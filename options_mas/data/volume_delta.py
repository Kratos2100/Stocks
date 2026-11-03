"""Individual-candle volume delta: classification and analytics.

Priority order (see docs/VOLUME_DELTA_METHODOLOGY.md):

1. ``BID_ASK_PROVIDER``  – the data vendor supplies aggressor-classified
   buy/sell volume per bar. The vendor's methodology must be documented.
2. ``BID_ASK_QUOTE_RULE`` – we receive trades *and* the prevailing best
   bid/ask and classify each trade ourselves (quote rule, Lee-Ready fallback
   at the mid).
3. ``ESTIMATED_TICK_RULE`` – trades without quotes: up-tick = buy.
4. ``ESTIMATED_BAR_CLV``  – OHLCV only: close-location value split.

Sources 3 and 4 are **estimates** and are labelled as such everywhere.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..models import Candle, DeltaSource


@dataclass(frozen=True)
class Trade:
    price: float
    qty: float
    bid: Optional[float] = None
    ask: Optional[float] = None


def classify_trades_quote_rule(trades: Sequence[Trade]) -> tuple[float, float, DeltaSource]:
    """Classify trades against the prevailing bid/ask.

    * price >= ask -> buyer-initiated; price <= bid -> seller-initiated
    * strictly inside the spread -> compare with mid
    * exactly at mid -> tick rule fallback (Lee-Ready)

    If *any* trade lacks a quote, the whole bar is downgraded to the tick rule
    so that estimated and bid/ask-classified volume are never mixed.
    """
    if any(t.bid is None or t.ask is None for t in trades):
        b, s = classify_trades_tick_rule(trades)
        return b, s, DeltaSource.ESTIMATED_TICK_RULE
    buy = sell = 0.0
    last_price: Optional[float] = None
    last_dir = 0
    for t in trades:
        mid = (t.bid + t.ask) / 2.0  # type: ignore[operator]
        if t.price >= t.ask:  # type: ignore[operator]
            d = 1
        elif t.price <= t.bid:  # type: ignore[operator]
            d = -1
        elif t.price > mid:
            d = 1
        elif t.price < mid:
            d = -1
        else:
            d = _tick_dir(t.price, last_price, last_dir)
        if d > 0:
            buy += t.qty
        elif d < 0:
            sell += t.qty
        last_dir = d or last_dir
        last_price = t.price
    return buy, sell, DeltaSource.BID_ASK_QUOTE_RULE


def _tick_dir(price: float, last_price: Optional[float], last_dir: int) -> int:
    if last_price is None:
        return 0
    if price > last_price:
        return 1
    if price < last_price:
        return -1
    return last_dir  # zero-tick: inherit previous direction


def classify_trades_tick_rule(trades: Sequence[Trade], prev_price: Optional[float] = None) -> tuple[float, float]:
    buy = sell = 0.0
    last_price = prev_price
    last_dir = 0
    for t in trades:
        d = _tick_dir(t.price, last_price, last_dir)
        if d > 0:
            buy += t.qty
        elif d < 0:
            sell += t.qty
        else:  # unclassifiable first trade: split evenly
            buy += t.qty / 2.0
            sell += t.qty / 2.0
        last_dir = d or last_dir
        last_price = t.price
    return buy, sell


def estimate_bar_clv(candle: Candle) -> tuple[float, float]:
    """Close-location-value split: buy = V * (C - L) / (H - L).

    This is a *price-shape* model, not order flow. A red candle with a high
    close-location (long lower wick) produces a positive estimated delta.
    """
    rng = candle.high - candle.low
    if rng <= 0:
        return candle.volume / 2.0, candle.volume / 2.0
    frac = (candle.close - candle.low) / rng
    buy = candle.volume * frac
    return buy, candle.volume - buy


@dataclass
class DeltaMetrics:
    buy_volume: float
    sell_volume: float
    total_volume: float
    delta: float
    delta_pct: float
    avg_abs_delta: Optional[float]
    avg_delta: Optional[float]
    relative_delta: Optional[float]
    imbalance_ratio: Optional[float]
    dominance: str
    source: DeltaSource
    cvd: Optional[float] = None

    @property
    def is_estimated(self) -> bool:
        return self.source.is_estimated

    def to_dict(self) -> dict:
        return {
            "buy_volume": round(self.buy_volume, 2),
            "sell_volume": round(self.sell_volume, 2),
            "total_volume": round(self.total_volume, 2),
            "delta": round(self.delta, 2),
            "delta_pct": round(self.delta_pct, 3),
            "avg_abs_delta": None if self.avg_abs_delta is None else round(self.avg_abs_delta, 2),
            "avg_delta": None if self.avg_delta is None else round(self.avg_delta, 2),
            "relative_delta": None if self.relative_delta is None else round(self.relative_delta, 3),
            "imbalance_ratio": None if self.imbalance_ratio is None else round(self.imbalance_ratio, 3),
            "dominance": self.dominance,
            "delta_source": self.source.value,
            "delta_is_estimated": self.is_estimated,
            "cvd": None if self.cvd is None else round(self.cvd, 2),
        }


class DeltaTracker:
    """Rolling per-contract delta statistics.

    ``relative_delta`` = |delta| / mean(|delta|) of the *previous*
    ``lookback`` candles, so the current candle never contaminates its own
    baseline (no look-ahead).
    """

    def __init__(self, lookback: int = 20, neutral_delta_pct: float = 5.0, estimate_when_missing: bool = True):
        self.lookback = lookback
        self.neutral = neutral_delta_pct
        self.estimate_when_missing = estimate_when_missing
        self._abs_hist: deque[float] = deque(maxlen=lookback)
        self._hist: deque[float] = deque(maxlen=lookback)
        self.cvd = 0.0
        self._cvd_session = None

    def resolve(self, candle: Candle) -> Optional[tuple[float, float, DeltaSource]]:
        if candle.has_delta:
            src = candle.delta_source if candle.delta_source != DeltaSource.NONE else DeltaSource.BID_ASK_PROVIDER
            return float(candle.buy_volume), float(candle.sell_volume), src  # type: ignore[arg-type]
        if self.estimate_when_missing:
            b, s = estimate_bar_clv(candle)
            return b, s, DeltaSource.ESTIMATED_BAR_CLV
        return None

    def update(self, candle: Candle) -> Optional[DeltaMetrics]:
        resolved = self.resolve(candle)
        if resolved is None:
            return None
        buy, sell, src = resolved
        sess = candle.ts.date()
        if sess != self._cvd_session:
            self.cvd = 0.0
            self._cvd_session = sess
        total = buy + sell
        delta = buy - sell
        delta_pct = (delta / total * 100.0) if total > 0 else 0.0
        avg_abs = (sum(self._abs_hist) / len(self._abs_hist)) if self._abs_hist else None
        avg = (sum(self._hist) / len(self._hist)) if self._hist else None
        rel = (abs(delta) / avg_abs) if avg_abs and avg_abs > 0 else None
        imb = (buy / sell) if sell > 0 else None
        if abs(delta_pct) < self.neutral:
            dom = "BALANCED"
        elif delta > 0:
            dom = "BUYING"
        else:
            dom = "SELLING"
        self.cvd += delta
        self._abs_hist.append(abs(delta))
        self._hist.append(delta)
        return DeltaMetrics(buy, sell, total, delta, delta_pct, avg_abs, avg, rel, imb, dom, src, self.cvd)


def aggregate_trades_to_delta(trades: Iterable[Trade]) -> tuple[float, float, DeltaSource]:
    trades = list(trades)
    if not trades:
        return 0.0, 0.0, DeltaSource.NONE
    return classify_trades_quote_rule(trades)
