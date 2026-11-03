"""Deterministic, streaming indicator implementations.

Every indicator is updated exactly once per *completed* candle, which makes
repainting impossible by construction: a value computed for candle ``t`` only
ever uses candles ``<= t``. Vectorised reference implementations are provided
for validation against pandas / TradingView.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .models import Candle


class EMA:
    """Exponential moving average seeded with the SMA of the first ``period``
    values (matches TradingView ``ta.ema``)."""

    def __init__(self, period: int):
        if period < 1:
            raise ValueError("EMA period must be >= 1")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value: Optional[float] = None
        self._seed: list[float] = []

    @property
    def ready(self) -> bool:
        return self.value is not None

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._seed.append(float(x))
            if len(self._seed) >= self.period:
                self.value = sum(self._seed) / len(self._seed)
                self._seed = []
            return self.value
        self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
        self._seed = []


class SessionVWAP:
    """Session-anchored VWAP using typical price (H+L+C)/3 (TradingView default
    source ``hlc3``). Call :meth:`reset` at each session start."""

    def __init__(self) -> None:
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.value: Optional[float] = None

    def reset(self) -> None:
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self.value = None

    def update(self, candle: Candle) -> Optional[float]:
        tp = (candle.high + candle.low + candle.close) / 3.0
        if candle.volume and candle.volume > 0:
            self.cum_pv += tp * candle.volume
            self.cum_v += candle.volume
            self.value = self.cum_pv / self.cum_v
        return self.value


PIVOT_ORDER = ("S3", "S2", "S1", "P", "R1", "R2", "R3")


@dataclass(frozen=True)
class FibPivots:
    P: float
    R1: float
    R2: float
    R3: float
    S1: float
    S2: float
    S3: float
    source_high: float
    source_low: float
    source_close: float

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in PIVOT_ORDER}

    def sorted_levels(self, names: Iterable[str] = PIVOT_ORDER) -> list[tuple[str, float]]:
        d = self.as_dict()
        return sorted(((n, d[n]) for n in names), key=lambda kv: kv[1])


def fibonacci_pivots(high: float, low: float, close: float) -> FibPivots:
    """Fibonacci pivot points from the previous completed session's H/L/C."""
    if high < low:
        raise ValueError("high must be >= low")
    p = (high + low + close) / 3.0
    rng = high - low
    return FibPivots(
        P=p,
        R1=p + 0.382 * rng,
        R2=p + 0.618 * rng,
        R3=p + 1.000 * rng,
        S1=p - 0.382 * rng,
        S2=p - 0.618 * rng,
        S3=p - 1.000 * rng,
        source_high=high,
        source_low=low,
        source_close=close,
    )


# --------------------------------------------------------------------------
# Vectorised reference implementations (validation / dashboards / tests)
# --------------------------------------------------------------------------

def ema_series(values: Sequence[float], period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    e = EMA(period)
    for i, v in enumerate(values):
        r = e.update(v)
        if r is not None:
            out[i] = r
    return out


def vwap_series(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    session_ids: Sequence[object],
) -> np.ndarray:
    h, l, c, v = (np.asarray(x, dtype=float) for x in (high, low, close, volume))
    tp = (h + l + c) / 3.0
    out = np.full(len(c), np.nan)
    cum_pv = cum_v = 0.0
    prev_sid = object()
    last = np.nan
    for i, sid in enumerate(session_ids):
        if sid != prev_sid:
            cum_pv = cum_v = 0.0
            last = np.nan
            prev_sid = sid
        if v[i] > 0:
            cum_pv += tp[i] * v[i]
            cum_v += v[i]
            last = cum_pv / cum_v
        out[i] = last
    return out
