"""Paper / manual-approval / live runtime driven by a tick feed.

``MarketDataFeed`` is the abstraction a broker market-data websocket
implements. :class:`BarBuilder` turns ticks into completed 5-minute candles
(with bid/ask-classified delta when quotes are present) and calls the exact
same :class:`TradingSystem` used by the backtester.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import AsyncIterator, Iterable, Optional

from .backtest.engine import ohlc_path
from .data.volume_delta import Trade as TapeTrade, classify_trades_quote_rule
from .models import Candle, DeltaSource
from .system import TradingSystem


@dataclass(frozen=True)
class Tick:
    key: str
    ts: datetime
    price: float
    qty: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None


class MarketDataFeed(ABC):
    @abstractmethod
    def subscribe(self, keys: Iterable[str]) -> None: ...

    @abstractmethod
    def stream(self) -> AsyncIterator[Tick]: ...


class BarBuilder:
    """Aggregates ticks into bars anchored at 09:15. A bar is emitted only
    when a tick of a later bar arrives (or :meth:`close_all` is called), so
    an incomplete bar is never treated as completed."""

    def __init__(self, timeframe_min: int = 5, anchor_minute: int = 9 * 60 + 15):
        self.tf = timeframe_min
        self.anchor = anchor_minute
        self.cur: dict[str, dict] = {}

    def bar_start(self, ts: datetime) -> datetime:
        m = ts.hour * 60 + ts.minute
        s = m - ((m - self.anchor) % self.tf)
        return ts.replace(hour=s // 60, minute=s % 60, second=0, microsecond=0)

    def on_tick(self, t: Tick) -> Optional[Candle]:
        start = self.bar_start(t.ts)
        b = self.cur.get(t.key)
        done = None
        if b is not None and b["start"] != start:
            done = self._finish(t.key)
            b = None
        if b is None:
            b = {"start": start, "o": t.price, "h": t.price, "l": t.price, "c": t.price, "v": 0.0, "trades": []}
            self.cur[t.key] = b
        b["h"] = max(b["h"], t.price)
        b["l"] = min(b["l"], t.price)
        b["c"] = t.price
        b["v"] += t.qty
        if t.qty:
            b["trades"].append(TapeTrade(t.price, t.qty, t.bid, t.ask))
        return done

    def _finish(self, key: str) -> Candle:
        b = self.cur.pop(key)
        if b["trades"]:
            buy, sell, src = classify_trades_quote_rule(b["trades"])
        else:
            buy = sell = None
            src = DeltaSource.NONE
        return Candle(b["start"], b["o"], b["h"], b["l"], b["c"], b["v"], buy, sell, src, timeframe_min=self.tf)

    def close_due(self, now: datetime) -> list[tuple[str, Candle]]:
        """Close bars whose time window has fully elapsed (illiquid contracts
        that have not traded since the bar ended)."""
        due = [k for k, b in self.cur.items() if b["start"] + timedelta(minutes=self.tf) <= now]
        return [(k, self._finish(k)) for k in due]

    def close_all(self) -> list[tuple[str, Candle]]:
        return [(k, self._finish(k)) for k in list(self.cur)]


class CandleReplayFeed(MarketDataFeed):
    """Replays historical candles as ticks along the O-(L/H)-(H/L)-C path.
    Useful for paper-trading rehearsals without a broker connection."""

    def __init__(self, candles: dict[str, list[Candle]], speed: float = 0.0):
        self.candles = candles
        self.speed = speed
        self.keys: list[str] = []

    def subscribe(self, keys: Iterable[str]) -> None:
        self.keys = list(keys)

    async def stream(self) -> AsyncIterator[Tick]:
        events = []
        for k in self.keys:
            for c in self.candles.get(k, []):
                path = ohlc_path(c)
                step = c.timeframe_min * 60 / (len(path) + 1)
                share = c.volume / len(path) if path else 0
                for i, p in enumerate(path):
                    events.append(Tick(k, c.ts + timedelta(seconds=step * i), p, share))
        events.sort(key=lambda t: (t.ts, t.key))
        for t in events:
            if self.speed:
                await asyncio.sleep(self.speed)
            yield t


class StreamingRunner:
    """Feeds ticks into the TradingSystem (paper / manual / live modes)."""

    def __init__(self, system: TradingSystem, feed: MarketDataFeed, timeframe_min: int = 5):
        self.system = system
        self.feed = feed
        self.builder = BarBuilder(timeframe_min)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        self.feed.subscribe(self.system.contracts.keys())
        last_ts = None
        async for t in self.feed.stream():
            if self._stop.is_set():
                break
            if last_ts is not None and t.ts.date() != last_ts.date():
                self._close(self.builder.close_all())
                self.system.end_session(last_ts)
            done = self.builder.on_tick(t)
            # completed bars are processed BEFORE the new bar's first tick
            self._close(([(t.key, done)] if done else []) + self.builder.close_due(t.ts))
            self.system.on_price(t.key, t.price, t.ts)
            last_ts = t.ts
        self._close(self.builder.close_all())
        if last_ts is not None:
            self.system.end_session(last_ts)

    def _close(self, items: list[tuple[str, Candle]]) -> None:
        if items:
            self.system.on_candles_closed(items)
