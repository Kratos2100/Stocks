"""Live monitor: Angel One ticks -> 5-minute bars -> the three agents, with
**simulated (paper) fills only**, shown on the same dashboard page as the
replay::

    python -m options_mas.cli live --underlying NIFTY --strikes-each-side 5   # http://127.0.0.1:8000

Nothing here can place an order with Angel One: the system is built with
:class:`SimulatedBroker`, and the only Angel One calls are login, the
instrument master, historical candles (warm-up) and the market-data websocket.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, AsyncIterator, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

from ..config import AppConfig
from ..live import MarketDataFeed, StreamingRunner, Tick
from ..models import Candle, OptionContract, OptionType
from ..storage import Store
from ..system import EventBus, TradingSystem
from .server import RuntimeViews

IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


@dataclass
class _Label:
    label: str


class CountingFeed(MarketDataFeed):
    """Wraps a feed and records what the page shows about it: tick count,
    last tick time and last traded price per contract."""

    def __init__(self, inner: MarketDataFeed, session: Optional[tuple[time, time]] = None):
        self.inner = inner
        self.session = session
        self.outside_session = 0
        self.ticks = 0
        self.last_tick: Optional[datetime] = None
        self.last_received: Optional[datetime] = None
        self.ltp: dict[str, dict[str, Any]] = {}
        self.started: Optional[datetime] = None

    def subscribe(self, keys: Iterable[str]) -> None:
        self.inner.subscribe(keys)

    async def stream(self) -> AsyncIterator[Tick]:
        self.started = ist_now()
        async for t in self.inner.stream():
            self.ticks += 1
            self.last_tick = t.ts
            self.last_received = ist_now()
            self.ltp[t.key] = {"price": t.price, "ts": t.ts.isoformat()}
            if self.session and not (self.session[0] <= t.ts.time() < self.session[1]):
                self.outside_session += 1   # quotes after the close: shown, never built into bars
                continue
            yield t

    def snapshot(self, now: datetime, stale_after_s: float = 30.0) -> dict[str, Any]:
        age = (now - self.last_received).total_seconds() if self.last_received else None
        return {"ticks": self.ticks, "outside_session": self.outside_session, "last_tick": self.last_tick.isoformat() if self.last_tick else None,
                "seconds_since_last_tick": None if age is None else round(age, 1),
                "stale": age is None or age > stale_after_s,
                "started": self.started.isoformat() if self.started else None, "ltp": self.ltp}


class LiveRuntime(RuntimeViews):
    """Same interface as ``ReplayRuntime`` for the FastAPI app and page."""
    live = True

    def __init__(self, cfg: AppConfig, system: TradingSystem, feed: MarketDataFeed, label: str,
                 store: Optional[Store] = None, warnings: Iterable[str] = (), clock: Callable[[], datetime] = ist_now,
                 broadcast_every: float = 1.0, close_grace_s: float = 2.0):
        self.cfg = cfg
        self.system = system
        self.bus: EventBus = system.bus
        self.feed = CountingFeed(feed, (cfg.session.market_open, cfg.session.market_close))
        self.ds = _Label(label)
        self.warnings = list(warnings)
        self.events: list[dict[str, Any]] = []
        self.bus.subscribe(lambda k, p: self.events.append({"kind": k, "payload": p}))
        if store:
            self.bus.subscribe(store.handle)
        self.clock = clock
        self.broadcast_every = broadcast_every
        self.close_grace = timedelta(seconds=close_grace_s)
        self.runner = StreamingRunner(system, self.feed, cfg.session.timeframe_min)
        self.paused = False
        self.done = False
        self.error: Optional[str] = None
        self.clients: set = set()

    @property
    def now(self) -> Optional[datetime]:
        return self.feed.last_tick

    def step(self) -> bool:  # the app calls this when not autostarting; nothing to advance
        return False

    def tick_clock(self) -> None:
        """Close 5-minute bars that ended on the wall clock (no tick needed)."""
        self.runner.close_due(self.clock() - self.close_grace)

    async def loop(self) -> None:  # pragma: no cover - needs a live feed
        task = asyncio.create_task(self.runner.run())
        try:
            while not task.done():
                self.tick_clock()
                await self.broadcast()
                await asyncio.sleep(self.broadcast_every)
            exc = task.exception()
            if exc:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            task.cancel()
            self.done = True
            await self.broadcast()

    async def broadcast(self) -> None:
        evs, self.events = self.events, []
        msg = json.dumps({"type": "update", "now": self.now.isoformat() if self.now else None,
                          "done": self.done, "events": evs[-200:]}, default=str)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def status(self) -> dict[str, Any]:
        return {"now": self.now.isoformat() if self.now else None, "done": self.done, "paused": False,
                "live": True, "data_label": self.ds.label, "warnings": self.warnings, "error": self.error,
                "feed": self.feed.snapshot(self.clock()), "clock": self.clock().isoformat(timespec="seconds"),
                **self.system.snapshot()}


# ------------------------------------------------------------------ Angel One wiring
def bars_to_candles(bars: list[dict], tf: int) -> list[Candle]:
    from ..data.providers.common import to_ist_naive
    out = []
    for b in bars:
        ts = to_ist_naive(b["timestamp"]).to_pydatetime()
        out.append(Candle(ts, float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"]),
                          float(b.get("volume") or 0.0), timeframe_min=tf))
    return out


def split_warmup(candles: list[Candle], now: datetime, sessions: int,
                 session: tuple[time, time] = (time(9, 15), time(15, 30))) -> list[Candle]:
    """Previous ``sessions`` sessions plus today's bars that have already
    *completed* by ``now``; the forming bar is dropped (the feed rebuilds it).
    Bars outside the regular session (Angel One also serves 15:30/15:35 bars)
    are dropped, as in the historical download."""
    candles = [c for c in candles if session[0] <= c.ts.time() < session[1]]
    today = now.date()
    prev_days = sorted({c.ts.date() for c in candles if c.ts.date() < today})[-sessions:] if sessions > 0 else []
    keep = set(prev_days)
    return [c for c in candles if (c.ts.date() in keep) or (c.ts.date() == today and c.end_ts <= now)]


def pick_strikes(listed: Iterable[float], atm_price: float, each_side: int) -> list[float]:
    ks = sorted(set(float(k) for k in listed))
    if not ks:
        return []
    atm = min(ks, key=lambda k: (abs(k - atm_price), k))
    i = ks.index(atm)
    return ks[max(0, i - each_side): i + each_side + 1]


def build_angelone_runtime(cfg: AppConfig, underlying: str = "NIFTY", expiry: Optional[date] = None,
                           strikes_each_side: int = 5, enabled_agents: Optional[list[str]] = None,
                           store: Optional[Store] = None, client: Any = None, feed_factory: Any = None,
                           now: Optional[datetime] = None, log: Callable[[str], None] = print) -> LiveRuntime:
    from ..agents import build_agents
    from ..execution.broker import SimulatedBroker
    from ..data.providers.angelone import AngelOneClient, AngelOneFeed, listed_expiries

    und = underlying.upper()
    now = now or ist_now()
    cfg.execution.mode = "paper"          # fills are simulated; no order ever reaches Angel One
    client = client or AngelOneClient()
    master = client.master()
    after_close = now.time() >= cfg.session.market_close
    # an expiry settles at the close: after it, the next expiry is the live one
    exps = [e for e in listed_expiries(master, und) if e > now.date() or (e == now.date() and not after_close)]
    if not exps:
        raise RuntimeError(f"no listed {und} expiries on or after {now.date()}")
    expiry = expiry or exps[0]
    con = client.contracts(und, expiry)
    if con.empty:
        raise RuntimeError(f"{und} {expiry} is not in the Angel One instrument master; listed: "
                           f"{[e.isoformat() for e in exps[:6]]}")

    idx = client.index_candles(und, now.date() - timedelta(days=7), now.date())
    if idx.empty:
        raise RuntimeError(f"no {und} index candles in the last 7 days to locate ATM")
    spot = float(idx["close"].iloc[-1])
    strikes = pick_strikes(con["strike"], spot, strikes_each_side)
    log(f"{und} {expiry}: index {spot:.2f} -> strikes {strikes[0]:g}..{strikes[-1]:g} ({len(strikes)} each CE/PE)")

    tf = cfg.session.timeframe_min
    system = TradingSystem(cfg, build_agents(cfg, enabled_agents), SimulatedBroker(cfg.execution))
    tokens: dict[str, tuple[str, str]] = {}
    for r in con[con["strike"].isin(strikes)].sort_values(["strike", "option_type"]).itertuples():
        oc = OptionContract(und, expiry, float(r.strike), OptionType(r.option_type), "NFO",
                            int(r.lot_size) or cfg.instruments[und].lot_size, r.symbol)
        bars = client.candles("NFO", r.token, now.date() - timedelta(days=10), now.date())
        warm = split_warmup(bars_to_candles(bars, tf), now, cfg.backtest.warmup_sessions,
                            (cfg.session.market_open, cfg.session.market_close))
        system.register_contract(oc, warm)
        tokens[oc.key] = ("NFO", str(r.token))
        log(f"  {r.symbol}: {len(warm)} warm-up bars")

    feed = (feed_factory or (lambda s, t: AngelOneFeed(s, t, mode=2)))(client.s, tokens)
    label = f"ANGELONE LIVE {und} {expiry} ±{strikes_each_side} strikes · paper fills"
    warnings = ["PAPER: fills are simulated; no orders are sent to Angel One.",
                "Live ticks carry no bid/ask: Agent 03 delta is ESTIMATED_TICK_RULE.",
                "Bars are built from ticks; the first bar after start-up is partial."]
    return LiveRuntime(cfg, system, feed, label, store=store, warnings=warnings)
