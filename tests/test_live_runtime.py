"""Live monitor wiring, without the network: a fake Angel One client and a
scripted tick feed. Values are synthetic."""
import asyncio
from datetime import date, datetime, timedelta

import pandas as pd
from fastapi.testclient import TestClient

from options_mas.api.live_runtime import (LiveRuntime, build_angelone_runtime, pick_strikes,
                                          split_warmup)
from options_mas.api.server import create_app
from options_mas.config import AppConfig
from options_mas.data.providers import angelone as a
from options_mas.execution.broker import SimulatedBroker
from options_mas.live import MarketDataFeed, Tick
from options_mas.models import Candle

DAY = date(2026, 11, 4)
NOW = datetime(2026, 11, 4, 10, 2)


def _master():
    rows = []
    for k in range(23000, 23650, 50):
        for ot in ("CE", "PE"):
            rows.append({"token": f"{k}{1 if ot == 'CE' else 2}", "symbol": f"NIFTY10NOV26{k}{ot}", "name": "NIFTY",
                         "expiry": "10NOV2026", "strike": f"{k * 100}.000000", "lotsize": "65",
                         "instrumenttype": "OPTIDX", "exch_seg": "NFO"})
    rows.append({"token": "99926000", "symbol": "Nifty 50", "name": "NIFTY", "expiry": "", "strike": "0.000000",
                 "lotsize": "1", "instrumenttype": "AMXIDX", "exch_seg": "NSE"})
    return rows


def _bars(day: date, n: int, start: float = 100.0):
    t0 = datetime.combine(day, datetime.min.time()).replace(hour=9, minute=15)
    return [{"timestamp": (t0 + timedelta(minutes=5 * i)).isoformat() + "+05:30", "open": start + i,
             "high": start + i + 2, "low": start + i - 1, "close": start + i + 1, "volume": 1000} for i in range(n)]


class FakeClient:
    def __init__(self):
        self.m = _master()
        self.s = a.Session("k", "c", "jwt", "feed")
        self.candle_calls = []

    def master(self):
        return self.m

    def contracts(self, und, expiry=None):
        return a.parse_scrip_master(self.m, und, expiry)

    def index_candles(self, und, start, end):
        return pd.DataFrame({"underlying": [und], "timestamp": [pd.Timestamp("2026-11-04 09:55")],
                             "open": [23290.0], "high": [23320.0], "low": [23280.0], "close": [23312.0],
                             "volume": [0]})

    def candles(self, exchange, token, start, end, interval="FIVE_MINUTE"):
        self.candle_calls.append((exchange, token))
        return _bars(date(2026, 11, 2), 75) + _bars(date(2026, 11, 3), 75) + _bars(DAY, 10)


class ScriptedFeed(MarketDataFeed):
    def __init__(self, ticks):
        self.ticks = ticks
        self.subscribed = []

    def subscribe(self, keys):
        self.subscribed = list(keys)

    async def stream(self):
        for t in self.ticks:
            yield t


def test_pick_strikes_centres_on_atm():
    ks = [23000 + 50 * i for i in range(13)]
    assert pick_strikes(ks, 23312, 2) == [23200.0, 23250.0, 23300.0, 23350.0, 23400.0]
    assert pick_strikes(ks, 22000, 1) == [23000.0, 23050.0]
    assert pick_strikes([], 1, 1) == []


def test_split_warmup_drops_forming_bar_and_old_sessions():
    c = lambda d, h, m: Candle(datetime(2026, 11, d, h, m), 1, 2, 0.5, 1.5)  # noqa: E731
    cs = [c(1, 9, 15), c(2, 9, 15), c(3, 9, 15), c(3, 15, 30), c(3, 15, 35), c(4, 9, 50), c(4, 9, 55), c(4, 10, 0)]
    out = split_warmup(cs, datetime(2026, 11, 4, 10, 2), sessions=2)
    assert [x.ts for x in out] == [datetime(2026, 11, 2, 9, 15), datetime(2026, 11, 3, 9, 15),
                                   datetime(2026, 11, 4, 9, 50), datetime(2026, 11, 4, 9, 55)]


def test_build_runtime_selects_atm_contracts_and_paper_broker():
    cfg = AppConfig()
    client = FakeClient()
    made = {}

    def feed_factory(session, tokens):
        made["tokens"] = tokens
        return ScriptedFeed([])

    rt = build_angelone_runtime(cfg, "NIFTY", strikes_each_side=2, client=client, feed_factory=feed_factory,
                                now=NOW, log=lambda m: None)
    assert isinstance(rt.system.broker, SimulatedBroker) and cfg.execution.mode == "paper"
    strikes = sorted({c.strike for c in rt.system.contracts.values()})
    assert strikes == [23200.0, 23250.0, 23300.0, 23350.0, 23400.0]
    assert len(rt.system.contracts) == 10 and len(client.candle_calls) == 10
    assert made["tokens"]["NIFTY-20261110-23300-CE"] == ("NFO", "233001")
    assert all(c.lot_size == 65 for c in rt.system.contracts.values())
    assert "paper fills" in rt.ds.label and any("no orders" in w for w in rt.warnings)


def test_live_runtime_builds_bars_from_ticks_and_serves_page():
    cfg = AppConfig()
    key = "NIFTY-20261110-23300-CE"
    t0 = datetime(2026, 11, 4, 10, 0, 5)
    ticks = [Tick(key, t0 + timedelta(seconds=20 * i), 110 + (i % 4), 0) for i in range(30)]  # 10:00:05..10:09:45
    rt = build_angelone_runtime(cfg, "NIFTY", strikes_each_side=1, client=FakeClient(),
                                feed_factory=lambda s, t: ScriptedFeed(ticks), now=NOW, log=lambda m: None)
    rt.clock = lambda: datetime(2026, 11, 4, 10, 10, 30)
    rt.runner.builder.close_all = lambda: []          # keep the forming bar open, as during a session
    n_before = len(rt.system.agents["A01"].contexts[key].candles)
    asyncio.run(rt.runner.run())
    ctx = rt.system.agents["A01"].contexts[key]
    assert len(ctx.candles) == n_before + 1 and ctx.candles[-1].ts == datetime(2026, 11, 4, 10, 0)
    rt.tick_clock()                                   # 10:05 bar ended on the wall clock
    assert ctx.candles[-1].ts == datetime(2026, 11, 4, 10, 5)

    app = TestClient(create_app(rt, autostart=False))
    st = app.get("/api/status").json()
    assert st["live"] is True and st["feed"]["ticks"] == 30
    assert st["feed"]["ltp"][key]["price"] == ticks[-1].price
    assert st["data_label"].startswith("ANGELONE LIVE")
    assert app.post("/api/control/pause").status_code == 400
    ch = app.get(f"/api/chart/{key}").json()
    assert ch["candles"][-1]["close"] == ctx.candles[-1].close
    assert app.get("/api/performance").json()["data_label"] == rt.ds.label
    assert "Live quotes" in app.get("/").text


def test_stale_feed_flag():
    from options_mas.api.live_runtime import CountingFeed
    f = CountingFeed(ScriptedFeed([]))
    assert f.snapshot(NOW)["stale"] is True
    f.last_received = NOW - timedelta(seconds=5)
    assert f.snapshot(NOW)["stale"] is False
    assert f.snapshot(NOW + timedelta(seconds=60))["stale"] is True


def test_after_close_uses_next_expiry_and_ignores_post_close_ticks():
    client = FakeClient()
    client.m = client.m + [dict(r, expiry="04NOV2026", symbol=r["symbol"].replace("10NOV", "04NOV"),
                                token="9" + r["token"]) for r in _master() if r["exch_seg"] == "NFO"]
    late = datetime(2026, 11, 4, 16, 0)
    rt = build_angelone_runtime(AppConfig(), "NIFTY", strikes_each_side=1, client=client, now=late,
                                log=lambda m: None, feed_factory=lambda s, t: ScriptedFeed([]))
    assert {c.expiry for c in rt.system.contracts.values()} == {date(2026, 11, 10)}
    before = build_angelone_runtime(AppConfig(), "NIFTY", strikes_each_side=1, client=client, now=NOW,
                                    log=lambda m: None, feed_factory=lambda s, t: ScriptedFeed([]))
    assert {c.expiry for c in before.system.contracts.values()} == {date(2026, 11, 4)}

    key = "NIFTY-20261110-23300-CE"
    ticks = [Tick(key, datetime(2026, 11, 4, 15, 39, 59), 110.9, 0)]
    rt2 = build_angelone_runtime(AppConfig(), "NIFTY", strikes_each_side=1, client=FakeClient(), now=late,
                                 log=lambda m: None, feed_factory=lambda s, t: ScriptedFeed(ticks))
    n = len(rt2.system.agents["A01"].contexts[key].candles)
    asyncio.run(rt2.runner.run())
    snap = rt2.feed.snapshot(late)
    assert snap["ltp"][key]["price"] == 110.9 and snap["outside_session"] == 1
    assert len(rt2.system.agents["A01"].contexts[key].candles) == n
