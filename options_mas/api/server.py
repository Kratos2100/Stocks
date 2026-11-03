"""FastAPI monitoring dashboard + REST/WebSocket API.

Run a replay of a dataset (paper-style, fills simulated) and watch the three
agents live in the browser::

    python -m options_mas.cli dashboard --synthetic          # demo data (labelled SYNTHETIC)
    python -m options_mas.cli dashboard --data options.csv   # your historical premium data
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..backtest.engine import Backtester
from ..backtest.metrics import full_report
from ..config import AppConfig
from ..data.dataset import BacktestDataset
from ..storage import Store
from ..system import EventBus

STATIC = Path(__file__).parent / "static"


def _epoch(ts: datetime) -> int:
    # Exchange-local naive timestamps are sent as if UTC so the chart shows IST labels.
    return int((ts - datetime(1970, 1, 1)).total_seconds())


class ReplayRuntime:
    def __init__(self, cfg: AppConfig, dataset: BacktestDataset, enabled_agents: Optional[list[str]] = None,
                 speed: float = 0.25, store: Optional[Store] = None):
        self.cfg = cfg
        self.ds = dataset
        self.bus = EventBus()
        self.events: list[dict[str, Any]] = []
        self.bus.subscribe(lambda k, p: self.events.append({"kind": k, "payload": p}))
        if store:
            self.bus.subscribe(store.handle)
        self.bt = Backtester(cfg, dataset, enabled_agents, bus=self.bus)
        self.gen = self.bt.iter_run()
        self.speed = speed
        self.paused = False
        self.done = False
        self.now: Optional[datetime] = None
        self.clients: set[WebSocket] = set()

    @property
    def system(self):
        return self.bt.system

    def step(self) -> bool:
        try:
            self.now = next(self.gen)
            return True
        except StopIteration:
            self.done = True
            return False

    async def loop(self) -> None:
        self.step()  # initialise system
        while not self.done:
            if not self.paused:
                self.step()
                await self.broadcast()
            await asyncio.sleep(self.speed)
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

    # ------------------------------------------------------------- views
    def status(self) -> dict[str, Any]:
        s = self.system
        snap = s.snapshot() if s else {}
        warnings = []
        if "SYNTHETIC" in self.ds.label.upper():
            warnings.append("SYNTHETIC DATA – NOT REAL MARKET PERFORMANCE")
        return {"now": self.now.isoformat() if self.now else None, "done": self.done, "paused": self.paused,
                "data_label": self.ds.label, "warnings": warnings, **snap}

    def chart(self, key: str) -> dict[str, Any]:
        s = self.system
        if s is None or key not in s.contracts:
            raise KeyError(key)
        agents = [a for a in s.agents.values() if a.has_contract(key)]
        if not agents:
            raise KeyError(key)
        base = agents[0].contexts[key]
        candles = base.candles
        times = [_epoch(c.ts) for c in candles]
        out: dict[str, Any] = {
            "contract": s.contracts[key].to_dict(),
            "candles": [{"time": t, "open": c.open, "high": c.high, "low": c.low, "close": c.close}
                        for t, c in zip(times, candles)],
            "lines": {}, "delta": [], "markers": [], "pivots": None,
        }

        def line(values):
            return [{"time": t, "value": round(v, 4)} for t, v in zip(times, values) if v is not None]

        tf = self.cfg.session.timeframe_min
        for a in agents:
            ctx = a.contexts[key]
            ser = ctx.series
            n = len(times)
            if "ema50" in ser:
                out["lines"]["ema50"] = line(ser["ema50"][-n:])
                out["lines"]["vwap"] = line(ser["vwap"][-n:])
            if "ema9" in ser:
                out["lines"]["ema9"] = line(ser["ema9"][-n:])
                piv = ser.get("pivots", [])[-n:]
                lv: dict[str, list] = {}
                for t, p in zip(times, piv):
                    if p:
                        for name, val in p.items():
                            lv.setdefault(name, []).append({"time": t, "value": round(val, 2)})
                out["pivots"] = lv
            if "delta" in ser:
                src = ser.get("delta_source", [])[-n:]
                out["delta"] = [{"time": t, "value": round(v, 2), "source": sv}
                                for t, v, sv in zip(times, ser["delta"][-n:], src) if v is not None]
                out["delta_estimated"] = bool(src and src[-1] not in ("BID_ASK_PROVIDER", "BID_ASK_QUOTE_RULE"))
                out["delta_source"] = src[-1] if src else None
            first = candles[0].ts if candles else None
            for m in ctx.markers:
                ts = a._bar_start(datetime.fromisoformat(m["ts"]))
                if first and ts >= first:
                    out["markers"].append({**m, "time": _epoch(ts)})
        out["markers"].sort(key=lambda m: m["time"])
        return out


def create_app(runtime: ReplayRuntime, autostart: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if autostart:  # pragma: no cover - exercised manually
            task = asyncio.create_task(runtime.loop())
        else:
            runtime.step()
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Options Multi-Agent Trading System", version="0.1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return JSONResponse(json.loads(json.dumps(runtime.status(), default=str)))

    @app.get("/api/contracts")
    def contracts():
        s = runtime.system
        if not s:
            return []
        return [{**c.to_dict(), "agents": s.monitors.get(k, []), "active": k in s.monitors}
                for k, c in s.contracts.items()]

    @app.get("/api/chart/{key}")
    def chart(key: str):
        try:
            return JSONResponse(json.loads(json.dumps(runtime.chart(key), default=str)))
        except KeyError:
            raise HTTPException(404, f"unknown contract {key}")

    @app.get("/api/signals")
    def signals(limit: int = 200, agent: Optional[str] = None):
        s = runtime.system
        sig = [x.to_dict() for x in (s.orchestrator.signals if s else [])]
        if agent:
            sig = [x for x in sig if x["agent_id"] == agent]
        return JSONResponse(json.loads(json.dumps(sig[-limit:][::-1], default=str)))

    @app.get("/api/decisions")
    def decisions(limit: int = 200):
        s = runtime.system
        return [d.to_dict() for d in list(s.orchestrator.decisions)[-limit:][::-1]] if s else []

    @app.get("/api/trades")
    def trades():
        s = runtime.system
        return [t.to_dict() for t in s.execution.trades][::-1] if s else []

    @app.get("/api/performance")
    def performance():
        s = runtime.system
        tr = list(s.execution.trades) if s else []
        return {"data_label": runtime.ds.label, **full_report(tr, runtime.cfg.risk.capital)}

    @app.get("/api/reconcile")
    def reconcile():
        return runtime.system.execution.reconcile()

    @app.post("/api/approvals/{order_id}/{action}")
    def approval(order_id: str, action: str):
        s = runtime.system
        ex = s.execution
        if order_id not in ex.approvals:
            raise HTTPException(404, "no such pending approval")
        ts = runtime.now or datetime.now()
        o = ex.approve(order_id, ts) if action == "approve" else ex.reject_approval(order_id, ts)
        return o.to_dict()

    @app.post("/api/emergency-stop")
    def emergency_stop():
        runtime.system.emergency_stop(runtime.now or datetime.now(), "dashboard emergency stop")
        return runtime.system.risk.snapshot()

    @app.post("/api/emergency-clear")
    def emergency_clear():
        runtime.system.risk.clear_emergency()
        return runtime.system.risk.snapshot()

    @app.post("/api/control/{action}")
    def control(action: str):
        if action == "pause":
            runtime.paused = True
        elif action == "resume":
            runtime.paused = False
        elif action == "step":
            runtime.step()
        else:
            raise HTTPException(400, "action must be pause|resume|step")
        return {"paused": runtime.paused, "now": runtime.now.isoformat() if runtime.now else None}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):  # pragma: no cover - exercised manually
        await websocket.accept()
        runtime.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            runtime.clients.discard(websocket)

    return app
