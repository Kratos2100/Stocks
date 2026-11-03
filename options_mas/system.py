"""Wires agents, orchestrator, risk and execution into one event-driven system.

The same :class:`TradingSystem` instance is driven by the backtester, the
paper-trading runner and (after explicit authorisation) a live runner, so the
strategy code path is identical everywhere.
"""
from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any, Callable, Iterable, Optional

from .agents import BaseStrategyAgent
from .config import AppConfig
from .execution.broker import BrokerAdapter, SimulatedBroker
from .execution.charges import ChargesCalculator
from .execution.engine import ExecutionCallbacks, ExecutionEngine
from .models import Candle, ExitReason, OptionContract, Trade
from .orchestrator import Orchestrator, StrategyRanker
from .risk import RiskEngine


class EventBus:
    def __init__(self) -> None:
        self._subs: list[Callable[[str, dict], None]] = []
        self.recent: deque[dict[str, Any]] = deque(maxlen=500)

    def subscribe(self, fn: Callable[[str, dict], None]) -> None:
        self._subs.append(fn)

    def publish(self, kind: str, payload: dict) -> None:
        self.recent.append({"kind": kind, "payload": payload})
        for fn in list(self._subs):
            fn(kind, payload)


class TradingSystem:
    def __init__(self, cfg: AppConfig, agents: dict[str, BaseStrategyAgent],
                 broker: Optional[BrokerAdapter] = None, bus: Optional[EventBus] = None,
                 live_confirmed: bool = False):
        self.cfg = cfg
        self.agents = agents
        self.bus = bus or EventBus()
        self.broker = broker or SimulatedBroker(cfg.execution)
        self.charges = ChargesCalculator(cfg.charges)
        self.risk = RiskEngine(cfg, self.charges)
        self.contracts: dict[str, OptionContract] = {}
        self.monitors: dict[str, list[str]] = {}
        self.last_price: dict[str, float] = {}
        self.regime_by_date: dict[date, str] = {}           # analytics (same-day, reporting only)
        self.decision_regime_by_date: dict[date, str] = {}  # previous-session regime (no look-ahead)
        self.suppress_after_entry = False
        self.suppressed: set[tuple[str, str]] = set()
        cb = ExecutionCallbacks(
            entry_filled=self._entry_filled, entry_rejected=self._entry_rejected,
            exit_filled=self._exit_filled, trade_closed=self._trade_closed,
            order_update=lambda o: self.bus.publish("order", o.to_dict()),
        )
        self.execution = ExecutionEngine(cfg, self.broker, self.charges, cb, live_confirmed)
        self.execution.set_regime_fn(self.regime)
        self.ranker = StrategyRanker(cfg.orchestrator)
        self.orchestrator = Orchestrator(cfg, self.risk, self.execution, self.contracts, self.ranker,
                                         self.decision_regime, self._orchestrator_reject, self.bus.publish)

    # --------------------------------------------------------------- setup
    def regime(self, ts: datetime) -> str:
        return self.regime_by_date.get(ts.date(), "UNKNOWN")

    def decision_regime(self, ts: datetime) -> str:
        return self.decision_regime_by_date.get(ts.date(), "UNKNOWN")

    def register_contract(self, contract: OptionContract, warmup: Iterable[Candle] = (),
                          agent_ids: Optional[list[str]] = None) -> None:
        warm = list(warmup)
        self.contracts[contract.key] = contract
        ids = agent_ids or list(self.agents)
        mons = self.monitors.setdefault(contract.key, [])
        for aid in ids:
            if aid not in self.agents:
                continue
            # new contexts are warmed up; existing contexts receive any missed history
            self.agents[aid].register_contract(contract, warm)
            if aid not in mons:
                mons.append(aid)
        self.bus.publish("contract", contract.to_dict())

    def unregister_contract(self, key: str) -> None:
        self.monitors.pop(key, None)

    def price_levels(self, key: str) -> list[float]:
        lv: list[float] = []
        for aid in self.monitors.get(key, []):
            if (aid, key) in self.suppressed:
                continue
            lv.extend(self.agents[aid].price_levels(key))
        return lv

    # ---------------------------------------------------------- events
    def on_price(self, key: str, price: float, ts: datetime) -> None:
        self.last_price[key] = price
        self.execution.on_price(key, price, ts)
        self._risk_check(ts, "intrabar")
        signals = []
        for aid in self.monitors.get(key, []):
            if (aid, key) in self.suppressed:
                continue
            signals.extend(self.agents[aid].on_price(key, price, ts))
        if signals:
            self.orchestrator.process(signals, ts, "intrabar", self.last_price)

    def on_candles_closed(self, items: list[tuple[str, Candle]]) -> None:
        if not items:
            return
        signals = []
        ts = max(c.end_ts for _, c in items)
        for key, candle in items:
            self.last_price[key] = candle.close
            for aid in self.monitors.get(key, []):
                signals.extend(self.agents[aid].on_candle_close(key, candle))
            self.bus.publish("candle", {"contract_key": key, **candle.to_dict()})
        self.orchestrator.process(signals, ts, "close", self.last_price)
        self._risk_check(ts, "close")
        if self.risk.is_eod(ts):
            self.execution.exit_all(ExitReason.EOD_SQUARE_OFF, ts, "close")

    def begin_bar(self) -> None:
        self.suppressed.clear()

    def end_session(self, ts: datetime) -> None:
        """Flatten anything still open at the last known prices and cancel
        pending setups."""
        for key, price in list(self.last_price.items()):
            self.execution.flush(key, price, ts)
        if self.execution.open_positions() or self.execution.approvals:
            self.execution.exit_all(ExitReason.EOD_SQUARE_OFF, ts, "intrabar")
            for key, price in list(self.last_price.items()):
                self.execution.flush(key, price, ts)
        signals = []
        for key, aids in self.monitors.items():
            for aid in aids:
                signals.extend(self.agents[aid].end_session(key, ts))
        if signals:
            self.orchestrator.process(signals, ts, "close", self.last_price)

    def emergency_stop(self, ts: datetime, reason: str = "manual emergency stop") -> None:
        self.risk.emergency_stop(reason)
        self.execution.exit_all(ExitReason.EMERGENCY_STOP, ts, "intrabar")
        self.bus.publish("risk", self.risk.snapshot())

    def _risk_check(self, ts: datetime, phase: str) -> None:
        actions = self.risk.update_unrealized(ts, self.execution.open_positions())
        if actions and self.execution.open_positions():
            self.execution.exit_all(actions[0], ts, phase)

    # ------------------------------------------------------- callbacks
    def _entry_filled(self, aid: str, key: str, price: float, qty: int, ts: datetime) -> None:
        self.agents[aid].on_entry_filled(key, price, qty, ts)
        if self.suppress_after_entry:
            self.suppressed.add((aid, key))
        self.bus.publish("fill", {"agent_id": aid, "contract_key": key, "side": "BUY", "price": price,
                                  "qty": qty, "ts": ts.isoformat()})

    def _entry_rejected(self, aid: str, key: str, reason: str, ts: datetime) -> None:
        self.risk.release_entry()
        self.agents[aid].on_entry_rejected(key, reason, ts)

    def _orchestrator_reject(self, aid: str, key: str, reason: str, ts: datetime) -> None:
        self.agents[aid].on_entry_rejected(key, reason, ts)

    def _exit_filled(self, aid: str, key: str, price: float, ts: datetime, reason: str, pnl: float) -> None:
        self.agents[aid].on_exit_filled(key, price, ts, reason, pnl)
        self.bus.publish("fill", {"agent_id": aid, "contract_key": key, "side": "SELL", "price": price,
                                  "ts": ts.isoformat(), "reason": reason})

    def _trade_closed(self, trade: Trade) -> None:
        self.risk.on_trade_closed(trade)
        self.ranker.record(trade)
        self.bus.publish("trade", trade.to_dict())

    # ------------------------------------------------------- snapshots
    def snapshot(self) -> dict[str, Any]:
        return {
            "agents": [a.snapshot({k for k, ids in self.monitors.items() if a.agent_id in ids})
                       for a in self.agents.values()],
            "positions": [p.to_dict() for p in self.execution.positions.values() if p.is_open],
            "risk": self.risk.snapshot(),
            "performance": self.orchestrator.performance(),
            "approvals": [pa.order.to_dict() for pa in self.execution.approvals.values()],
            "mode": self.cfg.execution.mode,
            "orchestrator_mode": self.cfg.orchestrator.mode,
        }
