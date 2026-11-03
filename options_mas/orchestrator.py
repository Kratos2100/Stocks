"""Central orchestrator.

Responsibilities: receive signals from all agents, keep per-agent state
views, drop duplicate orders, detect conflicting signals, manage simultaneous
opportunities, coordinate capital allocation (through the risk engine),
prevent unintended duplicate positions and monitor strategy performance.

It NEVER alters a strategy's entry or exit rules. Consensus (mode B) is
analytics only and is never required for a trade.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .config import AppConfig, OrchestratorConfig
from .execution.engine import ExecutionEngine
from .models import ExitReason, OptionContract, OptionType, Signal, SignalStatus, SignalType, Trade
from .risk import RiskEngine


class StrategyRanker:
    """Ranks agents by historical *out-of-sample* net P&L per trade, shrunk
    towards zero with ``prior_trades`` pseudo-trades. Regime-specific stats are
    used when they have enough samples."""

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.stats: dict[str, dict[str, list[float]]] = {}
        if cfg.ranking_file and Path(cfg.ranking_file).exists():
            self.load(cfg.ranking_file)

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        for agent, regimes in data.items():
            for regime, st in regimes.items():
                self.stats.setdefault(agent, {})[regime] = [float(st["trades"]), float(st["net_pnl"])]

    def record(self, trade: Trade) -> None:
        for regime in ("ALL", trade.regime):
            s = self.stats.setdefault(trade.agent_id, {}).setdefault(regime, [0.0, 0.0])
            s[0] += 1
            s[1] += trade.net_pnl

    def score(self, agent_id: str, regime: str = "ALL") -> tuple[float, int]:
        a = self.stats.get(agent_id, {})
        st = a.get(regime)
        if st is None or st[0] < self.cfg.ranking_min_trades:
            st = a.get("ALL")
        if st is None or st[0] < self.cfg.ranking_min_trades:
            return 0.0, int(st[0]) if st else 0
        return st[1] / (st[0] + self.cfg.ranking_prior_trades), int(st[0])

    def table(self) -> dict[str, Any]:
        return {a: {r: {"trades": int(v[0]), "net_pnl": round(v[1], 2)} for r, v in regs.items()}
                for a, regs in self.stats.items()}


@dataclass
class Decision:
    ts: datetime
    signal_id: str
    agent_id: str
    contract_key: str
    action: str             # "SUBMITTED" | "REJECTED" | "EXIT_SUBMITTED" | "LOGGED"
    reasons: list[str]
    consensus: list[str]
    conflict: Optional[str] = None
    qty: int = 0
    rank_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(), "signal_id": self.signal_id, "agent_id": self.agent_id,
            "contract_key": self.contract_key, "action": self.action, "reasons": self.reasons,
            "consensus": self.consensus, "conflict": self.conflict, "qty": self.qty, "rank_score": self.rank_score,
        }


class Orchestrator:
    def __init__(self, cfg: AppConfig, risk: RiskEngine, execution: ExecutionEngine,
                 contracts: dict[str, OptionContract], ranker: Optional[StrategyRanker] = None,
                 regime_fn: Callable[[datetime], str] = lambda ts: "UNKNOWN",
                 on_reject: Callable[[str, str, str, datetime], None] = lambda *a: None,
                 publish: Callable[[str, dict], None] = lambda k, p: None):
        self.cfg = cfg
        self.oc = cfg.orchestrator
        if self.oc.mode not in ("A", "B", "C"):
            raise ValueError("orchestrator.mode must be A, B or C")
        self.risk = risk
        self.execution = execution
        self.contracts = contracts
        self.ranker = ranker or StrategyRanker(self.oc)
        self.regime_fn = regime_fn
        self.on_reject = on_reject
        self.publish = publish
        self.agent_states: dict[str, dict[str, str]] = {}
        self.signals: deque[Signal] = deque(maxlen=5000)
        self.decisions: deque[Decision] = deque(maxlen=5000)
        self.seen_events: set[tuple[str, str]] = set()
        self._recent: deque[tuple[datetime, str, str, OptionType, str]] = deque(maxlen=2000)
        self.conflicts: list[dict[str, Any]] = []

    # --------------------------------------------------------------- main
    def process(self, signals: list[Signal], ts: datetime, phase: str, prices: dict[str, float]) -> list[Decision]:
        out: list[Decision] = []
        for s in signals:
            self.signals.append(s)
            if s.agent_state is not None:
                self.agent_states.setdefault(s.agent_id, {})[s.contract_key] = s.agent_state.value
            if s.signal_type in (SignalType.BUY_CE, SignalType.BUY_PE) and s.status in (
                SignalStatus.ARMED, SignalStatus.TRIGGERED, SignalStatus.CANDIDATE
            ):
                self._recent.append((s.signal_ts, s.agent_id, s.underlying, s.option_type, s.contract_key))
            self.publish("signal", s.to_dict())

        for s in (x for x in signals if x.is_exit_trigger):
            order = self.execution.submit_exit(s.agent_id, s.contract_key, s.exit_reason or ExitReason.MANUAL, ts,
                                               prices.get(s.contract_key, s.trigger_price), phase)
            d = Decision(ts, s.signal_id, s.agent_id, s.contract_key,
                         "EXIT_SUBMITTED" if order else "LOGGED",
                         [s.exit_reason.value if s.exit_reason else "exit"], [])
            out.append(self._log(d))

        entries = [x for x in signals if x.is_entry_trigger]
        scores = {}
        if self.oc.mode == "C" and entries:
            regime = self.regime_fn(ts)
            scores = {e.signal_id: self.ranker.score(e.agent_id, regime) for e in entries}
            entries.sort(key=lambda e: -scores[e.signal_id][0])
        batch_conflicts = self._batch_conflicts(entries)
        for s in entries:
            out.append(self._route_entry(s, ts, phase, prices, scores.get(s.signal_id), batch_conflicts))
        return out

    # ---------------------------------------------------------- helpers
    def consensus_for(self, s: Signal) -> list[str]:
        """Other agents with a same-direction buy idea on the same underlying
        within the consensus window (analytics only)."""
        win = timedelta(minutes=self.oc.consensus_window_minutes)
        agents = {a for (t, a, u, ot, _k) in self._recent
                  if a != s.agent_id and u == s.underlying and ot == s.option_type and abs(s.signal_ts - t) <= win}
        return sorted(agents)

    def _batch_conflicts(self, entries: list[Signal]) -> set[str]:
        by_und: dict[str, set[OptionType]] = {}
        for e in entries:
            by_und.setdefault(e.underlying, set()).add(e.option_type)
        return {u for u, types in by_und.items() if len(types) > 1}

    def _opposite_exposure(self, s: Signal) -> Optional[str]:
        for p in self.execution.open_positions():
            if p.contract.underlying == s.underlying and p.contract.option_type != s.option_type:
                return f"open {p.contract.option_type.value} position {p.contract.key} ({p.agent_id})"
        return None

    def _reject(self, s: Signal, ts: datetime, reasons: list[str], consensus, conflict=None, score=None) -> Decision:
        d = Decision(ts, s.signal_id, s.agent_id, s.contract_key, "REJECTED", reasons, consensus, conflict, 0, score)
        self.on_reject(s.agent_id, s.contract_key, "; ".join(reasons), ts)
        return self._log(d)

    def _route_entry(self, s: Signal, ts: datetime, phase: str, prices: dict[str, float],
                     score: Optional[tuple[float, int]], batch_conflicts: set[str]) -> Decision:
        consensus = self.consensus_for(s)
        sc = score[0] if score else None
        if s.event_id:
            k = (s.agent_id, s.event_id)
            if k in self.seen_events:
                return self._reject(s, ts, ["DUPLICATE_EVENT: entry already processed for this event"], consensus, score=sc)
            self.seen_events.add(k)
        if self.execution.position_for(s.agent_id, s.contract_key) or \
                (s.agent_id, s.contract_key) in self.execution.pending_entries:
            return self._reject(s, ts, ["DUPLICATE_POSITION: agent already holds/pending this contract"], consensus, score=sc)
        other = self.execution.contract_held_by_other(s.agent_id, s.contract_key)
        if other and not self.oc.allow_multiple_agents_same_contract:
            return self._reject(s, ts, [f"DUPLICATE_POSITION: contract already held by {other}"], consensus, score=sc)
        if self.oc.mode == "C" and self.oc.block_negative_expectancy and score and score[1] >= self.oc.ranking_min_trades \
                and score[0] < 0:
            return self._reject(s, ts, [f"MODE_C: negative out-of-sample expectancy ({score[0]:.2f})"], consensus, score=sc)

        conflict = None
        opp = self._opposite_exposure(s)
        if s.underlying in batch_conflicts:
            conflict = "simultaneous BUY_CE and BUY_PE on the same underlying"
        elif opp:
            conflict = f"opposite exposure: {opp}"
        if conflict:
            self.conflicts.append({"ts": ts.isoformat(), "signal_id": s.signal_id, "agent_id": s.agent_id,
                                   "conflict": conflict, "policy": self.oc.conflict_policy})
            pol = self.oc.conflict_policy
            if pol == "block_both" or (pol == "first_come" and opp):
                return self._reject(s, ts, [f"CONFLICT ({pol}): {conflict}"], consensus, conflict, sc)

        contract = self.contracts[s.contract_key]
        price = s.trigger_price if s.trigger_price is not None else prices.get(s.contract_key)
        decision = self.risk.evaluate_entry(s, price, ts, self.execution.open_positions(), contract.lot_size)
        if not decision.approved:
            return self._reject(s, ts, ["RISK: " + r for r in decision.reasons], consensus, conflict, sc)
        self.execution.submit_entry(s, contract, decision.qty, ts, price, phase, decision.risk_amount)
        d = Decision(ts, s.signal_id, s.agent_id, s.contract_key, "SUBMITTED", decision.reasons, consensus,
                     conflict, decision.qty, sc)
        return self._log(d)

    def _log(self, d: Decision) -> Decision:
        self.decisions.append(d)
        self.publish("decision", d.to_dict())
        return d

    # ---------------------------------------------------------- analytics
    def performance(self) -> dict[str, Any]:
        per: dict[str, dict[str, float]] = {}
        for t in self.execution.trades:
            p = per.setdefault(t.agent_id, {"trades": 0, "wins": 0, "net_pnl": 0.0})
            p["trades"] += 1
            p["wins"] += 1 if t.net_pnl > 0 else 0
            p["net_pnl"] = round(p["net_pnl"] + t.net_pnl, 2)
        for p in per.values():
            p["win_rate"] = round(100.0 * p["wins"] / p["trades"], 2) if p["trades"] else 0.0
        return {"by_agent": per, "ranking": self.ranker.table(), "mode": self.oc.mode}
