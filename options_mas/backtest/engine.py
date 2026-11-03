"""Event-driven historical backtester.

Bias controls
-------------
* **Look-ahead / repainting** – agents only ever receive completed candles via
  ``on_candle_close`` and intrabar prices via ``on_price``; indicators are
  streaming. Warm-up uses candles strictly before the session.
* **Unrealistic fills** – entries fill at the trigger level when the simulated
  intrabar path crosses it, or at the bar OPEN if the bar gaps through it.
  Close-evaluated signals fill at the NEXT bar open (configurable). Slippage,
  tick rounding, a participation cap (vs the *previous* bar's volume) and
  full Indian F&O charges are applied.
* **Intrabar ordering** – with OHLC-only data the path is O→L→H→C (bullish
  bar) or O→H→L→C (bearish bar). Price-based exits are not assumed inside the
  bar where the entry filled unless ``intrabar_exit_on_entry_bar`` is set.
  When 1-minute sub-bars are supplied the path is built from them instead.
* **Survivorship** – contracts are taken from historical data/snapshots for
  each session (expired contracts included); selection uses only snapshots at
  or before the selection time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Iterator, Optional

from ..agents import build_agents
from ..config import AppConfig
from ..contract_selection import ContractSelector
from ..data.dataset import BacktestDataset, classify_regimes, previous_session_regimes
from ..execution.broker import SimulatedBroker
from ..models import Candle, OptionType, Trade, reset_ids
from ..system import EventBus, TradingSystem
from .metrics import equity_drawdown, full_report

ContractPlan = dict[date, list[str]]


def ohlc_path(c: Candle) -> list[float]:
    pts = [c.open, c.low, c.high, c.close] if c.close >= c.open else [c.open, c.high, c.low, c.close]
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    return out


@dataclass
class BacktestResult:
    data_label: str
    agents: list[str]
    sessions: list[str]
    trades: list[Trade]
    report: dict[str, Any]
    equity_curve: list[tuple[str, float]]
    mtm_max_drawdown: float
    signals: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    selections: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, include_signals: bool = False) -> dict[str, Any]:
        d = {
            "disclaimer": ("Simulated results on the dataset labelled below. Not a guarantee of future "
                           "performance. Results on SYNTHETIC data are NOT real market performance."),
            "data_label": self.data_label,
            "agents": self.agents,
            "sessions": self.sessions,
            "report": self.report,
            "mtm_max_drawdown": round(self.mtm_max_drawdown, 2),
            "equity_curve": self.equity_curve,
            "trades": [t.to_dict() for t in self.trades],
            "conflicts": self.conflicts,
            "selections": self.selections,
            "warnings": self.warnings,
        }
        if include_signals:
            d["signals"] = self.signals
            d["decisions"] = self.decisions
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(**kw), indent=2, default=str)


class Backtester:
    def __init__(self, cfg: AppConfig, dataset: BacktestDataset, enabled_agents: Optional[list[str]] = None,
                 plan: Optional[ContractPlan] = None, selection_time: time = time(9, 20),
                 bus: Optional[EventBus] = None):
        self.cfg = cfg.copy()
        self.cfg.execution.mode = "backtest"
        self.ds = dataset
        self.enabled = enabled_agents
        self.plan = plan
        self.selection_time = selection_time
        self.bus = bus or EventBus()
        self.selections: dict[str, Any] = {}
        self.system: Optional[TradingSystem] = None

    # ------------------------------------------------------------ plan
    def _plan_for(self, day: date) -> list[str]:
        if self.plan is not None:
            return [k for k in self.plan.get(day, []) if k in self.ds.candles]
        if self.ds.chains is not None:
            return self._select(day)
        return self.ds.contracts_on(day)

    def _select(self, day: date) -> list[str]:
        cfg = self.cfg
        sel = ContractSelector(cfg.selection, {k: v.lot_size for k, v in cfg.instruments.items()},
                               {k: v.strike_step for k, v in cfg.instruments.items()})
        at = datetime.combine(day, self.selection_time)
        keys, rec = [], {}
        for und in sorted(set(self.ds.chains["underlying"])):
            res = sel.select_from_snapshots(self.ds.chains, und, at)
            for ot, s in res.items():
                rec[f"{und}:{ot.value}"] = s.to_dict() | {"rejected": len(s.rejected)}
                if s.contract and s.contract.key in self.ds.candles:
                    keys.append(s.contract.key)
        self.selections[day.isoformat()] = rec
        return keys

    # ------------------------------------------------------------- run
    def run(self) -> BacktestResult:
        for _ in self.iter_run():
            pass
        return self.result()

    def iter_run(self, sessions: Optional[list[date]] = None) -> Iterator[datetime]:
        """Generator that yields after every bar timestamp (used by the
        dashboard replay). Call :meth:`result` when exhausted."""
        reset_ids()
        cfg = self.cfg
        agents = build_agents(cfg, self.enabled)
        broker = SimulatedBroker(cfg.execution)
        system = TradingSystem(cfg, agents, broker, self.bus)
        system.suppress_after_entry = not cfg.backtest.intrabar_exit_on_entry_bar
        regimes = classify_regimes(self.ds.underlying)
        system.regime_by_date = regimes
        system.decision_regime_by_date = previous_session_regimes(regimes)
        self.system = system
        self._equity: list[tuple[str, float]] = []
        self._sessions_run: list[str] = []
        days = sessions or self.ds.sessions()
        tf = cfg.session.timeframe_min
        for day in days:
            keys = self._plan_for(day)
            if not keys:
                continue
            self._sessions_run.append(day.isoformat())
            current = set(keys)
            for k in list(system.monitors):
                if k not in current:
                    system.unregister_contract(k)
            for k in keys:
                contract = self.ds.contracts[k]
                ctxs = [a.contexts[k] for a in agents.values() if a.has_contract(k)]
                if not ctxs:
                    warm = self.ds.warmup(k, day, cfg.backtest.warmup_sessions)
                else:  # re-monitored contract: feed only the history it missed
                    last = max((cx.candles[-1].ts for cx in ctxs if cx.candles), default=None)
                    warm = [c for c in self.ds.warmup(k, day, 10_000) if last is None or c.ts > last]
                system.register_contract(contract, warm)
            bars: dict[datetime, list[tuple[str, Candle]]] = {}
            for k in keys:
                for c in self.ds.by_date(k).get(day, []):
                    bars.setdefault(c.ts, []).append((k, c))
            prev_vol: dict[str, float] = {}
            last_ts = None
            for ts in sorted(bars):
                items = sorted(bars[ts], key=lambda kc: kc[0])
                for key, candle in items:
                    broker.set_liquidity(key, prev_vol.get(key))
                    self._walk_bar(system, key, candle, tf)
                    prev_vol[key] = candle.volume
                system.on_candles_closed(items)
                last_ts = ts + timedelta(minutes=tf)
                eq = system.risk.cumulative_realized + sum(p.unrealized_pnl() for p in system.execution.open_positions())
                self._equity.append((last_ts.isoformat(), round(cfg.risk.capital + eq, 2)))
                yield last_ts
            if last_ts is not None:
                system.end_session(last_ts)

    def _walk_bar(self, system: TradingSystem, key: str, candle: Candle, tf: int) -> None:
        subs = self.ds.subbars_for(key, candle)
        units = [(s.ts, ohlc_path(s), 1) for s in subs] if subs else [(candle.ts, ohlc_path(candle), tf)]
        for start, verts, minutes in units:
            system.begin_bar()
            step = minutes * 60.0 / (len(verts) + 1)
            n = 0

            def emit(price: float) -> None:
                nonlocal n
                t = start + timedelta(seconds=min(step * n, minutes * 60 - 1))
                n += 1
                system.on_price(key, price, t)

            cur = verts[0]
            emit(cur)
            for nxt in verts[1:]:
                done: set[float] = set()
                while True:
                    lv = [l for l in system.price_levels(key) if l not in done and
                          ((cur < l < nxt) if nxt > cur else (nxt < l < cur))]
                    if not lv:
                        break
                    l = min(lv) if nxt > cur else max(lv)
                    done.add(l)
                    emit(l)
                    cur = l
                emit(nxt)
                cur = nxt

    def result(self) -> BacktestResult:
        s = self.system
        assert s is not None, "run() first"
        trades = list(s.execution.trades)
        warnings = []
        if "SYNTHETIC" in self.ds.label.upper():
            warnings.append("SYNTHETIC DATA – results are NOT real market performance.")
        if not self.ds.subbars:
            warnings.append("OHLC-only intrabar model: fill ordering inside bars is an assumption.")
        a02 = self.cfg.agent02
        if (self.enabled is None or "A02" in self.enabled) and a02.enabled and a02.pe_mode == "as_specified":
            warnings.append("Agent 02 PE uses pe_mode='as_specified' (breakdown on PE premium, entry above "
                            "breakdown candle high). See docs/STRATEGIES.md for the economic caveat.")
        mdd, _ = equity_drawdown(v for _, v in self._equity) if self._equity else (0.0, 0.0)
        return BacktestResult(
            data_label=self.ds.label, agents=list(s.agents), sessions=self._sessions_run, trades=trades,
            report=full_report(trades, self.cfg.risk.capital), equity_curve=self._equity, mtm_max_drawdown=mdd,
            signals=[x.to_dict() for x in s.orchestrator.signals],
            decisions=[d.to_dict() for d in s.orchestrator.decisions],
            conflicts=list(s.orchestrator.conflicts), selections=self.selections, warnings=warnings,
        )
