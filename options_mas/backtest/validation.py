"""In-sample / out-of-sample splits, walk-forward validation and parameter
sensitivity analysis. All numbers come from actual simulations on the
supplied dataset – nothing is estimated or extrapolated."""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Optional

from ..config import AppConfig, set_param
from ..data.dataset import BacktestDataset
from .engine import Backtester
from .metrics import summarize

Objective = Callable[[dict[str, Any]], float]


def objective_net_pnl(summary: dict[str, Any]) -> float:
    return float(summary["net_pnl"])


def objective_profit_factor(summary: dict[str, Any]) -> float:
    pf = summary.get("profit_factor")
    if pf is None:
        return 0.0
    return min(float(pf), 10.0) if summary["total_trades"] >= 5 else 0.0


def split_sessions(sessions: list[date], oos_fraction: float = 0.3) -> tuple[list[date], list[date]]:
    """Chronological split: the out-of-sample block is always LATER in time."""
    n = len(sessions)
    k = max(1, int(round(n * (1 - oos_fraction))))
    return sessions[:k], sessions[k:]


def _grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(param_grid[k] for k in keys))]


def _run(cfg: AppConfig, ds: BacktestDataset, params: dict[str, Any], sessions: list[date],
         agents: Optional[list[str]]) -> dict[str, Any]:
    c = cfg.copy()
    for k, v in params.items():
        set_param(c, k, v)
    bt = Backtester(c, ds, enabled_agents=agents)
    for _ in bt.iter_run(sessions):
        pass
    res = bt.result()
    return {"summary": res.report["overall"], "by_agent": res.report["by_agent"], "trades": res.trades}


@dataclass
class WalkForwardResult:
    folds: list[dict[str, Any]]
    oos_summary: dict[str, Any]
    data_label: str

    def to_dict(self) -> dict[str, Any]:
        return {"data_label": self.data_label, "folds": self.folds, "out_of_sample_summary": self.oos_summary}

    def ranking_table(self) -> dict[str, Any]:
        """Out-of-sample per-agent stats in the format read by Mode C."""
        out: dict[str, Any] = {}
        for f in self.folds:
            for t in f["_oos_trades"]:
                for regime in ("ALL", t.regime):
                    s = out.setdefault(t.agent_id, {}).setdefault(regime, {"trades": 0, "net_pnl": 0.0})
                    s["trades"] += 1
                    s["net_pnl"] = round(s["net_pnl"] + t.net_pnl, 2)
        return out


def walk_forward(cfg: AppConfig, ds: BacktestDataset, param_grid: dict[str, list[Any]], train_sessions: int,
                 test_sessions: int, step: Optional[int] = None, objective: Objective = objective_net_pnl,
                 agents: Optional[list[str]] = None) -> WalkForwardResult:
    days = ds.sessions()
    step = step or test_sessions
    grid = _grid(param_grid) or [{}]
    folds = []
    oos_trades = []
    start = 0
    while start + train_sessions + test_sessions <= len(days):
        train = days[start:start + train_sessions]
        test = days[start + train_sessions:start + train_sessions + test_sessions]
        scored = []
        for params in grid:
            r = _run(cfg, ds, params, train, agents)
            scored.append((objective(r["summary"]), params, r["summary"]))
        scored.sort(key=lambda x: -x[0])
        best_score, best_params, best_is = scored[0]
        r = _run(cfg, ds, best_params, test, agents)
        oos_trades.extend(r["trades"])
        folds.append({
            "train": [train[0].isoformat(), train[-1].isoformat()],
            "test": [test[0].isoformat(), test[-1].isoformat()],
            "best_params": best_params, "in_sample": best_is, "out_of_sample": r["summary"],
            "_oos_trades": r["trades"],
        })
        start += step
    wf = WalkForwardResult(folds, summarize(oos_trades, cfg.risk.capital), ds.label)
    return wf


def walk_forward_json(wf: WalkForwardResult) -> str:
    d = wf.to_dict()
    d["folds"] = [{k: v for k, v in f.items() if k != "_oos_trades"} for f in d["folds"]]
    return json.dumps(d, indent=2, default=str)


def sensitivity(cfg: AppConfig, ds: BacktestDataset, param: str, values: list[Any],
                agents: Optional[list[str]] = None, sessions: Optional[list[date]] = None) -> list[dict[str, Any]]:
    days = sessions or ds.sessions()
    rows = []
    for v in values:
        r = _run(cfg, ds, {param: v}, days, agents)
        rows.append({"param": param, "value": v, **r["summary"]})
    return rows


def in_out_of_sample(cfg: AppConfig, ds: BacktestDataset, oos_fraction: float = 0.3,
                     agents: Optional[list[str]] = None) -> dict[str, Any]:
    ins, oos = split_sessions(ds.sessions(), oos_fraction)
    return {
        "data_label": ds.label,
        "in_sample_sessions": [ins[0].isoformat(), ins[-1].isoformat()] if ins else None,
        "out_of_sample_sessions": [oos[0].isoformat(), oos[-1].isoformat()] if oos else None,
        "in_sample": _run(cfg, ds, {}, ins, agents)["summary"] if ins else None,
        "out_of_sample": _run(cfg, ds, {}, oos, agents)["summary"] if oos else None,
    }
