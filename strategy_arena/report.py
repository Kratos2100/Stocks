"""Plain-text and JSON reports."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .agents import Agent
from .selector import CASH, SelectorReport
from .tournament import Tournament


def _pct(x):
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+6.1f}%"


def _table(df: pd.DataFrame, cols: dict[str, tuple[str, str]]) -> str:
    """cols maps column -> (header, kind) where kind is pct|num|int."""
    name_w = max(len(str(i)) for i in df.index) + 2
    head = f"{'':<{name_w}}" + "".join(f"{h:>10}" for h, _ in cols.values())
    lines = [head, "-" * len(head)]
    for name, row in df.iterrows():
        cells = []
        for c, (_, kind) in cols.items():
            v = row.get(c)
            cells.append(_pct(v) if kind == "pct" else f"{v:.2f}" if kind == "num" else f"{int(v)}")
        lines.append(f"{name:<{name_w}}" + "".join(f"{x:>10}" for x in cells))
    return "\n".join(lines)


def suggested_strategy(tour: Tournament, sel: SelectorReport) -> tuple[str, str]:
    """Decide what to tell the user, and why."""
    rec, diag, comp = sel.recommendation, sel.diagnostics, sel.comparison
    champion = tour.leaderboard.index[0]
    meta_sharpe = comp.loc["Regression Selector", "sharpe"]
    bh_sharpe = comp.loc["Buy & Hold", "sharpe"] if "Buy & Hold" in comp.index else -np.inf
    trusted = diag["rank_ic"] > 0 and meta_sharpe >= bh_sharpe
    if trusted:
        return rec.agent, ("the regression selector's picks were positively correlated with what actually "
                           f"won (rank IC {diag['rank_ic']:+.2f}) and it out-traded Buy & Hold walk-forward, "
                           "so its pick for today's regime is the suggestion")
    return champion, ("the regression selector did not prove itself on this stock (rank IC "
                      f"{diag['rank_ic']:+.2f}, selector Sharpe {meta_sharpe:.2f} vs Buy & Hold {bh_sharpe:.2f}), "
                      "so the suggestion falls back to the tournament champion by Sharpe")


def text_report(ticker: str, df: pd.DataFrame, agents: list[Agent], tour: Tournament, sel: SelectorReport,
                cost_bps: float, allow_short: bool) -> str:
    out = []
    out.append(f"=== Strategy Arena: {ticker} ===")
    out.append(f"Prices {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} ({len(df)} days). "
               f"Evaluated from {tour.start:%Y-%m-%d}, cost {cost_bps:g} bps per unit traded, "
               f"{'long/short' if allow_short else 'long/flat only'}.")
    out.append("")
    out.append(f"--- Tournament leaderboard ({len(tour.rounds)} rounds) ---")
    out.append(_table(tour.leaderboard, {
        "cagr": ("CAGR", "pct"), "sharpe": ("Sharpe", "num"), "max_drawdown": ("MaxDD", "pct"),
        "exposure": ("InMarket", "pct"), "trades": ("Trades", "int"),
        "rounds_won": ("RoundsWon", "int"), "points": ("Points", "int")}))
    out.append("")
    d = sel.diagnostics
    out.append(f"--- Regression selector, walk-forward from {sel.picks.index[0]:%Y-%m-%d} "
               f"({d['decisions']} decisions, {sel.recommendation.horizon}-day horizon) ---")
    out.append(_table(sel.comparison, {
        "cagr": ("CAGR", "pct"), "sharpe": ("Sharpe", "num"), "max_drawdown": ("MaxDD", "pct"),
        "exposure": ("InMarket", "pct")}))
    out.append(f"rank IC {d['rank_ic']:+.3f} | picked an above-median agent {d['hit_rate_vs_median']:.0%} "
               f"| picked a top-3 agent {d['top3_rate']:.0%} | avg regret {_pct(d['avg_regret'])}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(d["pick_counts"].items(), key=lambda kv: -kv[1]))
    out.append(f"picks: {counts}")
    out.append("")
    rec = sel.recommendation
    out.append(f"--- Regime forecast as of {rec.as_of:%Y-%m-%d}: predicted next {rec.horizon} days ---")
    for name, v in rec.predicted.items():
        marker = " <- regression pick" if name == rec.agent else ""
        out.append(f"  {name:<22}{_pct(np.expm1(v))}{marker}")
    if rec.agent == CASH:
        out.append("  every agent is predicted to lose money: the regression pick is to stay in cash")
    if rec.drivers:
        out.append(f"  why {rec.agent}: " + "; ".join(
            f"{f}={v:.3g} ({'+' if c >= 0 else '-'}{abs(c) * 100:.2f}%)" for f, v, c in rec.drivers))
    out.append("")
    pick, why = suggested_strategy(tour, sel)
    styles = {a.name: a.style for a in agents}
    out.append(f">>> Suggested strategy for {ticker}: {pick}" + (f" ({styles[pick]})" if pick in styles else ""))
    out.append(f"    because {why}.")
    out.append("    Backtests are not promises: this is research, not financial advice.")
    return "\n".join(out)


def json_report(ticker: str, tour: Tournament, sel: SelectorReport) -> str:
    pick, why = suggested_strategy(tour, sel)
    rec = sel.recommendation
    payload = {
        "ticker": ticker,
        "as_of": f"{rec.as_of:%Y-%m-%d}",
        "suggested_strategy": pick,
        "reason": why,
        "regression_pick": rec.agent,
        "predicted_forward_return": {k: float(np.expm1(v)) for k, v in rec.predicted.items()},
        "horizon_days": rec.horizon,
        "drivers": [{"feature": f, "value": v, "contribution": c} for f, v, c in rec.drivers],
        "leaderboard": tour.leaderboard.reset_index().to_dict(orient="records"),
        "selector_comparison": sel.comparison.reset_index().to_dict(orient="records"),
        "selector_diagnostics": sel.diagnostics,
    }
    return json.dumps(payload, indent=2, default=float)
