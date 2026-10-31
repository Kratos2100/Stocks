"""Plain-text and JSON reports."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .agents import Agent
from .backtest import excess_tstat
from .indicators import bars_per_session, has_volume, is_intraday, session_ids
from .selector import BUY_HOLD, CASH, SelectorReport
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
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("n/a")
            else:
                cells.append(_pct(v) if kind == "pct" else f"{v:.2f}" if kind == "num" else f"{int(v)}")
        lines.append(f"{name:<{name_w}}" + "".join(f"{x:>10}" for x in cells))
    return "\n".join(lines)


# The selector's edge over Buy & Hold must clear this t-statistic (about 95% confidence)
TRUST_T = 2.0


def _fmt_t(t: float) -> str:
    return "n/a" if np.isnan(t) else f"{t:+.1f}"


def suggested_strategy(tour: Tournament, sel: SelectorReport, ticker: str = "this market") -> tuple[str, str]:
    """Decide what to tell the user, and why."""
    rec, diag = sel.recommendation, sel.diagnostics
    champion = tour.leaderboard.index[0]
    t_sel = diag.get("tstat_vs_buy_hold", float("nan"))
    if diag["rank_ic"] > 0 and t_sel >= TRUST_T:
        return rec.agent, ("the regression selector's picks were positively correlated with what actually "
                           f"won (rank IC {diag['rank_ic']:+.2f}) and it beat Buy & Hold walk-forward by a "
                           f"statistically significant margin (t = {_fmt_t(t_sel)}), so its pick for today's "
                           "regime is the suggestion")
    why = (f"the regression selector did not prove itself on {ticker} (rank IC {diag['rank_ic']:+.2f}, "
           f"t = {_fmt_t(t_sel)} vs Buy & Hold; it needs a positive rank IC and t >= {TRUST_T:g}), "
           "so the suggestion falls back to the tournament champion by Sharpe")
    if champion != BUY_HOLD and BUY_HOLD in tour.results:
        start = tour.start
        t_champ = excess_tstat(tour.results[champion].returns.loc[start:],
                               tour.results[BUY_HOLD].returns.loc[start:], rec.horizon)
        verdict = "significant" if t_champ >= TRUST_T else "NOT significant: treat it as no better than Buy & Hold"
        why += f". Its edge over Buy & Hold: t = {_fmt_t(t_champ)} ({verdict})"
    return champion, why


def text_report(ticker: str, df: pd.DataFrame, agents: list[Agent], tour: Tournament, sel: SelectorReport,
                cost_bps: float, allow_short: bool, square_off: bool = False) -> str:
    k = bars_per_session(df.index)
    fmt = "%Y-%m-%d %H:%M" if k > 1 else "%Y-%m-%d"
    bars = f"{len(df)} bars, ~{k:.0f} per session" if k > 1 else f"{len(df)} daily bars"
    out = []
    out.append(f"=== Strategy Arena: {ticker} ===")
    out.append(f"Prices {df.index[0]:{fmt}} -> {df.index[-1]:{fmt}} ({bars}). "
               f"Evaluated from {tour.start:{fmt}}, cost {cost_bps:g} bps per unit traded, "
               f"{'long/short' if allow_short else 'long/flat only (add --allow-short for futures/puts)'}.")
    if is_intraday(df.index):
        out.append("Every position is squared off at each session's last bar." if square_off else
                   "Positions may be held overnight (--hold-overnight).")
    if not has_volume(df):
        out.append("Note: no volume in this data (common for indices), so VWAP is an equal-weighted average price.")
    evaluated = df.loc[tour.start:]
    years = len(evaluated) / tour.bars_per_year
    if years < 1:
        sessions = len(np.unique(session_ids(evaluated.index)))
        out.append(f"WARNING: only {sessions} sessions (~{years * 12:.1f} months) were evaluated. That is too little "
                   "to trust any ranking; CAGR and Sharpe are annualized from it and exaggerate. Use more history "
                   "(e.g. --interval 1h gives ~2 years, or --csv with your broker's data).")
    out.append("")
    out.append(f"--- Tournament leaderboard ({len(tour.rounds)} rounds) ---")
    out.append(_table(tour.leaderboard, {
        "total_return": ("Return", "pct"), "cagr": ("CAGR", "pct"), "sharpe": ("Sharpe", "num"),
        "max_drawdown": ("MaxDD", "pct"), "exposure": ("InMarket", "pct"), "trades": ("Trades", "int"),
        "costs": ("Costs", "pct"), "rounds_won": ("RoundsWon", "int")}))
    out.append("Return is for the whole evaluated period, after costs; Costs is what trading costs took out.")
    out.append("")
    d = sel.diagnostics
    out.append(f"--- Regression selector, walk-forward from {sel.picks.index[0]:{fmt}} "
               f"({d['decisions']} decisions, {sel.recommendation.horizon}-bar horizon) ---")
    out.append(_table(sel.comparison, {
        "total_return": ("Return", "pct"), "cagr": ("CAGR", "pct"), "sharpe": ("Sharpe", "num"),
        "max_drawdown": ("MaxDD", "pct"), "exposure": ("InMarket", "pct"), "costs": ("Costs", "pct")}))
    out.append(f"vs Buy & Hold t = {_fmt_t(d.get('tstat_vs_buy_hold', float('nan')))} | rank IC {d['rank_ic']:+.3f} | picked an above-median agent {d['hit_rate_vs_median']:.0%} "
               f"| picked a top-3 agent {d['top3_rate']:.0%} | avg regret {_pct(d['avg_regret'])}")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(d["pick_counts"].items(), key=lambda kv: -kv[1]))
    out.append(f"picks: {counts}")
    out.append("")
    rec = sel.recommendation
    out.append(f"--- Regime forecast as of {rec.as_of:{fmt}}: predicted next {rec.horizon} bars ---")
    for name, v in rec.predicted.items():
        marker = " <- regression pick" if name == rec.agent else ""
        out.append(f"  {name:<22}{_pct(np.expm1(v))}{marker}")
    if rec.agent == CASH:
        out.append("  every agent is predicted to lose money: the regression pick is to stay in cash")
    if rec.drivers:
        out.append(f"  why {rec.agent}: " + "; ".join(
            f"{f}={v:.3g} ({'+' if c >= 0 else '-'}{abs(c) * 100:.2f}%)" for f, v, c in rec.drivers))
    out.append("")
    pick, why = suggested_strategy(tour, sel, ticker)
    styles = {a.name: a.style for a in agents}
    out.append(f">>> Suggested strategy for {ticker}: {pick}" + (f" ({styles[pick]})" if pick in styles else ""))
    out.append(f"    because {why}.")
    out.append("    Backtests are not promises: this is research, not financial advice.")
    return "\n".join(out)


def json_report(ticker: str, tour: Tournament, sel: SelectorReport) -> str:
    pick, why = suggested_strategy(tour, sel, ticker)
    rec = sel.recommendation
    payload = {
        "ticker": ticker,
        "as_of": str(rec.as_of),
        "suggested_strategy": pick,
        "reason": why,
        "regression_pick": rec.agent,
        "predicted_forward_return": {k: float(np.expm1(v)) for k, v in rec.predicted.items()},
        "horizon_bars": rec.horizon,
        "drivers": [{"feature": f, "value": v, "contribution": c} for f, v, c in rec.drivers],
        "leaderboard": tour.leaderboard.reset_index().to_dict(orient="records"),
        "selector_comparison": sel.comparison.reset_index().to_dict(orient="records"),
        "selector_diagnostics": sel.diagnostics,
    }
    return json.dumps(payload, indent=2, default=float)
