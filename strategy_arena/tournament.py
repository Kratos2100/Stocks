"""The arena: every agent trades the same stock over the same days; we score each round."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents import Agent
from .backtest import BacktestResult, metrics, run_backtest


@dataclass
class Tournament:
    results: dict[str, BacktestResult]  # full-history backtests, one per agent
    leaderboard: pd.DataFrame           # overall stats over the evaluation period
    rounds: pd.DataFrame                # per-round returns (rows: round end date, cols: agents)
    start: pd.Timestamp                 # first evaluated day (after indicator warm-up)


def run_agents(df: pd.DataFrame, agents: list[Agent], cost_bps: float, allow_short: bool) -> dict[str, BacktestResult]:
    return {a.name: run_backtest(df.Close, a.positions(df), cost_bps, allow_short) for a in agents}


def run_tournament(df: pd.DataFrame, agents: list[Agent], cost_bps: float = 5.0, allow_short: bool = False,
                   warmup: int = 252, round_len: int = 63) -> Tournament:
    if len(df) < warmup + round_len:
        raise ValueError(f"need at least {warmup + round_len} days of prices, got {len(df)}")
    results = run_agents(df, agents, cost_bps, allow_short)
    start = df.index[warmup]
    rets = pd.DataFrame({name: r.returns for name, r in results.items()}).loc[start:]

    round_id = np.arange(len(rets)) // round_len
    rounds = (1 + rets).groupby(round_id).prod() - 1
    rounds.index = rets.index.to_series().groupby(round_id).last().values
    rounds.index.name = "round_end"
    ranks = rounds.rank(axis=1, ascending=False, method="min")
    n = len(agents)

    rows = []
    for name, res in results.items():
        m = metrics(res.returns.loc[start:], res.positions.loc[start:])
        m["agent"] = name
        m["rounds_won"] = int((ranks[name] == 1).sum())
        m["avg_rank"] = float(ranks[name].mean())
        m["points"] = int((n - ranks[name] + 1).sum())  # n points for 1st place, 1 for last
        rows.append(m)
    board = pd.DataFrame(rows).set_index("agent").sort_values(["sharpe", "points"], ascending=False)
    return Tournament(results, board, rounds, start)
