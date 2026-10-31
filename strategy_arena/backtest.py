"""Vectorized backtest: the position chosen at the close of day t earns day t+1's return."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    returns: pd.Series    # per-bar strategy returns, net of costs
    positions: pd.Series  # target positions (decided at each close)
    equity: pd.Series     # growth of $1
    fills: pd.Series | None = None  # exit prices for exits inside a bar (e.g. a take-profit)
    costs: pd.Series | None = None  # trading costs charged per bar, as a fraction of capital


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Bars per year, used to annualize: 252 for daily bars, 252 x bars-per-session intraday."""
    from .indicators import bars_per_session

    return TRADING_DAYS * bars_per_session(index)


def clip_positions(positions: pd.Series, allow_short: bool) -> pd.Series:
    return positions.fillna(0.0).clip(-1.0 if allow_short else 0.0, 1.0)


def run_backtest(close: pd.Series, positions: pd.Series, cost_bps: float = 5.0,
                 allow_short: bool = False, fills: pd.Series | None = None) -> BacktestResult:
    """`fills[t]`, when set, is the price at which the position held into bar t was closed
    inside that bar (a target hit); that bar then earns fill / previous close - 1 instead of
    the close-to-close move. The agent must be flat at the close of a bar it fills on."""
    pos = clip_positions(positions.reindex(close.index), allow_short)
    exit_px = close if fills is None else fills.reindex(close.index).fillna(close)
    asset = (exit_px / close.shift(1) - 1).fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    costs = turnover * cost_bps / 1e4
    rets = held * asset - costs
    return BacktestResult(rets, pos, (1 + rets).cumprod(), fills, costs)


def excess_tstat(a: pd.Series, b: pd.Series, block: int) -> float:
    """t-statistic of strategy a's return over strategy b. Returns are summed in
    non-overlapping blocks of `block` bars first, so autocorrelation does not inflate it."""
    d = (a - b).dropna().to_numpy()
    n = len(d) // max(block, 1)
    if n < 3:
        return float("nan")
    sums = d[len(d) - n * block:].reshape(n, block).sum(axis=1)
    sd = sums.std(ddof=1)
    return float(sums.mean() / sd * np.sqrt(n)) if sd > 0 else float("nan")


def metrics(returns: pd.Series, positions: pd.Series | None = None, bars_per_year: float = TRADING_DAYS,
            costs: pd.Series | None = None) -> dict:
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {}
    equity = (1 + r).cumprod()
    total = equity.iloc[-1] - 1
    years = n / bars_per_year
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(bars_per_year)
    sharpe = r.mean() / r.std() * np.sqrt(bars_per_year) if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = r.mean() / downside * np.sqrt(bars_per_year) if downside and downside > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()
    out = {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0,
    }
    if positions is not None:
        p = positions.reindex(r.index).fillna(0.0)
        out["trades"] = int((p.diff().abs() > 0).sum())
        out["exposure"] = float((p != 0).mean())
    if costs is not None:
        out["costs"] = float(costs.reindex(r.index).fillna(0.0).sum())
    return out
