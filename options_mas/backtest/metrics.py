"""Performance metrics computed strictly from simulated trades. Nothing here
invents or extrapolates results: with zero trades every metric is zero/None."""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from ..models import Trade


def _max_drawdown(values: list[float]) -> tuple[float, float]:
    peak = -math.inf
    mdd = 0.0
    mdd_pct = 0.0
    for v in values:
        peak = max(peak, v)
        dd = peak - v
        if dd > mdd:
            mdd = dd
            mdd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
    return mdd, mdd_pct


def summarize(trades: list[Trade], starting_capital: float = 0.0) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "win_rate_pct": 0.0,
                "net_pnl": 0.0, "gross_pnl": 0.0, "charges": 0.0, "profit_factor": None, "expectancy": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "avg_holding_minutes": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_risk_reward": None, "avg_r_multiple": None,
                "max_consecutive_losses": 0, "max_consecutive_wins": 0, "largest_win": 0.0, "largest_loss": 0.0}
    nets = [t.net_pnl for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    gp, gl = sum(wins), -sum(losses)
    eq = [starting_capital]
    for x in sorted(trades, key=lambda t: t.exit_ts):
        eq.append(eq[-1] + x.net_pnl)
    mdd, mdd_pct = _max_drawdown(eq)
    streak = max_l = max_w = wstreak = 0
    for x in (t.net_pnl for t in sorted(trades, key=lambda t: t.exit_ts)):
        if x <= 0:
            streak += 1
            wstreak = 0
        else:
            wstreak += 1
            streak = 0
        max_l, max_w = max(max_l, streak), max(max_w, wstreak)
    avg_win = gp / len(wins) if wins else 0.0
    avg_loss = -gl / len(losses) if losses else 0.0
    rs = [t.r_multiple for t in trades if t.r_multiple is not None]
    return {
        "total_trades": n,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "net_pnl": round(sum(nets), 2),
        "gross_pnl": round(sum(t.gross_pnl for t in trades), 2),
        "charges": round(sum(t.charges for t in trades), 2),
        # None when there are no losing trades (undefined, never reported as infinite)
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "profit_factor_note": None if gl > 0 else "undefined: no losing trades",
        "expectancy": round(sum(nets) / n, 2),
        "max_drawdown": round(mdd, 2),
        "max_drawdown_pct": round(mdd_pct, 3),
        "avg_holding_minutes": round(sum(t.holding_minutes for t in trades) / n, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_risk_reward": round(avg_win / abs(avg_loss), 3) if avg_loss else None,
        "avg_r_multiple": round(sum(rs) / len(rs), 3) if rs else None,
        "max_consecutive_losses": max_l,
        "max_consecutive_wins": max_w,
        "largest_win": round(max(nets), 2),
        "largest_loss": round(min(nets), 2),
    }


def breakdown(trades: list[Trade], key) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Trade]] = {}
    for t in trades:
        groups.setdefault(str(key(t)), []).append(t)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def full_report(trades: list[Trade], starting_capital: float) -> dict[str, Any]:
    return {
        "overall": summarize(trades, starting_capital),
        "by_agent": breakdown(trades, lambda t: t.agent_id),
        "by_option_type": breakdown(trades, lambda t: t.option_type.value),
        "by_agent_and_option_type": breakdown(trades, lambda t: f"{t.agent_id}:{t.option_type.value}"),
        "by_underlying": breakdown(trades, lambda t: t.underlying),
        "by_regime": breakdown(trades, lambda t: t.regime),
        "by_exit_reason": breakdown(trades, lambda t: t.exit_reason.value),
    }


def equity_drawdown(equity: Iterable[float]) -> tuple[float, float]:
    return _max_drawdown(list(equity))
