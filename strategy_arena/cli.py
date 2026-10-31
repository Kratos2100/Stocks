"""Command line: python -m strategy_arena TSLA [--start 2015-01-01] ..."""
from __future__ import annotations

import argparse
import sys

from .agents import default_agents
from .data import load_prices, synthetic_prices
from .indicators import bars_per_session
from .report import json_report, text_report
from .selector import RegressionSelector
from .tournament import run_tournament


def bar_defaults(df) -> dict:
    """Round length, forecast horizon and decision step in bars.

    Daily: a quarter, a month, a week. Intraday: a week, a session, a sixth of a session.
    """
    k = bars_per_session(df.index)
    if k <= 1:
        return {"round_len": 63, "horizon": 21, "step": 5}
    k = int(round(k))
    return {"round_len": 5 * k, "horizon": k, "step": max(1, k // 6)}


def analyze(df, ticker, cost_bps=5.0, allow_short=False, horizon=None, step=None, round_len=None, allow_cash=True,
            square_off=True):
    """Run the tournament and the regression selector. `square_off` only affects intraday bars."""
    d = bar_defaults(df)
    horizon, step, round_len = horizon or d["horizon"], step or d["step"], round_len or d["round_len"]
    agents = default_agents(square_off)
    tour = run_tournament(df, agents, cost_bps, allow_short, round_len=round_len)
    sel = RegressionSelector(horizon=horizon, step=step, allow_cash=allow_cash).run(df, tour, cost_bps)
    return agents, tour, sel


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="strategy_arena",
                                description="Tiny trading agents compete on a stock; ridge regression picks the "
                                            "strategy that suits the stock's current regime.")
    p.add_argument("tickers", nargs="*", help="ticker symbols, e.g. TSLA GME")
    p.add_argument("--csv", help="OHLCV CSV with a Date column (instead of downloading)")
    p.add_argument("--synthetic", action="store_true", help="use a synthetic regime-switching market (offline demo)")
    p.add_argument("--interval", default="1d",
                   help="bar size: 1d (default), 1h, 30m, 15m, 5m, 1m. yfinance keeps only the last 60 days of "
                        "5m/15m bars and 730 days of 1h bars")
    p.add_argument("--start", default="2012-01-01", help="first date for daily bars")
    p.add_argument("--end")
    p.add_argument("--cost-bps", type=float, default=5.0, help="trading cost per unit of turnover, in basis points")
    p.add_argument("--allow-short", action="store_true", help="let agents go short (default: long or flat)")
    p.add_argument("--horizon", type=int, help="bars ahead the regression predicts (default: 21 daily bars, "
                                                "or one session intraday)")
    p.add_argument("--step", type=int, help="bars between selector decisions (default: 5 daily bars, "
                                             "or a sixth of a session intraday)")
    p.add_argument("--round-len", type=int, help="bars per tournament round (default: 63 daily bars, "
                                                  "or 5 sessions intraday)")
    p.add_argument("--hold-overnight", action="store_true",
                   help="intraday: keep positions across sessions (default: square off at every session's close)")
    p.add_argument("--no-cash", action="store_true", help="selector must always pick an agent")
    p.add_argument("--json", help="also write a JSON report to this path (one ticker) or prefix (several)")
    args = p.parse_args(argv)

    jobs = []
    if args.synthetic:
        intraday = args.interval.endswith("m") or args.interval.endswith("h")
        jobs.append(("SYNTHETIC", synthetic_prices(78 * 120, bars_per_day=78) if intraday else synthetic_prices()))
    elif args.csv:
        jobs.append((args.tickers[0] if args.tickers else args.csv, load_prices(csv=args.csv, start=args.start, end=args.end)))
    elif args.tickers:
        for t in args.tickers:
            try:
                jobs.append((t.upper(), load_prices(t, args.start, args.end, interval=args.interval)))
            except Exception as exc:
                print(f"skipping {t}: {exc}", file=sys.stderr)
    else:
        p.error("give at least one ticker, --csv PATH or --synthetic")

    for i, (ticker, df) in enumerate(jobs):
        square_off = not args.hold_overnight
        agents, tour, sel = analyze(df, ticker, args.cost_bps, args.allow_short, args.horizon, args.step,
                                    args.round_len, not args.no_cash, square_off)
        if i:
            print("\n")
        print(text_report(ticker, df, agents, tour, sel, args.cost_bps, args.allow_short, square_off))
        if args.json:
            path = args.json if len(jobs) == 1 else f"{args.json}_{ticker}.json"
            with open(path, "w") as f:
                f.write(json_report(ticker, tour, sel))
    return 0 if jobs else 1
