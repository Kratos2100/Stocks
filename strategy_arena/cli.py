"""Command line: python -m strategy_arena TSLA [--start 2015-01-01] ..."""
from __future__ import annotations

import argparse
import sys

from .agents import default_agents
from .data import load_prices, synthetic_prices
from .report import json_report, text_report
from .selector import RegressionSelector
from .tournament import run_tournament


def analyze(df, ticker, cost_bps=5.0, allow_short=False, horizon=21, step=5, round_len=63, allow_cash=True):
    agents = default_agents()
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
    p.add_argument("--start", default="2012-01-01")
    p.add_argument("--end")
    p.add_argument("--cost-bps", type=float, default=5.0, help="trading cost per unit of turnover, in basis points")
    p.add_argument("--allow-short", action="store_true", help="let agents go short (default: long or flat)")
    p.add_argument("--horizon", type=int, default=21, help="days ahead the regression predicts")
    p.add_argument("--step", type=int, default=5, help="days between selector decisions")
    p.add_argument("--round-len", type=int, default=63, help="days per tournament round")
    p.add_argument("--no-cash", action="store_true", help="selector must always pick an agent")
    p.add_argument("--json", help="also write a JSON report to this path (one ticker) or prefix (several)")
    args = p.parse_args(argv)

    jobs = []
    if args.synthetic:
        jobs.append(("SYNTHETIC", synthetic_prices()))
    elif args.csv:
        jobs.append((args.tickers[0] if args.tickers else args.csv, load_prices(csv=args.csv, start=args.start, end=args.end)))
    elif args.tickers:
        for t in args.tickers:
            try:
                jobs.append((t.upper(), load_prices(t, args.start, args.end)))
            except Exception as exc:
                print(f"skipping {t}: {exc}", file=sys.stderr)
    else:
        p.error("give at least one ticker, --csv PATH or --synthetic")

    for i, (ticker, df) in enumerate(jobs):
        agents, tour, sel = analyze(df, ticker, args.cost_bps, args.allow_short, args.horizon, args.step,
                                    args.round_len, not args.no_cash)
        if i:
            print("\n")
        print(text_report(ticker, df, agents, tour, sel, args.cost_bps, args.allow_short))
        if args.json:
            path = args.json if len(jobs) == 1 else f"{args.json}_{ticker}.json"
            with open(path, "w") as f:
                f.write(json_report(ticker, tour, sel))
    return 0 if jobs else 1
