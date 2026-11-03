# Getting real options data

Every downloader writes the long-format CSV that `backtest --data` reads, so
the workflow is always **fetch → backtest**. Credentials are read from
environment variables only. Never put tokens in config files or commit them.

| Source | Library | Account / cost | 5-min intraday | Expired contracts | Best for |
|---|---|---|---|---|---|
| NSE F&O bhavcopy | `nselib` | free | no (end of day) | yes | pivots' previous-session H/L/C, contract universe, picking liquid strikes |
| NSE live option chain | `nselib` | free | snapshots only | no | contract selection, building a free forward archive |
| **Upstox** | `requests` | free API app; the expired-instruments API may need a paid plan | yes, with OI | **yes** | **backtesting** (5-min history of expired weeklies) |
| Zerodha Kite | `kiteconnect` | Kite Connect + historical add-on (paid) | yes, with OI | no | paper/live on Kite; build an archive daily |
| Dhan | `dhanhq` | DhanHQ API (data plan) | yes, with OI | rolling ATM-relative only | paper/live on Dhan |

`yfinance` and similar global sources do **not** carry NSE option contracts.

## Install

```bash
pip install -e ".[data]"        # nselib, kiteconnect, dhanhq
```

## Free: NSE bhavcopy and option chain

```bash
# end-of-day F&O bhavcopy for a date range (holidays are skipped)
python -m options_mas.cli fetch bhavcopy --start 2026-10-01 --end 2026-10-31 --out data/bhavcopy_fo.csv

# one live option-chain snapshot (feeds contract selection: --chains)
python -m options_mas.cli fetch chain --symbol NIFTY --out data/chains.csv

# record the chain all day (run on a machine that can reach nseindia.com), then build candles
python -m options_mas.cli fetch record --symbols NIFTY,BANKNIFTY --every 30 --out-dir data/recorded
python -m options_mas.cli fetch snapshots-to-candles --chains data/recorded/chains.csv --out data/recorded/options_5m.csv
```

Candles built from snapshots are **sampled**: high and low only reflect the
moments a snapshot was taken, so they understate the true range. Volume is
the growth of NSE's cumulative traded volume. Label such backtests
`--label "NSE_CHAIN_SNAPSHOTS (sampled)"`. NSE's site is rate-limited and
its terms of use apply, so keep `--every` at 30 seconds or more.

## Upstox (recommended for backtesting)

1. Create an app at the Upstox developer console and complete the OAuth login
   to obtain an access token. Tokens expire daily.
2. Run:

```bash
export UPSTOX_ACCESS_TOKEN=...            # never commit this
# an expired weekly expiry, ATM ± range, plus index candles for regime analytics
python -m options_mas.cli fetch upstox --underlying NIFTY --expiry 2026-10-28 \
    --start 2026-10-14 --end 2026-10-28 --strike-min 25000 --strike-max 26500 \
    --out data/nifty_5m.csv --index-out data/nifty_index_5m.csv
# or let the bhavcopy choose the most liquid strikes
python -m options_mas.cli fetch upstox --underlying NIFTY --expiry 2026-10-28 \
    --start 2026-10-14 --end 2026-10-28 --bhavcopy data/bhavcopy_fo.csv --top 10
```

Repeat for each expiry you want. Output files are **merged** (deduplicated),
so the archive grows. Fetch at least 3 sessions before the period you test:
the 50 EMA needs 50 bars of warm-up and the pivots need the previous session.

## Zerodha Kite / Dhan

```bash
export KITE_API_KEY=... KITE_ACCESS_TOKEN=...
python -m options_mas.cli fetch kite --underlying BANKNIFTY --expiry 2026-11-25 --start 2026-11-01 --end 2026-11-03 \
    --strike-min 55000 --strike-max 58000

export DHAN_CLIENT_ID=... DHAN_ACCESS_TOKEN=...
python -m options_mas.cli fetch dhan --underlying NIFTY --expiry 2026-11-04 --start 2026-10-28 --end 2026-11-03 \
    --strike-min 25000 --strike-max 26000
```

Kite only serves currently listed contracts. Run it daily after the close to
build history.

## Then backtest

```bash
python -m options_mas.cli backtest --data data/nifty_5m.csv --underlying data/nifty_index_5m.csv \
    --label "UPSTOX NIFTY 5m Oct-2026" --agents A01
```

None of these sources provide **bid/ask-classified (aggressor) volume**, so
Agent 03 runs on its `ESTIMATED_BAR_CLV` delta. That delta is labelled as an
estimate everywhere, and by default it never opens trades. See
[VOLUME_DELTA_METHODOLOGY.md](VOLUME_DELTA_METHODOLOGY.md).

## Verification status

The Upstox, Kite and Dhan adapters were written against each vendor's
official Python SDK, which defines the method names, paths and parameters.
Parsing is unit tested with fixture payloads shaped like the documented
responses. They have **not** been run against the live APIs from this
repository's environment, which has no access to NSE or broker hosts. Before
relying on the data, run one small fetch and inspect it. Two items in
particular are unverified:

* the Upstox expired-candle interval string `5minute`
* Dhan's response array names
