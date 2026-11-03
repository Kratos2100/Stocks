# Getting real options data

Every downloader writes the long-format CSV that `backtest --data` reads, so
the workflow is always **fetch → backtest**. Credentials are read from
environment variables only. Never put tokens in config files or commit them.

| Source | Library | Account / cost | 5-min intraday | Expired contracts | Best for |
|---|---|---|---|---|---|
| NSE F&O bhavcopy | `nselib` | free | no (end of day) | yes | pivots' previous-session H/L/C, contract universe, picking liquid strikes |
| NSE live option chain | `nselib` | free | snapshots only | no | contract selection, building a free forward archive |
| **Angel One SmartAPI** | `requests`, `websockets` | free SmartAPI app (key) + TOTP | yes | **no** (listed contracts only) | live feed + history of listed contracts; build an archive daily |
| **Groww** | `requests` | Groww Trade API subscription + API key | yes, with OI | **yes** (backtesting API) | **backtesting**, if you trade on Groww |
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

## Angel One SmartAPI

SmartAPI is free for Angel One clients and gives both 5-minute history and a
live websocket feed. Its historical API only serves contracts in the
**current** instrument master, so expired weeklies cannot be downloaded. You
can backtest on contracts that are still listed (they trade for weeks before
expiry) and grow the archive by fetching after every session. The adapter
calls the REST paths and websocket protocol of the official `smartapi-python`
SDK directly, without the SDK itself, because the SDK writes request headers
(including the API key) to a log file on errors and disables TLS checks on
its websocket.

1. At smartapi.angelone.in, create an app to get an **API key**. Enable TOTP
   for your Angel One account and keep the TOTP **secret** (the text behind
   the QR code).
2. Set the credentials as environment variables (never in files or chat):

```bash
export ANGEL_API_KEY=...  ANGEL_CLIENT_CODE=...  ANGEL_PIN=...  ANGEL_TOTP_SECRET=...
```

3. In a restricted network, allow `apiconnect.angelone.in`,
   `margincalculator.angelone.in` (instrument master) and
   `smartapisocket.angelone.in` (live feed).
4. Check the response format once, then download a listed expiry:

```bash
python -m options_mas.cli fetch angelone-probe --underlying NIFTY --day 2026-11-02
python -m options_mas.cli fetch angelone --underlying NIFTY --expiry 2026-11-10 \
    --start 2026-10-20 --end 2026-11-03 --strike-min 25000 --strike-max 26200 \
    --out data/nifty_angel_5m.csv --index-out data/nifty_index_5m.csv
```

Observed on the live API (checked 2026-11-03):

* The historical API rate-limits bursts with HTTP 403 and the plain-text body
  "Access denied because of exceeding access rate"; the client retries with
  backoff. A three-expiry, ±1500-point download (378 contracts) took ~20 minutes.
* 5-minute bars with no trades are omitted, so illiquid strikes have fewer
  than 75 bars a day. The NIFTY index series was also missing the 15:15 and
  15:20 bars on most days.
* Expiries that settle today are still listed until the master refreshes, so
  fetch each weekly on its expiry day at the latest.

### Live monitor page

```bash
python -m options_mas.cli live --underlying NIFTY --strikes-each-side 5   # open http://127.0.0.1:8000
```

On start-up it logs in, picks the nearest listed expiry (the next one after
15:30 on expiry day), finds ATM from the latest index candle, and subscribes
to ATM ± N strikes, CE and PE. Each contract is warmed up with the previous
`warmup_sessions` sessions plus today's completed bars from the historical
API, then the websocket ticks are built into 5-minute bars and fed to A01–A03.
Fills are simulated (`SimulatedBroker`, mode `paper`); no order ever reaches
Angel One. The page shows a feed pill (tick count, time of the last tick, red
when no tick has arrived for 30 s), last traded prices, agent states, signals,
paper positions and trades, and the same candle chart as the replay
dashboard. Quotes outside 09:15–15:30 are shown but never built into bars.
Bars close on the wall clock even when a contract stops trading. Starting
mid-session, the first bar is partial. Options: `--expiry`, `--agents`,
`--db` (SQLite event log, default `data/live.sqlite`), `--host`, `--port`.

For live runs, `AngelOneFeed` (in `options_mas/data/providers/angelone.py`)
implements `MarketDataFeed`: it streams ticks for the subscribed contracts
into the same 5-minute bar builder and `TradingSystem` the backtester uses.
Live ticks are snapshots without bid/ask, so Agent 03 sees the labelled
`ESTIMATED_TICK_RULE` delta.

## Groww

Groww's Trade API has dedicated backtesting endpoints: expiries, contracts
(including expired ones) and 5-minute candles with OI. The adapter calls them
with `requests`, using the paths, parameters and auth flow from Groww's
official `growwapi` SDK. The SDK itself is not needed: it pins `pandas<3`
and makes a network call when constructed.

1. On groww.in → Trade API, create an API key. Choose the **approval
   (key + secret)** type, or **TOTP**.
2. Put the credentials in environment variables on **your** machine. Never
   paste them into chat or commit them:

```bash
export GROWW_API_KEY=...       # plus ONE of:
export GROWW_API_SECRET=...    #   approval key: the token is generated automatically each run
# export GROWW_TOTP=123456     #   TOTP key: the current 6-digit code
# export GROWW_TOTP_SECRET=... #   TOTP seed (pip install pyotp) to generate codes automatically
# or simply: export GROWW_ACCESS_TOKEN=...   (a token generated on groww.in, valid for the day)
```

3. Check the response format once, then download:

```bash
# prints raw responses for one small request (expiries, a few contracts, a few candles)
python -m options_mas.cli fetch groww-probe --underlying NIFTY --expiry 2026-10-28 --day 2026-10-27

python -m options_mas.cli fetch groww --underlying NIFTY --expiry 2026-10-28 \
    --start 2026-10-14 --end 2026-10-28 --strike-min 25000 --strike-max 26500 \
    --out data/nifty_groww_5m.csv --index-out data/nifty_index_5m.csv
```

If `groww-probe` shows `parsed_contracts: 0` or `parsed_candles: 0` while
the raw sample has data, the payload shape differs from what the parser
expects. Paste the probe output (it contains no credentials) and the parser
can be adjusted.

## Upstox

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

The Angel One, Groww, Upstox, Kite and Dhan adapters were written against each vendor's
official Python SDK, which defines the method names, paths and parameters.
Parsing is unit tested with fixture payloads shaped like the documented
responses. They have **not** been run against the live APIs from this
repository's environment, which has no access to NSE or broker hosts. Before
relying on the data, run one small fetch and inspect it. Two items in
particular are unverified:

* Angel One's candle and instrument-master payloads and the websocket
  packet layout (taken from the SDK; use `fetch angelone-probe`)
* Groww's payload bodies for expiries, contracts and candles (the SDK does
  not define them; use `fetch groww-probe`). **Live check, 2026-11-03:** the
  approval-key token flow worked (`/user/detail` returned 200), but
  `/historical/expiries`, `/historical/contracts`, `/historical/candles`,
  `/historical/candle/range` and `/live-data/ltp` all returned HTTP 403
  `"Access forbidden for this request."` for that account. The requests match
  the SDK byte for byte, so this is an account entitlement (Trade API data
  subscription) issue, and the payload shapes remain unverified until an
  entitled account runs the probe.
* the Groww groww-symbol format `NSE-NIFTY-04Nov26-25000-CE` and the index
  symbols `NSE-NIFTY` / `NSE-BANKNIFTY`
* the Upstox expired-candle interval string `5minute`
* Dhan's response array names
