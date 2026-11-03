# Multi-Agent Options Trading System: NSE NIFTY & BANKNIFTY

A modular, deterministic research and trading platform for **intraday options
buying** (BUY CE / BUY PE) on NSE index options. Three independent strategy
agents analyse **option premium charts directly**. A central orchestrator, a
deterministic risk engine and a broker-independent execution layer manage
actual trades.

| Agent | Strategy | Entry | Exit |
|---|---|---|---|
| **A01** | 50 EMA + VWAP crossover candle breakout | separate breakout candle high **+ 2.00** | completed close < previous candle low |
| **A02** | Fibonacci pivots + 9 EMA breakout | breakout candle high **+ 2.00** (CE and PE) | next pivot level **or** close across the 9 EMA |
| **A03** | Colour divergence: individual-candle volume delta | configurable, **off by default** (creator rules unverified) | configurable |

CE and PE are monitored independently and may use different strikes. A single
agent can trade on its own. There is no mandatory consensus and no
underlying-index confirmation. Live trading is **disabled by default**.

> ⚠️ This repository contains **no performance claims**. The bundled demo
> data is SYNTHETIC and labelled so everywhere. Read the economic caveat on
> Agent 02's PE leg in [docs/STRATEGIES.md](docs/STRATEGIES.md#agent-02-fibonacci-pivot-points--9-ema-breakout)
> before using it.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q                                                     # unit tests

python -m options_mas.cli backtest --synthetic --agents A01   # one agent, demo data
python -m options_mas.cli fetch groww --underlying NIFTY --expiry 2026-10-28 --start 2026-10-14 \
    --end 2026-10-28 --strike-min 25000 --strike-max 26500 --out data/nifty_5m.csv   # real data (see docs)
python -m options_mas.cli backtest --data opt5m.csv --label "VENDOR_X 2025" --chains chains.csv
python -m options_mas.cli walkforward --synthetic --agents A02 --grid agent02.ema_period=7,9,11 --train 8 --test 4
python -m options_mas.cli sensitivity --synthetic --param agent01.entry_offset --values 1,2,3
python -m options_mas.cli paper --synthetic                   # tick-driven paper rehearsal
python -m options_mas.cli dashboard --synthetic               # http://127.0.0.1:8000
```

## Documentation

* [Getting real options data](docs/DATA_SOURCES.md): NSE (free), Angel One SmartAPI, Groww, Upstox, Kite, Dhan downloaders
* [Architecture](docs/ARCHITECTURE.md): components, data flow, orchestrator modes
* [Strategies](docs/STRATEGIES.md): exact rules as implemented, configurable choices, caveats
* [Signal protocol](docs/SIGNAL_PROTOCOL.md): the common JSON message schema
* [Volume delta methodology](docs/VOLUME_DELTA_METHODOLOGY.md): actual vs estimated order flow
* [Backtesting](docs/BACKTESTING.md): data format, bias controls, metrics, walk-forward
* [Deployment](docs/DEPLOYMENT.md): modes, live-trading authorisation, broker integration, Docker
* `pine/`: TradingView validation indicators for each agent

## Development roadmap status

| Phase | Scope | Status |
|---|---|---|
| 1 | Agent 01 + validation tests | implemented, unit tested |
| 2 | Agent 02 + validation tests | implemented, unit tested |
| 3 | Agent 03 delta classification | implemented, unit tested; entry rules await verification |
| 4 | Signal protocol + orchestrator (modes A/B/C) | implemented |
| 5 | Contract selection + risk engine | implemented |
| 6 | Backtesting + paper trading | implemented (needs real historical premium data) |
| 7 | Dashboard | implemented (FastAPI + TradingView Lightweight Charts) |
| 8 | Out-of-sample validation | tooling implemented; **run on real data** |
| 9 | Strategy selection / RL | recommendation-only bandit scaffold; RL deliberately not started |
| 10 | Live brokerage | broker template + authorisation guard only; **disabled** |

## Repository layout

```
options_mas/      Python package (agents, orchestrator, risk, execution, backtest, api, ...)
config/           default.toml
docs/             architecture & methodology documents
pine/             TradingView validation indicators
tests/            pytest suite
```
