# Architecture

```
                         ┌───────────────────────── market data ─────────────────────────┐
                         │ historical CSV/Parquet │ CandleReplayFeed │ broker websocket (TBD) │
                         └───────────────┬───────────────────────────────────────────────┘
                                         │ ticks / completed 5-min candles (per CONTRACT)
                     ┌───────────────────▼────────────────────┐
                     │  Contract Selection Agent              │  CE and PE chosen independently
                     │  (expiry, strike, liquidity, spread,   │  (no directional view)
                     │   IV, delta, theta)                    │
                     └───────────────────┬────────────────────┘
                                         │ monitored contracts
     ┌───────────────────────────────────▼──────────────────────────────────┐
     │ TradingSystem (system.py) – identical in backtest / paper / live     │
     │                                                                      │
     │  ┌────────────┐  ┌────────────┐  ┌────────────┐                      │
     │  │ Agent 01   │  │ Agent 02   │  │ Agent 03   │  one ContractContext │
     │  │ EMA50+VWAP │  │ Fib + EMA9 │  │ Vol delta  │  per contract        │
     │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                      │
     │        └──── Signal (common schema) ───┘                             │
     │                        ▼                                             │
     │              Central Orchestrator (A / B / C)                        │
     │   dedupe · conflicts · consensus analytics · OOS ranking             │
     │                        ▼                                             │
     │              Risk Engine (deterministic)                             │
     │   sizing · daily loss · trades/day · concurrency · exposure ·        │
     │   allocation · profit protection · emergency stop · EOD square-off   │
     │                        ▼                                             │
     │              Execution Agent ── BrokerAdapter                        │
     │   orders · partial fills · rejects · approvals · reconciliation     │
     │                        │        ├─ SimulatedBroker (backtest/paper)  │
     │                        │        └─ IndianBrokerAdapterTemplate (live)│
     └────────────────────────┼─────────────────────────────────────────────┘
                              ▼
          EventBus ─▶ SQLite Store · FastAPI REST/WebSocket · Dashboard
```

## Module map

| Module | Responsibility |
|---|---|
| `models.py` | Contracts, candles, Signal schema, orders, positions, trades, enums |
| `config.py`, `config/default.toml` | Typed configuration; every open strategy choice is a parameter |
| `indicators.py` | Streaming EMA, session VWAP, Fibonacci pivots (+ vectorised reference versions) |
| `data/volume_delta.py` | Bid/ask, tick-rule and CLV classification; per-candle delta analytics |
| `data/dataset.py` | Dataset loading, warm-up slicing, regime labelling |
| `data/synthetic.py` | Labelled SYNTHETIC demo data |
| `agents/` | The three deterministic strategy agents + shared base class |
| `orchestrator.py` | Central orchestrator, `StrategyRanker` (Mode C) |
| `contract_selection.py` | Contract selection agent, Black–Scholes helpers |
| `risk.py` | Risk engine |
| `execution/` | Charges, broker abstraction, execution engine |
| `system.py` | Wiring + EventBus |
| `backtest/` | Event-driven backtester, metrics, walk-forward / sensitivity |
| `live.py` | Tick feed abstraction, bar builder, streaming runner (paper/manual/live) |
| `storage.py` | SQLite persistence |
| `api/` | FastAPI server + dashboard |
| `learning/bandit.py` | Phase 9 research: recommendation-only Thompson-sampling bandit |
| `pine/` | TradingView indicators to validate each agent's signals |

## Design principles

* **No LLM per agent.** Agents are plain Python state machines. They are
  deterministic and unit tested.
* **One code path.** The backtester, paper runner and live runner all drive
  the same `TradingSystem`, so a strategy cannot behave differently in a backtest.
* **Separation of powers.** Agents decide setups. The orchestrator coordinates
  them. The risk engine approves, sizes and can add hard exits. Execution
  trades. None of them can change another's rules. Risk never suppresses a
  strategy exit, and exits are never gated by approval.
* **Contract isolation.** Each (agent, contract) pair has its own context.
  CE and PE can use different strikes.
* **Duplicate protection twice.** Agents consume `event_id`s. The orchestrator
  also tracks `(agent, event_id)` and blocks a second position in the same
  contract (unless configured otherwise).

## Orchestrator modes

* **A – Independent:** every triggered signal goes to the risk engine on its own.
* **B – Consensus analytics:** the same as A. Each decision also lists other
  agents with a same-direction idea on the same underlying within
  `consensus_window_minutes`. Consensus is **never** required.
* **C – Intelligent selection:** simultaneous signals are processed in order of
  out-of-sample score. The score is shrunk net P&L per trade, by regime when
  enough samples exist, and comes from `--ranking-out` of a walk-forward run or
  from online results. Optionally (`block_negative_expectancy`) agents with
  negative OOS expectancy are skipped. The regime used is the previous
  session's, which avoids look-ahead.

Conflicts (BUY_CE and BUY_PE on the same underlying in one batch, or against an
open opposite position) are always logged. The policy is `allow_and_flag`
(default), `first_come` or `block_both`.
