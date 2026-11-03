# Common agent signal protocol

Every agent publishes `options_mas.models.Signal`, serialised as JSON with
`Signal.to_dict()`. The orchestrator, storage, dashboard and WebSocket all use
the same message.

| Field | Description |
|---|---|
| `signal_id` | Unique id |
| `agent_id` / `strategy_name` | `A01`, `A02`, `A03` and a human-readable name |
| `exchange`, `underlying`, `option_type`, `strike`, `expiry`, `contract_key` | The monitored contract |
| `timeframe` | e.g. `5m` |
| `signal_type` | `BUY_CE`, `BUY_PE`, `WAIT`, `NO_TRADE`, `EXIT_POSITION` |
| `status` | `INFO`, `CANDIDATE`, `ARMED`, `TRIGGERED`, `REJECTED`, `CANCELLED` |
| `entry_trigger` | Calculated entry price (e.g. breakout high + 2.00) |
| `breakout_candle_ts`, `breakout_candle_high`, `breakout_candle` | The actual breakout candle |
| `reference_candle` | Agent 01: the crossover candle. Agent 02: `{level, price, ema9, pivots}`. Agent 03: the signal candle. |
| `stop_loss`, `target` | Set only when the strategy defines them |
| `exit_condition`, `exit_reason` | Human-readable exit rule / the reason that fired |
| `signal_ts` | Time the signal was generated (candle close or tick time) |
| `trigger_price` | Price observed when the signal became `TRIGGERED` |
| `reason` | Why the signal was generated or rejected |
| `conditions_satisfied` / `conditions_not_satisfied` | Explicit checklists |
| `rejection_reason` | Why a setup was rejected or cancelled |
| `event_id` | Identity of the setup, used for duplicate protection |
| `agent_state` | State-machine state when the signal was emitted |
| `sizing_reference` | Per-unit risk reference used **only** for position sizing (never an exit) |
| `extras` | Strategy analytics (e.g. Agent 03 delta metrics, `delta_is_estimated`) |

Only `TRIGGERED` buys and `TRIGGERED` exits are actionable. Everything else is
informational and is stored for audit.

## Lifecycle

```
agent ──Signal──▶ orchestrator ──(dedupe, conflicts, ranking)──▶ risk engine ──RiskDecision──▶ execution
  ▲                                                                                          │
  └───────────── entry_filled / entry_rejected / exit_filled callbacks ◀──────────────────────┘
```

Orchestrator decisions (`SUBMITTED`, `REJECTED`, `EXIT_SUBMITTED`) are
published as separate `decision` messages. Each carries the reasons, the
consensus list (analytics only), the conflict flag and the rank score.

## Transport

* In-process: `system.EventBus` with message kinds `signal`, `decision`,
  `order`, `fill`, `trade`, `candle`, `contract`, `risk`.
* WebSocket: `/ws` pushes batched bus events.
* REST: `/api/signals`, `/api/decisions`, `/api/trades`, `/api/status`, …
* Persistence: `storage.Store` (SQLite tables `events`, `signals`, `trades`).
