# Deployment

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m options_mas.cli dashboard --synthetic          # http://127.0.0.1:8000
```

## Docker

```bash
docker build -t options-mas .
docker run --rm -p 8000:8000 options-mas                   # synthetic replay dashboard
docker run --rm -p 8000:8000 -v $PWD/data:/data options-mas \
    python -m options_mas.cli dashboard --host 0.0.0.0 --data /data/opt5m.csv --label "VENDOR_X"
```

## Execution modes

| Mode | Broker | Orders | Enable with |
|---|---|---|---|
| `backtest` | `SimulatedBroker` | simulated on historical bars | `cli backtest` |
| `paper` | `SimulatedBroker` | simulated on live/replayed ticks | `execution.mode = "paper"` |
| `manual` | any | every entry waits for operator approval (`/api/approvals/{id}/approve`) and expires after `approval_timeout_minutes`; exits are never gated | `execution.mode = "manual"` |
| `live` | real `BrokerAdapter` | real orders | see below |

### Live trading is disabled by default

`assert_live_authorized` (in `execution/broker.py`) refuses live execution
unless **all** of these hold:

1. `execution.live_enabled = true` in the config file
2. `execution.mode = "live"`
3. environment variable `OPTIONS_MAS_LIVE_AUTHORIZATION=I_AUTHORIZE_LIVE_TRADING`
4. a per-run confirmation flag passed in code (`live_confirmed=True` / a
   `--confirm-live` CLI flag that you add with your broker runner)

Do not enable live trading before Phase 10: successful out-of-sample and paper
results, and your explicit authorisation.

## Integrating an Indian broker

Implement `IndianBrokerAdapterTemplate` (Zerodha Kite Connect, Upstox, Angel
One SmartAPI, Dhan, Fyers, …):

* map `OptionContract` to the broker symbol / instrument token (segment `NFO`,
  product `MIS`)
* `place_order` → broker order API; translate order-update websocket events
  into `BrokerEvent("FILL" | "REJECT" | "CANCEL")`
* `positions()` / `available_funds()` → used by `ExecutionEngine.reconcile()`
  and the risk engine
* implement `MarketDataFeed.stream()` for the broker's tick websocket, with
  best bid/ask if available. `BarBuilder` then produces `BID_ASK_QUOTE_RULE`
  delta.
* run `reconcile()` periodically and alert on mismatches. Use the dashboard's
  emergency stop or `TradingSystem.emergency_stop()` to flatten.

Respect SEBI's retail algo-trading framework: broker-registered algos, static
IP, order-rate limits and API order tagging, as applicable to your setup.

## Database

SQLite (`db_path`) for development. For production, point the same schema
(plain SQL in `storage.py`) at PostgreSQL: swap `sqlite3` for `psycopg` and
change `?` placeholders to `%s`. The tables are `events`, `signals` and
`trades`.

## Operational checklist

* Verify lot sizes (`[instruments.*]`) and charges (`[charges]`) against
  current NSE, SEBI and broker documents.
* Check that the server clock is synchronised (NTP) to IST, and check the
  session times.
* Pine scripts in `pine/` validate signals on TradingView. Compile them in the
  Pine editor first, because they were written to mirror the Python logic but
  have not been compiled in this repository's CI.
