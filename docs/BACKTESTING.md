# Backtesting and validation

## Data you need

| File | Required | Columns |
|---|---|---|
| Option candles (5-min) | yes | `underlying, expiry, strike, option_type, timestamp, open, high, low, close, volume` (+ `buy_volume, sell_volume, delta_source`, `oi`) |
| Option candles (1-min) | optional | same; used for intrabar ordering |
| Chain snapshots | optional | `timestamp, underlying, spot, expiry, strike, option_type, ltp, bid, ask, volume, oi` (+ `iv, delta, theta`) |
| Underlying candles | optional | `underlying, timestamp, open, high, low, close, volume` (regime analytics only) |

`timestamp` is the bar **start** time in IST. Include **expired** contracts
and several prior sessions of each contract: EMA warm-up needs 50 bars and the
pivots need the previous session.

## Running

```bash
python -m options_mas.cli backtest --data opt5m.csv --subbars opt1m.csv --chains chains.csv \
    --underlying index.csv --label "VENDOR_X NIFTY 2025-H1" --delta-source BID_ASK_PROVIDER \
    --agents A01 --out reports/a01.json
```

Test each agent on its own first (`--agents A01`, then `A02`, then `A03`),
then run them together.

Without `--chains`, every contract that has data on a session is monitored.
Use `--chains` so the contract selection agent picks one CE and one PE per
underlying per session, using the snapshot at or before 09:20.

## Bias controls

| Risk | Control |
|---|---|
| Look-ahead / repainting | Streaming indicators. Agents only see completed candles and intrabar prices. Warm-up uses only earlier sessions. A test perturbs future data and checks that earlier signals do not change. |
| Data leakage | Walk-forward folds are chronological. Parameters are chosen in-sample only. The Mode C regime label is the *previous* session's. Relative delta uses previous candles only. |
| Unrealistic fills | Fills at the trigger level, or at the open on a gap. Close-based signals fill at the **next bar open**. Slippage (points and/or %), tick rounding, participation cap against the **previous** bar's volume (entries are IOC, exits keep working) and full charges are applied. |
| Intrabar ordering | O→L→H→C for bullish bars and O→H→L→C for bearish bars, or the 1-minute path when supplied. Price exits are not assumed in the bar where the entry filled (`intrabar_exit_on_entry_bar=false`). |
| Survivorship | Contracts come from historical data per session and include expired ones. Selection uses snapshots available at that moment only. |

## Metrics

`backtest/metrics.py`: total, winning and losing trades, win rate, net and
gross P&L, charges, profit factor (`null` when there are no losing trades),
expectancy, max drawdown on closed trades plus a mark-to-market equity
drawdown, average holding time, average risk/reward (avg win / avg loss),
average R-multiple, max consecutive losses and wins, and breakdowns by agent,
option type, agent × option type, underlying, regime and exit reason.

## Validation

```bash
# chronological in-sample / out-of-sample and walk-forward
python -m options_mas.cli walkforward --data ... --agents A02 \
    --grid agent02.ema_period=7,9,11 agent02.target_reference=broken_level,entry_price \
    --train 40 --test 10 --out reports/wf_a02.json --ranking-out reports/ranking.json

# parameter sensitivity
python -m options_mas.cli sensitivity --data ... --agents A01 --param agent01.entry_offset --values 1,1.5,2,2.5,3
```

Strategy rule parameters (offsets, EMA lengths) are fixed by the
specification. Use sensitivity runs to check robustness, not to replace the
rules. Feed `reports/ranking.json` to Mode C with
`[orchestrator] ranking_file = "reports/ranking.json"`.

## Honesty rules

* Every report carries `data_label` and a disclaimer. Synthetic data is
  labelled `SYNTHETIC … NOT REAL MARKET DATA` everywhere.
* Results are only ever computed from simulated trades on the supplied data.
  Nothing is extrapolated, and with no trades every metric is zero or `null`.
* No performance figures are published in this repository.
