# Strategy specifications (as implemented)

All three agents run **directly on option premium charts** (5-minute candles by
default). CE and PE contracts are monitored independently. Each contract has
its own OHLCV, indicators, reference candle, breakout candle, entry price,
exit conditions and position. No agent uses the underlying index for entry
confirmation, and no agent needs another agent to agree.

Common timing rules (all agents):

* Candle-close logic runs only on **completed** candles (`on_candle_close`).
* Entry triggers and price-reach exits run on **intrabar prices** (`on_price`).
* An entry trigger fires when `premium >= entry price`. If a bar opens above
  the trigger (gap), the backtester fills at that open, never at the trigger.
* Each setup has an `event_id`. It is consumed when its entry triggers, so the
  same event can never enter twice. The orchestrator checks for duplicates a
  second time.

---

## Agent 01: 50 EMA + VWAP crossover candle breakout

| Step | Rule | Code |
|---|---|---|
| Indicators | 50 EMA (seeded with an SMA, same as TradingView `ta.ema`) and session VWAP using `hlc3` (resets at every session open) | `indicators.py` |
| Crossover | An **actual** crossover is a change in the sign of `EMA − VWAP` between two completed candles. Zero values are skipped, so touching the line is not a crossover. The candle where this happens is the **crossover (reference) candle**, and its OHLC is stored. | `_update_indicators` |
| Breakout | A **later** candle whose high is above the crossover candle's high (`breakout_basis="close"` uses the close instead). The crossover candle can never be the breakout candle. | `_advance_pending` |
| Completion | The breakout candle's high is read only after that candle closes. | state `BREAKOUT_IDENTIFIED` |
| Entry | `ENTRY = breakout candle high + 2.00`. Entry triggers when the premium reaches it, on any later candle. | `_on_price` |
| Exit | A **completed** candle with `close < previous completed candle low`. A wick below the low does not count. This is also checked on the candle in which the entry filled (`evaluate_exit_on_entry_candle`). | `_check_exit` |

States: `WAITING_FOR_CROSSOVER → CROSSOVER_DETECTED → WAITING_FOR_BREAKOUT →
BREAKOUT_IDENTIFIED → ENTRY_TRIGGERED → POSITION_OPEN → EXIT_TRIGGERED →
TRADE_COMPLETED`.

Choices where the specification was silent (all configurable):

* Crossover direction: `any` by default, because the spec names no direction.
* A new crossover while a setup is still waiting replaces the old reference
  candle (`reset_on_new_crossover=true`).
* Setups that have not entered expire at session end. They have no time limit
  inside the session unless `breakout_max_wait_bars` or `entry_valid_bars` is set.
* The EMA carries across sessions (`ema_reset_daily=false`), so 50 bars of
  warm-up history from earlier sessions of the same contract are used.

## Agent 02: Fibonacci pivot points + 9 EMA breakout

Pivots come from the **contract's own previous completed session** H/L/C:

```
P = (H+L+C)/3     R1 = P+0.382(H−L)   R2 = P+0.618(H−L)   R3 = P+(H−L)
                  S1 = P−0.382(H−L)   S2 = P−0.618(H−L)   S3 = P−(H−L)
```

If there is no previous session for the contract, the agent publishes
`NO_TRADE` (state `NO_PIVOT_DATA`).

**CE (bullish breakout):**

* Relevant level `L` = the highest configured level below the close.
* The breakout candle is the completed candle on which
  `close > L AND close > EMA9` becomes true, meaning it was not true for the
  same `L` on the previous candle. **Both** conditions are required.
* `ENTRY = breakout candle HIGH + 2.00`.
* Exit on whichever comes first:
  * `NEXT_PIVOT_TARGET`: the premium reaches the next level above `L`. If that
    level is at or below the entry price, the first level above the entry price
    is used instead.
  * `EMA_EXIT`: a completed candle closes below the 9 EMA.

**PE, `pe_mode = "as_specified"` (default, exactly as written in the spec):**

* Relevant level `L` = the lowest configured level above the close.
* The breakdown candle is the completed candle on which
  `close < L AND close < EMA9` becomes true.
* `ENTRY = breakdown candle HIGH + 2.00`. The offset is measured above the
  high, not below the low.
* Exit on whichever comes first:
  * `NEXT_PIVOT_TARGET`: the premium reaches the next support below `L`.
  * `EMA_EXIT`: a completed candle closes above the 9 EMA.

> **Economic caveat (please review).** Under `as_specified`, a long PE
> position is opened only after the PE premium **falls** through support and
> then rallies back above the breakdown candle's high. Its "target" is a
> **lower** premium, which is a loss for a long option, and its EMA exit fires
> when the premium closes **above** the EMA, which is usually soon after such a
> rally. Read literally, this rule set loses money on the PE leg. It looks
> like logic written for the *underlying* chart (index breaks support, so buy
> puts) applied to the PE *premium* chart. The literal version is the default
> because the specification lists it as non-negotiable, and backtest reports
> warn about it. `pe_mode = "mirror_ce"` applies the CE logic to the PE premium
> chart instead: PE premium breaks above resistance and the 9 EMA, target is
> the next higher level, exit is a close below the 9 EMA. This mirrors how
> Agent 01 treats CE and PE.

Exit reasons are logged as `NEXT_PIVOT_TARGET` or `EMA_EXIT`. No 50 EMA or
VWAP confirmation is added.

## Agent 03: colour divergence / individual-candle volume delta

* `delta = buy volume − sell volume`
* `delta % = delta / total volume × 100`
* `relative delta = |delta| / mean(|delta|)` over the **previous**
  `avg_lookback` candles
* Imbalance ratio = buy / sell
* Dominance is BUYING, SELLING or BALANCED, using the neutral threshold
* Optional session CVD (secondary only)

The normal candle colour (close vs open) is never changed.

| Classification | Condition |
|---|---|
| BULLISH_ORDER_FLOW_DISAGREEMENT | red candle (close < open) **and** delta > 0 |
| BEARISH_ORDER_FLOW_DISAGREEMENT | green candle (close > open) **and** delta < 0 |
| NEUTRAL | doji, \|delta %\| < neutral threshold, volume < minimum, colour and delta agree, or no delta |
| Strength `WEAK` | below `min_delta_pct`, `min_relative_delta` or `min_signal_strength` |
| Strength `STRONG` | \|delta %\| ≥ `strong_delta_pct` **and** relative delta ≥ `strong_relative_delta` |
| Strength `MODERATE` | everything else |

Signal strength score (0–100) = `50·min(|Δ%|/strong_Δ%,1) + 50·min(rel/strong_rel,1)`.

Mapping (`mapping`):

* `literal` (default): a bullish disagreement on the **CE** chart is a BUY_CE
  candidate; a bearish disagreement on the **PE** chart is a BUY_PE candidate.
  Other combinations are informational. The same economic caveat as Agent 02's
  PE leg applies to the bearish → BUY_PE leg on a premium chart.
* `per_contract`: a bullish disagreement on a contract's own chart is a
  candidate to buy that contract.

**Entry, stop and exit rules are not invented.** The creator-specific rules
are unverified, so `entry_rule = "none"` by default and the agent publishes
`CANDIDATE` signals only. These are never sent for execution. Configurable
options, all off by default:

* entry: `signal_high_break` (signal candle high + offset) or `next_candle_open`
* stop: `signal_candle_low`, `fixed_points` or `pct`
* exits: `exit_on_opposite_disagreement`, `exit_on_close_below_prev_low`,
  `target_points`, `max_holding_bars`

Estimated delta never opens a position unless
`allow_estimated_delta_for_entries = true`. A colour mismatch on its own does
not prove absorption and does not guarantee a reversal.
