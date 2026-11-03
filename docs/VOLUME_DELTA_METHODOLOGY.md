# Volume delta: sources, classification and labelling

Every candle carries a `delta_source`. Every signal, chart and API response
reports it, together with `delta_is_estimated`.

| Priority | `delta_source` | Meaning | Treated as order flow? |
|---|---|---|---|
| 1 | `BID_ASK_PROVIDER` | Your data vendor supplies buy (aggressor = buyer) and sell (aggressor = seller) volume per bar | Yes, **if** the vendor documents it as aggressor-classified |
| 1 | `BID_ASK_QUOTE_RULE` | We classify each trade against the best bid/ask at that moment (`live.BarBuilder` / `classify_trades_quote_rule`): price ≥ ask → buy; price ≤ bid → sell; inside the spread → compared with the mid; exactly at the mid → tick rule (Lee-Ready) | Yes (bid/ask-classified) |
| 2 | `ESTIMATED_TICK_RULE` | Trades without quotes: an up-tick is a buy, a down-tick is a sell, a zero-tick inherits the previous direction. If even one trade in a bar lacks a quote, the whole bar is downgraded to this source, so the two kinds are never mixed. | **No (estimate)** |
| 2 | `ESTIMATED_BAR_CLV` | OHLCV only: `buy = V·(C−L)/(H−L)`. This is a price-shape model. A red candle with a long lower wick gives a positive "delta". | **No (estimate)** |
| — | `SYNTHETIC` | Generated demo data | **No** |

## Rules enforced in code

* When a CSV contains `buy_volume` and `sell_volume` but no `delta_source`
  column, loading fails unless you pass `--delta-source` explicitly. The
  provenance of order-flow data is never assumed.
* Agent 03 never opens positions on estimated or synthetic delta by default
  (`allow_estimated_delta_for_entries = false`).
* The dashboard shows "NOT bid/ask-classified order flow" whenever the delta
  shown is estimated or synthetic.

## Documenting your vendor

Before using `BID_ASK_PROVIDER`, record the following in your data README:

1. How the vendor determines the aggressor: exchange trade flags, a quote rule
   against which quote snapshot, or a tick rule.
2. Quote / trade timestamp resolution and how they are synchronised.
3. How auction, odd-lot and block trades are handled.
4. Whether the split covers all trades or a sample.

NSE's public tick-by-tick broadcast does not carry an aggressor flag. Most
vendor "buy/sell volume" for NSE options is therefore **inferred**. Unless the
vendor confirms quote-based classification, label it `ESTIMATED_TICK_RULE`.
