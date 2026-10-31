# Stocks

- `Final Assignment(1).ipynb` — extracting and plotting TSLA / GME prices and revenue.
- `strategy_arena/` — tiny trading agents compete on a stock, and regression learning
  picks the strategy that suits that stock's current market regime.

## Strategy Arena

```bash
pip install -r requirements.txt
python -m strategy_arena TSLA                 # download prices with yfinance (Stooq as fallback)
python -m strategy_arena TSLA GME --start 2015-01-01 --json report
python -m strategy_arena AAPL --csv aapl.csv  # your own OHLCV file (needs Date and Close columns)
python -m strategy_arena --synthetic          # offline demo on a regime-switching market

# intraday bars, long and short (futures, or calls/puts on the signal)
python -m strategy_arena ES=F NQ=F --interval 5m --allow-short
python -m strategy_arena ^NSEBANK --interval 15m --allow-short --cost-bps 2
```

### 1. The agents

Each agent sees prices up to the close of bar *t* and decides its position for bar *t+1*
(long, flat, or short; shorts become flat unless you pass `--allow-short`). A take-profit
inside a bar is filled at the target price.

| Agent | Style | Rule |
|---|---|---|
| Buy & Hold | benchmark | always long |
| SMA Cross 20/50 | trend | long when the 20-day average is above the 50-day |
| Momentum 63d | trend | long when the last 3 months' return is positive |
| MACD 12/26/9 | trend | long when MACD is above its signal line |
| Donchian 20/10 | breakout | buy a 20-day high, exit at a 10-day low |
| RSI Reversion 14 | mean reversion | buy RSI < 30, exit at RSI > 50 |
| Bollinger 20/2 | mean reversion | buy 2 std devs under the 20-day average, exit at the average |
| LinReg Trend 40 | regression trend | fit a line to 40 days of log prices; trade its slope when R² ≥ 0.5 |
| Ridge Forecaster | regression ML | walk-forward ridge regression of the next bar's return on 8 technical features |
| EMA50 + VWAP Breakout | EMA + VWAP breakout | enter when a candle closes above **both** the 50 EMA and VWAP (it was not above both on the candle before); exit when a later candle closes below the previous candle's low. Shorts mirror it |
| Fib Pivot + EMA9 | pivot breakout | enter when a candle closes through a Fibonacci pivot level (S3…P…R3) and is on the same side of the 9 EMA; exit at the next pivot level (filled at that price) or when a candle closes back through the 9 EMA. Shorts mirror it |

Details of the two breakout agents:

- **VWAP** is the session VWAP on intraday bars (reset every day). Daily bars have no
  intraday session, so they use a rolling 20-bar VWAP. Data without volume (indices such
  as NIFTY or BANKNIFTY) gets an equal-weighted average price, and the report says so.
- **"Closes below the previous candle"** means below its *low* by default.
  `EmaVwapBreakout(exit_rule="close")` exits on a close below the previous *close* instead.
- **Fibonacci pivots** come from the previous session's high, low and close on intraday
  bars, and from the previous week on daily bars (`FibPivotEma(period="D" | "W" | "M")`):
  P = (H + L + C) / 3, R/S 1-3 = P ± 0.382 / 0.618 / 1.0 × (H − L). Once a target fills,
  the agent waits for the next candle before trading again. A close back through the EMA
  that also breaks a pivot the other way reverses the position.

### Options and futures

- Run with `--allow-short`. Otherwise short signals become "flat", which is right for a
  cash stock account but not for futures (sell the contract) or options (buy puts).
- Use `--interval 5m` / `15m` / `1h` for intraday bars from yfinance. Yahoo keeps 60 days of
  5m/15m bars and 730 days of 1h bars; for longer history export bars from your broker
  and use `--csv` (timestamps with a UTC offset are kept in exchange time).
- Set `--cost-bps` to your real costs. Futures round trips are often 1-2 bps; option
  spreads can be far wider.
- The backtest trades the **underlying price**. Futures P&L follows it closely (use the
  continuous contract, e.g. `ES=F`). Option P&L does not: delta, time decay and implied
  volatility change it. Treat the result as the quality of the *direction signal*, not as
  options returns.
- On intraday data the windows are counted in bars: rounds are 5 sessions, the
  regression predicts one session ahead and decides every sixth of a session (override
  with `--round-len`, `--horizon`, `--step`).

### 2. The competition

All agents trade the same stock over the same days, paying `--cost-bps` (default 5) per unit
traded. The evaluation is split into rounds (`--round-len`, default 63 days ≈ a quarter);
each round is ranked, and the leaderboard shows CAGR, Sharpe, max drawdown, time in the
market, rounds won and points (n points for 1st place, 1 for last).

### 3. Regression learning: which agent for which regime?

No strategy wins everywhere: trend followers win in strong trends, mean-reversion agents in
choppy markets. For **every agent** the selector fits a ridge regression

```
agent's return over the next 21 days  ~  market regime today + agent's own last-63-day form
```

where the regime is 12 features: 21/63/126-day returns, volatility and its ratio, the slope
and R² of a 63-day trend line, the efficiency ratio, return autocorrelation, distance from
the 200-day average, RSI and the drawdown from the 52-week high. The ridge penalty is
picked on the most recent quarter of training data, with a gap so overlapping targets
cannot leak.

The selector is tested **walk-forward**: every 5 days it trains only on samples whose
21-day outcome was already known, predicts each agent's return, and hands the money to
the agent with the highest prediction (or cash if all are negative). The report shows how
that would have done against every agent, its rank IC (correlation between predicted and
actual ranking), how often it picked an above-median or top-3 agent, and its regret.

Finally it refits on all history and forecasts each agent for today's regime, with the
features that drive the pick.

### 4. The suggestion

- If the selector earned trust walk-forward (positive rank IC and at least Buy & Hold's
  Sharpe), the suggested strategy is its pick for the current regime.
- Otherwise the suggestion falls back to the tournament champion by Sharpe, and the
  report says why.

This is a research tool, not financial advice: backtests ignore slippage beyond the cost
setting, taxes and the fact that the past regimes may not repeat.

### Tests

```bash
python -m pytest -q
```

Tests check that no agent or feature looks ahead (changing future prices never changes a
past decision), the backtest's timing, costs and target fills, candle-by-candle entries and
exits of the EMA + VWAP and pivot agents, session VWAP and pivot levels, the ridge solver,
and the full pipeline on daily and intraday bars.
