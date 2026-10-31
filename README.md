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
```

### 1. The agents

Each agent sees prices up to the close of day *t* and decides its position for day *t+1*
(long, flat, or short; shorts become flat unless you pass `--allow-short`).

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
| Ridge Forecaster | regression ML | walk-forward ridge regression of tomorrow's return on 8 technical features |

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
past decision), the backtest's timing and costs, the ridge solver, and the full pipeline.
