"""Tiny trading agents. Each one looks at prices up to the close of day t and outputs a
target position for the next day: +1 long, 0 flat, -1 short (shorts are clipped to flat
by the backtester unless shorting is allowed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .ridge import Ridge


class Agent:
    name = "agent"
    style = ""

    def positions(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self):
        return self.name


def _hold_state(enter_long, exit_long, enter_short, exit_short) -> np.ndarray:
    """Turn entry/exit conditions into a held position (+1/0/-1)."""
    n = len(enter_long)
    out = np.zeros(n)
    pos = 0.0
    for t in range(n):
        if pos == 0:
            if enter_long[t]:
                pos = 1.0
            elif enter_short[t]:
                pos = -1.0
        elif pos > 0 and (exit_long[t] or enter_short[t]):
            pos = -1.0 if enter_short[t] else 0.0
        elif pos < 0 and (exit_short[t] or enter_long[t]):
            pos = 1.0 if enter_long[t] else 0.0
        out[t] = pos
    return out


class BuyAndHold(Agent):
    name, style = "Buy & Hold", "benchmark"

    def positions(self, df):
        return pd.Series(1.0, index=df.index)


class SMACrossover(Agent):
    style = "trend"

    def __init__(self, fast=20, slow=50):
        self.fast, self.slow = fast, slow
        self.name = f"SMA Cross {fast}/{slow}"

    def positions(self, df):
        f, s = ind.sma(df.Close, self.fast), ind.sma(df.Close, self.slow)
        return pd.Series(np.where(f > s, 1.0, -1.0), index=df.index).where(s.notna(), 0.0)


class Momentum(Agent):
    style = "trend"

    def __init__(self, lookback=63):
        self.lookback = lookback
        self.name = f"Momentum {lookback}d"

    def positions(self, df):
        r = df.Close.pct_change(self.lookback)
        return np.sign(r).fillna(0.0)


class MACD(Agent):
    name, style = "MACD 12/26/9", "trend"

    def positions(self, df):
        macd = ind.ema(df.Close, 12) - ind.ema(df.Close, 26)
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        return pd.Series(np.where(macd > signal, 1.0, -1.0), index=df.index).where(signal.notna(), 0.0)


class DonchianBreakout(Agent):
    style = "breakout"

    def __init__(self, entry=20, exit=10):
        self.entry, self.exit = entry, exit
        self.name = f"Donchian {entry}/{exit}"

    def positions(self, df):
        hi_e = df.High.rolling(self.entry).max().shift(1)
        lo_e = df.Low.rolling(self.entry).min().shift(1)
        hi_x = df.High.rolling(self.exit).max().shift(1)
        lo_x = df.Low.rolling(self.exit).min().shift(1)
        c = df.Close
        held = _hold_state((c > hi_e).to_numpy(), (c < lo_x).to_numpy(),
                           (c < lo_e).to_numpy(), (c > hi_x).to_numpy())
        return pd.Series(held, index=df.index)


class RSIReversion(Agent):
    style = "mean reversion"

    def __init__(self, n=14, low=30, high=70):
        self.n, self.low, self.high = n, low, high
        self.name = f"RSI Reversion {n}"

    def positions(self, df):
        r = ind.rsi(df.Close, self.n)
        held = _hold_state((r < self.low).to_numpy(), (r > 50).to_numpy(),
                           (r > self.high).to_numpy(), (r < 50).to_numpy())
        return pd.Series(held, index=df.index)


class BollingerReversion(Agent):
    style = "mean reversion"

    def __init__(self, n=20, k=2.0):
        self.n, self.k = n, k
        self.name = f"Bollinger {n}/{k:g}"

    def positions(self, df):
        z = ind.zscore(df.Close, self.n)
        held = _hold_state((z < -self.k).to_numpy(), (z > 0).to_numpy(),
                           (z > self.k).to_numpy(), (z < 0).to_numpy())
        return pd.Series(held, index=df.index)


class LinRegTrend(Agent):
    """Fits a straight line to the last n log prices; trades in the slope's direction only
    when the line explains the prices well (R^2 above a threshold)."""
    style = "regression trend"

    def __init__(self, n=40, min_r2=0.5):
        self.n, self.min_r2 = n, min_r2
        self.name = f"LinReg Trend {n}"

    def positions(self, df):
        slope, r2 = ind.rolling_trend(np.log(df.Close), self.n)
        pos = np.where(r2 >= self.min_r2, np.sign(slope), 0.0)
        return pd.Series(pos, index=df.index).fillna(0.0)


def forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df.Close
    lr = np.log(c).diff()
    slope, r2 = ind.rolling_trend(np.log(c), 21)
    return pd.DataFrame({
        "ret_1": lr,
        "ret_5": lr.rolling(5).sum(),
        "ret_21": lr.rolling(21).sum(),
        "vol_21": lr.rolling(21).std(),
        "dist_sma20": c / ind.sma(c, 20) - 1,
        "dist_sma50": c / ind.sma(c, 50) - 1,
        "rsi_14": ind.rsi(c, 14) / 100 - 0.5,
        "trend_21": slope * r2,
    }, index=df.index)


class RidgeForecaster(Agent):
    """Walk-forward ridge regression of the next day's return on technical features.

    Retrains every `refit` days on the last `train_window` days and only ever uses
    targets that were already known on the training date.
    """
    name, style = "Ridge Forecaster", "regression ML"

    def __init__(self, train_window=500, refit=21, min_train=250, alpha=50.0, threshold=0.0):
        self.train_window, self.refit, self.min_train = train_window, refit, min_train
        self.alpha, self.threshold = alpha, threshold

    def positions(self, df):
        X = forecast_features(df)
        y = np.log(df.Close).diff().shift(-1)  # next-day return, known one day later
        Xv, yv = X.to_numpy(), y.to_numpy()
        valid = ~np.isnan(Xv).any(axis=1)
        pos = np.zeros(len(df))
        model = None
        for t in range(len(df)):
            if t % self.refit == 0:
                lo = max(0, t - self.train_window)
                rows = np.arange(lo, t)  # row i's target is return i+1 <= t: known at t
                rows = rows[valid[rows] & ~np.isnan(yv[rows])]
                if len(rows) >= self.min_train:
                    model = Ridge(self.alpha).fit(Xv[rows], yv[rows])
            if model is not None and valid[t]:
                pred = model.predict(Xv[t:t + 1])[0]
                pos[t] = 1.0 if pred > self.threshold else (-1.0 if pred < -self.threshold else 0.0)
        return pd.Series(pos, index=df.index)


def default_agents() -> list[Agent]:
    return [
        BuyAndHold(),
        SMACrossover(20, 50),
        Momentum(63),
        MACD(),
        DonchianBreakout(20, 10),
        RSIReversion(14),
        BollingerReversion(20, 2.0),
        LinRegTrend(40),
        RidgeForecaster(),
    ]
