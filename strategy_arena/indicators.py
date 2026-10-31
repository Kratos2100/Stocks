"""Causal technical indicators. Every value at row t uses only data up to t."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - sma(s, n)) / s.rolling(n, min_periods=n).std()


def rolling_trend(log_price: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of log price on time: returns (slope per bar, R^2)."""
    y = log_price.to_numpy(dtype=float)
    slope = np.full(len(y), np.nan)
    r2 = np.full(len(y), np.nan)
    if len(y) >= n:
        w = sliding_window_view(y, n)
        x = np.arange(n) - (n - 1) / 2
        sxx = (x**2).sum()
        yc = w - w.mean(axis=1, keepdims=True)
        b = yc @ x / sxx
        sst = (yc**2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2[n - 1:] = np.where(sst > 0, b**2 * sxx / sst, 0.0)
        slope[n - 1:] = b
    return pd.Series(slope, index=log_price.index), pd.Series(r2, index=log_price.index)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / path length. 1 = pure trend, 0 = pure noise."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0, np.nan)
