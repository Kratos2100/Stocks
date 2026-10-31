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


def is_intraday(index: pd.DatetimeIndex) -> bool:
    """True when bars are less than a day apart."""
    return len(index) > 1 and pd.Series(index).diff().median() < pd.Timedelta(hours=20)


def session_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """Number the trading sessions. A session starts after a break longer than the usual bar
    spacing plus 30 minutes: the overnight close for NSE or US stocks, and the 17:00-18:00
    daily halt for CME futures (whose sessions cross midnight). Daily bars: one per bar."""
    if not is_intraday(index):
        return np.arange(len(index))
    gaps = pd.Series(index).diff()
    return (gaps > gaps.median() + pd.Timedelta(minutes=30)).cumsum().to_numpy()


def session_ends(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the last bar of every session (and on the final bar of the data)."""
    ids = session_ids(index)
    return np.append(ids[1:] != ids[:-1], True) if len(ids) else np.zeros(0, dtype=bool)


def bars_per_session(index: pd.DatetimeIndex) -> float:
    """Median number of bars per session: 1 for daily data, ~75 for 5-minute NSE bars."""
    if not is_intraday(index):
        return 1.0
    return float(pd.Series(session_ids(index)).value_counts().median())


def has_volume(df: pd.DataFrame) -> bool:
    """Indices (NIFTY, ^GSPC on intraday) often come with zero volume."""
    return bool((df.Volume.fillna(0) > 0).mean() >= 0.5)


def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-weighted average of the typical price (H+L+C)/3.

    Intraday bars: the classic session VWAP, reset at the first bar of every session.
    Daily bars: a rolling VWAP over the last `window` bars (a daily session VWAP would be
    the bar itself). Without volume data every bar gets the same weight.
    """
    tp = (df.High + df.Low + df.Close) / 3
    vol = df.Volume.fillna(0).clip(lower=0) if has_volume(df) else pd.Series(1.0, index=df.index)
    pv = tp * vol
    if is_intraday(df.index):
        session = session_ids(df.index)
        num, den = pv.groupby(session).cumsum(), vol.groupby(session).cumsum()
    else:
        num, den = pv.rolling(window, min_periods=window).sum(), vol.rolling(window, min_periods=window).sum()
    return (num / den.replace(0, np.nan)).where(den > 0)


FIB_LEVELS = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]


def fib_pivots(df: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
    """Fibonacci pivot levels from the previous period's high, low and close.

    P = (H + L + C) / 3, R/S n = P +/- {0.382, 0.618, 1.0} x (H - L).
    `period` is "D" (previous session; the default for intraday bars), "W" (previous week;
    the default for daily bars) or "M" (previous month).
    """
    if period is None:
        period = "D" if is_intraday(df.index) else "W"
    key = session_ids(df.index) if period == "D" else df.index.to_period(period)
    g = df.groupby(key)
    prev = pd.DataFrame({"H": g.High.max(), "L": g.Low.min(), "C": g.Close.last()}).shift(1)
    p = (prev.H + prev.L + prev.C) / 3
    rng = prev.H - prev.L
    levels = pd.DataFrame({
        "S3": p - rng, "S2": p - 0.618 * rng, "S1": p - 0.382 * rng, "P": p,
        "R1": p + 0.382 * rng, "R2": p + 0.618 * rng, "R3": p + rng,
    })
    out = levels.reindex(key)
    out.index = df.index
    return out[FIB_LEVELS]
