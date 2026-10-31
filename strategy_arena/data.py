"""Price loading: yfinance, Stooq, a local CSV, or a synthetic regime-switching market."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Date" in df.columns:
        df = df.set_index("Date")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:  # keep the exchange's wall-clock time so sessions split on local dates
        idx = idx.tz_localize(None)
    df.index = idx
    df.index.name = "Date"
    missing = [c for c in ("Close",) if c not in df.columns]
    if missing:
        raise ValueError(f"price data has no {missing} column")
    for c in ("Open", "High", "Low"):
        if c not in df.columns:
            df[c] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def from_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    date_col = next((c for c in raw.columns if str(c).strip().lower() in ("date", "datetime", "time", "timestamp")), None)
    if date_col is not None:
        raw = raw.rename(columns={date_col: "Date"})
        try:
            raw["Date"] = pd.to_datetime(raw["Date"])
        except (ValueError, TypeError):  # mixed UTC offsets (e.g. across a DST change)
            raw["Date"] = pd.to_datetime(raw["Date"], utc=True)
    return _normalize(raw)


# yfinance only serves intraday bars for a limited look-back window
INTRADAY_PERIOD = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "90m": "60d",
                   "60m": "730d", "1h": "730d"}


def from_yfinance(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    if interval in INTRADAY_PERIOD:
        hist = yf.Ticker(ticker).history(period=INTRADAY_PERIOD[interval], interval=interval, auto_adjust=True)
    else:
        hist = yf.Ticker(ticker).history(start=start, end=end, period=None if start else "max",
                                         interval=interval, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    return _normalize(hist)


def from_stooq(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    if interval != "1d":
        raise ValueError("Stooq only serves daily bars")
    symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
    df = _normalize(pd.read_csv(f"https://stooq.com/q/d/l/?s={symbol}&i=d"))
    return df.loc[start:end]


def load_prices(ticker: str | None = None, start: str | None = None, end: str | None = None,
                csv: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """Load OHLCV bars. Tries a CSV first if given, then yfinance, then Stooq (daily only).

    `interval` is a yfinance interval: 1d (default), 1h, 30m, 15m, 5m, 1m...
    """
    if csv:
        return from_csv(csv).loc[start:end]
    if not ticker:
        raise ValueError("give a ticker or a CSV path")
    errors = []
    for loader in (from_yfinance, from_stooq):
        try:
            df = loader(ticker, start, end, interval)
            if len(df):
                return df
        except Exception as exc:  # network down, package missing, unknown symbol
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(f"could not load {ticker}:\n  " + "\n  ".join(errors))


def synthetic_prices(n: int = 2500, seed: int = 7, start: str = "2015-01-02", bars_per_day: int = 1,
                     bar_minutes: int = 5) -> pd.DataFrame:
    """A market that switches between bull trends, bear trends and mean-reverting chop.

    Different agents win in different regimes, which is what the regression selector has
    to learn. Useful offline and in tests. With `bars_per_day > 1` it makes intraday bars
    (sessions open 09:30, one bar every `bar_minutes`), scaled so a day moves like a daily bar.
    """
    rng = np.random.default_rng(seed)
    k = bars_per_day
    regimes = {0: (0.0012, 0.012), 1: (-0.0012, 0.02), 2: (0.0, 0.015)}  # bull, bear, chop (per day)
    state, level = 0, np.log(100.0)
    anchor = level
    logp = np.empty(n)
    for t in range(n):
        if rng.random() < 1 / (90 * k):  # average regime length ~90 days
            state = int(rng.choice([s for s in regimes if s != state]))
            anchor = level
        mu, sigma = regimes[state]
        pull = -0.08 / k * (level - anchor) if state == 2 else 0.0
        level += mu / k + pull + sigma / np.sqrt(k) * rng.standard_normal()
        logp[t] = level
    close = np.exp(logp)
    bar_noise = 0.003 / np.sqrt(k)
    open_ = close * np.exp(bar_noise * rng.standard_normal(n))
    spread = np.abs(0.008 / np.sqrt(k) * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float) / k
    if k == 1:
        idx = pd.bdate_range(start, periods=n, name="Date")
    else:
        days = pd.bdate_range(start, periods=-(-n // k))
        offsets = pd.to_timedelta(9.5 * 60 + bar_minutes * np.arange(k), unit="min")
        idx = pd.DatetimeIndex((days.values[:, None] + offsets.values[None, :]).ravel()[:n], name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
