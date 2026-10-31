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
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
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
    return _normalize(pd.read_csv(path))


def from_yfinance(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(start=start, end=end, period=None if start else "max", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    return _normalize(hist)


def from_stooq(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
    df = _normalize(pd.read_csv(f"https://stooq.com/q/d/l/?s={symbol}&i=d"))
    return df.loc[start:end]


def load_prices(ticker: str | None = None, start: str | None = None, end: str | None = None,
                csv: str | None = None) -> pd.DataFrame:
    """Load daily OHLCV. Tries a CSV first if given, then yfinance, then Stooq."""
    if csv:
        return from_csv(csv).loc[start:end]
    if not ticker:
        raise ValueError("give a ticker or a CSV path")
    errors = []
    for loader in (from_yfinance, from_stooq):
        try:
            df = loader(ticker, start, end)
            if len(df):
                return df
        except Exception as exc:  # network down, package missing, unknown symbol
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(f"could not load {ticker}:\n  " + "\n  ".join(errors))


def synthetic_prices(n: int = 2500, seed: int = 7, start: str = "2015-01-02") -> pd.DataFrame:
    """A market that switches between bull trends, bear trends and mean-reverting chop.

    Different agents win in different regimes, which is what the regression selector has
    to learn. Useful offline and in tests.
    """
    rng = np.random.default_rng(seed)
    regimes = {0: (0.0012, 0.012), 1: (-0.0012, 0.02), 2: (0.0, 0.015)}  # bull, bear, chop
    state, level = 0, np.log(100.0)
    anchor = level
    logp = np.empty(n)
    for t in range(n):
        if rng.random() < 1 / 90:  # average regime length ~90 bars
            state = int(rng.choice([s for s in regimes if s != state]))
            anchor = level
        mu, sigma = regimes[state]
        pull = -0.08 * (level - anchor) if state == 2 else 0.0
        level += mu + pull + sigma * rng.standard_normal()
        logp[t] = level
    close = np.exp(logp)
    open_ = close * np.exp(0.003 * rng.standard_normal(n))
    spread = np.abs(0.008 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range(start, periods=n, name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
