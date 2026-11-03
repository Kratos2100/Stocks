"""Shared helpers: every provider returns the same long-format frame that
:func:`options_mas.data.dataset.load_dataset` reads, so downloaded data can be
backtested directly::

    underlying, expiry, strike, option_type, timestamp, open, high, low, close, volume, oi

``timestamp`` is the bar START in IST (naive). Providers never invent
buy/sell volume: broker candle APIs do not supply it, so Agent 03 falls back
to its clearly labelled ESTIMATED_BAR_CLV delta for such data.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable, Iterator, Optional

import pandas as pd

OPTION_COLUMNS = ["underlying", "expiry", "strike", "option_type", "timestamp",
                  "open", "high", "low", "close", "volume", "oi"]
CHAIN_COLUMNS = ["timestamp", "underlying", "spot", "expiry", "strike", "option_type",
                 "ltp", "bid", "ask", "volume", "oi", "iv"]
UNDERLYING_COLUMNS = ["underlying", "timestamp", "open", "high", "low", "close", "volume"]
IST = "Asia/Kolkata"


def to_ist_naive(ts) -> pd.Timestamp:
    """Accept epoch seconds, ISO strings or datetimes; return naive IST."""
    if isinstance(ts, (int, float)):
        t = pd.Timestamp(ts, unit="s", tz="UTC")
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(IST).tz_localize(None)
    return t


def empty_options_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OPTION_COLUMNS)


def make_option_rows(underlying: str, expiry: date, strike: float, option_type: str,
                     bars: Iterable[dict]) -> pd.DataFrame:
    """``bars`` = dicts with timestamp/open/high/low/close/volume[/oi]."""
    rows = [{
        "underlying": underlying, "expiry": pd.Timestamp(expiry).date(), "strike": float(strike),
        "option_type": option_type.upper(), "timestamp": to_ist_naive(b["timestamp"]),
        "open": float(b["open"]), "high": float(b["high"]), "low": float(b["low"]), "close": float(b["close"]),
        "volume": float(b.get("volume") or 0.0), "oi": None if b.get("oi") is None else float(b["oi"]),
    } for b in bars]
    if not rows:
        return empty_options_frame()
    return pd.DataFrame(rows, columns=OPTION_COLUMNS).sort_values("timestamp").reset_index(drop=True)


def date_chunks(start: date, end: date, max_days: int) -> Iterator[tuple[date, date]]:
    """Split [start, end] into inclusive windows of at most ``max_days``."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=max_days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def in_session(df: pd.DataFrame, open_t=(9, 15), close_t=(15, 30)) -> pd.DataFrame:
    """Drop bars outside the regular NSE session."""
    if df.empty:
        return df
    t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df[(t >= open_t[0] * 60 + open_t[1]) & (t < close_t[0] * 60 + close_t[1])]


def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return empty_options_frame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["underlying", "expiry", "strike", "option_type", "timestamp"]).sort_values(
        ["underlying", "expiry", "strike", "option_type", "timestamp"]).reset_index(drop=True)


def strikes_in_range(strikes: Iterable[float], lo: Optional[float], hi: Optional[float]) -> list[float]:
    return sorted(s for s in set(float(x) for x in strikes)
                  if (lo is None or s >= lo) and (hi is None or s <= hi))
