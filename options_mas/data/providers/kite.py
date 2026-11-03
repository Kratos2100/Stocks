"""Zerodha Kite Connect – 5-minute option candles via ``kiteconnect``.

``pip install kiteconnect``. Requires a Kite Connect app with the historical
data add-on and a daily access token (``KITE_API_KEY`` / ``KITE_ACCESS_TOKEN``).
Kite serves history only for contracts that are **currently listed**; expired
option contracts are not available, so build a backtest archive by running
this daily (or use Upstox / Dhan for expired contracts).
"""
from __future__ import annotations

import os
import time as _time
from datetime import date, datetime, time
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .common import concat, date_chunks, empty_options_frame, in_session, make_option_rows, strikes_in_range


def make_kite(api_key: Optional[str] = None, access_token: Optional[str] = None):
    from kiteconnect import KiteConnect  # optional dependency
    api_key = api_key or os.environ.get("KITE_API_KEY")
    access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise RuntimeError("set KITE_API_KEY and KITE_ACCESS_TOKEN")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


class KiteSource:
    MAX_DAYS_5MIN = 100  # Kite limit per request for minute intervals

    def __init__(self, kite: Any, pause: float = 0.35):
        self.kite = kite
        self.pause = pause  # Kite allows ~3 historical requests / second
        self._instruments: Optional[pd.DataFrame] = None

    def instruments(self) -> pd.DataFrame:
        if self._instruments is None:
            df = pd.DataFrame(self.kite.instruments("NFO"))
            df = df[df["instrument_type"].isin(["CE", "PE"])].copy()
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            self._instruments = df
        return self._instruments

    def option_contracts(self, underlying: str, expiry: date) -> pd.DataFrame:
        df = self.instruments()
        return df[(df["name"] == underlying.upper()) & (df["expiry"] == expiry)]

    def fetch_options(self, underlying: str, expiry: date, start: date, end: date,
                      strike_min: Optional[float] = None, strike_max: Optional[float] = None,
                      strikes: Optional[Iterable[float]] = None, interval: str = "5minute",
                      log: Callable[[str], None] = print) -> pd.DataFrame:
        con = self.option_contracts(underlying, expiry)
        if con.empty:
            log(f"no listed {underlying} contracts for {expiry} (Kite has no expired-option history)")
            return empty_options_frame()
        wanted = set(float(s) for s in strikes) if strikes else set(
            strikes_in_range(con["strike"], strike_min, strike_max))
        frames = []
        for r in con[con["strike"].astype(float).isin(wanted)].itertuples():
            bars = []
            for a, b in date_chunks(start, end, self.MAX_DAYS_5MIN):
                rows = self.kite.historical_data(r.instrument_token, datetime.combine(a, time(9, 15)),
                                                 datetime.combine(b, time(15, 30)), interval, oi=True)
                bars.extend({"timestamp": x["date"], "open": x["open"], "high": x["high"], "low": x["low"],
                             "close": x["close"], "volume": x.get("volume", 0), "oi": x.get("oi")} for x in rows)
                _time.sleep(self.pause)
            log(f"{underlying} {expiry} {r.strike:g} {r.instrument_type}: {len(bars)} bars")
            frames.append(in_session(make_option_rows(underlying, expiry, r.strike, r.instrument_type, bars)))
        return concat(frames)
