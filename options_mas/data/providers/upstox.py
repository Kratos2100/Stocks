"""Upstox API v2/v3 – historical 5-minute option candles (with OI) for active
**and expired** NIFTY / BANKNIFTY contracts.

Endpoints (from the official ``upstox-python-sdk``):

* ``GET /v2/option/contract?instrument_key=…&expiry_date=YYYY-MM-DD``               active contracts
* ``GET /v2/expired-instruments/expiries?instrument_key=…``                         expired expiries
* ``GET /v2/expired-instruments/option/contract?instrument_key=…&expiry_date=…``    expired contracts
* ``GET /v3/historical-candle/{key}/minutes/{n}/{to}/{from}``                       active contract candles
* ``GET /v2/expired-instruments/historical-candle/{key}/{interval}/{to}/{from}``     expired contract candles

Candle rows are ``[timestamp, open, high, low, close, volume, oi]``, newest
first. Authentication: an OAuth access token (valid for one trading day),
passed via the ``UPSTOX_ACCESS_TOKEN`` environment variable – never commit
it. The expired-instruments API may require a paid Upstox plan; check your
account. Only plain ``requests`` is needed.
"""
from __future__ import annotations

import os
import time as _time
from datetime import date
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .common import concat, date_chunks, empty_options_frame, in_session, make_option_rows, strikes_in_range

BASE = "https://api.upstox.com"
UNDERLYING_KEYS = {"NIFTY": "NSE_INDEX|Nifty 50", "BANKNIFTY": "NSE_INDEX|Nifty Bank"}


class UpstoxError(RuntimeError):
    pass


class UpstoxClient:
    def __init__(self, access_token: Optional[str] = None, session: Any = None, pause: float = 0.25):
        self.token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not self.token:
            raise UpstoxError("set UPSTOX_ACCESS_TOKEN (Upstox developer console -> OAuth access token)")
        if session is None:
            import requests
            session = requests.Session()
        self.http = session
        self.pause = pause

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        r = self.http.get(BASE + path, params=params,
                          headers={"Accept": "application/json", "Authorization": f"Bearer {self.token}"},
                          timeout=30)
        if getattr(r, "status_code", 200) >= 400:
            raise UpstoxError(f"GET {path} -> HTTP {r.status_code}: {getattr(r, 'text', '')[:300]}")
        body = r.json()
        if body.get("status") not in (None, "success"):
            raise UpstoxError(f"GET {path} -> {body}")
        if self.pause:
            _time.sleep(self.pause)
        return body.get("data")

    # ---------------------------------------------------------- contracts
    def option_contracts(self, underlying: str, expiry: date, expired: bool = False) -> pd.DataFrame:
        key = UNDERLYING_KEYS.get(underlying.upper(), underlying)
        path = "/v2/expired-instruments/option/contract" if expired else "/v2/option/contract"
        data = self._get(path, {"instrument_key": key, "expiry_date": expiry.isoformat()}) or []
        df = pd.DataFrame(data)
        if df.empty:
            return df
        return pd.DataFrame({
            "instrument_key": df["instrument_key"], "underlying": underlying.upper(),
            "expiry": pd.to_datetime(df["expiry"]).dt.date, "strike": df["strike_price"].astype(float),
            "option_type": df["instrument_type"].str.upper(), "lot_size": df.get("lot_size"),
            "trading_symbol": df.get("trading_symbol"),
        })

    def expired_expiries(self, underlying: str) -> list[date]:
        key = UNDERLYING_KEYS.get(underlying.upper(), underlying)
        return sorted(pd.to_datetime(x).date() for x in (self._get("/v2/expired-instruments/expiries",
                                                                      {"instrument_key": key}) or []))

    # ------------------------------------------------------------ candles
    def candles(self, instrument_key: str, start: date, end: date, minutes: int = 5,
                expired: bool = False) -> list[dict]:
        """Raw bars (oldest first). Requests are chunked to stay within the
        API's per-call date range limits."""
        bars: list[dict] = []
        for a, b in date_chunks(start, end, 28):
            if expired:
                path = f"/v2/expired-instruments/historical-candle/{instrument_key}/{minutes}minute/{b}/{a}"
            else:
                path = f"/v3/historical-candle/{instrument_key}/minutes/{minutes}/{b}/{a}"
            data = self._get(path) or {}
            bars.extend(parse_candles(data.get("candles", [])))
        bars.sort(key=lambda r: str(r["timestamp"]))
        return bars

    def index_candles(self, underlying: str, start: date, end: date, minutes: int = 5) -> pd.DataFrame:
        key = UNDERLYING_KEYS[underlying.upper()]
        bars = self.candles(key, start, end, minutes)
        df = pd.DataFrame(bars)
        if df.empty:
            return df
        from .common import to_ist_naive
        df["timestamp"] = df["timestamp"].map(to_ist_naive)
        df.insert(0, "underlying", underlying.upper())
        return df[["underlying", "timestamp", "open", "high", "low", "close", "volume"]]

    # --------------------------------------------------------- high level
    def fetch_options(self, underlying: str, expiry: date, start: date, end: date,
                      strike_min: Optional[float] = None, strike_max: Optional[float] = None,
                      strikes: Optional[Iterable[float]] = None, expired: Optional[bool] = None,
                      minutes: int = 5, log: Callable[[str], None] = print) -> pd.DataFrame:
        """Download candles for every CE and PE of ``expiry`` within the strike
        range (or the explicit ``strikes``) and return the long format."""
        if expired is None:
            expired = expiry < date.today()
        contracts = self.option_contracts(underlying, expiry, expired)
        if contracts.empty:
            log(f"no {underlying} contracts for expiry {expiry} (expired={expired})")
            return empty_options_frame()
        wanted = set(float(s) for s in strikes) if strikes else set(
            strikes_in_range(contracts["strike"], strike_min, strike_max))
        sel = contracts[contracts["strike"].isin(wanted)]
        frames = []
        for r in sel.itertuples():
            bars = self.candles(r.instrument_key, start, min(end, expiry), minutes, expired)
            log(f"{underlying} {expiry} {r.strike:g} {r.option_type}: {len(bars)} bars")
            frames.append(in_session(make_option_rows(underlying, expiry, r.strike, r.option_type, bars)))
        return concat(frames)


def parse_candles(rows: list) -> list[dict]:
    """``[ts, o, h, l, c, volume, oi]`` -> dicts (timestamps keep their offset)."""
    out = []
    for row in rows:
        out.append({"timestamp": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4],
                    "volume": row[5] if len(row) > 5 else 0, "oi": row[6] if len(row) > 6 else None})
    return out
