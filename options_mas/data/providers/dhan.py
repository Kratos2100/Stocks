"""DhanHQ v2 – 5-minute option candles via ``dhanhq`` (``pip install dhanhq``).

Credentials: ``DHAN_CLIENT_ID`` and ``DHAN_ACCESS_TOKEN``.

* Listed contracts: ``dhan.intraday_minute_data(security_id, "NSE_FNO",
  "OPTIDX", from, to, interval=5, oi=True)``. Security ids come from Dhan's
  public scrip master CSV.
* Expired contracts: ``dhan.expired_options_data(...)`` returns **rolling**
  series relative to ATM (e.g. ``ATM+1``): the actual strike changes from
  bar to bar. :func:`split_rolling_by_strike` splits such a series back
  into fixed-strike contracts, but each contract then only has bars for the
  periods when it sat at that ATM offset – too sparse for 50-EMA warm-up.
  Prefer Upstox for complete expired-contract history.

The response layout handled here (parallel arrays ``open/high/low/close/
volume/timestamp[/open_interest]``) follows Dhan's v2 documentation; verify it
against your account before relying on it.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .common import concat, date_chunks, empty_options_frame, in_session, make_option_rows, strikes_in_range

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


def make_dhan(client_id: Optional[str] = None, access_token: Optional[str] = None):
    from dhanhq import DhanContext, dhanhq  # optional dependency
    cid = client_id or os.environ.get("DHAN_CLIENT_ID")
    tok = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
    if not cid or not tok:
        raise RuntimeError("set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")
    return dhanhq(DhanContext(cid, tok))


def parse_scrip_master(df: pd.DataFrame, underlying: str, expiry: date) -> pd.DataFrame:
    df = df[(df["SEM_EXM_EXCH_ID"] == "NSE") & (df["SEM_INSTRUMENT_NAME"] == "OPTIDX")].copy()
    df["underlying"] = df["SEM_TRADING_SYMBOL"].astype(str).str.split("-").str[0]
    df["expiry"] = pd.to_datetime(df["SEM_EXPIRY_DATE"]).dt.date
    df = df[(df["underlying"] == underlying.upper()) & (df["expiry"] == expiry)]
    return pd.DataFrame({"security_id": df["SEM_SMST_SECURITY_ID"].astype(str), "underlying": underlying.upper(),
                         "expiry": df["expiry"], "strike": df["SEM_STRIKE_PRICE"].astype(float),
                         "option_type": df["SEM_OPTION_TYPE"].astype(str), "lot_size": df.get("SEM_LOT_UNITS")})


def parse_arrays(resp: dict) -> list[dict]:
    """Dhan chart response -> bar dicts. Accepts the raw API body or the SDK
    wrapper ``{"status", "remarks", "data"}``."""
    if isinstance(resp, dict) and resp.get("status") == "failure":
        raise RuntimeError(f"Dhan error: {resp.get('remarks')}")
    d = resp.get("data", resp) if isinstance(resp, dict) else {}
    ts = d.get("timestamp") or []
    oi = d.get("open_interest") or d.get("oi") or [None] * len(ts)
    return [{"timestamp": ts[i], "open": d["open"][i], "high": d["high"][i], "low": d["low"][i],
             "close": d["close"][i], "volume": (d.get("volume") or [0] * len(ts))[i], "oi": oi[i]}
            for i in range(len(ts))]


def split_rolling_by_strike(resp: dict, underlying: str, expiry: date, option_type: str) -> pd.DataFrame:
    """Expired-options (rolling) response -> fixed-strike long format."""
    d = resp.get("data", resp)
    leg = d.get("ce") or d.get("pe") or d
    bars = parse_arrays({"data": leg})
    strikes = leg.get("strike") or []
    frames = []
    by: dict[float, list[dict]] = {}
    for b, k in zip(bars, strikes):
        by.setdefault(float(k), []).append(b)
    for k, lst in by.items():
        frames.append(make_option_rows(underlying, expiry, k, option_type, lst))
    return concat(frames)


class DhanSource:
    MAX_DAYS = 90

    def __init__(self, dhan: Any, scrip_master: Optional[pd.DataFrame] = None):
        self.dhan = dhan
        self._master = scrip_master

    def master(self) -> pd.DataFrame:
        if self._master is None:
            self._master = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)
        return self._master

    def fetch_options(self, underlying: str, expiry: date, start: date, end: date,
                      strike_min: Optional[float] = None, strike_max: Optional[float] = None,
                      strikes: Optional[Iterable[float]] = None, minutes: int = 5,
                      log: Callable[[str], None] = print) -> pd.DataFrame:
        con = parse_scrip_master(self.master(), underlying, expiry)
        if con.empty:
            log(f"no listed {underlying} contracts for {expiry} in Dhan scrip master")
            return empty_options_frame()
        wanted = set(float(s) for s in strikes) if strikes else set(
            strikes_in_range(con["strike"], strike_min, strike_max))
        frames = []
        for r in con[con["strike"].isin(wanted)].itertuples():
            bars = []
            for a, b in date_chunks(start, end, self.MAX_DAYS):
                resp = self.dhan.intraday_minute_data(r.security_id, "NSE_FNO", "OPTIDX", a.isoformat(),
                                                      b.isoformat(), interval=minutes, oi=True)
                bars.extend(parse_arrays(resp))
            log(f"{underlying} {expiry} {r.strike:g} {r.option_type}: {len(bars)} bars")
            frames.append(in_session(make_option_rows(underlying, expiry, r.strike, r.option_type, bars)))
        return concat(frames)
