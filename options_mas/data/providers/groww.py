"""Groww Trade API – historical 5-minute option candles (active **and expired**
contracts, via Groww's backtesting endpoints) for NIFTY / BANKNIFTY.

Endpoints, parameters and the auth flow are taken from the official
``growwapi`` SDK (v1.5.0, ``growwapi/groww/client.py``):

* ``POST /v1/token/api/access``                     API key (+ secret checksum or TOTP) -> access token
* ``GET  /v1/historical/expiries``   exchange, underlying_symbol, year[, month]
* ``GET  /v1/historical/contracts``  exchange, underlying_symbol, expiry_date (YYYY-MM-DD)
* ``GET  /v1/historical/candles``    exchange, segment, groww_symbol, start_time, end_time,
                                     candle_interval ("5minute", ...)

Responses are ``{"status": ..., "payload": {...}}``. The SDK does not
document the payload bodies, so the parsers below accept the plausible
shapes (lists of groww symbols or of dicts; candles as arrays or dicts;
string or epoch timestamps). Run ``cli fetch groww-probe`` once with your key
to print the real shapes before a large download.

The SDK itself is **not** required (it pins ``pandas<3`` and makes a network
call in its constructor); only ``requests`` is used.

Credentials come from the environment, never from files or chat:

* ``GROWW_ACCESS_TOKEN`` – a token generated on groww.in (valid for the day), or
* ``GROWW_API_KEY`` + ``GROWW_API_SECRET`` – "approval" key type, or
* ``GROWW_API_KEY`` + ``GROWW_TOTP`` – TOTP key type (current 6-digit code), or
  ``GROWW_API_KEY`` + ``GROWW_TOTP_SECRET`` with ``pip install pyotp``.
"""
from __future__ import annotations

import hashlib
import os
import re
import time as _time
import uuid
from datetime import date, datetime, time
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from .common import (concat, date_chunks, empty_options_frame, in_session, make_option_rows,
                     strikes_in_range, to_ist_naive)

BASE = "https://api.groww.in/v1"
TOKEN_URL = BASE + "/token/api/access"
INDEX_SYMBOLS = {"NIFTY": "NSE-NIFTY", "BANKNIFTY": "NSE-BANKNIFTY"}
# e.g. NSE-NIFTY-04Nov26-25000-CE  (exchange-underlying-ddMonyy-strike-type)
_SYM = re.compile(r"^(?:(?P<ex>[A-Z]+)-)?(?P<und>[A-Z0-9&]+)-(?P<d>\d{1,2})(?P<mon>[A-Za-z]{3})(?P<yy>\d{2})-"
                  r"(?P<strike>\d+(?:\.\d+)?)-(?P<ot>CE|PE)$")


class GrowwError(RuntimeError):
    pass


def _headers(token: str) -> dict:
    return {"x-request-id": str(uuid.uuid4()), "Authorization": "Bearer " + token,
            "Content-Type": "application/json", "x-client-id": "growwapi",
            "x-client-platform": "growwapi-python-client", "x-client-platform-version": "1.5.0",
            "x-api-version": "1.0"}


def approval_payload(secret: str, now: Optional[int] = None) -> dict:
    """Same checksum scheme as the SDK: sha256(secret + str(epoch_seconds))."""
    ts = int(now if now is not None else _time.time())
    return {"key_type": "approval", "checksum": hashlib.sha256((secret + str(ts)).encode()).hexdigest(),
            "timestamp": ts}


def get_access_token(api_key: str, secret: Optional[str] = None, totp: Optional[str] = None,
                     session: Any = None) -> str:
    if (secret is None) == (totp is None):
        raise GrowwError("provide exactly one of secret or totp")
    body = approval_payload(secret) if secret is not None else {"key_type": "totp", "totp": totp.strip()}
    if session is None:
        import requests
        session = requests
    r = session.post(TOKEN_URL, headers=_headers(api_key), json=body, timeout=15)
    if r.status_code >= 400:
        raise GrowwError(f"token request failed: HTTP {r.status_code} {getattr(r, 'text', '')[:200]}")
    return r.json()["token"]


def token_from_env(session: Any = None) -> str:
    env = os.environ
    if env.get("GROWW_ACCESS_TOKEN"):
        return env["GROWW_ACCESS_TOKEN"]
    key = env.get("GROWW_API_KEY")
    if not key:
        raise GrowwError("set GROWW_ACCESS_TOKEN, or GROWW_API_KEY with GROWW_API_SECRET / GROWW_TOTP / "
                         "GROWW_TOTP_SECRET")
    if env.get("GROWW_API_SECRET"):
        return get_access_token(key, secret=env["GROWW_API_SECRET"], session=session)
    totp = env.get("GROWW_TOTP")
    if not totp and env.get("GROWW_TOTP_SECRET"):
        import pyotp  # optional
        totp = pyotp.TOTP(env["GROWW_TOTP_SECRET"]).now()
    if totp:
        return get_access_token(key, totp=totp, session=session)
    raise GrowwError("GROWW_API_KEY is set but no GROWW_API_SECRET / GROWW_TOTP / GROWW_TOTP_SECRET")


# ------------------------------------------------------------------ parsers
def parse_groww_symbol(sym: str) -> Optional[dict]:
    m = _SYM.match(sym.strip())
    if not m:
        return None
    exp = datetime.strptime(f"{m['d']}{m['mon'].title()}{m['yy']}", "%d%b%y").date()
    return {"underlying": m["und"], "expiry": exp, "strike": float(m["strike"]), "option_type": m["ot"],
            "groww_symbol": sym.strip()}


def parse_contracts(payload: Any, underlying: str, expiry: date) -> pd.DataFrame:
    items = payload.get("contracts", payload.get("data", payload)) if isinstance(payload, dict) else payload
    rows = []
    for it in items or []:
        if isinstance(it, str):
            p = parse_groww_symbol(it)
        else:
            sym = it.get("groww_symbol") or it.get("symbol") or ""
            p = parse_groww_symbol(sym) if sym else None
            if p is None and it.get("strike_price") is not None:
                p = {"underlying": underlying.upper(), "expiry": expiry, "strike": float(it["strike_price"]),
                     "option_type": str(it.get("instrument_type", "")).upper(), "groww_symbol": sym}
        if p and p["option_type"] in ("CE", "PE"):
            rows.append(p)
    return pd.DataFrame(rows, columns=["underlying", "expiry", "strike", "option_type", "groww_symbol"])


def parse_expiries(payload: Any) -> list[date]:
    items = payload.get("expiries", payload.get("data", payload)) if isinstance(payload, dict) else payload
    return sorted(pd.Timestamp(x).date() for x in items or [])


def _ts(v: Any):
    if isinstance(v, (int, float)):
        return pd.Timestamp(v, unit="ms" if v > 1e11 else "s", tz="UTC")  # epoch ms or s
    return v  # ISO string: naive strings are already IST


def parse_candles(payload: Any) -> list[dict]:
    rows = payload.get("candles", []) if isinstance(payload, dict) else payload or []
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append({"timestamp": _ts(r.get("timestamp") or r.get("time")), "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"], "volume": r.get("volume") or 0,
                        "oi": r.get("open_interest", r.get("oi"))})
        else:
            out.append({"timestamp": _ts(r[0]), "open": r[1], "high": r[2], "low": r[3], "close": r[4],
                        "volume": (r[5] if len(r) > 5 else 0) or 0, "oi": r[6] if len(r) > 6 else None})
    out.sort(key=lambda b: to_ist_naive(b["timestamp"]))
    return out


# ------------------------------------------------------------------ client
class GrowwClient:
    CHUNK_DAYS = 7  # conservative request window for minute candles

    def __init__(self, access_token: Optional[str] = None, session: Any = None, pause: float = 0.15):
        if session is None:
            import requests
            session = requests.Session()
        self.http = session
        self.token = access_token or token_from_env(session)
        self.pause = pause

    def _get(self, path: str, params: dict) -> Any:
        r = self.http.get(BASE + path, params=params, headers=_headers(self.token), timeout=30)
        try:
            body = r.json()
        except Exception:
            body = {}
        if body.get("status") == "FAILURE":
            err = body.get("error", {})
            raise GrowwError(f"GET {path}: {err.get('code')} {err.get('message')}")
        if r.status_code >= 400:
            raise GrowwError(f"GET {path} -> HTTP {r.status_code}: {getattr(r, 'text', '')[:300]}")
        if self.pause:
            _time.sleep(self.pause)
        return body.get("payload", body)

    def expiries(self, underlying: str, year: int, month: Optional[int] = None) -> list[date]:
        params = {"exchange": "NSE", "underlying_symbol": underlying.upper(), "year": str(year)}
        if month:
            params["month"] = str(month)
        return parse_expiries(self._get("/historical/expiries", params))

    def contracts(self, underlying: str, expiry: date) -> pd.DataFrame:
        return parse_contracts(self._get("/historical/contracts", {
            "exchange": "NSE", "underlying_symbol": underlying.upper(), "expiry_date": expiry.isoformat()}),
            underlying, expiry)

    def candles(self, groww_symbol: str, start: date, end: date, segment: str = "FNO",
                interval: str = "5minute") -> list[dict]:
        bars: list[dict] = []
        for a, b in date_chunks(start, end, self.CHUNK_DAYS):
            bars.extend(parse_candles(self._get("/historical/candles", {
                "exchange": "NSE", "segment": segment, "groww_symbol": groww_symbol,
                "start_time": datetime.combine(a, time(9, 15)).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.combine(b, time(15, 30)).strftime("%Y-%m-%d %H:%M:%S"),
                "candle_interval": interval})))
        return bars

    def index_candles(self, underlying: str, start: date, end: date, interval: str = "5minute") -> pd.DataFrame:
        bars = self.candles(INDEX_SYMBOLS[underlying.upper()], start, end, segment="CASH", interval=interval)
        if not bars:
            return pd.DataFrame(columns=["underlying", "timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(bars)
        df["timestamp"] = df["timestamp"].map(to_ist_naive)
        df.insert(0, "underlying", underlying.upper())
        return df[["underlying", "timestamp", "open", "high", "low", "close", "volume"]]

    def fetch_options(self, underlying: str, expiry: date, start: date, end: date,
                      strike_min: Optional[float] = None, strike_max: Optional[float] = None,
                      strikes: Optional[Iterable[float]] = None, interval: str = "5minute",
                      log: Callable[[str], None] = print) -> pd.DataFrame:
        con = self.contracts(underlying, expiry)
        if con.empty:
            log(f"no {underlying} contracts returned for expiry {expiry}")
            return empty_options_frame()
        wanted = {float(s) for s in strikes} if strikes else set(
            strikes_in_range(con["strike"], strike_min, strike_max))
        sel = con[con["strike"].isin(wanted)]
        frames = []
        for r in sel.itertuples():
            bars = self.candles(r.groww_symbol, start, min(end, expiry), interval=interval)
            log(f"{r.groww_symbol}: {len(bars)} bars")
            frames.append(in_session(make_option_rows(underlying.upper(), expiry, r.strike, r.option_type, bars)))
        return concat(frames)

    def probe(self, underlying: str, expiry: date, day: date) -> dict:
        """Raw responses for one small request each – use this to confirm the
        payload shapes before a large download."""
        out: dict[str, Any] = {"expiries": self._get("/historical/expiries", {
            "exchange": "NSE", "underlying_symbol": underlying.upper(), "year": str(expiry.year),
            "month": str(expiry.month)})}
        raw_con = self._get("/historical/contracts", {"exchange": "NSE", "underlying_symbol": underlying.upper(),
                                                       "expiry_date": expiry.isoformat()})
        out["contracts_sample"] = raw_con if not isinstance(raw_con, (list, dict)) else (
            raw_con[:5] if isinstance(raw_con, list) else {k: (v[:5] if isinstance(v, list) else v)
                                                           for k, v in raw_con.items()})
        con = parse_contracts(raw_con, underlying, expiry)
        out["parsed_contracts"] = len(con)
        if not con.empty:
            sym = con.iloc[len(con) // 2]["groww_symbol"]
            raw = self._get("/historical/candles", {
                "exchange": "NSE", "segment": "FNO", "groww_symbol": sym,
                "start_time": f"{day} 09:15:00", "end_time": f"{day} 10:00:00", "candle_interval": "5minute"})
            out["candles_symbol"] = sym
            out["candles_sample"] = {k: (v[:3] if isinstance(v, list) else v) for k, v in raw.items()} \
                if isinstance(raw, dict) else raw
            out["parsed_candles"] = len(parse_candles(raw))
        return out
