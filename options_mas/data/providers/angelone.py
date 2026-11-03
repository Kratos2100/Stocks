"""Angel One SmartAPI – 5-minute option / index candles and a live tick feed
for NIFTY / BANKNIFTY.

Paths, request bodies, headers and the websocket binary layout are taken from
Angel One's official ``smartapi-python`` SDK (v1.5.5, ``SmartApi/smartConnect.py``
and ``SmartApi/smartWebSocketV2.py``):

* ``POST /rest/auth/angelbroking/user/v1/loginByPassword``  clientcode, password (MPIN), totp
  -> ``data.jwtToken`` / ``refreshToken`` / ``feedToken``
* ``POST /rest/secure/angelbroking/historical/v1/getCandleData``
  exchange, symboltoken, interval ("FIVE_MINUTE"), fromdate / todate ("YYYY-MM-DD HH:MM")
  -> ``data`` = ``[[iso_ts, open, high, low, close, volume], ...]``
* instrument master (public JSON): token, symbol, name, expiry ("04NOV2026"),
  strike (x100), lotsize, instrumenttype, exch_seg
* websocket ``wss://smartapisocket.angelone.in/smart-stream`` (little-endian
  binary ticks, prices in paise)

The SDK itself is **not** used: on a failed request it writes the request
headers (including the API key) to ``logs/<date>/app.log``, it calls an
external IP-lookup service on import, and its websocket disables TLS
certificate checks. Only ``requests`` (REST) and ``websockets`` (live feed)
are needed.

**Limitation:** the historical API serves contracts that are in the current
instrument master, i.e. *listed* contracts. Expired weeklies drop out of the
master, so their history cannot be requested. Build history by fetching after
every session (or by recording the live feed).

Credentials come from the environment, never from files or chat:

* ``ANGEL_API_KEY``      SmartAPI app key
* ``ANGEL_CLIENT_CODE``  Angel One client code
* ``ANGEL_PIN``          login MPIN
* ``ANGEL_TOTP_SECRET``  TOTP seed shown when enabling TOTP (codes are generated
  locally, no extra package), or ``ANGEL_TOTP`` with a current 6-digit code
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import struct
import time as _time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, AsyncIterator, Callable, Iterable, Optional

import pandas as pd

from ...live import MarketDataFeed, Tick
from .common import (concat, date_chunks, empty_options_frame, in_session, make_option_rows,
                     strikes_in_range, to_ist_naive)

ROOT = "https://apiconnect.angelone.in"
LOGIN_PATH = "/rest/auth/angelbroking/user/v1/loginByPassword"
CANDLE_PATH = "/rest/secure/angelbroking/historical/v1/getCandleData"
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
WS_URL = "wss://smartapisocket.angelone.in/smart-stream"
# index tokens on the NSE cash segment (also looked up in the master when available)
INDEX_TOKENS = {"NIFTY": "99926000", "BANKNIFTY": "99926009"}
INDEX_NAMES = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank"}
EXCHANGE_TYPE = {"NSE": 1, "NFO": 2}


class AngelOneError(RuntimeError):
    pass


# ------------------------------------------------------------------ auth
def totp_now(secret: str, now: Optional[float] = None, digits: int = 6, step: int = 30) -> str:
    """RFC 6238 TOTP (HMAC-SHA1), the scheme authenticator apps use."""
    key = base64.b32decode(secret.replace(" ", "").upper() + "=" * (-len(secret.replace(" ", "")) % 8))
    counter = int((now if now is not None else _time.time()) // step)
    h = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = h[-1] & 0x0F
    code = (struct.unpack(">I", h[o:o + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def _headers(api_key: str, jwt: Optional[str] = None) -> dict:
    h = {"Content-type": "application/json", "Accept": "application/json",
         "X-ClientLocalIP": os.environ.get("ANGEL_LOCAL_IP", "127.0.0.1"),
         "X-ClientPublicIP": os.environ.get("ANGEL_PUBLIC_IP", "127.0.0.1"),
         "X-MACAddress": ":".join(f"{(uuid.getnode() >> s) & 0xFF:02x}" for s in range(40, -1, -8)),
         "X-PrivateKey": api_key, "X-UserType": "USER", "X-SourceID": "WEB"}
    if jwt:
        h["Authorization"] = "Bearer " + jwt
    return h


@dataclass
class Session:
    api_key: str
    client_code: str
    jwt: str
    feed_token: str
    refresh_token: str = ""


def _check(body: Any, what: str) -> Any:
    if not isinstance(body, dict):
        raise AngelOneError(f"{what}: unexpected response {str(body)[:200]}")
    if body.get("status") is False or body.get("errorcode") not in (None, "") or body.get("error_type"):
        raise AngelOneError(f"{what}: {body.get('errorcode') or body.get('error_type')} {body.get('message')}")
    return body.get("data")


def login(api_key: str, client_code: str, pin: str, totp: str, http: Any = None) -> Session:
    if http is None:
        import requests
        http = requests
    r = http.post(ROOT + LOGIN_PATH, headers=_headers(api_key), timeout=20,
                  data=json.dumps({"clientcode": client_code, "password": pin, "totp": totp}))
    try:
        body = r.json()
    except Exception:
        raise AngelOneError(f"login: HTTP {r.status_code} (non-JSON response)")
    d = _check(body, "login") or {}
    if not d.get("jwtToken"):
        raise AngelOneError(f"login: no jwtToken in response (HTTP {r.status_code}, message {body.get('message')})")
    return Session(api_key, client_code, d["jwtToken"], d.get("feedToken", ""), d.get("refreshToken", ""))


def session_from_env(http: Any = None) -> Session:
    env = os.environ
    missing = [k for k in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN") if not env.get(k)]
    if missing:
        raise AngelOneError("set " + ", ".join(missing) + " (and ANGEL_TOTP_SECRET or ANGEL_TOTP)")
    totp = env.get("ANGEL_TOTP") or (totp_now(env["ANGEL_TOTP_SECRET"]) if env.get("ANGEL_TOTP_SECRET") else None)
    if not totp:
        raise AngelOneError("set ANGEL_TOTP_SECRET (or ANGEL_TOTP with a current code)")
    return login(env["ANGEL_API_KEY"], env["ANGEL_CLIENT_CODE"], env["ANGEL_PIN"], totp, http)


# ------------------------------------------------------------------ parsers
def parse_scrip_master(items: Iterable[dict], underlying: str, expiry: Optional[date] = None) -> pd.DataFrame:
    """Instrument master -> NFO index-option contracts of one underlying."""
    rows = []
    und = underlying.upper()
    for it in items:
        if it.get("exch_seg") != "NFO" or it.get("instrumenttype") != "OPTIDX" or it.get("name") != und:
            continue
        sym = str(it.get("symbol", ""))
        ot = sym[-2:]
        if ot not in ("CE", "PE"):
            continue
        try:
            exp = datetime.strptime(str(it["expiry"]).title(), "%d%b%Y").date()
        except (KeyError, ValueError):
            continue
        if expiry is not None and exp != expiry:
            continue
        rows.append({"token": str(it["token"]), "underlying": und, "expiry": exp,
                     "strike": float(it["strike"]) / 100.0, "option_type": ot, "symbol": sym,
                     "lot_size": int(float(it.get("lotsize") or 0))})
    return pd.DataFrame(rows, columns=["token", "underlying", "expiry", "strike", "option_type", "symbol", "lot_size"])


def listed_expiries(items: Iterable[dict], underlying: str) -> list[date]:
    return sorted(set(parse_scrip_master(items, underlying)["expiry"]))


def index_token(items: Optional[Iterable[dict]], underlying: str) -> str:
    und = underlying.upper()
    for it in items or []:
        # the master lists the index as name "NIFTY", symbol "Nifty 50", instrumenttype "AMXIDX"
        if it.get("exch_seg") == "NSE" and str(it.get("instrumenttype")) in ("AMXIDX", "INDEX") \
                and (it.get("symbol") == INDEX_NAMES.get(und) or it.get("name") == INDEX_NAMES.get(und)):
            return str(it["token"])
    return INDEX_TOKENS[und]


def parse_candles(data: Any) -> list[dict]:
    out = [{"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4],
            "volume": (r[5] if len(r) > 5 else 0) or 0, "oi": None} for r in (data or [])]
    out.sort(key=lambda b: to_ist_naive(b["timestamp"]))
    return out


# ------------------------------------------------------------------ REST client
class AngelOneClient:
    CHUNK_DAYS = 30   # Angel allows up to 100 days per FIVE_MINUTE request; stay well inside it

    def __init__(self, session: Optional[Session] = None, http: Any = None, pause: float = 0.4,
                 scrip_master: Optional[list[dict]] = None, retries: int = 6, backoff: float = 1.0):
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self.s = session or session_from_env(http)
        self.pause = pause          # historical API rate limit is a few requests per second
        self._master = scrip_master
        self.retries = retries
        self.backoff = backoff

    def master(self) -> list[dict]:
        if self._master is None:
            r = self.http.get(SCRIP_MASTER_URL, timeout=120)
            if r.status_code >= 400:
                raise AngelOneError(f"instrument master: HTTP {r.status_code}")
            self._master = r.json()
        return self._master

    def _post(self, path: str, body: dict) -> Any:
        for attempt in range(self.retries + 1):
            r = self.http.post(ROOT + path, headers=_headers(self.s.api_key, self.s.jwt), data=json.dumps(body),
                               timeout=30)
            if self.pause:
                _time.sleep(self.pause)
            # rate limit: HTTP 403 (sometimes 429) with the plain-text body
            # "Access denied because of exceeding access rate"
            if r.status_code in (403, 429) and "access rate" in (getattr(r, "text", "") or "").lower():
                if attempt < self.retries:
                    _time.sleep(self.backoff * 2 ** attempt)
                    continue
                raise AngelOneError(f"POST {path}: HTTP {r.status_code} rate limit, gave up after {attempt + 1} tries")
            break
        try:
            payload = r.json()
        except Exception:
            raise AngelOneError(f"POST {path}: HTTP {r.status_code} (non-JSON response)")
        return _check(payload, f"POST {path}")

    def candles(self, exchange: str, token: str, start: date, end: date, interval: str = "FIVE_MINUTE") -> list[dict]:
        bars: list[dict] = []
        for a, b in date_chunks(start, end, self.CHUNK_DAYS):
            bars.extend(parse_candles(self._post(CANDLE_PATH, {
                "exchange": exchange, "symboltoken": str(token), "interval": interval,
                "fromdate": datetime.combine(a, time(9, 15)).strftime("%Y-%m-%d %H:%M"),
                "todate": datetime.combine(b, time(15, 30)).strftime("%Y-%m-%d %H:%M")})))
        return bars

    def contracts(self, underlying: str, expiry: Optional[date] = None) -> pd.DataFrame:
        return parse_scrip_master(self.master(), underlying, expiry)

    def index_candles(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        bars = self.candles("NSE", index_token(self.master(), underlying), start, end)
        cols = ["underlying", "timestamp", "open", "high", "low", "close", "volume"]
        if not bars:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(bars)
        df["timestamp"] = df["timestamp"].map(to_ist_naive)
        df.insert(0, "underlying", underlying.upper())
        df = df[cols]
        t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        return df[(t >= 555) & (t < 930)].reset_index(drop=True)

    def fetch_options(self, underlying: str, expiry: date, start: date, end: date,
                      strike_min: Optional[float] = None, strike_max: Optional[float] = None,
                      strikes: Optional[Iterable[float]] = None, log: Callable[[str], None] = print) -> pd.DataFrame:
        con = self.contracts(underlying, expiry)
        if con.empty:
            listed = listed_expiries(self.master(), underlying)
            log(f"no listed {underlying} contracts for {expiry} in the Angel One instrument master "
                f"(expired contracts are not served). Listed expiries: {[d.isoformat() for d in listed[:8]]}")
            return empty_options_frame()
        wanted = {float(s) for s in strikes} if strikes else set(
            strikes_in_range(con["strike"], strike_min, strike_max))
        frames = []
        for r in con[con["strike"].isin(wanted)].itertuples():
            bars = self.candles("NFO", r.token, start, min(end, expiry))
            log(f"{r.symbol}: {len(bars)} bars")
            frames.append(in_session(make_option_rows(underlying.upper(), expiry, r.strike, r.option_type, bars)))
        return concat(frames)

    def probe(self, underlying: str, day: date) -> dict:
        """One small request each, to confirm the response shapes before a large download."""
        m = self.master()
        con = parse_scrip_master(m, underlying)
        out: dict[str, Any] = {"master_rows": len(m), "listed_contracts": len(con),
                               "listed_expiries": [d.isoformat() for d in sorted(set(con["expiry"]))[:6]],
                               "index_token": index_token(m, underlying)}
        raw_idx = self._post(CANDLE_PATH, {"exchange": "NSE", "symboltoken": out["index_token"],
                                           "interval": "FIVE_MINUTE", "fromdate": f"{day} 09:15",
                                           "todate": f"{day} 10:00"})
        out["index_candles_sample"] = (raw_idx or [])[:3]
        out["parsed_index_candles"] = len(parse_candles(raw_idx))
        if not con.empty:
            first = con[con["expiry"] == con["expiry"].min()]
            ref = None
            try:
                ref = float(raw_idx[-1][4]) if raw_idx else None
            except (IndexError, TypeError, ValueError):
                ref = None
            if ref is None:
                r = first.iloc[len(first) // 2]
            else:   # the strike nearest the index: far-OTM strikes often have no trades
                r = first.iloc[int((first["strike"] - ref).abs().to_numpy().argmin())]
            raw = self._post(CANDLE_PATH, {"exchange": "NFO", "symboltoken": r["token"], "interval": "FIVE_MINUTE",
                                           "fromdate": f"{day} 09:15", "todate": f"{day} 10:00"})
            out["option_symbol"] = r["symbol"]
            out["option_candles_sample"] = (raw or [])[:3]
            out["parsed_option_candles"] = len(parse_candles(raw))
        return out


# ------------------------------------------------------------------ live feed
def parse_tick_packet(buf: bytes) -> dict:
    """Decode one SmartWebSocketV2 binary packet (LTP / Quote / SnapQuote modes).
    Prices arrive in paise and are returned in rupees."""
    u = lambda a, b, f: struct.unpack("<" + f, buf[a:b])[0]  # noqa: E731
    tok = buf[2:27].split(b"\x00", 1)[0].decode()
    d = {"mode": u(0, 1, "B"), "exchange_type": u(1, 2, "B"), "token": tok, "sequence": u(27, 35, "q"),
         "exchange_ts_ms": u(35, 43, "q"), "ltp": u(43, 51, "q") / 100.0}
    if d["mode"] in (2, 3) and len(buf) >= 123:
        d.update(last_qty=u(51, 59, "q"), avg_price=u(59, 67, "q") / 100.0, volume_today=u(67, 75, "q"),
                 total_buy_qty=u(75, 83, "d"), total_sell_qty=u(83, 91, "d"), open=u(91, 99, "q") / 100.0,
                 high=u(99, 107, "q") / 100.0, low=u(107, 115, "q") / 100.0, close=u(115, 123, "q") / 100.0)
    if d["mode"] == 3 and len(buf) >= 347:
        d["last_trade_ts"] = u(123, 131, "q")
        d["oi"] = u(131, 139, "q")
        bids, asks = [], []
        for i in range(147, 347, 20):   # flag 1 = buy side, 0 = sell side
            flag, qty, px = u(i, i + 2, "H"), u(i + 2, i + 10, "q"), u(i + 10, i + 18, "q") / 100.0
            (bids if flag == 1 else asks).append((px, qty))
        d["best_bid"] = max((p for p, q in bids if q > 0), default=None)
        d["best_ask"] = min((p for p, q in asks if q > 0), default=None)
    return d


class AngelOneFeed(MarketDataFeed):
    """Live ticks from SmartAPI's websocket for the runtime in ``options_mas.live``.

    ``contracts`` maps our contract key (``NIFTY-20261110-25000-CE``) to
    ``(exchange, token)``, e.g. ``("NFO", "45678")``. Each tick's quantity is
    the growth of the exchange's cumulative day volume since the previous tick
    of that contract, so bar volume matches the exchange's. Ticks are
    snapshots, not every trade, and carry no bid/ask, so live bars get the
    labelled ``ESTIMATED_TICK_RULE`` delta, never a bid/ask-classified one.
    """

    def __init__(self, session: Session, contracts: dict[str, tuple[str, str]], mode: int = 2,
                 reconnect_delay: float = 2.0, connect: Optional[Callable[..., Any]] = None):
        self.s = session
        self.all = contracts
        self.mode = mode
        self.reconnect_delay = reconnect_delay
        self._connect = connect
        self.keys: list[str] = []
        self._by_token: dict[tuple[int, str], str] = {}
        self._last_vol: dict[str, int] = {}

    def subscribe(self, keys: Iterable[str]) -> None:
        self.keys = [k for k in keys if k in self.all]
        self._by_token = {(EXCHANGE_TYPE[self.all[k][0]], str(self.all[k][1])): k for k in self.keys}

    def subscribe_message(self) -> str:
        groups: dict[int, list[str]] = {}
        for (ex, tok) in self._by_token:
            groups.setdefault(ex, []).append(tok)
        return json.dumps({"correlationID": "optionsmas", "action": 1, "params": {
            "mode": self.mode, "tokenList": [{"exchangeType": ex, "tokens": t} for ex, t in sorted(groups.items())]}})

    def to_tick(self, d: dict) -> Optional[Tick]:
        key = self._by_token.get((d["exchange_type"], d["token"]))
        if key is None or d["ltp"] <= 0:
            return None
        qty = 0.0
        if "volume_today" in d:
            prev = self._last_vol.get(key)
            qty = float(max(0, d["volume_today"] - prev)) if prev is not None else 0.0
            self._last_vol[key] = d["volume_today"]
        ts = to_ist_naive(pd.Timestamp(d["exchange_ts_ms"], unit="ms", tz="UTC")).to_pydatetime()
        return Tick(key, ts, d["ltp"], qty)

    def _ws_headers(self) -> dict:
        return {"Authorization": self.s.jwt, "x-api-key": self.s.api_key, "x-client-code": self.s.client_code,
                "x-feed-token": self.s.feed_token}

    async def stream(self) -> AsyncIterator[Tick]:  # pragma: no cover - needs the live socket
        connect = self._connect
        if connect is None:
            import websockets
            connect = websockets.connect
        while True:
            try:
                async with connect(WS_URL, additional_headers=self._ws_headers(), ping_interval=None) as ws:
                    await ws.send(self.subscribe_message())
                    hb = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for msg in ws:
                            if isinstance(msg, (bytes, bytearray)) and len(msg) >= 51:
                                t = self.to_tick(parse_tick_packet(bytes(msg)))
                                if t is not None:
                                    yield t
                    finally:
                        hb.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # dropped connection: resubscribe after a pause
                print(f"Angel One feed disconnected ({type(e).__name__}); reconnecting")
            await asyncio.sleep(self.reconnect_delay)

    @staticmethod
    async def _heartbeat(ws) -> None:  # pragma: no cover
        while True:
            await asyncio.sleep(10)
            await ws.send("ping")
