"""Angel One SmartAPI adapter tests with hand-written fixtures (not market data).

Request paths, bodies and the websocket packet layout mirror smartapi-python
1.5.5. The response bodies are shaped like Angel One's documented responses;
The tests at the end use the shapes seen on the live API (2026-11-03) with synthetic values.
"""
import asyncio
import json
import struct
from datetime import date, datetime

import pandas as pd
import pytest

from options_mas.backtest.engine import Backtester
from options_mas.config import AppConfig
from options_mas.data.dataset import load_dataset
from options_mas.data.providers import angelone as a

MASTER = [
    {"token": "50001", "symbol": "NIFTY10NOV2625000CE", "name": "NIFTY", "expiry": "10NOV2026",
     "strike": "2500000.000000", "lotsize": "75", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"token": "50002", "symbol": "NIFTY10NOV2625000PE", "name": "NIFTY", "expiry": "10NOV2026",
     "strike": "2500000.000000", "lotsize": "75", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"token": "50003", "symbol": "NIFTY10NOV2627000CE", "name": "NIFTY", "expiry": "10NOV2026",
     "strike": "2700000.000000", "lotsize": "75", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"token": "50004", "symbol": "NIFTY17NOV2625000CE", "name": "NIFTY", "expiry": "17NOV2026",
     "strike": "2500000.000000", "lotsize": "75", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"token": "60001", "symbol": "BANKNIFTY24NOV2657000CE", "name": "BANKNIFTY", "expiry": "24NOV2026",
     "strike": "5700000.000000", "lotsize": "35", "instrumenttype": "OPTIDX", "exch_seg": "NFO"},
    {"token": "70001", "symbol": "NIFTY24NOV26FUT", "name": "NIFTY", "expiry": "24NOV2026",
     "strike": "-1.000000", "lotsize": "75", "instrumenttype": "FUTIDX", "exch_seg": "NFO"},
    {"token": "99926000", "symbol": "Nifty 50", "name": "Nifty 50", "expiry": "", "strike": "0.000000",
     "lotsize": "1", "instrumenttype": "AMXIDX", "exch_seg": "NSE"},
]


def test_totp_matches_rfc6238_vector():
    # RFC 6238 SHA1 seed "12345678901234567890" (base32 below), T=59 -> 94287082 (8 digits)
    seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    assert a.totp_now(seed, now=59, digits=8) == "94287082"
    assert a.totp_now(seed, now=59) == "287082"
    assert a.totp_now(seed.lower(), now=1111111109, digits=8) == "07081804"


def test_scrip_master_parsing():
    con = a.parse_scrip_master(MASTER, "NIFTY", date(2026, 11, 10))
    assert sorted(zip(con.strike, con.option_type)) == [(25000.0, "CE"), (25000.0, "PE"), (27000.0, "CE")]
    assert set(con.lot_size) == {75} and con.iloc[0]["token"] == "50001"
    assert a.listed_expiries(MASTER, "NIFTY") == [date(2026, 11, 10), date(2026, 11, 17)]
    assert a.index_token(MASTER, "NIFTY") == "99926000" and a.index_token(None, "BANKNIFTY") == "99926009"


def test_candles_parse_ist_offset_and_sort():
    bars = a.parse_candles([["2026-11-02T09:20:00+05:30", 2, 3, 1, 2.5, 10],
                            ["2026-11-02T09:15:00+05:30", 1, 2, 0.5, 1.5, 20]])
    from options_mas.data.providers.common import to_ist_naive
    assert [to_ist_naive(b["timestamp"]) for b in bars] == [pd.Timestamp("2026-11-02 09:15"),
                                                           pd.Timestamp("2026-11-02 09:20")]
    assert bars[0]["volume"] == 20 and bars[0]["oi"] is None and a.parse_candles(None) == []


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeAngel:
    def __init__(self, login_ok=True):
        self.posts, self.login_ok = [], login_ok

    def post(self, url, headers=None, data=None, timeout=None):
        body = json.loads(data)
        self.posts.append((url, headers, body))
        if url.endswith(a.LOGIN_PATH):
            if not self.login_ok:
                return Resp({"status": False, "message": "Invalid totp", "errorcode": "AB1050", "data": None})
            return Resp({"status": True, "message": "SUCCESS", "errorcode": "",
                         "data": {"jwtToken": "JWT", "refreshToken": "R", "feedToken": "F"}})
        assert url.endswith(a.CANDLE_PATH) and headers["Authorization"] == "Bearer JWT"
        day = body["fromdate"][:10]
        rows = [[f"{day}T{9 + (15 + 5 * i) // 60:02d}:{(15 + 5 * i) % 60:02d}:00+05:30", 100 + i, 101 + i, 99 + i,
                 100.5 + i, 1000] for i in range(75)]
        return Resp({"status": True, "message": "SUCCESS", "errorcode": "", "data": rows})

    def get(self, url, timeout=None):
        return Resp(MASTER)


def test_login_from_env(monkeypatch):
    monkeypatch.delenv("ANGEL_TOTP", raising=False)
    for k, v in {"ANGEL_API_KEY": "KEY", "ANGEL_CLIENT_CODE": "C123", "ANGEL_PIN": "1111",
                 "ANGEL_TOTP_SECRET": "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"}.items():
        monkeypatch.setenv(k, v)
    http = FakeAngel()
    s = a.session_from_env(http)
    assert (s.jwt, s.feed_token, s.client_code) == ("JWT", "F", "C123")
    url, headers, body = http.posts[0]
    assert url == "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword"
    assert headers["X-PrivateKey"] == "KEY" and "Authorization" not in headers
    assert body["clientcode"] == "C123" and body["password"] == "1111" and len(body["totp"]) == 6
    with pytest.raises(a.AngelOneError, match="AB1050 Invalid totp"):
        a.session_from_env(FakeAngel(login_ok=False))
    monkeypatch.delenv("ANGEL_PIN")
    with pytest.raises(a.AngelOneError, match="ANGEL_PIN"):
        a.session_from_env(http)


def test_fetch_listed_contracts_index_and_backtest(tmp_path):
    http = FakeAngel()
    c = a.AngelOneClient(a.Session("KEY", "C123", "JWT", "F"), http=http, pause=0)
    df = c.fetch_options("NIFTY", date(2026, 11, 10), date(2026, 11, 2), date(2026, 11, 3),
                         strike_min=24900, strike_max=25100, log=lambda m: None)
    cand = [b for u, h, b in http.posts if u.endswith(a.CANDLE_PATH)]
    assert {b["symboltoken"] for b in cand} == {"50001", "50002"}
    assert all(b["exchange"] == "NFO" and b["interval"] == "FIVE_MINUTE" for b in cand)
    assert cand[0]["fromdate"] == "2026-11-02 09:15" and cand[0]["todate"] == "2026-11-03 15:30"
    assert len(df) == 2 * 75 and df["timestamp"].min() == pd.Timestamp("2026-11-02 09:15")
    idx = c.index_candles("NIFTY", date(2026, 11, 2), date(2026, 11, 2))
    assert http.posts[-1][2]["symboltoken"] == "99926000" and http.posts[-1][2]["exchange"] == "NSE"
    assert len(idx) == 75 and list(idx.columns)[:2] == ["underlying", "timestamp"]
    msgs = []
    assert c.fetch_options("NIFTY", date(2026, 10, 27), date(2026, 10, 20), date(2026, 10, 27),
                           log=msgs.append).empty
    assert "expired contracts are not served" in msgs[0] and "2026-11-10" in msgs[0]
    p = tmp_path / "angel.csv"
    df.to_csv(p, index=False)
    ds = load_dataset(p, {"NIFTY": 75}, label="ANGELONE fixture")
    assert set(ds.contracts) == {"NIFTY-20261110-25000-CE", "NIFTY-20261110-25000-PE"}
    Backtester(AppConfig(), ds).run()


def test_api_error_is_raised():
    class Bad(FakeAngel):
        def post(self, url, headers=None, data=None, timeout=None):
            return Resp({"status": False, "message": "Invalid Token", "errorcode": "AG8001", "data": None})
    c = a.AngelOneClient(a.Session("KEY", "C", "JWT", "F"), http=Bad(), pause=0)
    with pytest.raises(a.AngelOneError, match="AG8001 Invalid Token"):
        c.candles("NFO", "50001", date(2026, 11, 2), date(2026, 11, 2))


def packet(token, ts_ms, ltp_paise, mode=2, exch=2, vol=0, bids=(), asks=()):
    b = struct.pack("<BB", mode, exch) + token.encode().ljust(25, b"\x00")
    b += struct.pack("<qqq", 1, ts_ms, ltp_paise)
    if mode in (2, 3):
        b += struct.pack("<qqqddqqqq", 75, ltp_paise, vol, 1e5, 2e5, 10000, 12000, 9000, 9500)
    if mode == 3:
        b += struct.pack("<qqq", ts_ms, 123456, 0)
        levels = [(1, q, p) for p, q in bids] + [(0, q, p) for p, q in asks]
        levels += [(0, 0, 0)] * (10 - len(levels))
        b += b"".join(struct.pack("<HqqH", f, q, p, 3) for f, q, p in levels)
        b += struct.pack("<qqqq", 0, 0, 0, 0)
    return b


def test_tick_packet_decoding():
    ts = int(pd.Timestamp("2026-11-03 09:15:07", tz="Asia/Kolkata").timestamp() * 1000)
    d = a.parse_tick_packet(packet("50001", ts, 12345, mode=1))
    assert (d["token"], d["exchange_type"], d["ltp"]) == ("50001", 2, 123.45) and "volume_today" not in d
    q = a.parse_tick_packet(packet("50001", ts, 12345, vol=5000))
    assert q["volume_today"] == 5000 and q["open"] == 100.0 and q["close"] == 95.0
    s = a.parse_tick_packet(packet("50001", ts, 12345, mode=3, vol=5000, bids=[(12340, 75)], asks=[(12350, 150)]))
    assert (s["best_bid"], s["best_ask"], s["oi"]) == (123.40, 123.50, 123456)


def test_feed_maps_tokens_volume_and_streams():
    ts0 = int(pd.Timestamp("2026-11-03 09:15:00", tz="Asia/Kolkata").timestamp() * 1000)
    feed_map = {"NIFTY-20261110-25000-CE": ("NFO", "50001"), "NIFTY-20261110-25000-PE": ("NFO", "50002")}
    msgs = [packet("50001", ts0 + 1000, 10000, vol=1000), packet("99999", ts0 + 1500, 5, vol=1),
            packet("50001", ts0 + 2000, 10050, vol=1300), packet("50002", ts0 + 3000, 8000, vol=10)]
    sent = []

    class FakeWS:
        async def send(self, m):
            sent.append(m)

        def __aiter__(self):
            async def gen():
                for m in msgs:
                    yield m
            return gen()

    class FakeConnect:
        def __init__(self, url, additional_headers=None, ping_interval=None):
            self.headers = additional_headers

        async def __aenter__(self):
            assert self.headers["x-feed-token"] == "F" and self.headers["x-client-code"] == "C"
            return FakeWS()

        async def __aexit__(self, *exc):
            return False

    feed = a.AngelOneFeed(a.Session("KEY", "C", "JWT", "F"), feed_map, connect=FakeConnect)
    feed.subscribe(["NIFTY-20261110-25000-CE", "NIFTY-20261110-25000-PE", "UNKNOWN"])
    sub = json.loads(feed.subscribe_message())
    assert sub["action"] == 1 and sub["params"]["mode"] == 2
    assert sub["params"]["tokenList"] == [{"exchangeType": 2, "tokens": ["50001", "50002"]}]

    async def take(n):
        out = []
        async for t in feed.stream():
            out.append(t)
            if len(out) == n:
                break
        return out

    ticks = asyncio.run(take(3))
    assert json.loads(sent[0]) == sub
    assert [t.key[-2:] for t in ticks] == ["CE", "CE", "PE"]
    assert ticks[0].ts == datetime(2026, 11, 3, 9, 15, 1) and ticks[1].price == 100.5
    assert [t.qty for t in ticks] == [0.0, 300.0, 0.0]          # growth of cumulative day volume
    assert all(t.bid is None and t.ask is None for t in ticks)  # never claims bid/ask-classified flow


# ---- shapes observed on the live API (2026-11-03); values below are synthetic
LIVE_SHAPE_MASTER = [
    {"token": "11111", "symbol": "NIFTY29JUN2730000CE", "name": "NIFTY", "expiry": "29JUN2027",
     "strike": "3000000.000000", "lotsize": "65", "instrumenttype": "OPTIDX", "exch_seg": "NFO",
     "tick_size": "5.000000", "freeze_qty": "1801", "is_cas_enabled": False},
    # the live master names the index "NIFTY" and puts "Nifty 50" in symbol
    {"token": "22222", "symbol": "Nifty 50", "name": "NIFTY", "expiry": "", "strike": "0.000000",
     "lotsize": "1", "instrumenttype": "AMXIDX", "exch_seg": "NSE", "tick_size": "0.000000",
     "freeze_qty": "0", "is_cas_enabled": False},
]


def test_live_master_shape():
    con = a.parse_scrip_master(LIVE_SHAPE_MASTER, "NIFTY")
    assert list(zip(con.strike, con.option_type, con.lot_size)) == [(30000.0, "CE", 65)]
    assert con.iloc[0]["expiry"] == date(2027, 6, 29)
    assert a.index_token(LIVE_SHAPE_MASTER, "NIFTY") == "22222"


class _Resp:
    def __init__(self, status, text):
        self.status_code, self.text = status, text

    def json(self):
        return json.loads(self.text)


class _RateLimitedHttp:
    """Answers like Angel One when the historical API is called too fast: HTTP 403 with
    a plain-text body, then a normal JSON success."""
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def post(self, url, headers=None, data=None, timeout=None):
        self.calls += 1
        if self.calls <= self.fails:
            return _Resp(403, "Access denied because of exceeding access rate")
        return _Resp(200, json.dumps({"status": True, "message": "SUCCESS", "errorcode": "", "data": [
            ["2026-11-02T09:15:00+05:30", 100.0, 101.0, 99.0, 100.5, 0]]}))


def _client(http, retries=6):
    s = a.Session("k", "c", "jwt", "feed")
    return a.AngelOneClient(session=s, http=http, pause=0, scrip_master=[], retries=retries, backoff=0)


def test_rate_limit_403_is_retried():
    http = _RateLimitedHttp(fails=2)
    bars = _client(http).candles("NSE", "22222", date(2026, 11, 2), date(2026, 11, 2))
    assert http.calls == 3 and len(bars) == 1 and bars[0]["close"] == 100.5


def test_rate_limit_gives_up_with_clear_error():
    with pytest.raises(a.AngelOneError, match="rate limit"):
        _client(_RateLimitedHttp(fails=10), retries=2).candles("NSE", "1", date(2026, 11, 2), date(2026, 11, 2))
