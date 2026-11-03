"""Groww adapter tests with hand-written fixtures (not market data).

The auth scheme mirrors growwapi 1.5.0 ``GrowwAPI._build_request_data``:
approval keys send sha256(secret + str(epoch_seconds)) with the timestamp.
Payload shapes for expiries / contracts / candles are not documented in the
SDK, so the parsers are tested against every plausible shape.
"""
import hashlib
from datetime import date

import pandas as pd
import pytest

from options_mas.backtest.engine import Backtester
from options_mas.config import AppConfig
from options_mas.data.dataset import load_dataset
from options_mas.data.providers import groww as g


def test_approval_checksum_matches_sdk_scheme():
    p = g.approval_payload("s3cr3t", now=1767325500)
    assert p == {"key_type": "approval", "timestamp": 1767325500,
                 "checksum": hashlib.sha256(b"s3cr3t1767325500").hexdigest()}
    h = g._headers("tok")
    assert h["Authorization"] == "Bearer tok" and h["x-api-version"] == "1.0" and h["x-client-id"] == "growwapi"


def test_groww_symbol_parsing():
    p = g.parse_groww_symbol("NSE-NIFTY-04Nov26-25000-CE")
    assert p == {"underlying": "NIFTY", "expiry": date(2026, 11, 4), "strike": 25000.0, "option_type": "CE",
                 "groww_symbol": "NSE-NIFTY-04Nov26-25000-CE"}
    assert g.parse_groww_symbol("NSE-BANKNIFTY-25Nov26-57500.5-PE")["strike"] == 57500.5
    assert g.parse_groww_symbol("NSE-NIFTY") is None


def test_contract_and_expiry_payload_shapes():
    as_strings = {"contracts": ["NSE-NIFTY-04Nov26-25000-CE", "NSE-NIFTY-04Nov26-25000-PE", "junk"]}
    as_dicts = [{"groww_symbol": "NSE-NIFTY-04Nov26-25100-CE"},
                {"symbol": "", "strike_price": 25200, "instrument_type": "PE"}]
    a = g.parse_contracts(as_strings, "NIFTY", date(2026, 11, 4))
    b = g.parse_contracts(as_dicts, "NIFTY", date(2026, 11, 4))
    assert len(a) == 2 and set(a.option_type) == {"CE", "PE"}
    assert sorted(b.strike) == [25100.0, 25200.0]
    assert g.parse_expiries({"expiries": ["2026-11-11", "2026-11-04"]}) == [date(2026, 11, 4), date(2026, 11, 11)]


def test_candle_payload_shapes_and_timezones():
    iso = g.parse_candles({"candles": [["2026-11-03T09:20:00", 2, 3, 1, 2.5, 10, 99],
                                       ["2026-11-03T09:15:00", 1, 2, 0.5, 1.5, 20, 98]]})
    assert [b["open"] for b in iso] == [1, 2] and iso[0]["oi"] == 98      # sorted oldest first
    epoch_s = g.parse_candles({"candles": [[1767325500, 1, 2, 0.5, 1.5, 20]]})
    epoch_ms = g.parse_candles({"candles": [[1767325500000, 1, 2, 0.5, 1.5, 20]]})
    from options_mas.data.providers.common import to_ist_naive
    assert to_ist_naive(epoch_s[0]["timestamp"]) == to_ist_naive(epoch_ms[0]["timestamp"]) \
        == pd.Timestamp("2026-01-02 09:15")
    d = g.parse_candles([{"timestamp": "2026-11-03T09:15:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5}])
    assert d[0]["volume"] == 0 and d[0]["oi"] is None


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, str(body)

    def json(self):
        return self.body


class FakeGroww:
    def __init__(self):
        self.gets, self.posts = [], []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, headers, json))
        return Resp({"token": "ACCESS"})

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, dict(params)))
        assert headers["Authorization"] == "Bearer ACCESS"
        if url.endswith("/historical/contracts"):
            return Resp({"status": "SUCCESS", "payload": {"contracts": [
                "NSE-NIFTY-04Nov26-25000-CE", "NSE-NIFTY-04Nov26-25000-PE", "NSE-NIFTY-04Nov26-27000-CE"]}})
        if url.endswith("/historical/candles"):
            day = params["start_time"][:10]
            rows = [[f"{day}T{9 + (15 + 5 * i) // 60:02d}:{(15 + 5 * i) % 60:02d}:00", 100 + i, 101 + i, 99 + i,
                     100.5 + i, 1000, 5000] for i in range(75)]
            return Resp({"status": "SUCCESS", "payload": {"candles": rows}})
        return Resp({"status": "FAILURE", "error": {"code": "GA004", "message": "not found"}}, 404)


def test_token_from_env_approval_flow(monkeypatch):
    for k in ("GROWW_ACCESS_TOKEN", "GROWW_TOTP", "GROWW_TOTP_SECRET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROWW_API_KEY", "KEY")
    monkeypatch.setenv("GROWW_API_SECRET", "SECRET")
    http = FakeGroww()
    assert g.token_from_env(http) == "ACCESS"
    url, headers, body = http.posts[0]
    assert url == "https://api.groww.in/v1/token/api/access" and headers["Authorization"] == "Bearer KEY"
    assert body["key_type"] == "approval" and body["checksum"] == hashlib.sha256(
        ("SECRET" + str(body["timestamp"])).encode()).hexdigest()
    monkeypatch.delenv("GROWW_API_SECRET")
    with pytest.raises(g.GrowwError):
        g.token_from_env(http)


def test_fetch_expired_contracts_and_backtest(tmp_path):
    http = FakeGroww()
    c = g.GrowwClient("ACCESS", session=http, pause=0)
    df = c.fetch_options("NIFTY", date(2026, 11, 4), date(2026, 11, 2), date(2026, 11, 3),
                         strike_min=24900, strike_max=25100, log=lambda m: None)
    cand = [p for u, p in http.gets if u.endswith("/historical/candles")]
    assert {p["groww_symbol"] for p in cand} == {"NSE-NIFTY-04Nov26-25000-CE", "NSE-NIFTY-04Nov26-25000-PE"}
    assert all(p["segment"] == "FNO" and p["candle_interval"] == "5minute" and p["exchange"] == "NSE" for p in cand)
    assert cand[0]["start_time"] == "2026-11-02 09:15:00" and cand[0]["end_time"] == "2026-11-03 15:30:00"
    assert len(cand) == 2 and len(df) == 2 * 75      # one 7-day chunk per contract; fixture returns one day
    p = tmp_path / "groww.csv"
    df.to_csv(p, index=False)
    ds = load_dataset(p, {"NIFTY": 65}, label="GROWW fixture")
    assert set(ds.contracts) == {"NIFTY-20261104-25000-CE", "NIFTY-20261104-25000-PE"}
    Backtester(AppConfig(), ds).run()


def test_api_failure_is_raised():
    c = g.GrowwClient("ACCESS", session=FakeGroww(), pause=0)
    with pytest.raises(g.GrowwError, match="GA004"):
        c.expiries("NIFTY", 2026)


def test_forbidden_data_endpoint_explains_entitlement():
    # shape of Groww's live 403 body (token valid, data endpoints not entitled)
    class Forbidden(FakeGroww):
        def get(self, url, params=None, headers=None, timeout=None):
            return Resp({"status": "FAILURE", "error": {"code": "403", "message": "Access forbidden for this request.",
                                                        "metadata": None}}, 403)
    c = g.GrowwClient("ACCESS", session=Forbidden(), pause=0)
    with pytest.raises(g.GrowwError, match="Access forbidden.*not entitled"):
        c.expiries("NIFTY", 2026, 10)
