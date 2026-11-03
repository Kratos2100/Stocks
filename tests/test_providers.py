"""Parsing tests for the data providers.

The payloads below are small hand-written fixtures shaped like each API's
documented response. They exercise the conversion code only; they are not
market data and they do not prove the live endpoints behave identically.
"""
from datetime import date, datetime

import pandas as pd
import pytest

from options_mas.backtest.engine import Backtester
from options_mas.config import AppConfig
from options_mas.data.dataset import load_dataset
from options_mas.data.providers.common import date_chunks, to_ist_naive
from options_mas.data.providers.dhan import parse_arrays, parse_scrip_master, split_rolling_by_strike
from options_mas.data.providers.nse import (ChainRecorder, liquid_strikes, normalize_fo_bhavcopy,
                                            parse_nse_option_chain, snapshots_to_candles)
from options_mas.data.providers.upstox import UpstoxClient, parse_candles


def test_date_chunks_and_timezones():
    assert list(date_chunks(date(2026, 1, 1), date(2026, 1, 10), 4)) == [
        (date(2026, 1, 1), date(2026, 1, 4)), (date(2026, 1, 5), date(2026, 1, 8)), (date(2026, 1, 9), date(2026, 1, 10))]
    assert to_ist_naive("2026-01-02T09:15:00+05:30") == pd.Timestamp("2026-01-02 09:15")
    assert to_ist_naive(1767325500) == pd.Timestamp("2026-01-02 09:15")  # epoch seconds (UTC) -> IST


def test_bhavcopy_udiff_and_legacy():
    udiff = pd.DataFrame([
        {"TradDt": "2026-10-30", "FinInstrmTp": "IDO", "TckrSymb": "NIFTY", "XpryDt": "2026-11-04", "StrkPric": 25000,
         "OptnTp": "CE", "OpnPric": 100, "HghPric": 120, "LwPric": 90, "ClsPric": 110, "SttlmPric": 110,
         "TtlTradgVol": 5000, "OpnIntrst": 90000, "NewBrdLotQty": 65, "UndrlygPric": 25010},
        {"TradDt": "2026-10-30", "FinInstrmTp": "IDF", "TckrSymb": "NIFTY", "XpryDt": "2026-11-25", "StrkPric": None,
         "OptnTp": None, "OpnPric": 1, "HghPric": 1, "LwPric": 1, "ClsPric": 1},
    ])
    out = normalize_fo_bhavcopy(udiff)
    assert len(out) == 1 and out.iloc[0]["high"] == 120 and out.iloc[0]["lot_size"] == 65
    legacy = pd.DataFrame([{"INSTRUMENT": "OPTIDX", "SYMBOL": "BANKNIFTY", "EXPIRY_DT": "27-Jun-2024",
                            "STRIKE_PR": 50000, "OPTION_TYP": "PE", "OPEN": 1, "HIGH": 2, "LOW": 0.5, "CLOSE": 1.5,
                            "SETTLE_PR": 1.5, "CONTRACTS": 10, "OPEN_INT": 100, "TIMESTAMP": "20-Jun-2024"}])
    out2 = normalize_fo_bhavcopy(legacy)
    assert out2.iloc[0]["expiry"] == date(2024, 6, 27) and out2.iloc[0]["option_type"] == "PE"


def test_liquid_strikes_by_open_interest():
    bhav = pd.DataFrame({"underlying": "NIFTY", "expiry": date(2026, 11, 4), "date": date(2026, 10, 30),
                         "strike": [24900, 25000, 25000, 25100, 25200], "option_type": ["CE", "CE", "PE", "CE", "CE"],
                         "oi": [10, 50, 60, 70, 5]})
    assert liquid_strikes(bhav, "NIFTY", date(2026, 11, 4), 2) == [25000.0, 25100.0]


NSE_CHAIN = {"records": {"timestamp": "30-Oct-2026 10:00:05", "underlyingValue": 25010.5, "data": [
    {"strikePrice": 25000, "expiryDates": "04-Nov-2026",
     "CE": {"lastPrice": 120.5, "buyPrice1": 120.4, "sellPrice1": 120.6, "totalTradedVolume": 1000,
            "openInterest": 5000, "impliedVolatility": 12.5},
     "PE": {"lastPrice": 98.0, "buyPrice1": 97.9, "sellPrice1": 98.1, "totalTradedVolume": 800,
            "openInterest": 4000, "impliedVolatility": 13.0}}]}}


def test_parse_nse_option_chain():
    df = parse_nse_option_chain(NSE_CHAIN, "NIFTY")
    assert len(df) == 2
    ce = df[df.option_type == "CE"].iloc[0]
    assert ce.bid == 120.4 and ce.ask == 120.6 and ce.iv == pytest.approx(0.125) and ce.spot == 25010.5
    assert ce.expiry == date(2026, 11, 4) and ce.timestamp == pd.Timestamp("2026-10-30 10:00:05")


def test_chain_recorder_and_snapshot_candles(tmp_path):
    ticks = iter([(120.0, 1000), (125.0, 1300), (118.0, 1500), (119.0, 1600)])

    def fake_fetch(sym):
        ltp, vol = next(ticks)
        p = {"records": {**NSE_CHAIN["records"], "data": [{**NSE_CHAIN["records"]["data"][0],
             "CE": {**NSE_CHAIN["records"]["data"][0]["CE"], "lastPrice": ltp, "totalTradedVolume": vol}}]}}
        return parse_nse_option_chain(p, sym)
    rec = ChainRecorder(["NIFTY"], tmp_path, 30, fetch=fake_fetch, log=lambda m: None)
    for t in ("10:00:05", "10:02:00", "10:04:30", "10:05:10"):
        rec.poll_once(datetime.fromisoformat(f"2026-10-30T{t}"))
    candles = snapshots_to_candles(pd.read_csv(tmp_path / "chains.csv"))
    ce = candles[candles.option_type == "CE"].reset_index(drop=True)
    assert list(ce.timestamp.astype(str)) == ["2026-10-30 10:00:00", "2026-10-30 10:05:00"]
    assert (ce.loc[0, ["open", "high", "low", "close"]].tolist()) == [120.0, 125.0, 118.0, 118.0]
    assert ce.loc[0, "volume"] == 500 and ce.loc[1, "volume"] == 100   # increments of cumulative volume


class FakeResp:
    def __init__(self, body, status=200):
        self.body, self.status_code, self.text = body, status, str(body)

    def json(self):
        return self.body


class FakeUpstox:
    """Routes the documented Upstox paths to tiny fixtures."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        assert headers["Authorization"] == "Bearer TEST"
        if url.endswith("/v2/expired-instruments/option/contract"):
            return FakeResp({"status": "success", "data": [
                {"instrument_key": "NSE_FO|1|04-11-2026", "expiry": "2026-11-04", "strike_price": 25000.0,
                 "instrument_type": "CE", "lot_size": 65, "trading_symbol": "NIFTY 25000 CE"},
                {"instrument_key": "NSE_FO|2|04-11-2026", "expiry": "2026-11-04", "strike_price": 26000.0,
                 "instrument_type": "CE", "lot_size": 65, "trading_symbol": "NIFTY 26000 CE"}]})
        if "/historical-candle/" in url:
            day = "2026-11-03"
            rows = [[f"{day}T{9 + (15 + 5 * i) // 60:02d}:{(15 + 5 * i) % 60:02d}:00+05:30", 100 + i, 101 + i, 99 + i,
                     100.5 + i, 1000, 5000] for i in range(75)][::-1]  # newest first, like the API
            rows.append([f"{day}T15:30:00+05:30", 1, 1, 1, 1, 0, 0])      # outside session -> dropped
            return FakeResp({"status": "success", "data": {"candles": rows}})
        return FakeResp({"status": "error", "errors": ["unexpected"]}, 404)


def test_upstox_expired_contract_download_and_backtest(tmp_path):
    http = FakeUpstox()
    client = UpstoxClient("TEST", session=http, pause=0)
    df = client.fetch_options("NIFTY", date(2026, 11, 4), date(2026, 11, 3), date(2026, 11, 3),
                              strike_min=24900, strike_max=25100, expired=True, log=lambda m: None)
    assert any("/v2/expired-instruments/historical-candle/NSE_FO|1|04-11-2026/5minute/" in c for c in http.calls)
    assert len(df) == 75 and set(df.strike) == {25000.0}
    assert df.timestamp.is_monotonic_increasing and df.timestamp.iloc[0] == pd.Timestamp("2026-11-03 09:15")
    p = tmp_path / "opt.csv"
    df.to_csv(p, index=False)
    ds = load_dataset(p, {"NIFTY": 65}, label="UPSTOX fixture")
    assert len(ds.candles) == 1
    Backtester(AppConfig(), ds).run()  # the downloaded format feeds the backtester directly


def test_upstox_active_path_and_errors():
    http = FakeUpstox()
    client = UpstoxClient("TEST", session=http, pause=0)
    bars = client.candles("NSE_FO|1", date(2026, 11, 3), date(2026, 11, 3))
    assert "/v3/historical-candle/NSE_FO|1/minutes/5/2026-11-03/2026-11-03" in http.calls[0]
    assert bars[0]["timestamp"] < bars[-1]["timestamp"]
    from options_mas.data.providers.upstox import UpstoxError
    with pytest.raises(UpstoxError):
        client.option_contracts("NIFTY", date(2026, 11, 4), expired=False)
    assert parse_candles([["2026-11-03T09:15:00+05:30", 1, 2, 0.5, 1.5, 10]])[0]["oi"] is None


def test_dhan_parsers():
    resp = {"status": "success", "remarks": "", "data": {
        "open": [10, 11], "high": [12, 13], "low": [9, 10], "close": [11, 12], "volume": [100, 200],
        "timestamp": [1767325500, 1767325800], "open_interest": [5, 6]}}
    bars = parse_arrays(resp)
    assert len(bars) == 2 and bars[1]["oi"] == 6
    with pytest.raises(RuntimeError):
        parse_arrays({"status": "failure", "remarks": "bad token"})
    rolling = {"data": {"ce": {"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3],
                               "volume": [1, 1, 1], "timestamp": [1767325500, 1767325800, 1767326100],
                               "strike": [25000, 25000, 25050]}}}
    df = split_rolling_by_strike(rolling, "NIFTY", date(2026, 1, 6), "CE")
    assert df.groupby("strike").size().to_dict() == {25000.0: 2, 25050.0: 1}
    master = pd.DataFrame([{"SEM_EXM_EXCH_ID": "NSE", "SEM_INSTRUMENT_NAME": "OPTIDX",
                            "SEM_TRADING_SYMBOL": "NIFTY-Nov2026-25000-CE", "SEM_EXPIRY_DATE": "2026-11-04 14:30:00",
                            "SEM_SMST_SECURITY_ID": 12345, "SEM_STRIKE_PRICE": 25000.0, "SEM_OPTION_TYPE": "CE",
                            "SEM_LOT_UNITS": 65}])
    con = parse_scrip_master(master, "NIFTY", date(2026, 11, 4))
    assert con.iloc[0]["security_id"] == "12345"
