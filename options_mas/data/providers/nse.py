"""Free NSE sources (no broker account needed).

* **F&O bhavcopy** (end-of-day, every contract incl. expired ones) through
  ``nselib`` – official daily OHLC/OI/volume per option contract. Use it for
  previous-session H/L/C, the contract universe / lot sizes, and to pick the
  liquid strikes worth downloading intraday.
* **Live option chain** (current snapshot) through ``nselib``'s NSE session
  handling – LTP, best bid/ask, volume, OI and IV for every strike. Feeds the
  contract-selection agent.
* **Chain recorder** – polls the live chain during market hours and stores
  snapshots; :func:`snapshots_to_candles` turns them into premium candles.
  Those candles are *sampled*: their high/low only reflect the snapshots
  taken, so they understate the true range. They are labelled accordingly.

NSE does not offer free historical *intraday* option candles. For 5-minute
history use a broker API (see ``upstox.py``, ``kite.py``, ``dhan.py``).
NSE's website is rate-limited and its terms of use apply – poll gently.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from .common import CHAIN_COLUMNS, OPTION_COLUMNS, to_ist_naive

INDEX_OPTION_CODES = {"IDO", "OPTIDX"}


# ------------------------------------------------------------------ bhavcopy
def normalize_fo_bhavcopy(raw: pd.DataFrame, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Normalise an NSE F&O bhavcopy (new UDiFF *or* legacy layout) to daily
    option candles: ``underlying, expiry, strike, option_type, date, open,
    high, low, close, settle, volume, oi, lot_size, underlying_price``."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" in df.columns:  # UDiFF (NSE format since July 2024, archives back to 2018)
        df = df[df["FinInstrmTp"].astype(str).str.strip().isin(INDEX_OPTION_CODES | {"STO"})]
        out = pd.DataFrame({
            "underlying": df["TckrSymb"].astype(str).str.strip(),
            "expiry": pd.to_datetime(df["XpryDt"]).dt.date,
            "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
            "option_type": df["OptnTp"].astype(str).str.strip(),
            "date": pd.to_datetime(df["TradDt"]).dt.date,
            "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
            "high": pd.to_numeric(df["HghPric"], errors="coerce"),
            "low": pd.to_numeric(df["LwPric"], errors="coerce"),
            "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
            "settle": pd.to_numeric(df.get("SttlmPric"), errors="coerce"),
            "volume": pd.to_numeric(df.get("TtlTradgVol"), errors="coerce"),
            "oi": pd.to_numeric(df.get("OpnIntrst"), errors="coerce"),
            "lot_size": pd.to_numeric(df.get("NewBrdLotQty"), errors="coerce"),
            "underlying_price": pd.to_numeric(df.get("UndrlygPric"), errors="coerce"),
        })
    elif "INSTRUMENT" in df.columns:  # legacy fo<DD><MON><YYYY>bhav.csv
        df = df[df["INSTRUMENT"].astype(str).str.strip().isin({"OPTIDX", "OPTSTK"})]
        out = pd.DataFrame({
            "underlying": df["SYMBOL"].astype(str).str.strip(),
            "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y").dt.date,
            "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
            "option_type": df["OPTION_TYP"].astype(str).str.strip(),
            "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y").dt.date,
            "open": pd.to_numeric(df["OPEN"], errors="coerce"),
            "high": pd.to_numeric(df["HIGH"], errors="coerce"),
            "low": pd.to_numeric(df["LOW"], errors="coerce"),
            "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
            "settle": pd.to_numeric(df.get("SETTLE_PR"), errors="coerce"),
            "volume": pd.to_numeric(df.get("CONTRACTS"), errors="coerce"),
            "oi": pd.to_numeric(df.get("OPEN_INT"), errors="coerce"),
            "lot_size": None,
            "underlying_price": None,
        })
    else:
        raise ValueError("unrecognised bhavcopy layout")
    out = out[out["option_type"].isin(["CE", "PE"])]
    if symbols:
        out = out[out["underlying"].isin({s.upper() for s in symbols})]
    return out.reset_index(drop=True)


def fetch_fo_bhavcopy(trade_date: date, symbols: Optional[Iterable[str]] = ("NIFTY", "BANKNIFTY")) -> pd.DataFrame:
    """Download one day's F&O bhavcopy with ``nselib`` (``pip install nselib``)."""
    from nselib import derivatives  # optional dependency
    raw = derivatives.fno_bhav_copy(trade_date=trade_date.strftime("%d-%m-%Y"))
    return normalize_fo_bhavcopy(raw, symbols)


def fetch_fo_bhavcopy_range(start: date, end: date, symbols=("NIFTY", "BANKNIFTY"),
                            pause: float = 1.0, log: Callable[[str], None] = print) -> pd.DataFrame:
    frames = []
    for d in pd.bdate_range(start, end):
        try:
            frames.append(fetch_fo_bhavcopy(d.date(), symbols))
            log(f"bhavcopy {d.date()}: ok")
        except Exception as exc:  # holidays raise "data not found"
            log(f"bhavcopy {d.date()}: skipped ({type(exc).__name__}: {exc})")
        _time.sleep(pause)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def liquid_strikes(bhav: pd.DataFrame, underlying: str, expiry: date, n: int = 10,
                   on: Optional[date] = None) -> list[float]:
    """The ``n`` strikes with the highest combined CE+PE open interest – a
    sensible list of contracts to download intraday history for."""
    df = bhav[(bhav["underlying"] == underlying) & (bhav["expiry"] == expiry)]
    if on is not None:
        df = df[df["date"] == on]
    elif not df.empty:
        df = df[df["date"] == df["date"].max()]
    agg = df.groupby("strike")["oi"].sum().sort_values(ascending=False)
    return sorted(float(s) for s in agg.head(n).index)


# --------------------------------------------------------------- option chain
def parse_nse_option_chain(payload: dict, underlying: str, fetched_at: Optional[datetime] = None) -> pd.DataFrame:
    """Convert NSE's option-chain JSON into the ``chains`` format used by the
    contract-selection agent (IV converted from % to a fraction)."""
    rec = payload.get("records", payload)
    ts = fetched_at
    if ts is None and rec.get("timestamp"):
        ts = datetime.strptime(rec["timestamp"], "%d-%b-%Y %H:%M:%S")
    ts = ts or datetime.now()
    spot = rec.get("underlyingValue")
    rows = []
    for item in rec.get("data", []):
        exp_s = item.get("expiryDates") or item.get("expiryDate")
        for ot in ("CE", "PE"):
            leg = item.get(ot)
            if not leg:
                continue
            exp = exp_s or leg.get("expiryDate")
            s = leg.get("underlyingValue", spot)
            iv = leg.get("impliedVolatility")
            rows.append({
                "timestamp": pd.Timestamp(ts), "underlying": underlying.upper(), "spot": s,
                "expiry": datetime.strptime(exp, "%d-%b-%Y").date(),
                "strike": float(item.get("strikePrice", leg.get("strikePrice"))), "option_type": ot,
                "ltp": leg.get("lastPrice"), "bid": leg.get("buyPrice1", leg.get("bidprice")),
                "ask": leg.get("sellPrice1", leg.get("askPrice")),
                "volume": leg.get("totalTradedVolume"), "oi": leg.get("openInterest"),
                "iv": (float(iv) / 100.0) if iv else None,
            })
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def fetch_option_chain(symbol: str = "NIFTY", expiry: Optional[date] = None) -> pd.DataFrame:
    """Live NSE option chain via ``nselib`` (``pip install nselib``). With no
    expiry, the nearest listed expiry is used."""
    from nselib.derivatives import derivative_data, get_func  # optional dependency
    if expiry is None:
        exps = derivative_data.expiry_dates_option_index().get(symbol.upper(), [])
        if not exps:
            raise RuntimeError(f"NSE returned no expiries for {symbol}")
        expiry = min(datetime.strptime(e, "%d-%b-%Y").date() for e in exps
                     if datetime.strptime(e, "%d-%b-%Y").date() >= date.today())
    resp = get_func.get_nse_option_chain(symbol.upper(), expiry.strftime("%d-%b-%Y"))
    resp.raise_for_status()
    return parse_nse_option_chain(resp.json(), symbol)


class ChainRecorder:
    """Poll the live NSE chain and append snapshots to ``out_dir/chains.csv``.

    Run it on a machine that can reach nseindia.com during market hours,
    then convert with :func:`snapshots_to_candles`. Keep ``every_seconds``
    at 30 s or more.
    """

    def __init__(self, symbols: Iterable[str], out_dir: str | Path, every_seconds: int = 30,
                 fetch: Callable[[str], pd.DataFrame] = fetch_option_chain,
                 session=(time(9, 15), time(15, 30)), log: Callable[[str], None] = print):
        self.symbols = [s.upper() for s in symbols]
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.every = max(int(every_seconds), 5)
        self.fetch = fetch
        self.session = session
        self.log = log

    def poll_once(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now()
        n = 0
        path = self.out / "chains.csv"
        for sym in self.symbols:
            try:
                df = self.fetch(sym)
            except Exception as exc:
                self.log(f"{now:%H:%M:%S} {sym}: fetch failed ({type(exc).__name__}: {exc})")
                continue
            df["fetched_at"] = now
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            n += len(df)
        return n

    def run(self, until: Optional[datetime] = None) -> None:
        while True:
            now = datetime.now()
            if until and now >= until:
                return
            if self.session[0] <= now.time() < self.session[1] and now.weekday() < 5:
                self.log(f"{now:%H:%M:%S} recorded {self.poll_once(now)} rows")
            elif now.time() >= self.session[1]:
                return
            _time.sleep(self.every)


def snapshots_to_candles(chains: pd.DataFrame, timeframe_min: int = 5) -> pd.DataFrame:
    """Build premium candles from recorded chain snapshots.

    OHLC come from the sampled LTPs (high/low are therefore approximate) and
    volume is the increase of NSE's cumulative ``totalTradedVolume`` within the
    bar. Use ``--label "NSE_CHAIN_SNAPSHOTS (sampled)"`` when backtesting.
    """
    if chains.empty:
        return pd.DataFrame(columns=OPTION_COLUMNS)
    df = chains.copy()
    tcol = "fetched_at" if "fetched_at" in df.columns else "timestamp"
    df["t"] = pd.to_datetime(df[tcol])
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df = df.dropna(subset=["ltp"])
    df = df[df["ltp"] > 0]
    anchor = df["t"].dt.normalize() + pd.Timedelta(hours=9, minutes=15)
    mins = ((df["t"] - anchor).dt.total_seconds() // 60).astype(int)
    df["bar"] = anchor + pd.to_timedelta((mins // timeframe_min) * timeframe_min, unit="m")
    df = df[(df["t"] >= anchor)]
    keys = ["underlying", "expiry", "strike", "option_type"]
    df = df.sort_values(keys + ["t"])
    g = df.groupby(keys + ["bar"], sort=True)
    out = g.agg(open=("ltp", "first"), high=("ltp", "max"), low=("ltp", "min"), close=("ltp", "last"),
                cumvol=("volume", "last"), firstvol=("volume", "first"), oi=("oi", "last")).reset_index()
    out["day"] = out["bar"].dt.date
    prev = out.groupby(keys + ["day"])["cumvol"].shift(1)
    # first bar of the day: growth since the first snapshot; later bars: growth since the previous bar
    out["volume"] = (out["cumvol"] - prev.where(prev.notna(), out["firstvol"])).clip(lower=0)
    out = out.rename(columns={"bar": "timestamp"})
    return out[OPTION_COLUMNS].reset_index(drop=True)
