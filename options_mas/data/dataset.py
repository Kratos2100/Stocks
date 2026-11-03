"""Historical dataset container and loaders.

Long-format CSV / Parquet for option candles (one row per contract per bar)::

    underlying,expiry,strike,option_type,timestamp,open,high,low,close,volume
    [,buy_volume,sell_volume,delta_source][,oi]

* ``timestamp`` = bar START time in IST.
* ``delta_source`` must be one of :class:`~options_mas.models.DeltaSource`.
  If buy/sell volume columns are present without a source column, the
  ``default_delta_source`` argument is used – it must be set explicitly so the
  provenance of order-flow data is never assumed.

Optional option-chain snapshots (for contract selection)::

    timestamp,underlying,spot,expiry,strike,option_type,ltp,bid,ask,volume,oi[,iv,delta,theta]

Optional underlying index candles (moneyness + regime analytics only, NEVER
entry confirmation)::

    underlying,timestamp,open,high,low,close,volume
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..models import Candle, DeltaSource, OptionContract, OptionType


@dataclass
class BacktestDataset:
    candles: dict[str, list[Candle]]
    contracts: dict[str, OptionContract]
    subbars: dict[str, list[Candle]] = field(default_factory=dict)
    underlying: dict[str, list[Candle]] = field(default_factory=dict)
    chains: Optional[pd.DataFrame] = None
    label: str = "UNSPECIFIED"
    _by_date: dict[str, dict[date, list[Candle]]] = field(default_factory=dict, repr=False)
    _sub_ts: dict[str, list[datetime]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for k in self.candles:
            self.candles[k].sort(key=lambda c: c.ts)
        for k in self.subbars:
            self.subbars[k].sort(key=lambda c: c.ts)
            self._sub_ts[k] = [c.ts for c in self.subbars[k]]

    def sessions(self) -> list[date]:
        days: set[date] = set()
        for lst in self.candles.values():
            days.update(c.ts.date() for c in lst)
        return sorted(days)

    def by_date(self, key: str) -> dict[date, list[Candle]]:
        if key not in self._by_date:
            d: dict[date, list[Candle]] = {}
            for c in self.candles.get(key, []):
                d.setdefault(c.ts.date(), []).append(c)
            self._by_date[key] = d
        return self._by_date[key]

    def warmup(self, key: str, day: date, sessions: int) -> list[Candle]:
        """Candles of *previous* sessions only (strictly before ``day``)."""
        bd = self.by_date(key)
        prev = sorted(d for d in bd if d < day)[-sessions:] if sessions > 0 else []
        out: list[Candle] = []
        for d in prev:
            out.extend(bd[d])
        return out

    def subbars_for(self, key: str, candle: Candle) -> list[Candle]:
        ts = self._sub_ts.get(key)
        if not ts:
            return []
        lo = bisect.bisect_left(ts, candle.ts)
        hi = bisect.bisect_left(ts, candle.end_ts)
        return self.subbars[key][lo:hi]

    def contracts_on(self, day: date) -> list[str]:
        return sorted(k for k in self.candles if day in self.by_date(k))

    def subset(self, days: list[date]) -> "BacktestDataset":
        """Dataset restricted to ``days`` (keeps earlier history for warm-up)."""
        last = max(days)
        cut = {k: [c for c in v if c.ts.date() <= last] for k, v in self.candles.items()}
        return BacktestDataset(cut, self.contracts, self.subbars, self.underlying, self.chains, self.label)


def _row_candle(r, tf: int, default_src: Optional[DeltaSource]) -> Candle:
    bv = getattr(r, "buy_volume", None)
    sv = getattr(r, "sell_volume", None)
    bv = None if bv is None or (isinstance(bv, float) and math.isnan(bv)) else float(bv)
    sv = None if sv is None or (isinstance(sv, float) and math.isnan(sv)) else float(sv)
    src = getattr(r, "delta_source", None)
    if bv is not None and sv is not None:
        if isinstance(src, str) and src:
            source = DeltaSource(src)
        elif default_src is not None:
            source = default_src
        else:
            raise ValueError("buy/sell volume present but no delta_source column and no default_delta_source given")
    else:
        bv = sv = None
        source = DeltaSource.NONE
    oi = getattr(r, "oi", None)
    return Candle(
        ts=pd.Timestamp(r.timestamp).to_pydatetime().replace(tzinfo=None),
        open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close),
        volume=float(getattr(r, "volume", 0.0) or 0.0), buy_volume=bv, sell_volume=sv, delta_source=source,
        oi=None if oi is None or (isinstance(oi, float) and math.isnan(oi)) else float(oi), timeframe_min=tf,
    )


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["timestamp"] = ts
    return df


def candles_from_frame(df: pd.DataFrame, lot_sizes: dict[str, int], tf: int = 5,
                       default_delta_source: Optional[DeltaSource] = None) -> tuple[dict, dict]:
    candles: dict[str, list[Candle]] = {}
    contracts: dict[str, OptionContract] = {}
    df = df.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    for (und, exp, strike, ot), g in df.groupby(["underlying", "expiry", "strike", "option_type"], sort=True):
        c = OptionContract(und, exp, float(strike), OptionType(ot), lot_size=lot_sizes.get(und, 1))
        contracts[c.key] = c
        candles[c.key] = [_row_candle(r, tf, default_delta_source) for r in g.sort_values("timestamp").itertuples()]
    return candles, contracts


def load_dataset(options_path: str | Path, lot_sizes: dict[str, int], *, timeframe_min: int = 5,
                 subbars_path: str | Path | None = None, underlying_path: str | Path | None = None,
                 chains_path: str | Path | None = None, label: str = "UNSPECIFIED",
                 default_delta_source: Optional[DeltaSource] = None) -> BacktestDataset:
    candles, contracts = candles_from_frame(_read(options_path), lot_sizes, timeframe_min, default_delta_source)
    sub: dict[str, list[Candle]] = {}
    if subbars_path:
        sub, _ = candles_from_frame(_read(subbars_path), lot_sizes, 1, default_delta_source)
    und: dict[str, list[Candle]] = {}
    if underlying_path:
        udf = _read(underlying_path)
        for name, g in udf.groupby("underlying"):
            und[name] = [_row_candle(r, timeframe_min, None) for r in g.sort_values("timestamp").itertuples()]
    chains = None
    if chains_path:
        chains = _read(chains_path)
        chains["expiry"] = pd.to_datetime(chains["expiry"]).dt.date
    return BacktestDataset(candles, contracts, sub, und, chains, label)


def dataset_to_frame(ds: BacktestDataset) -> pd.DataFrame:
    rows = []
    for key, lst in ds.candles.items():
        c = ds.contracts[key]
        for b in lst:
            rows.append({"underlying": c.underlying, "expiry": c.expiry, "strike": c.strike,
                         "option_type": c.option_type.value, "timestamp": b.ts, "open": b.open, "high": b.high,
                         "low": b.low, "close": b.close, "volume": b.volume, "buy_volume": b.buy_volume,
                         "sell_volume": b.sell_volume, "delta_source": b.delta_source.value, "oi": b.oi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Market regime classification (analytics / Mode C ranking only)
# --------------------------------------------------------------------------

def classify_regimes(underlying: dict[str, list[Candle]], vol_lookback: int = 20) -> dict[date, str]:
    """Label each session TRENDING/RANGING x HIGH_VOL/LOW_VOL from the first
    underlying series. The volatility threshold uses only *previous* sessions."""
    if not underlying:
        return {}
    name = sorted(underlying)[0]
    days: dict[date, list[Candle]] = {}
    for c in underlying[name]:
        days.setdefault(c.ts.date(), []).append(c)
    out: dict[date, str] = {}
    hist: list[float] = []
    for d in sorted(days):
        bars = days[d]
        o, h, l, cl = bars[0].open, max(b.high for b in bars), min(b.low for b in bars), bars[-1].close
        trend = abs(cl - o) / (h - l) if h > l else 0.0
        closes = np.array([b.close for b in bars], dtype=float)
        rets = np.diff(np.log(closes)) if len(closes) > 1 else np.array([0.0])
        vol = float(np.std(rets))
        if len(hist) >= 5:
            med = float(np.median(hist[-vol_lookback:]))
            vlabel = "HIGH_VOL" if vol > med else "LOW_VOL"
        else:
            vlabel = "NORMAL_VOL"
        out[d] = ("TRENDING" if trend >= 0.5 else "RANGING") + "_" + vlabel
        hist.append(vol)
    return out


def previous_session_regimes(regimes: dict[date, str]) -> dict[date, str]:
    days = sorted(regimes)
    return {d: regimes[days[i - 1]] for i, d in enumerate(days) if i > 0}
