"""Deterministic SYNTHETIC dataset generator for demos and tests.

Everything produced here is fake and labelled ``SYNTHETIC``. The buy/sell
volume split uses ``DeltaSource.SYNTHETIC`` – it is neither bid/ask-classified
nor an estimate of real order flow. Never quote performance obtained on this
data as real performance.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..models import Candle, DeltaSource, OptionContract, OptionType
from .dataset import BacktestDataset

_SQRT2 = math.sqrt(2.0)


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7)
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _ncdf(x):
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _bs(S, K, T, r, sig, is_call):
    T = np.maximum(T, 1e-7)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    call = S * _ncdf(d1) - K * np.exp(-r * T) * _ncdf(d2)
    put = K * np.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)
    price = np.where(is_call, call, put)
    delta = np.where(is_call, _ncdf(d1), _ncdf(d1) - 1.0)
    pdf = np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    theta = (-S * pdf * sig / (2 * np.sqrt(T))) / 365.0
    return price, delta, theta


UNDERLYINGS = {
    "NIFTY": {"spot": 24500.0, "vol": 0.14, "step": 50, "lot": 65, "weekly": True},
    "BANKNIFTY": {"spot": 52000.0, "vol": 0.17, "step": 100, "lot": 30, "weekly": False},
}


def _business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _expiries(days: list[date], weekly: bool) -> list[date]:
    """Tuesday expiries (weekly) or last Tuesday of month (monthly)."""
    first, last = days[0], days[-1] + timedelta(days=40)
    out, d = [], first
    while d <= last:
        if d.weekday() == 1:
            if weekly:
                out.append(d)
            else:
                nxt = d + timedelta(days=7)
                if nxt.month != d.month:
                    out.append(d)
        d += timedelta(days=1)
    return out


def _tick(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.round(x / 0.05) * 0.05, 0.05)


def generate_synthetic_dataset(sessions: int = 15, start: date = date(2026, 6, 1), seed: int = 7,
                               underlyings: tuple[str, ...] = ("NIFTY", "BANKNIFTY"), strikes_each_side: int = 6,
                               timeframe_min: int = 5, with_subbars: bool = False,
                               listing_days: int = 12) -> BacktestDataset:
    rng = np.random.default_rng(seed)
    days = _business_days(start, sessions)
    minutes = 375  # 09:15 -> 15:30
    candles: dict[str, list[Candle]] = {}
    subbars: dict[str, list[Candle]] = {}
    contracts: dict[str, OptionContract] = {}
    und_candles: dict[str, list[Candle]] = {}
    chain_rows = []
    r = 0.065

    for und in underlyings:
        p = UNDERLYINGS[und]
        step = p["step"]
        sig_ann = p["vol"]
        dt = 1.0 / (252 * minutes)
        spot = p["spot"]
        # --- spot path for all sessions (1-minute), with daily regime drift
        paths = []
        for d in days:
            drift = rng.normal(0, 0.6) * sig_ann  # annualised daily-regime drift
            vol = sig_ann * math.exp(rng.normal(0, 0.25))
            z = rng.standard_normal(minutes)
            gap = rng.normal(0, 0.003)
            s0 = spot * math.exp(gap)
            logret = (drift - 0.5 * vol ** 2) * dt + vol * math.sqrt(dt) * z
            path = s0 * np.exp(np.cumsum(logret))
            path = np.concatenate([[s0], path])  # 376 points: minute boundaries
            paths.append(path)
            spot = float(path[-1])
        lo = min(float(x.min()) for x in paths)
        hi = max(float(x.max()) for x in paths)
        strikes = np.arange((math.floor(lo / step) - strikes_each_side) * step,
                            (math.ceil(hi / step) + strikes_each_side + 1) * step, step, dtype=float)
        exps = _expiries(days, p["weekly"])
        und_candles[und] = []
        base_iv = sig_ann * 1.05
        iv_state = 0.0
        for di, d in enumerate(days):
            path = paths[di]
            ts0 = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15)
            # underlying candles (analytics only)
            for b in range(0, minutes, timeframe_min):
                seg = path[b:b + timeframe_min + 1]
                und_candles[und].append(Candle(ts0 + timedelta(minutes=b), float(seg[0]), float(seg.max()),
                                               float(seg.min()), float(seg[-1]), 0.0, timeframe_min=timeframe_min))
            iv_state = 0.9 * iv_state + rng.normal(0, 0.02)
            live = [e for e in exps if e >= d and (e - d).days <= listing_days]
            for e in live:
                exp_dt = datetime.combine(e, datetime.min.time()).replace(hour=15, minute=30)
                tgrid = np.array([(exp_dt - (ts0 + timedelta(minutes=m))).total_seconds() / (365 * 86400)
                                  for m in range(minutes + 1)])
                for ot in (OptionType.CE, OptionType.PE):
                    is_call = ot == OptionType.CE
                    K = strikes[:, None]
                    S = path[None, :]
                    mny = np.log(K / S)
                    iv = np.maximum(base_iv + iv_state + 0.35 * mny ** 2 * 100 / 10 + (0.02 if not is_call else 0.0)
                                    + rng.normal(0, 0.004, size=(len(strikes), minutes + 1)).cumsum(axis=1) * 0.2, 0.05)
                    price, delta, theta = _bs(S, K, tgrid[None, :], r, iv, is_call)
                    noise = rng.normal(0, 0.15, size=price.shape)
                    price = _tick(price + noise)
                    absm = np.abs(mny[:, 0]) * 100
                    for si, k in enumerate(strikes):
                        c = OptionContract(und, e, float(k), ot, lot_size=p["lot"])
                        key = c.key
                        contracts[key] = c
                        pr = price[si]
                        if pr[0] < 0.5 and pr.max() < 1.0:
                            continue
                        base_v = 40000 * math.exp(-absm[si] * 1.2) / max(p["lot"] / 65, 1)
                        u_shape = 1.0 + 1.5 * np.exp(-np.arange(minutes) / 30.0) + 0.8 * np.exp(-(minutes - np.arange(minutes)) / 30.0)
                        mvol = np.floor(base_v / minutes * u_shape * rng.lognormal(0, 0.6, minutes) / p["lot"]) * p["lot"]
                        mret = np.diff(pr)
                        buy_frac = np.clip(0.5 + 0.3 * np.tanh(mret / (np.std(mret) + 1e-9)) + rng.normal(0, 0.18, minutes), 0.02, 0.98)
                        lst = candles.setdefault(key, [])
                        sub = subbars.setdefault(key, []) if with_subbars else None
                        for b in range(0, minutes, timeframe_min):
                            seg = pr[b:b + timeframe_min + 1]
                            v = float(mvol[b:b + timeframe_min].sum())
                            bv = float((mvol[b:b + timeframe_min] * buy_frac[b:b + timeframe_min]).sum())
                            o = float(seg[0])
                            h = float(max(seg.max(), o))
                            l = float(min(seg.min(), o))
                            lst.append(Candle(ts0 + timedelta(minutes=b), o, h, l, float(seg[-1]), v,
                                              round(bv), round(v - bv), DeltaSource.SYNTHETIC, timeframe_min=timeframe_min))
                            if sub is not None:
                                for m in range(b, b + timeframe_min):
                                    s2 = pr[m:m + 2]
                                    vv = float(mvol[m])
                                    sub.append(Candle(ts0 + timedelta(minutes=m), float(s2[0]), float(s2.max()),
                                                      float(s2.min()), float(s2[-1]), vv, round(vv * buy_frac[m]),
                                                      round(vv - vv * buy_frac[m]), DeltaSource.SYNTHETIC, timeframe_min=1))
                        # chain snapshot at 09:20 (bar index 5)
                        m5 = 5
                        mid = float(pr[m5])
                        spr = max(0.05, round(mid * 0.004 / 0.05) * 0.05)
                        chain_rows.append({
                            "timestamp": ts0 + timedelta(minutes=m5), "underlying": und, "spot": float(path[m5]),
                            "expiry": e, "strike": float(k), "option_type": ot.value, "ltp": mid,
                            "bid": round(mid - spr / 2, 2), "ask": round(mid + spr / 2, 2),
                            "volume": float(mvol[:m5].sum()) * 20, "oi": float(base_v * 30),
                            "iv": float(iv[si, m5]), "delta": float(delta[si, m5]), "theta": float(theta[si, m5]),
                        })
    chains = pd.DataFrame(chain_rows)
    candles = {k: v for k, v in candles.items() if v}
    subbars = {k: v for k, v in subbars.items() if v}
    contracts = {k: v for k, v in contracts.items() if k in candles}
    return BacktestDataset(candles, contracts, subbars, und_candles, chains,
                           label=f"SYNTHETIC (generated, seed={seed}) – NOT REAL MARKET DATA")
