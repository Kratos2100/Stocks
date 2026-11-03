"""Options contract selection agent.

Chooses which CE and which PE premium charts are monitored – *independently*
(they may have different strikes). It evaluates expiry, strike, moneyness,
premium, volume, OI, bid-ask spread, IV, delta, theta and liquidity.

It never generates a directional view and never adds an entry condition. The
underlying spot price is used solely to classify moneyness (ATM/ITM/OTM).
It never simply picks the cheapest option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from .config import ContractSelectionConfig
from .models import OptionContract, OptionType


# ------------------------------------------------------------------ Black-Scholes helpers
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, ot: OptionType) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if ot == OptionType.CE else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if ot == OptionType.CE:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, ot: OptionType) -> dict[str, float]:
    T = max(T, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if ot == OptionType.CE:
        delta = _ncdf(d1)
        theta = -S * _npdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _ncdf(d2)
    else:
        delta = _ncdf(d1) - 1
        theta = -S * _npdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _ncdf(-d2)
    return {"delta": delta, "theta_per_day": theta / 365.0}


def implied_vol(price: float, S: float, K: float, T: float, r: float, ot: OptionType) -> Optional[float]:
    intrinsic = max(0.0, S - K) if ot == OptionType.CE else max(0.0, K - S)
    if price <= intrinsic or T <= 0:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, r, mid, ot) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@dataclass
class Selection:
    contract: Optional[OptionContract]
    option_type: OptionType
    score: Optional[float]
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"contract": self.contract.to_dict() if self.contract else None, "option_type": self.option_type.value,
                "score": self.score, "metrics": self.metrics, "reasons": self.reasons, "rejected": self.rejected}


class ContractSelector:
    def __init__(self, cfg: ContractSelectionConfig, lot_sizes: dict[str, int], strike_steps: dict[str, float]):
        self.cfg = cfg
        self.lot_sizes = lot_sizes
        self.strike_steps = strike_steps

    def _pick_expiry(self, expiries: list[date], today: date) -> Optional[date]:
        c = self.cfg
        valid = sorted(e for e in expiries if (e - today).days >= c.min_days_to_expiry)
        if c.avoid_expiry_day:
            valid = [e for e in valid if e != today]
        if not valid:
            return None
        return valid[1] if (c.expiry_rule == "next" and len(valid) > 1) else valid[0]

    def select(self, chain: pd.DataFrame, underlying: str, now: datetime) -> dict[OptionType, Selection]:
        """``chain`` = one snapshot for one underlying (see data/dataset.py)."""
        c = self.cfg
        snap = chain[chain["underlying"] == underlying]
        if snap.empty:
            return {ot: Selection(None, ot, None, reasons=["empty chain"]) for ot in OptionType}
        spot = float(snap["spot"].iloc[0])
        expiry = self._pick_expiry(sorted(set(snap["expiry"])), now.date())
        if expiry is None:
            return {ot: Selection(None, ot, None, reasons=["no valid expiry"]) for ot in OptionType}
        snap = snap[snap["expiry"] == expiry]
        step = self.strike_steps.get(underlying, 50)
        atm = round(spot / step) * step
        T = max(((datetime.combine(expiry, datetime.min.time()).replace(hour=15, minute=30) - now).total_seconds())
                / (365 * 24 * 3600), 1e-6)
        out = {}
        for ot in OptionType:
            out[ot] = self._select_side(snap[snap["option_type"] == ot.value], ot, underlying, expiry, spot, atm, step, T)
        return out

    def _select_side(self, df, ot, und, expiry, spot, atm, step, T) -> Selection:
        c = self.cfg
        # preferred strike by moneyness preference
        sign = 1 if ot == OptionType.CE else -1   # OTM direction for CE is higher strikes
        pref = atm
        if c.moneyness == "OTM":
            pref = atm + sign * c.moneyness_steps * step
        elif c.moneyness == "ITM":
            pref = atm - sign * c.moneyness_steps * step
        lo, hi = pref - c.candidate_strikes_each_side * step, pref + c.candidate_strikes_each_side * step
        cands, rejected = [], []
        for r in df.itertuples():
            K = float(r.strike)
            if not (lo <= K <= hi):
                continue
            why = []
            ltp = float(r.ltp)
            bid, ask = float(getattr(r, "bid", float("nan"))), float(getattr(r, "ask", float("nan")))
            mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else ltp
            spread = (ask - bid) if (bid > 0 and ask > 0) else float("nan")
            spread_pct = spread / mid * 100 if mid > 0 and not math.isnan(spread) else float("nan")
            iv = getattr(r, "iv", None)
            iv = None if iv is None or (isinstance(iv, float) and math.isnan(iv)) else float(iv)
            if iv is None:
                iv = implied_vol(mid, spot, K, T, c.risk_free_rate, ot)
            delta = getattr(r, "delta", None)
            theta = getattr(r, "theta", None)
            if (delta is None or (isinstance(delta, float) and math.isnan(delta))) and iv:
                g = bs_greeks(spot, K, T, c.risk_free_rate, iv, ot)
                delta, theta = g["delta"], g["theta_per_day"]
            if delta is None or (isinstance(delta, float) and math.isnan(delta)):
                delta = None
            vol, oi = float(r.volume), float(r.oi)
            if vol < c.min_volume:
                why.append(f"volume {vol:.0f} < {c.min_volume:.0f}")
            if oi < c.min_oi:
                why.append(f"OI {oi:.0f} < {c.min_oi:.0f}")
            if math.isnan(spread):
                why.append("no bid/ask")
            elif spread > c.max_spread_abs or spread_pct > c.max_spread_pct:
                why.append(f"spread {spread:.2f} ({spread_pct:.2f}%) too wide")
            if not (c.min_premium <= mid <= c.max_premium):
                why.append(f"premium {mid:.2f} outside [{c.min_premium}, {c.max_premium}]")
            if iv is not None and iv > c.max_iv:
                why.append(f"IV {iv:.2f} above cap")
            if delta is not None and not (c.min_abs_delta <= abs(delta) <= c.max_abs_delta):
                why.append(f"|delta| {abs(delta):.2f} outside band")
            moneyness = "ATM" if K == atm else (("ITM" if K < atm else "OTM") if ot == OptionType.CE
                                                else ("ITM" if K > atm else "OTM"))
            m = {"strike": K, "moneyness": moneyness, "premium": round(mid, 2), "ltp": ltp, "bid": bid, "ask": ask,
                 "spread": None if math.isnan(spread) else round(spread, 2),
                 "spread_pct": None if math.isnan(spread_pct) else round(spread_pct, 3),
                 "volume": vol, "oi": oi, "iv": None if iv is None else round(iv, 4),
                 "delta": None if delta is None else round(float(delta), 4),
                 "theta": None if theta is None or (isinstance(theta, float) and math.isnan(theta)) else round(float(theta), 4)}
            if why:
                rejected.append({**m, "rejected_because": why})
                continue
            cands.append(m)
        if not cands:
            return Selection(None, ot, None, reasons=["no contract passed liquidity/spread/premium/delta filters"],
                             rejected=rejected)
        # scoring – normalised within the candidate set
        max_lv = max(math.log1p(x["volume"]) + math.log1p(x["oi"]) for x in cands) or 1.0
        for x in cands:
            liq = (math.log1p(x["volume"]) + math.log1p(x["oi"])) / max_lv
            spr = 1.0 - min((x["spread_pct"] or 0) / c.max_spread_pct, 1.0)
            dlt = 1.0 - min(abs(abs(x["delta"] or c.target_abs_delta) - c.target_abs_delta) / 0.5, 1.0)
            th = 1.0 - min(abs(x["theta"] or 0.0) / max(x["premium"], 1e-6) / 0.1, 1.0)
            x["score"] = round(c.w_liquidity * liq + c.w_spread * spr + c.w_delta * dlt + c.w_theta * th, 5)
            x["score_parts"] = {"liquidity": round(liq, 3), "spread": round(spr, 3), "delta": round(dlt, 3),
                                "theta": round(th, 3)}
        # deterministic tie-break: score, then closeness to preferred strike (never "cheapest")
        best = sorted(cands, key=lambda x: (-x["score"], abs(x["strike"] - pref)))[0]
        contract = OptionContract(und, expiry, best["strike"], ot, lot_size=self.lot_sizes.get(und, 1))
        reasons = [f"expiry {expiry}", f"{best['moneyness']} strike {best['strike']:g} (pref {pref:g})",
                   f"score {best['score']} {best['score_parts']}"]
        return Selection(contract, ot, best["score"], best, reasons, rejected)

    def select_from_snapshots(self, chains: pd.DataFrame, underlying: str, at: datetime) -> dict[OptionType, Selection]:
        """Use the latest snapshot at or before ``at`` (no look-ahead)."""
        snaps = chains[(chains["underlying"] == underlying) & (chains["timestamp"] <= at)]
        if snaps.empty:
            return {ot: Selection(None, ot, None, reasons=["no snapshot at/before selection time"]) for ot in OptionType}
        last = snaps["timestamp"].max()
        return self.select(snaps[snaps["timestamp"] == last], underlying, at)
