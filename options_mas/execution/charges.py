"""Brokerage and statutory charges for NSE index options (buy & sell legs).

Rates are configurable in :class:`ChargesConfig` and MUST be verified against
the current broker contract note / NSE & SEBI circulars before use.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ChargesConfig
from ..models import OrderSide


@dataclass
class ChargeBreakdown:
    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp: float
    gst: float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange_txn + self.sebi + self.stamp + self.gst

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["total"] = self.total
        return {k: round(v, 4) for k, v in d.items()}


class ChargesCalculator:
    def __init__(self, cfg: ChargesConfig | None = None):
        self.cfg = cfg or ChargesConfig()

    def compute(self, side: OrderSide, price: float, qty: int, orders: int = 1) -> ChargeBreakdown:
        c = self.cfg
        if not c.enabled or qty <= 0:
            return ChargeBreakdown(0, 0, 0, 0, 0, 0)
        turnover = price * qty
        brokerage = c.brokerage_per_order * orders
        stt = turnover * c.stt_sell_pct / 100.0 if side == OrderSide.SELL else 0.0
        exch = turnover * c.exchange_txn_pct / 100.0
        sebi = turnover * c.sebi_per_crore / 1e7
        stamp = turnover * c.stamp_buy_pct / 100.0 if side == OrderSide.BUY else 0.0
        gst = (brokerage + exch + sebi) * c.gst_pct / 100.0
        return ChargeBreakdown(brokerage, stt, exch, sebi, stamp, gst)

    def total(self, side: OrderSide, price: float, qty: int, orders: int = 1) -> float:
        return self.compute(side, price, qty, orders).total

    def round_trip(self, entry: float, exit_: float, qty: int) -> float:
        return self.total(OrderSide.BUY, entry, qty) + self.total(OrderSide.SELL, exit_, qty)
