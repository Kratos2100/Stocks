"""Broker abstraction layer.

* :class:`BrokerAdapter`          – the interface every broker must implement.
* :class:`SimulatedBroker`        – deterministic fill simulator used for
  historical backtesting *and* paper trading.
* :class:`IndianBrokerAdapterTemplate` – skeleton for a future live
  integration (Zerodha Kite, Upstox, Angel One, Dhan, Fyers, ...). It refuses to
  instantiate unless live trading has been explicitly authorised.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..config import AppConfig, ExecutionConfig
from ..models import Order, OrderSide, OrderStatus

LIVE_AUTH_ENV = "OPTIONS_MAS_LIVE_AUTHORIZATION"
LIVE_AUTH_PHRASE = "I_AUTHORIZE_LIVE_TRADING"
TICK_SIZE = 0.05


class LiveTradingNotAuthorized(RuntimeError):
    pass


def assert_live_authorized(cfg: AppConfig, cli_confirmed: bool = False) -> None:
    """Live trading requires ALL of:

    1. ``execution.live_enabled = true`` in the config file,
    2. ``execution.mode = "live"``,
    3. the environment variable ``OPTIONS_MAS_LIVE_AUTHORIZATION`` set to
       ``I_AUTHORIZE_LIVE_TRADING``,
    4. an explicit per-run confirmation flag (``--confirm-live``).
    """
    problems = []
    if not cfg.execution.live_enabled:
        problems.append("execution.live_enabled is false")
    if cfg.execution.mode != "live":
        problems.append("execution.mode is not 'live'")
    if os.environ.get(LIVE_AUTH_ENV) != LIVE_AUTH_PHRASE:
        problems.append(f"env {LIVE_AUTH_ENV} not set to the authorisation phrase")
    if not cli_confirmed:
        problems.append("per-run confirmation flag not given")
    if problems:
        raise LiveTradingNotAuthorized("Live trading is disabled: " + "; ".join(problems))


@dataclass
class BrokerEvent:
    kind: str                # "FILL" | "REJECT" | "CANCEL" | "ACK"
    order_id: str
    ts: datetime
    qty: int = 0
    price: float = 0.0
    reason: Optional[str] = None


class BrokerAdapter(ABC):
    name = "abstract"
    is_live = False

    @abstractmethod
    def place_order(self, order: Order, market_price: Optional[float], ts: datetime, phase: str) -> list[BrokerEvent]:
        """Submit an order. ``phase`` is "intrabar" (a live price is known) or
        "close" (generated after a candle closed)."""

    @abstractmethod
    def cancel_order(self, order_id: str, ts: datetime) -> list[BrokerEvent]: ...

    def on_price(self, key: str, price: float, ts: datetime) -> list[BrokerEvent]:
        return []

    def flush(self, key: str, price: float, ts: datetime) -> list[BrokerEvent]:
        return []

    @abstractmethod
    def positions(self) -> dict[str, int]: ...

    def available_funds(self) -> Optional[float]:
        return None


def round_to_tick(price: float, side: OrderSide) -> float:
    ticks = price / TICK_SIZE
    ticks = math.ceil(ticks - 1e-9) if side == OrderSide.BUY else math.floor(ticks + 1e-9)
    return round(max(ticks * TICK_SIZE, TICK_SIZE), 2)


class SimulatedBroker(BrokerAdapter):
    """Deterministic fill model.

    * Market orders submitted intrabar fill at the current price +/- slippage.
    * Orders generated at a candle close fill at the *next* price update for
      that contract (normally the next bar open) when
      ``close_signal_fill = "next_open"``.
    * Optional participation cap: an order can fill at most
      ``max_participation_pct`` of the *previous* bar's volume (no look-ahead).
      Entry remainders are cancelled (IOC); exit remainders stay working.
    """

    name = "simulated"

    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self.queued: dict[str, list[Order]] = {}
        self.working_exits: dict[str, list[Order]] = {}
        self.capacity: dict[str, Optional[float]] = {}
        self._positions: dict[str, int] = {}
        self.orders: dict[str, Order] = {}
        self.last_price: dict[str, float] = {}

    # liquidity hint from the data layer (previous completed bar volume)
    def set_liquidity(self, key: str, prev_bar_volume: Optional[float]) -> None:
        pct = self.cfg.max_participation_pct
        if pct is None or prev_bar_volume is None:
            self.capacity[key] = None
        else:
            self.capacity[key] = math.floor(prev_bar_volume * pct / 100.0)

    def _slip(self, price: float, side: OrderSide) -> float:
        slip = self.cfg.slippage_points + price * self.cfg.slippage_pct / 100.0
        p = price + slip if side == OrderSide.BUY else price - slip
        return round_to_tick(p, side)

    def place_order(self, order: Order, market_price: Optional[float], ts: datetime, phase: str) -> list[BrokerEvent]:
        self.orders[order.order_id] = order
        lot = max(order.contract.lot_size, 1)
        if order.qty <= 0 or order.qty % lot != 0:
            return [BrokerEvent("REJECT", order.order_id, ts, reason=f"qty {order.qty} not a multiple of lot {lot}")]
        key = order.contract.key
        if phase == "close" and self.cfg.close_signal_fill == "next_open":
            self.queued.setdefault(key, []).append(order)
            return [BrokerEvent("ACK", order.order_id, ts)]
        if market_price is None or market_price <= 0:
            if self.cfg.reject_if_no_price:
                return [BrokerEvent("REJECT", order.order_id, ts, reason="no market price")]
            self.queued.setdefault(key, []).append(order)
            return [BrokerEvent("ACK", order.order_id, ts)]
        return [BrokerEvent("ACK", order.order_id, ts)] + self._execute(order, market_price, ts)

    def _execute(self, order: Order, price: float, ts: datetime) -> list[BrokerEvent]:
        key = order.contract.key
        lot = max(order.contract.lot_size, 1)
        remaining = order.qty - order.filled_qty
        cap = self.capacity.get(key)
        qty = remaining if cap is None else min(remaining, int(cap // lot) * lot)
        events: list[BrokerEvent] = []
        if qty > 0:
            fill_px = self._slip(price, order.side)
            if cap is not None:
                self.capacity[key] = cap - qty
            sign = 1 if order.side == OrderSide.BUY else -1
            self._positions[key] = self._positions.get(key, 0) + sign * qty
            events.append(BrokerEvent("FILL", order.order_id, ts, qty=qty, price=fill_px))
        left = remaining - qty
        if left > 0:
            if order.side == OrderSide.BUY:
                events.append(BrokerEvent("CANCEL", order.order_id, ts, qty=left,
                                          reason="insufficient liquidity (participation cap); remainder cancelled"))
            else:
                self.working_exits.setdefault(key, []).append(order)
        return events

    def on_price(self, key: str, price: float, ts: datetime) -> list[BrokerEvent]:
        self.last_price[key] = price
        events: list[BrokerEvent] = []
        q = self.queued.pop(key, [])
        w = self.working_exits.pop(key, [])
        for o in q + w:
            if o.status.is_terminal:
                continue
            events.extend(self._execute(o, price, ts))
        return events

    def flush(self, key: str, price: float, ts: datetime) -> list[BrokerEvent]:
        """Force-fill everything outstanding (session end), ignoring the cap."""
        saved = self.capacity.get(key)
        self.capacity[key] = None
        ev = self.on_price(key, price, ts)
        self.capacity[key] = saved
        return ev

    def cancel_order(self, order_id: str, ts: datetime) -> list[BrokerEvent]:
        for book in (self.queued, self.working_exits):
            for key, lst in book.items():
                for o in list(lst):
                    if o.order_id == order_id:
                        lst.remove(o)
                        return [BrokerEvent("CANCEL", order_id, ts, qty=o.remaining_qty, reason="cancelled")]
        return []

    def positions(self) -> dict[str, int]:
        return {k: v for k, v in self._positions.items() if v != 0}


class IndianBrokerAdapterTemplate(BrokerAdapter):
    """Template for a live Indian broker integration.

    Implement the methods using the broker's official SDK. Map
    :class:`~options_mas.models.OptionContract` to the broker's trading symbol
    / instrument token, use exchange segment ``NFO``, product ``MIS`` (intraday)
    and handle order-update websockets by translating them to
    :class:`BrokerEvent` objects.
    """

    name = "live-template"
    is_live = True

    def __init__(self, cfg: AppConfig, cli_confirmed: bool = False):
        assert_live_authorized(cfg, cli_confirmed)
        self.cfg = cfg

    def place_order(self, order, market_price, ts, phase):  # pragma: no cover - template
        raise NotImplementedError("Implement with your broker SDK")

    def cancel_order(self, order_id, ts):  # pragma: no cover - template
        raise NotImplementedError

    def positions(self):  # pragma: no cover - template
        raise NotImplementedError

    def available_funds(self):  # pragma: no cover - template
        raise NotImplementedError
