"""Execution agent: order lifecycle, partial fills, rejections, positions,
manual approval and broker reconciliation. Brokerage independent."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..config import AppConfig
from ..models import (
    ExitReason,
    Fill,
    OptionContract,
    Order,
    OrderPurpose,
    OrderSide,
    OrderStatus,
    Position,
    Signal,
    Trade,
    new_id,
)
from .broker import BrokerAdapter, BrokerEvent, assert_live_authorized
from .charges import ChargesCalculator


@dataclass
class ExecutionCallbacks:
    entry_filled: Callable[[str, str, float, int, datetime], None] = lambda *a: None
    entry_rejected: Callable[[str, str, str, datetime], None] = lambda *a: None
    exit_filled: Callable[[str, str, float, datetime, str, float], None] = lambda *a: None
    trade_closed: Callable[[Trade], None] = lambda t: None
    order_update: Callable[[Order], None] = lambda o: None


@dataclass
class PendingApproval:
    order: Order
    signal: Signal
    risk_amount: Optional[float]
    expires: datetime


class ExecutionEngine:
    MODES = ("backtest", "paper", "manual", "live")

    def __init__(self, cfg: AppConfig, broker: BrokerAdapter, charges: Optional[ChargesCalculator] = None,
                 callbacks: Optional[ExecutionCallbacks] = None, live_confirmed: bool = False):
        mode = cfg.execution.mode
        if mode not in self.MODES:
            raise ValueError(f"unknown execution mode {mode!r}")
        if mode == "live" or broker.is_live:
            assert_live_authorized(cfg, live_confirmed)
        self.cfg = cfg
        self.mode = mode
        self.broker = broker
        self.charges = charges or ChargesCalculator(cfg.charges)
        self.cb = callbacks or ExecutionCallbacks()
        self.orders: dict[str, Order] = {}
        self.positions: dict[str, Position] = {}
        self.open_index: dict[tuple[str, str], str] = {}   # (agent_id, key) -> position_id
        self.pending_entries: dict[tuple[str, str], str] = {}  # (agent_id, key) -> order_id
        self.trades: list[Trade] = []
        self.approvals: dict[str, PendingApproval] = {}
        self.last_price: dict[str, float] = {}
        self._risk_amount: dict[str, Optional[float]] = {}
        self._regime_fn: Callable[[datetime], str] = lambda ts: "UNKNOWN"
        self.log: list[dict[str, Any]] = []
        self._exit_retry: dict[str, ExitReason] = {}   # position_id -> reason

    def set_regime_fn(self, fn: Callable[[datetime], str]) -> None:
        self._regime_fn = fn

    # ----------------------------------------------------------- queries
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.is_open]

    def position_for(self, agent_id: str, key: str) -> Optional[Position]:
        pid = self.open_index.get((agent_id, key))
        return self.positions.get(pid) if pid else None

    def contract_held_by_other(self, agent_id: str, key: str) -> Optional[str]:
        for (aid, k), _ in self.open_index.items():
            if k == key and aid != agent_id:
                return aid
        for (aid, k), _ in self.pending_entries.items():
            if k == key and aid != agent_id:
                return aid
        return None

    # ----------------------------------------------------------- entries
    def submit_entry(self, signal: Signal, contract: OptionContract, qty: int, ts: datetime,
                     market_price: Optional[float], phase: str, risk_amount: Optional[float] = None) -> Order:
        order = Order(
            order_id=new_id("ORD"), contract=contract, side=OrderSide.BUY, qty=qty,
            purpose=OrderPurpose.ENTRY, agent_id=signal.agent_id, created_ts=ts, signal_id=signal.signal_id,
        )
        self.orders[order.order_id] = order
        self._risk_amount[order.order_id] = risk_amount
        self.pending_entries[(signal.agent_id, contract.key)] = order.order_id
        if self.mode == "manual":
            order.status = OrderStatus.PENDING_APPROVAL
            self.approvals[order.order_id] = PendingApproval(
                order, signal, risk_amount, ts + timedelta(minutes=self.cfg.execution.approval_timeout_minutes))
            self._emit(order)
            return order
        self._send(order, market_price, ts, phase)
        return order

    def approve(self, order_id: str, ts: datetime) -> Order:
        pa = self.approvals.pop(order_id)
        if ts > pa.expires:
            self._finalize_unfilled(pa.order, ts, OrderStatus.CANCELLED, "approval expired")
            return pa.order
        pa.order.status = OrderStatus.NEW
        self._send(pa.order, self.last_price.get(pa.order.contract.key), ts, "intrabar")
        return pa.order

    def reject_approval(self, order_id: str, ts: datetime, reason: str = "rejected by operator") -> Order:
        pa = self.approvals.pop(order_id)
        self._finalize_unfilled(pa.order, ts, OrderStatus.REJECTED, reason)
        return pa.order

    def _expire_approvals(self, ts: datetime) -> None:
        for oid in [o for o, pa in self.approvals.items() if ts > pa.expires]:
            pa = self.approvals.pop(oid)
            self._finalize_unfilled(pa.order, ts, OrderStatus.CANCELLED, "approval expired")

    # ------------------------------------------------------------- exits
    def submit_exit(self, agent_id: str, key: str, reason: ExitReason, ts: datetime,
                    market_price: Optional[float], phase: str) -> Optional[Order]:
        pos = self.position_for(agent_id, key)
        if pos is None:
            # cancel a not-yet-filled entry instead
            oid = self.pending_entries.get((agent_id, key))
            if oid:
                self.cancel_entry(oid, ts, f"exit requested before fill ({reason.value})")
            return None
        if pos.exit_order_id and not self.orders[pos.exit_order_id].status.is_terminal:
            return self.orders[pos.exit_order_id]  # never duplicate an exit
        order = Order(
            order_id=new_id("ORD"), contract=pos.contract, side=OrderSide.SELL, qty=pos.open_qty,
            purpose=OrderPurpose.EXIT, agent_id=agent_id, created_ts=ts, position_id=pos.position_id,
            exit_reason=reason,
        )
        pos.exit_order_id = order.order_id
        pos.exit_reason = pos.exit_reason or reason
        self.orders[order.order_id] = order
        self._send(order, market_price, ts, phase)
        return order

    def cancel_entry(self, order_id: str, ts: datetime, reason: str) -> None:
        order = self.orders[order_id]
        if order.status.is_terminal:
            return
        if order_id in self.approvals:
            self.approvals.pop(order_id)
        else:
            self._handle_events(self.broker.cancel_order(order_id, ts))
        if not order.status.is_terminal:
            self._finalize_unfilled(order, ts, OrderStatus.CANCELLED, reason)

    def exit_all(self, reason: ExitReason, ts: datetime, phase: str = "intrabar") -> list[Order]:
        out = []
        for oid in list(self.approvals):
            self.reject_approval(oid, ts, f"{reason.value}: pending approvals cleared")
        for pos in list(self.open_positions()):
            o = self.submit_exit(pos.agent_id, pos.contract.key, reason, ts,
                                 self.last_price.get(pos.contract.key), phase)
            if o:
                out.append(o)
        return out

    # ------------------------------------------------------- market data
    def on_price(self, key: str, price: float, ts: datetime) -> None:
        self.last_price[key] = price
        for pos in self.open_positions():
            if pos.contract.key == key:
                pos.last_price = price
        if self.approvals:
            self._expire_approvals(ts)
        self._handle_events(self.broker.on_price(key, price, ts))
        for pid, reason in list(self._exit_retry.items()):
            pos = self.positions.get(pid)
            if pos is None or not pos.is_open:
                self._exit_retry.pop(pid, None)
            elif pos.contract.key == key:
                self._exit_retry.pop(pid, None)
                self.submit_exit(pos.agent_id, key, reason, ts, price, "intrabar")

    def flush(self, key: str, price: float, ts: datetime) -> None:
        self.last_price[key] = price
        self._handle_events(self.broker.flush(key, price, ts))

    # ---------------------------------------------------------- internal
    def _send(self, order: Order, price: Optional[float], ts: datetime, phase: str) -> None:
        order.status = OrderStatus.SUBMITTED
        order.updated_ts = ts
        self._emit(order)
        self._handle_events(self.broker.place_order(order, price, ts, phase))

    def _emit(self, order: Order) -> None:
        self.log.append(order.to_dict())
        self.cb.order_update(order)

    def _handle_events(self, events: list[BrokerEvent]) -> None:
        for ev in events:
            order = self.orders.get(ev.order_id)
            if order is None or order.status.is_terminal:
                continue
            if ev.kind == "ACK":
                continue
            if ev.kind == "FILL":
                self._on_fill(order, ev)
            elif ev.kind == "REJECT":
                if order.filled_qty > 0:
                    order.status = OrderStatus.CANCELLED
                    self._emit(order)
                else:
                    self._finalize_unfilled(order, ev.ts, OrderStatus.REJECTED, ev.reason or "rejected")
            elif ev.kind == "CANCEL":
                if order.filled_qty > 0:
                    order.status = OrderStatus.CANCELLED  # partial fill, remainder cancelled
                    order.reject_reason = ev.reason
                    order.updated_ts = ev.ts
                    self.pending_entries.pop((order.agent_id, order.contract.key), None)
                    self._emit(order)
                else:
                    self._finalize_unfilled(order, ev.ts, OrderStatus.CANCELLED, ev.reason or "cancelled")

    def _finalize_unfilled(self, order: Order, ts: datetime, status: OrderStatus, reason: str) -> None:
        order.status = status
        order.reject_reason = reason
        order.updated_ts = ts
        self._emit(order)
        if order.purpose == OrderPurpose.ENTRY:
            self.pending_entries.pop((order.agent_id, order.contract.key), None)
            self.cb.entry_rejected(order.agent_id, order.contract.key, reason, ts)
        else:
            pos = self.positions.get(order.position_id or "")
            if pos and pos.is_open:
                pos.exit_order_id = None
                # strategy exits are never dropped: retry on the next price update
                self._exit_retry[pos.position_id] = order.exit_reason or ExitReason.MANUAL

    def _on_fill(self, order: Order, ev: BrokerEvent) -> None:
        first = order.filled_qty == 0
        charges = self.charges.total(order.side, ev.price, ev.qty, orders=1 if first else 0)
        fill = Fill(order.order_id, ev.ts, ev.qty, ev.price, charges)
        order.fills.append(fill)
        new_qty = order.filled_qty + ev.qty
        order.avg_fill_price = (order.avg_fill_price * order.filled_qty + ev.price * ev.qty) / new_qty
        order.filled_qty = new_qty
        order.updated_ts = ev.ts
        order.status = OrderStatus.FILLED if order.remaining_qty == 0 else OrderStatus.PARTIALLY_FILLED
        self._emit(order)
        key = order.contract.key
        if order.purpose == OrderPurpose.ENTRY:
            if order.status == OrderStatus.FILLED:
                self.pending_entries.pop((order.agent_id, key), None)
            pid = self.open_index.get((order.agent_id, key))
            if pid is None:
                pos = Position(
                    position_id=new_id("POS"), agent_id=order.agent_id, contract=order.contract, qty=ev.qty,
                    entry_price=ev.price, entry_ts=ev.ts, signal_id=order.signal_id, entry_charges=charges,
                    last_price=ev.price,
                )
                self.positions[pos.position_id] = pos
                self.open_index[(order.agent_id, key)] = pos.position_id
                self.cb.entry_filled(order.agent_id, key, ev.price, ev.qty, ev.ts)
            else:
                pos = self.positions[pid]
                pos.entry_price = (pos.entry_price * pos.qty + ev.price * ev.qty) / (pos.qty + ev.qty)
                pos.qty += ev.qty
                pos.entry_charges += charges
            return
        pos = self.positions[order.position_id]  # exit
        pos.exit_qty += ev.qty
        pos.exit_value += ev.price * ev.qty
        pos.exit_charges += charges
        pos.exit_ts = ev.ts
        if pos.open_qty <= 0:
            pending = self.pending_entries.get((pos.agent_id, key))
            if pending:  # stop any still-working entry remainder
                self.cancel_entry(pending, ev.ts, "position closed")
            self._close_position(pos, order)

    def _close_position(self, pos: Position, order: Order) -> None:
        pos.is_open = False
        self.open_index.pop((pos.agent_id, pos.contract.key), None)
        exit_px = pos.exit_value / pos.exit_qty
        gross = (exit_px - pos.entry_price) * pos.qty
        charges = pos.entry_charges + pos.exit_charges
        reason = order.exit_reason or pos.exit_reason or ExitReason.MANUAL
        entry_oid = next((o.order_id for o in self.orders.values()
                          if o.signal_id == pos.signal_id and o.purpose == OrderPurpose.ENTRY), None)
        trade = Trade(
            trade_id=new_id("TRD"), agent_id=pos.agent_id, contract_key=pos.contract.key,
            underlying=pos.contract.underlying, option_type=pos.contract.option_type, strike=pos.contract.strike,
            expiry=pos.contract.expiry, qty=pos.qty, entry_ts=pos.entry_ts, entry_price=round(pos.entry_price, 4),
            exit_ts=pos.exit_ts or order.updated_ts, exit_price=round(exit_px, 4), exit_reason=reason,
            gross_pnl=round(gross, 2), charges=round(charges, 2), net_pnl=round(gross - charges, 2),
            signal_id=pos.signal_id, regime=self._regime_fn(pos.entry_ts),
            risk_amount=self._risk_amount.get(entry_oid) if entry_oid else None,
        )
        self.trades.append(trade)
        self.cb.trade_closed(trade)
        self.cb.exit_filled(pos.agent_id, pos.contract.key, exit_px, trade.exit_ts, reason.value, trade.net_pnl)

    # ----------------------------------------------------- reconciliation
    def reconcile(self) -> dict[str, Any]:
        internal: dict[str, int] = {}
        for p in self.open_positions():
            internal[p.contract.key] = internal.get(p.contract.key, 0) + p.open_qty
        broker = self.broker.positions()
        keys = set(internal) | set(broker)
        mismatches = [
            {"contract_key": k, "internal_qty": internal.get(k, 0), "broker_qty": broker.get(k, 0)}
            for k in sorted(keys) if internal.get(k, 0) != broker.get(k, 0)
        ]
        return {"ok": not mismatches, "mismatches": mismatches, "internal": internal, "broker": broker}
