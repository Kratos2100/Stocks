"""Deterministic risk management engine.

The risk engine is independent of every strategy and of any learning
component. It can *approve/size/reject* entries and it can *add* hard safety
exits (EOD square-off, max daily loss, profit protection, emergency stop). It
never modifies or suppresses a strategy's own exit rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional

from .config import AppConfig
from .execution.charges import ChargesCalculator
from .models import ExitReason, OrderSide, Position, RiskDecision, Signal, Trade


@dataclass
class DayState:
    day: Optional[date] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_pnl: float = 0.0
    entries: int = 0
    halted: bool = False
    halt_reason: Optional[str] = None
    trades: list[str] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class RiskEngine:
    def __init__(self, cfg: AppConfig, charges: Optional[ChargesCalculator] = None):
        self.cfg = cfg
        self.rc = cfg.risk
        self.charges = charges or ChargesCalculator(cfg.charges)
        self.day = DayState()
        self.emergency = False
        self.emergency_reason: Optional[str] = None
        self.cumulative_realized = 0.0

    # ------------------------------------------------------------ state
    def _roll(self, ts: datetime) -> None:
        if self.day.day != ts.date():
            self.day = DayState(day=ts.date())

    @property
    def equity(self) -> float:
        return self.rc.capital + self.cumulative_realized

    def emergency_stop(self, reason: str = "manual emergency stop") -> None:
        self.emergency = True
        self.emergency_reason = reason

    def clear_emergency(self) -> None:
        self.emergency = False
        self.emergency_reason = None

    def on_trade_closed(self, trade: Trade) -> None:
        self._roll(trade.exit_ts)
        self.day.realized_pnl += trade.net_pnl
        self.cumulative_realized += trade.net_pnl
        self.day.trades.append(trade.trade_id)
        self.day.peak_pnl = max(self.day.peak_pnl, self.day.total_pnl)

    def update_unrealized(self, ts: datetime, positions: Iterable[Position]) -> list[ExitReason]:
        """Mark-to-market check. Returns hard-exit reasons that apply to ALL
        open positions (risk-layer exits)."""
        self._roll(ts)
        self.day.unrealized_pnl = sum(p.unrealized_pnl() for p in positions if p.is_open)
        self.day.peak_pnl = max(self.day.peak_pnl, self.day.total_pnl)
        actions: list[ExitReason] = []
        if self.emergency:
            actions.append(ExitReason.EMERGENCY_STOP)
        if self.day.total_pnl <= -abs(self.rc.max_daily_loss):
            if not self.day.halted:
                self.day.halted, self.day.halt_reason = True, "max daily loss reached"
            if self.rc.flatten_on_daily_loss:
                actions.append(ExitReason.MAX_DAILY_LOSS)
        act = self.rc.profit_protection_activate
        if act is not None and self.day.peak_pnl >= act:
            floor = self.day.peak_pnl * (1 - self.rc.profit_protection_giveback_pct / 100.0)
            if self.day.total_pnl < floor:
                if not self.day.halted:
                    self.day.halted, self.day.halt_reason = True, f"profit protection (floor {floor:.0f})"
                if self.rc.flatten_on_profit_lock:
                    actions.append(ExitReason.PROFIT_PROTECTION)
        return actions

    def is_eod(self, ts: datetime) -> bool:
        return ts.time() >= self.cfg.session.eod_squareoff

    # ---------------------------------------------------------- entries
    def evaluate_entry(
        self,
        signal: Signal,
        price: float,
        ts: datetime,
        open_positions: list[Position],
        lot_size: int,
    ) -> RiskDecision:
        self._roll(ts)
        rc = self.rc
        reasons: list[str] = []
        checks: dict = {}
        s = self.cfg.session

        def fail(msg: str) -> RiskDecision:
            reasons.append(msg)
            return RiskDecision(False, reasons=reasons, checks=checks)

        if self.emergency:
            return fail(f"EMERGENCY_STOP active: {self.emergency_reason}")
        if self.day.halted:
            return fail(f"trading halted: {self.day.halt_reason}")
        t = ts.time()
        if t < s.market_open or t >= s.no_new_entries_after:
            return fail(f"outside entry window ({s.market_open}-{s.no_new_entries_after})")
        if price is None or price <= 0:
            return fail("no valid price")
        if self.day.entries >= rc.max_trades_per_day:
            return fail(f"max trades per day {rc.max_trades_per_day} reached")
        open_now = [p for p in open_positions if p.is_open]
        if len(open_now) >= rc.max_concurrent_positions:
            return fail(f"max concurrent positions {rc.max_concurrent_positions} reached")
        same_und = [p for p in open_now if p.contract.underlying == signal.underlying]
        if len(same_und) >= rc.max_positions_per_underlying:
            return fail(f"max positions for {signal.underlying} reached")

        equity = self.equity
        # --- position sizing
        risk_budget = equity * rc.risk_per_trade_pct / 100.0
        per_unit_risk = None
        if signal.stop_loss is not None and price > signal.stop_loss:
            per_unit_risk = price - signal.stop_loss
            checks["risk_basis"] = "signal stop_loss"
        elif signal.sizing_reference:
            per_unit_risk = signal.sizing_reference
            checks["risk_basis"] = "signal sizing_reference (indicative only, not an exit)"
        if per_unit_risk is None or per_unit_risk <= 0:
            per_unit_risk = price * rc.assumed_risk_pct_of_premium / 100.0
            checks["risk_basis"] = f"assumed {rc.assumed_risk_pct_of_premium}% of premium"
        # configurable floor on per-unit sizing risk (option ranges are wide)
        per_unit_risk = max(per_unit_risk, price * rc.min_risk_pct_of_premium / 100.0)
        lot_cost = price * lot_size
        lots_by_risk = math.floor(risk_budget / (per_unit_risk * lot_size)) if per_unit_risk > 0 else 0

        alloc = rc.agent_allocation.get(signal.agent_id, 1.0)
        agent_cap = equity * alloc
        agent_used = sum(p.cost for p in open_now if p.agent_id == signal.agent_id)
        total_used = sum(p.cost for p in open_now)
        exposure_cap = equity * rc.max_total_exposure_pct / 100.0
        per_trade_cap = equity * rc.max_capital_per_trade_pct / 100.0
        capital_room = min(per_trade_cap, agent_cap - agent_used, exposure_cap - total_used, equity - total_used)
        est_charges = self.charges.total(OrderSide.BUY, price, lot_size)
        lots_by_capital = math.floor(max(capital_room, 0) / (lot_cost + est_charges)) if lot_cost > 0 else 0
        lots = min(lots_by_risk, lots_by_capital, rc.max_lots_per_trade)
        checks.update(
            risk_budget=round(risk_budget, 2), per_unit_risk=round(per_unit_risk, 2), lots_by_risk=lots_by_risk,
            agent_capital=round(agent_cap, 2), agent_used=round(agent_used, 2), capital_room=round(capital_room, 2),
            lots_by_capital=lots_by_capital, max_lots=rc.max_lots_per_trade, lot_size=lot_size,
        )
        if lots < 1:
            limiting = "risk budget" if lots_by_risk < 1 else "capital/exposure limits"
            return fail(f"position size < 1 lot ({limiting})")
        self.day.entries += 1
        qty = lots * lot_size
        reasons.append(f"approved {lots} lot(s) = {qty} qty")
        return RiskDecision(True, qty=qty, lots=lots, reasons=reasons, checks=checks,
                            risk_amount=round(per_unit_risk * qty, 2))

    def release_entry(self) -> None:
        """Undo an entry count when an approved order is rejected/unfilled."""
        self.day.entries = max(0, self.day.entries - 1)

    def snapshot(self) -> dict:
        d = self.day
        return {
            "day": d.day.isoformat() if d.day else None,
            "realized_pnl": round(d.realized_pnl, 2),
            "unrealized_pnl": round(d.unrealized_pnl, 2),
            "peak_pnl": round(d.peak_pnl, 2),
            "entries_today": d.entries,
            "halted": d.halted,
            "halt_reason": d.halt_reason,
            "emergency": self.emergency,
            "emergency_reason": self.emergency_reason,
            "equity": round(self.equity, 2),
            "limits": {
                "max_daily_loss": self.rc.max_daily_loss,
                "max_trades_per_day": self.rc.max_trades_per_day,
                "max_concurrent_positions": self.rc.max_concurrent_positions,
                "risk_per_trade_pct": self.rc.risk_per_trade_pct,
            },
        }
