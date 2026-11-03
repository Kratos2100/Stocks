"""Base class for strategy agents.

An agent is a lightweight, deterministic Python worker (no LLM). It keeps an
independent :class:`ContractContext` for every option contract it monitors, so
CE and PE (and different strikes) never share candles, indicators, reference
candles, breakout candles, entry prices or positions.

Event interface (identical in backtest, paper and live):

* :meth:`on_candle_close` – a *completed* candle for one contract.
* :meth:`on_price`        – an intrabar price update (entry triggers and
  price-reach exits only).
* :meth:`on_entry_filled` / :meth:`on_entry_rejected` / :meth:`on_exit_filled`
  – feedback from the execution layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from ..models import (
    AgentState,
    Candle,
    ExitReason,
    OptionContract,
    OptionType,
    Signal,
    SignalStatus,
    SignalType,
)

MAX_HISTORY = 3000


@dataclass
class ContractContext:
    contract: OptionContract
    state: AgentState
    candles: list[Candle] = field(default_factory=list)
    series: dict[str, list[Any]] = field(default_factory=dict)
    state_history: list[dict[str, Any]] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)
    session_date: Optional[date] = None
    consumed_events: set[str] = field(default_factory=set)
    # setup / position bookkeeping (strategy specific keys)
    setup: dict[str, Any] = field(default_factory=dict)
    position: Optional[dict[str, Any]] = None
    last_signal: Optional[Signal] = None
    last_exit_reason: Optional[str] = None
    realized_pnl: float = 0.0
    trades: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.contract.key

    def push_series(self, name: str, value: Any) -> None:
        lst = self.series.setdefault(name, [])
        lst.append(value)
        if len(lst) > MAX_HISTORY:
            del lst[: len(lst) - MAX_HISTORY]

    def push_candle(self, candle: Candle) -> None:
        self.candles.append(candle)
        if len(self.candles) > MAX_HISTORY:
            del self.candles[: len(self.candles) - MAX_HISTORY]


class BaseStrategyAgent(ABC):
    agent_id: str = "A00"
    strategy_name: str = "base"
    initial_state: AgentState = AgentState.WAITING_FOR_SIGNAL

    def __init__(self, config: Any, timeframe_min: int = 5):
        self.config = config
        self.timeframe = f"{timeframe_min}m"
        self.timeframe_min = timeframe_min
        self.contexts: dict[str, ContractContext] = {}

    # ------------------------------------------------------------------ setup
    def register_contract(self, contract: OptionContract, warmup: list[Candle] | tuple = ()) -> ContractContext:
        """Register a contract and feed historical candles for indicator
        warm-up. Warm-up candles never generate signals."""
        if contract.key in self.contexts:
            ctx = self.contexts[contract.key]
        else:
            ctx = ContractContext(contract=contract, state=self.initial_state)
            self.contexts[contract.key] = ctx
            self._init_context(ctx)
        for c in warmup:
            self._roll_session_if_needed(ctx, c, live=False)
            ctx.push_candle(c)
            self._update_indicators(ctx, c)
        return ctx

    def has_contract(self, key: str) -> bool:
        return key in self.contexts

    # --------------------------------------------------------------- events
    def on_candle_close(self, key: str, candle: Candle) -> list[Signal]:
        ctx = self.contexts[key]
        out: list[Signal] = []
        out.extend(self._roll_session_if_needed(ctx, candle, live=True))
        ctx.push_candle(candle)
        info = self._update_indicators(ctx, candle)
        out.extend(self._on_close(ctx, candle, info))
        if out:
            ctx.last_signal = out[-1]
        return out

    def on_price(self, key: str, price: float, ts: datetime) -> list[Signal]:
        ctx = self.contexts.get(key)
        if ctx is None:
            return []
        out = self._on_price(ctx, price, ts)
        if out:
            ctx.last_signal = out[-1]
        return out

    def price_levels(self, key: str) -> list[float]:
        """Active intrabar trigger levels (entry triggers, targets, stops).
        The backtester inserts these into the simulated intrabar path so that
        fills occur exactly at the level when it is crossed."""
        ctx = self.contexts.get(key)
        return self._price_levels(ctx) if ctx else []

    def on_entry_filled(self, key: str, price: float, qty: int, ts: datetime) -> None:
        ctx = self.contexts[key]
        ctx.position = {
            "entry_price": price,
            "qty": qty,
            "entry_ts": ts,
            "entry_bar_ts": self._bar_start(ts),
            "bars_held": 0,
            **ctx.setup.get("position_seed", {}),
        }
        self._set_state(ctx, AgentState.POSITION_OPEN, ts, f"entry filled @ {price:.2f} x {qty}")
        self._marker(ctx, ts, "entry", price, f"{self.agent_id} BUY {ctx.contract.option_type.value} {price:.2f}")

    def on_entry_rejected(self, key: str, reason: str, ts: datetime) -> None:
        ctx = self.contexts[key]
        if ctx.state != AgentState.ENTRY_TRIGGERED:
            return  # never disturb an open position or an unrelated state
        # The event is already consumed, so the same setup can never re-enter.
        ctx.setup = {}
        ctx.position = None
        self._set_state(ctx, self.idle_state(), ts, f"entry rejected: {reason}")

    def on_exit_filled(self, key: str, price: float, ts: datetime, reason: str, pnl: float = 0.0) -> None:
        ctx = self.contexts[key]
        ctx.last_exit_reason = reason
        ctx.realized_pnl += pnl
        ctx.trades += 1
        ctx.position = None
        ctx.setup = {}
        self._marker(ctx, ts, "exit", price, f"{self.agent_id} EXIT {reason} {price:.2f}")
        self._set_state(ctx, AgentState.TRADE_COMPLETED, ts, f"exit filled @ {price:.2f} ({reason})")
        self._set_state(ctx, self.idle_state(), ts, "ready for next setup")

    def end_session(self, key: str, ts: datetime) -> list[Signal]:
        """Cancel pending (not-yet-entered) setups at session end."""
        ctx = self.contexts.get(key)
        if ctx is None:
            return []
        return self._expire_pending(ctx, ts, "session ended")

    # ------------------------------------------------------- abstract hooks
    @abstractmethod
    def _init_context(self, ctx: ContractContext) -> None: ...

    @abstractmethod
    def _update_indicators(self, ctx: ContractContext, candle: Candle) -> dict[str, Any]: ...

    @abstractmethod
    def _on_close(self, ctx: ContractContext, candle: Candle, info: dict[str, Any]) -> list[Signal]: ...

    @abstractmethod
    def _on_price(self, ctx: ContractContext, price: float, ts: datetime) -> list[Signal]: ...

    def _price_levels(self, ctx: ContractContext) -> list[float]:
        return []

    def _on_new_session(self, ctx: ContractContext, prev_session: Optional[date], new_session: date) -> None:
        """Strategy-specific session rollover (VWAP reset, pivots...)."""

    def idle_state(self) -> AgentState:
        return self.initial_state

    def _expire_pending(self, ctx: ContractContext, ts: datetime, why: str) -> list[Signal]:
        if ctx.state in (
            AgentState.CROSSOVER_DETECTED,
            AgentState.WAITING_FOR_BREAKOUT,
            AgentState.BREAKOUT_IDENTIFIED,
            AgentState.WAITING_FOR_ENTRY,
        ):
            ref = ctx.setup.get("reference_candle")
            sig = self._signal(
                ctx,
                SignalType.NO_TRADE,
                SignalStatus.CANCELLED,
                ts,
                reason=f"Pending setup cancelled: {why}",
                rejection_reason=why,
                reference=ref,
            )
            ctx.setup = {}
            self._set_state(ctx, self.idle_state(), ts, f"setup expired: {why}")
            return [sig]
        return []

    # ------------------------------------------------------------- helpers
    def _roll_session_if_needed(self, ctx: ContractContext, candle: Candle, live: bool) -> list[Signal]:
        d = candle.ts.date()
        if ctx.session_date == d:
            return []
        prev = ctx.session_date
        out: list[Signal] = []
        if live and prev is not None and getattr(self.config, "expire_setups_at_session_end", True):
            out.extend(self._expire_pending(ctx, candle.ts, "new session"))
        ctx.session_date = d
        self._on_new_session(ctx, prev, d)
        return out

    def _bar_start(self, ts: datetime) -> datetime:
        minute = (ts.hour * 60 + ts.minute)
        # bars anchored at 09:15
        anchor = 9 * 60 + 15
        off = (minute - anchor) % self.timeframe_min
        start_min = minute - off
        return ts.replace(hour=start_min // 60, minute=start_min % 60, second=0, microsecond=0)

    def _set_state(self, ctx: ContractContext, state: AgentState, ts: datetime, note: str = "") -> None:
        ctx.state = state
        ctx.state_history.append({"ts": ts.isoformat(), "state": state.value, "note": note})
        if len(ctx.state_history) > MAX_HISTORY:
            del ctx.state_history[: len(ctx.state_history) - MAX_HISTORY]

    def _marker(self, ctx: ContractContext, ts: datetime, kind: str, price: float, text: str) -> None:
        ctx.markers.append({"ts": ts.isoformat(), "kind": kind, "price": price, "text": text, "agent": self.agent_id})
        if len(ctx.markers) > MAX_HISTORY:
            del ctx.markers[: len(ctx.markers) - MAX_HISTORY]

    def _buy_type(self, ctx: ContractContext) -> SignalType:
        return SignalType.BUY_CE if ctx.contract.option_type == OptionType.CE else SignalType.BUY_PE

    def _signal(
        self,
        ctx: ContractContext,
        signal_type: SignalType,
        status: SignalStatus,
        ts: datetime,
        *,
        reason: str = "",
        entry_trigger: Optional[float] = None,
        breakout: Optional[Candle] = None,
        reference: Optional[Candle | dict] = None,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        exit_condition: str = "",
        satisfied: Optional[list[str]] = None,
        not_satisfied: Optional[list[str]] = None,
        rejection_reason: Optional[str] = None,
        event_id: Optional[str] = None,
        trigger_price: Optional[float] = None,
        exit_reason: Optional[ExitReason] = None,
        sizing_reference: Optional[float] = None,
        extras: Optional[dict[str, Any]] = None,
    ) -> Signal:
        c = ctx.contract
        ref_dict = reference.to_dict() if isinstance(reference, Candle) else reference
        return Signal(
            agent_id=self.agent_id,
            strategy_name=self.strategy_name,
            exchange=c.exchange,
            underlying=c.underlying,
            option_type=c.option_type,
            strike=c.strike,
            expiry=c.expiry,
            timeframe=self.timeframe,
            signal_type=signal_type,
            status=status,
            signal_ts=ts,
            contract_key=c.key,
            reason=reason,
            entry_trigger=entry_trigger,
            breakout_candle_ts=breakout.ts if breakout else None,
            breakout_candle_high=breakout.high if breakout else None,
            stop_loss=stop_loss,
            target=target,
            exit_condition=exit_condition,
            conditions_satisfied=list(satisfied or []),
            conditions_not_satisfied=list(not_satisfied or []),
            reference_candle=ref_dict,
            breakout_candle=breakout.to_dict() if breakout else None,
            rejection_reason=rejection_reason,
            event_id=event_id,
            trigger_price=trigger_price,
            exit_reason=exit_reason,
            agent_state=ctx.state,
            sizing_reference=sizing_reference,
            extras=dict(extras or {}),
        )

    # ------------------------------------------------------------ dashboard
    def snapshot(self, keys: Optional[set[str]] = None) -> dict[str, Any]:
        out = []
        for key, ctx in self.contexts.items():
            if keys is not None and key not in keys:
                continue
            sig = ctx.last_signal
            setup = ctx.setup
            ref = setup.get("reference_candle")
            brk = setup.get("breakout_candle")
            out.append(
                {
                    "contract": ctx.contract.to_dict(),
                    "state": ctx.state.value,
                    "current_signal": sig.to_dict() if sig else None,
                    "reference_candle": ref.to_dict() if isinstance(ref, Candle) else ref,
                    "breakout_candle": brk.to_dict() if isinstance(brk, Candle) else brk,
                    "entry_trigger": setup.get("entry_price"),
                    "position": {
                        k: (v.isoformat() if isinstance(v, datetime) else v)
                        for k, v in (ctx.position or {}).items()
                    }
                    or None,
                    "last_exit_reason": ctx.last_exit_reason,
                    "realized_pnl": round(ctx.realized_pnl, 2),
                    "trades": ctx.trades,
                }
            )
        return {
            "agent_id": self.agent_id,
            "strategy_name": self.strategy_name,
            "enabled": bool(getattr(self.config, "enabled", True)),
            "contracts": out,
        }

    def chart_data(self, key: str) -> dict[str, Any]:
        ctx = self.contexts[key]
        return {
            "series": {k: list(v) for k, v in ctx.series.items()},
            "markers": list(ctx.markers),
            "extras": self._chart_extras(ctx),
        }

    def _chart_extras(self, ctx: ContractContext) -> dict[str, Any]:
        return {}
