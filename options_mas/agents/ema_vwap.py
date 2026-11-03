"""AGENT 01 – 50 EMA + VWAP Crossover Candle Breakout.

Rules (applied independently to every monitored CE and PE premium chart):

1. Compute the 50 EMA and session VWAP on the option premium chart.
2. Detect an *actual* crossover (sign change of EMA - VWAP between two
   completed candles). The candle on which it occurs is the **crossover
   (reference) candle**; store its OHLC.
3. Wait for a **subsequent** candle to break above the crossover candle high.
   The crossover candle can never be the breakout candle.
4. Only after the breakout candle has **closed**, record its high and compute
   ``ENTRY = breakout high + 2.00``.
5. Trigger BUY CE / BUY PE when the premium reaches the entry price.
6. Exit when a **completed** candle closes below the low of the immediately
   preceding completed candle. Wicks never trigger the exit.

No VWAP-position, EMA-slope, volume, underlying-index or cross-agent
confirmation is applied.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from ..config import Agent01Config
from ..indicators import EMA, SessionVWAP
from ..models import AgentState, Candle, ExitReason, Signal, SignalStatus, SignalType
from .base import BaseStrategyAgent, ContractContext


class EmaVwapBreakoutAgent(BaseStrategyAgent):
    agent_id = "A01"
    strategy_name = "50 EMA + VWAP Crossover Candle Breakout"
    initial_state = AgentState.WAITING_FOR_CROSSOVER

    def __init__(self, config: Optional[Agent01Config] = None, timeframe_min: int = 5):
        super().__init__(config or Agent01Config(), timeframe_min)
        self.config: Agent01Config

    # ------------------------------------------------------------ indicators
    def _init_context(self, ctx: ContractContext) -> None:
        ctx.scratch["ema"] = EMA(self.config.ema_period)
        ctx.scratch["vwap"] = SessionVWAP()
        ctx.scratch["last_sign"] = None

    def _on_new_session(self, ctx: ContractContext, prev: Optional[date], new: date) -> None:
        ctx.scratch["vwap"].reset()
        if self.config.ema_reset_daily:
            ctx.scratch["ema"].reset()
            ctx.scratch["last_sign"] = None

    def _update_indicators(self, ctx: ContractContext, candle: Candle) -> dict[str, Any]:
        ema = ctx.scratch["ema"].update(candle.close)
        vwap = ctx.scratch["vwap"].update(candle)
        ctx.push_series("ema50", ema)
        ctx.push_series("vwap", vwap)
        crossed = None
        if ema is not None and vwap is not None:
            diff = ema - vwap
            if diff != 0:
                sign = 1 if diff > 0 else -1
                last = ctx.scratch["last_sign"]
                if last is not None and sign != last:
                    crossed = "EMA_ABOVE_VWAP" if sign > 0 else "EMA_BELOW_VWAP"
                ctx.scratch["last_sign"] = sign
        return {"ema": ema, "vwap": vwap, "crossed": crossed}

    def _direction_allowed(self, crossed: str) -> bool:
        d = self.config.crossover_direction
        if d == "any":
            return True
        if d == "ema_above_vwap":
            return crossed == "EMA_ABOVE_VWAP"
        if d == "ema_below_vwap":
            return crossed == "EMA_BELOW_VWAP"
        raise ValueError(f"unknown crossover_direction {d!r}")

    # ---------------------------------------------------------- close logic
    def _on_close(self, ctx: ContractContext, candle: Candle, info: dict[str, Any]) -> list[Signal]:
        st = ctx.state
        ts = candle.end_ts
        out: list[Signal] = []

        if st == AgentState.POSITION_OPEN:
            return self._check_exit(ctx, candle)
        if st in (AgentState.ENTRY_TRIGGERED, AgentState.EXIT_TRIGGERED):
            return out  # waiting for execution feedback

        crossed = info["crossed"]
        if crossed and self._direction_allowed(crossed):
            if st in (AgentState.WAITING_FOR_BREAKOUT, AgentState.BREAKOUT_IDENTIFIED):
                if not self.config.reset_on_new_crossover:
                    return self._advance_pending(ctx, candle)
                out.append(
                    self._signal(
                        ctx, SignalType.NO_TRADE, SignalStatus.CANCELLED, ts,
                        reason="Previous setup superseded by a new EMA/VWAP crossover",
                        rejection_reason="superseded by new crossover",
                        reference=ctx.setup.get("reference_candle"),
                    )
                )
            out.extend(self._register_crossover(ctx, candle, info))
            return out

        if st in (AgentState.WAITING_FOR_BREAKOUT, AgentState.BREAKOUT_IDENTIFIED):
            return self._advance_pending(ctx, candle)
        return out

    def _register_crossover(self, ctx: ContractContext, candle: Candle, info: dict[str, Any]) -> list[Signal]:
        ts = candle.end_ts
        event_id = f"A01:{ctx.key}:XO:{candle.ts.isoformat()}"
        ctx.setup = {
            "event_id": event_id,
            "reference_candle": candle,
            "crossover": info["crossed"],
            "bars_waiting": 0,
        }
        self._set_state(ctx, AgentState.CROSSOVER_DETECTED, ts, f"{info['crossed']} on candle {candle.ts:%H:%M}")
        self._marker(ctx, candle.ts, "crossover", candle.high, "XO")
        self._set_state(ctx, AgentState.WAITING_FOR_BREAKOUT, ts, f"reference high {candle.high:.2f}")
        return [
            self._signal(
                ctx, SignalType.WAIT, SignalStatus.INFO, ts,
                reason=(
                    f"50 EMA/VWAP crossover ({info['crossed']}) on {candle.ts:%Y-%m-%d %H:%M}; "
                    f"waiting for a subsequent candle to break above {candle.high:.2f}"
                ),
                reference=candle,
                satisfied=[f"Actual 50 EMA/VWAP crossover ({info['crossed']})"],
                not_satisfied=["Subsequent breakout candle above crossover candle high"],
                event_id=event_id,
                extras={"ema50": info["ema"], "vwap": info["vwap"]},
            )
        ]

    def _advance_pending(self, ctx: ContractContext, candle: Candle) -> list[Signal]:
        ts = candle.end_ts
        cfg = self.config
        setup = ctx.setup
        ref: Candle = setup["reference_candle"]

        if ctx.state == AgentState.WAITING_FOR_BREAKOUT:
            setup["bars_waiting"] += 1
            basis = candle.high if cfg.breakout_basis == "high" else candle.close
            if candle.ts > ref.ts and basis > ref.high:
                entry = round(candle.high + cfg.entry_offset, 4)
                setup.update(
                    breakout_candle=candle,
                    entry_price=entry,
                    bars_since_breakout=0,
                    position_seed={"reference_high": ref.high, "breakout_high": candle.high},
                )
                self._set_state(ctx, AgentState.BREAKOUT_IDENTIFIED, ts, f"breakout candle {candle.ts:%H:%M}, entry {entry:.2f}")
                self._marker(ctx, candle.ts, "breakout", candle.high, "BO")
                return [
                    self._signal(
                        ctx, self._buy_type(ctx), SignalStatus.ARMED, ts,
                        reason=(
                            f"Breakout candle {candle.ts:%H:%M} (high {candle.high:.2f}) broke crossover "
                            f"candle {ref.ts:%H:%M} high {ref.high:.2f}; entry armed at {entry:.2f}"
                        ),
                        entry_trigger=entry,
                        breakout=candle,
                        reference=ref,
                        exit_condition="Completed candle close < previous completed candle low",
                        satisfied=[
                            "Actual 50 EMA/VWAP crossover",
                            "Subsequent candle broke above crossover candle high",
                            "Breakout candle completed",
                        ],
                        not_satisfied=[f"Premium must reach entry {entry:.2f} (breakout high + {cfg.entry_offset:.2f})"],
                        event_id=setup["event_id"],
                        sizing_reference=max(entry - candle.low, 0.0) or None,
                    )
                ]
            if cfg.breakout_max_wait_bars is not None and setup["bars_waiting"] >= cfg.breakout_max_wait_bars:
                return self._cancel(ctx, ts, f"no breakout within {cfg.breakout_max_wait_bars} bars")
            return []

        if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
            setup["bars_since_breakout"] += 1
            if cfg.entry_valid_bars is not None and setup["bars_since_breakout"] >= cfg.entry_valid_bars:
                return self._cancel(ctx, ts, f"entry not reached within {cfg.entry_valid_bars} bars")
        return []

    def _cancel(self, ctx: ContractContext, ts: datetime, why: str) -> list[Signal]:
        sig = self._signal(
            ctx, SignalType.NO_TRADE, SignalStatus.REJECTED, ts,
            reason=f"Setup rejected: {why}", rejection_reason=why,
            reference=ctx.setup.get("reference_candle"),
            breakout=ctx.setup.get("breakout_candle"),
            event_id=ctx.setup.get("event_id"),
        )
        ctx.setup = {}
        self._set_state(ctx, AgentState.WAITING_FOR_CROSSOVER, ts, why)
        return [sig]

    def _check_exit(self, ctx: ContractContext, candle: Candle) -> list[Signal]:
        pos = ctx.position or {}
        ts = candle.end_ts
        if len(ctx.candles) < 2:
            return []
        if candle.ts < pos.get("entry_bar_ts", candle.ts):
            return []
        is_entry_candle = candle.ts == pos.get("entry_bar_ts")
        if is_entry_candle and not self.config.evaluate_exit_on_entry_candle:
            return []
        pos["bars_held"] = pos.get("bars_held", 0) + 1
        prev = ctx.candles[-2]
        if candle.close < prev.low:
            self._set_state(ctx, AgentState.EXIT_TRIGGERED, ts, f"close {candle.close:.2f} < prev low {prev.low:.2f}")
            return [
                self._signal(
                    ctx, SignalType.EXIT_POSITION, SignalStatus.TRIGGERED, ts,
                    reason=(
                        f"Completed candle {candle.ts:%H:%M} closed at {candle.close:.2f}, below previous "
                        f"candle {prev.ts:%H:%M} low {prev.low:.2f}"
                    ),
                    exit_condition="CURRENT CANDLE CLOSE < PREVIOUS CANDLE LOW",
                    satisfied=["Current candle completed", "Close below previous completed candle low"],
                    exit_reason=ExitReason.PREV_CANDLE_LOW_CLOSE,
                    trigger_price=candle.close,
                    reference=prev,
                    event_id=ctx.setup.get("event_id"),
                )
            ]
        return []

    # ---------------------------------------------------------- price logic
    def _on_price(self, ctx: ContractContext, price: float, ts: datetime) -> list[Signal]:
        if ctx.state != AgentState.BREAKOUT_IDENTIFIED:
            return []
        setup = ctx.setup
        entry = setup["entry_price"]
        if price < entry:
            return []
        event_id = setup["event_id"]
        if event_id in ctx.consumed_events:  # duplicate protection
            return []
        ctx.consumed_events.add(event_id)
        self._set_state(ctx, AgentState.ENTRY_TRIGGERED, ts, f"premium {price:.2f} >= entry {entry:.2f}")
        brk: Candle = setup["breakout_candle"]
        ref: Candle = setup["reference_candle"]
        return [
            self._signal(
                ctx, self._buy_type(ctx), SignalStatus.TRIGGERED, ts,
                reason=(
                    f"Premium reached entry {entry:.2f} = breakout candle high {brk.high:.2f} + "
                    f"{self.config.entry_offset:.2f}"
                ),
                entry_trigger=entry,
                breakout=brk,
                reference=ref,
                exit_condition="Completed candle close < previous completed candle low",
                satisfied=[
                    "Actual 50 EMA/VWAP crossover",
                    "Separate subsequent breakout candle above crossover candle high",
                    "Breakout candle completed",
                    f"Premium reached breakout high + {self.config.entry_offset:.2f}",
                ],
                event_id=event_id,
                trigger_price=price,
                sizing_reference=max(entry - brk.low, 0.0) or None,
            )
        ]

    def _price_levels(self, ctx: ContractContext) -> list[float]:
        if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
            return [ctx.setup["entry_price"]]
        return []
