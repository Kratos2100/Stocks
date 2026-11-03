"""AGENT 02 – Fibonacci Pivot Points + 9 EMA Breakout.

Pivots are computed from the *contract's own* previous completed session
H/L/C (option premium, never the underlying index). The 9 EMA is computed on
the same premium chart.

CE (and PE when ``pe_mode = "mirror_ce"``) – bullish breakout:
  * relevant level L = the highest configured pivot level below the close
  * breakout candle = the completed candle on which ``close > L AND close > EMA9``
    becomes true (it was not true for the same L on the previous candle)
  * ENTRY = breakout candle HIGH + 2.00
  * EXIT  = premium reaches the next higher level (NEXT_PIVOT_TARGET)
            OR a completed candle closes below the 9 EMA (EMA_EXIT)

PE with ``pe_mode = "as_specified"`` (default, exactly as written in the spec):
  * relevant level L = the lowest configured pivot level above the close
  * breakdown candle = the completed candle on which ``close < L AND close < EMA9``
    becomes true
  * ENTRY = breakdown candle HIGH + 2.00 (offset is measured above the HIGH)
  * EXIT  = premium reaches the next lower support (NEXT_PIVOT_TARGET)
            OR a completed candle closes above the 9 EMA (EMA_EXIT)

Both entry conditions are mandatory; either exit condition is sufficient.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from ..config import Agent02Config
from ..indicators import EMA, FibPivots, fibonacci_pivots
from ..models import AgentState, Candle, ExitReason, OptionType, Signal, SignalStatus, SignalType
from .base import BaseStrategyAgent, ContractContext


class FibPivotEmaAgent(BaseStrategyAgent):
    agent_id = "A02"
    strategy_name = "Fibonacci Pivot Points + 9 EMA Breakout"
    initial_state = AgentState.WAITING_FOR_BREAKOUT

    def __init__(self, config: Optional[Agent02Config] = None, timeframe_min: int = 5):
        super().__init__(config or Agent02Config(), timeframe_min)
        self.config: Agent02Config
        if self.config.pe_mode not in ("as_specified", "mirror_ce"):
            raise ValueError("pe_mode must be 'as_specified' or 'mirror_ce'")

    # ----------------------------------------------------------- helpers
    def direction(self, ctx: ContractContext) -> str:
        if ctx.contract.option_type == OptionType.CE or self.config.pe_mode == "mirror_ce":
            return "UP"
        return "DOWN"

    def set_previous_session(self, key: str, high: float, low: float, close: float) -> None:
        """Explicitly inject previous-session H/L/C (e.g. from a vendor EOD file)."""
        ctx = self.contexts[key]
        piv = fibonacci_pivots(high, low, close)
        ctx.scratch["pivots"] = piv
        ctx.scratch["pivots_injected"] = piv

    # ------------------------------------------------------- indicators
    def _init_context(self, ctx: ContractContext) -> None:
        ctx.scratch.update(ema=EMA(self.config.ema_period), pivots=None, sess_hlc=None,
                           level_counts={}, cross_bars={})

    def _on_new_session(self, ctx: ContractContext, prev: Optional[date], new: date) -> None:
        hlc = ctx.scratch.get("sess_hlc")
        ctx.scratch["pivots"] = fibonacci_pivots(*hlc) if (hlc and prev is not None) else ctx.scratch.get("pivots_injected")
        ctx.scratch["sess_hlc"] = None
        ctx.scratch["level_counts"] = {}
        ctx.scratch["cross_bars"] = {}
        ctx.scratch["no_pivot_reported"] = False
        if self.config.ema_reset_daily:
            ctx.scratch["ema"].reset()

    def _update_indicators(self, ctx: ContractContext, candle: Candle) -> dict[str, Any]:
        sc = ctx.scratch
        hlc = sc.get("sess_hlc")
        sc["sess_hlc"] = (candle.high, candle.low, candle.close) if hlc is None else (
            max(hlc[0], candle.high), min(hlc[1], candle.low), candle.close)
        prev_ema = sc["ema"].value
        ema = sc["ema"].update(candle.close)
        ctx.push_series("ema9", ema)
        piv: Optional[FibPivots] = sc.get("pivots")
        ctx.push_series("pivots", piv.as_dict() if piv else None)
        # record level crosses by close (for optional lookback constraint)
        if piv and len(ctx.candles) >= 2:
            pc = ctx.candles[-2].close
            idx = len(ctx.candles) - 1
            for name, lvl in piv.sorted_levels(self.config.levels):
                if (pc <= lvl < candle.close) or (pc >= lvl > candle.close):
                    sc["cross_bars"][name] = idx
        return {"ema": ema, "prev_ema": prev_ema, "pivots": piv}

    # ------------------------------------------------------ close logic
    def _on_close(self, ctx: ContractContext, candle: Candle, info: dict[str, Any]) -> list[Signal]:
        ts = candle.end_ts
        piv: Optional[FibPivots] = info["pivots"]
        ema = info["ema"]
        st = ctx.state

        if st == AgentState.POSITION_OPEN:
            return self._check_ema_exit(ctx, candle, ema)
        if st in (AgentState.ENTRY_TRIGGERED, AgentState.EXIT_TRIGGERED):
            return []
        if piv is None:
            if not ctx.scratch.get("no_pivot_reported"):
                ctx.scratch["no_pivot_reported"] = True
                self._set_state(ctx, AgentState.NO_PIVOT_DATA, ts, "no previous session data for this contract")
                return [self._signal(ctx, SignalType.NO_TRADE, SignalStatus.REJECTED, ts,
                                     reason="No previous completed session for this contract; pivots unavailable",
                                     rejection_reason="pivots unavailable",
                                     not_satisfied=["Previous session H/L/C available"])]
            return []
        if st == AgentState.NO_PIVOT_DATA:
            self._set_state(ctx, AgentState.WAITING_FOR_BREAKOUT, ts, "pivots available")
        if ema is None or info["prev_ema"] is None or len(ctx.candles) < 2:
            return []

        out: list[Signal] = []
        event = self._detect_event(ctx, candle, ema, info["prev_ema"], piv)
        if event is not None:
            if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
                out.append(self._signal(ctx, SignalType.NO_TRADE, SignalStatus.CANCELLED, ts,
                                        reason="Pending breakout superseded by a new breakout event",
                                        rejection_reason="superseded", breakout=ctx.setup.get("breakout_candle"),
                                        event_id=ctx.setup.get("event_id")))
                ctx.setup = {}
                self._set_state(ctx, AgentState.WAITING_FOR_BREAKOUT, ts, "superseded")
            out.extend(self._arm(ctx, candle, ema, piv, *event))
            return out

        if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
            setup = ctx.setup
            setup["bars_since_breakout"] += 1
            cfg = self.config
            if cfg.entry_valid_bars is not None and setup["bars_since_breakout"] >= cfg.entry_valid_bars:
                return out + self._cancel(ctx, ts, f"entry not reached within {cfg.entry_valid_bars} bars")
            if cfg.cancel_pending_on_invalidation and not self._joint(self.direction(ctx), candle.close, ema, setup["level"]):
                return out + self._cancel(ctx, ts, "breakout invalidated before entry")
        return out

    @staticmethod
    def _joint(direction: str, close: float, ema: float, level: float) -> bool:
        if direction == "UP":
            return close > level and close > ema
        return close < level and close < ema

    def _detect_event(self, ctx, candle: Candle, ema: float, prev_ema: float, piv: FibPivots):
        direction = self.direction(ctx)
        levels = piv.sorted_levels(self.config.levels)
        close = candle.close
        prev_close = ctx.candles[-2].close
        if direction == "UP":
            below = [(n, v) for n, v in levels if v < close]
            if not below:
                return None
            name, lvl = below[-1]
        else:
            above = [(n, v) for n, v in levels if v > close]
            if not above:
                return None
            name, lvl = above[0]
        if not self._joint(direction, close, ema, lvl):
            return None
        if self._joint(direction, prev_close, prev_ema, lvl):
            return None  # already satisfied on the previous candle -> not a new breakout
        cfg = self.config
        if cfg.level_cross_lookback_bars is not None:
            idx = ctx.scratch["cross_bars"].get(name)
            if idx is None or (len(ctx.candles) - 1 - idx) > cfg.level_cross_lookback_bars:
                return None
        if cfg.max_entries_per_level_per_session is not None:
            if ctx.scratch["level_counts"].get(name, 0) >= cfg.max_entries_per_level_per_session:
                return None
        return name, lvl

    def _target(self, direction: str, piv: FibPivots, level: float, entry: float) -> tuple[Optional[str], Optional[float]]:
        levels = piv.sorted_levels(self.config.levels)
        if direction == "UP":
            ref = entry if self.config.target_reference == "entry_price" else level
            nxt = [(n, v) for n, v in levels if v > ref]
            if nxt and nxt[0][1] <= entry:
                nxt = [(n, v) for n, v in levels if v > entry]
            return nxt[0] if nxt else (None, None)
        nxt = [(n, v) for n, v in levels if v < level]
        return nxt[-1] if nxt else (None, None)

    def _arm(self, ctx, candle: Candle, ema: float, piv: FibPivots, name: str, lvl: float) -> list[Signal]:
        ts = candle.end_ts
        direction = self.direction(ctx)
        entry = round(candle.high + self.config.entry_offset, 4)
        tname, target = self._target(direction, piv, lvl, entry)
        event_id = f"A02:{ctx.key}:{ctx.session_date}:{direction}:{name}:{candle.ts.isoformat()}"
        ctx.setup = {
            "event_id": event_id,
            "breakout_candle": candle,
            "reference_candle": {"level": name, "price": lvl, "ema9": ema},
            "level_name": name,
            "level": lvl,
            "target_name": tname,
            "target": target,
            "entry_price": entry,
            "direction": direction,
            "bars_since_breakout": 0,
            "position_seed": {"target": target, "target_name": tname, "direction": direction, "level_name": name},
        }
        word = "breakout above" if direction == "UP" else "breakdown below"
        self._set_state(ctx, AgentState.BREAKOUT_IDENTIFIED, ts, f"{word} {name} {lvl:.2f} and EMA9; entry {entry:.2f}")
        self._marker(ctx, candle.ts, "breakout", candle.high, f"BO {name}" if direction == "UP" else f"BD {name}")
        cmp = ">" if direction == "UP" else "<"
        exit_desc = (f"Premium reaches {tname} {target:.2f}" if target is not None else "No further pivot level") + \
            (" OR candle close < 9 EMA" if direction == "UP" else " OR candle close > 9 EMA")
        return [self._signal(
            ctx, self._buy_type(ctx), SignalStatus.ARMED, ts,
            reason=(f"Candle {candle.ts:%H:%M} completed a {word} {name} ({lvl:.2f}) and 9 EMA ({ema:.2f}); "
                    f"entry armed at breakout candle high {candle.high:.2f} + {self.config.entry_offset:.2f} = {entry:.2f}"),
            entry_trigger=entry, breakout=candle,
            reference={"level": name, "price": lvl, "ema9": ema, "pivots": piv.as_dict()},
            target=target, exit_condition=exit_desc,
            satisfied=[f"close {candle.close:.2f} {cmp} {name} {lvl:.2f}", f"close {candle.close:.2f} {cmp} EMA9 {ema:.2f}",
                       "Breakout candle completed"],
            not_satisfied=[f"Premium must reach {entry:.2f}"],
            event_id=event_id,
            extras={"direction": direction, "pe_mode": self.config.pe_mode, "target_name": tname},
        )]

    def _cancel(self, ctx, ts: datetime, why: str) -> list[Signal]:
        sig = self._signal(ctx, SignalType.NO_TRADE, SignalStatus.REJECTED, ts, reason=f"Setup rejected: {why}",
                           rejection_reason=why, breakout=ctx.setup.get("breakout_candle"),
                           event_id=ctx.setup.get("event_id"))
        ctx.setup = {}
        self._set_state(ctx, AgentState.WAITING_FOR_BREAKOUT, ts, why)
        return [sig]

    def _check_ema_exit(self, ctx, candle: Candle, ema: Optional[float]) -> list[Signal]:
        if ema is None:
            return []
        pos = ctx.position or {}
        direction = pos.get("direction", self.direction(ctx))
        hit = candle.close < ema if direction == "UP" else candle.close > ema
        if not hit:
            return []
        ts = candle.end_ts
        rel = "<" if direction == "UP" else ">"
        self._set_state(ctx, AgentState.EXIT_TRIGGERED, ts, f"EMA_EXIT close {candle.close:.2f} {rel} EMA9 {ema:.2f}")
        return [self._signal(
            ctx, SignalType.EXIT_POSITION, SignalStatus.TRIGGERED, ts,
            reason=f"Completed candle {candle.ts:%H:%M} closed {candle.close:.2f} {rel} 9 EMA {ema:.2f}",
            exit_condition=f"CANDLE CLOSE {rel} 9 EMA", exit_reason=ExitReason.EMA_EXIT,
            satisfied=[f"close {rel} EMA9"], trigger_price=candle.close, target=pos.get("target"),
            event_id=ctx.setup.get("event_id"))]

    # ------------------------------------------------------ price logic
    def _on_price(self, ctx: ContractContext, price: float, ts: datetime) -> list[Signal]:
        if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
            setup = ctx.setup
            if price < setup["entry_price"]:
                return []
            eid = setup["event_id"]
            if eid in ctx.consumed_events:
                return []
            ctx.consumed_events.add(eid)
            counts = ctx.scratch["level_counts"]
            counts[setup["level_name"]] = counts.get(setup["level_name"], 0) + 1
            self._set_state(ctx, AgentState.ENTRY_TRIGGERED, ts, f"premium {price:.2f} >= entry {setup['entry_price']:.2f}")
            brk: Candle = setup["breakout_candle"]
            return [self._signal(
                ctx, self._buy_type(ctx), SignalStatus.TRIGGERED, ts,
                reason=f"Premium reached entry {setup['entry_price']:.2f} = breakout candle high {brk.high:.2f} + {self.config.entry_offset:.2f}",
                entry_trigger=setup["entry_price"], breakout=brk, reference=setup["reference_candle"],
                target=setup["target"], exit_condition="NEXT_PIVOT_TARGET or EMA_EXIT",
                satisfied=[f"Pivot level {setup['level_name']} broken", "9 EMA broken", "Breakout candle completed",
                           f"Premium reached breakout high + {self.config.entry_offset:.2f}"],
                event_id=eid, trigger_price=price,
                sizing_reference=max(setup["entry_price"] - brk.low, 0.0) or None,
                extras={"direction": setup["direction"], "level": setup["level_name"]},
            )]
        if ctx.state == AgentState.POSITION_OPEN:
            pos = ctx.position or {}
            tgt = pos.get("target")
            if tgt is None:
                return []
            hit = price >= tgt if pos.get("direction") == "UP" else price <= tgt
            if not hit:
                return []
            self._set_state(ctx, AgentState.EXIT_TRIGGERED, ts, f"NEXT_PIVOT_TARGET {pos.get('target_name')} {tgt:.2f}")
            return [self._signal(
                ctx, SignalType.EXIT_POSITION, SignalStatus.TRIGGERED, ts,
                reason=f"Premium {price:.2f} reached next pivot {pos.get('target_name')} {tgt:.2f}",
                exit_condition="PRICE REACHES NEXT PIVOT LEVEL", exit_reason=ExitReason.NEXT_PIVOT_TARGET,
                target=tgt, trigger_price=price, satisfied=["Next pivot level reached"],
                event_id=ctx.setup.get("event_id"))]
        return []

    def _price_levels(self, ctx: ContractContext) -> list[float]:
        if ctx.state == AgentState.BREAKOUT_IDENTIFIED:
            return [ctx.setup["entry_price"]]
        if ctx.state == AgentState.POSITION_OPEN and ctx.position and ctx.position.get("target") is not None:
            return [ctx.position["target"]]
        return []

    def _chart_extras(self, ctx: ContractContext) -> dict[str, Any]:
        piv = ctx.scratch.get("pivots")
        return {"pivots": piv.as_dict() if piv else None, "direction": self.direction(ctx)}
