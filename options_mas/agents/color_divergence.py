"""AGENT 03 – Colour Divergence / Individual Candle Volume Delta.

For every completed candle on an option premium chart the agent compares the
*normal* candle colour (close vs open, never recoloured) with the volume delta
of that *same* candle:

* bearish (red) candle  + positive delta  -> BULLISH order-flow disagreement
* bullish (green) candle + negative delta -> BEARISH order-flow disagreement

This is not RSI divergence, not HH/LL divergence and not primarily CVD (CVD is
computed only as an optional secondary feature).

A colour mismatch alone does not prove absorption or guarantee reversal. The
creator-specific entry / confirmation / stop / exit rules are unverified, so
by default (``entry_rule = "none"``) the agent publishes CANDIDATE signals
only. Every enabled rule below is optional and configurable.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from ..config import Agent03Config
from ..data.volume_delta import DeltaMetrics, DeltaTracker
from ..models import AgentState, Candle, ExitReason, OptionType, Signal, SignalStatus, SignalType
from .base import BaseStrategyAgent, ContractContext

BULLISH = "BULLISH_ORDER_FLOW_DISAGREEMENT"
BEARISH = "BEARISH_ORDER_FLOW_DISAGREEMENT"
NEUTRAL = "NEUTRAL"
WEAK, MODERATE, STRONG = "WEAK", "MODERATE", "STRONG"


def classify(candle: Candle, m: Optional[DeltaMetrics], cfg: Agent03Config) -> dict[str, Any]:
    """Pure classification function (independently testable)."""
    res: dict[str, Any] = {"classification": NEUTRAL, "strength": None, "score": 0.0, "notes": []}
    if m is None:
        res["notes"].append("volume delta unavailable")
        return res
    if m.total_volume <= 0 or m.total_volume < cfg.min_total_volume:
        res["notes"].append(f"total volume {m.total_volume:.0f} below minimum {cfg.min_total_volume:.0f}")
        return res
    if not (candle.is_bullish or candle.is_bearish):
        res["notes"].append("doji candle (close == open)")
        return res
    if abs(m.delta_pct) < cfg.neutral_delta_pct:
        res["notes"].append(f"|delta%| {abs(m.delta_pct):.2f} < neutral threshold {cfg.neutral_delta_pct}")
        return res
    if candle.is_bearish and m.delta > 0:
        cls = BULLISH
    elif candle.is_bullish and m.delta < 0:
        cls = BEARISH
    else:
        res["notes"].append("candle colour and delta agree (no disagreement)")
        return res
    rel = m.relative_delta
    pct_score = min(abs(m.delta_pct) / cfg.strong_delta_pct, 1.0) if cfg.strong_delta_pct > 0 else 1.0
    rel_score = min((rel or 0.0) / cfg.strong_relative_delta, 1.0) if cfg.strong_relative_delta > 0 else 1.0
    score = round(50.0 * pct_score + 50.0 * rel_score, 2)
    weak_reasons = []
    if abs(m.delta_pct) < cfg.min_delta_pct:
        weak_reasons.append(f"|delta%| {abs(m.delta_pct):.2f} < min {cfg.min_delta_pct}")
    if rel is None or rel < cfg.min_relative_delta:
        weak_reasons.append(f"relative delta {rel if rel is None else round(rel, 3)} < min {cfg.min_relative_delta}")
    if score < cfg.min_signal_strength:
        weak_reasons.append(f"strength {score} < min {cfg.min_signal_strength}")
    if weak_reasons:
        strength = WEAK
    elif abs(m.delta_pct) >= cfg.strong_delta_pct and rel is not None and rel >= cfg.strong_relative_delta:
        strength = STRONG
    else:
        strength = MODERATE
    res.update(classification=cls, strength=strength, score=score, notes=weak_reasons)
    return res


class ColorDivergenceAgent(BaseStrategyAgent):
    agent_id = "A03"
    strategy_name = "Color Divergence – Individual Candle Volume Delta"
    initial_state = AgentState.WAITING_FOR_SIGNAL

    def __init__(self, config: Optional[Agent03Config] = None, timeframe_min: int = 5):
        super().__init__(config or Agent03Config(), timeframe_min)
        self.config: Agent03Config

    def _init_context(self, ctx: ContractContext) -> None:
        cfg = self.config
        ctx.scratch["tracker"] = DeltaTracker(cfg.avg_lookback, cfg.neutral_delta_pct, cfg.estimate_when_missing)

    def _update_indicators(self, ctx: ContractContext, candle: Candle) -> dict[str, Any]:
        m = ctx.scratch["tracker"].update(candle)
        cls = classify(candle, m, self.config)
        ctx.push_series("delta", m.delta if m else None)
        ctx.push_series("delta_pct", m.delta_pct if m else None)
        ctx.push_series("relative_delta", m.relative_delta if m else None)
        ctx.push_series("cvd", m.cvd if (m and self.config.include_cvd) else None)
        ctx.push_series("delta_source", m.source.value if m else None)
        ctx.push_series("divergence", cls["classification"])
        ctx.push_series("divergence_strength", cls["strength"])
        return {"metrics": m, "cls": cls}

    def _actionable_type(self, ctx: ContractContext, cls: str) -> Optional[SignalType]:
        ot = ctx.contract.option_type
        if self.config.mapping == "per_contract":
            return self._buy_type(ctx) if cls == BULLISH else None
        # literal mapping from the specification
        if cls == BULLISH and ot == OptionType.CE:
            return SignalType.BUY_CE
        if cls == BEARISH and ot == OptionType.PE:
            return SignalType.BUY_PE
        return None

    # ------------------------------------------------------------ close
    def _on_close(self, ctx: ContractContext, candle: Candle, info: dict[str, Any]) -> list[Signal]:
        ts = candle.end_ts
        m: Optional[DeltaMetrics] = info["metrics"]
        c = info["cls"]
        cls, strength = c["classification"], c["strength"]
        cfg = self.config
        out: list[Signal] = []
        extras = {"delta_metrics": m.to_dict() if m else None, "classification": cls, "strength": strength,
                  "signal_strength": c["score"], "notes": c["notes"]}
        if cls != NEUTRAL:
            label = ("BULL" if cls == BULLISH else "BEAR") + (f"-{strength[0]}" if strength else "")
            self._marker(ctx, candle.ts, "div_bull" if cls == BULLISH else "div_bear",
                         candle.low if cls == BULLISH else candle.high, label)

        if ctx.state == AgentState.POSITION_OPEN:
            return self._check_exits_on_close(ctx, candle, cls, strength, extras)
        if ctx.state in (AgentState.ENTRY_TRIGGERED, AgentState.EXIT_TRIGGERED):
            return out
        if ctx.state == AgentState.WAITING_FOR_ENTRY:
            ctx.setup["bars_waiting"] += 1
            if cfg.entry_valid_bars is not None and ctx.setup["bars_waiting"] >= cfg.entry_valid_bars:
                out.append(self._signal(ctx, SignalType.NO_TRADE, SignalStatus.REJECTED, ts,
                                        reason="Entry trigger not reached in time", rejection_reason="entry expired",
                                        reference=ctx.setup.get("reference_candle"), event_id=ctx.setup.get("event_id")))
                ctx.setup = {}
                self._set_state(ctx, AgentState.WAITING_FOR_SIGNAL, ts, "entry expired")
            else:
                return out

        if cls == NEUTRAL or strength == WEAK:
            return out
        stype = self._actionable_type(ctx, cls)
        if stype is None:
            out.append(self._signal(ctx, SignalType.NO_TRADE, SignalStatus.INFO, ts,
                                    reason=f"{cls} ({strength}) on {ctx.contract.option_type.value} chart: informational under mapping '{cfg.mapping}'",
                                    reference=candle, extras=extras))
            return out

        event_id = f"A03:{ctx.key}:{candle.ts.isoformat()}"
        satisfied = [
            f"price candle is {'bearish' if candle.is_bearish else 'bullish'}",
            f"individual candle delta {m.delta:+.0f} ({m.delta_pct:+.2f}%)",
            f"{'buying' if m.delta > 0 else 'selling'} volume dominates",
            f"strength {strength} (score {c['score']})",
        ]
        not_sat: list[str] = []
        est_block = m.is_estimated and not cfg.allow_estimated_delta_for_entries
        if m.is_estimated:
            not_sat.append(f"delta is ESTIMATED ({m.source.value}), not bid/ask-classified")
        if cfg.entry_rule == "none" or est_block:
            if cfg.entry_rule == "none":
                not_sat.append("entry rule not configured (creator rules unverified)")
            if est_block:
                not_sat.append("entries on estimated delta disabled")
            out.append(self._signal(ctx, stype, SignalStatus.CANDIDATE, ts,
                                    reason=f"{cls} candidate ({strength}); not actionable",
                                    reference=candle, satisfied=satisfied, not_satisfied=not_sat,
                                    event_id=event_id, extras=extras))
            return out

        stop = self._stop_price(candle, candle.high + cfg.entry_offset)
        ctx.setup = {"event_id": event_id, "reference_candle": candle, "classification": cls,
                     "bars_waiting": 0, "position_seed": {"classification": cls}}
        if cfg.entry_rule == "signal_high_break":
            entry = round(candle.high + cfg.entry_offset, 4)
            ctx.setup["entry_price"] = entry
            ctx.setup["breakout_candle"] = candle
            self._set_state(ctx, AgentState.WAITING_FOR_ENTRY, ts, f"{cls}; entry armed {entry:.2f}")
            out.append(self._signal(ctx, stype, SignalStatus.ARMED, ts,
                                    reason=f"{cls} ({strength}); entry armed at signal candle high + {cfg.entry_offset}",
                                    entry_trigger=entry, reference=candle, breakout=candle, stop_loss=stop,
                                    satisfied=satisfied, not_satisfied=not_sat + [f"premium must reach {entry:.2f}"],
                                    event_id=event_id, extras=extras))
            return out
        if cfg.entry_rule == "next_candle_open":
            ctx.consumed_events.add(event_id)
            self._set_state(ctx, AgentState.ENTRY_TRIGGERED, ts, f"{cls}; enter at next open")
            out.append(self._signal(ctx, stype, SignalStatus.TRIGGERED, ts,
                                    reason=f"{cls} ({strength}); configured entry: next candle open",
                                    entry_trigger=candle.close, reference=candle, stop_loss=self._stop_price(candle, candle.close),
                                    satisfied=satisfied, not_satisfied=not_sat, event_id=event_id,
                                    trigger_price=candle.close, extras=extras,
                                    sizing_reference=self._sizing(candle, candle.close)))
            return out
        raise ValueError(f"unknown entry_rule {cfg.entry_rule!r}")

    def _stop_price(self, candle: Candle, entry: float) -> Optional[float]:
        cfg = self.config
        if cfg.stop_rule == "signal_candle_low":
            return candle.low
        if cfg.stop_rule == "fixed_points":
            return entry - cfg.stop_value
        if cfg.stop_rule == "pct":
            return entry * (1 - cfg.stop_value / 100.0)
        return None

    def _sizing(self, candle: Candle, entry: float) -> Optional[float]:
        s = self._stop_price(candle, entry)
        return (entry - s) if (s is not None and entry > s) else None

    def _check_exits_on_close(self, ctx, candle: Candle, cls: str, strength, extras) -> list[Signal]:
        cfg = self.config
        pos = ctx.position or {}
        ts = candle.end_ts
        if candle.ts < pos.get("entry_bar_ts", candle.ts):
            return []
        pos["bars_held"] = pos.get("bars_held", 0) + 1
        reason = None
        if cfg.exit_on_opposite_disagreement and cls != NEUTRAL and strength != WEAK and cls != pos.get("classification"):
            reason = (ExitReason.OPPOSITE_DISAGREEMENT, f"opposite disagreement {cls}")
        elif cfg.exit_on_close_below_prev_low and len(ctx.candles) >= 2 and candle.close < ctx.candles[-2].low:
            reason = (ExitReason.CLOSE_BELOW_PREV_LOW, "close below previous candle low")
        elif cfg.max_holding_bars is not None and pos["bars_held"] >= cfg.max_holding_bars:
            reason = (ExitReason.TIME_EXIT, f"held {pos['bars_held']} bars")
        if reason is None:
            return []
        self._set_state(ctx, AgentState.EXIT_TRIGGERED, ts, reason[1])
        return [self._signal(ctx, SignalType.EXIT_POSITION, SignalStatus.TRIGGERED, ts, reason=reason[1],
                             exit_reason=reason[0], exit_condition=reason[0].value, trigger_price=candle.close,
                             event_id=ctx.setup.get("event_id"), extras=extras)]

    # ------------------------------------------------------------ price
    def on_entry_filled(self, key: str, price: float, qty: int, ts: datetime) -> None:
        super().on_entry_filled(key, price, qty, ts)
        ctx = self.contexts[key]
        ref = ctx.setup.get("reference_candle")
        cfg = self.config
        ctx.position["stop"] = self._stop_price(ref, price) if ref else None
        ctx.position["target"] = (price + cfg.target_points) if cfg.target_points else None

    def _on_price(self, ctx: ContractContext, price: float, ts: datetime) -> list[Signal]:
        if ctx.state == AgentState.WAITING_FOR_ENTRY and "entry_price" in ctx.setup:
            entry = ctx.setup["entry_price"]
            eid = ctx.setup["event_id"]
            if price >= entry and eid not in ctx.consumed_events:
                ctx.consumed_events.add(eid)
                self._set_state(ctx, AgentState.ENTRY_TRIGGERED, ts, f"premium {price:.2f} >= {entry:.2f}")
                ref = ctx.setup["reference_candle"]
                return [self._signal(ctx, self._buy_type(ctx), SignalStatus.TRIGGERED, ts,
                                     reason=f"Premium reached configured entry {entry:.2f}", entry_trigger=entry,
                                     reference=ref, breakout=ref, stop_loss=self._stop_price(ref, entry),
                                     event_id=eid, trigger_price=price, sizing_reference=self._sizing(ref, entry),
                                     satisfied=[ctx.setup["classification"], "configured entry trigger reached"])]
            return []
        if ctx.state == AgentState.POSITION_OPEN and ctx.position:
            stop, tgt = ctx.position.get("stop"), ctx.position.get("target")
            reason = None
            if stop is not None and price <= stop:
                reason = (ExitReason.STOP_LOSS, f"stop {stop:.2f} hit")
            elif tgt is not None and price >= tgt:
                reason = (ExitReason.TARGET, f"target {tgt:.2f} hit")
            if reason:
                self._set_state(ctx, AgentState.EXIT_TRIGGERED, ts, reason[1])
                return [self._signal(ctx, SignalType.EXIT_POSITION, SignalStatus.TRIGGERED, ts, reason=reason[1],
                                     exit_reason=reason[0], exit_condition=reason[0].value, trigger_price=price,
                                     stop_loss=stop, target=tgt, event_id=ctx.setup.get("event_id"))]
        return []

    def _price_levels(self, ctx: ContractContext) -> list[float]:
        if ctx.state == AgentState.WAITING_FOR_ENTRY and "entry_price" in ctx.setup:
            return [ctx.setup["entry_price"]]
        if ctx.state == AgentState.POSITION_OPEN and ctx.position:
            return [v for v in (ctx.position.get("stop"), ctx.position.get("target")) if v is not None]
        return []

    def _expire_pending(self, ctx, ts, why):
        if ctx.state == AgentState.WAITING_FOR_ENTRY:
            return super()._expire_pending(ctx, ts, why)
        return []
