import pytest

from options_mas.agents import EmaVwapBreakoutAgent
from options_mas.config import Agent01Config
from options_mas.models import AgentState, ExitReason, OptionType, SignalStatus, SignalType
from tests.helpers import D1, bar, contract, ts

CLOSES_DOWN_UP = [100, 98, 96, 94, 92, 90, 89, 95, 101, 107, 113]


def make_agent(**kw):
    return EmaVwapBreakoutAgent(Agent01Config(ema_period=3, **kw))


def feed(agent, key, closes, start=0):
    sigs = []
    for i, c in enumerate(closes, start=start):
        o = closes[i - start - 1] if i > start else c
        sigs.append(agent.on_candle_close(key, bar(D1, i, o, max(o, c) + 0.5, min(o, c) - 0.5, c)))
    return sigs


def run_to_breakout():
    a = make_agent()
    c = contract()
    a.register_contract(c)
    out = []
    i = 0
    for i, cl in enumerate(CLOSES_DOWN_UP):
        o = CLOSES_DOWN_UP[i - 1] if i else cl
        out += a.on_candle_close(c.key, bar(D1, i, o, max(o, cl) + 0.5, min(o, cl) - 0.5, cl))
        if a.contexts[c.key].state == AgentState.BREAKOUT_IDENTIFIED:
            break
    return a, c, out, i


def test_crossover_then_separate_breakout_candle_and_entry_offset():
    a, c, out, _ = run_to_breakout()
    xo = [s for s in out if s.signal_type == SignalType.WAIT and s.status == SignalStatus.INFO]
    armed = [s for s in out if s.status == SignalStatus.ARMED]
    assert len(xo) == 1 and len(armed) == 1
    ref = xo[0].reference_candle
    s = armed[0]
    # the breakout candle is a different, later candle than the crossover candle
    assert s.breakout_candle["ts"] > ref["ts"]
    assert s.breakout_candle_high > ref["high"]
    assert s.entry_trigger == pytest.approx(s.breakout_candle_high + 2.0)
    assert s.signal_type == SignalType.BUY_CE


def test_breakout_candle_must_complete_before_entry():
    a = make_agent()
    c = contract()
    a.register_contract(c)
    for i, cl in enumerate(CLOSES_DOWN_UP):
        o = CLOSES_DOWN_UP[i - 1] if i else cl
        a.on_candle_close(c.key, bar(D1, i, o, max(o, cl) + 0.5, min(o, cl) - 0.5, cl))
        if a.contexts[c.key].state == AgentState.WAITING_FOR_BREAKOUT:
            break
    # an intrabar spike far above the crossover high must not trigger anything
    assert a.on_price(c.key, 10_000.0, ts(D1, i + 1)) == []
    assert a.contexts[c.key].state == AgentState.WAITING_FOR_BREAKOUT


def test_no_entry_before_two_point_offset_then_trigger_once():
    a, c, out, i = run_to_breakout()
    entry = a.contexts[c.key].setup["entry_price"]
    assert a.on_price(c.key, entry - 0.05, ts(D1, i + 1)) == []
    sig = a.on_price(c.key, entry, ts(D1, i + 1))
    assert len(sig) == 1 and sig[0].is_entry_trigger and sig[0].trigger_price == entry
    # duplicate protection: same crossover event never triggers again
    a.on_entry_rejected(c.key, "risk", ts(D1, i + 1))
    assert a.on_price(c.key, entry + 5, ts(D1, i + 1)) == []
    assert a.contexts[c.key].state == AgentState.WAITING_FOR_CROSSOVER


def test_exit_requires_close_below_previous_low_not_wick():
    a, c, out, i = run_to_breakout()
    entry = a.contexts[c.key].setup["entry_price"]
    a.on_price(c.key, entry, ts(D1, i + 1))
    a.on_entry_filled(c.key, entry, 65, ts(D1, i + 1))
    # entry candle closes strong
    j = i + 1
    assert a.on_candle_close(c.key, bar(D1, j, entry - 1, entry + 4, entry - 1.5, entry + 3)) == []
    prev_low = entry - 1.5
    # wick far below previous low but close above it -> no exit
    assert a.on_candle_close(c.key, bar(D1, j + 1, entry + 3, entry + 4, prev_low - 10, prev_low + 0.5)) == []
    prev_low = prev_low - 10
    # close below the previous completed candle's low -> exit
    sigs = a.on_candle_close(c.key, bar(D1, j + 2, prev_low + 1, prev_low + 2, prev_low - 2, prev_low - 0.05))
    assert len(sigs) == 1 and sigs[0].is_exit_trigger
    assert sigs[0].exit_reason == ExitReason.PREV_CANDLE_LOW_CLOSE
    assert a.contexts[c.key].state == AgentState.EXIT_TRIGGERED


def test_ce_and_pe_are_independent():
    a = make_agent()
    ce, pe = contract(OptionType.CE, 24000), contract(OptionType.PE, 23900)
    a.register_contract(ce)
    a.register_contract(pe)
    for i, cl in enumerate(CLOSES_DOWN_UP):
        o = CLOSES_DOWN_UP[i - 1] if i else cl
        a.on_candle_close(ce.key, bar(D1, i, o, max(o, cl) + 0.5, min(o, cl) - 0.5, cl))
    assert a.contexts[ce.key].state != AgentState.WAITING_FOR_CROSSOVER
    assert a.contexts[pe.key].state == AgentState.WAITING_FOR_CROSSOVER
    assert a.contexts[pe.key].candles == []


def test_pe_uses_upside_breakout_and_buy_pe():
    a = make_agent()
    pe = contract(OptionType.PE)
    a.register_contract(pe)
    out = []
    for i, cl in enumerate(CLOSES_DOWN_UP):
        o = CLOSES_DOWN_UP[i - 1] if i else cl
        out += a.on_candle_close(pe.key, bar(D1, i, o, max(o, cl) + 0.5, min(o, cl) - 0.5, cl))
    armed = [s for s in out if s.status == SignalStatus.ARMED]
    assert armed and armed[0].signal_type == SignalType.BUY_PE
    assert armed[0].entry_trigger == pytest.approx(armed[0].breakout_candle_high + 2.0)


def test_new_crossover_resets_reference_candle():
    a = make_agent()
    c = contract()
    a.register_contract(c)
    i = 0
    for i, cl in enumerate(CLOSES_DOWN_UP):
        o = CLOSES_DOWN_UP[i - 1] if i else cl
        a.on_candle_close(c.key, bar(D1, i, o, max(o, cl) + 0.5, min(o, cl) - 0.5, cl))
        if a.contexts[c.key].state == AgentState.WAITING_FOR_BREAKOUT:
            break
    first_ref = a.contexts[c.key].setup["reference_candle"]
    price, out = first_ref.close, []
    for k in range(i + 1, i + 30):  # decline (never above the reference high) until EMA crosses back below VWAP
        o, price = price, price - 3
        out += a.on_candle_close(c.key, bar(D1, k, o, o + 0.1, price - 0.1, price))
        if any(s.signal_type == SignalType.WAIT for s in out):
            break
    assert any(s.status == SignalStatus.CANCELLED for s in out), "old setup must be superseded"
    new_ref = a.contexts[c.key].setup["reference_candle"]
    assert new_ref.ts > first_ref.ts
    assert a.contexts[c.key].state == AgentState.WAITING_FOR_BREAKOUT


def test_setup_expires_at_session_end():
    a, c, out, i = run_to_breakout()
    sig = a.end_session(c.key, ts(D1, 74))
    assert sig and sig[0].signal_type == SignalType.NO_TRADE
    assert a.contexts[c.key].state == AgentState.WAITING_FOR_CROSSOVER
