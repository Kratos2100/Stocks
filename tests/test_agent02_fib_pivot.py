import pytest

from options_mas.agents import FibPivotEmaAgent
from options_mas.config import Agent02Config
from options_mas.models import AgentState, ExitReason, OptionType, SignalStatus, SignalType
from tests.helpers import D0, D1, bar, contract, ts

# previous session: H=120, L=80, C=100 -> P=100, R1=115.28, R2=124.72, R3=140, S1=84.72, S2=75.28
WARMUP = [bar(D0, 0, 100, 120, 80, 100), bar(D0, 1, 100, 105, 95, 100), bar(D0, 2, 100, 102, 98, 100)]


def setup(ot=OptionType.CE, k0=(100, 101, 94, 95), **kw):
    a = FibPivotEmaAgent(Agent02Config(ema_period=3, **kw))
    c = contract(ot)
    a.register_contract(c, WARMUP)
    a.on_candle_close(c.key, bar(D1, 0, *k0))
    return a, c


def test_pivots_computed_from_contracts_own_previous_session():
    a, c = setup()
    piv = a.contexts[c.key].scratch["pivots"]
    assert piv.P == pytest.approx(100)
    assert piv.R1 == pytest.approx(115.28)
    assert piv.S2 == pytest.approx(75.28)


def test_ce_requires_both_pivot_and_ema_breakout():
    a, c = setup()
    a.contexts[c.key].scratch["ema"].value = 130.0
    # closes above R1 but below the 9 EMA -> no breakout
    assert a.on_candle_close(c.key, bar(D1, 1, 110, 118, 109, 117)) == []      # ema -> 123.5
    assert a.on_candle_close(c.key, bar(D1, 2, 117, 123, 116, 122)) == []      # ema -> 122.75
    sig = a.on_candle_close(c.key, bar(D1, 3, 122, 127, 121, 126))             # ema -> 124.375; > R2
    assert len(sig) == 1 and sig[0].status == SignalStatus.ARMED and sig[0].signal_type == SignalType.BUY_CE
    s = sig[0]
    assert s.reference_candle["level"] == "R2"
    assert s.entry_trigger == pytest.approx(127 + 2.0)
    assert s.target == pytest.approx(140.0)  # next higher resistance (R3)


def test_ce_entry_and_next_pivot_target_exit():
    a, c = setup()
    a.contexts[c.key].scratch["ema"].value = 130.0
    for i, k in enumerate([(110, 118, 109, 117), (117, 123, 116, 122), (122, 127, 121, 126)], start=1):
        a.on_candle_close(c.key, bar(D1, i, *k))
    assert a.on_price(c.key, 128.95, ts(D1, 4)) == []
    sig = a.on_price(c.key, 129.0, ts(D1, 4))
    assert sig[0].is_entry_trigger
    a.on_entry_filled(c.key, 129.0, 65, ts(D1, 4))
    assert a.on_price(c.key, 139.9, ts(D1, 4)) == []
    ex = a.on_price(c.key, 140.0, ts(D1, 4))
    assert ex[0].is_exit_trigger and ex[0].exit_reason == ExitReason.NEXT_PIVOT_TARGET


def test_ce_ema_exit_on_completed_close_below_ema():
    a, c = setup()
    a.contexts[c.key].scratch["ema"].value = 130.0
    for i, k in enumerate([(110, 118, 109, 117), (117, 123, 116, 122), (122, 127, 121, 126)], start=1):
        a.on_candle_close(c.key, bar(D1, i, *k))
    a.on_price(c.key, 129.0, ts(D1, 4))
    a.on_entry_filled(c.key, 129.0, 65, ts(D1, 4))
    ex = a.on_candle_close(c.key, bar(D1, 4, 129, 130, 118, 119))  # ema = .5*119 + .5*124.375 = 121.69
    assert ex[0].is_exit_trigger and ex[0].exit_reason == ExitReason.EMA_EXIT


def test_pe_as_specified_breakdown_entry_above_high_and_exits():
    a, c = setup(OptionType.PE, k0=(100, 102, 99, 101))
    a.contexts[c.key].scratch["ema"].value = 80.0
    assert a.on_candle_close(c.key, bar(D1, 1, 101, 102, 85, 86)) == []      # close > ema (83)
    sig = a.on_candle_close(c.key, bar(D1, 2, 86, 87, 81, 82))               # < S1 and < ema 82.5
    s = sig[0]
    assert s.signal_type == SignalType.BUY_PE and s.status == SignalStatus.ARMED
    assert s.reference_candle["level"] == "S1"
    assert s.entry_trigger == pytest.approx(87 + 2.0)   # measured above the HIGH
    assert s.target == pytest.approx(75.28)             # next lower support (S2)
    assert a.on_price(c.key, 88.95, ts(D1, 3)) == []
    assert a.on_price(c.key, 89.0, ts(D1, 3))[0].is_entry_trigger
    a.on_entry_filled(c.key, 89.0, 65, ts(D1, 3))
    ex = a.on_candle_close(c.key, bar(D1, 3, 89, 95, 88, 94))               # close > 9 EMA
    assert ex[0].exit_reason == ExitReason.EMA_EXIT


def test_pe_mirror_mode_uses_bullish_logic():
    a, c = setup(OptionType.PE, pe_mode="mirror_ce")
    assert a.direction(a.contexts[c.key]) == "UP"
    a.contexts[c.key].scratch["ema"].value = 130.0
    for i, k in enumerate([(110, 118, 109, 117), (117, 123, 116, 122)], start=1):
        a.on_candle_close(c.key, bar(D1, i, *k))
    sig = a.on_candle_close(c.key, bar(D1, 3, 122, 127, 121, 126))
    assert sig[0].signal_type == SignalType.BUY_PE and sig[0].entry_trigger == pytest.approx(129)


def test_no_pivots_without_previous_session():
    a = FibPivotEmaAgent(Agent02Config(ema_period=3))
    c = contract()
    a.register_contract(c)
    sig = a.on_candle_close(c.key, bar(D1, 0, 100, 101, 99, 100))
    assert sig[0].signal_type == SignalType.NO_TRADE
    assert a.contexts[c.key].state == AgentState.NO_PIVOT_DATA


def test_breakout_event_is_not_duplicated():
    a, c = setup()
    a.contexts[c.key].scratch["ema"].value = 130.0
    for i, k in enumerate([(110, 118, 109, 117), (117, 123, 116, 122), (122, 127, 121, 126)], start=1):
        a.on_candle_close(c.key, bar(D1, i, *k))
    assert a.on_price(c.key, 129.0, ts(D1, 4))
    a.on_entry_rejected(c.key, "risk", ts(D1, 4))
    assert a.on_price(c.key, 131.0, ts(D1, 4)) == []
    # the condition remains true on the next candle -> it is NOT a new breakout
    assert a.on_candle_close(c.key, bar(D1, 4, 126, 128, 125, 127)) == []
