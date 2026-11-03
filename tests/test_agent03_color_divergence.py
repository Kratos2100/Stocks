import pytest

from options_mas.agents import ColorDivergenceAgent
from options_mas.agents.color_divergence import BEARISH, BULLISH, NEUTRAL, STRONG, WEAK, classify
from options_mas.config import Agent03Config
from options_mas.data.volume_delta import DeltaTracker
from options_mas.models import AgentState, DeltaSource, OptionType, SignalStatus, SignalType
from tests.helpers import D1, bar, contract, ts

P = DeltaSource.BID_ASK_PROVIDER


def metrics_for(candles, cfg=None):
    cfg = cfg or Agent03Config()
    t = DeltaTracker(cfg.avg_lookback, cfg.neutral_delta_pct, cfg.estimate_when_missing)
    return [t.update(c) for c in candles][-1]


def baseline(n=5):
    return [bar(D1, i, 100, 101, 99, 100.5, 1000, 520, 480, P) for i in range(n)]


def test_red_candle_with_positive_delta_is_bullish_disagreement():
    c = bar(D1, 5, 100, 101, 95, 96, 1000, 800, 200, P)
    m = metrics_for(baseline() + [c])
    assert m.delta == 600 and m.delta_pct == pytest.approx(60)
    assert m.relative_delta == pytest.approx(600 / 40)  # baseline |delta| = 40 (previous bars only)
    r = classify(c, m, Agent03Config())
    assert r["classification"] == BULLISH and r["strength"] == STRONG


def test_green_candle_with_negative_delta_is_bearish_disagreement():
    c = bar(D1, 5, 100, 106, 99, 105, 1000, 300, 700, P)
    r = classify(c, metrics_for(baseline() + [c]), Agent03Config())
    assert r["classification"] == BEARISH


def test_agreement_doji_and_small_delta_are_neutral():
    cfg = Agent03Config()
    agree = bar(D1, 5, 100, 106, 99, 105, 1000, 700, 300, P)
    doji = bar(D1, 5, 100, 106, 99, 100, 1000, 300, 700, P)
    small = bar(D1, 5, 100, 106, 99, 105, 1000, 490, 510, P)  # |delta%| = 2 < neutral 5
    for c in (agree, doji, small):
        assert classify(c, metrics_for(baseline() + [c]), cfg)["classification"] == NEUTRAL


def test_weak_disagreement_below_minimums():
    c = bar(D1, 5, 100, 101, 95, 96, 1000, 540, 460, P)  # delta% 8 < min 10
    r = classify(c, metrics_for(baseline() + [c]), Agent03Config())
    assert r["classification"] == BULLISH and r["strength"] == WEAK


def test_missing_order_flow_is_labelled_estimated():
    c = bar(D1, 0, 100, 110, 90, 108, 1000)  # no buy/sell split
    m = metrics_for([c])
    assert m.source == DeltaSource.ESTIMATED_BAR_CLV and m.is_estimated
    assert m.buy_volume == pytest.approx(900)


def test_default_config_only_publishes_candidates():
    a = ColorDivergenceAgent(Agent03Config())
    c = contract(OptionType.CE)
    a.register_contract(c, baseline())
    sig = a.on_candle_close(c.key, bar(D1, 5, 100, 101, 95, 96, 1000, 800, 200, P))
    assert sig[0].signal_type == SignalType.BUY_CE and sig[0].status == SignalStatus.CANDIDATE
    assert not sig[0].is_entry_trigger
    assert any("not configured" in x for x in sig[0].conditions_not_satisfied)


def test_estimated_delta_never_enters_by_default():
    a = ColorDivergenceAgent(Agent03Config(entry_rule="signal_high_break"))
    c = contract(OptionType.CE)
    warm = [bar(D1, i, 100, 101, 99, 100.5, 1000) for i in range(5)]
    a.register_contract(c, warm)
    sig = a.on_candle_close(c.key, bar(D1, 5, 100, 101, 80, 99, 5000))  # long lower wick -> +est. delta
    assert sig and sig[0].status == SignalStatus.CANDIDATE
    assert sig[0].extras["delta_metrics"]["delta_is_estimated"] is True


def test_configured_entry_literal_mapping_ce_and_pe():
    cfg = Agent03Config(entry_rule="signal_high_break", stop_rule="signal_candle_low", exit_on_opposite_disagreement=True)
    a = ColorDivergenceAgent(cfg)
    ce, pe = contract(OptionType.CE), contract(OptionType.PE)
    a.register_contract(ce, baseline())
    a.register_contract(pe, baseline())
    s = a.on_candle_close(ce.key, bar(D1, 5, 100, 101, 95, 96, 1000, 800, 200, P))[0]
    assert s.status == SignalStatus.ARMED and s.entry_trigger == pytest.approx(103) and s.stop_loss == 95
    # literal mapping: a bullish disagreement on the PE chart is informational only
    s2 = a.on_candle_close(pe.key, bar(D1, 5, 100, 101, 95, 96, 1000, 800, 200, P))[0]
    assert s2.signal_type == SignalType.NO_TRADE
    # bearish disagreement on the PE chart -> BUY_PE
    s3 = a.on_candle_close(pe.key, bar(D1, 6, 96, 106, 95, 105, 1000, 200, 800, P))[0]
    assert s3.signal_type == SignalType.BUY_PE and s3.status == SignalStatus.ARMED
    # CE entry + stop
    t = a.on_price(ce.key, 103, ts(D1, 6))[0]
    assert t.is_entry_trigger
    a.on_entry_filled(ce.key, 103, 65, ts(D1, 6))
    ex = a.on_price(ce.key, 95, ts(D1, 6))
    assert ex[0].is_exit_trigger and ex[0].exit_reason.value == "STOP_LOSS"


def test_ce_pe_states_independent():
    a = ColorDivergenceAgent(Agent03Config(entry_rule="signal_high_break"))
    ce, pe = contract(OptionType.CE), contract(OptionType.PE, 23900)
    a.register_contract(ce, baseline())
    a.register_contract(pe, baseline())
    a.on_candle_close(ce.key, bar(D1, 5, 100, 101, 95, 96, 1000, 800, 200, P))
    assert a.contexts[ce.key].state == AgentState.WAITING_FOR_ENTRY
    assert a.contexts[pe.key].state == AgentState.WAITING_FOR_SIGNAL
