from datetime import datetime

from options_mas.agents import build_agents
from options_mas.config import AppConfig
from options_mas.models import OptionType, Signal, SignalStatus, SignalType, Trade, ExitReason
from options_mas.system import TradingSystem
from tests.helpers import contract

T = datetime(2026, 6, 2, 10, 0)


def system(**orch):
    cfg = AppConfig()
    cfg.execution.max_participation_pct = None
    for k, v in orch.items():
        setattr(cfg.orchestrator, k, v)
    s = TradingSystem(cfg, build_agents(cfg))
    for ot in (OptionType.CE, OptionType.PE):
        s.register_contract(contract(ot))
    return s


def trig(agent, ot, event, ts=T, und="NIFTY"):
    c = contract(ot)
    return Signal(agent, "x", "NSE", und, ot, c.strike, c.expiry, "5m",
                  SignalType.BUY_CE if ot == OptionType.CE else SignalType.BUY_PE, SignalStatus.TRIGGERED, ts,
                  c.key, entry_trigger=100, trigger_price=100, event_id=event)


def _force_trigger_state(s, agent, key):
    from options_mas.models import AgentState
    s.agents[agent].contexts[key].state = AgentState.ENTRY_TRIGGERED


def test_single_agent_can_trade_without_consensus():
    s = system()
    k = contract().key
    _force_trigger_state(s, "A01", k)
    d = s.orchestrator.process([trig("A01", OptionType.CE, "e1")], T, "intrabar", {k: 100})
    assert d[0].action == "SUBMITTED" and d[0].consensus == []
    assert s.execution.position_for("A01", k)


def test_duplicate_event_and_duplicate_position_rejected():
    s = system()
    k = contract().key
    _force_trigger_state(s, "A01", k)
    s.orchestrator.process([trig("A01", OptionType.CE, "e1")], T, "intrabar", {k: 100})
    d = s.orchestrator.process([trig("A01", OptionType.CE, "e1")], T, "intrabar", {k: 100})
    assert d[0].action == "REJECTED" and "DUPLICATE_EVENT" in d[0].reasons[0]
    _force_trigger_state(s, "A02", k)
    d = s.orchestrator.process([trig("A02", OptionType.CE, "e2")], T, "intrabar", {k: 100})
    assert "held by A01" in d[0].reasons[0]


def test_conflict_detection_policies():
    s = system(conflict_policy="block_both")
    ce, pe = contract(OptionType.CE).key, contract(OptionType.PE).key
    d = s.orchestrator.process([trig("A01", OptionType.CE, "a"), trig("A02", OptionType.PE, "b")], T, "intrabar",
                               {ce: 100, pe: 100})
    assert all(x.action == "REJECTED" and x.conflict for x in d)
    s2 = system()  # default: allow and flag
    for a, k in (("A01", ce), ("A02", pe)):
        _force_trigger_state(s2, a, k)
    d = s2.orchestrator.process([trig("A01", OptionType.CE, "a"), trig("A02", OptionType.PE, "b")], T, "intrabar",
                                {ce: 100, pe: 100})
    assert all(x.action == "SUBMITTED" and x.conflict for x in d)
    assert len(s2.orchestrator.conflicts) == 2


def test_consensus_is_analytics_only():
    s = system(mode="B")
    k = contract().key
    armed = trig("A02", OptionType.CE, "z")
    armed.status = SignalStatus.ARMED
    _force_trigger_state(s, "A01", k)
    d = s.orchestrator.process([armed, trig("A01", OptionType.CE, "e")], T, "intrabar", {k: 100})
    sub = [x for x in d if x.action == "SUBMITTED"][0]
    assert sub.consensus == ["A02"]


def test_mode_c_ranks_by_out_of_sample_score():
    s = system(mode="C", ranking_min_trades=1, ranking_prior_trades=0)
    c = contract()
    for agent, pnl in (("A01", -500.0), ("A02", 800.0)):
        s.ranker.record(Trade("t", agent, c.key, "NIFTY", OptionType.CE, c.strike, c.expiry, 65, T, 100, T, 101,
                              ExitReason.EMA_EXIT, pnl, 0, pnl))
    s.cfg.risk.max_concurrent_positions = 1
    ce, pe = contract(OptionType.CE).key, contract(OptionType.PE).key
    for a, k in (("A01", ce), ("A02", pe)):
        _force_trigger_state(s, a, k)
    d = s.orchestrator.process([trig("A01", OptionType.CE, "a"), trig("A02", OptionType.PE, "b")], T, "intrabar",
                               {ce: 100, pe: 100})
    assert d[0].agent_id == "A02" and d[0].action == "SUBMITTED"
    assert d[1].action == "REJECTED"
