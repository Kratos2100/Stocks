from datetime import datetime

import pytest

from options_mas.config import AppConfig
from options_mas.execution import (ChargesCalculator, ExecutionCallbacks, ExecutionEngine, IndianBrokerAdapterTemplate,
                                   LiveTradingNotAuthorized, SimulatedBroker)
from options_mas.models import (ExitReason, OptionType, OrderSide, OrderStatus, Position, Signal, SignalStatus,
                                SignalType)
from options_mas.risk import RiskEngine
from tests.helpers import contract

T = datetime(2026, 6, 2, 10, 0)


def sig(agent="A01", ot=OptionType.CE, key=None, stop=None, event="e1"):
    c = contract(ot)
    return Signal(agent, "s", "NSE", "NIFTY", ot, c.strike, c.expiry, "5m",
                  SignalType.BUY_CE if ot == OptionType.CE else SignalType.BUY_PE, SignalStatus.TRIGGERED, T,
                  key or c.key, entry_trigger=100, stop_loss=stop, event_id=event, trigger_price=100)


# ----------------------------------------------------------------- risk
def test_position_sizing_uses_risk_budget_and_lot_size():
    cfg = AppConfig()
    cfg.risk.capital = 500_000
    cfg.risk.risk_per_trade_pct = 1.0         # 5,000 rupees
    r = RiskEngine(cfg)
    d = r.evaluate_entry(sig(stop=80), 100, T, [], 65)   # 20/unit * 65 = 1300/lot -> 3 lots
    assert d.approved and d.lots == 3 and d.qty == 195


def test_entry_window_and_limits():
    cfg = AppConfig()
    cfg.risk.max_trades_per_day = 1
    r = RiskEngine(cfg)
    assert not r.evaluate_entry(sig(), 100, datetime(2026, 6, 2, 15, 5), [], 65).approved  # after 15:00
    assert r.evaluate_entry(sig(), 100, T, [], 65).approved
    d = r.evaluate_entry(sig(), 100, T, [], 65)
    assert not d.approved and "max trades" in d.reasons[0]


def test_max_concurrent_and_emergency_stop():
    cfg = AppConfig()
    cfg.risk.max_concurrent_positions = 1
    r = RiskEngine(cfg)
    pos = Position("P1", "A02", contract(), 65, 100, T)
    assert "concurrent" in r.evaluate_entry(sig(), 100, T, [pos], 65).reasons[0]
    r.emergency_stop("test")
    assert "EMERGENCY" in r.evaluate_entry(sig(), 100, T, [], 65).reasons[0]


def test_daily_loss_halts_and_requests_flatten():
    cfg = AppConfig()
    cfg.risk.max_daily_loss = 1000
    r = RiskEngine(cfg)
    pos = Position("P1", "A01", contract(), 65, 100, T, last_price=80)  # -1300 unrealised
    assert ExitReason.MAX_DAILY_LOSS in r.update_unrealized(T, [pos])
    assert not r.evaluate_entry(sig(), 100, T, [], 65).approved


def test_charges_breakdown():
    ch = ChargesCalculator()
    buy = ch.compute(OrderSide.BUY, 100, 65)
    sell = ch.compute(OrderSide.SELL, 100, 65)
    assert buy.stt == 0 and sell.stt == pytest.approx(6.5)
    assert buy.stamp == pytest.approx(6500 * 0.00003)
    assert buy.gst == pytest.approx((20 + 6500 * 0.0003503 + 6500 * 1e-6) * 0.18)


# ------------------------------------------------------------ execution
def engine(mode="backtest", **ex):
    cfg = AppConfig()
    cfg.execution.mode = mode
    for k, v in ex.items():
        setattr(cfg.execution, k, v)
    events = {"filled": [], "rejected": [], "exited": []}
    cb = ExecutionCallbacks(entry_filled=lambda *a: events["filled"].append(a),
                            entry_rejected=lambda *a: events["rejected"].append(a),
                            exit_filled=lambda *a: events["exited"].append(a))
    return ExecutionEngine(cfg, SimulatedBroker(cfg.execution), callbacks=cb), events


def test_entry_fill_with_slippage_and_exit_creates_trade():
    ex, ev = engine(slippage_points=0.5, max_participation_pct=None)
    s = sig()
    o = ex.submit_entry(s, contract(), 65, T, 100.0, "intrabar")
    assert o.status == OrderStatus.FILLED and o.avg_fill_price == pytest.approx(100.5)
    assert ev["filled"]
    ex.submit_exit("A01", contract().key, ExitReason.PREV_CANDLE_LOW_CLOSE, T, 110.0, "intrabar")
    t = ex.trades[0]
    assert t.exit_price == pytest.approx(109.5) and t.gross_pnl == pytest.approx(9.0 * 65)
    assert t.net_pnl < t.gross_pnl and t.exit_reason == ExitReason.PREV_CANDLE_LOW_CLOSE


def test_close_signals_fill_at_next_open():
    ex, ev = engine(slippage_points=0.0, max_participation_pct=None)
    ex.submit_entry(sig(), contract(), 65, T, 100.0, "intrabar")
    o = ex.submit_exit("A01", contract().key, ExitReason.EMA_EXIT, T, 90.0, "close")
    assert o.status == OrderStatus.SUBMITTED  # queued
    ex.on_price(contract().key, 88.0, T)       # next bar open
    assert ex.trades[0].exit_price == pytest.approx(88.0)


def test_partial_fill_entry_remainder_cancelled_exit_keeps_working():
    ex, ev = engine(slippage_points=0.0, max_participation_pct=10.0)
    k = contract().key
    ex.broker.set_liquidity(k, 1300)            # cap = 130 qty = 2 lots
    o = ex.submit_entry(sig(), contract(), 260, T, 100.0, "intrabar")
    assert o.filled_qty == 130 and o.status == OrderStatus.CANCELLED
    assert ex.position_for("A01", k).qty == 130
    ex.broker.set_liquidity(k, 650)             # 65 qty
    x = ex.submit_exit("A01", k, ExitReason.EMA_EXIT, T, 101.0, "intrabar")
    assert x.status == OrderStatus.PARTIALLY_FILLED
    ex.broker.set_liquidity(k, 650)
    ex.on_price(k, 102.0, T)
    assert x.status == OrderStatus.FILLED and len(ex.trades) == 1


def test_rejection_for_non_lot_multiple():
    ex, ev = engine()
    o = ex.submit_entry(sig(), contract(), 50, T, 100.0, "intrabar")
    assert o.status == OrderStatus.REJECTED and ev["rejected"]


def test_manual_approval_flow():
    ex, ev = engine("manual", max_participation_pct=None)
    o = ex.submit_entry(sig(), contract(), 65, T, 100.0, "intrabar")
    assert o.status == OrderStatus.PENDING_APPROVAL and not ev["filled"]
    ex.on_price(contract().key, 100.0, T)
    ex.approve(o.order_id, T)
    assert o.status == OrderStatus.FILLED


def test_reconcile_detects_mismatch():
    ex, _ = engine(max_participation_pct=None)
    ex.submit_entry(sig(), contract(), 65, T, 100.0, "intrabar")
    assert ex.reconcile()["ok"]
    ex.broker._positions[contract().key] = 130
    rep = ex.reconcile()
    assert not rep["ok"] and rep["mismatches"][0]["broker_qty"] == 130


def test_live_trading_disabled_by_default(monkeypatch):
    cfg = AppConfig()
    assert cfg.execution.live_enabled is False
    with pytest.raises(LiveTradingNotAuthorized):
        IndianBrokerAdapterTemplate(cfg)
    cfg.execution.mode = "live"
    with pytest.raises(LiveTradingNotAuthorized):
        ExecutionEngine(cfg, SimulatedBroker(cfg.execution))
    cfg.execution.live_enabled = True
    monkeypatch.setenv("OPTIONS_MAS_LIVE_AUTHORIZATION", "I_AUTHORIZE_LIVE_TRADING")
    with pytest.raises(LiveTradingNotAuthorized):   # still needs the per-run confirmation
        ExecutionEngine(cfg, SimulatedBroker(cfg.execution))
    ExecutionEngine(cfg, SimulatedBroker(cfg.execution), live_confirmed=True)
