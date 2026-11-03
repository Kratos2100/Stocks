import asyncio
import json
from datetime import datetime

import pandas as pd
import pytest

from options_mas.backtest.engine import Backtester, ohlc_path
from options_mas.backtest.metrics import summarize
from options_mas.backtest.validation import split_sessions, walk_forward
from options_mas.config import AppConfig, load_config
from options_mas.contract_selection import ContractSelector, bs_greeks, bs_price, implied_vol
from options_mas.data.dataset import BacktestDataset, dataset_to_frame, load_dataset
from options_mas.data.synthetic import generate_synthetic_dataset
from options_mas.data.volume_delta import Trade as Tape, classify_trades_quote_rule
from options_mas.learning.bandit import StrategyBandit
from options_mas.live import BarBuilder, Tick
from options_mas.models import Candle, DeltaSource, OptionType
from tests.helpers import D1, bar


@pytest.fixture(scope="module")
def ds():
    return generate_synthetic_dataset(sessions=6, seed=3)


def test_ohlc_path_ordering():
    assert ohlc_path(bar(D1, 0, 10, 12, 9, 11)) == [10, 9, 12, 11]
    assert ohlc_path(bar(D1, 0, 10, 12, 9, 9.5)) == [10, 12, 9, 9.5]


def test_backtest_is_deterministic_and_labelled(ds):
    r1 = Backtester(AppConfig(), ds).run()
    r2 = Backtester(AppConfig(), ds).run()
    assert [t.to_dict() for t in r1.trades] == [t.to_dict() for t in r2.trades]
    assert "SYNTHETIC" in r1.data_label and any("SYNTHETIC" in w for w in r1.warnings)
    json.loads(r1.to_json())


def test_no_look_ahead_future_bars_do_not_change_past_signals(ds):
    cut = ds.sessions()[3]
    full = Backtester(AppConfig(), ds).run()
    # corrupt every bar after the cut-off session
    corrupted = {k: [c if c.ts.date() <= cut else Candle(c.ts, c.open * 3, c.high * 3, c.low * 3, c.close * 3,
                                                          c.volume, c.buy_volume, c.sell_volume, c.delta_source)
                     for c in v] for k, v in ds.candles.items()}
    ds2 = BacktestDataset(corrupted, ds.contracts, {}, ds.underlying, ds.chains, ds.label)
    part = Backtester(AppConfig(), ds2).run()

    def upto(res):
        return [(s["signal_ts"], s["agent_id"], s["contract_key"], s["signal_type"], s["status"], s["entry_trigger"])
                for s in res.signals if s["signal_ts"] < f"{cut.isoformat()}T15:30:00"]
    assert upto(full) == upto(part)
    assert [t.to_dict() for t in full.trades if t.exit_ts.date() < cut] == \
           [t.to_dict() for t in part.trades if t.exit_ts.date() < cut]


def test_entries_fill_at_trigger_or_worse_never_better(ds):
    cfg = AppConfig()
    cfg.execution.slippage_points = 0.0
    bt = Backtester(cfg, ds, enabled_agents=["A01"])
    res = bt.run()
    sigs = {s["signal_id"]: s for s in res.signals}
    for t in res.trades:
        s = sigs[t.signal_id]
        assert t.entry_price >= s["entry_trigger"] - 1e-9


def test_each_agent_can_be_tested_alone(ds):
    for a in ("A01", "A02", "A03"):
        res = Backtester(AppConfig(), ds, enabled_agents=[a]).run()
        assert res.agents == [a]
        assert all(t.agent_id == a for t in res.trades)


def test_metrics_on_known_trades():
    from options_mas.models import ExitReason, Trade
    T = datetime(2026, 6, 2, 10)
    mk = lambda pnl: Trade("t", "A01", "k", "NIFTY", OptionType.CE, 1, D1, 65, T, 1, T.replace(minute=30), 1,
                           ExitReason.EMA_EXIT, pnl, 0, pnl)
    s = summarize([mk(100), mk(-50), mk(-50), mk(200)], 1000)
    assert s["total_trades"] == 4 and s["win_rate_pct"] == 50.0 and s["net_pnl"] == 200
    assert s["profit_factor"] == 3.0 and s["expectancy"] == 50 and s["max_drawdown"] == 100
    assert s["max_consecutive_losses"] == 2 and s["avg_holding_minutes"] == 30
    assert summarize([])["total_trades"] == 0


def test_walk_forward_runs_and_oos_is_later(ds):
    wf = walk_forward(AppConfig(), ds, {"agent01.entry_offset": [1.0, 2.0]}, 3, 2, agents=["A01"])
    assert wf.folds
    for f in wf.folds:
        assert f["test"][0] > f["train"][1]
    from options_mas.backtest.validation import walk_forward_json
    json.loads(walk_forward_json(wf))
    assert isinstance(wf.ranking_table(), dict)   # still usable after serialisation
    ins, oos = split_sessions(ds.sessions(), 0.34)
    assert max(ins) < min(oos)


def test_csv_roundtrip_requires_explicit_delta_source(tmp_path, ds):
    p = tmp_path / "o.csv"
    df = dataset_to_frame(ds)
    df.drop(columns=["delta_source"]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_dataset(p, {"NIFTY": 65, "BANKNIFTY": 30})
    d2 = load_dataset(p, {"NIFTY": 65, "BANKNIFTY": 30}, default_delta_source=DeltaSource.BID_ASK_PROVIDER, label="X")
    assert len(d2.candles) == len(ds.candles)


# -------------------------------------------------------- contract selection
def chain():
    rows = []
    for k, ltp, vol, spread in ((24000, 120, 50000, 0.3), (24050, 90, 40000, 0.3), (24100, 70, 30000, 0.2),
                                (24150, 45, 100, 0.2), (24200, 21, 90000, 0.1)):
        rows.append(dict(timestamp=datetime(2026, 6, 2, 9, 20), underlying="NIFTY", spot=24010.0,
                         expiry=datetime(2026, 6, 9).date(), strike=k, option_type="CE", ltp=ltp, bid=ltp - spread / 2,
                         ask=ltp + spread / 2, volume=vol, oi=500000))
    for k, ltp in ((24000, 115), (23900, 75)):
        rows.append(dict(timestamp=datetime(2026, 6, 2, 9, 20), underlying="NIFTY", spot=24010.0,
                         expiry=datetime(2026, 6, 9).date(), strike=k, option_type="PE", ltp=ltp, bid=ltp - 6,
                         ask=ltp + 6, volume=50000, oi=500000))
    return pd.DataFrame(rows)


def test_selection_is_independent_and_not_cheapest():
    cfg = AppConfig()
    sel = ContractSelector(cfg.selection, {"NIFTY": 65}, {"NIFTY": 50})
    res = sel.select(chain(), "NIFTY", datetime(2026, 6, 2, 9, 20))
    ce = res[OptionType.CE]
    assert ce.contract is not None and ce.contract.strike != 24200   # cheapest is not chosen
    rejected = {r["strike"]: r["rejected_because"] for r in ce.rejected}
    assert any("volume" in x for x in rejected.get(24150.0, []))      # illiquid
    pe = res[OptionType.PE]
    assert pe.contract is None and all("spread" in " ".join(r["rejected_because"]) for r in pe.rejected)


def test_black_scholes_helpers():
    p = bs_price(24000, 24000, 7 / 365, 0.065, 0.14, OptionType.CE)
    iv = implied_vol(p, 24000, 24000, 7 / 365, 0.065, OptionType.CE)
    assert iv == pytest.approx(0.14, abs=1e-4)
    g = bs_greeks(24000, 24000, 7 / 365, 0.065, 0.14, OptionType.PE)
    assert -0.6 < g["delta"] < -0.4 and g["theta_per_day"] < 0


# ---------------------------------------------------------------- live
def test_bar_builder_and_quote_rule_delta():
    bb = BarBuilder(5)
    t0 = datetime(2026, 6, 2, 9, 15, 5)
    ticks = [Tick("K", t0, 100, 10, 99.5, 100), Tick("K", t0.replace(second=30), 99.5, 20, 99.5, 100),
             Tick("K", t0.replace(minute=17), 101, 5, 100.5, 101)]
    assert all(bb.on_tick(t) is None for t in ticks)
    done = bb.on_tick(Tick("K", datetime(2026, 6, 2, 9, 20, 1), 102, 1, 101.5, 102))
    assert done.ts == datetime(2026, 6, 2, 9, 15) and done.high == 101 and done.close == 101
    assert done.buy_volume == 15 and done.sell_volume == 20 and done.delta_source == DeltaSource.BID_ASK_QUOTE_RULE


def test_quote_rule_downgrades_when_quotes_missing():
    b, s, src = classify_trades_quote_rule([Tape(100, 1, 99, 100), Tape(101, 1)])
    assert src == DeltaSource.ESTIMATED_TICK_RULE


def test_streaming_runner_paper(ds):
    from options_mas.agents import build_agents
    from options_mas.execution import SimulatedBroker
    from options_mas.live import CandleReplayFeed, StreamingRunner
    from options_mas.system import TradingSystem
    cfg = AppConfig()
    cfg.execution.mode = "paper"
    s = TradingSystem(cfg, build_agents(cfg), SimulatedBroker(cfg.execution))
    day = ds.sessions()[-1]
    keys = ds.contracts_on(day)[:4]
    for k in keys:
        s.register_contract(ds.contracts[k], ds.warmup(k, day, 3))
    asyncio.run(StreamingRunner(s, CandleReplayFeed({k: ds.by_date(k)[day] for k in keys})).run())
    assert not s.execution.open_positions()
    assert s.execution.reconcile()["ok"]


def test_bandit_is_recommendation_only():
    b = StrategyBandit(["A01", "A02"], min_trades=5)
    r = b.recommend()
    assert r["ready"] is False and set(r["ranking"]) == {"A01", "A02"}


def test_default_toml_loads():
    cfg = load_config("config/default.toml")
    assert cfg.agent01.entry_offset == 2.0 and cfg.agent02.pe_mode == "as_specified"
    assert cfg.execution.live_enabled is False
