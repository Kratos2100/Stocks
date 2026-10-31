import json

import numpy as np
import pandas as pd
import pytest

from strategy_arena import default_agents, market_features, metrics, run_backtest, run_tournament, synthetic_prices
from strategy_arena.cli import analyze, main
from strategy_arena.data import from_csv
from strategy_arena.report import json_report
from strategy_arena.ridge import Ridge, fit_ridge_cv
from strategy_arena.selector import CASH


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(1200, seed=3)


@pytest.fixture(scope="module")
def analysis(prices):
    return analyze(prices, "TEST")


@pytest.mark.parametrize("agent", default_agents(), ids=lambda a: a.name)
def test_agents_do_not_look_ahead(prices, agent):
    """Changing future prices must not change any position decided before them."""
    cut = 900
    shocked = prices.copy()
    shocked.iloc[cut:] *= np.linspace(0.5, 2.0, len(prices) - cut)[:, None]
    a = agent.positions(prices).iloc[:cut]
    b = agent.positions(shocked).iloc[:cut]
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("agent", default_agents(), ids=lambda a: a.name)
def test_agent_positions_are_valid(prices, agent):
    pos = agent.positions(prices)
    assert pos.index.equals(prices.index)
    assert pos.notna().all()
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}


def test_market_features_do_not_look_ahead(prices):
    cut = 900
    shocked = prices.copy()
    shocked.iloc[cut:] *= 3
    pd.testing.assert_frame_equal(market_features(prices).iloc[:cut], market_features(shocked).iloc[:cut])


def test_backtest_trades_next_day_and_charges_costs():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    pos = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
    res = run_backtest(close, pos, cost_bps=10)
    # day 1: bought at day 0 close, earns +10%, pays 10 bps for the buy
    assert res.returns.iloc[1] == pytest.approx(0.10 - 0.001)
    # day 2: still long (sold at day 2 close), earns -10%; no trade cost yet
    assert res.returns.iloc[2] == pytest.approx(-0.10)
    # day 3: the sell is charged, no exposure
    assert res.returns.iloc[3] == pytest.approx(-0.001)
    assert res.returns.iloc[0] == 0


def test_backtest_clips_shorts_unless_allowed():
    idx = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([100.0, 90.0, 81.0], index=idx)
    pos = pd.Series(-1.0, index=idx)
    assert run_backtest(close, pos, 0).returns.sum() == 0
    assert run_backtest(close, pos, 0, allow_short=True).returns.iloc[1] == pytest.approx(0.10)


def test_metrics_on_constant_growth():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    m = metrics(r)
    assert m["total_return"] == pytest.approx(1.001**252 - 1)
    assert m["max_drawdown"] == 0
    assert m["cagr"] == pytest.approx(m["total_return"])


def test_ridge_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = 2 * X[:, 0] - 1 * X[:, 2] + 0.5 + 0.01 * rng.normal(size=500)
    m = Ridge(alpha=1e-6).fit(X, y)
    assert np.allclose(m.coef_ / m.scale_, [2, 0, -1], atol=0.01)
    assert np.allclose(m.predict(X), y, atol=0.05)
    assert fit_ridge_cv(X, y).alpha <= 1.0  # little noise -> little shrinkage


def test_tournament_leaderboard(analysis):
    agents, tour, _ = analysis
    board = tour.leaderboard
    assert set(board.index) == {a.name for a in agents}
    assert board["rounds_won"].sum() >= len(tour.rounds)  # ties can share a win
    assert board["sharpe"].is_monotonic_decreasing


def test_selector_recommends_a_known_strategy(analysis):
    agents, tour, sel = analysis
    rec = sel.recommendation
    assert rec.agent in {a.name for a in agents} | {CASH}
    assert set(rec.predicted.index) == {a.name for a in agents}
    assert "Regression Selector" in sel.comparison.index
    assert 0 <= sel.diagnostics["hit_rate_vs_median"] <= 1
    assert sel.picks.index.min() > tour.start  # needs training history before its first decision


def test_json_report_round_trips(analysis):
    _, tour, sel = analysis
    data = json.loads(json_report("TEST", tour, sel))
    assert data["ticker"] == "TEST"
    assert data["suggested_strategy"]
    assert len(data["leaderboard"]) == len(tour.leaderboard)


def test_cli_with_csv(tmp_path, prices, capsys):
    path = tmp_path / "prices.csv"
    prices.reset_index().to_csv(path, index=False)
    assert from_csv(str(path))["Close"].iloc[-1] == pytest.approx(prices["Close"].iloc[-1])
    out_json = tmp_path / "report.json"
    assert main(["DEMO", "--csv", str(path), "--start", "2000-01-01", "--json", str(out_json)]) == 0
    text = capsys.readouterr().out
    assert "Suggested strategy for DEMO" in text
    assert json.loads(out_json.read_text())["ticker"] == "DEMO"
