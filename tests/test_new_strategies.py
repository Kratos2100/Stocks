"""Candle-by-candle checks of the EMA + VWAP breakout and the Fibonacci pivot + 9 EMA agents."""
import numpy as np
import pandas as pd
import pytest

from strategy_arena.agents import EmaVwapBreakout, FibPivotEma, breakout_trail, pivot_trades
from strategy_arena.backtest import periods_per_year, run_backtest
from strategy_arena.cli import analyze, bar_defaults
from strategy_arena.data import from_csv, synthetic_prices
from strategy_arena.indicators import fib_pivots, vwap


def arr(*xs):
    return np.array(xs, dtype=float)


def bars(day_closes: dict, volume=1000.0):
    """Intraday frame from {date: [closes]} with 5-minute bars and a 1-point range."""
    idx, close = [], []
    for day, cs in day_closes.items():
        idx += list(pd.date_range(f"{day} 09:30", periods=len(cs), freq="5min"))
        close += cs
    c = np.array(close, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": volume},
                        index=pd.DatetimeIndex(idx, name="Date"))


class TestEmaVwapBreakout:
    # one line at 10 stands in for both max(EMA50, VWAP) and min(EMA50, VWAP)
    line = np.full(6, 10.0)

    def test_enters_on_breakout_and_exits_on_close_below_previous_low(self):
        close = arr(9, 11, 12, 13, 12.5, 11.5)
        low = arr(8.5, 10.5, 11.5, 12.6, 12.0, 11.0)
        pos = breakout_trail(close, close + 0.5, low, self.line, self.line, "low")
        # bar 4 closes 12.5, below bar 3's low of 12.6 -> exit
        assert pos.tolist() == [0, 1, 1, 1, 0, 0]

    def test_holds_while_closes_stay_above_previous_low(self):
        close = arr(9, 11, 10.8, 10.9, 11.5, 12)
        low = arr(8.5, 10.5, 10.6, 10.7, 11, 11.5)
        pos = breakout_trail(close, close + 0.5, low, self.line, self.line, "low")
        assert pos.tolist() == [0, 1, 1, 1, 1, 1]  # 10.8 dips below 11 but not below the low 10.5

    def test_close_rule_exits_on_a_lower_close(self):
        close = arr(9, 11, 12, 11.8, 13, 13)
        pos = breakout_trail(close, close + 1, close - 1, self.line, self.line, "close")
        assert pos.tolist() == [0, 1, 1, 0, 0, 0]  # no fresh breakout after the exit

    def test_needs_a_fresh_breakout_above_both_lines(self):
        close = arr(9, 11, 12)
        upper = arr(10, 11.5, 11.5)  # bar 1 clears the lower line only
        pos = breakout_trail(close, close + 1, close - 1, upper, arr(9.5, 10, 10), "low")
        assert pos.tolist() == [0, 0, 1]

    def test_short_side_mirrors(self):
        close = arr(11, 9, 8, 8.4, 8.9)
        high = arr(11.5, 9.5, 8.5, 8.8, 9.0)
        pos = breakout_trail(close, high, close - 0.5, self.line, self.line, "low")
        # 8.4 stays under bar 2's high 8.5; 8.9 closes above bar 3's high 8.8 -> cover
        assert pos.tolist() == [0, -1, -1, -1, 0]

    def test_exit_does_not_flip_to_short(self):
        close = arr(9, 11, 9.5)  # bar 2 closes below bar 1's low and below both lines
        pos = breakout_trail(close, close + 0.5, close - 0.4, np.full(3, 10.0), np.full(3, 10.0), "low")
        assert pos.tolist() == [0, 1, 0]

    def test_squares_off_at_session_end(self):
        close = arr(9, 11, 12, 13, 14)
        ends = np.array([False, False, True, False, False])
        pos = breakout_trail(close, close + 0.5, close - 0.5, np.full(5, 10.0), np.full(5, 10.0), "low", ends)
        assert pos.tolist() == [0, 1, 0, 0, 0]  # next day needs a fresh breakout above both lines

    def test_bad_exit_rule_is_rejected(self):
        with pytest.raises(ValueError):
            breakout_trail(arr(1, 2), arr(1, 2), arr(1, 2), arr(1, 1), arr(1, 1), "open")


class TestFibPivotEma:
    levels = np.tile(arr(90, 95, 100, 105, 110), (6, 1))
    ema = np.full(6, 99.0)

    def run(self, close, high=None, low=None, open_=None, ema=None):
        close = np.asarray(close, dtype=float)
        n = len(close)
        return pivot_trades(close if open_ is None else open_, close + 0.5 if high is None else high,
                            close - 0.5 if low is None else low, close,
                            self.ema[:n] if ema is None else ema, self.levels[:n])

    def test_long_breakout_takes_profit_at_next_pivot(self):
        close = arr(98, 101, 103, 104)
        high = arr(98.5, 101.5, 103.5, 106)  # bar 3 trades through the 105 target
        pos, fills = self.run(close, high=high)
        assert pos.tolist() == [0, 1, 1, 0]
        assert fills[3] == 105 and np.isnan(fills[:3]).all()

    def test_gap_through_target_fills_at_open(self):
        close = arr(98, 101, 108)
        pos, fills = self.run(close, open_=arr(98, 101, 107), high=arr(98.5, 101.5, 108.5))
        assert pos.tolist() == [0, 1, 0] and fills[2] == 107

    def test_exits_on_close_below_ema(self):
        close = arr(98, 101, 102, 100.5)
        pos, fills = self.run(close, ema=arr(99, 99, 99, 101))
        assert pos.tolist() == [0, 1, 1, 0] and np.isnan(fills).all()

    def test_exit_does_not_flip_to_short(self):
        close = arr(98, 101, 102, 98.5, 97)  # bar 3 loses the EMA (99) and breaks the 100 pivot
        pos, _ = self.run(close, ema=np.full(5, 99.0))
        assert pos.tolist() == [0, 1, 1, 0, 0]  # exit only; bar 4 crosses no pivot, so no short

    def test_short_needs_a_fresh_breakdown_after_the_exit(self):
        close = arr(98, 101, 102, 98.5, 94)  # bar 4 breaks the 95 pivot below the EMA
        pos, _ = self.run(close, ema=np.full(5, 99.0))
        assert pos.tolist() == [0, 1, 1, 0, -1]

    def test_squares_off_at_session_end_and_waits_for_a_new_breakout(self):
        close = arr(98, 101, 102, 103, 103.5)
        ends = np.array([False, False, True, False, False])
        pos, fills = pivot_trades(close, close + 0.5, close - 0.5, close, np.full(5, 99.0), self.levels[:5], ends)
        assert pos.tolist() == [0, 1, 0, 0, 0]  # still above the EMA next day, but no fresh breakout

    def test_breakout_below_ema_is_ignored(self):
        pos, _ = self.run(arr(98, 101, 102), ema=np.full(3, 101.5))
        assert pos.tolist() == [0, 0, 0]

    def test_short_breakdown_mirrors(self):
        close = arr(101, 99, 97)
        low = arr(100.5, 98.5, 94.5)  # target is the next pivot down, 95
        pos, fills = self.run(close, low=low, ema=np.full(3, 100.0))
        assert pos.tolist() == [0, -1, 0] and fills[2] == 95

    def test_no_reentry_on_the_bar_that_hit_the_target(self):
        close = arr(98, 101, 106)  # bar 2 hits 105 and also closes above it
        pos, fills = self.run(close, open_=arr(98, 101, 103), high=arr(98.5, 101.5, 106.5))
        assert pos[2] == 0 and fills[2] == 105


def test_backtest_uses_the_fill_price_on_exit_bars():
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.Series([100.0, 101.0, 104.0], index=idx)
    pos = pd.Series([1.0, 0.0, 0.0], index=idx)
    fills = pd.Series([np.nan, 102.0, np.nan], index=idx)
    res = run_backtest(close, pos, cost_bps=0, fills=fills)
    assert res.returns.iloc[1] == pytest.approx(0.02)  # sold at the 102 target, not the 101 close
    assert res.returns.iloc[2] == 0


def test_session_vwap_resets_each_day():
    df = bars({"2024-03-04": [100, 102, 104], "2024-03-05": [90, 92]})
    v = vwap(df)
    assert v.iloc[0] == pytest.approx(100)
    assert v.iloc[2] == pytest.approx(102)  # equal volume: mean of 100, 102, 104
    assert v.iloc[3] == pytest.approx(90)   # new session starts over


def test_vwap_without_volume_uses_equal_weights():
    df = bars({"2024-03-04": [100, 110]}, volume=0.0)
    assert vwap(df).iloc[1] == pytest.approx(105)


def test_vwap_weights_by_volume():
    df = bars({"2024-03-04": [100, 110]})
    df["Volume"] = [3000.0, 1000.0]
    assert vwap(df).iloc[1] == pytest.approx((100 * 3 + 110) / 4)


def test_fib_pivots_come_from_the_previous_session():
    df = bars({"2024-03-04": [95, 109, 100], "2024-03-05": [101, 102]})
    lv = fib_pivots(df)
    assert lv.iloc[:3].isna().all().all()  # no previous session yet
    # previous session: H = 110, L = 94, C = 100 -> P = 101.33, range 16
    p = (110 + 94 + 100) / 3
    assert lv.iloc[3]["P"] == pytest.approx(p)
    assert lv.iloc[3]["R1"] == pytest.approx(p + 0.382 * 16)
    assert lv.iloc[3]["S2"] == pytest.approx(p - 0.618 * 16)
    assert lv.iloc[4]["R3"] == pytest.approx(p + 16)


def test_daily_bars_use_previous_week_pivots():
    df = synthetic_prices(15, seed=1)
    lv = fib_pivots(df)
    week2 = df.loc["2015-01-05":"2015-01-09"]
    prev = df.loc[:"2015-01-04"]  # the first (partial) week
    p = (prev.High.max() + prev.Low.min() + prev.Close.iloc[-1]) / 3
    assert np.allclose(lv.loc[week2.index, "P"], p)


def test_agents_run_on_intraday_bars_without_look_ahead():
    df = synthetic_prices(78 * 30, seed=2, bars_per_day=78)
    cut = 78 * 20 + 10
    shocked = df.copy()
    shocked.iloc[cut:] *= 1.5
    for agent in (EmaVwapBreakout(), FibPivotEma()):
        pos, fills = agent.trade(df)
        pos2, fills2 = agent.trade(shocked)
        pd.testing.assert_series_equal(pos.iloc[:cut], pos2.iloc[:cut])
        if fills is not None:
            pd.testing.assert_series_equal(fills.iloc[:cut], fills2.iloc[:cut])
        assert (pos != 0).any(), f"{agent.name} never traded"


def test_intraday_pipeline_and_annualization():
    df = synthetic_prices(78 * 40, seed=4, bars_per_day=78)
    assert periods_per_year(df.index) == 252 * 78
    assert bar_defaults(df) == {"round_len": 390, "horizon": 78, "step": 13}
    agents, tour, sel = analyze(df, "INTRA", allow_short=True)
    assert {"EMA50 + VWAP Breakout", "Fib Pivot + EMA9"} <= set(tour.leaderboard.index)
    assert sel.recommendation.horizon == 78


def test_csv_with_timezone_keeps_exchange_time(tmp_path):
    path = tmp_path / "nifty_5m.csv"
    path.write_text("Datetime,Open,High,Low,Close,Volume\n"
                    "2024-03-04 09:15:00+05:30,100,101,99,100.5,0\n"
                    "2024-03-04 09:20:00+05:30,100.5,102,100,101.5,0\n")
    df = from_csv(str(path))
    assert df.index[0] == pd.Timestamp("2024-03-04 09:15")


def test_sessions_split_on_trading_breaks_not_midnight():
    from strategy_arena.indicators import bars_per_session, session_ends, session_ids

    # CME-style 15-minute bars: 18:00 -> 17:00 with a one-hour halt, crossing midnight
    idx = pd.date_range("2026-03-01 18:00", "2026-03-04 16:45", freq="15min")
    idx = idx[idx.hour != 17]
    assert len(np.unique(session_ids(idx))) == 3
    assert bars_per_session(idx) == 92
    assert [t.strftime("%m-%d %H:%M") for t in idx[session_ends(idx)]] == ["03-02 16:45", "03-03 16:45", "03-04 16:45"]
    daily = pd.bdate_range("2024-01-01", periods=30)
    assert bars_per_session(daily) == 1 and len(np.unique(session_ids(daily))) == 30


def test_square_off_makes_every_agent_flat_overnight():
    from strategy_arena.agents import default_agents
    from strategy_arena.indicators import session_ends

    df = synthetic_prices(78 * 10, seed=5, bars_per_day=78)
    ends = session_ends(df.index)
    for agent in default_agents(square_off=True):
        pos, _ = agent.trade(df)
        assert (pos[ends] == 0).all(), agent.name
    held_overnight = [a.name for a in default_agents(square_off=False) if (a.trade(df)[0][ends] != 0).any()]
    assert "Buy & Hold" in held_overnight


def test_square_off_is_ignored_on_daily_bars():
    from strategy_arena.agents import BuyAndHold

    agent = BuyAndHold()
    agent.square_off = True
    assert (agent.trade(synthetic_prices(50))[0] == 1).all()


def test_excess_tstat_detects_a_real_edge_only():
    from strategy_arena.backtest import excess_tstat

    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(2000)
    b = pd.Series(rng.normal(0, 0.01, 2000), index=idx)
    assert excess_tstat(b + 0.002 + rng.normal(0, 0.001, 2000), b, 20) > 5
    assert abs(excess_tstat(b + rng.normal(0, 0.01, 2000), b, 20)) < 3
    assert np.isnan(excess_tstat(b.iloc[:30], b.iloc[:30], 20))  # fewer than 3 blocks


def test_intraday_report_warns_about_short_history_and_shows_costs():
    from strategy_arena.report import text_report

    df = synthetic_prices(78 * 40, seed=6, bars_per_day=78)
    agents, tour, sel = analyze(df, "INTRA", cost_bps=2, allow_short=True)
    text = text_report("INTRA", df, agents, tour, sel, 2, True, square_off=True)
    assert "squared off at each session's last bar" in text
    assert "WARNING: only" in text
    assert "Costs" in text and "vs Buy & Hold t =" in text
    assert (tour.leaderboard["costs"] >= 0).all()
    assert tour.leaderboard.loc["Buy & Hold", "trades"] > 0  # re-buys every session when squaring off
