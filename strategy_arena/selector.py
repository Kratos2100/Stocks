"""Regression learning on top of the arena.

For every agent we fit a ridge regression that maps the current market regime
(trend strength, volatility, momentum, mean reversion, drawdown, the agent's own recent
form) to that agent's return over the next `horizon` trading days. At each decision date
the selector hands capital to the agent with the highest predicted return (or to cash if
every prediction is negative).

Everything is walk-forward: a model trained at date t only sees samples whose forward
returns had fully played out by t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import metrics, run_backtest
from .ridge import Ridge, fit_ridge_cv
from .tournament import Tournament

CASH = "Cash"


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regime descriptors known at the close of each day."""
    c = df.Close
    lr = np.log(c).diff()
    slope63, r2_63 = ind.rolling_trend(np.log(c), 63)
    vol21, vol63 = lr.rolling(21).std(), lr.rolling(63).std()
    return pd.DataFrame({
        "ret_21d": lr.rolling(21).sum(),
        "ret_63d": lr.rolling(63).sum(),
        "ret_126d": lr.rolling(126).sum(),
        "vol_21d": vol21 * np.sqrt(252),
        "vol_ratio": vol21 / vol63,
        "trend_slope_63d": slope63 * 252,
        "trend_r2_63d": r2_63,
        "efficiency_21d": ind.efficiency_ratio(c, 21),
        "autocorr_63d": lr.rolling(63).corr(lr.shift(1)),
        "dist_sma200": c / ind.sma(c, 200) - 1,
        "rsi_14": ind.rsi(c, 14),
        "drawdown_252d": c / c.rolling(252, min_periods=1).max() - 1,
    }, index=df.index)


@dataclass
class Recommendation:
    as_of: pd.Timestamp
    horizon: int
    agent: str                       # best agent for the current regime (or "Cash")
    predicted: pd.Series             # predicted forward log return per agent, best first
    drivers: list[tuple[str, float, float]] = field(default_factory=list)  # (feature, value, contribution)


@dataclass
class SelectorReport:
    picks: pd.DataFrame          # one row per decision: date, chosen agent, predicted, realized
    comparison: pd.DataFrame     # selector vs every agent over the walk-forward period
    diagnostics: dict            # rank IC, hit rate, regret...
    recommendation: Recommendation
    equity: pd.Series            # selector equity curve


class RegressionSelector:
    def __init__(self, horizon: int = 21, step: int = 5, min_train: int = 60, allow_cash: bool = True,
                 alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.horizon, self.step, self.min_train = horizon, step, min_train
        self.allow_cash, self.alphas = allow_cash, alphas

    # ---- dataset -----------------------------------------------------------------
    def _dataset(self, df: pd.DataFrame, tour: Tournament):
        names = list(tour.results)
        rets = pd.DataFrame({n: tour.results[n].returns for n in names})
        logr = np.log1p(rets)
        H = self.horizon
        # forward H-day return realized from t+1..t+H by the position set at t's close
        fwd = logr[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1)
        own_form = logr.rolling(63, min_periods=63).sum()  # each agent's trailing 63d return
        feats = market_features(df)
        start = df.index.get_loc(tour.start)
        decision_idx = np.arange(start, len(df), self.step)
        decision_idx = decision_idx[feats.iloc[decision_idx].notna().all(axis=1).to_numpy()]
        return names, rets, feats, own_form, fwd, decision_idx

    def _X(self, feats, own_form, name, idx):
        return np.column_stack([feats.to_numpy()[idx], own_form[name].to_numpy()[idx]])

    def _fit(self, feats, own_form, fwd, names, train_idx) -> dict[str, Ridge]:
        gap = int(np.ceil(self.horizon / self.step))
        models = {}
        for n in names:
            X = self._X(feats, own_form, n, train_idx)
            y = fwd[n].to_numpy()[train_idx]
            ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            models[n] = fit_ridge_cv(X[ok], y[ok], self.alphas, gap=gap)
        return models

    def _predict(self, models, feats, own_form, names, t) -> pd.Series:
        preds = {}
        for n in names:
            x = self._X(feats, own_form, n, np.array([t]))
            preds[n] = np.nan if np.isnan(x).any() else float(models[n].predict(x)[0])
        return pd.Series(preds)

    def _choose(self, pred: pd.Series) -> str:
        pred = pred.dropna()
        if pred.empty:
            return CASH
        best = pred.idxmax()
        return CASH if self.allow_cash and pred[best] <= 0 else best

    # ---- walk-forward + recommendation --------------------------------------------
    def run(self, df: pd.DataFrame, tour: Tournament, cost_bps: float = 5.0) -> SelectorReport:
        names, rets, feats, own_form, fwd, dec = self._dataset(df, tour)
        H = self.horizon
        positions = {n: tour.results[n].positions for n in names}

        fills = {n: tour.results[n].fills for n in names}
        rows, meta_pos = [], pd.Series(np.nan, index=df.index)
        meta_fill = pd.Series(np.nan, index=df.index)
        prev_choice = None
        for k, t in enumerate(dec):
            train = dec[dec + H <= t]  # forward window fully realized by day t
            if len(train) < self.min_train:
                continue
            models = self._fit(feats, own_form, fwd, names, train)
            pred = self._predict(models, feats, own_form, names, t)
            choice = self._choose(pred)
            end = dec[k + 1] if k + 1 < len(dec) else len(df)
            meta_pos.iloc[t:end] = 0.0 if choice == CASH else positions[choice].iloc[t:end].to_numpy()
            if choice != CASH and fills[choice] is not None:
                # bar t still holds the previous pick's position, so its fill only applies if unchanged
                lo = t if choice == prev_choice else t + 1
                meta_fill.iloc[lo:end] = fills[choice].iloc[lo:end].to_numpy()
            prev_choice = choice
            realized = fwd.iloc[t]
            rows.append({
                "date": df.index[t], "choice": choice,
                "predicted": 0.0 if choice == CASH else pred[choice],
                "realized": 0.0 if choice == CASH else realized[choice],
                "best_realized_agent": realized.idxmax() if realized.notna().all() else None,
                "_pred": pred, "_real": realized,
            })
        if not rows:
            raise ValueError("not enough history for the regression selector; load more years of prices")

        picks = pd.DataFrame(rows).set_index("date")
        first = picks.index[0]
        meta = run_backtest(df.Close, meta_pos.fillna(0.0), cost_bps, allow_short=True,  # already clipped
                            fills=meta_fill)

        bpy = tour.bars_per_year
        comp = {"Regression Selector": metrics(meta.returns.loc[first:], meta.positions.loc[first:], bpy)}
        for n in names:
            r = tour.results[n]
            comp[n] = metrics(r.returns.loc[first:], r.positions.loc[first:], bpy)
        comparison = pd.DataFrame(comp).T.sort_values("sharpe", ascending=False)
        comparison.index.name = "strategy"

        diagnostics = self._diagnostics(picks)
        picks = picks.drop(columns=["_pred", "_real"])

        # Final model: every sample whose outcome is known today, applied to today's regime.
        last = len(df) - 1
        train = dec[dec + H <= last]
        models = self._fit(feats, own_form, fwd, names, train)
        pred = self._predict(models, feats, own_form, names, last).sort_values(ascending=False)
        choice = self._choose(pred)
        drivers = []
        if choice != CASH:
            m = models[choice]
            x = self._X(feats, own_form, choice, np.array([last]))
            contrib = (m.standardize(x)[0] * m.coef_)
            cols = list(feats.columns) + ["own_form_63d"]
            order = np.argsort(-np.abs(contrib))[:4]
            drivers = [(cols[i], float(x[0, i]), float(contrib[i])) for i in order]
        rec = Recommendation(df.index[last], H, choice, pred, drivers)
        return SelectorReport(picks, comparison, diagnostics, rec, meta.equity.loc[first:])

    @staticmethod
    def _diagnostics(picks: pd.DataFrame) -> dict:
        ics, hits, top3, regret = [], [], [], []
        for _, row in picks.iterrows():
            p, r = row["_pred"], row["_real"]
            ok = p.notna() & r.notna()
            if ok.sum() < 3:
                continue
            p, r = p[ok], r[ok]
            ics.append(p.rank().corr(r.rank()))
            chosen = 0.0 if row["choice"] == CASH else r[row["choice"]]
            hits.append(chosen >= r.median())
            top3.append(row["choice"] in r.nlargest(3).index)
            regret.append(r.max() - chosen)
        return {
            "decisions": len(picks),
            "rank_ic": float(np.nanmean(ics)) if ics else float("nan"),
            "hit_rate_vs_median": float(np.mean(hits)) if hits else float("nan"),
            "top3_rate": float(np.mean(top3)) if top3 else float("nan"),
            "avg_regret": float(np.mean(regret)) if regret else float("nan"),
            "pick_counts": picks["choice"].value_counts().to_dict(),
        }
