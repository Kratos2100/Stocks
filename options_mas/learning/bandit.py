"""PHASE 9 research component – contextual strategy-selection bandit.

Recommendation-only: the bandit can *suggest* which agent to prioritise in a
market regime (its output can feed Mode C ranking), but it never places
orders, never bypasses the risk engine and never changes strategy rules.
Disabled by default; use only after enough real trade data exists.

Model: per (regime, agent) Gaussian Thompson sampling on R-multiples (or net
P&L normalised by a scale), with a Normal-Inverse-Gamma style shrinkage.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..models import Trade


@dataclass
class _Arm:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 1.0


@dataclass
class StrategyBandit:
    agents: list[str]
    min_trades: int = 30
    prior_strength: float = 10.0
    pnl_scale: float = 1000.0
    seed: int = 0
    arms: dict[tuple[str, str], _Arm] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _reward(self, t: Trade) -> float:
        r = t.r_multiple
        return r if r is not None else t.net_pnl / self.pnl_scale

    def fit(self, trades: Iterable[Trade]) -> None:
        for t in trades:
            for reg in ("ALL", t.regime):
                self.arms.setdefault((reg, t.agent_id), _Arm()).update(self._reward(t))

    def recommend(self, regime: str = "ALL") -> dict:
        """Returns a ranking with sampled scores. ``ready`` is False until every
        agent has ``min_trades`` observations – consumers should ignore the
        ranking while not ready."""
        scores = {}
        ready = True
        for a in self.agents:
            arm = self.arms.get((regime, a)) or self.arms.get(("ALL", a)) or _Arm()
            if arm.n < self.min_trades:
                ready = False
            k = self.prior_strength
            post_mean = arm.n * arm.mean / (arm.n + k)
            post_sd = math.sqrt(arm.var / (arm.n + k))
            scores[a] = self._rng.gauss(post_mean, post_sd)
        ranking = sorted(scores, key=lambda a: -scores[a])
        return {"regime": regime, "ready": ready, "ranking": ranking,
                "sampled_scores": {a: round(s, 4) for a, s in scores.items()},
                "note": "Recommendation only. Hard risk limits and strategy rules are never overridden."}
