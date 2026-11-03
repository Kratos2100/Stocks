"""Strategy agents. Each one is a deterministic, independently testable worker."""
from .base import BaseStrategyAgent, ContractContext
from .color_divergence import ColorDivergenceAgent
from .ema_vwap import EmaVwapBreakoutAgent
from .fib_pivot import FibPivotEmaAgent


def build_agents(cfg, enabled=None):
    """Instantiate the enabled agents from an :class:`AppConfig`."""
    tf = cfg.session.timeframe_min
    agents = {}
    for cls, sub in ((EmaVwapBreakoutAgent, cfg.agent01), (FibPivotEmaAgent, cfg.agent02), (ColorDivergenceAgent, cfg.agent03)):
        if enabled is not None and cls.agent_id not in enabled:
            continue
        if enabled is None and not sub.enabled:
            continue
        agents[cls.agent_id] = cls(sub, tf)
    return agents


__all__ = [
    "BaseStrategyAgent", "ContractContext", "EmaVwapBreakoutAgent", "FibPivotEmaAgent",
    "ColorDivergenceAgent", "build_agents",
]
