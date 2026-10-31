"""Strategy Arena: tiny trading agents compete on a stock, and a ridge-regression selector
learns which agent suits which market regime."""
from .agents import Agent, default_agents
from .backtest import metrics, run_backtest
from .data import load_prices, synthetic_prices
from .selector import RegressionSelector, market_features
from .tournament import run_tournament

__all__ = ["Agent", "default_agents", "metrics", "run_backtest", "load_prices", "synthetic_prices",
           "RegressionSelector", "market_features", "run_tournament"]
