"""Typed configuration with safe defaults.

Configuration is loaded from TOML (``config/default.toml``) and merged over the
dataclass defaults below. Every strategy parameter that the specification
leaves open is exposed here instead of being hard-coded.
"""
from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import time
from pathlib import Path
from typing import Any, Optional


def _t(s: str) -> time:
    h, m = s.split(":")[:2]
    return time(int(h), int(m))


@dataclass
class SessionConfig:
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    timeframe_min: int = 5
    no_new_entries_after: time = time(15, 0)
    # Risk-layer intraday square-off. This is a hard *risk* exit for an
    # intraday system; it does not replace any strategy exit.
    eod_squareoff: time = time(15, 15)


@dataclass
class InstrumentConfig:
    # VERIFY lot sizes against the latest NSE circular before any live use.
    lot_size: int
    strike_step: float


def _default_instruments() -> dict[str, InstrumentConfig]:
    return {
        "NIFTY": InstrumentConfig(lot_size=65, strike_step=50),
        "BANKNIFTY": InstrumentConfig(lot_size=30, strike_step=100),
    }


@dataclass
class Agent01Config:
    """50 EMA + VWAP crossover candle breakout."""

    enabled: bool = True
    ema_period: int = 50
    entry_offset: float = 2.0
    # "any" | "ema_above_vwap" | "ema_below_vwap". Spec: any actual crossover.
    crossover_direction: str = "any"
    # "high": a subsequent candle whose HIGH exceeds the crossover candle high.
    # "close": a subsequent candle that CLOSES above the crossover candle high.
    breakout_basis: str = "high"
    # A new crossover while waiting replaces the old reference candle.
    reset_on_new_crossover: bool = True
    breakout_max_wait_bars: Optional[int] = None   # None = wait until session end
    entry_valid_bars: Optional[int] = None         # None = valid until session end
    expire_setups_at_session_end: bool = True
    # Evaluate "close < previous candle low" on the candle in which the entry filled.
    evaluate_exit_on_entry_candle: bool = True
    ema_reset_daily: bool = False


@dataclass
class Agent02Config:
    """Fibonacci pivot points + 9 EMA breakout."""

    enabled: bool = True
    ema_period: int = 9
    entry_offset: float = 2.0
    levels: tuple[str, ...] = ("S3", "S2", "S1", "P", "R1", "R2", "R3")
    # "as_specified": PE = breakdown below support AND 9 EMA, entry = breakdown
    #                 candle HIGH + offset, exit = next lower support OR close > 9 EMA
    #                 (exactly as written in the specification).
    # "mirror_ce":    PE premium chart uses the same bullish logic as CE.
    pe_mode: str = "as_specified"
    # "broken_level": target = next level beyond the broken level (if that is not
    #                 beyond the entry price, the first level beyond the entry price).
    # "entry_price":  target = first level beyond the entry price.
    target_reference: str = "broken_level"
    level_cross_lookback_bars: Optional[int] = None
    max_entries_per_level_per_session: Optional[int] = None
    entry_valid_bars: Optional[int] = None
    cancel_pending_on_invalidation: bool = False
    expire_setups_at_session_end: bool = True
    ema_reset_daily: bool = False


@dataclass
class Agent03Config:
    """Colour divergence / individual-candle volume delta.

    Classification thresholds are fully configurable. Entry/stop/exit rules are
    *disabled by default* because the creator-specific rules are unverified.
    """

    enabled: bool = True
    min_delta_pct: float = 10.0
    min_relative_delta: float = 1.0
    min_total_volume: float = 0.0
    neutral_delta_pct: float = 5.0
    min_signal_strength: float = 0.0
    strong_delta_pct: float = 30.0
    strong_relative_delta: float = 1.5
    avg_lookback: int = 20
    # When a candle has no buy/sell split, estimate it from the bar (clearly labelled).
    estimate_when_missing: bool = True
    # Safety default: never open positions on *estimated* delta.
    allow_estimated_delta_for_entries: bool = False
    include_cvd: bool = True
    # "literal":      bullish disagreement on the CE chart -> BUY_CE;
    #                 bearish disagreement on the PE chart -> BUY_PE.
    # "per_contract": bullish disagreement on a chart -> buy that contract.
    mapping: str = "literal"
    # Entry rule: "none" (candidates only) | "signal_high_break" | "next_candle_open"
    entry_rule: str = "none"
    entry_offset: float = 2.0
    entry_valid_bars: Optional[int] = 3
    # Stop rule: "none" | "signal_candle_low" | "fixed_points" | "pct"
    stop_rule: str = "none"
    stop_value: float = 0.0
    # Exit rules (any that are enabled can close the trade)
    exit_on_opposite_disagreement: bool = False
    exit_on_close_below_prev_low: bool = False
    target_points: Optional[float] = None
    max_holding_bars: Optional[int] = None


@dataclass
class OrchestratorConfig:
    # "A" independent execution, "B" consensus analytics, "C" intelligent selection
    mode: str = "A"
    allow_multiple_agents_same_contract: bool = False
    # "allow_and_flag" | "first_come" | "block_both"
    conflict_policy: str = "allow_and_flag"
    consensus_window_minutes: int = 15
    ranking_min_trades: int = 20
    ranking_prior_trades: int = 10
    block_negative_expectancy: bool = False
    ranking_file: Optional[str] = None


@dataclass
class ContractSelectionConfig:
    expiry_rule: str = "nearest"            # "nearest" | "next"
    min_days_to_expiry: int = 0
    avoid_expiry_day: bool = False
    moneyness: str = "ATM"                  # "ATM" | "ITM" | "OTM"
    moneyness_steps: int = 0                # strikes away from ATM for ITM/OTM
    candidate_strikes_each_side: int = 4
    min_volume: float = 1000
    min_oi: float = 10000
    max_spread_pct: float = 2.0             # of mid price
    max_spread_abs: float = 3.0
    min_premium: float = 20.0
    max_premium: float = 800.0
    target_abs_delta: float = 0.5
    min_abs_delta: float = 0.25
    max_abs_delta: float = 0.75
    max_iv: float = 2.0                     # 200% sanity cap
    risk_free_rate: float = 0.065
    w_liquidity: float = 0.35
    w_spread: float = 0.25
    w_delta: float = 0.25
    w_theta: float = 0.15


@dataclass
class RiskConfig:
    capital: float = 500_000.0
    risk_per_trade_pct: float = 1.0
    # Per-unit risk assumed for sizing when a signal carries no stop/sizing reference.
    assumed_risk_pct_of_premium: float = 30.0
    # Floor on the per-unit risk used for sizing (protects against tiny stops).
    min_risk_pct_of_premium: float = 10.0
    max_lots_per_trade: int = 10
    max_capital_per_trade_pct: float = 20.0
    max_daily_loss: float = 10_000.0
    max_trades_per_day: int = 12
    max_concurrent_positions: int = 3
    max_positions_per_underlying: int = 2
    max_total_exposure_pct: float = 50.0
    agent_allocation: dict[str, float] = field(
        default_factory=lambda: {"A01": 0.34, "A02": 0.33, "A03": 0.33}
    )
    flatten_on_daily_loss: bool = True
    profit_protection_activate: Optional[float] = None   # e.g. 8000 (rupees)
    profit_protection_giveback_pct: float = 40.0
    flatten_on_profit_lock: bool = False


@dataclass
class ChargesConfig:
    """Indian F&O option charges. VERIFY current rates before relying on them."""

    brokerage_per_order: float = 20.0
    stt_sell_pct: float = 0.1            # % of sell-side premium turnover
    exchange_txn_pct: float = 0.03503    # % of premium turnover (NSE)
    sebi_per_crore: float = 10.0
    stamp_buy_pct: float = 0.003         # % of buy-side turnover
    gst_pct: float = 18.0                # on brokerage + exchange + SEBI fees
    enabled: bool = True


@dataclass
class ExecutionConfig:
    # "backtest" | "paper" | "manual" | "live"
    mode: str = "backtest"
    live_enabled: bool = False           # live trading is OFF by default
    slippage_points: float = 0.5
    slippage_pct: float = 0.0
    # Close-evaluated signals: "next_open" (realistic) or "close" (idealised)
    close_signal_fill: str = "next_open"
    # Cap fill size to this % of the bar's traded volume (None = unlimited)
    max_participation_pct: Optional[float] = 20.0
    reject_if_no_price: bool = True
    approval_timeout_minutes: int = 2


@dataclass
class BacktestConfig:
    # With OHLC-only data the order of prices inside a bar is unknown. When
    # False, price-based exits (targets/stops) are NOT assumed to hit inside
    # the same bar (or 1-minute sub-bar) in which the entry filled.
    intrabar_exit_on_entry_bar: bool = False
    warmup_sessions: int = 3
    data_label: str = "UNSPECIFIED"


@dataclass
class AppConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    instruments: dict[str, InstrumentConfig] = field(default_factory=_default_instruments)
    agent01: Agent01Config = field(default_factory=Agent01Config)
    agent02: Agent02Config = field(default_factory=Agent02Config)
    agent03: Agent03Config = field(default_factory=Agent03Config)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    selection: ContractSelectionConfig = field(default_factory=ContractSelectionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    charges: ChargesConfig = field(default_factory=ChargesConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    db_path: str = "options_mas.sqlite3"

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def lot_size(self, underlying: str) -> int:
        inst = self.instruments.get(underlying)
        return inst.lot_size if inst else 1


_TIME_FIELDS = {"market_open", "market_close", "no_new_entries_after", "eod_squareoff"}


def _merge(obj: Any, data: dict[str, Any]) -> Any:
    for f in fields(obj):
        if f.name not in data:
            continue
        val = data[f.name]
        cur = getattr(obj, f.name)
        if f.name == "instruments":
            setattr(
                obj,
                "instruments",
                {k: InstrumentConfig(**v) for k, v in val.items()},
            )
        elif is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        elif f.name in _TIME_FIELDS and isinstance(val, str):
            setattr(obj, f.name, _t(val))
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(obj, f.name, tuple(val))
        else:
            setattr(obj, f.name, val)
    return obj


def apply_overrides(cfg: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    """Apply nested overrides, e.g. {"agent01": {"entry_offset": 3.0}}."""
    return _merge(cfg, overrides)


def set_param(cfg: AppConfig, dotted: str, value: Any) -> AppConfig:
    """Set a single parameter by dotted path, e.g. ``agent02.ema_period``."""
    parts = dotted.split(".")
    target: Any = cfg
    for p in parts[:-1]:
        target = getattr(target, p)
    if not hasattr(target, parts[-1]):
        raise AttributeError(f"Unknown config parameter: {dotted}")
    setattr(target, parts[-1], value)
    return cfg


def load_config(path: str | Path | None = None) -> AppConfig:
    cfg = AppConfig()
    if path is None:
        return cfg
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return _merge(cfg, data)
