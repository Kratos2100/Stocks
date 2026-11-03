"""Core domain models shared by every layer of the system.

All models are plain dataclasses so that strategy logic stays deterministic,
dependency-free and trivially serialisable to the standard JSON message format
(see ``docs/SIGNAL_PROTOCOL.md``).

Timestamp convention
--------------------
All timestamps are *naive* ``datetime`` objects expressed in exchange local
time (IST, Asia/Kolkata). A candle's ``ts`` is the bar **start** time; the bar
is complete at ``ts + timeframe``.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Optional


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class OptionType(StrEnum):
    CE = "CE"
    PE = "PE"


class SignalType(StrEnum):
    BUY_CE = "BUY_CE"
    BUY_PE = "BUY_PE"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"
    EXIT_POSITION = "EXIT_POSITION"


class SignalStatus(StrEnum):
    INFO = "INFO"                # state change / informational, not actionable
    CANDIDATE = "CANDIDATE"      # setup classified, entry rule not satisfied / not configured
    ARMED = "ARMED"              # entry trigger price known, waiting for price to reach it
    TRIGGERED = "TRIGGERED"      # actionable: entry trigger reached or exit condition met
    REJECTED = "REJECTED"        # setup rejected / invalidated / cancelled
    CANCELLED = "CANCELLED"


class AgentState(StrEnum):
    # Agent 01 states (exact names from the specification)
    WAITING_FOR_CROSSOVER = "WAITING_FOR_CROSSOVER"
    CROSSOVER_DETECTED = "CROSSOVER_DETECTED"
    WAITING_FOR_BREAKOUT = "WAITING_FOR_BREAKOUT"
    BREAKOUT_IDENTIFIED = "BREAKOUT_IDENTIFIED"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_TRIGGERED = "EXIT_TRIGGERED"
    TRADE_COMPLETED = "TRADE_COMPLETED"
    # Additional states used by Agent 02 / Agent 03
    NO_PIVOT_DATA = "NO_PIVOT_DATA"
    WAITING_FOR_SIGNAL = "WAITING_FOR_SIGNAL"
    WAITING_FOR_ENTRY = "WAITING_FOR_ENTRY"
    WARMING_UP = "WARMING_UP"


class DeltaSource(StrEnum):
    """Where an individual candle's buy/sell volume split came from.

    Only the two ``BID_ASK_*`` sources represent bid/ask-classified trade
    volume. Everything starting with ``ESTIMATED_`` is a model and must never
    be presented as actual exchange order flow. ``SYNTHETIC`` is generated
    test data.
    """

    BID_ASK_PROVIDER = "BID_ASK_PROVIDER"        # vendor-supplied aggressor-classified volume
    BID_ASK_QUOTE_RULE = "BID_ASK_QUOTE_RULE"    # we classified trades against prevailing bid/ask
    ESTIMATED_TICK_RULE = "ESTIMATED_TICK_RULE"  # trade-to-trade price change (no quotes)
    ESTIMATED_BAR_CLV = "ESTIMATED_BAR_CLV"      # close-location value of the OHLC bar
    SYNTHETIC = "SYNTHETIC"                      # synthetic demo/test data
    NONE = "NONE"

    @property
    def is_bid_ask_classified(self) -> bool:
        return self in (DeltaSource.BID_ASK_PROVIDER, DeltaSource.BID_ASK_QUOTE_RULE)

    @property
    def is_estimated(self) -> bool:
        return not self.is_bid_ask_classified


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderPurpose(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class OrderStatus(StrEnum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED)


class ExitReason(StrEnum):
    PREV_CANDLE_LOW_CLOSE = "PREV_CANDLE_LOW_CLOSE"   # Agent 01
    NEXT_PIVOT_TARGET = "NEXT_PIVOT_TARGET"           # Agent 02
    EMA_EXIT = "EMA_EXIT"                             # Agent 02
    STOP_LOSS = "STOP_LOSS"                           # Agent 03 (configurable)
    TARGET = "TARGET"                                 # Agent 03 (configurable)
    OPPOSITE_DISAGREEMENT = "OPPOSITE_DISAGREEMENT"   # Agent 03 (configurable)
    CLOSE_BELOW_PREV_LOW = "CLOSE_BELOW_PREV_LOW"     # Agent 03 (configurable)
    TIME_EXIT = "TIME_EXIT"                           # Agent 03 (configurable)
    # Risk-layer exits (never replace strategy exits, only add hard safety exits)
    EOD_SQUARE_OFF = "EOD_SQUARE_OFF"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    PROFIT_PROTECTION = "PROFIT_PROTECTION"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MANUAL = "MANUAL"


_id_counter = itertools.count(1)


def new_id(prefix: str) -> str:
    """Deterministic, monotonically increasing ids (stable across runs)."""
    return f"{prefix}-{next(_id_counter):07d}"


def reset_ids() -> None:
    global _id_counter
    _id_counter = itertools.count(1)


def random_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    exchange: str = "NSE"
    lot_size: int = 1
    tradingsymbol: str = ""

    @property
    def key(self) -> str:
        strike = f"{self.strike:g}"
        return f"{self.underlying}-{self.expiry:%Y%m%d}-{strike}-{self.option_type.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "underlying": self.underlying,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type.value,
            "exchange": self.exchange,
            "lot_size": self.lot_size,
            "tradingsymbol": self.tradingsymbol,
        }


@dataclass(frozen=True)
class Candle:
    ts: datetime                      # bar start time
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    delta_source: DeltaSource = DeltaSource.NONE
    oi: Optional[float] = None
    timeframe_min: int = 5

    @property
    def end_ts(self) -> datetime:
        return self.ts + timedelta(minutes=self.timeframe_min)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def has_delta(self) -> bool:
        return self.buy_volume is not None and self.sell_volume is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "delta_source": self.delta_source.value,
        }


@dataclass
class Signal:
    """The common agent communication schema (Section 6 of the spec)."""

    agent_id: str
    strategy_name: str
    exchange: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: date
    timeframe: str
    signal_type: SignalType
    status: SignalStatus
    signal_ts: datetime
    contract_key: str
    reason: str = ""
    entry_trigger: Optional[float] = None
    breakout_candle_ts: Optional[datetime] = None
    breakout_candle_high: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    exit_condition: str = ""
    conditions_satisfied: list[str] = field(default_factory=list)
    conditions_not_satisfied: list[str] = field(default_factory=list)
    reference_candle: Optional[dict[str, Any]] = None
    breakout_candle: Optional[dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    event_id: Optional[str] = None
    trigger_price: Optional[float] = None    # price observed when TRIGGERED
    exit_reason: Optional[ExitReason] = None
    agent_state: Optional[AgentState] = None
    sizing_reference: Optional[float] = None  # indicative per-unit risk reference used ONLY for sizing
    extras: dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: new_id("SIG"))

    @property
    def is_entry_trigger(self) -> bool:
        return (
            self.signal_type in (SignalType.BUY_CE, SignalType.BUY_PE)
            and self.status == SignalStatus.TRIGGERED
        )

    @property
    def is_exit_trigger(self) -> bool:
        return self.signal_type == SignalType.EXIT_POSITION and self.status == SignalStatus.TRIGGERED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, (datetime, date)):
                d[k] = v.isoformat()
        return d


@dataclass
class Fill:
    order_id: str
    ts: datetime
    qty: int
    price: float
    charges: float = 0.0


@dataclass
class Order:
    order_id: str
    contract: OptionContract
    side: OrderSide
    qty: int
    purpose: OrderPurpose
    agent_id: str
    created_ts: datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    signal_id: Optional[str] = None
    position_id: Optional[str] = None
    exit_reason: Optional[ExitReason] = None
    reject_reason: Optional[str] = None
    updated_ts: Optional[datetime] = None
    broker_order_id: Optional[str] = None
    fills: list[Fill] = field(default_factory=list)

    @property
    def remaining_qty(self) -> int:
        return self.qty - self.filled_qty

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "contract_key": self.contract.key,
            "side": self.side.value,
            "qty": self.qty,
            "purpose": self.purpose.value,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": round(self.avg_fill_price, 4),
            "created_ts": self.created_ts.isoformat(),
            "updated_ts": self.updated_ts.isoformat() if self.updated_ts else None,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "reject_reason": self.reject_reason,
            "signal_id": self.signal_id,
        }


@dataclass
class Position:
    position_id: str
    agent_id: str
    contract: OptionContract
    qty: int
    entry_price: float
    entry_ts: datetime
    signal_id: Optional[str] = None
    entry_charges: float = 0.0
    exit_qty: int = 0
    exit_value: float = 0.0
    exit_charges: float = 0.0
    exit_ts: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None
    exit_order_id: Optional[str] = None
    is_open: bool = True
    last_price: Optional[float] = None
    sizing_risk_per_unit: Optional[float] = None

    @property
    def open_qty(self) -> int:
        return self.qty - self.exit_qty

    @property
    def cost(self) -> float:
        return self.entry_price * self.open_qty

    def unrealized_pnl(self, price: Optional[float] = None) -> float:
        p = price if price is not None else self.last_price
        if p is None:
            return 0.0
        return (p - self.entry_price) * self.open_qty

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "agent_id": self.agent_id,
            "contract_key": self.contract.key,
            "option_type": self.contract.option_type.value,
            "qty": self.qty,
            "open_qty": self.open_qty,
            "entry_price": round(self.entry_price, 4),
            "entry_ts": self.entry_ts.isoformat(),
            "last_price": self.last_price,
            "unrealized_pnl": round(self.unrealized_pnl(), 2),
            "is_open": self.is_open,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
        }


@dataclass
class Trade:
    """A completed round trip. P&L is net of the modelled charges."""

    trade_id: str
    agent_id: str
    contract_key: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: date
    qty: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: ExitReason
    gross_pnl: float
    charges: float
    net_pnl: float
    signal_id: Optional[str] = None
    regime: str = "UNKNOWN"
    risk_amount: Optional[float] = None

    @property
    def holding_minutes(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds() / 60.0

    @property
    def r_multiple(self) -> Optional[float]:
        if not self.risk_amount:
            return None
        return self.net_pnl / self.risk_amount

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, (datetime, date)):
                d[k] = v.isoformat()
        d["holding_minutes"] = round(self.holding_minutes, 2)
        return d


@dataclass
class RiskDecision:
    approved: bool
    qty: int = 0
    lots: int = 0
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    risk_amount: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
