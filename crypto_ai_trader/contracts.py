from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class AlphaPrediction:
    """Model output contract; it describes forecasts and never places orders."""

    timestamp: str
    symbol: str
    horizon: str
    expected_return: float | None
    p_up: float
    p_down: float
    volatility_forecast: float | None
    confidence: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyDecision:
    """Explainable target-position request consumed by paper execution."""

    target_direction: int
    target_exposure: float
    stop_loss: float | None
    take_profit: float | None
    holding_period: int | None
    reason_code: str

    def __post_init__(self) -> None:
        if self.target_direction not in {-1, 0, 1}:
            raise ValueError("target_direction must be -1, 0, or 1")
        if self.target_exposure < 0:
            raise ValueError("target_exposure must be non-negative")


class RiskLevel(StrEnum):
    """Stable risk severity values shared by reports and control surfaces."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class MarketRegime(StrEnum):
    """Stable market-state labels shared by strategy, risk, and reports."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    CRASH = "crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"


class VolatilityState(StrEnum):
    """Causal volatility bucket derived from trailing observations."""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


class LiquidityState(StrEnum):
    """Liquidity condition inferred from available volume proxies."""

    NORMAL = "normal"
    THIN = "thin"
    CRISIS = "crisis"


@dataclass(frozen=True)
class RegimeState:
    """Market-state output; it is context for decisions, not an order."""

    timestamp: str
    regime: MarketRegime
    confidence: float
    volatility_state: VolatilityState
    liquidity_state: LiquidityState
    risk_off: bool
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime"] = self.regime.value
        payload["volatility_state"] = self.volatility_state.value
        payload["liquidity_state"] = self.liquidity_state.value
        return payload


@dataclass(frozen=True)
class MetaSignal:
    """Fused research signal consumed by strategy rules, never by an exchange."""

    timestamp: str
    symbol: str
    horizon: str
    long_score: float
    short_score: float
    trade_score: float
    confidence: float
    expected_return: float | None
    volatility_forecast: float | None
    regime: MarketRegime
    risk_off: bool
    reason: str
    components: dict[str, float]

    def __post_init__(self) -> None:
        for name in ("long_score", "short_score", "trade_score", "confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regime"] = self.regime.value
        return payload


@dataclass(frozen=True)
class RiskDecision:
    """Final risk authority output applied after a strategy decision."""

    allow_trade: bool
    risk_level: RiskLevel
    max_position_size: float
    reason: str

    def __post_init__(self) -> None:
        if self.max_position_size < 0:
            raise ValueError("max_position_size must be non-negative")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_level"] = self.risk_level.value
        return payload


@dataclass(frozen=True)
class PortfolioAssetInput:
    """One strategy/risk-approved asset request for portfolio construction."""

    symbol: str
    direction: int
    signal_strength: float
    confidence: float
    volatility: float
    liquidity_score: float
    max_weight: float
    correlation_cluster: str
    sector: str = "unclassified"
    input_available: bool = True
    input_reason: str = "portfolio_input_available"

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        for name in ("signal_strength", "confidence", "liquidity_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.volatility < 0.0:
            raise ValueError("volatility must be non-negative")
        if self.max_weight < 0.0:
            raise ValueError("max_weight must be non-negative")
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if not self.correlation_cluster.strip():
            raise ValueError("correlation_cluster must not be empty")
        if not self.sector.strip():
            raise ValueError("sector must not be empty")
        if not self.input_reason.strip():
            raise ValueError("input_reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioDecision:
    """Portfolio target weights after cross-asset constraints."""

    weights: dict[str, float]
    gross_exposure: float
    net_exposure: float
    cluster_exposure: dict[str, float]
    sector_exposure: dict[str, float]
    allow_portfolio: bool
    risk_level: RiskLevel
    reason: str
    asset_reasons: dict[str, str]

    def __post_init__(self) -> None:
        if self.gross_exposure < 0.0:
            raise ValueError("gross_exposure must be non-negative")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk_level"] = self.risk_level.value
        return payload
