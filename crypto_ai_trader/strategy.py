from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .contracts import (
    MarketRegime,
    MetaSignal,
    RegimeState,
    RiskDecision,
    StrategyDecision,
)


class DecisionReason(StrEnum):
    """Stable reason codes shared by reports and user interfaces."""

    NO_ALPHA_SIGNAL = "no_alpha_signal"
    BLOCK_HORIZON = "blocked_horizon_confirmation"
    BLOCK_SIDE_POLICY = "blocked_side_policy"
    BLOCK_COST = "blocked_cost_edge"
    BLOCK_TRADEABILITY = "blocked_tradeability"
    BLOCK_REGIME = "blocked_market_regime"
    BLOCK_VOLATILITY = "blocked_volatility_regime"
    BLOCK_ARCHETYPE = "blocked_strategy_archetype"
    BLOCK_MARKET_STRUCTURE = "blocked_market_structure_or_crowding"
    BLOCK_EVENT = "blocked_event_filter"
    BLOCK_RISK_REGIME = "blocked_risk_regime"
    BLOCK_FUNDING_CROWDING = "blocked_funding_crowding"
    BLOCK_DRAWDOWN = "blocked_drawdown_cooldown"
    ALLOW_LONG = "allow_long"
    ALLOW_SHORT = "allow_short"
    FLAT_AFTER_FILTERS = "flat_after_filters"


GATE_REASON_PRIORITY = (
    ("horizon_gate_pass", DecisionReason.BLOCK_HORIZON),
    ("side_policy_gate_pass", DecisionReason.BLOCK_SIDE_POLICY),
    ("cost_edge_pass", DecisionReason.BLOCK_COST),
    ("tradeability_gate_pass", DecisionReason.BLOCK_TRADEABILITY),
    ("regime_gate_pass", DecisionReason.BLOCK_REGIME),
    ("volatility_regime_policy_pass", DecisionReason.BLOCK_VOLATILITY),
    ("archetype_policy_gate_pass", DecisionReason.BLOCK_ARCHETYPE),
    ("platform_strategy_quality_gate_pass", DecisionReason.BLOCK_MARKET_STRUCTURE),
    ("platform_event_gate_pass", DecisionReason.BLOCK_EVENT),
    ("regime_risk_gate_pass", DecisionReason.BLOCK_RISK_REGIME),
    ("funding_crowding_gate_pass", DecisionReason.BLOCK_FUNDING_CROWDING),
    ("drawdown_cooldown_gate_pass", DecisionReason.BLOCK_DRAWDOWN),
)


def decision_reason_codes(detail: pd.DataFrame) -> pd.Series:
    """Explain each final backtest position using the first failed gate."""

    if "position" not in detail.columns or "model_position" not in detail.columns:
        raise ValueError("decision reasons require position and model_position columns")

    final_position = detail["position"].to_numpy(dtype=int)
    model_position = detail["model_position"].to_numpy(dtype=int)
    reasons = np.full(len(detail), DecisionReason.FLAT_AFTER_FILTERS.value, dtype=object)
    reasons[model_position == 0] = DecisionReason.NO_ALPHA_SIGNAL.value

    unresolved = (model_position != 0) & (final_position == 0)
    for column, reason in GATE_REASON_PRIORITY:
        if column not in detail.columns:
            continue
        gate_pass = detail[column].astype(bool).to_numpy()
        blocked = unresolved & ~gate_pass
        reasons[blocked] = reason.value
        unresolved &= gate_pass

    reasons[final_position > 0] = DecisionReason.ALLOW_LONG.value
    reasons[final_position < 0] = DecisionReason.ALLOW_SHORT.value
    return pd.Series(reasons, index=detail.index, dtype="object")


@dataclass(frozen=True)
class StrategyContext:
    """Inputs shared by explainable strategy rules."""

    meta_signal: MetaSignal
    regime: RegimeState
    current_position: int = 0
    risk_state: RiskDecision | None = None
    transaction_cost_estimate: float = 0.0
    max_exposure: float = 0.10
    stop_loss: float | None = None
    take_profit: float | None = None
    holding_period: int | None = None


class BaseStrategy(ABC):
    """Rule-only strategy interface; model training belongs to the Alpha layer."""

    name = "base"

    def __init__(self, min_score: float = 0.55):
        self.min_score = float(np.clip(min_score, 0.5, 1.0))

    @abstractmethod
    def decide(self, context: StrategyContext) -> StrategyDecision:
        """Interpret a fused signal as a requested target position."""

    def _flat(self, reason: str) -> StrategyDecision:
        return StrategyDecision(0, 0.0, None, None, None, reason)

    def _directional_decision(
        self,
        context: StrategyContext,
        *,
        long_score: float,
        short_score: float,
        reason_prefix: str,
    ) -> StrategyDecision:
        if context.meta_signal.risk_off:
            return self._flat(f"{reason_prefix}_risk_off")
        if context.risk_state is not None and not context.risk_state.allow_trade:
            return self._flat(f"{reason_prefix}_{context.risk_state.reason}")
        trade_score = float(context.meta_signal.trade_score)
        if trade_score < self.min_score:
            return self._flat(f"{reason_prefix}_below_threshold")
        direction = 1 if long_score > short_score else -1 if short_score > long_score else 0
        if direction == 0:
            return self._flat(f"{reason_prefix}_neutral")
        exposure_cap = max(float(context.max_exposure), 0.0)
        if context.risk_state is not None:
            exposure_cap = min(
                exposure_cap,
                max(float(context.risk_state.max_position_size), 0.0),
            )
        exposure = exposure_cap * float(
            np.clip(context.meta_signal.confidence, 0.0, 1.0)
        )
        if context.regime.regime == MarketRegime.HIGH_VOL:
            exposure *= 0.5
        if exposure <= 0.0:
            return self._flat(f"{reason_prefix}_zero_risk_capacity")
        return StrategyDecision(
            target_direction=direction,
            target_exposure=exposure,
            stop_loss=context.stop_loss,
            take_profit=context.take_profit,
            holding_period=context.holding_period,
            reason_code=(
                f"{reason_prefix}_long"
                if direction > 0
                else f"{reason_prefix}_short"
            ),
        )


class TrendStrategy(BaseStrategy):
    """Prefer trend Alpha only in detected directional regimes."""

    name = "trend"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        if context.regime.regime not in {
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
        }:
            return self._flat("trend_regime_not_directional")
        return self._directional_decision(
            context,
            long_score=context.meta_signal.components.get(
                "long_trend",
                context.meta_signal.long_score,
            ),
            short_score=context.meta_signal.components.get(
                "short_trend",
                context.meta_signal.short_score,
            ),
            reason_prefix="trend",
        )


class MeanReversionStrategy(BaseStrategy):
    """Prefer range-reversion Alpha only in range regimes."""

    name = "mean_reversion"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        if context.regime.regime != MarketRegime.RANGE:
            return self._flat("mean_reversion_regime_not_range")
        return self._directional_decision(
            context,
            long_score=context.meta_signal.components.get(
                "long_mean_reversion",
                context.meta_signal.long_score,
            ),
            short_score=context.meta_signal.components.get(
                "short_mean_reversion",
                context.meta_signal.short_score,
            ),
            reason_prefix="mean_reversion",
        )


class FundingStrategy(BaseStrategy):
    """Interpret the funding Alpha component without training a model."""

    name = "funding"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return self._directional_decision(
            context,
            long_score=context.meta_signal.components.get("long_funding", 0.5),
            short_score=context.meta_signal.components.get("short_funding", 0.5),
            reason_prefix="funding",
        )


class CrossSectionalStrategy(BaseStrategy):
    """Interpret a cross-sectional rank component when it is available."""

    name = "cross_sectional"

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return self._directional_decision(
            context,
            long_score=context.meta_signal.components.get(
                "long_cross_sectional",
                0.5,
            ),
            short_score=context.meta_signal.components.get(
                "short_cross_sectional",
                0.5,
            ),
            reason_prefix="cross_sectional",
        )


class StrategyOrchestrator:
    """Select the regime-appropriate rule decision with final risk awareness."""

    def __init__(
        self,
        strategies: tuple[BaseStrategy, ...] | None = None,
    ):
        self.strategies = strategies or (
            TrendStrategy(),
            MeanReversionStrategy(),
            FundingStrategy(min_score=0.60),
            CrossSectionalStrategy(min_score=0.60),
        )

    def decide(self, context: StrategyContext) -> StrategyDecision:
        """Return the strongest non-flat request; risk may still veto later."""

        if context.regime.risk_off or context.meta_signal.risk_off:
            return StrategyDecision(
                0,
                0.0,
                None,
                None,
                None,
                "orchestrator_risk_off",
            )
        if context.risk_state is not None and not context.risk_state.allow_trade:
            return StrategyDecision(
                0,
                0.0,
                None,
                None,
                None,
                f"orchestrator_{context.risk_state.reason}",
            )
        decisions = [strategy.decide(context) for strategy in self.strategies]
        active = [item for item in decisions if item.target_direction != 0]
        if active:
            return max(active, key=lambda item: item.target_exposure)
        return StrategyDecision(
            0,
            0.0,
            None,
            None,
            None,
            "orchestrator_no_strategy_passed",
        )
