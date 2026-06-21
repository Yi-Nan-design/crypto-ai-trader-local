from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    AlphaPrediction,
    LiquidityState,
    MarketRegime,
    MetaSignal,
    RegimeState,
    RiskDecision,
)


META_SIGNAL_VERSION = "2026-06-21-weighted-v1"


@dataclass(frozen=True)
class MetaSignalConfig:
    """Weights and gates for the explainable signal-fusion baseline."""

    alpha_weight: float = 1.0
    trend_weight: float = 0.75
    mean_reversion_weight: float = 0.65
    funding_weight: float = 0.35
    cross_sectional_weight: float = 0.35
    regime_weight_multiplier: float = 1.5
    high_vol_trade_scale: float = 0.60
    min_trade_score: float = 0.55


@dataclass(frozen=True)
class MetaSignalInputs:
    """Typed inputs accepted by the Meta Signal layer."""

    alpha: AlphaPrediction
    regime: RegimeState
    risk: RiskDecision | None = None
    transaction_cost_estimate: float = 0.0
    trend_long: float = 0.5
    trend_short: float = 0.5
    mean_reversion_long: float = 0.5
    mean_reversion_short: float = 0.5
    funding_long: float = 0.5
    funding_short: float = 0.5
    cross_sectional_long: float = 0.5
    cross_sectional_short: float = 0.5


def funding_alpha_scores(
    funding_rate: float,
    *,
    normalization_rate: float = 0.001,
) -> tuple[float, float]:
    """Convert signed funding into contrarian long/short desirability."""

    normalized = float(
        np.clip(
            float(funding_rate) / max(abs(float(normalization_rate)), 1e-12),
            -1.0,
            1.0,
        )
    )
    return (
        float(np.clip(0.5 - 0.5 * normalized, 0.0, 1.0)),
        float(np.clip(0.5 + 0.5 * normalized, 0.0, 1.0)),
    )


class WeightedMetaSignal:
    """Fuse heterogeneous alphas with regime, cost, liquidity, and risk gates."""

    def __init__(self, config: MetaSignalConfig | None = None):
        self.config = config or MetaSignalConfig()

    @staticmethod
    def _score(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def fuse(self, inputs: MetaSignalInputs) -> MetaSignal:
        """Return one explainable fused signal without creating an order."""

        cfg = self.config
        trend_multiplier = (
            cfg.regime_weight_multiplier
            if inputs.regime.regime
            in {MarketRegime.TREND_UP, MarketRegime.TREND_DOWN}
            else 1.0
        )
        mean_reversion_multiplier = (
            cfg.regime_weight_multiplier
            if inputs.regime.regime == MarketRegime.RANGE
            else 1.0
        )
        weights = {
            "alpha": max(float(cfg.alpha_weight), 0.0),
            "trend": max(float(cfg.trend_weight) * trend_multiplier, 0.0),
            "mean_reversion": max(
                float(cfg.mean_reversion_weight)
                * mean_reversion_multiplier,
                0.0,
            ),
            "funding": max(float(cfg.funding_weight), 0.0),
            "cross_sectional": max(float(cfg.cross_sectional_weight), 0.0),
        }
        total_weight = max(sum(weights.values()), 1e-12)
        long_components = {
            "alpha": self._score(inputs.alpha.p_up),
            "trend": self._score(inputs.trend_long),
            "mean_reversion": self._score(inputs.mean_reversion_long),
            "funding": self._score(inputs.funding_long),
            "cross_sectional": self._score(inputs.cross_sectional_long),
        }
        short_components = {
            "alpha": self._score(inputs.alpha.p_down),
            "trend": self._score(inputs.trend_short),
            "mean_reversion": self._score(inputs.mean_reversion_short),
            "funding": self._score(inputs.funding_short),
            "cross_sectional": self._score(inputs.cross_sectional_short),
        }
        long_score = sum(
            weights[name] * long_components[name] for name in weights
        ) / total_weight
        short_score = sum(
            weights[name] * short_components[name] for name in weights
        ) / total_weight
        confidence = float(np.clip(abs(long_score - short_score), 0.0, 1.0))
        trade_score = float(
            np.clip(
                max(long_score, short_score)
                * (0.75 + 0.25 * inputs.alpha.confidence),
                0.0,
                1.0,
            )
        )
        blocked_reason: str | None = None
        if inputs.regime.risk_off:
            blocked_reason = "meta_blocked_regime_risk_off"
        elif inputs.regime.liquidity_state == LiquidityState.CRISIS:
            blocked_reason = "meta_blocked_liquidity_crisis"
        elif inputs.risk is not None and not inputs.risk.allow_trade:
            blocked_reason = f"meta_blocked_{inputs.risk.reason}"
        elif (
            inputs.alpha.expected_return is not None
            and abs(float(inputs.alpha.expected_return))
            <= max(float(inputs.transaction_cost_estimate), 0.0)
        ):
            blocked_reason = "meta_blocked_cost_hurdle"
        if inputs.regime.regime == MarketRegime.HIGH_VOL:
            trade_score *= float(
                np.clip(cfg.high_vol_trade_scale, 0.0, 1.0)
            )
        if blocked_reason is not None:
            trade_score = 0.0
            reason = blocked_reason
        elif trade_score < float(cfg.min_trade_score):
            reason = "meta_below_trade_threshold"
        elif long_score > short_score:
            reason = "meta_long_preferred"
        elif short_score > long_score:
            reason = "meta_short_preferred"
        else:
            reason = "meta_neutral"
        components = {
            **{f"long_{key}": value for key, value in long_components.items()},
            **{
                f"short_{key}": value
                for key, value in short_components.items()
            },
            **{f"weight_{key}": value for key, value in weights.items()},
        }
        return MetaSignal(
            timestamp=inputs.alpha.timestamp,
            symbol=inputs.alpha.symbol,
            horizon=inputs.alpha.horizon,
            long_score=float(long_score),
            short_score=float(short_score),
            trade_score=float(np.clip(trade_score, 0.0, 1.0)),
            confidence=confidence,
            expected_return=inputs.alpha.expected_return,
            volatility_forecast=inputs.alpha.volatility_forecast,
            regime=inputs.regime.regime,
            risk_off=blocked_reason is not None,
            reason=reason,
            components=components,
        )
