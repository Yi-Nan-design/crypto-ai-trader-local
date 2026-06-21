from __future__ import annotations

import unittest

from crypto_ai_trader.contracts import (
    AlphaPrediction,
    LiquidityState,
    MarketRegime,
    RegimeState,
    RiskDecision,
    RiskLevel,
    VolatilityState,
)
from crypto_ai_trader.meta_signal import (
    MetaSignalInputs,
    WeightedMetaSignal,
    funding_alpha_scores,
)
from crypto_ai_trader.strategy import StrategyContext, StrategyOrchestrator


def alpha() -> AlphaPrediction:
    return AlphaPrediction(
        timestamp="2026-06-21T12:00:00+08:00",
        symbol="ETHUSDT",
        horizon="5m",
        expected_return=0.003,
        p_up=0.72,
        p_down=0.28,
        volatility_forecast=0.01,
        confidence=0.44,
        model_version="test",
    )


def regime(value: MarketRegime, *, risk_off: bool = False) -> RegimeState:
    return RegimeState(
        timestamp="2026-06-21T12:00:00+08:00",
        regime=value,
        confidence=0.8,
        volatility_state=VolatilityState.HIGH
        if value == MarketRegime.HIGH_VOL
        else VolatilityState.MID,
        liquidity_state=LiquidityState.CRISIS
        if value == MarketRegime.LIQUIDITY_CRISIS
        else LiquidityState.NORMAL,
        risk_off=risk_off,
        reason="test",
    )


class MetaStrategyTests(unittest.TestCase):
    def test_regime_weighted_signal_and_orchestrator_choose_trend_long(self) -> None:
        state = regime(MarketRegime.TREND_UP)
        risk = RiskDecision(True, RiskLevel.LOW, 0.20, "risk_allowed")
        signal = WeightedMetaSignal().fuse(
            MetaSignalInputs(
                alpha=alpha(),
                regime=state,
                risk=risk,
                trend_long=0.95,
                trend_short=0.05,
                mean_reversion_long=0.45,
                mean_reversion_short=0.55,
            )
        )
        decision = StrategyOrchestrator().decide(
            StrategyContext(
                meta_signal=signal,
                regime=state,
                risk_state=risk,
                max_exposure=0.20,
                stop_loss=0.02,
                take_profit=0.04,
                holding_period=6,
            )
        )

        self.assertGreater(signal.long_score, signal.short_score)
        self.assertEqual(decision.target_direction, 1)
        self.assertEqual(decision.reason_code, "trend_long")
        self.assertLessEqual(decision.target_exposure, 0.20)

    def test_risk_off_blocks_meta_signal_and_strategy(self) -> None:
        state = regime(MarketRegime.CRASH, risk_off=True)
        signal = WeightedMetaSignal().fuse(
            MetaSignalInputs(alpha=alpha(), regime=state)
        )
        decision = StrategyOrchestrator().decide(
            StrategyContext(meta_signal=signal, regime=state)
        )

        self.assertEqual(signal.trade_score, 0.0)
        self.assertTrue(signal.risk_off)
        self.assertEqual(decision.target_direction, 0)
        self.assertEqual(decision.reason_code, "orchestrator_risk_off")

    def test_positive_funding_favors_short_component(self) -> None:
        long_score, short_score = funding_alpha_scores(0.001)

        self.assertLess(long_score, short_score)
        self.assertEqual(long_score, 0.0)
        self.assertEqual(short_score, 1.0)


if __name__ == "__main__":
    unittest.main()
