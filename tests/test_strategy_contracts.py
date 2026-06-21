from __future__ import annotations

import unittest

import pandas as pd

from crypto_ai_trader.contracts import (
    AlphaPrediction,
    LiquidityState,
    MarketRegime,
    RegimeState,
    RiskDecision,
    RiskLevel,
    StrategyDecision,
    VolatilityState,
)
from crypto_ai_trader.strategy import DecisionReason, decision_reason_codes


class StrategyContractTests(unittest.TestCase):
    def test_alpha_prediction_serializes_stable_fields(self) -> None:
        prediction = AlphaPrediction(
            timestamp="2026-06-18T10:00:00+08:00",
            symbol="ETHUSDT",
            horizon="5m",
            expected_return=None,
            p_up=0.62,
            p_down=0.38,
            volatility_forecast=0.01,
            confidence=0.24,
            model_version="test-model",
        )

        self.assertEqual(prediction.to_dict()["symbol"], "ETHUSDT")
        self.assertEqual(prediction.to_dict()["p_down"], 0.38)

    def test_strategy_decision_rejects_invalid_direction(self) -> None:
        with self.assertRaises(ValueError):
            StrategyDecision(2, 0.1, None, None, None, "invalid")

    def test_regime_state_serializes_stable_fields(self) -> None:
        state = RegimeState(
            timestamp="2026-06-18T10:00:00+08:00",
            regime=MarketRegime.CRASH,
            confidence=0.8,
            volatility_state=VolatilityState.HIGH,
            liquidity_state=LiquidityState.THIN,
            risk_off=True,
            reason="negative_return_exceeded_crash_threshold",
        )

        self.assertEqual(state.to_dict()["regime"], "crash")
        self.assertEqual(state.to_dict()["volatility_state"], "high")
        self.assertTrue(state.to_dict()["risk_off"])

    def test_risk_decision_serializes_stable_fields(self) -> None:
        decision = RiskDecision(
            allow_trade=False,
            risk_level=RiskLevel.EXTREME,
            max_position_size=0.0,
            reason="risk_blocked_drawdown_cooldown",
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "allow_trade": False,
                "risk_level": "extreme",
                "max_position_size": 0.0,
                "reason": "risk_blocked_drawdown_cooldown",
            },
        )

    def test_reason_codes_use_first_failed_gate_and_final_side(self) -> None:
        detail = pd.DataFrame(
            {
                "model_position": [0, 1, -1, 1],
                "position": [0, 0, -1, 0],
                "horizon_gate_pass": [True, True, True, True],
                "side_policy_gate_pass": [True, True, True, True],
                "cost_edge_pass": [True, False, True, True],
                "tradeability_gate_pass": [True, True, True, True],
                "regime_gate_pass": [True, True, True, True],
                "regime_risk_gate_pass": [True, True, True, False],
            }
        )

        reasons = decision_reason_codes(detail).tolist()

        self.assertEqual(reasons[0], DecisionReason.NO_ALPHA_SIGNAL.value)
        self.assertEqual(reasons[1], DecisionReason.BLOCK_COST.value)
        self.assertEqual(reasons[2], DecisionReason.ALLOW_SHORT.value)
        self.assertEqual(reasons[3], DecisionReason.BLOCK_RISK_REGIME.value)


if __name__ == "__main__":
    unittest.main()
