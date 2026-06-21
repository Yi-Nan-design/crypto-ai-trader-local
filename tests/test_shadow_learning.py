from __future__ import annotations

import unittest

from crypto_ai_trader.shadow_learning import (
    build_shadow_learning_decision,
    select_shadow_threshold_candidate,
    shadow_portfolio_report,
)


def threshold(
    *,
    value: float,
    count: int,
    profit_factor: float,
    total_return: float,
    expectancy: float,
) -> dict:
    return {
        "threshold": value,
        "signal_count": count,
        "signal_profit_factor_after_cost": profit_factor,
        "signal_total_return_after_cost": total_return,
        "signal_expectancy_after_cost": expectancy,
    }


class ShadowLearningTests(unittest.TestCase):
    def test_threshold_selection_requires_positive_validation_quality(self) -> None:
        selected = select_shadow_threshold_candidate(
            [
                threshold(
                    value=0.82,
                    count=7,
                    profit_factor=3.0,
                    total_return=0.01,
                    expectancy=0.001,
                ),
                threshold(
                    value=0.74,
                    count=12,
                    profit_factor=1.5,
                    total_return=0.006,
                    expectancy=0.0005,
                ),
                threshold(
                    value=0.66,
                    count=30,
                    profit_factor=0.9,
                    total_return=-0.01,
                    expectancy=-0.0002,
                ),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["threshold"], 0.74)
        self.assertEqual(selected["signal_count"], 12)

    def test_shadow_short_signal_is_paper_only_and_risk_gated(self) -> None:
        report = {
            "directional_signal_report": {
                "directions": {
                    "short": {
                        "shadow_candidate": {
                            **threshold(
                                value=0.74,
                                count=13,
                                profit_factor=1.54,
                                total_return=0.0049,
                                expectancy=0.00037,
                            ),
                            "model_name": "short_shadow_model",
                        }
                    }
                }
            }
        }
        decision = build_shadow_learning_decision(
            report,
            {
                "long": 0.0,
                "short": 0.80,
                "long_model_available": False,
                "short_model_available": True,
            },
            execution_allowed=True,
            regime_risk_off=False,
            liquidity_score=0.80,
            min_liquidity_score=0.20,
            funding_rate=0.0,
            funding_crowding_limit=0.0005,
        )

        self.assertTrue(decision["eligible"])
        self.assertTrue(decision["latest_signal_active"])
        self.assertEqual(decision["target_direction"], -1)
        self.assertEqual(decision["target_exposure"], 0.05)
        self.assertFalse(decision["safety"]["live_trading_enabled"])

        blocked = build_shadow_learning_decision(
            report,
            {
                "long": 0.0,
                "short": 0.80,
                "long_model_available": False,
                "short_model_available": True,
            },
            execution_allowed=True,
            regime_risk_off=True,
            liquidity_score=0.80,
            min_liquidity_score=0.20,
            funding_rate=0.0,
            funding_crowding_limit=0.0005,
        )
        self.assertFalse(blocked["latest_signal_active"])
        self.assertIn("shadow_regime_risk_off", blocked["blockers"])

    def test_shadow_runtime_rejects_non_positive_validation_edge(self) -> None:
        report = {
            "directional_signal_report": {
                "directions": {
                    "long": {
                        "shadow_candidate": {
                            **threshold(
                                value=0.70,
                                count=20,
                                profit_factor=1.8,
                                total_return=0.0,
                                expectancy=0.0,
                            ),
                            "model_name": "invalid_shadow_model",
                        }
                    }
                }
            }
        }

        decision = build_shadow_learning_decision(
            report,
            {
                "long": 0.90,
                "short": 0.0,
                "long_model_available": True,
                "short_model_available": False,
            },
            execution_allowed=True,
            regime_risk_off=False,
            liquidity_score=0.80,
            min_liquidity_score=0.20,
            funding_rate=0.0,
            funding_crowding_limit=0.0005,
        )

        self.assertFalse(decision["eligible"])
        self.assertFalse(decision["latest_signal_active"])
        self.assertIn(
            "no_validation_qualified_shadow_side",
            decision["blockers"],
        )

    def test_shadow_short_report_maps_to_negative_portfolio_signal(self) -> None:
        converted = shadow_portfolio_report(
            {
                "symbol": "BTCUSDT",
                "latest_up_probability": 0.65,
                "latest_alpha_prediction": {
                    "volatility_forecast": 0.02,
                },
                "optimized_backtest_config": {
                    "trade_side_policy": "none",
                },
                "shadow_learning": {
                    "selected_side": "short",
                    "latest_probability": 0.80,
                    "signal_threshold": 0.74,
                    "latest_signal_active": True,
                    "max_position_fraction": 0.05,
                    "leverage": 1,
                    "blockers": [],
                },
            }
        )

        self.assertAlmostEqual(converted["latest_up_probability"], 0.20)
        self.assertEqual(
            converted["optimized_backtest_config"]["trade_side_policy"],
            "short_only",
        )
        self.assertAlmostEqual(
            converted["optimized_backtest_config"]["short_threshold"],
            0.26,
        )
        self.assertEqual(
            converted["latest_risk_decision"]["max_position_size"],
            0.05,
        )


if __name__ == "__main__":
    unittest.main()
