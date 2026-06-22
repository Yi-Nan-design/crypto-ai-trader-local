from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.shadow_learning import (
    apply_shadow_holding_period,
    build_shadow_learning_decision,
    evaluate_low_threshold_strategy_candidate,
    select_low_threshold_strategy_candidate,
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
    def test_shadow_holding_matches_forecast_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shadow_holding_5m.json"
            first, state = apply_shadow_holding_period(
                [
                    {
                        "symbol": "ETHUSDT",
                        "latest_open_time": 100,
                        "shadow_learning": {
                            "latest_signal_active": True,
                            "selected_side": "long",
                            "selected_model": "model",
                            "target_direction": 1,
                            "target_exposure": 0.025,
                            "leverage": 10,
                            "forecast_horizon_bars": 2,
                            "blockers": [],
                        },
                    }
                ],
                state_path=path,
                interval="5m",
            )
            self.assertTrue(
                first[0]["shadow_learning"]["holding_period_active"]
            )
            self.assertEqual(
                state["positions"]["ETHUSDT"]["remaining_hold_bars"],
                2,
            )

            second, state = apply_shadow_holding_period(
                [
                    {
                        "symbol": "ETHUSDT",
                        "latest_open_time": 200,
                        "shadow_learning": {
                            "latest_signal_active": False,
                            "target_direction": 0,
                            "target_exposure": 0.0,
                            "blockers": [
                                "shadow_probability_below_threshold",
                                "shadow_strategy_filter_blocked",
                            ],
                        },
                    }
                ],
                state_path=path,
                interval="5m",
            )
            self.assertTrue(
                second[0]["shadow_learning"]["latest_signal_active"]
            )
            self.assertEqual(
                second[0]["shadow_learning"]["remaining_hold_bars"],
                1,
            )

            third, state = apply_shadow_holding_period(
                [
                    {
                        "symbol": "ETHUSDT",
                        "latest_open_time": 300,
                        "shadow_learning": {
                            "latest_signal_active": False,
                            "target_direction": 0,
                            "target_exposure": 0.0,
                            "blockers": [
                                "shadow_probability_below_threshold"
                            ],
                        },
                    }
                ],
                state_path=path,
                interval="5m",
            )
            self.assertFalse(
                third[0]["shadow_learning"]["latest_signal_active"]
            )
            self.assertNotIn("ETHUSDT", state["positions"])

    def test_low_threshold_strategy_filter_stays_positive_across_gate(self) -> None:
        calibration = pd.DataFrame(
            {
                "future_return": [0.002] * 8 + [-0.004] * 4,
                "platform_strategy_long_score": [0.60] * 8 + [0.0] * 4,
                "liquidity_quality_score": [0.80] * 12,
            }
        )
        gate = pd.DataFrame(
            {
                "future_return": [0.002] * 3 + [-0.004] * 3,
                "platform_strategy_long_score": [0.60] * 3 + [0.0] * 3,
                "liquidity_quality_score": [0.80] * 6,
            }
        )

        candidate = select_low_threshold_strategy_candidate(
            np.full(len(calibration), 0.60),
            calibration,
            np.full(len(gate), 0.60),
            gate,
            direction="long",
            model_name="low_threshold_model",
            cost_buffer=0.00075,
        )

        self.assertIsNotNone(candidate)
        self.assertLessEqual(candidate["threshold"], 0.60)
        self.assertEqual(
            candidate["strategy_profile"],
            "platform_035_liquidity",
        )
        self.assertGreater(
            candidate["calibration"][
                "signal_total_return_after_cost"
            ],
            0.0,
        )
        self.assertGreater(
            candidate["gate"]["signal_total_return_after_cost"],
            0.0,
        )
        test_result = evaluate_low_threshold_strategy_candidate(
            np.full(len(gate), 0.60),
            gate,
            candidate,
        )
        self.assertEqual(test_result["signal_count"], 3.0)
        self.assertGreater(
            test_result["signal_total_return_after_cost"],
            0.0,
        )

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

    def test_ten_x_shadow_uses_bounded_isolated_margin(self) -> None:
        report = {
            "directional_signal_report": {
                "directions": {
                    "long": {
                        "shadow_candidate": {
                            **threshold(
                                value=0.57,
                                count=12,
                                profit_factor=1.4,
                                total_return=0.01,
                                expectancy=0.0008,
                            ),
                            "model_name": "bounded_ten_x",
                            "strategy_profile": "unfiltered",
                        }
                    }
                }
            }
        }
        decision = build_shadow_learning_decision(
            report,
            {
                "long": 0.70,
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
            max_position_fraction=0.025,
            leverage=10,
        )

        self.assertTrue(decision["latest_signal_active"])
        self.assertEqual(decision["leverage"], 10)
        self.assertAlmostEqual(decision["target_exposure"], 0.025)
        self.assertAlmostEqual(
            decision["target_notional_exposure"],
            0.25,
        )

        converted = shadow_portfolio_report(
            {
                "symbol": "ETHUSDT",
                "latest_alpha_prediction": {},
                "optimized_backtest_config": {},
                "shadow_learning": decision,
            }
        )
        self.assertEqual(
            converted["optimized_backtest_config"]["leverage"],
            10,
        )
        self.assertEqual(
            converted["optimized_backtest_config"][
                "max_allowed_leverage"
            ],
            10,
        )
        self.assertEqual(
            converted["optimized_backtest_config"]["margin_type"],
            "ISOLATED",
        )


if __name__ == "__main__":
    unittest.main()
