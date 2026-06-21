from __future__ import annotations

import unittest
from unittest.mock import patch
from dataclasses import asdict

import numpy as np
import pandas as pd

from crypto_ai_trader.backtest import BacktestConfig, BacktestResult
from crypto_ai_trader import model_optimization
from crypto_ai_trader.strategy_calibration import (
    build_strategy_calibration_search_space,
    build_no_trade_calibration_fallback,
    calibrate_directional_thresholds,
    cost_efficiency_ratio,
    directional_side_preflight_gate,
    directional_signal_capture_metrics,
    evaluate_strategy_calibration_candidate,
    iter_strategy_calibration_candidates,
    normalize_trade_side_policy,
    replay_independent_validation_gate,
    score_strategy_calibration_candidate,
    side_contribution_gate,
    small_account_risk_profiles,
    split_validation_for_strategy_calibration,
    StrategyCalibrationCandidate,
    validation_trading_gate,
)


class StrategyCalibrationTests(unittest.TestCase):
    @staticmethod
    def _result(**overrides: object) -> BacktestResult:
        values: dict[str, object] = {
            "final_balance": 101.0,
            "total_return": 0.01,
            "max_drawdown": -0.02,
            "trades": 24,
            "leverage": 1,
            "win_rate": 0.55,
            "profit_factor": 1.35,
            "sharpe_like": 0.8,
            "expectancy_per_trade_after_cost": 0.0002,
            "rr_gate_passed": True,
            "long_trades": 12,
            "short_trades": 12,
            "long_total_return": 0.006,
            "short_total_return": 0.004,
            "long_profit_factor": 1.30,
            "short_profit_factor": 1.25,
        }
        values.update(overrides)
        return BacktestResult(**values)

    def test_validation_split_is_chronological_and_purged(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": list(range(400)),
                "marker": [f"validation_{idx}" for idx in range(400)],
            }
        )

        split = split_validation_for_strategy_calibration(
            frame,
            purge_rows=12,
            calibration_fraction=0.55,
            min_rows=80,
        )

        self.assertTrue(split.report["enabled"])
        self.assertFalse(split.report["test_used_for_selection"])
        self.assertEqual(split.report["selection_dataset"], "validation_calibration")
        self.assertEqual(split.report["gate_dataset"], "validation_gate")
        self.assertIsNotNone(split.gate)
        assert split.gate is not None
        self.assertLess(
            int(split.calibration["open_time"].max()),
            int(split.gate["open_time"].min()),
        )
        missing = set(frame["open_time"]) - set(split.calibration["open_time"]) - set(
            split.gate["open_time"]
        )
        self.assertEqual(len(missing), 12)

    def test_small_validation_falls_back_without_gate(self) -> None:
        frame = pd.DataFrame({"open_time": list(range(120))})

        split = split_validation_for_strategy_calibration(
            frame,
            purge_rows=10,
        )

        self.assertFalse(split.report["enabled"])
        self.assertIsNone(split.gate)
        self.assertEqual(len(split.calibration), len(frame))
        self.assertFalse(split.report["test_used_for_selection"])

    def test_search_space_is_deterministic_and_preserves_configuration(self) -> None:
        cfg = BacktestConfig(
            long_threshold=0.61,
            short_threshold=0.39,
            trade_signal_threshold=0.67,
            trade_side_policy="short_only",
        )

        search = build_strategy_calibration_search_space(cfg, compact=True)

        self.assertTrue(search.compact)
        self.assertIn((0.61, 0.39), search.threshold_pairs)
        self.assertIn(0.67, search.trade_thresholds)
        self.assertIn("short_only", search.side_policies)
        self.assertEqual(len(search.threshold_pairs), len(set(search.threshold_pairs)))
        self.assertEqual(len(search.trade_thresholds), len(set(search.trade_thresholds)))

    def test_side_policy_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_trade_side_policy("long"), "long_only")
        self.assertEqual(normalize_trade_side_policy("short"), "short_only")
        self.assertEqual(normalize_trade_side_policy("no_trade"), "none")
        self.assertEqual(normalize_trade_side_policy("unexpected"), "both")

    def test_directional_capture_counts_long_and_short_signals(self) -> None:
        frame = pd.DataFrame(
            {"future_return": [0.02, -0.03, 0.0001, 0.015]}
        )
        probabilities = {
            "long": np.array([0.80, 0.10, 0.51, 0.70]),
            "short": np.array([0.10, 0.85, 0.49, 0.20]),
            "trade": np.array([0.90, 0.90, 0.20, 0.90]),
            "trade_threshold": np.full(4, 0.60),
        }

        metrics = directional_signal_capture_metrics(
            frame,
            probabilities,
            long_threshold=0.65,
            short_threshold=0.35,
            cutoff=0.01,
            neutral_cutoff=0.001,
        )

        self.assertEqual(metrics["long_signal_count"], 2.0)
        self.assertEqual(metrics["short_signal_count"], 1.0)
        self.assertEqual(metrics["long_signal_precision"], 1.0)
        self.assertEqual(metrics["short_signal_precision"], 1.0)
        self.assertEqual(metrics["large_up_capture"], 1.0)
        self.assertEqual(metrics["large_down_capture"], 1.0)
        self.assertEqual(metrics["false_trade_on_neutral_rate"], 0.0)

    def test_directional_preflight_is_explicit_for_pass_and_failure(self) -> None:
        passing_report = {
            "directions": {
                "short": {
                    "status": "trained",
                    "valid_rows": 1000,
                    "valid_positive_count": 120,
                    "valid_positive_rate": 0.12,
                    "ranking": [
                        {
                            "signal_count": 40,
                            "signal_rate": 0.04,
                            "signal_precision": 0.20,
                            "false_signal_on_negative_rate": 0.10,
                            "signal_expectancy_after_cost": 0.0002,
                            "signal_total_return_after_cost": 0.008,
                            "signal_profit_factor_after_cost": 1.4,
                        }
                    ],
                }
            }
        }

        passed = directional_side_preflight_gate(
            passing_report,
            "short",
        )
        failed = directional_side_preflight_gate(None, "short")

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertIn("short_preflight_not_trained", failed["reasons"])

    def test_side_contribution_can_auditably_override_preflight(self) -> None:
        gate = side_contribution_gate(
            self._result(
                trades=12,
                long_trades=12,
                short_trades=0,
                long_total_return=0.01,
                long_profit_factor=1.30,
            ),
            side_policy="long_only",
            min_trades=10,
            min_profit_factor=1.10,
            long_preflight_gate={
                "passed": False,
                "reasons": ["long_preflight_sparse_signal"],
            },
        )

        self.assertTrue(gate["passed"])
        self.assertTrue(
            gate["long_preflight_overridden_by_validation_edge"]
        )
        self.assertTrue(gate["long_candidate_allowed"])

    def test_validation_gate_rejects_cost_dominated_leverage(self) -> None:
        gate = validation_trading_gate(
            self._result(
                leverage=3,
                total_return=0.003,
                trades=20,
                total_cost_drag=0.006,
                expectancy_per_trade_after_cost=0.00005,
            ),
            min_trades=12,
            max_drawdown_floor=-0.10,
            min_profit_factor=1.10,
            min_total_return=0.002,
            max_fee_drag_to_abs_return=0.45,
            min_expectancy_per_trade_after_cost=0.00004,
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "fee_drag_ratio_above_validation_max",
            gate["reasons"],
        )
        self.assertIn(
            "leveraged_fee_drag_ratio_too_high",
            gate["reasons"],
        )

    def test_cost_efficiency_supports_current_and_legacy_cost_fields(
        self,
    ) -> None:
        current = self._result(
            total_return=0.03,
            total_cost_drag=0.01,
            fee_drag=0.20,
            funding_drag=0.20,
        )
        legacy = self._result(
            total_return=0.03,
            total_cost_drag=0.0,
            fee_drag=0.006,
            funding_drag=0.004,
        )
        empty = self._result(
            total_return=0.0,
            total_cost_drag=0.0,
            fee_drag=0.0,
            funding_drag=0.0,
        )

        self.assertAlmostEqual(cost_efficiency_ratio(current), 0.25)
        self.assertAlmostEqual(cost_efficiency_ratio(legacy), 0.25)
        self.assertEqual(cost_efficiency_ratio(empty), 0.0)

    def test_validation_gate_accepts_complete_quality_candidate(self) -> None:
        result = self._result(
            total_return=0.02,
            trades=30,
            profit_factor=1.50,
            max_drawdown=-0.03,
            total_cost_drag=0.002,
            expectancy_per_trade_after_cost=0.0003,
            rr_gate_passed=True,
        )

        gate = validation_trading_gate(
            result,
            min_trades=12,
            max_drawdown_floor=-0.10,
            min_profit_factor=1.10,
            min_total_return=0.002,
            max_fee_drag_to_abs_return=0.45,
            min_expectancy_per_trade_after_cost=0.00004,
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["reasons"], [])

    def test_old_model_optimization_import_path_remains_available(self) -> None:
        self.assertIs(
            model_optimization.directional_side_preflight_gate,
            directional_side_preflight_gate,
        )
        self.assertIs(
            model_optimization.small_account_risk_profiles,
            small_account_risk_profiles,
        )

    def test_small_account_profiles_preserve_compact_search_contract(self) -> None:
        cfg = BacktestConfig(
            max_position_fraction=0.30,
            risk_per_trade=0.01,
        )

        profiles = small_account_risk_profiles(cfg)
        compact = small_account_risk_profiles(cfg, compact=True)

        self.assertEqual(len(profiles), 19)
        self.assertEqual(
            [item["name"] for item in compact],
            [
                "micro_range_grid_long",
                "micro_capital_preservation",
                "platform_dual_trend_breakout",
                "platform_short_breakdown_swing",
                "platform_range_reversion_dual",
                "trend_breakout_long_swing",
                "event_momentum_breakout",
                "short_breakdown_event_only",
            ],
        )
        preservation = next(
            item
            for item in profiles
            if item["name"] == "micro_capital_preservation"
        )
        self.assertLessEqual(
            preservation["params"]["max_position_fraction"],
            0.18,
        )
        self.assertLessEqual(
            preservation["params"]["risk_per_trade"],
            0.0035,
        )
        base = asdict(cfg)
        for profile in profiles:
            rebuilt = BacktestConfig(
                **{
                    **base,
                    **profile["params"],
                }
            )
            self.assertGreater(rebuilt.leverage, 0)

    def test_typed_candidate_iteration_preserves_legacy_order(self) -> None:
        candidates = list(
            iter_strategy_calibration_candidates(
                [
                    {"name": "first", "params": {"leverage": 1}},
                    {"name": "second", "params": {"leverage": 2}},
                ],
                [(0.60, 0.40), (0.65, 0.35)],
                [0.55],
                ["long_only", "short_only"],
            )
        )

        self.assertEqual(
            [
                (
                    item.risk_profile,
                    item.long_threshold,
                    item.side_policy,
                )
                for item in candidates
            ],
            [
                ("first", 0.60, "long_only"),
                ("first", 0.60, "short_only"),
                ("first", 0.65, "long_only"),
                ("first", 0.65, "short_only"),
                ("second", 0.60, "long_only"),
                ("second", 0.60, "short_only"),
                ("second", 0.65, "long_only"),
                ("second", 0.65, "short_only"),
            ],
        )

    def test_score_breakdown_keeps_trade_count_stability_penalty(self) -> None:
        cfg = BacktestConfig()
        candidate = StrategyCalibrationCandidate(
            risk_profile="test",
            risk_params={},
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
            trade_signal_threshold=cfg.trade_signal_threshold,
            side_policy="long_only",
        )
        result = self._result(
            trades=6,
            total_return=0.01,
            expectancy_after_cost=0.0001,
            expectancy_per_trade_after_cost=0.0002,
        )

        score = score_strategy_calibration_candidate(
            candidate=candidate,
            config=cfg,
            base_cfg=cfg,
            result=result,
            capture={
                "large_move_capture": 0.20,
                "neutral_no_trade_rate": 0.50,
                "false_trade_on_neutral_rate": 0.0,
            },
            penalty=0.0,
            validation_rows=120,
            min_trades=12,
        )

        self.assertAlmostEqual(
            score.trade_count_stability_penalty,
            0.09,
        )
        self.assertEqual(
            score.report_fields()["trade_count_stability_penalty"],
            score.trade_count_stability_penalty,
        )

    def test_typed_evaluation_serializes_legacy_ranking_fields(self) -> None:
        frame = pd.DataFrame(
            {"future_return": np.linspace(-0.01, 0.01, 120)}
        )
        candidate = StrategyCalibrationCandidate(
            risk_profile="typed_test",
            risk_params={},
            long_threshold=0.57,
            short_threshold=0.43,
            trade_signal_threshold=0.57,
            side_policy="long_only",
        )
        result = self._result(
            trades=24,
            long_trades=24,
            short_trades=0,
            long_total_return=0.01,
            long_profit_factor=1.35,
            trade_side_policy="long_only",
        )
        direction_prob = {
            "up": np.full(len(frame), 0.60),
            "long": np.full(len(frame), 0.60),
            "short": np.full(len(frame), 0.40),
            "trade": np.full(len(frame), 0.80),
            "uses_directional_models": np.zeros(
                len(frame),
                dtype=bool,
            ),
        }

        with patch(
            "crypto_ai_trader.strategy_calibration.run_backtest",
            return_value=(result, pd.DataFrame()),
        ):
            evaluation = evaluate_strategy_calibration_candidate(
                frame,
                object(),  # type: ignore[arg-type]
                BacktestConfig(),
                candidate,
                direction_prob,
                cutoff=0.005,
                label_min_return=0.001,
                min_trades=12,
                max_drawdown_floor=-0.10,
                min_profit_factor=1.10,
                long_preflight_gate={"passed": True, "reasons": []},
                short_preflight_gate={"passed": True, "reasons": []},
            )

        report = evaluation.to_report()
        self.assertTrue(evaluation.passed)
        self.assertEqual(report["risk_profile"], "typed_test")
        self.assertEqual(report["side_policy"], "long_only")
        self.assertEqual(
            report["calibration_backtest"],
            report["backtest"],
        )
        self.assertEqual(
            report["calibration_validation_trading_gate"],
            report["validation_trading_gate"],
        )
        self.assertIn("score", report)
        self.assertIn("threshold_stability_penalty", report)

    def test_no_trade_finalization_is_explicit_and_typed(self) -> None:
        frame = pd.DataFrame(
            {"future_return": np.zeros(40, dtype=float)}
        )
        no_trade_result = self._result(
            total_return=0.0,
            trades=0,
            profit_factor=0.0,
            expectancy_per_trade_after_cost=0.0,
            rr_gate_passed=False,
            long_trades=0,
            short_trades=0,
        )

        with patch(
            "crypto_ai_trader.strategy_calibration.run_backtest",
            return_value=(no_trade_result, pd.DataFrame()),
        ):
            finalized = build_no_trade_calibration_fallback(
                frame,
                None,
                object(),  # type: ignore[arg-type]
                BacktestConfig(),
                min_trades=12,
                max_drawdown_floor=-0.10,
                min_profit_factor=1.10,
            )

        self.assertEqual(
            finalized.best["risk_profile"],
            "no_trade_recommended",
        )
        self.assertTrue(
            finalized.best["fallback_no_trade_recommended"]
        )
        self.assertFalse(
            finalized.best["validation_trading_gate"]["passed"]
        )
        self.assertEqual(
            finalized.validation_gate_evaluated_count,
            0,
        )
        self.assertEqual(finalized.final_gate_passed_count, 0)

    def test_independent_gate_finalization_preserves_calibration_audit(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {"future_return": np.linspace(-0.01, 0.01, 120)}
        )
        cfg = BacktestConfig(trade_side_policy="long_only")
        best = {
            "risk_profile": "typed_test",
            "side_policy": "long_only",
            "backtest_config": asdict(cfg),
            "calibration_validation_trading_gate": {
                "passed": True,
                "reasons": [],
            },
        }
        gate_result = self._result(
            total_return=0.02,
            trades=24,
            profit_factor=1.40,
            expectancy_per_trade_after_cost=0.0003,
            rr_gate_passed=True,
            long_trades=24,
            short_trades=0,
            long_total_return=0.02,
            long_profit_factor=1.40,
            trade_side_policy="long_only",
        )
        direction_prob = {
            "up": np.full(len(frame), 0.60),
            "long": np.full(len(frame), 0.60),
            "short": np.full(len(frame), 0.40),
            "trade": np.full(len(frame), 0.80),
            "uses_directional_models": np.zeros(
                len(frame),
                dtype=bool,
            ),
        }

        with patch(
            "crypto_ai_trader.strategy_calibration.run_backtest",
            return_value=(gate_result, pd.DataFrame()),
        ):
            finalized = replay_independent_validation_gate(
                best,
                frame,
                object(),  # type: ignore[arg-type]
                direction_prob,
                gate_cutoff=0.005,
                label_min_return=0.001,
                min_trades=12,
                max_drawdown_floor=-0.10,
                min_profit_factor=1.10,
                long_preflight_gate={
                    "passed": True,
                    "reasons": [],
                },
                short_preflight_gate={
                    "passed": True,
                    "reasons": [],
                },
            )

        gate = finalized.best["validation_trading_gate"]
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["calibration_gate_passed"])
        self.assertEqual(gate["decision_dataset"], "validation_gate")
        self.assertEqual(
            finalized.validation_gate_evaluated_count,
            1,
        )
        self.assertEqual(finalized.final_gate_passed_count, 1)

    def test_calibration_executor_respects_eval_limit_without_test_data(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "future_return": np.linspace(-0.01, 0.01, 120),
            }
        )

        class FakeBundle:
            def predict_direction_probabilities(
                self,
                values: np.ndarray,
            ) -> dict[str, np.ndarray]:
                rows = len(values)
                return {
                    "up": np.full(rows, 0.60),
                    "long": np.full(rows, 0.60),
                    "short": np.full(rows, 0.40),
                    "trade": np.full(rows, 0.80),
                    "uses_directional_models": np.zeros(
                        rows,
                        dtype=bool,
                    ),
                }

        passing_result = self._result(
            trades=24,
            long_trades=24,
            short_trades=0,
            long_total_return=0.01,
            long_profit_factor=1.35,
            trade_side_policy="long_only",
        )
        with (
            patch(
                "crypto_ai_trader.strategy_calibration.feature_matrix",
                return_value=(
                    np.zeros((len(frame), 1)),
                    np.zeros(len(frame)),
                    ["feature"],
                ),
            ),
            patch(
                "crypto_ai_trader.strategy_calibration.run_backtest",
                return_value=(passing_result, pd.DataFrame()),
            ),
        ):
            report = calibrate_directional_thresholds(
                frame,
                FakeBundle(),  # type: ignore[arg-type]
                BacktestConfig(),
                label_min_return=0.001,
                min_trades=12,
                max_drawdown_floor=-0.10,
                min_profit_factor=1.10,
                force_compact=True,
                max_threshold_evals=1,
                long_preflight_gate={"passed": True, "reasons": []},
                short_preflight_gate={"passed": True, "reasons": []},
            )

        self.assertEqual(report["searched"], 1)
        self.assertEqual(
            report["strategy_calibration_contract_version"],
            "2026-06-20-v1",
        )
        self.assertEqual(
            report["strategy_calibration_engine_contract_version"],
            "2026-06-20-typed-v1",
        )
        self.assertEqual(
            report["risk_profile_catalog_schema_version"],
            1,
        )
        self.assertEqual(
            report["risk_profile_catalog_version"],
            "2026-06-20-v1",
        )
        self.assertTrue(
            report["risk_profile_catalog_path"].endswith(
                "config\\strategy_calibration_profiles.toml"
            )
        )
        self.assertFalse(report["test_used_for_selection"])
        self.assertEqual(report["selection_dataset"], "validation_full")
        self.assertEqual(report["gate_dataset"], "same_as_selection")
        self.assertFalse(report["separate_gate_enabled"])
        self.assertEqual(
            report["best"]["risk_profile"],
            "micro_range_grid_long",
        )
        self.assertEqual(report["best"]["side_policy"], "long_only")


if __name__ == "__main__":
    unittest.main()
