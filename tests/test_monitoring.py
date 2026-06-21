from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.monitoring import (
    build_monitoring_snapshot,
    build_feature_reference,
    calibration_metrics,
    feature_drift_from_reference,
    ks_statistic,
    population_stability_index,
    rolling_performance,
)
from crypto_ai_trader.scheduled_optimizer import (
    acknowledge_monitoring_triggers,
    monitoring_triggered_targets,
)
from crypto_ai_trader.monitoring import (
    MONITORING_ALGORITHM_VERSION,
    MONITORING_SCHEMA_VERSION,
)


def thresholds() -> SimpleNamespace:
    return SimpleNamespace(
        monitoring_recent_rows=100,
        monitoring_psi_threshold=0.25,
        monitoring_ks_threshold=0.20,
        monitoring_min_confidence=0.12,
        monitoring_max_ece=0.15,
        monitoring_min_rolling_sharpe=0.0,
        monitoring_max_drawdown=0.08,
        monitoring_return_deviation=0.03,
        monitoring_regime_shift_threshold=0.35,
    )


class MonitoringTests(unittest.TestCase):
    def test_identical_distributions_have_no_drift(self) -> None:
        values = np.linspace(-1.0, 1.0, 200)

        self.assertAlmostEqual(population_stability_index(values, values), 0.0)
        self.assertAlmostEqual(ks_statistic(values, values), 0.0)

    def test_constant_reference_shift_is_visible_to_psi_and_compressed_reference(self) -> None:
        reference = np.zeros(200)
        current = np.ones(100)

        self.assertGreater(population_stability_index(reference, current), 1.0)
        profile = build_feature_reference(
            pd.DataFrame({"feature_a": reference}),
            ["feature_a"],
        )
        drift = feature_drift_from_reference(
            profile,
            pd.DataFrame({"feature_a": current}),
        )
        self.assertGreater(drift[0].psi, 1.0)
        self.assertGreater(drift[0].ks, 0.9)

    def test_shifted_distributions_trigger_retraining(self) -> None:
        rng = np.random.default_rng(42)
        reference = pd.DataFrame(
            {
                "feature_a": rng.normal(0.0, 1.0, 500),
                "micro_trend_regime": np.zeros(500),
            }
        )
        current = pd.DataFrame(
            {
                "feature_a": rng.normal(3.0, 1.0, 200),
                "micro_trend_regime": np.ones(200),
            }
        )
        detail = pd.DataFrame(
            {
                "open_time": np.arange(100) * 300_000,
                "strategy_return": np.full(100, -0.001),
                "notional_turnover": np.ones(100),
            }
        )

        snapshot = build_monitoring_snapshot(
            reference_frame=reference,
            current_frame=current,
            feature_columns=["feature_a"],
            calibration_targets=np.tile([0, 1], 100),
            calibration_probabilities=np.full(200, 0.5),
            recent_detail=detail,
            baseline_backtest={
                "total_return": 0.05,
                "max_drawdown": -0.02,
                "profit_factor": 1.4,
                "win_rate": 0.55,
            },
            recent_backtest={
                "total_return": -0.05,
                "max_drawdown": -0.10,
                "profit_factor": 0.6,
                "win_rate": 0.40,
            },
            cfg=thresholds(),
            frozen_backtest={
                "total_return": 0.05,
                "max_drawdown": -0.02,
                "profit_factor": 1.4,
                "win_rate": 0.55,
            },
            paper_metrics={
                "total_return": -0.05,
                "max_drawdown": -0.10,
                "profit_factor": 0.6,
                "win_rate": 0.40,
            },
            paper_status="comparable",
        )

        self.assertTrue(snapshot["retraining"]["triggered"])
        self.assertIn(
            "feature_psi_exceeded",
            snapshot["retraining"]["reasons"],
        )
        self.assertIn(
            "market_regime_distribution_shift",
            snapshot["retraining"]["reasons"],
        )
        self.assertIn(
            "paper_return_below_frozen_test",
            snapshot["retraining"]["reasons"],
        )
        self.assertIn(
            "recent_return_below_equal_window_baseline",
            snapshot["retraining"]["reasons"],
        )

    def test_calibration_metrics_reward_correct_confident_predictions(self) -> None:
        good = calibration_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.05, 0.10, 0.90, 0.95]),
        )
        bad = calibration_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.95, 0.90, 0.10, 0.05]),
        )

        self.assertLess(good.brier_score, bad.brier_score)
        self.assertLess(
            good.expected_calibration_error,
            bad.expected_calibration_error,
        )

    def test_rolling_performance_uses_initial_equity_peak(self) -> None:
        detail = pd.DataFrame(
            {
                "open_time": [0, 300_000],
                "strategy_return": [-0.10, 0.05],
                "notional_turnover": [1.0, 0.0],
                "equity": [9_000.0, 9_450.0],
            }
        )

        rolling = rolling_performance(detail, window=2)

        self.assertAlmostEqual(rolling.total_return, -0.055)
        self.assertAlmostEqual(rolling.max_drawdown, -0.10)
        self.assertEqual(rolling.trades, 1)

    def test_scheduled_optimizer_prioritizes_triggered_monitoring_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reports = Path(temp_dir)
            payload = {
                "schema_version": MONITORING_SCHEMA_VERSION,
                "algorithm_version": MONITORING_ALGORITHM_VERSION,
                "symbol": "ETHUSDT",
                "interval": "5m",
                "retraining": {
                    "triggered": True,
                    "active": True,
                    "acknowledged": False,
                    "severity": "high",
                    "reasons": ["feature_psi_exceeded"],
                    "trigger_id": "abc123",
                    "valid_until_beijing": "2099-01-01T00:00:00+08:00",
                },
            }
            (reports / "ETHUSDT_5m_monitoring.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            targets, details = monitoring_triggered_targets(
                reports,
                allowed_symbols=["ETHUSDT", "BNBUSDT"],
            )

            self.assertEqual(targets, [("ETHUSDT", "5m")])
            self.assertEqual(details[0]["severity"], "high")

            acknowledgements = acknowledge_monitoring_triggers(
                reports,
                {("ETHUSDT", "5m")},
                optimization_report="reports/model_optimization_test.json",
            )
            updated = json.loads(
                (reports / "ETHUSDT_5m_monitoring.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(acknowledgements), 1)
            self.assertFalse(updated["retraining"]["active"])
            self.assertTrue(updated["retraining"]["acknowledged"])

            expired = dict(payload)
            expired["symbol"] = "BNBUSDT"
            expired["retraining"] = dict(payload["retraining"])
            expired["retraining"]["valid_until_beijing"] = (
                "2000-01-01T00:00:00+08:00"
            )
            (reports / "BNBUSDT_5m_monitoring.json").write_text(
                json.dumps(expired),
                encoding="utf-8",
            )
            targets, _ = monitoring_triggered_targets(
                reports,
                allowed_symbols=["ETHUSDT", "BNBUSDT"],
            )
            self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
