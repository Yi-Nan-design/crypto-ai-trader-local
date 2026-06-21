from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto_ai_trader.backtest import BacktestConfig
from crypto_ai_trader.contracts import MarketRegime
from crypto_ai_trader.regime import (
    detect_regime_frame,
    latest_regime_state,
    summarize_regime_performance,
)


class RegimeLayerTests(unittest.TestCase):
    @staticmethod
    def _statistical_frame(rows: int = 120) -> pd.DataFrame:
        index = np.arange(rows)
        return pd.DataFrame(
            {
                "open_time": index,
                "atr_14": 0.01 + 0.004 * (index % 4) + 0.00001 * index,
                "quote_volume_z": np.sin(index / 7.0),
                "liquidity_quality_score": 0.55 + 0.1 * np.cos(index / 9.0),
                "return_3": 0.002 * np.sin(index / 5.0),
                "return_24": 0.01 * np.sin(index / 11.0),
                "efficiency_ratio_48": 0.08 + 0.08 * (index % 4),
                "trend_strength_12_48": 0.02 * np.sin(index / 13.0),
            }
        )

    @staticmethod
    def _statistical_config() -> BacktestConfig:
        return BacktestConfig(
            regime_detection_method="walk_forward_kmeans",
            regime_statistical_clusters=4,
            regime_statistical_min_history=40,
            regime_statistical_lookback=80,
            regime_statistical_refit_interval=10,
            regime_statistical_random_seed=42,
        )

    def test_rule_based_detector_emits_explainable_states(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [1, 2, 3, 4, 5],
                "atr_14": [0.01, 0.01, 0.01, 0.04, 0.04],
                "quote_volume_z": [0.0, 0.0, 0.0, 0.0, -3.0],
                "liquidity_quality_score": [0.7, 0.7, 0.7, 0.7, 0.1],
                "return_3": [0.0, 0.0, 0.0, -0.03, 0.0],
                "return_24": [0.0, 0.0, 0.0, -0.05, 0.0],
                "efficiency_ratio_48": [0.3, 0.3, 0.05, 0.3, 0.3],
                "trend_strength_12_48": [0.01, -0.01, 0.0, -0.02, 0.01],
            }
        )

        classified = detect_regime_frame(frame, BacktestConfig())

        self.assertEqual(
            classified["market_regime"].tolist(),
            [
                MarketRegime.TREND_UP.value,
                MarketRegime.TREND_DOWN.value,
                MarketRegime.RANGE.value,
                MarketRegime.CRASH.value,
                MarketRegime.LIQUIDITY_CRISIS.value,
            ],
        )
        self.assertTrue(classified.loc[3, "regime_risk_off"])
        self.assertTrue(classified.loc[4, "regime_risk_off"])
        self.assertEqual(latest_regime_state(pd.concat([frame, classified], axis=1)).regime, MarketRegime.LIQUIDITY_CRISIS)
        self.assertEqual(
            classified["regime_method_used"].unique().tolist(),
            ["rule_based"],
        )
        self.assertTrue((classified["regime_fallback_reason"] == "").all())

    def test_appended_future_rows_do_not_change_past_regimes(self) -> None:
        rows = 120
        base = pd.DataFrame(
            {
                "atr_14": np.linspace(0.005, 0.02, rows),
                "quote_volume_z": np.zeros(rows),
                "liquidity_quality_score": np.full(rows, 0.7),
                "return_3": np.zeros(rows),
                "return_24": np.zeros(rows),
                "efficiency_ratio_48": np.full(rows, 0.25),
                "trend_strength_12_48": np.full(rows, 0.01),
            }
        )
        future = pd.DataFrame(
            {
                "atr_14": np.full(20, 1.0),
                "quote_volume_z": np.full(20, -5.0),
                "liquidity_quality_score": np.zeros(20),
                "return_3": np.full(20, -0.2),
                "return_24": np.full(20, -0.4),
                "efficiency_ratio_48": np.zeros(20),
                "trend_strength_12_48": np.zeros(20),
            }
        )

        original = detect_regime_frame(base, BacktestConfig())
        extended = detect_regime_frame(
            pd.concat([base, future], ignore_index=True),
            BacktestConfig(),
        )

        self.assertEqual(
            original["market_regime"].tolist(),
            extended.iloc[:rows]["market_regime"].tolist(),
        )
        np.testing.assert_allclose(
            original["regime_confidence"],
            extended.iloc[:rows]["regime_confidence"],
        )

    def test_walk_forward_kmeans_never_refits_on_future_rows(self) -> None:
        base = self._statistical_frame()
        future = self._statistical_frame(30)
        future["open_time"] += len(base)
        future["atr_14"] += 5.0
        future["return_24"] -= 0.5
        cfg = self._statistical_config()

        original = detect_regime_frame(base, cfg)
        extended = detect_regime_frame(
            pd.concat([base, future], ignore_index=True),
            cfg,
        )

        self.assertGreater(
            int((original["regime_method_used"] == "walk_forward_kmeans").sum()),
            0,
        )
        pd.testing.assert_frame_equal(
            original.reset_index(drop=True),
            extended.iloc[: len(base)].reset_index(drop=True),
        )

    def test_walk_forward_kmeans_falls_back_when_sklearn_is_unavailable(self) -> None:
        frame = self._statistical_frame(80)
        cfg = self._statistical_config()
        rule = detect_regime_frame(frame, BacktestConfig())

        with patch("crypto_ai_trader.statistical_regime._SklearnKMeans", None):
            classified = detect_regime_frame(frame, cfg)

        pd.testing.assert_frame_equal(
            classified[
                [
                    "market_regime",
                    "regime_confidence",
                    "regime_risk_off",
                    "volatility_state",
                    "liquidity_state",
                    "regime_reason",
                ]
            ],
            rule[
                [
                    "market_regime",
                    "regime_confidence",
                    "regime_risk_off",
                    "volatility_state",
                    "liquidity_state",
                    "regime_reason",
                ]
            ],
        )
        self.assertTrue(
            (classified["regime_method_used"] == "rule_based_fallback").all()
        )
        self.assertEqual(
            classified["regime_fallback_reason"].unique().tolist(),
            ["sklearn_unavailable"],
        )

    def test_crash_and_liquidity_rules_override_statistical_regime(self) -> None:
        frame = self._statistical_frame(90)
        frame.loc[75, ["return_3", "return_24"]] = [-0.08, -0.12]
        frame.loc[76, ["quote_volume_z", "liquidity_quality_score"]] = [-4.0, 0.05]

        classified = detect_regime_frame(frame, self._statistical_config())

        self.assertEqual(
            classified.loc[75, "market_regime"],
            MarketRegime.CRASH.value,
        )
        self.assertEqual(
            classified.loc[76, "market_regime"],
            MarketRegime.LIQUIDITY_CRISIS.value,
        )
        self.assertEqual(
            classified.loc[75, "regime_method_used"],
            "rule_based_risk_override",
        )
        self.assertEqual(
            classified.loc[76, "regime_method_used"],
            "rule_based_risk_override",
        )
        self.assertTrue(classified.loc[75, "regime_risk_off"])
        self.assertTrue(classified.loc[76, "regime_risk_off"])
        self.assertEqual(classified.loc[75, "regime_fallback_reason"], "")
        self.assertEqual(classified.loc[76, "regime_fallback_reason"], "")
        self.assertEqual(classified.loc[75, "regime_override_reason"], "risk_override")
        self.assertEqual(classified.loc[76, "regime_override_reason"], "risk_override")

    def test_crash_wins_when_crash_and_liquidity_overlap(self) -> None:
        frame = self._statistical_frame(90)
        frame.loc[
            75,
            [
                "return_3",
                "return_24",
                "quote_volume_z",
                "liquidity_quality_score",
            ],
        ] = [-0.08, -0.12, -4.0, 0.05]

        classified = detect_regime_frame(frame, self._statistical_config())

        self.assertEqual(
            classified.loc[75, "market_regime"],
            MarketRegime.CRASH.value,
        )
        self.assertEqual(classified.loc[75, "liquidity_state"], "crisis")
        self.assertEqual(
            classified.loc[75, "regime_method_used"],
            "rule_based_risk_override",
        )

    def test_statistical_regime_ignores_forward_label_columns(self) -> None:
        base = self._statistical_frame()
        with_labels = base.copy()
        with_labels["future_return"] = np.linspace(-10.0, 10.0, len(base))
        with_labels["edge_long_target"] = np.arange(len(base)) % 2
        with_labels["meta_label"] = 1
        cfg = self._statistical_config()

        original = detect_regime_frame(base, cfg)
        classified = detect_regime_frame(with_labels, cfg)

        pd.testing.assert_frame_equal(original, classified)

    def test_statistical_regime_config_rejects_invalid_windows(self) -> None:
        invalid = (
            {"regime_detection_method": "unknown"},
            {"regime_statistical_clusters": 1},
            {"regime_statistical_clusters": 9},
            {"regime_statistical_min_history": 10},
            {
                "regime_statistical_min_history": 100,
                "regime_statistical_lookback": 80,
            },
            {"regime_statistical_refit_interval": 0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    BacktestConfig(**values)

    def test_regime_summary_uses_executed_exposure(self) -> None:
        detail = pd.DataFrame(
            {
                "market_regime": ["trend_up", "trend_up", "range"],
                "strategy_return": [0.01, -0.002, 0.0],
                "executed_notional_position": [0.2, 0.2, 0.0],
                "notional_turnover": [0.2, 0.0, 0.0],
                "total_cost": [0.001, 0.0, 0.0],
            }
        )

        summary = summarize_regime_performance(detail)

        self.assertEqual(summary["trend_up"]["active_rows"], 2)
        self.assertEqual(summary["trend_up"]["execution_events"], 1)
        self.assertEqual(summary["range"]["active_rows"], 0)


if __name__ == "__main__":
    unittest.main()
