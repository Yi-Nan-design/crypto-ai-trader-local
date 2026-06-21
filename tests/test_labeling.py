from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.features import FEATURE_COLUMNS
from crypto_ai_trader.labeling import (
    META_LABEL_COLUMNS,
    LabelConfig,
    MetaLabelConfig,
    add_cross_sectional_rank_labels,
    add_forward_labels,
    add_meta_labels,
    add_path_dependent_labels,
    future_realized_volatility,
    label_columns,
)


class LabelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 102.0, 104.0],
                "event_return_threshold": [0.02] * 5,
                "edge_return_threshold": [0.005] * 5,
            }
        )

    def test_path_labels_record_first_barrier_and_excursions(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0, 100.0],
                "high": [100.0, 103.0, 101.0, 100.0],
                "low": [100.0, 99.0, 96.0, 100.0],
            }
        )

        labeled = add_path_dependent_labels(
            frame,
            horizon=2,
            upper_barrier=0.02,
            lower_barrier=0.02,
        )

        self.assertEqual(int(labeled.loc[0, "triple_barrier_label"]), 1)
        self.assertEqual(int(labeled.loc[0, "triple_barrier_hit_step"]), 1)
        self.assertAlmostEqual(
            float(labeled.loc[0, "max_favorable_excursion_label"]),
            0.03,
        )
        self.assertAlmostEqual(
            float(labeled.loc[0, "max_adverse_excursion_label"]),
            -0.04,
        )
        self.assertTrue(pd.isna(labeled.loc[2, "triple_barrier_label"]))

    def test_cross_sectional_rank_stays_within_timestamp_group(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [1, 1, 1, 2, 2],
                "symbol": ["BTC", "ETH", "SOL", "BTC", "ETH"],
                "future_risk_adjusted_return": [0.1, 0.3, -0.2, -0.5, 0.5],
            }
        )

        labeled = add_cross_sectional_rank_labels(frame, bins=5)
        first_group = labeled[labeled["open_time"] == 1]

        self.assertEqual(
            int(
                first_group.loc[
                    first_group["symbol"] == "ETH",
                    "cross_sectional_relevance_label",
                ].iloc[0]
            ),
            4,
        )
        self.assertLess(
            float(
                first_group.loc[
                    first_group["symbol"] == "SOL",
                    "cross_sectional_rank_label",
                ].iloc[0]
            ),
            float(
                first_group.loc[
                    first_group["symbol"] == "BTC",
                    "cross_sectional_rank_label",
                ].iloc[0]
            ),
        )

    def test_forward_labels_preserve_legacy_edge_semantics(self) -> None:
        result = add_forward_labels(
            self.frame,
            LabelConfig(horizon=2, min_return=0.001, multi_horizon_steps=(1, 2)),
        )

        self.assertAlmostEqual(float(result.loc[0, "future_return"]), 0.03, places=12)
        self.assertAlmostEqual(float(result.loc[0, "long_edge_after_cost"]), 0.025, places=12)
        self.assertEqual(int(result.loc[0, "edge_long_target"]), 1)
        self.assertEqual(int(result.loc[0, "edge_short_target"]), 0)
        self.assertTrue(np.isnan(result.loc[4, "future_return"]))

    def test_risk_adjusted_return_uses_only_future_label_path(self) -> None:
        result = add_forward_labels(
            self.frame,
            LabelConfig(horizon=2, min_return=0.001, multi_horizon_steps=(2,)),
        )
        one_bar_returns = np.array([101.0 / 100.0 - 1.0, 103.0 / 101.0 - 1.0])
        expected_volatility = float(np.sqrt(np.mean(one_bar_returns**2)))
        expected_risk_adjusted = 0.03 / expected_volatility

        self.assertAlmostEqual(
            float(result.loc[0, "future_realized_volatility"]),
            expected_volatility,
            places=12,
        )
        self.assertAlmostEqual(
            float(result.loc[0, "future_risk_adjusted_return"]),
            expected_risk_adjusted,
            places=12,
        )
        self.assertEqual(int(result.loc[0, "risk_adjusted_long_target"]), 1)

    def test_future_volatility_requires_complete_horizon(self) -> None:
        volatility = future_realized_volatility(self.frame["close"], horizon=3)

        self.assertTrue(np.isfinite(volatility.iloc[0]))
        self.assertTrue(np.isnan(volatility.iloc[-2]))
        self.assertTrue(np.isnan(volatility.iloc[-1]))

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LabelConfig(horizon=0)

    def test_meta_labels_score_long_and_short_primary_signals(self) -> None:
        frame = pd.DataFrame(
            {
                "primary_side_signal": [1, -1, 1, 0, -1],
                "future_return": [0.02, -0.03, -0.01, 0.50, np.nan],
                "edge_return_threshold": [0.005] * 5,
            }
        )

        result = add_meta_labels(frame, MetaLabelConfig())

        self.assertEqual(int(result.loc[0, "meta_label"]), 1)
        self.assertEqual(int(result.loc[1, "meta_label"]), 1)
        self.assertEqual(int(result.loc[2, "meta_label"]), 0)
        self.assertTrue(pd.isna(result.loc[3, "meta_label"]))
        self.assertTrue(pd.isna(result.loc[4, "meta_label"]))
        self.assertAlmostEqual(
            float(result.loc[1, "meta_edge_after_cost"]),
            0.025,
        )
        self.assertFalse(bool(result.loc[3, "meta_label_active"]))

    def test_meta_labels_reject_invalid_primary_side(self) -> None:
        frame = pd.DataFrame(
            {
                "primary_side_signal": [2],
                "future_return": [0.01],
                "edge_return_threshold": [0.001],
            }
        )

        with self.assertRaisesRegex(ValueError, "must be -1, 0, 1"):
            add_meta_labels(frame)

    def test_forward_label_pipeline_can_enable_meta_labels(self) -> None:
        frame = self.frame.copy()
        frame["primary_side_signal"] = [1, -1, 0, 1, -1]
        config = LabelConfig(
            horizon=2,
            min_return=0.001,
            multi_horizon_steps=(2,),
            meta_side_column="primary_side_signal",
        )

        result = add_forward_labels(frame, config)

        self.assertTrue(set(META_LABEL_COLUMNS).issubset(result.columns))
        self.assertTrue(
            set(META_LABEL_COLUMNS).issubset(label_columns(config))
        )
        self.assertTrue(
            set(META_LABEL_COLUMNS).isdisjoint(FEATURE_COLUMNS)
        )
        self.assertEqual(int(result.loc[0, "meta_label"]), 1)
        self.assertEqual(int(result.loc[1, "meta_label"]), 0)

    def test_empty_meta_side_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "meta_side_column must not be empty",
        ):
            LabelConfig(meta_side_column=" ")


if __name__ == "__main__":
    unittest.main()
