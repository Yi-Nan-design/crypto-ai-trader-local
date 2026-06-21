from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.feature_catalog import (
    add_cross_sectional_features,
    add_optional_open_interest_features,
    feature_group_map,
)
from crypto_ai_trader.features import FEATURE_COLUMNS, make_features


def source_frame(rows: int) -> pd.DataFrame:
    close = 100.0 + np.linspace(0.0, 8.0, rows)
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * 300_000,
            "open_datetime": pd.to_datetime(
                np.arange(rows) * 300_000,
                unit="ms",
                utc=True,
            ),
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.linspace(100.0, 300.0, rows),
            "quote_volume": np.linspace(10_000.0, 40_000.0, rows),
            "trades": np.linspace(100.0, 250.0, rows),
            "taker_buy_quote_volume": np.linspace(5_000.0, 22_000.0, rows),
            "funding_rate_8h": np.where(
                np.arange(rows) % 96 == 0,
                0.0001,
                np.nan,
            ),
        }
    )


class FeatureCatalogTests(unittest.TestCase):
    def test_required_feature_groups_have_model_columns(self) -> None:
        groups = feature_group_map()

        self.assertEqual(
            set(groups),
            {
                "technical_features",
                "volatility_features",
                "volume_features",
                "funding_features",
                "open_interest_features",
                "microstructure_features",
                "cross_sectional_features",
                "regime_features",
            },
        )
        for group in (
            "technical_features",
            "volatility_features",
            "volume_features",
            "funding_features",
            "microstructure_features",
            "regime_features",
        ):
            self.assertTrue(set(groups[group]).intersection(FEATURE_COLUMNS))

    def test_new_default_features_do_not_read_appended_future_rows(self) -> None:
        base = source_frame(260)
        extended = source_frame(280)
        base_features = make_features(base, drop_future_na=False)
        extended_features = make_features(extended, drop_future_na=False)
        columns = [
            "realized_volatility_24",
            "funding_rate_level",
            "funding_rate_change",
            "price_volume_divergence_24",
            "spread_proxy",
            "drawdown_from_high_96",
        ]

        pd.testing.assert_frame_equal(
            base_features[columns].reset_index(drop=True),
            extended_features.iloc[: len(base_features)][columns].reset_index(
                drop=True
            ),
        )

    def test_optional_context_features_are_explicit(self) -> None:
        open_interest = add_optional_open_interest_features(
            pd.DataFrame({"open_interest": np.linspace(100.0, 120.0, 60)})
        )
        self.assertIn("open_interest_change", open_interest.columns)
        cross = add_cross_sectional_features(
            pd.DataFrame(
                {
                    "open_time": [1, 1, 2, 2],
                    "symbol": ["BTCUSDT", "ETHUSDT"] * 2,
                    "return_24": [0.01, 0.03, -0.02, -0.01],
                }
            )
        )
        self.assertIn("cross_sectional_momentum_rank", cross.columns)
        self.assertAlmostEqual(
            float(cross.loc[1, "btc_relative_strength"]),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
