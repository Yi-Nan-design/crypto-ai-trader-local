from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.backtest import BacktestConfig, run_backtest
from crypto_ai_trader.liquidation import (
    assess_liquidations,
    liquidation_price_distance,
)


class LiquidationModelTests(unittest.TestCase):
    def test_liquidation_distance_uses_leverage_and_margin_assumptions(self) -> None:
        cfg = BacktestConfig(
            leverage=3,
            maintenance_margin_rate=0.005,
            liquidation_buffer=0.01,
        )

        self.assertAlmostEqual(
            liquidation_price_distance(cfg),
            1.0 / 3.0 - 0.005 - 0.01,
        )

    def test_protective_stop_precedes_intrabar_liquidation_but_not_gap(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "open": [100.0, 100.0],
                "high": [100.0, 101.0],
                "low": [100.0, 50.0],
            }
        )
        cfg = BacktestConfig(leverage=3)

        protected = assess_liquidations(
            frame,
            executed_notional_position=np.array([0.6, 0.0]),
            stop_loss=np.array([0.02, 0.02]),
            cfg=cfg,
        )
        unprotected = assess_liquidations(
            frame,
            executed_notional_position=np.array([0.6, 0.0]),
            stop_loss=np.array([0.50, 0.50]),
            cfg=cfg,
        )

        self.assertFalse(protected.triggered[0])
        self.assertTrue(unprotected.triggered[0])
        self.assertFalse(unprotected.gap_triggered[0])

        gap_frame = frame.copy()
        gap_frame.loc[1, "open"] = 50.0
        gap = assess_liquidations(
            gap_frame,
            executed_notional_position=np.array([0.6, 0.0]),
            stop_loss=np.array([0.02, 0.02]),
            cfg=cfg,
        )
        self.assertTrue(gap.triggered[0])
        self.assertTrue(gap.gap_triggered[0])

    def test_backtest_records_liquidation_and_forced_reentry(self) -> None:
        class LongBundle:
            feature_columns = ["close"]
            auxiliary_metadata: dict[str, object] = {}

            def predict_direction_probabilities(
                self, values: np.ndarray
            ) -> dict[str, np.ndarray]:
                rows = len(values)
                up = np.full(rows, 0.90)
                return {
                    "up": up,
                    "long": up,
                    "short": 1.0 - up,
                    "trade": np.ones(rows),
                    "uses_directional_models": np.zeros(rows, dtype=bool),
                    "uses_tradeability_model": np.zeros(rows, dtype=bool),
                }

            def predict_horizon_probabilities(
                self, values: np.ndarray
            ) -> dict[str, np.ndarray]:
                return {}

        frame = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "open": [100.0, 50.0, 100.0],
                "high": [100.0, 101.0, 101.0],
                "low": [100.0, 45.0, 99.0],
                "atr_14": [0.02, 0.02, 0.02],
                "quote_volume_z": [0.0, 0.0, 0.0],
            }
        )
        cfg = BacktestConfig(
            leverage=3,
            stop_loss=0.50,
            dynamic_position_sizing=False,
            max_position_fraction=0.20,
            max_notional_exposure=0.60,
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
            liquidation_fee_rate=0.01,
            cost_filter_enabled=False,
            regime_gate_enabled=False,
            horizon_confirmation_enabled=False,
            horizon_assist_enabled=False,
        )

        result, detail = run_backtest(frame, LongBundle(), cfg)

        self.assertEqual(result.liquidation_events, 1)
        self.assertEqual(result.liquidation_gap_events, 1)
        self.assertGreater(result.liquidation_fee_drag, 0.0)
        self.assertAlmostEqual(
            result.liquidation_forced_turnover,
            float(detail.loc[0, "executed_notional_position"]),
        )
        self.assertGreater(
            float(detail.loc[1, "executed_notional_change"]),
            0.0,
        )
        self.assertEqual(float(detail.loc[0, "end_notional_position"]), 0.0)
        self.assertEqual(detail.loc[0, "liquidation_reason"], "liquidation_gap_breach")


if __name__ == "__main__":
    unittest.main()
