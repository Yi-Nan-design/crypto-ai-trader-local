from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.performance_report import (
    annualized_geometric_return,
    calendar_performance,
    evaluate_backtest_performance,
    risk_adjusted_statistics,
)


class PerformanceReportTests(unittest.TestCase):
    def test_interval_aware_metrics_are_finite_and_ordered(self) -> None:
        returns = pd.Series([0.01, -0.004, 0.008, -0.002, 0.006])
        total_return = float((1.0 + returns).prod() - 1.0)
        equity = (1.0 + returns).cumprod()
        drawdown = float(
            (equity / equity.cummax() - 1.0).min()
        )
        metrics = risk_adjusted_statistics(
            returns,
            total_return=total_return,
            max_drawdown=drawdown,
            bar_hours=1.0,
            duration_days=5.0 / 24.0,
        )

        self.assertTrue(np.isfinite(metrics["annualized_return"]))
        self.assertGreater(metrics["sharpe_like"], 0.0)
        self.assertGreater(metrics["sortino_ratio"], 0.0)
        self.assertGreater(metrics["calmar_ratio"], 0.0)
        self.assertEqual(metrics["periods_per_year"], 365.0 * 24.0)

    def test_annualized_return_handles_total_loss_and_short_windows(self) -> None:
        self.assertEqual(
            annualized_geometric_return(-1.0, duration_days=365.0),
            -1.0,
        )
        self.assertEqual(
            annualized_geometric_return(1.0, duration_days=1.0),
            999.0,
        )

    def test_drawdown_includes_initial_equity_peak(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [0, 3_600_000],
                "strategy_return": [-0.10, 0.05],
                "equity": [9_000.0, 9_450.0],
            }
        )

        result = evaluate_backtest_performance(
            frame,
            initial_balance=10_000.0,
        )

        self.assertAlmostEqual(result.max_drawdown, -0.10)

    def test_calendar_grouping_uses_beijing_time(self) -> None:
        open_time = [
            int(
                pd.Timestamp(value).timestamp() * 1000
            )
            for value in [
                "2025-12-31T15:00:00Z",
                "2025-12-31T17:00:00Z",
                "2026-01-31T17:00:00Z",
            ]
        ]
        frame = pd.DataFrame(
            {
                "open_time": open_time,
                "strategy_return": [0.01, 0.02, -0.01],
                "total_cost": [0.001, 0.001, 0.001],
                "executed_notional_position": [1.0, 1.0, 1.0],
                "notional_turnover": [1.0, 0.0, 1.0],
                "notional_exposure": [1.0, 1.0, 1.0],
            }
        )

        yearly = calendar_performance(frame, frequency="year")
        monthly = calendar_performance(frame, frequency="month")

        self.assertEqual(yearly["2025"]["rows"], 1)
        self.assertEqual(yearly["2026"]["rows"], 2)
        self.assertEqual(monthly["2026-01"]["rows"], 1)
        self.assertEqual(monthly["2026-02"]["rows"], 1)

    def test_evaluation_reports_cost_ratio_and_symbol_grain(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.006, 0.0])
        costs = pd.Series([0.001, 0.001, 0.001, 0.0])
        frame = pd.DataFrame(
            {
                "open_time": np.arange(4) * 3_600_000,
                "strategy_return": returns,
                "total_cost": costs,
                "equity": 10_000.0 * (1.0 + returns).cumprod(),
                "executed_notional_position": [1.0, 1.0, 1.0, 0.0],
                "notional_turnover": [1.0, 0.0, 1.0, 1.0],
                "notional_exposure": [1.0, 1.0, 1.0, 0.0],
            }
        )
        frame.attrs["symbol"] = "ETHUSDT"

        result = evaluate_backtest_performance(
            frame,
            initial_balance=10_000.0,
        )

        self.assertAlmostEqual(
            result.total_return,
            float((1.0 + returns).prod() - 1.0),
        )
        self.assertLessEqual(result.max_drawdown, 0.0)
        self.assertGreater(result.fee_ratio, 0.0)
        self.assertGreater(result.gross_return_before_cost, 0.0)
        self.assertIn("ETHUSDT", result.performance_by_symbol)
        self.assertTrue(result.performance_by_month)


if __name__ == "__main__":
    unittest.main()
