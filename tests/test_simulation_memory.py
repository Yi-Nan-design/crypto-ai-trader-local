from __future__ import annotations

import unittest

from crypto_ai_trader.simulation_memory import compact_backtest


class SimulationMemoryTests(unittest.TestCase):
    def test_compact_backtest_preserves_extended_performance_metrics(self) -> None:
        compacted = compact_backtest(
            {
                "total_return": 0.12,
                "annualized_return": 0.26,
                "max_drawdown": -0.08,
                "sharpe_like": 1.4,
                "sortino_ratio": 2.1,
                "calmar_ratio": 3.25,
                "fee_ratio": 0.18,
                "notional_turnover": 4.2,
                "performance_by_month": {"2026-06": {"total_return": 0.03}},
            }
        )

        self.assertEqual(compacted["annualized_return"], 0.26)
        self.assertEqual(compacted["sortino_ratio"], 2.1)
        self.assertEqual(compacted["calmar_ratio"], 3.25)
        self.assertEqual(compacted["fee_ratio"], 0.18)
        self.assertEqual(compacted["notional_turnover"], 4.2)
        self.assertNotIn("performance_by_month", compacted)


if __name__ == "__main__":
    unittest.main()
