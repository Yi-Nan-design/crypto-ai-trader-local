from __future__ import annotations

import unittest

from crypto_ai_trader.execution_algorithms import (
    OrderStyle,
    learned_slippage_availability,
    limit_first_plan,
    twap_plan,
    vwap_plan,
)


class ExecutionAlgorithmTests(unittest.TestCase):
    def test_limit_first_keeps_market_as_remainder(self) -> None:
        plan = limit_first_plan(
            1_000.0,
            urgency=0.30,
            maker_fraction=0.70,
            participation_rate=0.01,
        )

        self.assertEqual(plan.slices[0].style, OrderStyle.LIMIT)
        self.assertEqual(plan.slices[1].style, OrderStyle.MARKET)
        self.assertAlmostEqual(
            sum(item.signed_notional_usdt for item in plan.slices),
            1_000.0,
        )
        self.assertFalse(plan.live_orders_allowed)

    def test_twap_and_vwap_preserve_signed_target(self) -> None:
        twap = twap_plan(-900.0, bars=3, participation_rate=0.02)
        vwap = vwap_plan(
            1_000.0,
            expected_volume_profile=[1.0, 2.0, 1.0],
            max_participation_rate=0.01,
        )

        self.assertAlmostEqual(
            sum(item.signed_notional_usdt for item in twap.slices),
            -900.0,
        )
        self.assertAlmostEqual(
            sum(item.signed_notional_usdt for item in vwap.slices),
            1_000.0,
        )
        self.assertGreater(
            vwap.slices[1].signed_notional_usdt,
            vwap.slices[0].signed_notional_usdt,
        )

    def test_learned_slippage_is_an_offline_optional_boundary(self) -> None:
        report = learned_slippage_availability()

        self.assertEqual(report["status"], "interface_ready_not_trained")
        self.assertFalse(report["live_orders_allowed"])


if __name__ == "__main__":
    unittest.main()
