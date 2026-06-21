from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.backtest import BacktestConfig, cost_edge_filter
from crypto_ai_trader.cost_model import (
    estimate_execution_costs,
    infer_bar_hours,
    simulate_execution_path,
)
from crypto_ai_trader.exchange_rules import normalize_order_notional_usdt
from crypto_ai_trader.features import feature_only_matrix, make_features


class CostModelTests(unittest.TestCase):
    def test_infer_bar_hours_from_open_time(self) -> None:
        frame = pd.DataFrame({"open_time": [0, 3_600_000, 7_200_000]})

        self.assertEqual(infer_bar_hours(frame), 1.0)

    def test_cost_breakdown_preserves_legacy_total(self) -> None:
        frame = pd.DataFrame({"open_time": [0, 3_600_000, 7_200_000]})
        cfg = BacktestConfig(
            fee_rate=0.001,
            slippage_rate=0.002,
            funding_rate_buffer=0.0008,
        )
        notional = np.array([0.2, -0.2, 0.0])

        costs = estimate_execution_costs(frame, notional, cfg)

        np.testing.assert_allclose(costs.notional_turnover, [0.2, 0.4, 0.2])
        np.testing.assert_allclose(costs.commission_cost, [0.0002, 0.0004, 0.0002])
        np.testing.assert_allclose(costs.slippage_cost, [0.0004, 0.0008, 0.0004])
        np.testing.assert_allclose(costs.trade_cost, costs.commission_cost + costs.slippage_cost)
        np.testing.assert_allclose(costs.total_cost, costs.trade_cost + costs.funding_cost)
        self.assertAlmostEqual(costs.funding_period_fraction, 0.125)
        self.assertAlmostEqual(float(costs.funding_cost.sum()), 0.00004)

    def test_default_execution_path_fills_target_immediately(self) -> None:
        target = np.array([0.2, -0.2, 0.0])

        path = simulate_execution_path(target, BacktestConfig())

        np.testing.assert_allclose(path.delayed_target_notional, target)
        np.testing.assert_allclose(path.executed_notional_position, target)
        np.testing.assert_allclose(path.fill_ratio, np.ones(3))
        self.assertFalse(path.minimum_order_rejected.any())

    def test_latency_partial_fill_and_minimum_order_are_applied(self) -> None:
        cfg = BacktestConfig(
            execution_latency_bars=1,
            partial_fill_ratio=0.5,
            min_order_notional_fraction=0.1,
        )

        path = simulate_execution_path(np.array([1.0, 1.0, 0.0, 0.0]), cfg)

        np.testing.assert_allclose(path.delayed_target_notional, [0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(path.executed_notional_position, [0.0, 0.5, 0.75, 0.375])
        np.testing.assert_allclose(path.fill_ratio, [1.0, 0.5, 0.5, 0.5])
        self.assertFalse(path.minimum_order_rejected.any())

        rejected = simulate_execution_path(
            np.array([0.05]),
            BacktestConfig(min_order_notional_fraction=0.1),
        )
        np.testing.assert_allclose(rejected.executed_notional_position, [0.0])
        self.assertTrue(rejected.minimum_order_rejected[0])

    def test_forced_flat_resets_position_and_charges_reentry_turnover(self) -> None:
        cfg = BacktestConfig(
            fee_rate=0.001,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
        )
        target = np.array([1.0, 1.0, 1.0])

        path = simulate_execution_path(
            target,
            cfg,
            forced_flat_after=np.array([True, False, False]),
        )
        costs = estimate_execution_costs(
            pd.DataFrame({"open_time": [0, 1, 2]}),
            target,
            cfg,
            forced_flat_after=np.array([True, False, False]),
        )

        np.testing.assert_allclose(path.executed_notional_change, [1.0, 1.0, 0.0])
        np.testing.assert_allclose(path.forced_flat_turnover, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(costs.notional_turnover, [2.0, 1.0, 0.0])
        np.testing.assert_allclose(costs.commission_cost, [0.002, 0.001, 0.0])

    def test_maker_taker_mix_changes_commission_and_slippage(self) -> None:
        frame = pd.DataFrame({"open_time": [0]})
        cfg = BacktestConfig(
            fee_rate=0.001,
            maker_fee_rate=0.0002,
            maker_fill_fraction=0.75,
            slippage_rate=0.002,
            funding_rate_buffer=0.0,
        )

        costs = estimate_execution_costs(frame, np.array([1.0]), cfg)

        np.testing.assert_allclose(costs.maker_notional_turnover, [0.75])
        np.testing.assert_allclose(costs.taker_notional_turnover, [0.25])
        np.testing.assert_allclose(costs.commission_cost, [0.0004])
        np.testing.assert_allclose(costs.slippage_cost, [0.0005])

    def test_historical_funding_is_causal_and_side_aware(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [0, 3_600_000, 7_200_000, 10_800_000],
                "close": [100.0] * 4,
                "funding_rate_8h": [0.0008, np.nan, -0.0004, np.nan],
            }
        )
        cfg = BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.5,
        )

        costs = estimate_execution_costs(
            frame,
            np.array([1.0, -1.0, 1.0, -1.0]),
            cfg,
        )

        self.assertEqual(costs.funding_rate_source, "historical_funding_rate_8h")
        np.testing.assert_allclose(
            costs.funding_rate_used,
            [0.0008, 0.0008, -0.0004, -0.0004],
        )
        np.testing.assert_allclose(
            costs.funding_cost,
            [0.0001, -0.0001, -0.00005, 0.00005],
        )

        prefix = estimate_execution_costs(
            frame.iloc[:3].copy(),
            np.ones(3),
            cfg,
        )
        extended = pd.concat(
            [
                frame.iloc[:3],
                pd.DataFrame(
                    {
                        "open_time": [10_800_000],
                        "close": [100.0],
                        "funding_rate_8h": [0.25],
                    }
                ),
            ],
            ignore_index=True,
        )
        extended_costs = estimate_execution_costs(extended, np.ones(4), cfg)
        np.testing.assert_allclose(
            prefix.funding_rate_used,
            extended_costs.funding_rate_used[:3],
        )

    def test_exact_funding_payments_are_not_prorated(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [0, 3_600_000, 7_200_000],
                "close": [100.0] * 3,
                "funding_payment_rate": [np.nan, 0.001, 0.001],
            }
        )
        cfg = BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.5,
        )

        costs = estimate_execution_costs(
            frame,
            np.array([1.0, 1.0, -1.0]),
            cfg,
        )

        self.assertEqual(costs.funding_rate_source, "historical_funding_payment")
        self.assertEqual(costs.funding_settlement_mode, "event")
        np.testing.assert_allclose(costs.funding_rate_used, [0.0, 0.001, 0.001])
        np.testing.assert_allclose(
            costs.execution_path.position_before_execution,
            [0.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(costs.funding_cost, [0.0, 0.001, 0.001])

    def test_liquidity_capacity_limits_fills_and_increases_slippage(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": [0, 3_600_000],
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "quote_volume": [1_000.0, 1_000.0],
            }
        )
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0002,
            funding_rate_buffer=0.0,
            liquidity_execution_enabled=True,
            max_bar_participation_rate=0.10,
            slippage_impact_coefficient=1.0,
            max_dynamic_slippage_rate=0.02,
        )

        costs = estimate_execution_costs(frame, np.array([0.05, 0.05]), cfg)

        np.testing.assert_allclose(
            costs.execution_path.executed_notional_position,
            [0.01, 0.02],
        )
        np.testing.assert_allclose(
            costs.execution_path.liquidity_fill_ratio,
            [0.20, 0.25],
        )
        self.assertTrue(costs.execution_path.liquidity_limited.all())
        np.testing.assert_allclose(
            costs.market_participation_rate,
            [0.10, 0.10],
        )
        self.assertTrue((costs.effective_slippage_rate > cfg.slippage_rate).all())
        self.assertTrue(
            (costs.effective_slippage_rate <= cfg.max_dynamic_slippage_rate).all()
        )

        extended = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "open_time": [7_200_000],
                        "open": [100.0],
                        "high": [120.0],
                        "low": [80.0],
                        "close": [100.0],
                        "quote_volume": [1_000_000_000.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        extended_costs = estimate_execution_costs(
            extended,
            np.array([0.05, 0.05, 0.05]),
            cfg,
        )
        np.testing.assert_allclose(
            costs.effective_slippage_rate,
            extended_costs.effective_slippage_rate[:2],
        )

    def test_cost_edge_filter_uses_dynamic_liquidity_cost(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "quote_volume": [1_000.0, 1_000_000.0],
                "atr_14": [0.10, 0.10],
            }
        )
        cfg = BacktestConfig(
            liquidity_execution_enabled=True,
            max_bar_participation_rate=0.10,
            slippage_impact_coefficient=1.0,
            max_dynamic_slippage_rate=0.02,
        )

        passed, required = cost_edge_filter(frame, cfg)

        self.assertTrue(passed.all())
        self.assertGreater(required[0], required[1])
        self.assertGreater(required[0], cfg.fee_rate + cfg.slippage_rate)

    def test_feature_pipeline_preserves_sparse_funding_as_execution_context(self) -> None:
        rows = 260
        rng = np.random.default_rng(42)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, rows)))
        open_price = np.r_[close[0], close[:-1]]
        volume = rng.uniform(8.0, 14.0, rows)
        frame = pd.DataFrame(
            {
                "open_time": np.arange(rows) * 3_600_000,
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.003,
                "low": np.minimum(open_price, close) * 0.997,
                "close": close,
                "volume": volume,
                "quote_volume": close * volume,
                "trades": rng.integers(15, 35, rows),
                "funding_rate_8h": np.where(
                    np.arange(rows) % 8 == 0,
                    0.0001,
                    np.nan,
                ),
                "exchange_available": np.where(
                    np.arange(rows) == rows - 1,
                    False,
                    True,
                ),
                "exchange_gap_before_bars": np.where(
                    np.arange(rows) == rows - 1,
                    1,
                    0,
                ),
                "exchange_unavailable_reason": np.where(
                    np.arange(rows) == rows - 1,
                    "test_outage",
                    "",
                ),
            }
        )

        features = make_features(frame, drop_future_na=False)
        _, feature_columns = feature_only_matrix(features)

        self.assertGreater(len(features), 0)
        self.assertIn("funding_rate_8h", features.columns)
        self.assertTrue(features["funding_rate_8h"].isna().any())
        self.assertNotIn("funding_rate_8h", feature_columns)
        self.assertIn("exchange_available", features.columns)
        self.assertIn("exchange_gap_before_bars", features.columns)
        self.assertIn("exchange_unavailable_reason", features.columns)
        self.assertNotIn("exchange_available", feature_columns)
        self.assertNotIn("exchange_gap_before_bars", feature_columns)
        self.assertNotIn("exchange_unavailable_reason", feature_columns)

    def test_exchange_order_rules_round_and_reject_orders(self) -> None:
        cfg = BacktestConfig(
            initial_balance=1_000.0,
            exchange_min_notional_usdt=5.0,
            exchange_min_quantity=0.02,
            exchange_quantity_step=0.01,
        )

        normalized = normalize_order_notional_usdt(10.9, 100.0, cfg)
        self.assertTrue(normalized.accepted)
        self.assertAlmostEqual(normalized.normalized_quantity, 0.10)
        self.assertAlmostEqual(normalized.normalized_notional_usdt, 10.0)
        self.assertAlmostEqual(normalized.rounding_loss_usdt, 0.9)

        path = simulate_execution_path(
            np.array([0.0109]),
            cfg,
            prices=np.array([100.0]),
        )
        np.testing.assert_allclose(path.executed_notional_position, [0.01])
        np.testing.assert_allclose(path.exchange_order_quantity, [0.10])
        np.testing.assert_allclose(path.quantity_rounding_loss_usdt, [0.9])
        self.assertFalse(path.exchange_filter_rejected[0])

        rejected = simulate_execution_path(
            np.array([0.004]),
            cfg,
            prices=np.array([100.0]),
        )
        self.assertTrue(rejected.minimum_order_rejected[0])
        self.assertTrue(rejected.exchange_filter_rejected[0])
        self.assertEqual(
            rejected.exchange_rejection_reason[0],
            "below_exchange_min_notional",
        )

    def test_exchange_max_quantity_is_split_across_bars(self) -> None:
        cfg = BacktestConfig(
            initial_balance=1_000.0,
            exchange_max_quantity=0.05,
            exchange_quantity_step=0.01,
        )

        normalized = normalize_order_notional_usdt(20.0, 100.0, cfg)
        self.assertTrue(normalized.accepted)
        self.assertTrue(normalized.maximum_quantity_limited)
        self.assertAlmostEqual(normalized.normalized_quantity, 0.05)
        self.assertEqual(normalized.reason, "quantity_capped_to_exchange_max")

        path = simulate_execution_path(
            np.array([0.02, 0.02, 0.02, 0.02]),
            cfg,
            prices=np.full(4, 100.0),
        )
        np.testing.assert_allclose(
            path.executed_notional_position,
            [0.005, 0.010, 0.015, 0.020],
        )
        self.assertTrue(path.maximum_quantity_limited[:3].all())

    def test_exchange_downtime_blocks_changes_but_not_forced_flat(self) -> None:
        cfg = BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
        )
        path = simulate_execution_path(
            np.array([0.0, 1.0, 1.0]),
            cfg,
            execution_available=np.array([True, False, True]),
            exchange_unavailable_reason=np.array(
                ["", "kline_gap_recovery", ""],
                dtype=object,
            ),
        )
        np.testing.assert_allclose(
            path.executed_notional_position,
            [0.0, 0.0, 1.0],
        )
        self.assertTrue(path.exchange_downtime_blocked[1])
        self.assertEqual(
            path.exchange_rejection_reason[1],
            "kline_gap_recovery",
        )

        liquidated = simulate_execution_path(
            np.array([1.0, 1.0]),
            cfg,
            execution_available=np.array([True, False]),
            exchange_unavailable_reason=np.array(
                ["", "exchange_explicitly_unavailable"],
                dtype=object,
            ),
            forced_flat_after=np.array([False, True]),
        )
        self.assertAlmostEqual(liquidated.forced_flat_turnover[1], 1.0)

    def test_cost_model_rejects_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            estimate_execution_costs(
                pd.DataFrame({"open_time": [0, 1]}),
                np.array([0.1]),
                BacktestConfig(),
            )

    def test_backtest_config_rejects_negative_cost_rates(self) -> None:
        for field in (
            "fee_rate",
            "maker_fee_rate",
            "slippage_rate",
            "funding_rate_buffer",
            "min_order_notional_fraction",
            "exchange_min_notional_usdt",
            "exchange_min_quantity",
            "exchange_max_quantity",
            "exchange_quantity_step",
            "exchange_price_tick_size",
            "max_bar_participation_rate",
            "slippage_impact_coefficient",
            "max_dynamic_slippage_rate",
            "maintenance_margin_rate",
            "liquidation_buffer",
            "liquidation_fee_rate",
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    BacktestConfig(**{field: -0.001})

    def test_backtest_config_rejects_invalid_execution_settings(self) -> None:
        for field in ("maker_fill_fraction", "partial_fill_ratio"):
            for value in (-0.1, 1.1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        BacktestConfig(**{field: value})
        for value in (-1, 1.5):
            with self.subTest(execution_latency_bars=value):
                with self.assertRaises(ValueError):
                    BacktestConfig(execution_latency_bars=value)
        with self.assertRaises(ValueError):
            BacktestConfig(initial_balance=0.0)
        with self.assertRaises(ValueError):
            BacktestConfig(
                liquidity_execution_enabled=True,
                max_bar_participation_rate=0.0,
            )
        with self.assertRaises(ValueError):
            BacktestConfig(
                slippage_rate=0.002,
                max_dynamic_slippage_rate=0.001,
            )
        with self.assertRaises(ValueError):
            BacktestConfig(liquidity_lookback_bars=0)
        with self.assertRaises(ValueError):
            BacktestConfig(exchange_gap_recovery_bars=-1)

    def test_backtest_config_rejects_immediate_liquidation_assumptions(self) -> None:
        with self.assertRaises(ValueError):
            BacktestConfig(
                leverage=3,
                maintenance_margin_rate=0.30,
                liquidation_buffer=0.04,
            )


if __name__ == "__main__":
    unittest.main()
