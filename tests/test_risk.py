from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from crypto_ai_trader.backtest import BacktestConfig, run_backtest
from crypto_ai_trader.contracts import RiskDecision, RiskLevel, StrategyDecision
from crypto_ai_trader.paper import PaperBroker, run_paper_replay
from crypto_ai_trader.risk import (
    RiskReason,
    build_risk_decision_frame,
    causal_atr_values,
    causal_ewma_daily_volatility,
    causal_ewma_volatility,
    drawdown_cooldown_gate,
    dynamic_position_fraction,
    evaluate_strategy_risk,
    funding_crowding_gate,
    regime_risk_gate,
    stop_distance_series,
    volatility_position_scale,
)


class RiskLayerTests(unittest.TestCase):
    def test_paper_replay_summary_contains_comparable_performance_metrics(self) -> None:
        class StubBundle:
            feature_columns = ["close"]
            model_name = "paper_test_model"

            def predict_up_probability(self, values: np.ndarray) -> np.ndarray:
                return np.array([0.8, 0.8, 0.2, 0.2], dtype=float)

        frame = pd.DataFrame(
            {
                "open_time": np.arange(4) * 3_600_000,
                "close": [100.0, 102.0, 101.0, 99.0],
                "high": [101.0, 103.0, 102.0, 100.0],
                "low": [99.0, 101.0, 100.0, 98.0],
                "quote_volume": [1_000_000.0] * 4,
            }
        )
        cfg = BacktestConfig(
            fee_rate=0.001,
            slippage_rate=0.001,
            funding_rate_buffer=0.0,
            regime_gate_enabled=False,
            cost_filter_enabled=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            _, summary_path = run_paper_replay(
                frame,
                StubBundle(),
                cfg,
                Path(temp_dir),
                "ETHUSDT_1h",
            )
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["model_name"], "paper_test_model")
        self.assertEqual(payload["symbol"], "ETHUSDT")
        self.assertEqual(payload["interval"], "1h")
        self.assertTrue(
            {
                "total_return",
                "annualized_return",
                "max_drawdown",
                "sharpe_like",
                "sortino_ratio",
                "calmar_ratio",
                "profit_factor",
                "win_rate",
                "trades",
                "fee_ratio",
                "notional_turnover",
                "average_exposure",
                "performance_by_year",
                "performance_by_month",
                "performance_by_symbol",
            }.issubset(payload["metrics"])
        )
        self.assertGreater(payload["metrics"]["fee_ratio"], 0.0)
        self.assertIn(
            "ETHUSDT",
            payload["metrics"]["performance_by_symbol"],
        )

    def test_backtest_config_rejects_invalid_leverage(self) -> None:
        for leverage in (0, -1, 1.5, float("nan")):
            with self.subTest(leverage=leverage):
                with self.assertRaises(ValueError):
                    BacktestConfig(leverage=leverage)
        with self.assertRaisesRegex(
            ValueError,
            "leverage must not exceed max_allowed_leverage",
        ):
            BacktestConfig(leverage=4, max_allowed_leverage=3)
        with self.assertRaises(ValueError):
            BacktestConfig(
                ewma_volatility_enabled=True,
                ewma_volatility_span=1,
            )
        with self.assertRaises(ValueError):
            BacktestConfig(
                ewma_volatility_enabled=True,
                ewma_daily_volatility_target=0.0,
            )

    def test_dynamic_position_respects_leveraged_notional_cap(self) -> None:
        cfg = BacktestConfig(
            leverage=3,
            max_position_fraction=0.50,
            max_notional_exposure=0.45,
            dynamic_position_sizing=False,
        )
        frame = pd.DataFrame({"close": [100.0, 101.0]})

        fractions = dynamic_position_fraction(
            frame,
            np.array([0.8, 0.2]),
            np.array([1, -1]),
            cfg,
        )

        np.testing.assert_allclose(fractions, np.array([0.15, 0.15]))

    def test_atr_stop_distance_controls_risk_budget(self) -> None:
        cfg = BacktestConfig(
            leverage=2,
            max_position_fraction=1.0,
            max_notional_exposure=2.0,
            risk_per_trade=0.01,
            min_confidence_gap=0.0,
            min_position_scale=1.0,
            use_atr_exits=True,
            stop_loss_atr_multiplier=2.0,
            min_exit_pct=0.001,
            max_exit_pct=0.20,
        )
        frame = pd.DataFrame(
            {
                "atr_14": [0.01, 0.02],
                "quote_volume_z": [0.0, 0.0],
            }
        )

        stops = stop_distance_series(frame, cfg)
        fractions = dynamic_position_fraction(
            frame,
            prob=np.array([1.0, 1.0]),
            position=np.array([1, 1]),
            cfg=cfg,
        )

        np.testing.assert_allclose(stops, np.array([0.02, 0.04]))
        np.testing.assert_allclose(fractions, np.array([0.25, 0.125]))

    def test_atr_fallback_does_not_read_future_rows(self) -> None:
        base = pd.DataFrame({"atr_14": [0.01, np.nan, 0.02]})
        extended = pd.concat(
            [base, pd.DataFrame({"atr_14": [1.0, 2.0]})],
            ignore_index=True,
        )
        np.testing.assert_allclose(
            causal_atr_values(base, fallback=0.005),
            causal_atr_values(extended, fallback=0.005)[: len(base)],
        )

    def test_ewma_volatility_is_causal(self) -> None:
        base = pd.DataFrame(
            {"return_1": [np.nan, 0.01, -0.02, 0.005, -0.004]}
        )
        extended = pd.concat(
            [
                base,
                pd.DataFrame({"return_1": [0.50, -0.40]}),
            ],
            ignore_index=True,
        )

        np.testing.assert_allclose(
            causal_ewma_volatility(
                base,
                span=4,
                fallback=0.01,
            ),
            causal_ewma_volatility(
                extended,
                span=4,
                fallback=0.01,
            )[: len(base)],
        )

    def test_ewma_volatility_reduces_high_volatility_position(self) -> None:
        cfg = BacktestConfig(
            leverage=1,
            max_position_fraction=1.0,
            max_notional_exposure=1.0,
            risk_per_trade=0.02,
            stop_loss=0.02,
            min_confidence_gap=0.0,
            min_position_scale=1.0,
            volatility_target=0.01,
            ewma_volatility_enabled=True,
            ewma_volatility_span=3,
            ewma_daily_volatility_target=0.03,
            min_volatility_scale=0.20,
            max_volatility_scale=1.0,
        )
        frame = pd.DataFrame(
            {
                "return_1": [
                    np.nan,
                    0.001,
                    -0.001,
                    0.001,
                    0.04,
                    -0.04,
                ],
                "quote_volume_z": [0.0] * 6,
            }
        )
        fractions = dynamic_position_fraction(
            frame,
            prob=np.ones(6),
            position=np.ones(6, dtype=int),
            cfg=cfg,
        )

        self.assertLess(fractions[-1], fractions[2])
        daily = causal_ewma_daily_volatility(
            frame,
            span=3,
            fallback_daily=0.03,
        )
        self.assertGreater(daily[-1], 0.03)
        scales = volatility_position_scale(frame, cfg)
        self.assertLess(scales[-1], scales[2])

    def test_drawdown_guard_blocks_following_cooldown_bars(self) -> None:
        cfg = BacktestConfig(
            drawdown_cooldown_enabled=True,
            cooldown_drawdown=0.006,
            cooldown_loss_streak=99,
            cooldown_bars=2,
        )

        gate = drawdown_cooldown_gate(
            np.array([-0.01, 0.02, 0.02, 0.02]),
            cfg,
        )

        np.testing.assert_array_equal(
            gate,
            np.array([True, False, False, True]),
        )

    def test_funding_crowding_blocks_only_the_paying_side(self) -> None:
        cfg = BacktestConfig(
            funding_crowding_guard_enabled=True,
            funding_crowding_max_rate=0.0005,
        )
        frame = pd.DataFrame(
            {
                "funding_rate_8h": [
                    0.001,
                    0.001,
                    -0.001,
                    -0.001,
                ]
            }
        )
        gate, funding = funding_crowding_gate(
            frame,
            np.array([1, -1, -1, 1]),
            cfg,
        )

        np.testing.assert_array_equal(
            gate,
            np.array([False, True, False, True]),
        )
        np.testing.assert_allclose(
            funding,
            frame["funding_rate_8h"].to_numpy(),
        )

        detail = build_risk_decision_frame(
            frame,
            proposed_position=np.array([1, -1, -1, 1]),
            target_fraction=np.full(4, 0.1),
            cooldown_gate=np.ones(4, dtype=bool),
            funding_gate=gate,
            cfg=cfg,
        )
        self.assertEqual(
            detail["risk_reason"].tolist(),
            [
                RiskReason.FUNDING_CROWDING_BLOCKED.value,
                RiskReason.ALLOWED.value,
                RiskReason.FUNDING_CROWDING_BLOCKED.value,
                RiskReason.ALLOWED.value,
            ],
        )

    def test_regime_risk_off_has_highest_risk_priority(self) -> None:
        cfg = BacktestConfig(regime_risk_guard_enabled=True)
        frame = pd.DataFrame(
            {
                "regime_risk_off": [True, False, True],
                "market_regime": ["crash", "trend_up", "liquidity_crisis"],
                "atr_14": [0.01, 0.01, 0.01],
                "quote_volume_z": [0.0, 0.0, -3.0],
            }
        )
        proposed = np.array([1, 1, -1])
        gate = regime_risk_gate(frame, proposed, cfg)
        detail = build_risk_decision_frame(
            frame,
            proposed_position=proposed,
            target_fraction=np.full(3, 0.1),
            cooldown_gate=np.array([False, True, True]),
            funding_gate=np.array([False, True, False]),
            regime_gate=gate,
            cfg=cfg,
        )

        np.testing.assert_array_equal(gate, np.array([False, True, False]))
        self.assertEqual(
            detail["risk_reason"].tolist(),
            [
                RiskReason.REGIME_RISK_OFF_BLOCKED.value,
                RiskReason.ALLOWED.value,
                RiskReason.REGIME_RISK_OFF_BLOCKED.value,
            ],
        )
        self.assertEqual(
            detail["risk_level"].tolist(),
            [
                RiskLevel.EXTREME.value,
                RiskLevel.LOW.value,
                RiskLevel.EXTREME.value,
            ],
        )

    def test_regime_risk_guard_is_enabled_by_default(self) -> None:
        self.assertTrue(BacktestConfig().regime_risk_guard_enabled)

    def test_risk_frame_exposes_block_and_reduction_reasons(self) -> None:
        cfg = BacktestConfig(volatility_target=0.01)
        frame = pd.DataFrame(
            {
                "atr_14": [0.008, 0.015, 0.008],
                "quote_volume_z": [0.0, 0.0, -1.2],
            }
        )

        detail = build_risk_decision_frame(
            frame,
            proposed_position=np.array([1, 1, -1]),
            target_fraction=np.array([0.2, 0.1, 0.08]),
            cooldown_gate=np.array([False, True, True]),
            cfg=cfg,
        )

        self.assertEqual(
            detail["risk_reason"].tolist(),
            [
                RiskReason.DRAWDOWN_BLOCKED.value,
                RiskReason.VOLATILITY_REDUCED.value,
                RiskReason.LIQUIDITY_REDUCED.value,
            ],
        )
        self.assertEqual(detail["risk_allow_trade"].tolist(), [False, True, True])

    def test_risk_frame_does_not_claim_reduction_when_sizing_is_static(self) -> None:
        cfg = BacktestConfig(
            dynamic_position_sizing=False,
            volatility_target=0.01,
        )
        frame = pd.DataFrame(
            {
                "atr_14": [0.05],
                "quote_volume_z": [-2.0],
            }
        )

        detail = build_risk_decision_frame(
            frame,
            proposed_position=np.array([1]),
            target_fraction=np.array([0.1]),
            cooldown_gate=np.array([True]),
            cfg=cfg,
        )

        self.assertEqual(detail.iloc[0]["risk_reason"], RiskReason.ALLOWED.value)

    def test_paper_broker_honors_risk_veto_and_target_exposure(self) -> None:
        cfg = BacktestConfig(initial_balance=10_000.0, leverage=2, fee_rate=0.0)
        broker = PaperBroker(cfg)
        row = pd.Series({"close": 100.0, "open_time": 1})
        strategy = StrategyDecision(
            target_direction=1,
            target_exposure=0.10,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="alpha_threshold_long",
        )
        blocked = RiskDecision(
            allow_trade=False,
            risk_level=RiskLevel.EXTREME,
            max_position_size=0.0,
            reason=RiskReason.DRAWDOWN_BLOCKED.value,
        )

        broker.step(row, strategy, 0.7, risk_decision=blocked)
        self.assertEqual(broker.state.position, 0)

        allowed = evaluate_strategy_risk(strategy, cfg)
        broker.step(row, strategy, 0.7, risk_decision=allowed)
        self.assertEqual(broker.state.position, 1)
        self.assertAlmostEqual(broker.state.units, 20.0)
        self.assertEqual(broker.ledger[-1]["target_exposure"], 0.10)

    def test_paper_broker_applies_exchange_minimum_order_rules(self) -> None:
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            leverage=1,
            fee_rate=0.0,
            exchange_min_notional_usdt=50.0,
            exchange_quantity_step=0.01,
        )
        broker = PaperBroker(cfg)
        strategy = StrategyDecision(
            target_direction=1,
            target_exposure=0.001,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="test_small_order",
        )

        broker.step(
            pd.Series({"close": 100.0, "open_time": 1}),
            strategy,
            0.8,
            risk_decision=evaluate_strategy_risk(strategy, cfg),
        )

        self.assertEqual(broker.state.position, 0)
        self.assertEqual(broker.state.order_rejections, 1)
        self.assertEqual(broker.ledger[-1]["action"], "ORDER_REJECTED")
        self.assertEqual(
            broker.ledger[-1]["reason"],
            "below_exchange_min_notional",
        )

    def test_paper_broker_uses_liquidity_partial_fills_and_funding(self) -> None:
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            leverage=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
            liquidity_execution_enabled=True,
            max_bar_participation_rate=0.01,
            slippage_impact_coefficient=0.0,
        )
        broker = PaperBroker(cfg)
        long_decision = StrategyDecision(
            target_direction=1,
            target_exposure=0.10,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="test_liquidity_long",
        )
        flat_decision = StrategyDecision(
            target_direction=0,
            target_exposure=0.0,
            stop_loss=None,
            take_profit=None,
            holding_period=None,
            reason_code="test_flat",
        )
        long_risk = evaluate_strategy_risk(long_decision, cfg)
        flat_risk = evaluate_strategy_risk(flat_decision, cfg)

        broker.step(
            pd.Series(
                {
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "quote_volume": 10_000.0,
                    "open_time": 1,
                }
            ),
            long_decision,
            0.8,
            risk_decision=long_risk,
        )
        self.assertEqual(broker.state.position, 1)
        self.assertAlmostEqual(broker.state.units, 1.0)
        self.assertEqual(broker.state.partial_fills, 1)
        self.assertEqual(broker.state.liquidity_limited_orders, 1)
        self.assertAlmostEqual(broker.ledger[-1]["fill_ratio"], 0.10)

        broker.step(
            pd.Series(
                {
                    "close": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "quote_volume": 10_000.0,
                    "funding_payment_rate": 0.001,
                    "open_time": 2,
                }
            ),
            flat_decision,
            0.5,
            risk_decision=flat_risk,
        )
        self.assertEqual(broker.state.position, 0)
        self.assertAlmostEqual(broker.state.funding_net_cost, 0.1)
        self.assertEqual(
            [entry["action"] for entry in broker.ledger],
            ["BUY_LONG", "FUNDING", "CLOSE"],
        )

    def test_paper_broker_blocks_orders_when_exchange_is_unavailable(self) -> None:
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
        )
        broker = PaperBroker(cfg)
        strategy = StrategyDecision(
            target_direction=-1,
            target_exposure=0.10,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="test_short",
        )
        risk = evaluate_strategy_risk(strategy, cfg)

        broker.step(
            pd.Series(
                {
                    "close": 100.0,
                    "open_time": 1,
                    "execution_available": False,
                    "exchange_downtime_reason": "kline_gap_recovery",
                }
            ),
            strategy,
            0.2,
            risk_decision=risk,
        )

        self.assertEqual(broker.state.position, 0)
        self.assertEqual(broker.state.exchange_downtime_blocks, 1)
        self.assertEqual(broker.ledger[-1]["action"], "EXCHANGE_UNAVAILABLE")
        self.assertEqual(broker.ledger[-1]["reason"], "kline_gap_recovery")

    def test_paper_broker_applies_configured_bar_latency(self) -> None:
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            leverage=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
            execution_latency_bars=1,
        )
        broker = PaperBroker(cfg)
        long_decision = StrategyDecision(
            target_direction=1,
            target_exposure=0.10,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="test_latency_long",
        )
        flat_decision = StrategyDecision(
            target_direction=0,
            target_exposure=0.0,
            stop_loss=None,
            take_profit=None,
            holding_period=None,
            reason_code="test_latency_flat",
        )

        broker.step(
            pd.Series({"close": 100.0, "open_time": 1}),
            long_decision,
            0.8,
            risk_decision=evaluate_strategy_risk(long_decision, cfg),
        )
        self.assertEqual(broker.state.position, 0)
        self.assertEqual(broker.state.latency_deferred_decisions, 1)

        broker.step(
            pd.Series({"close": 101.0, "open_time": 2}),
            flat_decision,
            0.5,
            risk_decision=evaluate_strategy_risk(flat_decision, cfg),
        )
        self.assertEqual(broker.state.position, 1)
        self.assertEqual(broker.ledger[-1]["reason_code"], "test_latency_long")

        broker.step(
            pd.Series({"close": 102.0, "open_time": 3}),
            flat_decision,
            0.5,
            risk_decision=evaluate_strategy_risk(flat_decision, cfg),
        )
        self.assertEqual(broker.state.position, 0)
        self.assertEqual(broker.ledger[-1]["reason"], "test_latency_flat")

    def test_paper_broker_continues_exchange_max_quantity_child_orders(self) -> None:
        cfg = BacktestConfig(
            initial_balance=1_000.0,
            leverage=1,
            fee_rate=0.0,
            slippage_rate=0.0,
            funding_rate_buffer=0.0,
            exchange_max_quantity=0.05,
            exchange_quantity_step=0.01,
        )
        broker = PaperBroker(cfg)
        strategy = StrategyDecision(
            target_direction=1,
            target_exposure=0.02,
            stop_loss=0.02,
            take_profit=0.03,
            holding_period=None,
            reason_code="test_child_orders",
        )
        risk = evaluate_strategy_risk(strategy, cfg)

        for open_time in range(4):
            broker.step(
                pd.Series(
                    {
                        "close": 100.0,
                        "open_time": open_time,
                        "exchange_available": True,
                    }
                ),
                strategy,
                0.8,
                risk_decision=risk,
            )

        self.assertAlmostEqual(broker.state.units, 0.20)
        self.assertAlmostEqual(broker.state.pending_open_notional_usdt, 0.0)
        self.assertEqual(broker.state.maximum_quantity_limited_orders, 3)
        self.assertEqual(
            [entry["action"] for entry in broker.ledger],
            ["BUY_LONG", "BUY_LONG", "BUY_LONG", "BUY_LONG"],
        )

    def test_backtest_detail_contains_risk_contract_columns(self) -> None:
        class StubBundle:
            feature_columns = ["close"]
            auxiliary_metadata: dict[str, object] = {}

            def predict_direction_probabilities(
                self, values: np.ndarray
            ) -> dict[str, np.ndarray]:
                rows = len(values)
                up = np.full(rows, 0.80)
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
                "close": [100.0, 101.0, 102.0, 101.5, 103.0],
                "atr_14": [0.01] * 5,
                "quote_volume_z": [0.0] * 5,
            }
        )
        cfg = BacktestConfig(
            horizon_confirmation_enabled=False,
            regime_gate_enabled=False,
            cost_filter_enabled=False,
        )

        result, detail = run_backtest(frame, StubBundle(), cfg)

        self.assertTrue(
            {
                "risk_allow_trade",
                "risk_level",
                "risk_max_position_size",
                "risk_reason",
                "risk_ewma_volatility",
                "risk_ewma_daily_volatility",
                "risk_position_volatility_measure",
                "risk_volatility_scale",
                "funding_crowding_gate_pass",
                "funding_crowding_rate",
                "decision_reason_code",
                "execution_reason_code",
                "execution_available",
                "exchange_downtime_blocked",
                "market_regime",
                "regime_confidence",
                "regime_risk_off",
                "volatility_state",
                "liquidity_state",
                "regime_reason",
                "liquidation_triggered",
                "liquidation_gap_triggered",
                "liquidation_price_distance",
                "liquidation_price_return",
                "liquidation_fee_cost",
                "liquidation_reason",
                "liquidation_forced_notional",
                "liquidation_forced_turnover",
                "end_notional_position",
                "commission_cost",
                "slippage_cost",
                "funding_cost",
                "funding_rate_used",
                "funding_rate_source",
                "funding_settlement_mode",
                "trade_cost",
                "total_cost",
                "target_notional_position",
                "delayed_target_notional",
                "executed_notional_position",
                "executed_position",
                "performance_side",
                "requested_notional_change",
                "position_before_execution",
                "executed_notional_change",
                "execution_fill_ratio",
                "minimum_order_rejected",
                "exchange_filter_rejected",
                "exchange_order_quantity",
                "exchange_order_notional_usdt",
                "quantity_rounding_loss_usdt",
                "maximum_quantity_limited",
                "exchange_rejection_reason",
                "liquidity_fill_ratio",
                "liquidity_capacity_usdt",
                "liquidity_limited",
                "effective_slippage_rate",
                "market_participation_rate",
                "liquidity_stress",
                "funding_notional_position",
                "maker_notional_turnover",
                "taker_notional_turnover",
            }.issubset(detail.columns)
        )
        self.assertFalse(result.ewma_volatility_enabled)
        np.testing.assert_allclose(
            detail["trade_cost"],
            detail["commission_cost"] + detail["slippage_cost"],
        )
        np.testing.assert_allclose(
            detail["total_cost"],
            detail["trade_cost"]
            + detail["funding_cost"]
            + detail["liquidation_fee_cost"],
        )
        self.assertTrue(result.regime_summary)

    def test_rejected_signal_is_not_counted_as_executed_trade(self) -> None:
        class StubBundle:
            feature_columns = ["close"]
            auxiliary_metadata: dict[str, object] = {}

            def predict_direction_probabilities(
                self, values: np.ndarray
            ) -> dict[str, np.ndarray]:
                rows = len(values)
                up = np.full(rows, 0.80)
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
                "close": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "quote_volume": [1_000_000.0] * 4,
            }
        )
        cfg = BacktestConfig(
            initial_balance=1_000.0,
            max_position_fraction=0.01,
            exchange_min_notional_usdt=100.0,
            horizon_confirmation_enabled=False,
            regime_gate_enabled=False,
            cost_filter_enabled=False,
        )

        result, detail = run_backtest(frame, StubBundle(), cfg)

        self.assertTrue((detail["position"] != 0).any())
        self.assertFalse((detail["notional_turnover"] > 0.0).any())
        self.assertEqual(result.trades, 0)
        self.assertEqual(result.execution_events, 0)
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.profit_factor, 0.0)

    def test_paper_broker_records_liquidation_before_reopening(self) -> None:
        cfg = BacktestConfig(
            initial_balance=10_000.0,
            leverage=3,
            max_position_fraction=0.20,
            max_notional_exposure=0.60,
            stop_loss=0.50,
            fee_rate=0.0,
            liquidation_fee_rate=0.01,
        )
        broker = PaperBroker(cfg)
        decision = StrategyDecision(
            target_direction=1,
            target_exposure=0.20,
            stop_loss=0.50,
            take_profit=0.60,
            holding_period=None,
            reason_code="test_long",
        )
        risk = evaluate_strategy_risk(decision, cfg)

        broker.step(
            pd.Series(
                {"close": 100.0, "open": 100.0, "high": 100.0, "low": 100.0}
            ),
            decision,
            0.9,
            risk_decision=risk,
        )
        broker.step(
            pd.Series(
                {"close": 90.0, "open": 50.0, "high": 95.0, "low": 45.0}
            ),
            decision,
            0.9,
            risk_decision=risk,
        )

        self.assertEqual(broker.state.liquidations, 1)
        self.assertGreater(broker.state.liquidation_fees, 0.0)
        self.assertEqual(
            [entry["action"] for entry in broker.ledger],
            ["BUY_LONG", "LIQUIDATION", "BUY_LONG"],
        )


if __name__ == "__main__":
    unittest.main()
