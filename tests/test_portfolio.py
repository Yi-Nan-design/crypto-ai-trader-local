from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from crypto_ai_trader.contracts import PortfolioAssetInput, RiskLevel
from crypto_ai_trader.portfolio import (
    apply_expected_shortfall_limit,
    apply_portfolio_volatility_target,
    build_portfolio_snapshot,
    construct_portfolio,
    correlation_clusters,
    historical_expected_shortfall,
    portfolio_inputs_from_reports,
)


def config() -> SimpleNamespace:
    return SimpleNamespace(
        portfolio_target_gross_exposure=0.60,
        portfolio_max_total_leverage=0.70,
        portfolio_max_single_weight=0.30,
        portfolio_max_sector_exposure=0.35,
        portfolio_max_cluster_exposure=0.40,
        portfolio_min_liquidity_score=0.20,
        portfolio_max_drawdown=0.10,
        portfolio_volatility_floor=0.001,
        portfolio_volatility_target_enabled=True,
        portfolio_target_daily_volatility=0.02,
        portfolio_min_volatility_observations=100,
        portfolio_correlation_threshold=0.75,
        portfolio_correlation_lookback=500,
        portfolio_cvar_confidence=0.95,
        portfolio_max_cvar_loss=0.01,
        portfolio_min_cvar_observations=100,
        max_daily_loss=0.03,
        default_leverage=2,
        portfolio_require_complete_inputs=True,
        portfolio_symbol_sectors={
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "layer1",
            "SOLUSDT": "layer1",
            "BNBUSDT": "exchange",
        },
    )


def asset(
    symbol: str,
    *,
    direction: int,
    volatility: float,
    liquidity: float = 1.0,
    cluster: str = "cluster_a",
    sector: str = "layer1",
) -> PortfolioAssetInput:
    return PortfolioAssetInput(
        symbol=symbol,
        direction=direction,
        signal_strength=0.8,
        confidence=0.75,
        volatility=volatility,
        liquidity_score=liquidity,
        max_weight=0.50,
        correlation_cluster=cluster,
        sector=sector,
    )


class PortfolioTests(unittest.TestCase):
    def test_lower_volatility_receives_more_weight_within_caps(self) -> None:
        decision = construct_portfolio(
            [
                asset("BTCUSDT", direction=1, volatility=0.01, cluster="large"),
                asset("ETHUSDT", direction=1, volatility=0.02, cluster="large"),
                asset("SOLUSDT", direction=-1, volatility=0.03, cluster="alt"),
            ],
            config(),
        )

        self.assertTrue(decision.allow_portfolio)
        self.assertGreater(
            abs(decision.weights["BTCUSDT"]),
            abs(decision.weights["ETHUSDT"]),
        )
        self.assertLessEqual(decision.gross_exposure, 0.60 + 1e-12)
        self.assertLessEqual(decision.cluster_exposure["large"], 0.40 + 1e-12)
        self.assertLessEqual(abs(decision.weights["BTCUSDT"]), 0.30 + 1e-12)

    def test_sector_exposure_caps_related_assets(self) -> None:
        decision = construct_portfolio(
            [
                asset(
                    "ETHUSDT",
                    direction=1,
                    volatility=0.01,
                    cluster="eth",
                    sector="layer1",
                ),
                asset(
                    "SOLUSDT",
                    direction=1,
                    volatility=0.01,
                    cluster="sol",
                    sector="layer1",
                ),
                asset(
                    "BNBUSDT",
                    direction=1,
                    volatility=0.01,
                    cluster="bnb",
                    sector="exchange",
                ),
            ],
            config(),
        )

        self.assertLessEqual(decision.sector_exposure["layer1"], 0.35 + 1e-12)
        self.assertEqual(
            decision.asset_reasons["ETHUSDT"],
            "portfolio_reduced_sector_cap",
        )

    def test_liquidity_and_drawdown_can_veto_portfolio(self) -> None:
        low_liquidity = construct_portfolio(
            [
                asset(
                    "ETHUSDT",
                    direction=1,
                    volatility=0.02,
                    liquidity=0.05,
                )
            ],
            config(),
        )
        self.assertFalse(low_liquidity.allow_portfolio)
        self.assertEqual(
            low_liquidity.asset_reasons["ETHUSDT"],
            "portfolio_blocked_low_liquidity",
        )

        drawdown = construct_portfolio(
            [asset("ETHUSDT", direction=1, volatility=0.02)],
            config(),
            current_drawdown=-0.12,
        )
        self.assertFalse(drawdown.allow_portfolio)
        self.assertEqual(drawdown.risk_level, RiskLevel.EXTREME)
        self.assertEqual(drawdown.reason, "portfolio_blocked_drawdown")

        daily_loss = construct_portfolio(
            [asset("ETHUSDT", direction=1, volatility=0.02)],
            config(),
            current_daily_return=-0.04,
        )
        self.assertFalse(daily_loss.allow_portfolio)
        self.assertEqual(daily_loss.risk_level, RiskLevel.EXTREME)
        self.assertEqual(daily_loss.reason, "portfolio_blocked_daily_loss")

        unavailable = construct_portfolio(
            [asset("ETHUSDT", direction=1, volatility=0.02)],
            config(),
            input_available=False,
            input_unavailable_reason=(
                "portfolio_blocked_unaligned_market_data"
            ),
        )
        self.assertFalse(unavailable.allow_portfolio)
        self.assertEqual(unavailable.risk_level, RiskLevel.EXTREME)
        self.assertEqual(
            unavailable.reason,
            "portfolio_blocked_unaligned_market_data",
        )

    def test_correlation_clusters_use_connected_components(self) -> None:
        base = np.linspace(-0.02, 0.02, 120)
        returns = pd.DataFrame(
            {
                "BTCUSDT": base,
                "ETHUSDT": base * 0.95,
                "SOLUSDT": np.sin(np.linspace(0, 12, 120)) * 0.01,
            }
        )
        clusters = correlation_clusters(
            returns,
            threshold=0.90,
            min_periods=48,
        )

        self.assertEqual(clusters["BTCUSDT"], clusters["ETHUSDT"])
        self.assertNotEqual(clusters["BTCUSDT"], clusters["SOLUSDT"])

    def test_report_conversion_respects_threshold_side_and_zero_risk(self) -> None:
        reports = [
            {
                "symbol": "ETHUSDT",
                "latest_open_time": 1_000,
                "latest_up_probability": 0.72,
                "latest_direction_probabilities": {
                    "long": 0.72,
                    "short": 0.28,
                    "trade": 0.90,
                },
                "latest_alpha_prediction": {
                    "confidence": 0.44,
                    "volatility_forecast": 0.02,
                },
                "latest_liquidity_score": 0.8,
                "latest_execution_decision": {
                    "allow_execution": True,
                },
                "latest_risk_decision": {
                    "allow_trade": True,
                    "max_position_size": 0.0,
                },
                "optimized_backtest_config": {
                    "long_threshold": 0.65,
                    "short_threshold": 0.35,
                    "trade_side_policy": "both",
                    "trade_signal_threshold": 0.57,
                },
            },
            {
                "symbol": "BNBUSDT",
                "latest_open_time": 1_000,
                "latest_up_probability": 0.20,
                "latest_direction_probabilities": {
                    "long": 0.20,
                    "short": 0.80,
                    "trade": 0.90,
                },
                "latest_alpha_prediction": {
                    "confidence": 0.60,
                    "volatility_forecast": 0.02,
                },
                "latest_liquidity_score": 0.8,
                "latest_execution_decision": {
                    "allow_execution": True,
                },
                "latest_risk_decision": {
                    "allow_trade": True,
                    "max_position_size": 0.10,
                },
                "optimized_backtest_config": {
                    "long_threshold": 0.65,
                    "short_threshold": 0.35,
                    "trade_side_policy": "long_only",
                    "trade_signal_threshold": 0.57,
                },
            },
        ]
        inputs = portfolio_inputs_from_reports(
            reports,
            clusters={},
            default_max_weight=0.25,
            leverage=2,
            require_complete_inputs=True,
            expected_open_time=1_000,
        )

        self.assertEqual(inputs[0].direction, 1)
        self.assertEqual(inputs[0].max_weight, 0.0)
        self.assertEqual(inputs[1].direction, 0)
        self.assertAlmostEqual(inputs[1].max_weight, 0.20)

    def test_incomplete_or_stale_report_fails_closed(self) -> None:
        reports = [
            {
                "symbol": "ETHUSDT",
                "latest_open_time": 900,
                "latest_up_probability": 0.72,
                "latest_direction_probabilities": {
                    "long": 0.72,
                    "short": 0.28,
                    "trade": 0.90,
                },
                "latest_alpha_prediction": {
                    "confidence": 0.44,
                    "volatility_forecast": 0.02,
                },
                "latest_liquidity_score": 0.8,
                "latest_execution_decision": {"allow_execution": True},
                "latest_risk_decision": {
                    "allow_trade": True,
                    "max_position_size": 0.10,
                },
            },
            {
                "symbol": "BNBUSDT",
                "latest_open_time": 1_000,
                "latest_up_probability": 0.72,
                "latest_direction_probabilities": {
                    "long": 0.72,
                    "short": 0.28,
                    "trade": 0.90,
                },
                "latest_alpha_prediction": {
                    "confidence": 0.44,
                    "volatility_forecast": 0.02,
                },
                "latest_liquidity_score": None,
                "latest_execution_decision": {"allow_execution": True},
                "latest_risk_decision": {
                    "allow_trade": True,
                    "max_position_size": 0.10,
                },
            },
        ]
        inputs = portfolio_inputs_from_reports(
            reports,
            clusters={},
            default_max_weight=0.25,
            require_complete_inputs=True,
            expected_open_time=1_000,
        )
        decision = construct_portfolio(inputs, config())

        self.assertFalse(decision.allow_portfolio)
        self.assertEqual(
            decision.asset_reasons["ETHUSDT"],
            "portfolio_blocked_stale_or_unaligned_open_time",
        )
        self.assertEqual(
            decision.asset_reasons["BNBUSDT"],
            "portfolio_blocked_invalid_liquidity_score",
        )

    def test_snapshot_is_explicitly_planning_only(self) -> None:
        reports = [
            {
                "symbol": "ETHUSDT",
                "latest_open_time": 1_000,
                "latest_up_probability": 0.72,
                "latest_direction_probabilities": {
                    "long": 0.72,
                    "short": 0.28,
                    "trade": 0.90,
                },
                "latest_alpha_prediction": {
                    "confidence": 0.44,
                    "volatility_forecast": 0.02,
                },
                "latest_liquidity_score": 0.8,
                "latest_execution_decision": {
                    "allow_execution": True,
                },
                "latest_risk_decision": {
                    "allow_trade": True,
                    "max_position_size": 0.10,
                },
                "optimized_backtest_config": {
                    "long_threshold": 0.65,
                    "short_threshold": 0.35,
                    "trade_side_policy": "both",
                    "trade_signal_threshold": 0.57,
                },
            }
        ]
        returns = pd.DataFrame({"ETHUSDT": np.linspace(-0.01, 0.01, 80)})
        snapshot = build_portfolio_snapshot(
            reports,
            returns,
            config(),
            interval="5m",
            expected_open_time=1_000,
        )

        self.assertEqual(snapshot["status"], "planning_only")
        self.assertFalse(snapshot["safety"]["live_trading_enabled"])
        self.assertEqual(
            snapshot["current_drawdown_source"],
            "no_live_portfolio_equity_planning_default",
        )

    def test_expected_shortfall_uses_weighted_closed_bar_returns(self) -> None:
        returns = pd.DataFrame(
            {
                "ETHUSDT": [-0.01] * 10 + [0.001] * 190,
                "BNBUSDT": [0.02] * 10 + [-0.001] * 190,
            }
        )
        risk = historical_expected_shortfall(
            returns,
            {"ETHUSDT": 0.20, "BNBUSDT": -0.10},
            confidence=0.95,
            min_observations=100,
        )

        self.assertTrue(risk["available"])
        self.assertEqual(risk["observations"], 200)
        self.assertGreater(risk["expected_shortfall_loss"], 0.0)

    def test_portfolio_volatility_target_only_reduces_exposure(self) -> None:
        cfg = config()
        decision = construct_portfolio(
            [
                asset(
                    "ETHUSDT",
                    direction=1,
                    volatility=0.01,
                    cluster="large",
                )
            ],
            cfg,
        )
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {"ETHUSDT": rng.normal(0.0, 0.03, 200)},
            index=np.arange(200) * 3_600_000,
        )

        adjusted, risk = apply_portfolio_volatility_target(
            decision,
            returns,
            cfg,
        )

        self.assertTrue(risk["target_applied"])
        self.assertLess(adjusted.gross_exposure, decision.gross_exposure)
        self.assertLessEqual(
            risk["post_target_daily_volatility"],
            cfg.portfolio_target_daily_volatility + 1e-12,
        )

    def test_expected_shortfall_limit_scales_portfolio(self) -> None:
        cfg = config()
        cfg.portfolio_max_cvar_loss = 0.002
        decision = construct_portfolio(
            [
                asset(
                    "ETHUSDT",
                    direction=1,
                    volatility=0.01,
                    cluster="large",
                ),
                asset(
                    "BNBUSDT",
                    direction=1,
                    volatility=0.02,
                    cluster="large",
                ),
            ],
            cfg,
        )
        returns = pd.DataFrame(
            {
                "ETHUSDT": [-0.03] * 10 + [0.001] * 190,
                "BNBUSDT": [-0.02] * 10 + [0.001] * 190,
            }
        )
        adjusted, risk = apply_expected_shortfall_limit(
            decision,
            returns,
            cfg,
        )

        self.assertTrue(risk["limit_applied"])
        self.assertLess(adjusted.gross_exposure, decision.gross_exposure)
        self.assertEqual(adjusted.reason, "portfolio_constrained_cvar")
        self.assertEqual(adjusted.risk_level, RiskLevel.HIGH)


if __name__ == "__main__":
    unittest.main()
