from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .cost_model import estimate_execution_costs, infer_bar_hours
from .features import feature_only_matrix
from .liquidation import assess_liquidations, liquidation_price_distance
from .liquidity_execution import (
    causal_liquidity_profile,
    estimate_dynamic_slippage,
)
from .models import ModelBundle
from .performance_report import evaluate_backtest_performance
from .regime import (
    causal_volatility_thresholds,
    detect_regime_frame,
    summarize_regime_performance,
)
from .risk import (
    apply_position_rebalance_rules,
    build_risk_decision_frame,
    drawdown_cooldown_gate,
    dynamic_position_fraction,
    causal_atr_values,
    funding_crowding_gate,
    regime_risk_gate,
    stop_distance_series,
    volatility_position_scale,
)
from .strategy import decision_reason_codes


@dataclass
class BacktestConfig:
    initial_balance: float = 10_000.0
    leverage: int = 1
    max_allowed_leverage: int = 3
    fee_rate: float = 0.00045
    maker_fee_rate: float = 0.0002
    maker_fill_fraction: float = 0.0
    slippage_rate: float = 0.0002
    partial_fill_ratio: float = 1.0
    execution_latency_bars: int = 0
    min_order_notional_fraction: float = 0.0
    exchange_min_notional_usdt: float = 0.0
    exchange_min_quantity: float = 0.0
    exchange_max_quantity: float = 0.0
    exchange_quantity_step: float = 0.0
    exchange_price_tick_size: float = 0.0
    exchange_downtime_guard_enabled: bool = True
    exchange_gap_recovery_bars: int = 1
    liquidity_execution_enabled: bool = False
    max_bar_participation_rate: float = 0.01
    liquidity_lookback_bars: int = 48
    slippage_impact_coefficient: float = 1.0
    max_dynamic_slippage_rate: float = 0.02
    long_threshold: float = 0.57
    short_threshold: float = 0.43
    max_position_fraction: float = 0.35
    stop_loss: float = 0.02
    take_profit: float = 0.035
    use_atr_exits: bool = False
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 3.0
    min_exit_pct: float = 0.004
    max_exit_pct: float = 0.06
    risk_per_trade: float = 0.005
    min_confidence_gap: float = 0.07
    dynamic_position_sizing: bool = True
    min_position_scale: float = 0.35
    volatility_target: float = 0.01
    ewma_volatility_enabled: bool = False
    ewma_volatility_span: int = 48
    ewma_daily_volatility_target: float = 0.03
    min_volatility_scale: float = 0.35
    max_volatility_scale: float = 1.25
    cost_filter_enabled: bool = True
    funding_rate_buffer: float = 0.0001
    min_atr_cost_multiplier: float = 2.0
    regime_gate_enabled: bool = True
    regime_detection_method: str = "rule_based"
    regime_statistical_clusters: int = 4
    regime_statistical_min_history: int = 240
    regime_statistical_lookback: int = 720
    regime_statistical_refit_interval: int = 24
    regime_statistical_random_seed: int = 42
    trend_gate_min_efficiency: float = 0.18
    range_gate_max_efficiency: float = 0.16
    range_long_max_position: float = 0.35
    range_short_min_position: float = 0.65
    trend_breakout_min_score: float = 0.0002
    range_reversion_min_score: float = 0.08
    direction_signal_margin: float = 0.08
    trade_signal_threshold: float = 0.57
    horizon_confirmation_enabled: bool = True
    horizon_assist_enabled: bool = True
    horizon_assist_long_min_main: float = 0.52
    horizon_assist_short_max_main: float = 0.48
    horizon_long_min_probability: float = 0.50
    horizon_short_max_probability: float = 0.50
    horizon_assist_long_probability: float = 0.58
    horizon_assist_short_probability: float = 0.42
    trade_side_policy: str = "both"
    platform_event_gate_enabled: bool = False
    platform_event_min_score: float = 0.65
    drawdown_cooldown_enabled: bool = False
    cooldown_drawdown: float = 0.006
    cooldown_loss_streak: int = 3
    cooldown_bars: int = 12
    strategy_archetype_policy: str = "all"
    volatility_regime_policy: str = "all"
    volatility_regime_lookback: int = 240
    volatility_regime_low_quantile: float = 0.33
    volatility_regime_high_quantile: float = 0.67
    regime_crash_return_3: float = -0.02
    regime_crash_return_24: float = -0.04
    regime_liquidity_z_threshold: float = -2.0
    regime_liquidity_quality_threshold: float = 0.15
    event_position_boost_enabled: bool = False
    event_position_min_score: float = 0.55
    event_position_boost_strength: float = 0.80
    position_rebalance_band: float = 0.0
    max_position_fraction_step: float = 1.0
    market_structure_gate_enabled: bool = False
    min_market_structure_score: float = 0.55
    crowding_filter_enabled: bool = False
    max_crowding_risk: float = 0.82
    funding_crowding_guard_enabled: bool = False
    funding_crowding_max_rate: float = 0.0005
    regime_risk_guard_enabled: bool = True
    max_notional_exposure: float = 0.45
    liquidation_guard_enabled: bool = True
    maintenance_margin_rate: float = 0.005
    liquidation_buffer: float = 0.01
    liquidation_fee_rate: float = 0.005

    def __post_init__(self) -> None:
        leverage = float(self.leverage)
        if not np.isfinite(leverage) or leverage < 1 or not leverage.is_integer():
            raise ValueError("leverage must be a positive whole number")
        self.leverage = int(leverage)
        max_allowed = float(self.max_allowed_leverage)
        if (
            not np.isfinite(max_allowed)
            or max_allowed < 1
            or not max_allowed.is_integer()
        ):
            raise ValueError(
                "max_allowed_leverage must be a positive whole number"
            )
        self.max_allowed_leverage = int(max_allowed)
        if self.leverage > self.max_allowed_leverage:
            raise ValueError(
                "leverage must not exceed max_allowed_leverage"
            )
        if not 0.0 <= float(self.max_position_fraction) <= 1.0:
            raise ValueError("max_position_fraction must be between 0 and 1")
        if float(self.max_notional_exposure) < 0:
            raise ValueError("max_notional_exposure must be non-negative")
        if float(self.risk_per_trade) < 0:
            raise ValueError("risk_per_trade must be non-negative")
        for name in (
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
            "funding_crowding_max_rate",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(float(self.initial_balance)) or float(self.initial_balance) <= 0:
            raise ValueError("initial_balance must be finite and positive")
        if float(self.maintenance_margin_rate) >= 1.0:
            raise ValueError("maintenance_margin_rate must be below 1")
        if bool(self.liquidation_guard_enabled) and (
            float(self.maintenance_margin_rate) + float(self.liquidation_buffer)
            >= 1.0 / float(self.leverage)
        ):
            raise ValueError(
                "maintenance margin plus liquidation buffer must stay below initial margin"
            )
        for name in ("maker_fill_fraction", "partial_fill_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if float(self.max_bar_participation_rate) > 1.0:
            raise ValueError("max_bar_participation_rate must not exceed 1")
        if bool(self.liquidity_execution_enabled) and float(
            self.max_bar_participation_rate
        ) <= 0.0:
            raise ValueError(
                "max_bar_participation_rate must be positive when liquidity execution is enabled"
            )
        if float(self.max_dynamic_slippage_rate) < float(self.slippage_rate):
            raise ValueError(
                "max_dynamic_slippage_rate must be at least slippage_rate"
            )
        lookback = float(self.liquidity_lookback_bars)
        if not np.isfinite(lookback) or lookback < 1 or not lookback.is_integer():
            raise ValueError("liquidity_lookback_bars must be a positive whole number")
        self.liquidity_lookback_bars = int(lookback)
        latency = float(self.execution_latency_bars)
        if not np.isfinite(latency) or latency < 0 or not latency.is_integer():
            raise ValueError("execution_latency_bars must be a non-negative whole number")
        self.execution_latency_bars = int(latency)
        recovery_bars = float(self.exchange_gap_recovery_bars)
        if (
            not np.isfinite(recovery_bars)
            or recovery_bars < 0
            or not recovery_bars.is_integer()
        ):
            raise ValueError(
                "exchange_gap_recovery_bars must be a non-negative whole number"
            )
        self.exchange_gap_recovery_bars = int(recovery_bars)
        ewma_span = float(self.ewma_volatility_span)
        if (
            not np.isfinite(ewma_span)
            or ewma_span < 2
            or not ewma_span.is_integer()
        ):
            raise ValueError(
                "ewma_volatility_span must be a whole number of at least 2"
            )
        self.ewma_volatility_span = int(ewma_span)
        if bool(self.ewma_volatility_enabled) and (
            not np.isfinite(float(self.ewma_daily_volatility_target))
            or float(self.ewma_daily_volatility_target) <= 0.0
        ):
            raise ValueError(
                "ewma_daily_volatility_target must be positive when EWMA volatility is enabled"
            )
        regime_method = str(self.regime_detection_method or "").strip().lower()
        if regime_method not in {"rule_based", "walk_forward_kmeans"}:
            raise ValueError(
                "regime_detection_method must be rule_based or walk_forward_kmeans"
            )
        self.regime_detection_method = regime_method
        for name, minimum in (
            ("regime_statistical_clusters", 2),
            ("regime_statistical_min_history", 20),
            ("regime_statistical_lookback", 20),
            ("regime_statistical_refit_interval", 1),
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < minimum or not value.is_integer():
                raise ValueError(
                    f"{name} must be a whole number of at least {minimum}"
                )
            setattr(self, name, int(value))
        if self.regime_statistical_clusters > 8:
            raise ValueError("regime_statistical_clusters must not exceed 8")
        if self.regime_statistical_lookback < self.regime_statistical_min_history:
            raise ValueError(
                "regime_statistical_lookback must be at least regime_statistical_min_history"
            )
        seed = float(self.regime_statistical_random_seed)
        if not np.isfinite(seed) or not seed.is_integer():
            raise ValueError("regime_statistical_random_seed must be a whole number")
        self.regime_statistical_random_seed = int(seed)


@dataclass
class BacktestResult:
    final_balance: float
    total_return: float
    max_drawdown: float
    trades: int
    leverage: int
    win_rate: float
    profit_factor: float
    sharpe_like: float
    report_path: str | None = None
    annualized_return: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    fee_ratio: float = 0.0
    gross_return_before_cost: float = 0.0
    duration_days: float = 0.0
    periods_per_year: float = 0.0
    configured_risk_reward_ratio: float = 0.0
    avg_win_return: float = 0.0
    avg_loss_return: float = 0.0
    realized_avg_win_loss_ratio: float = 0.0
    expectancy_after_cost: float = 0.0
    expectancy_per_trade_after_cost: float = 0.0
    rr_gate_passed: bool = False
    avg_exposure: float = 0.0
    active_avg_exposure: float = 0.0
    active_avg_position_fraction: float = 0.0
    ewma_volatility_enabled: bool = False
    ewma_volatility_span: int = 48
    average_ewma_volatility: float = 0.0
    ewma_daily_volatility_target: float = 0.03
    average_ewma_daily_volatility: float = 0.0
    average_volatility_position_scale: float = 1.0
    notional_turnover: float = 0.0
    commission_drag: float = 0.0
    slippage_drag: float = 0.0
    fee_drag: float = 0.0
    funding_drag: float = 0.0
    funding_debit: float = 0.0
    funding_credit: float = 0.0
    funding_rate_source: str = "configured_absolute_buffer"
    funding_settlement_mode: str = "prorated"
    total_cost_drag: float = 0.0
    average_slippage_cost: float = 0.0
    average_fill_ratio: float = 1.0
    execution_events: int = 0
    minimum_order_rejections: int = 0
    exchange_filter_rejections: int = 0
    quantity_rounding_loss_usdt: float = 0.0
    exchange_min_notional_usdt: float = 0.0
    exchange_min_quantity: float = 0.0
    exchange_max_quantity: float = 0.0
    exchange_quantity_step: float = 0.0
    exchange_price_tick_size: float = 0.0
    maximum_quantity_limited_orders: int = 0
    exchange_downtime_guard_enabled: bool = True
    exchange_gap_recovery_bars: int = 1
    exchange_downtime_blocked_orders: int = 0
    exchange_available_rate: float = 1.0
    liquidity_execution_enabled: bool = False
    max_bar_participation_rate: float = 0.0
    liquidity_limited_orders: int = 0
    average_liquidity_fill_ratio: float = 1.0
    average_market_participation_rate: float = 0.0
    average_effective_slippage_rate: float = 0.0
    max_effective_slippage_rate: float = 0.0
    maker_turnover: float = 0.0
    taker_turnover: float = 0.0
    execution_latency_bars: int = 0
    effective_long_threshold: float = 0.57
    effective_short_threshold: float = 0.43
    effective_short_signal_threshold: float = 0.57
    effective_trade_signal_threshold: float = 0.57
    effective_direction_long_threshold: float = 0.57
    effective_direction_short_threshold: float = 0.57
    effective_direction_trade_threshold: float = 0.57
    cost_edge_pass_rate: float = 1.0
    required_edge: float = 0.0
    regime_gate_pass_rate: float = 1.0
    tradeability_gate_pass_rate: float = 1.0
    horizon_gate_pass_rate: float = 1.0
    horizon_assisted_trades: int = 0
    horizon_confirmation_enabled: bool = False
    long_trades: int = 0
    short_trades: int = 0
    long_total_return: float = 0.0
    short_total_return: float = 0.0
    long_profit_factor: float = 0.0
    short_profit_factor: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_commission_drag: float = 0.0
    short_commission_drag: float = 0.0
    long_slippage_drag: float = 0.0
    short_slippage_drag: float = 0.0
    long_fee_drag: float = 0.0
    short_fee_drag: float = 0.0
    atr_exit_enabled: bool = False
    trade_side_policy: str = "both"
    platform_event_gate_enabled: bool = False
    platform_event_gate_pass_rate: float = 1.0
    drawdown_cooldown_enabled: bool = False
    cooldown_bars_triggered: int = 0
    strategy_archetype_policy: str = "all"
    strategy_archetype_policy_pass_rate: float = 1.0
    volatility_regime_policy: str = "all"
    volatility_regime_policy_pass_rate: float = 1.0
    regime_detection_method_requested: str = "rule_based"
    regime_method_counts: dict[str, int] = field(default_factory=dict)
    regime_model_version_counts: dict[str, int] = field(default_factory=dict)
    regime_fallback_reason_counts: dict[str, int] = field(default_factory=dict)
    regime_override_reason_counts: dict[str, int] = field(default_factory=dict)
    market_structure_gate_enabled: bool = False
    market_structure_gate_pass_rate: float = 1.0
    crowding_filter_enabled: bool = False
    crowding_filter_pass_rate: float = 1.0
    funding_crowding_guard_enabled: bool = False
    funding_crowding_gate_pass_rate: float = 1.0
    funding_crowding_blocked_rows: int = 0
    regime_risk_guard_enabled: bool = False
    regime_risk_gate_pass_rate: float = 1.0
    regime_risk_blocked_rows: int = 0
    max_notional_exposure: float = 0.45
    liquidation_guard_enabled: bool = True
    maintenance_margin_rate: float = 0.005
    liquidation_buffer: float = 0.01
    liquidation_fee_rate: float = 0.005
    liquidation_price_distance: float = 0.985
    liquidation_events: int = 0
    liquidation_gap_events: int = 0
    liquidation_fee_drag: float = 0.0
    liquidation_forced_turnover: float = 0.0
    strategy_archetype_summary: dict[str, dict[str, float | int]] = field(default_factory=dict)
    regime_summary: dict[str, dict[str, float | int]] = field(default_factory=dict)
    performance_by_year: dict[str, dict[str, float | int]] = field(default_factory=dict)
    performance_by_month: dict[str, dict[str, float | int]] = field(default_factory=dict)
    performance_by_symbol: dict[str, dict[str, float | int]] = field(default_factory=dict)


def effective_thresholds(cfg: BacktestConfig) -> tuple[float, float]:
    long_threshold = max(float(cfg.long_threshold), 0.5 + float(cfg.min_confidence_gap))
    short_threshold = min(float(cfg.short_threshold), 0.5 - float(cfg.min_confidence_gap))
    return long_threshold, short_threshold


def choose_position(prob_up: float, cfg: BacktestConfig) -> int:
    long_threshold, short_threshold = effective_thresholds(cfg)
    if prob_up >= long_threshold:
        return 1
    if prob_up <= short_threshold:
        return -1
    return 0


def choose_position_from_direction_scores(prob_long: float, prob_short: float, prob_trade: float, cfg: BacktestConfig) -> int:
    if prob_trade < float(getattr(cfg, "trade_signal_threshold", 0.57)):
        return 0
    long_threshold, short_threshold = effective_thresholds(cfg)
    short_signal_threshold = max(1.0 - short_threshold, 0.5 + float(cfg.min_confidence_gap))
    margin = float(getattr(cfg, "direction_signal_margin", 0.08))
    if prob_long >= long_threshold and prob_long >= prob_short + margin:
        return 1
    if prob_short >= short_signal_threshold and prob_short >= prob_long + margin:
        return -1
    return 0


def auxiliary_signal_threshold(bundle: ModelBundle, key: str, default: float) -> float:
    metadata = getattr(bundle, "auxiliary_metadata", {}) or {}
    item = metadata.get(key, {}) if isinstance(metadata, dict) else {}
    try:
        value = float(item.get("selected_signal_threshold", default))
    except (TypeError, ValueError, AttributeError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(np.clip(value, 0.01, 0.99))


def choose_positions_from_direction_arrays(
    prob_long: np.ndarray,
    prob_short: np.ndarray,
    prob_trade: np.ndarray,
    *,
    long_threshold: float,
    short_threshold: float,
    trade_threshold: float,
    margin: float,
) -> np.ndarray:
    long_signal = (
        (prob_trade >= float(trade_threshold))
        & (prob_long >= float(long_threshold))
        & (prob_long >= prob_short + float(margin))
    )
    short_signal = (
        (prob_trade >= float(trade_threshold))
        & (prob_short >= float(short_threshold))
        & (prob_short >= prob_long + float(margin))
    )
    return np.where(long_signal, 1, np.where(short_signal, -1, 0)).astype(int)


def horizon_signal_context(bundle: ModelBundle, x: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    horizon_probabilities = bundle.predict_horizon_probabilities(x)
    payload: dict[str, np.ndarray] = {}
    helper_keys = []
    for key, value in horizon_probabilities.items():
        array = np.asarray(value, dtype=float)
        payload[str(key)] = array
        if str(key) != "main":
            helper_keys.append(str(key))
    if not helper_keys:
        length = len(next(iter(payload.values()))) if payload else len(x)
        return payload, np.full(length, 0.5, dtype=float), False
    stacked = np.vstack([payload[key] for key in helper_keys])
    return payload, stacked.mean(axis=0), True


def apply_horizon_gate_and_assist(
    raw_position: np.ndarray,
    prob_up: np.ndarray,
    horizon_score: np.ndarray,
    has_horizon_signal: bool,
    cfg: BacktestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not has_horizon_signal:
        return raw_position, np.ones(len(raw_position), dtype=bool), np.zeros(len(raw_position), dtype=bool)

    position = raw_position.copy()
    assisted = np.zeros(len(position), dtype=bool)
    if cfg.horizon_assist_enabled:
        long_assist = (
            (position == 0)
            & (prob_up >= float(cfg.horizon_assist_long_min_main))
            & (horizon_score >= float(cfg.horizon_assist_long_probability))
        )
        short_assist = (
            (position == 0)
            & (prob_up <= float(cfg.horizon_assist_short_max_main))
            & (horizon_score <= float(cfg.horizon_assist_short_probability))
        )
        position = np.where(long_assist, 1, np.where(short_assist, -1, position))
        assisted = long_assist | short_assist

    if not cfg.horizon_confirmation_enabled:
        return position, np.ones(len(position), dtype=bool), assisted

    long_pass = np.where(position > 0, horizon_score >= float(cfg.horizon_long_min_probability), True)
    short_pass = np.where(position < 0, horizon_score <= float(cfg.horizon_short_max_probability), True)
    gate_pass = long_pass & short_pass
    return np.where(gate_pass, position, 0), gate_pass, assisted


def apply_side_policy(position: np.ndarray, cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray]:
    policy = str(getattr(cfg, "trade_side_policy", "both") or "both").strip().lower()
    if policy in {"long", "long_only"}:
        allowed = position >= 0
    elif policy in {"short", "short_only"}:
        allowed = position <= 0
    elif policy in {"none", "no_trade"}:
        allowed = np.zeros(len(position), dtype=bool)
    else:
        allowed = np.ones(len(position), dtype=bool)
    return np.where(allowed, position, 0), allowed


def cost_edge_filter(
    data: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Require enough ATR edge to cover causal expected execution costs."""

    maker_fraction = float(np.clip(cfg.maker_fill_fraction, 0.0, 1.0))
    expected_fee = (
        maker_fraction * float(cfg.maker_fee_rate)
        + (1.0 - maker_fraction) * float(cfg.fee_rate)
    )
    expected_order_fraction = min(
        float(cfg.max_notional_exposure),
        float(cfg.max_position_fraction) * float(cfg.leverage),
    )
    expected_taker_turnover = np.full(
        len(data),
        expected_order_fraction * (1.0 - maker_fraction),
        dtype=float,
    )
    slippage = estimate_dynamic_slippage(
        data,
        expected_taker_turnover,
        cfg,
        profile=causal_liquidity_profile(data, cfg),
    ).effective_rate
    required_edge = (
        float(cfg.min_atr_cost_multiplier) * (expected_fee + slippage)
        + float(cfg.funding_rate_buffer)
    )
    if not cfg.cost_filter_enabled or "atr_14" not in data.columns:
        return np.ones(len(data), dtype=bool), required_edge
    atr = pd.to_numeric(data["atr_14"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    atr_values = atr.fillna(0.0).to_numpy(dtype=float)
    return atr_values >= required_edge, required_edge


def regime_gate_filter(data: pd.DataFrame, position: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    if not cfg.regime_gate_enabled:
        return np.ones(len(data), dtype=bool)
    required = {"trend_strength_12_48", "efficiency_ratio_48", "range_position_20"}
    if not required.issubset(set(data.columns)):
        return np.ones(len(data), dtype=bool)

    trend = pd.to_numeric(data["trend_strength_12_48"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    efficiency = pd.to_numeric(data["efficiency_ratio_48"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    range_position = pd.to_numeric(data["range_position_20"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    zero_series = pd.Series(0.0, index=data.index)
    trend_breakout = pd.to_numeric(
        data.get("trend_breakout_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    trend_breakdown = pd.to_numeric(
        data.get("trend_breakdown_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    range_reversion_long = pd.to_numeric(
        data.get("grid_reversion_long_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    range_reversion_short = pd.to_numeric(
        data.get("grid_reversion_short_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)

    trend_long = ((trend >= 0.0) & (efficiency >= float(cfg.trend_gate_min_efficiency))) | (
        trend_breakout >= float(cfg.trend_breakout_min_score)
    )
    trend_short = ((trend <= 0.0) & (efficiency >= float(cfg.trend_gate_min_efficiency))) | (
        trend_breakdown >= float(cfg.trend_breakout_min_score)
    )
    range_mode = efficiency <= float(cfg.range_gate_max_efficiency)
    range_long = (range_mode & (range_position <= float(cfg.range_long_max_position))) | (
        range_reversion_long >= float(cfg.range_reversion_min_score)
    )
    range_short = (range_mode & (range_position >= float(cfg.range_short_min_position))) | (
        range_reversion_short >= float(cfg.range_reversion_min_score)
    )
    long_allowed = trend_long | range_long
    short_allowed = trend_short | range_short
    return np.where(position > 0, long_allowed, np.where(position < 0, short_allowed, True))


def volatility_regime_filter(data: pd.DataFrame, position: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    policy = str(getattr(cfg, "volatility_regime_policy", "all") or "all").strip().lower()
    if policy in {"", "all", "any"} or "atr_14" not in data.columns:
        return np.ones(len(data), dtype=bool)
    atr = pd.to_numeric(data["atr_14"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    low_cut, high_cut = causal_volatility_thresholds(
        atr,
        lookback=int(getattr(cfg, "volatility_regime_lookback", 240)),
        low_quantile=float(getattr(cfg, "volatility_regime_low_quantile", 0.33)),
        high_quantile=float(getattr(cfg, "volatility_regime_high_quantile", 0.67)),
    )
    values = atr.ffill().fillna(0.0).to_numpy(dtype=float)
    low_values = low_cut.to_numpy(dtype=float)
    high_values = high_cut.to_numpy(dtype=float)
    low_vol = values <= low_values
    high_vol = values >= high_values
    mid_vol = (~low_vol) & (~high_vol)
    if policy in {"high", "high_vol", "high_vol_only", "breakout_vol"}:
        allowed = high_vol
    elif policy in {"mid_high", "mid_high_vol", "not_low", "not_low_vol", "tradable_vol"}:
        allowed = mid_vol | high_vol
    elif policy in {"low", "low_vol", "low_vol_only", "range_low_vol"}:
        allowed = low_vol
    elif policy in {"mid", "mid_vol", "mid_vol_only"}:
        allowed = mid_vol
    else:
        allowed = np.ones(len(data), dtype=bool)
    return np.where(position != 0, allowed, True)


def platform_event_scores(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not len(data):
        return np.array([], dtype=float), np.array([], dtype=float)
    zero_series = pd.Series(0.0, index=data.index)
    if "platform_event_long_score" in data.columns and "platform_event_short_score" in data.columns:
        long_score = pd.to_numeric(data["platform_event_long_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        short_score = pd.to_numeric(data["platform_event_short_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.clip(long_score, 0.0, 1.0), np.clip(short_score, 0.0, 1.0)

    strategy_long = pd.to_numeric(data.get("platform_strategy_long_score", zero_series), errors="coerce").fillna(0.0)
    strategy_short = pd.to_numeric(data.get("platform_strategy_short_score", zero_series), errors="coerce").fillna(0.0)
    trend_long = pd.to_numeric(data.get("trend_breakout_score_20", zero_series), errors="coerce").fillna(0.0) / 0.00035
    trend_short = pd.to_numeric(data.get("trend_breakdown_score_20", zero_series), errors="coerce").fillna(0.0) / 0.00035
    follow_long = pd.to_numeric(data.get("breakout_followthrough_long_3", zero_series), errors="coerce").fillna(0.0) / 0.003
    follow_short = pd.to_numeric(data.get("breakout_followthrough_short_3", zero_series), errors="coerce").fillna(0.0) / 0.003
    grid_long = pd.to_numeric(data.get("grid_reversion_long_score_20", zero_series), errors="coerce").fillna(0.0)
    grid_short = pd.to_numeric(data.get("grid_reversion_short_score_20", zero_series), errors="coerce").fillna(0.0)
    exhaustion_long = pd.to_numeric(data.get("exhaustion_reversal_long_score", zero_series), errors="coerce").fillna(0.0) / 0.012
    exhaustion_short = pd.to_numeric(data.get("exhaustion_reversal_short_score", zero_series), errors="coerce").fillna(0.0) / 0.012
    long_score = pd.concat([strategy_long, trend_long, follow_long, grid_long, exhaustion_long], axis=1).clip(0.0, 1.0).max(axis=1)
    short_score = pd.concat([strategy_short, trend_short, follow_short, grid_short, exhaustion_short], axis=1).clip(0.0, 1.0).max(axis=1)
    return long_score.to_numpy(dtype=float), short_score.to_numpy(dtype=float)


def platform_strategy_quality_filter(
    data: pd.DataFrame,
    position: np.ndarray,
    cfg: BacktestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not len(data):
        empty = np.array([], dtype=float)
        return np.array([], dtype=bool), empty, empty, empty, empty
    zero_series = pd.Series(0.0, index=data.index)
    if "platform_strategy_long_score" in data.columns:
        long_score = pd.to_numeric(data["platform_strategy_long_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        long_score, _ = platform_event_scores(data)
    if "platform_strategy_short_score" in data.columns:
        short_score = pd.to_numeric(data["platform_strategy_short_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        _, short_score = platform_event_scores(data)
    long_risk = pd.to_numeric(data.get("crowding_long_risk", zero_series), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    short_risk = pd.to_numeric(data.get("crowding_short_risk", zero_series), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gate = np.ones(len(data), dtype=bool)
    active = position != 0
    if bool(getattr(cfg, "market_structure_gate_enabled", False)):
        min_score = float(getattr(cfg, "min_market_structure_score", 0.55))
        score_pass = np.where(position > 0, long_score >= min_score, np.where(position < 0, short_score >= min_score, True))
        gate = gate & np.where(active, score_pass, True)
    if bool(getattr(cfg, "crowding_filter_enabled", False)):
        max_risk = float(getattr(cfg, "max_crowding_risk", 0.82))
        risk_pass = np.where(position > 0, long_risk <= max_risk, np.where(position < 0, short_risk <= max_risk, True))
        gate = gate & np.where(active, risk_pass, True)
    return gate, np.clip(long_score, 0.0, 1.0), np.clip(short_score, 0.0, 1.0), np.clip(long_risk, 0.0, 1.0), np.clip(short_risk, 0.0, 1.0)


def platform_event_gate_filter(data: pd.DataFrame, position: np.ndarray, cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_score, short_score = platform_event_scores(data)
    if not bool(getattr(cfg, "platform_event_gate_enabled", False)):
        return np.ones(len(data), dtype=bool), long_score, short_score
    min_score = float(getattr(cfg, "platform_event_min_score", 0.65))
    long_allowed = long_score >= min_score
    short_allowed = short_score >= min_score
    gate_pass = np.where(position > 0, long_allowed, np.where(position < 0, short_allowed, True))
    return gate_pass, long_score, short_score


def exit_thresholds(data: pd.DataFrame, cfg: BacktestConfig) -> tuple[np.ndarray, np.ndarray, bool]:
    stop_loss = stop_distance_series(data, cfg)
    take_profit = np.full(len(data), max(float(cfg.take_profit), 1e-9), dtype=float)
    enabled = bool(cfg.use_atr_exits and "atr_14" in data.columns)
    if not enabled:
        return stop_loss, take_profit, False
    take_profit_multiplier = max(float(cfg.take_profit_atr_multiplier), 1e-9)
    atr_values = causal_atr_values(
        data,
        fallback=float(cfg.take_profit) / take_profit_multiplier,
    )
    min_exit = max(float(cfg.min_exit_pct), 1e-6)
    max_exit = max(float(cfg.max_exit_pct), min_exit)
    take_profit = np.clip(atr_values * take_profit_multiplier, min_exit, max_exit)
    return stop_loss, take_profit, True


def configured_risk_reward_ratio(cfg: BacktestConfig) -> float:
    if cfg.use_atr_exits:
        stop = max(float(cfg.stop_loss_atr_multiplier), 1e-9)
        target = max(float(cfg.take_profit_atr_multiplier), 0.0)
    else:
        stop = max(float(cfg.stop_loss), 1e-9)
        target = max(float(cfg.take_profit), 0.0)
    return float(target / stop)


def apply_return_model(
    data: pd.DataFrame,
    raw_ret: np.ndarray,
    prob_long: np.ndarray,
    prob_short: np.ndarray,
    cfg: BacktestConfig,
) -> bool:
    data["position_prev"] = data["position"].shift(1).fillna(0)
    data["turnover"] = (data["position"] - data["position_prev"]).abs()
    position = data["position"].to_numpy(dtype=float)
    signal_confidence = np.where(position > 0, prob_long, np.where(position < 0, prob_short, 0.5))
    target_position_fraction = dynamic_position_fraction(
        data,
        signal_confidence,
        position,
        cfg,
        event_scores=platform_event_scores(data),
    )
    position_fraction = apply_position_rebalance_rules(position, target_position_fraction, cfg)
    data["target_position_fraction"] = target_position_fraction
    data["position_fraction"] = position_fraction
    data["risk_volatility_scale"] = volatility_position_scale(data, cfg)
    target_notional_position = position * position_fraction * float(cfg.leverage)
    stop_loss, take_profit, atr_exit_enabled = exit_thresholds(data, cfg)
    forced_flat_after = np.zeros(len(data), dtype=bool)
    for _ in range(6):
        costs = estimate_execution_costs(
            data,
            target_notional_position,
            cfg,
            forced_flat_after=forced_flat_after,
        )
        liquidation = assess_liquidations(
            data,
            costs.execution_path.executed_notional_position,
            stop_loss,
            cfg,
        )
        if np.array_equal(liquidation.triggered, forced_flat_after):
            break
        forced_flat_after = liquidation.triggered
    costs = estimate_execution_costs(
        data,
        target_notional_position,
        cfg,
        forced_flat_after=forced_flat_after,
    )
    executed_notional_position = costs.execution_path.executed_notional_position
    executed_position = np.sign(executed_notional_position)
    liquidation = assess_liquidations(
        data,
        executed_notional_position,
        stop_loss,
        cfg,
    )
    capped_ret = np.where(
        executed_position >= 0,
        np.minimum(np.maximum(raw_ret, -stop_loss), take_profit),
        np.minimum(np.maximum(raw_ret, -take_profit), stop_loss),
    )
    capped_ret = np.where(
        liquidation.triggered,
        liquidation.price_return,
        capped_ret,
    )
    total_cost = costs.total_cost + liquidation.fee_cost
    strategy_ret = executed_notional_position * capped_ret - total_cost
    data["stop_loss_pct"] = stop_loss
    data["take_profit_pct"] = take_profit
    data["atr_exit_enabled"] = atr_exit_enabled
    data["target_notional_position"] = target_notional_position
    data["delayed_target_notional"] = costs.execution_path.delayed_target_notional
    data["executed_notional_position"] = executed_notional_position
    data["executed_position"] = executed_position.astype(int)
    data["requested_notional_change"] = costs.execution_path.requested_notional_change
    data["position_before_execution"] = (
        costs.execution_path.position_before_execution
    )
    data["executed_notional_change"] = costs.execution_path.executed_notional_change
    data["execution_fill_ratio"] = costs.execution_path.fill_ratio
    data["minimum_order_rejected"] = costs.execution_path.minimum_order_rejected
    data["exchange_filter_rejected"] = costs.execution_path.exchange_filter_rejected
    data["exchange_order_quantity"] = costs.execution_path.exchange_order_quantity
    data["exchange_order_notional_usdt"] = (
        costs.execution_path.exchange_order_notional_usdt
    )
    data["quantity_rounding_loss_usdt"] = (
        costs.execution_path.quantity_rounding_loss_usdt
    )
    data["maximum_quantity_limited"] = (
        costs.execution_path.maximum_quantity_limited
    )
    data["exchange_rejection_reason"] = (
        costs.execution_path.exchange_rejection_reason
    )
    data["execution_available"] = costs.execution_path.execution_available
    data["exchange_downtime_blocked"] = (
        costs.execution_path.exchange_downtime_blocked
    )
    data["exchange_downtime_reason"] = (
        costs.execution_path.exchange_downtime_reason
    )
    data["execution_reason_code"] = np.where(
        costs.execution_path.exchange_downtime_blocked,
        "blocked_exchange_unavailable",
        np.where(
            np.abs(costs.execution_path.executed_notional_change) > 1e-15,
            "execution_filled",
            np.where(
                np.abs(costs.execution_path.requested_notional_change) > 1e-15,
                costs.execution_path.exchange_rejection_reason,
                "no_execution_requested",
            ),
        ),
    )
    data["liquidity_fill_ratio"] = costs.execution_path.liquidity_fill_ratio
    data["liquidity_capacity_usdt"] = (
        costs.execution_path.liquidity_capacity_usdt
    )
    data["liquidity_limited"] = costs.execution_path.liquidity_limited
    data["liquidation_forced_notional"] = costs.execution_path.forced_flat_notional
    data["liquidation_forced_turnover"] = costs.execution_path.forced_flat_turnover
    data["end_notional_position"] = np.where(
        liquidation.triggered,
        0.0,
        executed_notional_position,
    )
    data["notional_exposure"] = np.abs(executed_notional_position)
    data["notional_turnover"] = costs.notional_turnover
    data["maker_notional_turnover"] = costs.maker_notional_turnover
    data["taker_notional_turnover"] = costs.taker_notional_turnover
    data["commission_cost"] = costs.commission_cost
    data["slippage_cost"] = costs.slippage_cost
    data["effective_slippage_rate"] = costs.effective_slippage_rate
    data["market_participation_rate"] = costs.market_participation_rate
    data["liquidity_stress"] = costs.liquidity_stress
    data["trade_cost"] = costs.trade_cost
    data["funding_cost"] = costs.funding_cost
    data["funding_notional_position"] = (
        costs.execution_path.position_before_execution
        if costs.funding_settlement_mode == "event"
        else executed_notional_position
    )
    data["funding_rate_used"] = costs.funding_rate_used
    data["funding_rate_source"] = costs.funding_rate_source
    data["funding_settlement_mode"] = costs.funding_settlement_mode
    data["liquidation_triggered"] = liquidation.triggered
    data["liquidation_gap_triggered"] = liquidation.gap_triggered
    data["liquidation_price_distance"] = liquidation.price_distance
    data["liquidation_price_return"] = liquidation.price_return
    data["liquidation_fee_cost"] = liquidation.fee_cost
    data["liquidation_reason"] = liquidation.reason
    data["total_cost"] = total_cost
    data["funding_period_fraction"] = costs.funding_period_fraction
    data["strategy_return"] = strategy_ret
    data["equity"] = cfg.initial_balance * (1 + data["strategy_return"]).cumprod()
    return atr_exit_enabled


def side_metrics(data: pd.DataFrame, side: int) -> dict[str, float | int]:
    side_source = data.get(
        "performance_side",
        data.get("executed_position", data["position"]),
    )
    side_mask = pd.to_numeric(
        side_source,
        errors="coerce",
    ).fillna(0.0) == side
    rows = data[side_mask]
    if len(data):
        side_cost_mask = side_mask
        costs = data.loc[side_cost_mask, "total_cost"] if "total_cost" in data.columns else data.loc[side_cost_mask, "trade_cost"]
        commissions = data.loc[side_cost_mask, "commission_cost"] if "commission_cost" in data.columns else pd.Series(dtype=float)
        slippage = data.loc[side_cost_mask, "slippage_cost"] if "slippage_cost" in data.columns else pd.Series(dtype=float)
        funding_costs = data.loc[side_cost_mask, "funding_cost"] if "funding_cost" in data.columns else pd.Series(dtype=float)
    else:
        costs = pd.Series(dtype=float)
        commissions = pd.Series(dtype=float)
        slippage = pd.Series(dtype=float)
        funding_costs = pd.Series(dtype=float)
    wins = rows[rows["strategy_return"] > 0]
    losses = rows[rows["strategy_return"] < 0]
    gross_profit = float(wins["strategy_return"].sum())
    gross_loss = abs(float(losses["strategy_return"].sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    avg_win = float(wins["strategy_return"].mean()) if len(wins) else 0.0
    avg_loss = abs(float(losses["strategy_return"].mean())) if len(losses) else 0.0
    return {
        "trades": int(
            (
                pd.to_numeric(
                    rows.get(
                        "notional_turnover",
                        pd.Series(0.0, index=rows.index),
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                > 1e-15
            ).sum()
        ),
        "total_return": float(rows["strategy_return"].sum()) if len(rows) else 0.0,
        "profit_factor": float(profit_factor),
        "win_rate": float(len(wins) / max(len(rows), 1)),
        "avg_win_return": avg_win,
        "avg_loss_return": avg_loss,
        "realized_avg_win_loss_ratio": float(avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0),
        "commission_drag": float(commissions.sum()) if len(commissions) else 0.0,
        "slippage_drag": float(slippage.sum()) if len(slippage) else 0.0,
        "fee_drag": float(costs.sum()) if len(costs) else 0.0,
        "funding_drag": float(funding_costs.sum()) if len(funding_costs) else 0.0,
    }


def classify_strategy_archetype(data: pd.DataFrame, position: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    if not len(data):
        return np.array([], dtype=object)
    zero_series = pd.Series(0.0, index=data.index)
    half_series = pd.Series(0.5, index=data.index)
    efficiency = pd.to_numeric(data.get("efficiency_ratio_48", zero_series), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    range_position = pd.to_numeric(data.get("range_position_20", half_series), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    trend = pd.to_numeric(data.get("trend_strength_12_48", zero_series), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trend_breakout = pd.to_numeric(
        data.get("trend_breakout_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    trend_breakdown = pd.to_numeric(
        data.get("trend_breakdown_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    range_reversion_long = pd.to_numeric(
        data.get("grid_reversion_long_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    range_reversion_short = pd.to_numeric(
        data.get("grid_reversion_short_score_20", zero_series), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    range_mode = efficiency <= float(cfg.range_gate_max_efficiency)
    range_long = (range_mode & (range_position <= float(cfg.range_long_max_position))) | (
        range_reversion_long >= float(cfg.range_reversion_min_score)
    )
    range_short = (range_mode & (range_position >= float(cfg.range_short_min_position))) | (
        range_reversion_short >= float(cfg.range_reversion_min_score)
    )
    trend_long = ((trend >= 0.0) & (efficiency >= float(cfg.trend_gate_min_efficiency))) | (
        trend_breakout >= float(cfg.trend_breakout_min_score)
    )
    trend_short = ((trend <= 0.0) & (efficiency >= float(cfg.trend_gate_min_efficiency))) | (
        trend_breakdown >= float(cfg.trend_breakout_min_score)
    )
    labels = np.full(len(data), "flat", dtype=object)
    labels = np.where((position > 0) & range_long, "range_grid_long", labels)
    labels = np.where((position < 0) & range_short, "range_grid_short", labels)
    labels = np.where((position > 0) & (labels == "flat") & trend_long, "trend_breakout_long", labels)
    labels = np.where((position < 0) & (labels == "flat") & trend_short, "trend_breakout_short", labels)
    labels = np.where((position != 0) & (labels == "flat"), "model_signal_other", labels)
    return labels


def archetype_policy_filter(
    data: pd.DataFrame,
    position: np.ndarray,
    cfg: BacktestConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = classify_strategy_archetype(data, position, cfg)
    policy = str(getattr(cfg, "strategy_archetype_policy", "all") or "all").strip().lower()
    if policy in {"", "all", "any"}:
        return np.ones(len(position), dtype=bool), labels, position
    active = position != 0
    if policy in {"trend", "trend_only", "trend_breakout_only"}:
        allowed = np.array([str(label).startswith("trend_breakout") for label in labels], dtype=bool)
    elif policy in {"range", "range_only", "range_grid_only", "grid_only"}:
        allowed = np.array([str(label).startswith("range_grid") for label in labels], dtype=bool)
    elif policy in {"range_grid_long", "range_grid_long_only", "grid_long_only"}:
        allowed = np.array([str(label) == "range_grid_long" for label in labels], dtype=bool)
    elif policy in {"range_grid_short", "range_grid_short_only", "grid_short_only"}:
        allowed = np.array([str(label) == "range_grid_short" for label in labels], dtype=bool)
    elif policy in {"trend_breakout_long", "trend_breakout_long_only", "trend_long_only"}:
        allowed = np.array([str(label) == "trend_breakout_long" for label in labels], dtype=bool)
    elif policy in {"trend_breakout_short", "trend_breakout_short_only", "trend_short_only"}:
        allowed = np.array([str(label) == "trend_breakout_short" for label in labels], dtype=bool)
    elif policy in {"long_archetypes", "long_archetypes_only", "range_or_trend_long", "range_or_trend_long_only"}:
        allowed = np.array(
            [str(label) in {"range_grid_long", "trend_breakout_long"} for label in labels],
            dtype=bool,
        )
    elif policy in {"short_archetypes", "short_archetypes_only", "range_or_trend_short", "range_or_trend_short_only"}:
        allowed = np.array(
            [str(label) in {"range_grid_short", "trend_breakout_short"} for label in labels],
            dtype=bool,
        )
    elif policy in {"named_archetype", "specific"}:
        allowed = np.array([str(label) != "flat" for label in labels], dtype=bool)
    elif policy in {"model_signal_other", "model_signal_only"}:
        allowed = np.array([str(label) == "model_signal_other" for label in labels], dtype=bool)
    elif "|" in policy or "," in policy or "+" in policy:
        normalized = policy.replace("+", "|").replace(",", "|")
        allowed_names = {item.strip() for item in normalized.split("|") if item.strip()}
        allowed = np.array([str(label) in allowed_names for label in labels], dtype=bool)
    else:
        allowed = np.ones(len(position), dtype=bool)
    gate = np.where(active, allowed, True)
    return gate, labels, np.where(gate, position, 0)


def strategy_archetype_metrics(data: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    if "strategy_archetype" not in data.columns:
        return {}
    summary: dict[str, dict[str, float | int]] = {}
    for label in sorted(str(value) for value in data["strategy_archetype"].dropna().unique()):
        active = (
            pd.to_numeric(
                data.get(
                    "executed_notional_position",
                    data.get("position", pd.Series(0.0, index=data.index)),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .abs()
            > 1e-15
        )
        rows = data[(data["strategy_archetype"] == label) & active]
        if rows.empty:
            continue
        wins = rows[rows["strategy_return"] > 0]
        losses = rows[rows["strategy_return"] < 0]
        gross_profit = float(wins["strategy_return"].sum())
        gross_loss = abs(float(losses["strategy_return"].sum()))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 0.0
        equity = (1.0 + rows["strategy_return"]).cumprod().to_numpy(dtype=float)
        peak = np.maximum.accumulate(equity) if len(equity) else np.array([], dtype=float)
        drawdown = equity / peak - 1.0 if len(peak) else np.array([], dtype=float)
        summary[label] = {
            "rows": int(len(rows)),
            "trades": int(
                (
                    pd.to_numeric(
                        rows.get(
                            "notional_turnover",
                            pd.Series(0.0, index=rows.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    > 1e-15
                ).sum()
            ),
            "total_return": float(rows["strategy_return"].sum()),
            "profit_factor": float(profit_factor),
            "win_rate": float(len(wins) / max(len(rows), 1)),
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "fee_drag": float(rows["trade_cost"].sum()) if "trade_cost" in rows.columns else 0.0,
            "long_rows": int((rows["position"] > 0).sum()),
            "short_rows": int((rows["position"] < 0).sum()),
        }
    return summary


def run_backtest(frame: pd.DataFrame, bundle: ModelBundle, cfg: BacktestConfig) -> tuple[BacktestResult, pd.DataFrame]:
    source_attrs = dict(frame.attrs)
    x, _ = feature_only_matrix(frame, bundle.feature_columns)
    direction_prob = bundle.predict_direction_probabilities(x)
    horizon_prob, horizon_score, has_horizon_signal = horizon_signal_context(bundle, x)
    prob = np.asarray(direction_prob["up"], dtype=float)
    prob_long = np.asarray(direction_prob["long"], dtype=float)
    prob_short = np.asarray(direction_prob["short"], dtype=float)
    prob_trade = np.asarray(direction_prob["trade"], dtype=float)
    uses_directional_models = bool(np.asarray(direction_prob["uses_directional_models"]).any())
    uses_tradeability_model = bool(np.asarray(direction_prob["uses_tradeability_model"]).any())
    data = frame.copy().reset_index(drop=True)
    data.attrs.update(source_attrs)
    regime_detail = detect_regime_frame(data, cfg)
    for column in regime_detail.columns:
        data[column] = regime_detail[column]
    data["prob_up"] = prob
    data["prob_long"] = prob_long
    data["prob_short"] = prob_short
    data["prob_trade"] = prob_trade
    for key, value in horizon_prob.items():
        data[f"horizon_prob_{key}"] = value
    data["horizon_signal_score"] = horizon_score
    data["uses_horizon_signal"] = has_horizon_signal
    data["uses_directional_models"] = uses_directional_models
    data["uses_tradeability_model"] = uses_tradeability_model
    effective_long_threshold, effective_short_threshold = effective_thresholds(cfg)
    effective_short_signal_threshold = max(1.0 - effective_short_threshold, 0.5 + float(cfg.min_confidence_gap))
    data["effective_long_threshold"] = effective_long_threshold
    data["effective_short_threshold"] = effective_short_threshold
    data["effective_short_signal_threshold"] = effective_short_signal_threshold
    direction_long_threshold = effective_long_threshold
    direction_short_threshold = effective_short_signal_threshold
    direction_trade_threshold = float(cfg.trade_signal_threshold)
    if uses_directional_models:
        direction_long_threshold = auxiliary_signal_threshold(bundle, "direction_long", direction_long_threshold)
        direction_short_threshold = auxiliary_signal_threshold(bundle, "direction_short", direction_short_threshold)
        direction_trade_threshold = auxiliary_signal_threshold(bundle, "direction_trade", direction_trade_threshold)
    data["effective_trade_signal_threshold"] = float(direction_trade_threshold)
    data["effective_direction_long_threshold"] = float(direction_long_threshold)
    data["effective_direction_short_threshold"] = float(direction_short_threshold)
    data["effective_direction_trade_threshold"] = float(direction_trade_threshold)
    if uses_directional_models:
        raw_position = choose_positions_from_direction_arrays(
            prob_long,
            prob_short,
            prob_trade,
            long_threshold=direction_long_threshold,
            short_threshold=direction_short_threshold,
            trade_threshold=direction_trade_threshold,
            margin=float(getattr(cfg, "direction_signal_margin", 0.08)),
        )
    else:
        raw_position = np.array([choose_position(float(p), cfg) for p in prob], dtype=int)
        if uses_tradeability_model:
            raw_position = np.where(prob_trade >= float(direction_trade_threshold), raw_position, 0)
    data["model_position"] = raw_position
    raw_position, horizon_pass, horizon_assisted = apply_horizon_gate_and_assist(
        raw_position,
        prob,
        horizon_score,
        has_horizon_signal,
        cfg,
    )
    data["horizon_gate_pass"] = horizon_pass
    data["horizon_assisted_signal"] = horizon_assisted
    raw_position, side_policy_pass = apply_side_policy(raw_position, cfg)
    data["side_policy_gate_pass"] = side_policy_pass
    data["trade_side_policy"] = str(getattr(cfg, "trade_side_policy", "both") or "both")
    cost_pass, required_edge = cost_edge_filter(data, cfg)
    data["cost_edge_pass"] = cost_pass
    data["required_edge"] = required_edge
    data["position"] = np.where(cost_pass, raw_position, 0)
    tradeability_pass = np.where(raw_position == 0, True, prob_trade >= float(direction_trade_threshold))
    data["tradeability_gate_pass"] = tradeability_pass
    data["position"] = np.where(tradeability_pass, data["position"], 0)
    regime_pass = regime_gate_filter(data, data["position"].to_numpy(dtype=int), cfg)
    data["regime_gate_pass"] = regime_pass
    data["position"] = np.where(regime_pass, data["position"], 0)
    volatility_policy_pass = volatility_regime_filter(data, data["position"].to_numpy(dtype=int), cfg)
    data["volatility_regime_policy_pass"] = volatility_policy_pass
    data["volatility_regime_policy"] = str(getattr(cfg, "volatility_regime_policy", "all") or "all")
    data["position"] = np.where(volatility_policy_pass, data["position"], 0)
    archetype_policy_pass, preliminary_archetype, filtered_position = archetype_policy_filter(
        data,
        data["position"].to_numpy(dtype=int),
        cfg,
    )
    data["preliminary_strategy_archetype"] = preliminary_archetype
    data["archetype_policy_gate_pass"] = archetype_policy_pass
    data["strategy_archetype_policy"] = str(getattr(cfg, "strategy_archetype_policy", "all") or "all")
    data["position"] = filtered_position
    platform_quality_pass, platform_strategy_long, platform_strategy_short, crowding_long, crowding_short = platform_strategy_quality_filter(
        data,
        data["position"].to_numpy(dtype=int),
        cfg,
    )
    data["platform_strategy_long_score"] = platform_strategy_long
    data["platform_strategy_short_score"] = platform_strategy_short
    data["crowding_long_risk"] = crowding_long
    data["crowding_short_risk"] = crowding_short
    data["platform_strategy_quality_gate_pass"] = platform_quality_pass
    data["position"] = np.where(platform_quality_pass, data["position"], 0)
    event_gate_pass, event_long_score, event_short_score = platform_event_gate_filter(
        data, data["position"].to_numpy(dtype=int), cfg
    )
    data["platform_event_long_score"] = event_long_score
    data["platform_event_short_score"] = event_short_score
    data["platform_event_gate_pass"] = event_gate_pass
    data["position"] = np.where(event_gate_pass, data["position"], 0)
    data["strategy_archetype"] = classify_strategy_archetype(data, data["position"].to_numpy(dtype=int), cfg)
    proposed_risk_position = data["position"].to_numpy(dtype=int).copy()
    regime_risk_gate_pass = regime_risk_gate(
        data,
        proposed_risk_position,
        cfg,
    )
    data["regime_risk_gate_pass"] = regime_risk_gate_pass
    data["position"] = np.where(
        regime_risk_gate_pass,
        data["position"],
        0,
    )
    funding_gate_pass, funding_rate_risk = funding_crowding_gate(
        data,
        proposed_risk_position,
        cfg,
    )
    data["funding_crowding_gate_pass"] = funding_gate_pass
    data["funding_crowding_rate"] = funding_rate_risk
    data["position"] = np.where(
        funding_gate_pass,
        data["position"],
        0,
    )

    raw_ret = (data["close"].shift(-1) / data["close"] - 1.0).fillna(0).to_numpy()
    atr_exit_enabled = apply_return_model(data, raw_ret, prob_long, prob_short, cfg)
    proposed_risk_fraction = data["target_position_fraction"].to_numpy(dtype=float).copy()
    cooldown_gate = drawdown_cooldown_gate(data["strategy_return"].to_numpy(dtype=float), cfg)
    data = data.copy()
    data["drawdown_cooldown_gate_pass"] = cooldown_gate
    if bool(getattr(cfg, "drawdown_cooldown_enabled", False)):
        data["position"] = np.where(cooldown_gate, data["position"], 0)
        data["strategy_archetype"] = classify_strategy_archetype(data, data["position"].to_numpy(dtype=int), cfg)
        atr_exit_enabled = apply_return_model(data, raw_ret, prob_long, prob_short, cfg)
    risk_detail = build_risk_decision_frame(
        data,
        proposed_risk_position,
        proposed_risk_fraction,
        cooldown_gate,
        cfg,
        funding_gate=funding_gate_pass,
        regime_gate=regime_risk_gate_pass,
    )
    data = pd.concat(
        [
            data.drop(
                columns=list(risk_detail.columns),
                errors="ignore",
            ),
            risk_detail,
        ],
        axis=1,
    ).copy()
    data.attrs.update(source_attrs)
    data["decision_reason_code"] = decision_reason_codes(data)

    equity = data["equity"].to_numpy()
    executed_side = pd.to_numeric(
        data["executed_position"],
        errors="coerce",
    ).fillna(0.0)
    prior_execution_side = np.sign(
        pd.to_numeric(
            data["position_before_execution"],
            errors="coerce",
        )
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    close_cost_side = (
        (executed_side.to_numpy(dtype=float) == 0.0)
        & (
            pd.to_numeric(
                data["executed_notional_change"],
                errors="coerce",
            )
            .fillna(0.0)
            .abs()
            .to_numpy(dtype=float)
            > 1e-15
        )
    )
    data["performance_side"] = np.where(
        close_cost_side,
        prior_execution_side,
        executed_side.to_numpy(dtype=float),
    ).astype(int)
    execution_active = (
        data["executed_notional_position"].abs() > 1e-15
    ) | (
        data["total_cost"].abs() > 1e-15
    )
    trade_rows = data[execution_active]
    wins = trade_rows[trade_rows["strategy_return"] > 0]
    losses = trade_rows[trade_rows["strategy_return"] < 0]
    win_rate = float(len(wins) / max(len(trade_rows), 1))
    gross_profit = float(wins["strategy_return"].sum())
    gross_loss = abs(float(losses["strategy_return"].sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    avg_win_return = float(wins["strategy_return"].mean()) if len(wins) else 0.0
    avg_loss_return = abs(float(losses["strategy_return"].mean())) if len(losses) else 0.0
    realized_avg_win_loss_ratio = (
        float(avg_win_return / avg_loss_return)
        if avg_loss_return > 0
        else (999.0 if avg_win_return > 0 else 0.0)
    )
    active_returns = trade_rows["strategy_return"] if len(trade_rows) else pd.Series(dtype=float)
    active_exposure = data.loc[execution_active, "notional_exposure"] if len(data) else pd.Series(dtype=float)
    active_position_fraction = data.loc[execution_active, "position_fraction"] if len(data) else pd.Series(dtype=float)
    trade_count = int((data["notional_turnover"] > 1e-15).sum())
    net_return_sum = float(data["strategy_return"].sum()) if len(data) else 0.0
    expectancy_after_cost = float(active_returns.mean()) if len(active_returns) else 0.0
    expectancy_per_trade_after_cost = float(net_return_sum / max(trade_count, 1))
    performance = evaluate_backtest_performance(
        data,
        initial_balance=cfg.initial_balance,
    )
    long_stats = side_metrics(data, 1)
    short_stats = side_metrics(data, -1)
    archetype_summary = strategy_archetype_metrics(data)
    regime_summary = summarize_regime_performance(data)
    regime_method_counts = {
        str(key): int(value)
        for key, value in data["regime_method_used"].value_counts().items()
    }
    regime_model_version_counts = {
        str(key): int(value)
        for key, value in data["regime_model_version"].value_counts().items()
    }
    regime_fallback_reason_counts = {
        str(key): int(value)
        for key, value in data.loc[
            data["regime_fallback_reason"].astype(str).str.len() > 0,
            "regime_fallback_reason",
        ]
        .value_counts()
        .items()
    }
    regime_override_reason_counts = {
        str(key): int(value)
        for key, value in data.loc[
            data["regime_override_reason"].astype(str).str.len() > 0,
            "regime_override_reason",
        ]
        .value_counts()
        .items()
    }

    result = BacktestResult(
        final_balance=float(equity[-1]) if len(equity) else cfg.initial_balance,
        total_return=performance.total_return,
        max_drawdown=performance.max_drawdown,
        trades=trade_count,
        leverage=int(getattr(cfg, "leverage", 1) or 1),
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_like=performance.sharpe_like,
        annualized_return=performance.annualized_return,
        sortino_ratio=performance.sortino_ratio,
        calmar_ratio=performance.calmar_ratio,
        fee_ratio=performance.fee_ratio,
        gross_return_before_cost=performance.gross_return_before_cost,
        duration_days=performance.duration_days,
        periods_per_year=performance.periods_per_year,
        configured_risk_reward_ratio=configured_risk_reward_ratio(cfg),
        avg_win_return=avg_win_return,
        avg_loss_return=avg_loss_return,
        realized_avg_win_loss_ratio=realized_avg_win_loss_ratio,
        expectancy_after_cost=expectancy_after_cost,
        expectancy_per_trade_after_cost=expectancy_per_trade_after_cost,
        rr_gate_passed=bool(realized_avg_win_loss_ratio >= 0.90),
        avg_exposure=float(data["notional_exposure"].mean()) if len(data) else 0.0,
        active_avg_exposure=float(active_exposure.mean()) if len(active_exposure) else 0.0,
        active_avg_position_fraction=float(active_position_fraction.mean()) if len(active_position_fraction) else 0.0,
        ewma_volatility_enabled=bool(
            getattr(cfg, "ewma_volatility_enabled", False)
        ),
        ewma_volatility_span=int(
            getattr(cfg, "ewma_volatility_span", 48)
        ),
        average_ewma_volatility=float(
            pd.to_numeric(
                data.get(
                    "risk_ewma_volatility",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        )
        if "risk_ewma_volatility" in data.columns and len(data)
        else 0.0,
        ewma_daily_volatility_target=float(
            getattr(cfg, "ewma_daily_volatility_target", 0.03)
        ),
        average_ewma_daily_volatility=float(
            pd.to_numeric(
                data.get(
                    "risk_ewma_daily_volatility",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        )
        if "risk_ewma_daily_volatility" in data.columns and len(data)
        else 0.0,
        average_volatility_position_scale=float(
            pd.to_numeric(
                data.get(
                    "risk_volatility_scale",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).mean()
        )
        if "risk_volatility_scale" in data.columns and len(data)
        else 1.0,
        notional_turnover=float(data["notional_turnover"].sum()) if len(data) else 0.0,
        commission_drag=float(data["commission_cost"].sum()) if len(data) else 0.0,
        slippage_drag=float(data["slippage_cost"].sum()) if len(data) else 0.0,
        fee_drag=float(data["trade_cost"].sum()) if len(data) else 0.0,
        funding_drag=float(data["funding_cost"].sum()) if len(data) else 0.0,
        funding_debit=(
            float(data["funding_cost"].clip(lower=0.0).sum()) if len(data) else 0.0
        ),
        funding_credit=(
            float((-data["funding_cost"].clip(upper=0.0)).sum())
            if len(data)
            else 0.0
        ),
        funding_rate_source=(
            str(data["funding_rate_source"].iloc[0])
            if len(data)
            else "configured_absolute_buffer"
        ),
        funding_settlement_mode=(
            str(data["funding_settlement_mode"].iloc[0])
            if len(data)
            else "prorated"
        ),
        total_cost_drag=float(data["total_cost"].sum()) if len(data) else 0.0,
        liquidation_guard_enabled=bool(cfg.liquidation_guard_enabled),
        maintenance_margin_rate=float(cfg.maintenance_margin_rate),
        liquidation_buffer=float(cfg.liquidation_buffer),
        liquidation_fee_rate=float(cfg.liquidation_fee_rate),
        liquidation_price_distance=liquidation_price_distance(cfg),
        liquidation_events=int(data["liquidation_triggered"].sum()),
        liquidation_gap_events=int(data["liquidation_gap_triggered"].sum()),
        liquidation_fee_drag=float(data["liquidation_fee_cost"].sum()),
        liquidation_forced_turnover=float(
            data["liquidation_forced_turnover"].sum()
        ),
        average_slippage_cost=(
            float(data.loc[data["notional_turnover"] > 0, "slippage_cost"].mean())
            if bool((data["notional_turnover"] > 0).any())
            else 0.0
        ),
        average_fill_ratio=(
            float(
                data.loc[
                    data["requested_notional_change"].abs() > 1e-15,
                    "execution_fill_ratio",
                ].mean()
            )
            if bool((data["requested_notional_change"].abs() > 1e-15).any())
            else 1.0
        ),
        execution_events=int((data["notional_turnover"] > 0).sum()),
        minimum_order_rejections=int(data["minimum_order_rejected"].sum()),
        exchange_filter_rejections=int(data["exchange_filter_rejected"].sum()),
        quantity_rounding_loss_usdt=float(
            data["quantity_rounding_loss_usdt"].sum()
        ),
        exchange_min_notional_usdt=float(cfg.exchange_min_notional_usdt),
        exchange_min_quantity=float(cfg.exchange_min_quantity),
        exchange_max_quantity=float(cfg.exchange_max_quantity),
        exchange_quantity_step=float(cfg.exchange_quantity_step),
        exchange_price_tick_size=float(cfg.exchange_price_tick_size),
        maximum_quantity_limited_orders=int(
            data["maximum_quantity_limited"].sum()
        ),
        exchange_downtime_guard_enabled=bool(
            cfg.exchange_downtime_guard_enabled
        ),
        exchange_gap_recovery_bars=int(cfg.exchange_gap_recovery_bars),
        exchange_downtime_blocked_orders=int(
            data["exchange_downtime_blocked"].sum()
        ),
        exchange_available_rate=(
            float(data["execution_available"].mean()) if len(data) else 1.0
        ),
        liquidity_execution_enabled=bool(cfg.liquidity_execution_enabled),
        max_bar_participation_rate=float(cfg.max_bar_participation_rate),
        liquidity_limited_orders=int(data["liquidity_limited"].sum()),
        average_liquidity_fill_ratio=(
            float(
                data.loc[
                    data["requested_notional_change"].abs() > 1e-15,
                    "liquidity_fill_ratio",
                ].mean()
            )
            if bool((data["requested_notional_change"].abs() > 1e-15).any())
            else 1.0
        ),
        average_market_participation_rate=(
            float(
                data.loc[
                    data["taker_notional_turnover"] > 1e-15,
                    "market_participation_rate",
                ].mean()
            )
            if bool((data["taker_notional_turnover"] > 1e-15).any())
            else 0.0
        ),
        average_effective_slippage_rate=(
            float(
                data.loc[
                    data["taker_notional_turnover"] > 1e-15,
                    "effective_slippage_rate",
                ].mean()
            )
            if bool((data["taker_notional_turnover"] > 1e-15).any())
            else float(cfg.slippage_rate)
        ),
        max_effective_slippage_rate=(
            float(data["effective_slippage_rate"].max())
            if len(data)
            else float(cfg.slippage_rate)
        ),
        maker_turnover=float(data["maker_notional_turnover"].sum()),
        taker_turnover=float(data["taker_notional_turnover"].sum()),
        execution_latency_bars=int(cfg.execution_latency_bars),
        effective_long_threshold=effective_long_threshold,
        effective_short_threshold=effective_short_threshold,
        effective_short_signal_threshold=effective_short_signal_threshold,
        effective_trade_signal_threshold=float(direction_trade_threshold),
        effective_direction_long_threshold=float(direction_long_threshold),
        effective_direction_short_threshold=float(direction_short_threshold),
        effective_direction_trade_threshold=float(direction_trade_threshold),
        cost_edge_pass_rate=float(data["cost_edge_pass"].mean()) if len(data) else 1.0,
        required_edge=float(np.mean(required_edge)) if len(required_edge) else 0.0,
        regime_gate_pass_rate=float(data["regime_gate_pass"].mean()) if len(data) else 1.0,
        platform_event_gate_enabled=bool(getattr(cfg, "platform_event_gate_enabled", False)),
        platform_event_gate_pass_rate=float(data["platform_event_gate_pass"].mean()) if len(data) else 1.0,
        drawdown_cooldown_enabled=bool(getattr(cfg, "drawdown_cooldown_enabled", False)),
        cooldown_bars_triggered=int((~data["drawdown_cooldown_gate_pass"].astype(bool)).sum()) if len(data) else 0,
        strategy_archetype_policy=str(getattr(cfg, "strategy_archetype_policy", "all") or "all"),
        strategy_archetype_policy_pass_rate=float(data["archetype_policy_gate_pass"].mean()) if len(data) else 1.0,
        volatility_regime_policy=str(getattr(cfg, "volatility_regime_policy", "all") or "all"),
        volatility_regime_policy_pass_rate=float(data["volatility_regime_policy_pass"].mean()) if len(data) else 1.0,
        regime_detection_method_requested=str(
            getattr(cfg, "regime_detection_method", "rule_based")
            or "rule_based"
        ),
        regime_method_counts=regime_method_counts,
        regime_model_version_counts=regime_model_version_counts,
        regime_fallback_reason_counts=regime_fallback_reason_counts,
        regime_override_reason_counts=regime_override_reason_counts,
        market_structure_gate_enabled=bool(getattr(cfg, "market_structure_gate_enabled", False)),
        market_structure_gate_pass_rate=float(data["platform_strategy_quality_gate_pass"].mean()) if len(data) else 1.0,
        crowding_filter_enabled=bool(getattr(cfg, "crowding_filter_enabled", False)),
        crowding_filter_pass_rate=float(data["platform_strategy_quality_gate_pass"].mean()) if len(data) else 1.0,
        funding_crowding_guard_enabled=bool(
            getattr(cfg, "funding_crowding_guard_enabled", False)
        ),
        funding_crowding_gate_pass_rate=float(
            data["funding_crowding_gate_pass"].mean()
        )
        if len(data)
        else 1.0,
        funding_crowding_blocked_rows=int(
            (~data["funding_crowding_gate_pass"]).sum()
        )
        if len(data)
        else 0,
        regime_risk_guard_enabled=bool(
            getattr(cfg, "regime_risk_guard_enabled", False)
        ),
        regime_risk_gate_pass_rate=float(
            data["regime_risk_gate_pass"].mean()
        )
        if len(data)
        else 1.0,
        regime_risk_blocked_rows=int(
            (~data["regime_risk_gate_pass"]).sum()
        )
        if len(data)
        else 0,
        max_notional_exposure=float(getattr(cfg, "max_notional_exposure", 0.0) or 0.0),
        tradeability_gate_pass_rate=float(data["tradeability_gate_pass"].mean()) if len(data) else 1.0,
        horizon_gate_pass_rate=float(data["horizon_gate_pass"].mean()) if len(data) else 1.0,
        horizon_assisted_trades=int(((data["turnover"] > 0) & data["horizon_assisted_signal"]).sum()) if len(data) else 0,
        horizon_confirmation_enabled=bool(has_horizon_signal and (cfg.horizon_confirmation_enabled or cfg.horizon_assist_enabled)),
        long_trades=int(long_stats["trades"]),
        short_trades=int(short_stats["trades"]),
        long_total_return=float(long_stats["total_return"]),
        short_total_return=float(short_stats["total_return"]),
        long_profit_factor=float(long_stats["profit_factor"]),
        short_profit_factor=float(short_stats["profit_factor"]),
        long_win_rate=float(long_stats["win_rate"]),
        short_win_rate=float(short_stats["win_rate"]),
        long_commission_drag=float(long_stats["commission_drag"]),
        short_commission_drag=float(short_stats["commission_drag"]),
        long_slippage_drag=float(long_stats["slippage_drag"]),
        short_slippage_drag=float(short_stats["slippage_drag"]),
        long_fee_drag=float(long_stats["fee_drag"]),
        short_fee_drag=float(short_stats["fee_drag"]),
        atr_exit_enabled=atr_exit_enabled,
        trade_side_policy=str(getattr(cfg, "trade_side_policy", "both") or "both"),
        strategy_archetype_summary=archetype_summary,
        regime_summary=regime_summary,
        performance_by_year=performance.performance_by_year,
        performance_by_month=performance.performance_by_month,
        performance_by_symbol=performance.performance_by_symbol,
    )
    return result, data


def save_backtest_report(
    result: BacktestResult,
    detail: pd.DataFrame,
    output_dir: str | Path,
    name: str,
) -> BacktestResult:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    detail_path = base / f"{name}_backtest.csv"
    summary_path = base / f"{name}_backtest_summary.json"
    detail.to_csv(detail_path, index=False)
    payload = asdict(result)
    payload["detail_path"] = str(detail_path)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    result.report_path = str(summary_path)
    return result
