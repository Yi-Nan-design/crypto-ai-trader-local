from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .exchange_rules import normalize_order_notional_usdt


class LiquidityExecutionConfig(Protocol):
    """Configuration required by the liquidity-aware execution model."""

    initial_balance: float
    partial_fill_ratio: float
    slippage_rate: float
    liquidity_execution_enabled: bool
    max_bar_participation_rate: float
    liquidity_lookback_bars: int
    slippage_impact_coefficient: float
    max_dynamic_slippage_rate: float
    fee_rate: float
    maker_fee_rate: float
    maker_fill_fraction: float
    exchange_min_notional_usdt: float
    exchange_min_quantity: float
    exchange_max_quantity: float
    exchange_quantity_step: float
    exchange_price_tick_size: float


@dataclass(frozen=True)
class LiquidityFill:
    """Fill result after deterministic and market-capacity constraints."""

    executed_notional_usdt: float
    fill_ratio: float
    liquidity_fill_ratio: float
    capacity_usdt: float
    liquidity_limited: bool


@dataclass(frozen=True)
class LiquidityProfile:
    """Causal per-bar liquidity inputs used by fills and slippage."""

    quote_volume_usdt: np.ndarray
    trailing_quote_volume_usdt: np.ndarray
    range_proxy: np.ndarray
    capacity_usdt: np.ndarray


@dataclass(frozen=True)
class SlippageEstimate:
    """Per-bar effective slippage rates and their causal drivers."""

    effective_rate: np.ndarray
    market_participation_rate: np.ndarray
    liquidity_stress: np.ndarray
    range_proxy: np.ndarray


@dataclass(frozen=True)
class SingleOrderExecution:
    """Shared scalar order estimate for paper and event-driven execution."""

    accepted: bool
    filled: bool
    requested_notional_usdt: float
    normalized_notional_usdt: float
    executed_notional_usdt: float
    normalized_quantity: float
    fill_ratio: float
    liquidity_fill_ratio: float
    liquidity_capacity_usdt: float
    liquidity_limited: bool
    maker_notional_usdt: float
    taker_notional_usdt: float
    commission_usdt: float
    slippage_usdt: float
    effective_slippage_rate: float
    market_participation_rate: float
    liquidity_stress: float
    quantity_rounding_loss_usdt: float
    reason: str
    maximum_quantity_limited: bool = False


def liquidity_execution_enabled(cfg: LiquidityExecutionConfig) -> bool:
    return bool(getattr(cfg, "liquidity_execution_enabled", False))


def _numeric_series(
    data: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
) -> pd.Series:
    if column not in data.columns:
        return pd.Series(default, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def causal_liquidity_profile(
    data: pd.DataFrame,
    cfg: LiquidityExecutionConfig,
) -> LiquidityProfile:
    """Build liquidity inputs using the current closed bar and prior baselines."""

    quote_volume = _numeric_series(data, "quote_volume").fillna(0.0).clip(lower=0.0)
    lookback = max(int(getattr(cfg, "liquidity_lookback_bars", 48)), 1)
    trailing = (
        quote_volume.shift(1)
        .rolling(lookback, min_periods=1)
        .median()
        .fillna(quote_volume)
        .fillna(0.0)
    )
    if {"high", "low", "close"}.issubset(data.columns):
        close = _numeric_series(data, "close").replace(0.0, np.nan)
        range_proxy = (
            (_numeric_series(data, "high") - _numeric_series(data, "low"))
            .abs()
            .div(close.abs())
        )
    elif "high_low_range" in data.columns:
        range_proxy = _numeric_series(data, "high_low_range")
    elif "atr_14" in data.columns:
        range_proxy = _numeric_series(data, "atr_14")
    else:
        range_proxy = pd.Series(0.0, index=data.index, dtype=float)
    range_proxy = range_proxy.fillna(0.0).clip(lower=0.0, upper=0.25)
    if liquidity_execution_enabled(cfg):
        capacity = quote_volume * max(float(cfg.max_bar_participation_rate), 0.0)
    else:
        capacity = pd.Series(0.0, index=data.index, dtype=float)
    return LiquidityProfile(
        quote_volume_usdt=quote_volume.to_numpy(dtype=float),
        trailing_quote_volume_usdt=trailing.to_numpy(dtype=float),
        range_proxy=range_proxy.to_numpy(dtype=float),
        capacity_usdt=capacity.to_numpy(dtype=float),
    )


def estimate_liquidity_fill(
    requested_notional_usdt: float,
    quote_volume_usdt: float,
    cfg: LiquidityExecutionConfig,
    *,
    capacity_override_usdt: float | None = None,
) -> LiquidityFill:
    """Apply configured partial fills and a per-bar market participation cap."""

    requested = float(requested_notional_usdt)
    requested_abs = abs(requested)
    if requested_abs <= 1e-15:
        return LiquidityFill(0.0, 1.0, 1.0, 0.0, False)
    deterministic_ratio = float(np.clip(cfg.partial_fill_ratio, 0.0, 1.0))
    if not liquidity_execution_enabled(cfg):
        executed = requested * deterministic_ratio
        return LiquidityFill(
            executed_notional_usdt=executed,
            fill_ratio=abs(executed) / requested_abs,
            liquidity_fill_ratio=1.0,
            capacity_usdt=0.0,
            liquidity_limited=False,
        )

    quote_volume = max(float(quote_volume_usdt), 0.0)
    configured_capacity = quote_volume * max(
        float(cfg.max_bar_participation_rate),
        0.0,
    )
    capacity = (
        min(configured_capacity, max(float(capacity_override_usdt), 0.0))
        if capacity_override_usdt is not None
        else configured_capacity
    )
    liquidity_ratio = float(np.clip(capacity / requested_abs, 0.0, 1.0))
    combined_ratio = deterministic_ratio * liquidity_ratio
    executed = requested * combined_ratio
    return LiquidityFill(
        executed_notional_usdt=executed,
        fill_ratio=abs(executed) / requested_abs,
        liquidity_fill_ratio=liquidity_ratio,
        capacity_usdt=capacity,
        liquidity_limited=liquidity_ratio < 1.0 - 1e-12,
    )


def dynamic_slippage_rate_scalar(
    taker_notional_usdt: float,
    quote_volume_usdt: float,
    range_proxy: float,
    trailing_quote_volume_usdt: float,
    cfg: LiquidityExecutionConfig,
) -> tuple[float, float, float]:
    """Estimate slippage from participation, volatility, and liquidity stress."""

    base = max(float(cfg.slippage_rate), 0.0)
    if not liquidity_execution_enabled(cfg):
        return base, 0.0, 1.0
    quote_volume = max(float(quote_volume_usdt), 0.0)
    participation = (
        abs(float(taker_notional_usdt)) / quote_volume
        if quote_volume > 1e-12
        else (1.0 if abs(float(taker_notional_usdt)) > 1e-12 else 0.0)
    )
    trailing = max(float(trailing_quote_volume_usdt), 0.0)
    stress = (
        float(np.sqrt(trailing / quote_volume))
        if quote_volume > 1e-12 and trailing > 0.0
        else 1.0
    )
    stress = float(np.clip(stress, 0.5, 3.0))
    observed_range = float(np.clip(range_proxy, 0.0, 0.25))
    impact = (
        max(float(cfg.slippage_impact_coefficient), 0.0)
        * float(np.sqrt(max(participation, 0.0)))
        * observed_range
        * stress
    )
    effective = float(
        np.clip(
            base + impact,
            base,
            max(float(cfg.max_dynamic_slippage_rate), base),
        )
    )
    return effective, float(participation), stress


def estimate_dynamic_slippage(
    data: pd.DataFrame,
    taker_turnover_fraction: np.ndarray,
    cfg: LiquidityExecutionConfig,
    *,
    profile: LiquidityProfile | None = None,
) -> SlippageEstimate:
    """Vectorize the causal dynamic-slippage estimate across a backtest."""

    liquidity = profile or causal_liquidity_profile(data, cfg)
    turnover = np.asarray(taker_turnover_fraction, dtype=float)
    if len(turnover) != len(data):
        raise ValueError("taker_turnover_fraction length must match data rows")
    rates = np.zeros(len(data), dtype=float)
    participation = np.zeros(len(data), dtype=float)
    stress = np.ones(len(data), dtype=float)
    for idx in range(len(data)):
        rates[idx], participation[idx], stress[idx] = dynamic_slippage_rate_scalar(
            turnover[idx] * float(cfg.initial_balance),
            liquidity.quote_volume_usdt[idx],
            liquidity.range_proxy[idx],
            liquidity.trailing_quote_volume_usdt[idx],
            cfg,
        )
    return SlippageEstimate(
        effective_rate=rates,
        market_participation_rate=participation,
        liquidity_stress=stress,
        range_proxy=liquidity.range_proxy,
    )


def estimate_single_order_execution(
    requested_notional_usdt: float,
    *,
    price: float,
    quote_volume_usdt: float,
    range_proxy: float,
    trailing_quote_volume_usdt: float,
    cfg: LiquidityExecutionConfig,
    capacity_override_usdt: float | None = None,
) -> SingleOrderExecution:
    """Estimate one order with the same filters, fills, fees, and slippage."""

    normalized = normalize_order_notional_usdt(
        requested_notional_usdt,
        price,
        cfg,
    )
    if not normalized.accepted:
        return SingleOrderExecution(
            accepted=False,
            filled=False,
            requested_notional_usdt=float(requested_notional_usdt),
            normalized_notional_usdt=0.0,
            executed_notional_usdt=0.0,
            normalized_quantity=normalized.normalized_quantity,
            fill_ratio=0.0,
            liquidity_fill_ratio=0.0,
            liquidity_capacity_usdt=0.0,
            liquidity_limited=False,
            maker_notional_usdt=0.0,
            taker_notional_usdt=0.0,
            commission_usdt=0.0,
            slippage_usdt=0.0,
            effective_slippage_rate=max(float(cfg.slippage_rate), 0.0),
            market_participation_rate=0.0,
            liquidity_stress=1.0,
            quantity_rounding_loss_usdt=normalized.rounding_loss_usdt,
            reason=normalized.reason,
            maximum_quantity_limited=normalized.maximum_quantity_limited,
        )
    fill = estimate_liquidity_fill(
        normalized.normalized_notional_usdt,
        quote_volume_usdt,
        cfg,
        capacity_override_usdt=capacity_override_usdt,
    )
    executed_abs = abs(fill.executed_notional_usdt)
    maker_fraction = float(np.clip(cfg.maker_fill_fraction, 0.0, 1.0))
    maker_notional = executed_abs * maker_fraction
    taker_notional = executed_abs - maker_notional
    commission = (
        maker_notional * max(float(cfg.maker_fee_rate), 0.0)
        + taker_notional * max(float(cfg.fee_rate), 0.0)
    )
    slippage_rate, participation, stress = dynamic_slippage_rate_scalar(
        taker_notional,
        quote_volume_usdt,
        range_proxy,
        trailing_quote_volume_usdt,
        cfg,
    )
    slippage = taker_notional * slippage_rate
    filled = executed_abs > 1e-12
    reason = (
        "liquidity_no_fill"
        if not filled
        else ("liquidity_partial_fill" if fill.liquidity_limited else normalized.reason)
    )
    return SingleOrderExecution(
        accepted=True,
        filled=filled,
        requested_notional_usdt=float(requested_notional_usdt),
        normalized_notional_usdt=normalized.normalized_notional_usdt,
        executed_notional_usdt=fill.executed_notional_usdt,
        normalized_quantity=normalized.normalized_quantity,
        fill_ratio=fill.fill_ratio,
        liquidity_fill_ratio=fill.liquidity_fill_ratio,
        liquidity_capacity_usdt=fill.capacity_usdt,
        liquidity_limited=fill.liquidity_limited,
        maker_notional_usdt=maker_notional,
        taker_notional_usdt=taker_notional,
        commission_usdt=commission,
        slippage_usdt=slippage,
        effective_slippage_rate=slippage_rate,
        market_participation_rate=participation,
        liquidity_stress=stress,
        quantity_rounding_loss_usdt=normalized.rounding_loss_usdt,
        reason=reason,
        maximum_quantity_limited=normalized.maximum_quantity_limited,
    )
