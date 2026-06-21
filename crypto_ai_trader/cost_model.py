from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .exchange_availability import resolve_exchange_availability
from .exchange_rules import exchange_rules_enabled, normalize_order_notional_usdt
from .liquidity_execution import (
    causal_liquidity_profile,
    estimate_dynamic_slippage,
    estimate_liquidity_fill,
    liquidity_execution_enabled,
)


class CostConfig(Protocol):
    """Configuration fields required by the vectorized execution cost model."""

    fee_rate: float
    maker_fee_rate: float
    maker_fill_fraction: float
    slippage_rate: float
    funding_rate_buffer: float
    partial_fill_ratio: float
    execution_latency_bars: int
    min_order_notional_fraction: float
    initial_balance: float
    exchange_min_notional_usdt: float
    exchange_min_quantity: float
    exchange_max_quantity: float
    exchange_quantity_step: float
    exchange_price_tick_size: float
    exchange_downtime_guard_enabled: bool
    exchange_gap_recovery_bars: int
    liquidity_execution_enabled: bool
    max_bar_participation_rate: float
    liquidity_lookback_bars: int
    slippage_impact_coefficient: float
    max_dynamic_slippage_rate: float


@dataclass(frozen=True)
class ExecutionPath:
    """Target-to-fill simulation for a normalized notional position path."""

    delayed_target_notional: np.ndarray
    requested_notional_change: np.ndarray
    position_before_execution: np.ndarray
    executed_notional_position: np.ndarray
    executed_notional_change: np.ndarray
    fill_ratio: np.ndarray
    minimum_order_rejected: np.ndarray
    exchange_filter_rejected: np.ndarray
    exchange_order_quantity: np.ndarray
    exchange_order_notional_usdt: np.ndarray
    quantity_rounding_loss_usdt: np.ndarray
    maximum_quantity_limited: np.ndarray
    exchange_rejection_reason: np.ndarray
    execution_available: np.ndarray
    exchange_downtime_blocked: np.ndarray
    exchange_downtime_reason: np.ndarray
    liquidity_fill_ratio: np.ndarray
    liquidity_capacity_usdt: np.ndarray
    liquidity_limited: np.ndarray
    forced_flat_notional: np.ndarray
    forced_flat_turnover: np.ndarray


@dataclass(frozen=True)
class ExecutionCostBreakdown:
    """Per-bar execution costs expressed as fractions of account equity."""

    notional_turnover: np.ndarray
    maker_notional_turnover: np.ndarray
    taker_notional_turnover: np.ndarray
    commission_cost: np.ndarray
    slippage_cost: np.ndarray
    effective_slippage_rate: np.ndarray
    market_participation_rate: np.ndarray
    liquidity_stress: np.ndarray
    funding_cost: np.ndarray
    funding_rate_used: np.ndarray
    funding_rate_source: str
    funding_settlement_mode: str
    funding_period_fraction: float
    execution_path: ExecutionPath

    @property
    def trade_cost(self) -> np.ndarray:
        """Legacy combined commission and slippage cost."""

        return self.commission_cost + self.slippage_cost

    @property
    def total_cost(self) -> np.ndarray:
        return self.trade_cost + self.funding_cost


def infer_bar_hours(data: pd.DataFrame) -> float:
    """Infer the median bar duration without using future market values."""

    if "open_time" in data.columns and len(data) > 1:
        values = (
            pd.to_numeric(data["open_time"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) > 1:
            diffs = np.diff(values)
            diffs = diffs[diffs > 0]
            if len(diffs):
                return float(np.median(diffs) / 3_600_000.0)
    if "open_datetime" in data.columns and len(data) > 1:
        times = pd.to_datetime(
            data["open_datetime"], errors="coerce", utc=True
        ).dropna()
        if len(times) > 1:
            diffs = times.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
            diffs = diffs[diffs > 0]
            if len(diffs):
                return float(np.median(diffs) / 3600.0)
    return 1.0


def resolve_funding_rates(
    data: pd.DataFrame,
    cfg: CostConfig,
) -> tuple[np.ndarray, str, str]:
    """Resolve exact payments, a continuous rate series, or the legacy buffer."""

    if "funding_payment_rate" in data.columns:
        payment = pd.to_numeric(data["funding_payment_rate"], errors="coerce")
        payment = payment.replace([np.inf, -np.inf], np.nan)
        if bool(payment.notna().any()):
            values = payment.fillna(0.0).to_numpy(dtype=float)
            return values, "historical_funding_payment", "event"
    if "funding_rate_8h" in data.columns:
        raw = pd.to_numeric(data["funding_rate_8h"], errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan)
        if bool(raw.notna().any()):
            values = raw.ffill().fillna(0.0).to_numpy(dtype=float)
            return values, "historical_funding_rate_8h", "prorated"
    return (
        np.full(len(data), max(float(cfg.funding_rate_buffer), 0.0), dtype=float),
        "configured_absolute_buffer",
        "prorated",
    )


def estimate_execution_costs(
    data: pd.DataFrame,
    notional_position: np.ndarray,
    cfg: CostConfig,
    *,
    forced_flat_after: np.ndarray | None = None,
) -> ExecutionCostBreakdown:
    """Estimate commission, slippage, and funding costs for a position path."""

    target_notional = np.asarray(notional_position, dtype=float)
    if len(target_notional) != len(data):
        raise ValueError("notional_position length must match data rows")
    liquidity_profile = causal_liquidity_profile(data, cfg)
    exchange_availability = resolve_exchange_availability(data, cfg)
    path = simulate_execution_path(
        target_notional,
        cfg,
        prices=pd.to_numeric(data["close"], errors="coerce").to_numpy(dtype=float)
        if "close" in data.columns
        else None,
        quote_volumes=liquidity_profile.quote_volume_usdt,
        execution_available=exchange_availability.available,
        exchange_unavailable_reason=exchange_availability.blocked_reason,
        forced_flat_after=forced_flat_after,
    )
    normal_turnover = np.abs(path.executed_notional_change)
    turnover = normal_turnover + path.forced_flat_turnover
    maker_fraction = float(np.clip(cfg.maker_fill_fraction, 0.0, 1.0))
    maker_turnover = normal_turnover * maker_fraction
    taker_turnover = normal_turnover - maker_turnover + path.forced_flat_turnover
    commission = (
        maker_turnover * max(float(cfg.maker_fee_rate), 0.0)
        + taker_turnover * max(float(cfg.fee_rate), 0.0)
    )
    slippage_estimate = estimate_dynamic_slippage(
        data,
        taker_turnover,
        cfg,
        profile=liquidity_profile,
    )
    slippage = taker_turnover * slippage_estimate.effective_rate
    funding_period_fraction = max(0.0, infer_bar_hours(data) / 8.0)
    funding_rate, funding_source, funding_mode = resolve_funding_rates(data, cfg)
    if funding_mode == "event":
        funding = path.position_before_execution * funding_rate
    elif funding_source == "historical_funding_rate_8h":
        funding = (
            path.executed_notional_position
            * funding_rate
            * funding_period_fraction
        )
    else:
        funding = (
            np.abs(path.executed_notional_position)
            * funding_rate
            * funding_period_fraction
        )
    return ExecutionCostBreakdown(
        notional_turnover=turnover,
        maker_notional_turnover=maker_turnover,
        taker_notional_turnover=taker_turnover,
        commission_cost=commission,
        slippage_cost=slippage,
        effective_slippage_rate=slippage_estimate.effective_rate,
        market_participation_rate=slippage_estimate.market_participation_rate,
        liquidity_stress=slippage_estimate.liquidity_stress,
        funding_cost=funding,
        funding_rate_used=funding_rate,
        funding_rate_source=funding_source,
        funding_settlement_mode=funding_mode,
        funding_period_fraction=funding_period_fraction,
        execution_path=path,
    )


def simulate_execution_path(
    target_notional_position: np.ndarray,
    cfg: CostConfig,
    *,
    prices: np.ndarray | None = None,
    quote_volumes: np.ndarray | None = None,
    execution_available: np.ndarray | None = None,
    exchange_unavailable_reason: np.ndarray | None = None,
    forced_flat_after: np.ndarray | None = None,
) -> ExecutionPath:
    """Apply latency, exchange filters, and deterministic partial fills."""

    target = np.asarray(target_notional_position, dtype=float)
    rows = len(target)
    latency = max(int(cfg.execution_latency_bars), 0)
    delayed = np.zeros(rows, dtype=float)
    if latency == 0:
        delayed = target.copy()
    elif latency < rows:
        delayed[latency:] = target[:-latency]

    minimum_order = max(float(cfg.min_order_notional_fraction), 0.0)
    requested = np.zeros(rows, dtype=float)
    position_before = np.zeros(rows, dtype=float)
    executed_change = np.zeros(rows, dtype=float)
    executed_position = np.zeros(rows, dtype=float)
    fill_ratio = np.ones(rows, dtype=float)
    rejected = np.zeros(rows, dtype=bool)
    exchange_rejected = np.zeros(rows, dtype=bool)
    exchange_quantity = np.zeros(rows, dtype=float)
    exchange_notional = np.zeros(rows, dtype=float)
    rounding_loss = np.zeros(rows, dtype=float)
    maximum_quantity_limited = np.zeros(rows, dtype=bool)
    rejection_reason = np.full(rows, "", dtype=object)
    availability = (
        np.ones(rows, dtype=bool)
        if execution_available is None
        else np.asarray(execution_available, dtype=bool)
    )
    unavailable_reason = (
        np.full(rows, "", dtype=object)
        if exchange_unavailable_reason is None
        else np.asarray(exchange_unavailable_reason, dtype=object)
    )
    downtime_blocked = np.zeros(rows, dtype=bool)
    liquidity_fill_ratio = np.ones(rows, dtype=float)
    liquidity_capacity = np.zeros(rows, dtype=float)
    liquidity_limited = np.zeros(rows, dtype=bool)
    forced_flat_notional = np.zeros(rows, dtype=float)
    forced_flat_turnover = np.zeros(rows, dtype=float)
    forced_flat = (
        np.zeros(rows, dtype=bool)
        if forced_flat_after is None
        else np.asarray(forced_flat_after, dtype=bool)
    )
    if len(forced_flat) != rows:
        raise ValueError("forced_flat_after length must match target rows")
    if len(availability) != rows:
        raise ValueError("execution_available length must match target rows")
    if len(unavailable_reason) != rows:
        raise ValueError("exchange_unavailable_reason length must match target rows")
    price_values = (
        np.full(rows, np.nan, dtype=float)
        if prices is None
        else np.asarray(prices, dtype=float)
    )
    if len(price_values) != rows:
        raise ValueError("prices length must match target rows")
    if exchange_rules_enabled(cfg) and prices is None:
        raise ValueError("prices are required when exchange order rules are enabled")
    quote_volume_values = (
        np.full(rows, np.nan, dtype=float)
        if quote_volumes is None
        else np.asarray(quote_volumes, dtype=float)
    )
    if len(quote_volume_values) != rows:
        raise ValueError("quote_volumes length must match target rows")
    if liquidity_execution_enabled(cfg) and quote_volumes is None:
        raise ValueError("quote_volumes are required when liquidity execution is enabled")
    previous = 0.0
    for idx, target_value in enumerate(delayed):
        position_before[idx] = previous
        change = float(target_value - previous)
        requested[idx] = change
        if abs(change) <= 1e-15:
            executed_position[idx] = previous
            if forced_flat[idx] and abs(previous) > 1e-15:
                forced_flat_notional[idx] = -previous
                forced_flat_turnover[idx] = abs(previous)
                previous = 0.0
            continue
        if not availability[idx]:
            fill_ratio[idx] = 0.0
            downtime_blocked[idx] = True
            rejection_reason[idx] = str(
                unavailable_reason[idx] or "exchange_unavailable"
            )
            executed_position[idx] = previous
            if forced_flat[idx] and abs(previous) > 1e-15:
                forced_flat_notional[idx] = -previous
                forced_flat_turnover[idx] = abs(previous)
                previous = 0.0
            continue
        if abs(change) < minimum_order:
            rejected[idx] = True
            rejection_reason[idx] = "below_min_order_notional_fraction"
            fill_ratio[idx] = 0.0
            executed_position[idx] = previous
            if forced_flat[idx] and abs(previous) > 1e-15:
                forced_flat_notional[idx] = -previous
                forced_flat_turnover[idx] = abs(previous)
                previous = 0.0
            continue
        normalized_change = change
        if exchange_rules_enabled(cfg):
            normalized = normalize_order_notional_usdt(
                change * float(cfg.initial_balance),
                price_values[idx],
                cfg,
            )
            exchange_quantity[idx] = normalized.normalized_quantity
            exchange_notional[idx] = abs(normalized.normalized_notional_usdt)
            rounding_loss[idx] = normalized.rounding_loss_usdt
            maximum_quantity_limited[idx] = normalized.maximum_quantity_limited
            rejection_reason[idx] = normalized.reason
            if not normalized.accepted:
                rejected[idx] = True
                exchange_rejected[idx] = True
                fill_ratio[idx] = 0.0
                executed_position[idx] = previous
                if forced_flat[idx] and abs(previous) > 1e-15:
                    forced_flat_notional[idx] = -previous
                    forced_flat_turnover[idx] = abs(previous)
                    previous = 0.0
                continue
            normalized_change = (
                normalized.normalized_notional_usdt / float(cfg.initial_balance)
            )
        fill = estimate_liquidity_fill(
            normalized_change * float(cfg.initial_balance),
            quote_volume_values[idx],
            cfg,
        )
        filled = fill.executed_notional_usdt / float(cfg.initial_balance)
        executed_change[idx] = filled
        fill_ratio[idx] = abs(filled) / abs(change)
        liquidity_fill_ratio[idx] = fill.liquidity_fill_ratio
        liquidity_capacity[idx] = fill.capacity_usdt
        liquidity_limited[idx] = fill.liquidity_limited
        previous += filled
        executed_position[idx] = previous
        if forced_flat[idx] and abs(previous) > 1e-15:
            forced_flat_notional[idx] = -previous
            forced_flat_turnover[idx] = abs(previous)
            previous = 0.0
    return ExecutionPath(
        delayed_target_notional=delayed,
        requested_notional_change=requested,
        position_before_execution=position_before,
        executed_notional_position=executed_position,
        executed_notional_change=executed_change,
        fill_ratio=fill_ratio,
        minimum_order_rejected=rejected,
        exchange_filter_rejected=exchange_rejected,
        exchange_order_quantity=exchange_quantity,
        exchange_order_notional_usdt=exchange_notional,
        quantity_rounding_loss_usdt=rounding_loss,
        maximum_quantity_limited=maximum_quantity_limited,
        exchange_rejection_reason=rejection_reason,
        execution_available=availability,
        exchange_downtime_blocked=downtime_blocked,
        exchange_downtime_reason=unavailable_reason,
        liquidity_fill_ratio=liquidity_fill_ratio,
        liquidity_capacity_usdt=liquidity_capacity,
        liquidity_limited=liquidity_limited,
        forced_flat_notional=forced_flat_notional,
        forced_flat_turnover=forced_flat_turnover,
    )
