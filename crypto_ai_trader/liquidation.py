from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class LiquidationConfig(Protocol):
    """Configuration required by the isolated-margin liquidation model."""

    leverage: int
    liquidation_guard_enabled: bool
    maintenance_margin_rate: float
    liquidation_buffer: float
    liquidation_fee_rate: float


@dataclass(frozen=True)
class LiquidationAssessment:
    """Per-bar liquidation outcomes for an executed notional position path."""

    triggered: np.ndarray
    gap_triggered: np.ndarray
    price_distance: np.ndarray
    price_return: np.ndarray
    fee_cost: np.ndarray
    reason: np.ndarray


def liquidation_price_distance(cfg: LiquidationConfig) -> float:
    """Return the adverse entry-price move that reaches liquidation."""

    leverage = max(float(cfg.leverage), 1.0)
    maintenance = max(float(cfg.maintenance_margin_rate), 0.0)
    buffer = max(float(cfg.liquidation_buffer), 0.0)
    return max(1.0 / leverage - maintenance - buffer, 1e-9)


def assess_liquidations(
    data: pd.DataFrame,
    executed_notional_position: np.ndarray,
    stop_loss: np.ndarray,
    cfg: LiquidationConfig,
) -> LiquidationAssessment:
    """Detect next-bar liquidation using open and intrabar adverse extremes.

    A protective stop closer than liquidation is assumed to execute first,
    except when the next bar opens beyond the liquidation threshold.
    """

    positions = np.asarray(executed_notional_position, dtype=float)
    stops = np.asarray(stop_loss, dtype=float)
    if len(positions) != len(data) or len(stops) != len(data):
        raise ValueError("liquidation inputs must match data rows")

    rows = len(data)
    triggered = np.zeros(rows, dtype=bool)
    gap_triggered = np.zeros(rows, dtype=bool)
    price_return = np.zeros(rows, dtype=float)
    fee_cost = np.zeros(rows, dtype=float)
    reason = np.full(rows, "not_liquidated", dtype=object)
    distance = np.full(rows, liquidation_price_distance(cfg), dtype=float)

    required = {"close", "open", "high", "low"}
    if (
        rows == 0
        or not bool(getattr(cfg, "liquidation_guard_enabled", True))
        or not required.issubset(data.columns)
    ):
        return LiquidationAssessment(
            triggered=triggered,
            gap_triggered=gap_triggered,
            price_distance=distance,
            price_return=price_return,
            fee_cost=fee_cost,
            reason=reason,
        )

    close = pd.to_numeric(data["close"], errors="coerce").replace(0, np.nan)
    next_open_return = (
        pd.to_numeric(data["open"], errors="coerce").shift(-1) / close - 1.0
    ).fillna(0.0).to_numpy(dtype=float)
    next_high_return = (
        pd.to_numeric(data["high"], errors="coerce").shift(-1) / close - 1.0
    ).fillna(0.0).to_numpy(dtype=float)
    next_low_return = (
        pd.to_numeric(data["low"], errors="coerce").shift(-1) / close - 1.0
    ).fillna(0.0).to_numpy(dtype=float)

    active_long = positions > 1e-15
    active_short = positions < -1e-15
    stop_outside_liquidation = stops + 1e-12 >= distance
    long_gap = active_long & (next_open_return <= -distance)
    short_gap = active_short & (next_open_return >= distance)
    long_intrabar = (
        active_long
        & stop_outside_liquidation
        & (next_low_return <= -distance)
    )
    short_intrabar = (
        active_short
        & stop_outside_liquidation
        & (next_high_return >= distance)
    )

    gap_triggered = long_gap | short_gap
    triggered = gap_triggered | long_intrabar | short_intrabar
    price_return = np.where(
        triggered & active_long,
        -distance,
        np.where(triggered & active_short, distance, 0.0),
    )
    liquidation_notional = np.abs(positions) * np.maximum(1.0 + price_return, 0.0)
    fee_cost = (
        liquidation_notional
        * max(float(getattr(cfg, "liquidation_fee_rate", 0.0)), 0.0)
        * triggered.astype(float)
    )
    reason[triggered] = "liquidation_intrabar_breach"
    reason[gap_triggered] = "liquidation_gap_breach"
    return LiquidationAssessment(
        triggered=triggered,
        gap_triggered=gap_triggered,
        price_distance=distance,
        price_return=price_return,
        fee_cost=fee_cost,
        reason=reason,
    )
