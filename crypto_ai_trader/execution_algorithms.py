from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np


class OrderStyle(StrEnum):
    """Supported simulated order styles."""

    LIMIT = "limit"
    MARKET = "market"


@dataclass(frozen=True)
class OrderSlice:
    """One simulated child order; it is not an exchange instruction."""

    sequence: int
    signed_notional_usdt: float
    style: OrderStyle
    participation_rate: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["style"] = self.style.value
        return payload


@dataclass(frozen=True)
class ExecutionPlan:
    """Deterministic order plan generated from a target notional change."""

    algorithm: str
    target_signed_notional_usdt: float
    slices: tuple[OrderSlice, ...]
    estimated_completion_bars: int
    live_orders_allowed: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["slices"] = [item.to_dict() for item in self.slices]
        return payload


class SlippagePredictor(Protocol):
    """Optional learned slippage interface for offline execution research."""

    name: str

    def predict_rate(self, features: np.ndarray) -> np.ndarray: ...


def twap_plan(
    signed_notional_usdt: float,
    *,
    bars: int,
    participation_rate: float,
    style: OrderStyle = OrderStyle.LIMIT,
) -> ExecutionPlan:
    """Split target notional evenly across a bounded number of bars."""

    count = max(int(bars), 1)
    per_slice = float(signed_notional_usdt) / count
    slices = tuple(
        OrderSlice(
            sequence=index + 1,
            signed_notional_usdt=per_slice,
            style=style,
            participation_rate=float(np.clip(participation_rate, 0.0, 1.0)),
            reason="twap_equal_slice",
        )
        for index in range(count)
    )
    return ExecutionPlan(
        algorithm="twap",
        target_signed_notional_usdt=float(signed_notional_usdt),
        slices=slices,
        estimated_completion_bars=count,
    )


def vwap_plan(
    signed_notional_usdt: float,
    *,
    expected_volume_profile: list[float] | np.ndarray,
    max_participation_rate: float,
) -> ExecutionPlan:
    """Allocate child orders in proportion to a non-negative volume profile."""

    profile = np.asarray(expected_volume_profile, dtype=float).reshape(-1)
    if len(profile) == 0 or not bool(np.isfinite(profile).all()):
        raise ValueError("expected_volume_profile must be finite and non-empty")
    profile = np.clip(profile, 0.0, None)
    if float(profile.sum()) <= 0.0:
        raise ValueError("expected_volume_profile must contain positive volume")
    weights = profile / profile.sum()
    slices = tuple(
        OrderSlice(
            sequence=index + 1,
            signed_notional_usdt=float(signed_notional_usdt) * float(weight),
            style=OrderStyle.LIMIT,
            participation_rate=float(
                np.clip(max_participation_rate, 0.0, 1.0)
            ),
            reason="vwap_volume_weighted_slice",
        )
        for index, weight in enumerate(weights)
    )
    return ExecutionPlan(
        algorithm="vwap",
        target_signed_notional_usdt=float(signed_notional_usdt),
        slices=slices,
        estimated_completion_bars=len(slices),
    )


def limit_first_plan(
    signed_notional_usdt: float,
    *,
    urgency: float,
    maker_fraction: float,
    participation_rate: float,
) -> ExecutionPlan:
    """Prefer maker liquidity and reserve market fallback for urgent remainder."""

    urgency = float(np.clip(urgency, 0.0, 1.0))
    maker_fraction = float(np.clip(maker_fraction, 0.0, 1.0))
    if urgency >= 0.80:
        maker_fraction *= 0.25
    limit_notional = float(signed_notional_usdt) * maker_fraction
    market_notional = float(signed_notional_usdt) - limit_notional
    slices: list[OrderSlice] = []
    if abs(limit_notional) > 1e-12:
        slices.append(
            OrderSlice(
                sequence=1,
                signed_notional_usdt=limit_notional,
                style=OrderStyle.LIMIT,
                participation_rate=float(
                    np.clip(participation_rate, 0.0, 1.0)
                ),
                reason="limit_first_maker_attempt",
            )
        )
    if abs(market_notional) > 1e-12:
        slices.append(
            OrderSlice(
                sequence=len(slices) + 1,
                signed_notional_usdt=market_notional,
                style=OrderStyle.MARKET,
                participation_rate=float(
                    np.clip(participation_rate, 0.0, 1.0)
                ),
                reason="market_fallback_for_unfilled_or_urgent_remainder",
            )
        )
    return ExecutionPlan(
        algorithm="limit_first",
        target_signed_notional_usdt=float(signed_notional_usdt),
        slices=tuple(slices),
        estimated_completion_bars=max(len(slices), 1),
    )


def learned_slippage_availability() -> dict[str, object]:
    """Describe the optional predictor boundary without requiring a model."""

    return {
        "status": "interface_ready_not_trained",
        "model_kind": "lightgbm_slippage_regressor",
        "required_inputs": [
            "spread_proxy",
            "market_participation_rate",
            "liquidity_stress",
            "high_low_range",
        ],
        "live_orders_allowed": False,
    }
