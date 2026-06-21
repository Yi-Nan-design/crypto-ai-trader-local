from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


SHADOW_LEARNING_CONTRACT_VERSION = "2026-06-21-v1"


def select_shadow_threshold_candidate(
    ranking: list[dict[str, Any]] | None,
    *,
    min_signal_count: int = 8,
    min_profit_factor: float = 1.20,
    min_total_return: float = 0.0,
) -> dict[str, Any] | None:
    """Select a validation-only threshold suitable for low-risk shadow use."""

    eligible: list[dict[str, Any]] = []
    for item in ranking or []:
        if not isinstance(item, dict):
            continue
        signal_count = int(item.get("signal_count", 0) or 0)
        profit_factor = float(
            item.get("signal_profit_factor_after_cost", 0.0) or 0.0
        )
        total_return = float(
            item.get("signal_total_return_after_cost", 0.0) or 0.0
        )
        expectancy = float(
            item.get("signal_expectancy_after_cost", 0.0) or 0.0
        )
        threshold = float(item.get("threshold", 0.0) or 0.0)
        if (
            signal_count < max(int(min_signal_count), 1)
            or profit_factor < float(min_profit_factor)
            or total_return <= float(min_total_return)
            or expectancy <= 0.0
            or not 0.5 <= threshold < 1.0
        ):
            continue
        eligible.append(
            {
                **item,
                "signal_count": signal_count,
                "signal_profit_factor_after_cost": profit_factor,
                "signal_total_return_after_cost": total_return,
                "signal_expectancy_after_cost": expectancy,
                "threshold": threshold,
            }
        )
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item["signal_total_return_after_cost"]),
            float(item["signal_profit_factor_after_cost"]),
            int(item["signal_count"]),
        ),
    )


def build_shadow_learning_decision(
    model_report: dict[str, Any],
    shadow_probabilities: dict[str, float | bool],
    *,
    execution_allowed: bool,
    regime_risk_off: bool,
    liquidity_score: float,
    min_liquidity_score: float,
    funding_rate: float,
    funding_crowding_limit: float,
    min_signal_count: int = 8,
    min_profit_factor: float = 1.20,
    max_position_fraction: float = 0.05,
    leverage: int = 1,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build a forward shadow-only decision from validation-qualified sides."""

    directional = (
        model_report.get("directional_signal_report", {}).get(
            "directions",
            {},
        )
        if isinstance(model_report, dict)
        else {}
    )
    candidates: list[dict[str, Any]] = []
    for side in ("long", "short"):
        side_report = directional.get(side, {})
        candidate = (
            side_report.get("shadow_candidate")
            if isinstance(side_report, dict)
            else None
        )
        model_available = bool(
            shadow_probabilities.get(f"{side}_model_available", False)
        )
        if not isinstance(candidate, dict) or not model_available:
            continue
        if (
            int(candidate.get("signal_count", 0) or 0)
            < max(int(min_signal_count), 1)
            or float(
                candidate.get(
                    "signal_profit_factor_after_cost",
                    0.0,
                )
                or 0.0
            )
            < float(min_profit_factor)
            or float(
                candidate.get(
                    "signal_total_return_after_cost",
                    0.0,
                )
                or 0.0
            )
            <= 0.0
            or float(
                candidate.get(
                    "signal_expectancy_after_cost",
                    0.0,
                )
                or 0.0
            )
            <= 0.0
        ):
            continue
        candidates.append(
            {
                **candidate,
                "side": side,
                "latest_probability": float(
                    shadow_probabilities.get(side, 0.0) or 0.0
                ),
                "probability_triggered": bool(
                    float(shadow_probabilities.get(side, 0.0) or 0.0)
                    >= float(candidate.get("threshold", 1.0) or 1.0)
                ),
            }
        )

    triggered_candidates = [
        item
        for item in candidates
        if bool(item.get("probability_triggered", False))
    ]
    selectable = triggered_candidates or candidates
    selected = (
        max(
            selectable,
            key=lambda item: (
                float(item.get("signal_total_return_after_cost", 0.0)),
                float(item.get("signal_profit_factor_after_cost", 0.0)),
                int(item.get("signal_count", 0)),
            ),
        )
        if selectable
        else None
    )
    blockers: list[str] = []
    if not enabled:
        blockers.append("shadow_learning_disabled")
    if selected is None:
        blockers.append("no_validation_qualified_shadow_side")

    latest_signal_active = False
    direction = 0
    if selected is not None:
        side = str(selected["side"])
        probability = float(selected["latest_probability"])
        threshold = float(selected.get("threshold", 1.0))
        if probability < threshold:
            blockers.append("shadow_probability_below_threshold")
        if not execution_allowed:
            blockers.append("shadow_exchange_unavailable")
        if regime_risk_off:
            blockers.append("shadow_regime_risk_off")
        if (
            not np.isfinite(float(liquidity_score))
            or float(liquidity_score) < float(min_liquidity_score)
        ):
            blockers.append("shadow_liquidity_below_minimum")
        if side == "long" and float(funding_rate) > abs(
            float(funding_crowding_limit)
        ):
            blockers.append("shadow_long_funding_crowded")
        if side == "short" and float(funding_rate) < -abs(
            float(funding_crowding_limit)
        ):
            blockers.append("shadow_short_funding_crowded")
        latest_signal_active = not blockers
        direction = 1 if latest_signal_active and side == "long" else (
            -1 if latest_signal_active and side == "short" else 0
        )

    return {
        "contract_version": SHADOW_LEARNING_CONTRACT_VERSION,
        "enabled": bool(enabled),
        "eligible": bool(selected is not None),
        "selected_side": (
            str(selected.get("side")) if selected is not None else ""
        ),
        "selected_model": (
            str(selected.get("model_name") or "")
            if selected is not None
            else ""
        ),
        "validation": selected or {},
        "latest_probability": (
            float(selected.get("latest_probability", 0.0))
            if selected is not None
            else 0.0
        ),
        "signal_threshold": (
            float(selected.get("threshold", 1.0))
            if selected is not None
            else 1.0
        ),
        "latest_signal_active": bool(latest_signal_active),
        "target_direction": int(direction),
        "target_exposure": (
            float(max_position_fraction) if latest_signal_active else 0.0
        ),
        "max_position_fraction": float(max_position_fraction),
        "leverage": int(leverage),
        "blockers": blockers,
        "mode": "shadow_paper_only",
        "strict_candidate_unchanged": True,
        "safety": {
            "live_trading_enabled": False,
            "real_orders_allowed": False,
            "api_keys_used": False,
        },
    }


def shadow_portfolio_report(report: dict[str, Any]) -> dict[str, Any]:
    """Convert a live report into a shadow-only portfolio planning input."""

    converted = deepcopy(report)
    shadow = converted.get("shadow_learning")
    shadow = shadow if isinstance(shadow, dict) else {}
    side = str(shadow.get("selected_side") or "")
    probability = float(shadow.get("latest_probability", 0.0) or 0.0)
    threshold = float(shadow.get("signal_threshold", 1.0) or 1.0)
    active = bool(shadow.get("latest_signal_active", False))
    max_position = float(
        shadow.get("max_position_fraction", 0.0) or 0.0
    )
    if side == "long":
        up_probability = probability
        long_threshold = threshold
        short_threshold = 0.0
        side_policy = "long_only"
    elif side == "short":
        up_probability = 1.0 - probability
        long_threshold = 1.0
        short_threshold = 1.0 - threshold
        side_policy = "short_only"
    else:
        up_probability = 0.5
        long_threshold = 1.0
        short_threshold = 0.0
        side_policy = "none"
    converted["latest_up_probability"] = float(up_probability)
    converted["latest_direction_probabilities"] = {
        "long": float(probability if side == "long" else 1.0 - probability),
        "short": float(probability if side == "short" else 1.0 - probability),
        "trade": 1.0 if active else 0.0,
        "up": float(up_probability),
    }
    alpha = dict(converted.get("latest_alpha_prediction") or {})
    alpha["confidence"] = float(
        np.clip(abs(probability - 0.5) * 2.0, 0.0, 1.0)
    )
    converted["latest_alpha_prediction"] = alpha
    converted["latest_risk_decision"] = {
        "allow_trade": active,
        "risk_level": "low" if active else "high",
        "max_position_size": max_position if active else 0.0,
        "reason": (
            "shadow_learning_signal_allowed"
            if active
            else str((shadow.get("blockers") or ["shadow_no_signal"])[0])
        ),
    }
    converted["optimized_backtest_config"] = {
        **dict(converted.get("optimized_backtest_config") or {}),
        "leverage": int(shadow.get("leverage", 1) or 1),
        "max_allowed_leverage": 1,
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
        "min_confidence_gap": 0.0,
        "trade_signal_threshold": 0.5,
        "trade_side_policy": side_policy,
        "max_position_fraction": max_position,
    }
    converted["candidate_mode"] = "shadow_learning"
    return converted
