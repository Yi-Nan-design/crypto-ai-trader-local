from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .progress import safe_replace_text


SHADOW_LEARNING_CONTRACT_VERSION = "2026-06-22-v2"
LOW_THRESHOLD_STRATEGY_VERSION = "2026-06-22-low-threshold-v1"
LOW_THRESHOLD_CANDIDATES = (0.54, 0.57, 0.60, 0.63, 0.66, 0.70)
LOW_THRESHOLD_RETURN_HORIZONS = (
    ("future_return_h1", 1),
    ("future_return_h2", 2),
    ("future_return_h3", 3),
    ("future_return_h6", 6),
    ("future_return_h12", 12),
)
SHADOW_HOLDING_SCHEMA_VERSION = 1
SHADOW_HARD_EXIT_BLOCKERS = {
    "shadow_learning_disabled",
    "no_validation_qualified_shadow_side",
    "shadow_exchange_unavailable",
    "shadow_regime_risk_off",
    "shadow_liquidity_below_minimum",
    "shadow_long_funding_crowded",
    "shadow_short_funding_crowded",
}


def _strategy_profile_specs(direction: str) -> dict[str, list[dict[str, Any]]]:
    side = "long" if direction == "long" else "short"
    return {
        "unfiltered": [],
        "platform_035_liquidity": [
            {
                "column": f"platform_strategy_{side}_score",
                "operator": ">=",
                "value": 0.35,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
        "platform_045_liquidity": [
            {
                "column": f"platform_strategy_{side}_score",
                "operator": ">=",
                "value": 0.45,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
        "trend_035_controlled_volatility": [
            {
                "column": f"trend_quality_{side}",
                "operator": ">=",
                "value": 0.35,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "trend_045_controlled_volatility": [
            {
                "column": f"trend_quality_{side}",
                "operator": ">=",
                "value": 0.45,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "range_035_controlled_volatility": [
            {
                "column": f"range_quality_{side}",
                "operator": ">=",
                "value": 0.35,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "range_045_controlled_volatility": [
            {
                "column": f"range_quality_{side}",
                "operator": ">=",
                "value": 0.45,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "volume_impulse_aligned_controlled_volatility": [
            {
                "column": "volume_price_impulse",
                "operator": "direction_aligned",
                "value": 0.0,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "volume_impulse_counter_controlled_volatility": [
            {
                "column": "volume_price_impulse",
                "operator": "direction_counter",
                "value": 0.0,
            },
            {
                "column": "event_volatility_budget",
                "operator": "<=",
                "value": 1.50,
            },
        ],
        "taker_pressure_aligned_liquidity": [
            {
                "column": "taker_pressure_3",
                "operator": "direction_aligned",
                "value": 0.0,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
        "taker_pressure_counter_liquidity": [
            {
                "column": "taker_pressure_3",
                "operator": "direction_counter",
                "value": 0.0,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
        "micro_trend_aligned_liquidity": [
            {
                "column": "micro_trend_regime",
                "operator": "direction_aligned",
                "value": 0.0,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
        "higher_timeframe_025_liquidity": [
            {
                "column": f"higher_tf_trend_alignment_{side}",
                "operator": ">=",
                "value": 0.25,
            },
            {
                "column": "liquidity_quality_score",
                "operator": ">=",
                "value": 0.35,
            },
        ],
    }


def shadow_strategy_filter_mask(
    frame: pd.DataFrame,
    direction: str,
    profile: str,
) -> np.ndarray:
    """Apply one saved causal strategy filter to feature rows."""

    specs = _strategy_profile_specs(direction)
    conditions = specs.get(str(profile))
    if conditions is None:
        return np.zeros(len(frame), dtype=bool)
    mask = np.ones(len(frame), dtype=bool)
    side_sign = 1.0 if direction == "long" else -1.0
    for condition in conditions:
        column = str(condition["column"])
        if column not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            .to_numpy(dtype=float)
        )
        finite = np.isfinite(values)
        operator = str(condition["operator"])
        threshold = float(condition["value"])
        if operator == ">=":
            passed = values >= threshold
        elif operator == "<=":
            passed = values <= threshold
        elif operator == "direction_aligned":
            passed = values * side_sign >= threshold
        elif operator == "direction_counter":
            passed = values * side_sign <= threshold
        else:
            return np.zeros(len(frame), dtype=bool)
        mask &= finite & passed
    return mask


def _signal_performance_after_cost(
    frame: pd.DataFrame,
    signal: np.ndarray,
    direction: str,
    cost_buffer: float,
    *,
    return_column: str = "future_return",
    hold_bars: int = 1,
) -> dict[str, float]:
    if return_column not in frame.columns:
        return {
            "signal_count": 0.0,
            "signal_rate": 0.0,
            "signal_total_return_after_cost": 0.0,
            "signal_expectancy_after_cost": 0.0,
            "signal_profit_factor_after_cost": 0.0,
            "signal_win_rate_after_cost": 0.0,
            "signal_cost_buffer": float(cost_buffer),
            "return_column": return_column,
            "hold_bars": float(max(int(hold_bars), 1)),
        }
    returns = (
        pd.to_numeric(frame[return_column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    raw_signal = np.asarray(signal, dtype=bool)
    independent_signal = np.zeros(len(raw_signal), dtype=bool)
    next_available = 0
    for index, active in enumerate(raw_signal):
        if active and index >= next_available:
            independent_signal[index] = True
            next_available = index + max(int(hold_bars), 1)
    signed = returns if direction == "long" else -returns
    net = signed[independent_signal] - float(cost_buffer)
    if not len(net):
        return {
            "signal_count": 0.0,
            "signal_rate": 0.0,
            "signal_total_return_after_cost": 0.0,
            "signal_expectancy_after_cost": 0.0,
            "signal_profit_factor_after_cost": 0.0,
            "signal_win_rate_after_cost": 0.0,
            "signal_cost_buffer": float(cost_buffer),
            "return_column": return_column,
            "hold_bars": float(max(int(hold_bars), 1)),
        }
    gross_profit = float(net[net > 0.0].sum())
    gross_loss = abs(float(net[net < 0.0].sum()))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else (999.0 if gross_profit > 0.0 else 0.0)
    )
    return {
        "signal_count": float(len(net)),
        "signal_rate": float(len(net) / max(len(frame), 1)),
        "signal_total_return_after_cost": float(net.sum()),
        "signal_expectancy_after_cost": float(net.mean()),
        "signal_profit_factor_after_cost": float(profit_factor),
        "signal_win_rate_after_cost": float((net > 0.0).mean()),
        "signal_cost_buffer": float(cost_buffer),
        "return_column": return_column,
        "hold_bars": float(max(int(hold_bars), 1)),
    }


def select_low_threshold_strategy_candidate(
    calibration_probability: np.ndarray,
    calibration_frame: pd.DataFrame,
    gate_probability: np.ndarray,
    gate_frame: pd.DataFrame,
    *,
    direction: str,
    model_name: str,
    cost_buffer: float,
    min_signal_count: int = 8,
    min_profit_factor: float = 1.05,
) -> dict[str, Any] | None:
    """Find a low raw threshold whose filtered signal survives a later gate."""

    if direction not in {"long", "short"}:
        return None
    calibration_probability = np.asarray(
        calibration_probability,
        dtype=float,
    )
    gate_probability = np.asarray(gate_probability, dtype=float)
    if (
        len(calibration_probability) != len(calibration_frame)
        or len(gate_probability) != len(gate_frame)
        or gate_frame.empty
    ):
        return None
    ranked: list[dict[str, Any]] = []
    profiles = _strategy_profile_specs(direction)
    horizons = [
        (column, bars)
        for column, bars in LOW_THRESHOLD_RETURN_HORIZONS
        if column in calibration_frame.columns
        and column in gate_frame.columns
    ]
    if not horizons and "future_return" in calibration_frame.columns:
        horizons = [("future_return", 1)]
    for return_column, hold_bars in horizons:
        for threshold in LOW_THRESHOLD_CANDIDATES:
            raw_calibration = calibration_probability >= float(threshold)
            raw_gate = gate_probability >= float(threshold)
            for profile, conditions in profiles.items():
                filtered_calibration = (
                    raw_calibration
                    & shadow_strategy_filter_mask(
                        calibration_frame,
                        direction,
                        profile,
                    )
                )
                calibration = _signal_performance_after_cost(
                    calibration_frame,
                    filtered_calibration,
                    direction,
                    cost_buffer,
                    return_column=return_column,
                    hold_bars=hold_bars,
                )
                if (
                    int(calibration["signal_count"])
                    < max(int(min_signal_count), 1)
                    or calibration["signal_total_return_after_cost"] <= 0.0
                    or calibration["signal_expectancy_after_cost"] <= 0.0
                    or calibration["signal_profit_factor_after_cost"]
                    < float(min_profit_factor)
                ):
                    continue
                filtered_gate = (
                    raw_gate
                    & shadow_strategy_filter_mask(
                        gate_frame,
                        direction,
                        profile,
                    )
                )
                gate = _signal_performance_after_cost(
                    gate_frame,
                    filtered_gate,
                    direction,
                    cost_buffer,
                    return_column=return_column,
                    hold_bars=hold_bars,
                )
                ranked.append(
                    {
                        "threshold": float(threshold),
                        "raw_threshold": float(threshold),
                        "strategy_profile": profile,
                        "strategy_conditions": conditions,
                        "strategy_version": LOW_THRESHOLD_STRATEGY_VERSION,
                        "return_column": return_column,
                        "forecast_horizon_bars": int(hold_bars),
                        "raw_calibration_signal_count": int(
                            raw_calibration.sum()
                        ),
                        "raw_gate_signal_count": int(raw_gate.sum()),
                        "calibration": calibration,
                        "gate": gate,
                        "model_name": str(model_name),
                        "direction": direction,
                        "selection_dataset": "validation_calibration",
                        "gate_dataset": "validation_gate",
                        "test_used_for_selection": False,
                        "score": float(
                            calibration[
                                "signal_total_return_after_cost"
                            ]
                            + 0.001
                            * min(
                                calibration[
                                    "signal_profit_factor_after_cost"
                                ],
                                10.0,
                            )
                        ),
                    }
                )
    ranked.sort(
        key=lambda item: (
            float(item["threshold"]),
            int(item.get("forecast_horizon_bars", 1)),
            -float(item["score"]),
        )
    )
    gate_min_signals = max(3, int(np.ceil(min_signal_count / 3.0)))
    for item in ranked:
        calibration = item["calibration"]
        gate = item["gate"]
        if (
            int(gate["signal_count"]) < gate_min_signals
            or gate["signal_total_return_after_cost"] <= 0.0
            or gate["signal_expectancy_after_cost"] <= 0.0
            or gate["signal_profit_factor_after_cost"] < 1.0
        ):
            continue
        return {
            **item,
            "signal_count": int(calibration["signal_count"]),
            "signal_rate": float(calibration["signal_rate"]),
            "signal_total_return_after_cost": float(
                min(
                    calibration["signal_total_return_after_cost"],
                    gate["signal_total_return_after_cost"],
                )
            ),
            "signal_expectancy_after_cost": float(
                min(
                    calibration["signal_expectancy_after_cost"],
                    gate["signal_expectancy_after_cost"],
                )
            ),
            "signal_profit_factor_after_cost": float(
                min(
                    calibration["signal_profit_factor_after_cost"],
                    gate["signal_profit_factor_after_cost"],
                )
            ),
            "signal_win_rate_after_cost": float(
                min(
                    calibration["signal_win_rate_after_cost"],
                    gate["signal_win_rate_after_cost"],
                )
            ),
            "gate_min_signal_count": int(gate_min_signals),
        }
    return None


def evaluate_low_threshold_strategy_candidate(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    candidate: dict[str, Any],
) -> dict[str, float]:
    """Evaluate one frozen low-threshold strategy without changing it."""

    direction = str(candidate.get("direction") or "")
    profile = str(candidate.get("strategy_profile") or "")
    threshold = float(candidate.get("threshold", 1.0) or 1.0)
    cost_buffer = float(
        (
            candidate.get("calibration")
            or candidate.get("gate")
            or {}
        ).get("signal_cost_buffer", 0.0)
        or 0.0
    )
    if direction not in {"long", "short"} or not profile:
        return _signal_performance_after_cost(
            frame,
            np.zeros(len(frame), dtype=bool),
            "long",
            cost_buffer,
        )
    signal = (
        np.asarray(probabilities, dtype=float) >= threshold
    ) & shadow_strategy_filter_mask(frame, direction, profile)
    return _signal_performance_after_cost(
        frame,
        signal,
        direction,
        cost_buffer,
        return_column=str(
            candidate.get("return_column") or "future_return"
        ),
        hold_bars=int(
            candidate.get("forecast_horizon_bars", 1) or 1
        ),
    )


def apply_shadow_holding_period(
    reports: list[dict[str, Any]],
    *,
    state_path: Path,
    interval: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep a validated shadow direction for its frozen forecast horizon."""

    updated_reports = deepcopy(reports)
    state: dict[str, Any] = {
        "schema_version": SHADOW_HOLDING_SCHEMA_VERSION,
        "interval": interval,
        "last_open_time": None,
        "positions": {},
        "safety": {
            "live_trading_enabled": False,
            "real_orders_allowed": False,
            "api_keys_used": False,
        },
    }
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if (
            isinstance(loaded, dict)
            and int(loaded.get("schema_version", 0))
            == SHADOW_HOLDING_SCHEMA_VERSION
            and str(loaded.get("interval") or "") == interval
        ):
            state.update(loaded)
    positions = {
        str(symbol).upper(): dict(value)
        for symbol, value in dict(state.get("positions") or {}).items()
        if isinstance(value, dict)
    }
    open_times: set[int] = set()
    for report in updated_reports:
        try:
            open_times.add(int(report.get("latest_open_time")))
        except (TypeError, ValueError):
            continue
    aligned_open_time = (
        next(iter(open_times)) if len(open_times) == 1 else None
    )
    previous_open_time = state.get("last_open_time")
    new_bar = bool(
        aligned_open_time is not None
        and (
            previous_open_time is None
            or int(aligned_open_time) > int(previous_open_time)
        )
    )
    if new_bar:
        for symbol in list(positions):
            remaining = max(
                int(positions[symbol].get("remaining_hold_bars", 0))
                - 1,
                0,
            )
            if remaining <= 0:
                del positions[symbol]
            else:
                positions[symbol]["remaining_hold_bars"] = remaining

    for report in updated_reports:
        symbol = str(report.get("symbol") or "").upper()
        decision = report.get("shadow_learning")
        if not symbol or not isinstance(decision, dict):
            continue
        blockers = {
            str(item)
            for item in decision.get("blockers") or []
        }
        hard_exit = bool(blockers & SHADOW_HARD_EXIT_BLOCKERS)
        existing = positions.get(symbol)
        if existing is not None and hard_exit:
            del positions[symbol]
            decision["holding_period_active"] = False
            decision["holding_exit_reason"] = sorted(
                blockers & SHADOW_HARD_EXIT_BLOCKERS
            )[0]
            existing = None
        if existing is not None:
            direction = int(existing.get("target_direction", 0))
            margin_fraction = float(
                existing.get("target_exposure", 0.0) or 0.0
            )
            leverage = max(int(existing.get("leverage", 1) or 1), 1)
            decision.update(
                {
                    "selected_side": str(existing.get("side") or ""),
                    "selected_model": str(
                        existing.get("selected_model") or ""
                    ),
                    "latest_signal_active": direction != 0,
                    "target_direction": direction,
                    "target_exposure": margin_fraction,
                    "target_notional_exposure": (
                        margin_fraction * leverage
                    ),
                    "leverage": leverage,
                    "holding_period_active": True,
                    "remaining_hold_bars": int(
                        existing.get("remaining_hold_bars", 0)
                    ),
                    "blockers": [
                        item
                        for item in decision.get("blockers") or []
                        if item
                        not in {
                            "shadow_probability_below_threshold",
                            "shadow_strategy_filter_blocked",
                        }
                    ],
                }
            )
            continue
        if (
            new_bar
            and bool(decision.get("latest_signal_active", False))
            and int(decision.get("target_direction", 0)) != 0
        ):
            horizon = max(
                int(decision.get("forecast_horizon_bars", 1) or 1),
                1,
            )
            positions[symbol] = {
                "side": str(decision.get("selected_side") or ""),
                "selected_model": str(
                    decision.get("selected_model") or ""
                ),
                "target_direction": int(
                    decision.get("target_direction", 0)
                ),
                "target_exposure": float(
                    decision.get("target_exposure", 0.0) or 0.0
                ),
                "leverage": max(
                    int(decision.get("leverage", 1) or 1),
                    1,
                ),
                "remaining_hold_bars": horizon,
                "opened_at": int(aligned_open_time),
            }
            decision["holding_period_active"] = True
            decision["remaining_hold_bars"] = horizon
        else:
            decision["holding_period_active"] = False
            decision["remaining_hold_bars"] = 0

    if new_bar:
        state["last_open_time"] = int(aligned_open_time)
    state["positions"] = positions
    state["updated_beijing"] = pd.Timestamp.now(
        tz="Asia/Shanghai"
    ).isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    safe_replace_text(
        state_path,
        json.dumps(state, indent=2, ensure_ascii=False),
    )
    return updated_reports, state


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
    latest_features: pd.DataFrame | dict[str, Any] | None = None,
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
    if isinstance(latest_features, dict):
        latest_feature_frame = pd.DataFrame([latest_features])
    elif isinstance(latest_features, pd.DataFrame):
        latest_feature_frame = latest_features.tail(1)
    else:
        latest_feature_frame = pd.DataFrame()
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
        strategy_profile = str(
            candidate.get("strategy_profile") or "unfiltered"
        )
        strategy_filter_passed = bool(
            strategy_profile == "unfiltered"
            or (
                len(latest_feature_frame)
                and shadow_strategy_filter_mask(
                    latest_feature_frame,
                    side,
                    strategy_profile,
                )[0]
            )
        )
        candidates.append(
            {
                **candidate,
                "side": side,
                "strategy_profile": strategy_profile,
                "strategy_filter_passed": strategy_filter_passed,
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
        and bool(item.get("strategy_filter_passed", False))
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
        if not bool(selected.get("strategy_filter_passed", False)):
            blockers.append("shadow_strategy_filter_blocked")
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
        "raw_probability_triggered": bool(
            selected.get("probability_triggered", False)
            if selected is not None
            else False
        ),
        "strategy_profile": (
            str(selected.get("strategy_profile") or "")
            if selected is not None
            else ""
        ),
        "strategy_filter_passed": bool(
            selected.get("strategy_filter_passed", False)
            if selected is not None
            else False
        ),
        "forecast_horizon_bars": int(
            selected.get("forecast_horizon_bars", 1) or 1
            if selected is not None
            else 1
        ),
        "return_column": (
            str(selected.get("return_column") or "future_return")
            if selected is not None
            else "future_return"
        ),
        "latest_signal_active": bool(latest_signal_active),
        "target_direction": int(direction),
        "target_exposure": (
            float(max_position_fraction) if latest_signal_active else 0.0
        ),
        "target_notional_exposure": (
            float(max_position_fraction) * max(int(leverage), 1)
            if latest_signal_active
            else 0.0
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
        "max_allowed_leverage": int(shadow.get("leverage", 1) or 1),
        "long_threshold": float(long_threshold),
        "short_threshold": float(short_threshold),
        "min_confidence_gap": 0.0,
        "trade_signal_threshold": 0.5,
        "trade_side_policy": side_policy,
        "max_position_fraction": max_position,
        "margin_type": "ISOLATED",
    }
    converted["candidate_mode"] = "shadow_learning"
    return converted
