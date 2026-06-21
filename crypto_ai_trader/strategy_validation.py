from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .binance_data import load_symbol_interval, resolve_exchange_rule_values
from .config import TraderConfig
from .features import FEATURE_VERSION, feature_matrix, make_features
from .model_optimization import (
    interval_requires_extra_publish_gates,
    train_accuracy_first_candidates,
    volatility_regime_backtest_report,
)
from .progress import safe_replace_text
from .strategy_config import primary_label_horizon, primary_label_min_return
from .time_utils import beijing_now_iso, beijing_stamp, to_beijing_iso


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    folds: int | None = None
    rolling_folds: int | None = None
    max_model_trials: int | None = None
    wf_train_rows: int | None = None
    wf_valid_rows: int | None = None
    wf_test_rows: int | None = None
    max_threshold_evals: int | None = None
    per_target_budget_minutes: float | None = None
    force_compact_threshold_search: bool = False
    promotion_allowed: bool = True
    description: str = ""


VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    "standard": ValidationProfile(
        name="standard",
        description="Default full validation behavior; preserves existing command semantics.",
    ),
    "large-sample-light": ValidationProfile(
        name="large-sample-light",
        folds=2,
        rolling_folds=0,
        max_model_trials=2,
        wf_train_rows=4200,
        wf_valid_rows=900,
        wf_test_rows=900,
        max_threshold_evals=160,
        per_target_budget_minutes=18.0,
        force_compact_threshold_search=True,
        description="Use larger history sources with bounded per-fold windows, compact threshold search, and frozen holdout.",
    ),
    "fast-screen": ValidationProfile(
        name="fast-screen",
        folds=1,
        rolling_folds=0,
        max_model_trials=1,
        wf_train_rows=2400,
        wf_valid_rows=500,
        wf_test_rows=500,
        max_threshold_evals=60,
        per_target_budget_minutes=8.0,
        force_compact_threshold_search=True,
        promotion_allowed=False,
        description="Fast research screening only; never promotes a paper candidate.",
    ),
    "deep-audit": ValidationProfile(
        name="deep-audit",
        folds=3,
        max_model_trials=3,
        wf_train_rows=8000,
        wf_valid_rows=1600,
        wf_test_rows=1600,
        max_threshold_evals=300,
        per_target_budget_minutes=45.0,
        force_compact_threshold_search=True,
        description="Slower audit profile for promising candidates with larger walk-forward windows.",
    ),
}


def validation_profile_or_default(name: str | None) -> ValidationProfile:
    key = str(name or "standard").strip().lower()
    if key not in VALIDATION_PROFILES:
        key = "standard"
    return VALIDATION_PROFILES[key]


def positive_int_or_none(value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def capped_profit_factor(value: float, cap: float = 10.0) -> float:
    if value != value or value < 0:
        return 0.0
    return min(float(value), cap)


def _payload_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return float(default)
    if value != value:
        return float(default)
    return float(value)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def holdout_gate_components(
    holdout_backtest: dict[str, Any],
    *,
    cost_stress: dict[str, Any],
    high_slippage_passed: bool,
    model_gate_passed: bool,
    threshold_gate_passed: bool,
    threshold_search_timed_out: bool,
    volatility_required: bool,
    volatility_regime: dict[str, Any],
    min_profit_factor: float,
    min_trades: int,
    max_drawdown_limit: float,
) -> dict[str, bool]:
    components = {
        "positive_return": _payload_float(holdout_backtest, "total_return") > 0.0,
        "profit_factor_gate": _payload_float(holdout_backtest, "profit_factor") >= float(min_profit_factor),
        "min_trades_gate": _payload_float(holdout_backtest, "trades") >= float(min_trades),
        "max_drawdown_gate": _payload_float(holdout_backtest, "max_drawdown") >= -abs(float(max_drawdown_limit)),
        "realized_risk_reward_gate": _payload_float(holdout_backtest, "realized_avg_win_loss_ratio") >= 0.90,
        "expectancy_per_trade_positive": _payload_float(holdout_backtest, "expectancy_per_trade_after_cost") > 0.0,
        "cost_stress_gate": _payload_float(cost_stress, "pass_rate") >= 0.50,
        "high_slippage_gate": bool(high_slippage_passed),
        "model_selection_gate": bool(model_gate_passed),
        "threshold_validation_gate": bool(threshold_gate_passed),
        "threshold_search_completed": not bool(threshold_search_timed_out),
        "volatility_regime_gate": (
            (not bool(volatility_required)) or bool(volatility_regime.get("gate_passed", False))
        ),
        "live_trading_disabled": True,
    }
    return components


def promotion_gate_from_components(components: dict[str, bool]) -> bool:
    required = [
        "positive_return",
        "profit_factor_gate",
        "min_trades_gate",
        "max_drawdown_gate",
        "realized_risk_reward_gate",
        "cost_stress_gate",
        "high_slippage_gate",
        "profit_quality_gate",
        "holdout_degradation_gate",
        "model_selection_gate",
        "threshold_validation_gate",
        "threshold_search_completed",
        "volatility_regime_gate",
    ]
    return bool(all(bool(components.get(key, False)) for key in required))


def shadow_candidate_from_components(components: dict[str, bool]) -> tuple[bool, str]:
    required = [
        "positive_return",
        "profit_factor_gate",
        "min_trades_gate",
        "max_drawdown_gate",
        "realized_risk_reward_gate",
        "expectancy_per_trade_positive",
        "cost_stress_gate",
        "high_slippage_gate",
        "profit_quality_gate",
        "holdout_degradation_gate",
        "threshold_validation_gate",
        "threshold_search_completed",
        "volatility_regime_gate",
        "live_trading_disabled",
    ]
    trade_quality_passed = bool(all(bool(components.get(key, False)) for key in required))
    if trade_quality_passed and bool(components.get("model_selection_gate", False)):
        return True, "holdout_trade_quality_positive_but_not_promoted"
    if trade_quality_passed:
        return False, "model_selection_gate_failed_research_only"
    return False, ""


def gate_blockers_from_components(components: dict[str, bool]) -> list[str]:
    labels = {
        "positive_return": "holdout_return_not_positive",
        "profit_factor_gate": "holdout_profit_factor_gate_failed",
        "min_trades_gate": "holdout_min_trades_gate_failed",
        "max_drawdown_gate": "holdout_drawdown_gate_failed",
        "realized_risk_reward_gate": "holdout_realized_rr_gate_failed",
        "expectancy_per_trade_positive": "holdout_expectancy_not_positive",
        "cost_stress_gate": "cost_stress_gate_failed",
        "high_slippage_gate": "high_slippage_gate_failed",
        "profit_quality_gate": "profit_quality_gate_failed",
        "holdout_degradation_gate": "holdout_degradation_gate_failed",
        "model_selection_gate": "model_selection_gate_failed",
        "threshold_validation_gate": "threshold_validation_gate_failed",
        "threshold_search_completed": "threshold_search_timed_out",
        "volatility_regime_gate": "volatility_regime_gate_failed",
        "live_trading_disabled": "live_trading_not_disabled",
    }
    return [label for key, label in labels.items() if not bool(components.get(key, False))]


def research_candidate_report(
    frozen_holdout: dict[str, Any],
    *,
    profitable_fold_rate: float,
    acceptable_profit_factor_rate: float,
    median_return_value: float,
    mean_profit_factor_value: float,
    worst_drawdown_value: float,
    mean_cost_stress_pass_rate: float,
    high_slippage_pass_rate: float,
    model_selection_pass_rate: float,
    min_fold_trades: float,
    no_trade_fold_present: bool,
    fee_drag_to_return_ratio: float,
    max_drawdown_limit: float,
) -> dict[str, Any]:
    """Research-only target: useful for learning, never for paper/testnet promotion."""

    def base_payload(status: str, blockers: list[str], reasons: list[str] | None = None) -> dict[str, Any]:
        usage = "offline_diagnostics_only"
        return {
            "eligible": status == "eligible",
            "research_status": status,
            "usage": usage,
            "research_usage": usage,
            "reasons": sorted(set(reasons or [])),
            "blockers": sorted(set(blockers)),
            "strict_gate_failures": sorted(set(blockers)),
            "promotion_blocked": True,
            "paper_candidate_allowed": False,
            "shadow_candidate_allowed": False,
            "testnet_candidate_allowed": False,
            "real_orders_allowed": False,
            "research_promotion_allowed": False,
            "research_shadow_allowed": False,
            "observe_only_override_allowed": False,
            "watchlist_allowed": status in {"watch", "eligible"},
            "next_action": "research_optimize_cost_edge" if status in {"watch", "eligible"} else "feature_and_label_optimization",
        }

    if not frozen_holdout.get("required_for_eligible", False):
        return base_payload("none", ["frozen_holdout_not_required"])
    if frozen_holdout.get("status") != "completed":
        return base_payload("none", ["frozen_holdout_not_completed"])

    backtest = as_dict(frozen_holdout.get("backtest"))
    components = as_dict(frozen_holdout.get("gate_components"))
    profit_quality = as_dict(frozen_holdout.get("profit_quality_gate"))
    degradation = as_dict(frozen_holdout.get("holdout_degradation_gate"))
    holdout_return = _payload_float(backtest, "total_return")
    holdout_profit_factor = _payload_float(backtest, "profit_factor")
    holdout_trades = _payload_float(backtest, "trades")
    holdout_expectancy = _payload_float(backtest, "expectancy_per_trade_after_cost")
    holdout_drawdown = _payload_float(backtest, "max_drawdown")
    blockers: list[str] = []
    reasons: list[str] = []

    if holdout_return <= 0.0:
        blockers.append("research_holdout_return_not_positive")
    if holdout_profit_factor < 1.20:
        blockers.append("research_holdout_profit_factor_below_1_2")
    if holdout_trades < 12.0:
        blockers.append("research_holdout_trades_below_12")
    if holdout_expectancy <= 0.0:
        blockers.append("research_holdout_expectancy_not_positive")
    if holdout_drawdown < -abs(float(max_drawdown_limit)):
        blockers.append("research_holdout_drawdown_too_deep")
    if not bool(profit_quality.get("passed", False)):
        blockers.append("research_profit_quality_gate_failed")
    if bool(degradation.get("validation_holdout_sign_flip", False)):
        blockers.append("research_validation_holdout_sign_flip")
    if bool(degradation.get("validation_pf_spike_not_replicated", False)):
        blockers.append("research_validation_pf_spike_not_replicated")
    if not bool(components.get("model_selection_gate", False)):
        blockers.append("research_model_selection_gate_failed")
    if model_selection_pass_rate < 0.50:
        blockers.append("research_model_selection_pass_rate_below_0_5")
    if not bool(components.get("threshold_validation_gate", False)):
        blockers.append("research_threshold_validation_gate_failed")
    if not bool(components.get("threshold_search_completed", False)):
        blockers.append("research_threshold_search_not_completed")
    if mean_cost_stress_pass_rate < 0.67:
        blockers.append("research_cost_stress_pass_rate_below_0_67")
    if high_slippage_pass_rate < 0.67:
        blockers.append("research_high_slippage_pass_rate_below_0_67")
    if profitable_fold_rate < 0.50:
        blockers.append("research_profitable_fold_rate_below_0_5")
    if median_return_value < 0.0:
        blockers.append("research_median_fold_return_negative")
    if worst_drawdown_value < -abs(float(max_drawdown_limit)):
        blockers.append("research_fold_drawdown_too_deep")
    if mean_profit_factor_value < 0.80:
        blockers.append("research_mean_profit_factor_below_0_8")
    if min_fold_trades <= 0.0 or no_trade_fold_present:
        blockers.append("research_no_trade_fold_present")
    if fee_drag_to_return_ratio > 0.75:
        blockers.append("research_fee_drag_ratio_above_0_75")

    if not components.get("min_trades_gate", False):
        reasons.append("holdout_trade_count_below_promotion_min")
    if not bool(degradation.get("passed", False)):
        reasons.extend(str(item) for item in degradation.get("reasons", []) or [])
    if not bool(components.get("model_selection_gate", False)):
        reasons.append("model_selection_gate_needs_improvement")
    if acceptable_profit_factor_rate < 1.0:
        reasons.append("fold_profit_factor_unstable")
    if fee_drag_to_return_ratio > 0.50:
        reasons.append("fee_drag_too_material_for_research_promotion")

    score = (
        2.0 * holdout_return
        + 0.015 * min(max(holdout_profit_factor - 1.0, 0.0), 4.0)
        + 25.0 * holdout_expectancy
        + 0.02 * profitable_fold_rate
        + 0.01 * acceptable_profit_factor_rate
        - 0.10 * abs(worst_drawdown_value)
    )
    status = "eligible" if not blockers and not reasons else ("watch" if not blockers else "blocked")
    payload = base_payload(status, blockers, reasons)
    payload.update({
        "score": float(score),
        "holdout_return": float(holdout_return),
        "holdout_profit_factor": float(holdout_profit_factor),
        "holdout_trades": float(holdout_trades),
        "holdout_expectancy_per_trade_after_cost": float(holdout_expectancy),
        "holdout_fee_drag_to_abs_return_ratio": fee_drag_to_abs_return_ratio(backtest),
        "profitable_fold_rate": float(profitable_fold_rate),
        "acceptable_profit_factor_rate": float(acceptable_profit_factor_rate),
        "median_fold_return": float(median_return_value),
        "mean_profit_factor": float(mean_profit_factor_value),
        "worst_drawdown": float(worst_drawdown_value),
        "mean_cost_stress_pass_rate": float(mean_cost_stress_pass_rate),
        "high_slippage_pass_rate": float(high_slippage_pass_rate),
        "model_selection_pass_rate": float(model_selection_pass_rate),
        "min_fold_trades": float(min_fold_trades),
        "no_trade_fold_present": bool(no_trade_fold_present),
        "fee_drag_to_return_ratio": float(fee_drag_to_return_ratio),
    })
    return payload


def fee_drag_to_abs_return_ratio(backtest: dict[str, Any]) -> float:
    fee_drag = max(
        _payload_float(backtest, "total_cost_drag")
        or (_payload_float(backtest, "fee_drag") + _payload_float(backtest, "funding_drag")),
        0.0,
    )
    total_return = abs(_payload_float(backtest, "total_return"))
    if total_return <= 1e-12:
        return 999.0 if fee_drag > 0 else 0.0
    return float(fee_drag / total_return)


def profit_quality_gate_report(
    backtest: dict[str, Any],
    *,
    min_return: float = 0.0020,
    max_fee_drag_ratio: float = 0.35,
    min_expectancy_per_trade: float = 0.00008,
) -> dict[str, Any]:
    total_return = _payload_float(backtest, "total_return")
    expectancy_per_trade = _payload_float(backtest, "expectancy_per_trade_after_cost")
    ratio = fee_drag_to_abs_return_ratio(backtest)
    reasons: list[str] = []
    if total_return < float(min_return):
        reasons.append("return_below_profit_quality_min")
    if expectancy_per_trade < float(min_expectancy_per_trade):
        reasons.append("expectancy_per_trade_below_quality_min")
    if ratio > float(max_fee_drag_ratio):
        reasons.append("fee_drag_ratio_above_quality_max")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "min_return": float(min_return),
        "max_fee_drag_to_abs_return_ratio": float(max_fee_drag_ratio),
        "min_expectancy_per_trade_after_cost": float(min_expectancy_per_trade),
        "total_return": float(total_return),
        "expectancy_per_trade_after_cost": float(expectancy_per_trade),
        "fee_drag_to_abs_return_ratio": float(ratio),
    }


def holdout_degradation_gate_report(
    holdout_backtest: dict[str, Any],
    fold_reports: list[dict[str, Any]],
    *,
    min_trades: int,
    selected_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fold_returns = [
        _payload_float(item.get("test_backtest", {}) if isinstance(item, dict) else {}, "total_return")
        for item in fold_reports
    ]
    fold_trades = [
        _payload_float(item.get("test_backtest", {}) if isinstance(item, dict) else {}, "trades")
        for item in fold_reports
    ]
    positive_fold_returns = [value for value in fold_returns if value > 0.0]
    median_fold_return = median(positive_fold_returns or fold_returns)
    median_fold_trades = median(fold_trades)
    holdout_return = _payload_float(holdout_backtest, "total_return")
    holdout_trades = _payload_float(holdout_backtest, "trades")
    holdout_expectancy = _payload_float(holdout_backtest, "expectancy_per_trade_after_cost")
    required_return = max(0.0, 0.25 * median_fold_return)
    required_trades = max(float(min_trades), 0.50 * median_fold_trades)
    return_retention = (
        float(holdout_return / median_fold_return)
        if median_fold_return > 1e-12
        else (1.0 if holdout_return > 0.0 else 0.0)
    )
    trade_retention = (
        float(holdout_trades / median_fold_trades)
        if median_fold_trades > 1e-12
        else (1.0 if holdout_trades > 0 else 0.0)
    )
    threshold_summary = selected_validation if isinstance(selected_validation, dict) else {}
    selected_validation_return = _payload_float(threshold_summary, "selected_total_return")
    selected_validation_pf = _payload_float(threshold_summary, "selected_profit_factor")
    validation_holdout_sign_flip = bool(selected_validation_return > 0.0 and holdout_return <= 0.0)
    validation_pf_spike_not_replicated = bool(selected_validation_pf > 10.0 and _payload_float(holdout_backtest, "profit_factor") < 1.0)
    reasons: list[str] = []
    if holdout_return <= 0.0:
        reasons.append("holdout_return_not_positive")
    if holdout_return < required_return:
        reasons.append("holdout_return_retention_too_low")
    if holdout_trades < required_trades:
        reasons.append("holdout_trade_retention_too_low")
    if holdout_expectancy <= 0.0:
        reasons.append("holdout_expectancy_not_positive")
    if validation_holdout_sign_flip:
        reasons.append("validation_holdout_sign_flip")
    if validation_pf_spike_not_replicated:
        reasons.append("validation_pf_spike_not_replicated")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "median_fold_return": float(median_fold_return),
        "median_fold_trades": float(median_fold_trades),
        "holdout_return": float(holdout_return),
        "holdout_trades": float(holdout_trades),
        "holdout_expectancy_per_trade_after_cost": float(holdout_expectancy),
        "required_holdout_return": float(required_return),
        "required_holdout_trades": float(required_trades),
        "holdout_return_retention": float(return_retention),
        "holdout_trade_retention": float(trade_retention),
        "selected_validation_return": float(selected_validation_return),
        "selected_validation_profit_factor": float(selected_validation_pf),
        "validation_holdout_sign_flip": validation_holdout_sign_flip,
        "validation_pf_spike_not_replicated": validation_pf_spike_not_replicated,
    }


def profit_bottleneck_report(
    *,
    mean_return_value: float,
    mean_fee_drag_value: float,
    mean_balanced_accuracy_value: float,
    mean_trades_value: float,
    min_required_trades: int,
    mean_exposure_value: float,
    mean_active_exposure_value: float,
    mean_active_position_fraction_value: float,
    total_long_trades: float,
    total_short_trades: float,
    mean_turnover_value: float,
) -> dict[str, Any]:
    fee_drag_to_return = (
        float(mean_fee_drag_value / abs(mean_return_value))
        if abs(mean_return_value) > 1e-12
        else 0.0
    )
    reasons: list[str] = []
    if mean_balanced_accuracy_value < 0.53:
        reasons.append("weak_direction_edge")
    if mean_exposure_value < 0.02:
        reasons.append("very_low_average_exposure")
    elif mean_exposure_value < 0.08:
        reasons.append("low_average_exposure")
    if mean_active_exposure_value > 0.0 and mean_active_exposure_value < 0.08:
        reasons.append("low_active_exposure")
    if total_short_trades <= 0 and total_long_trades > 0:
        reasons.append("short_side_not_contributing")
    if fee_drag_to_return >= 0.35:
        reasons.append("fee_drag_material_vs_return")
    if mean_trades_value < max(float(min_required_trades) * 1.5, float(min_required_trades + 3)):
        reasons.append("low_trade_count")
    if mean_turnover_value < 0.75:
        reasons.append("low_notional_turnover")
    return {
        "reasons": reasons,
        "mean_return": float(mean_return_value),
        "mean_fee_drag": float(mean_fee_drag_value),
        "fee_drag_to_return_ratio": float(fee_drag_to_return),
        "mean_balanced_accuracy": float(mean_balanced_accuracy_value),
        "mean_trades": float(mean_trades_value),
        "mean_avg_exposure": float(mean_exposure_value),
        "mean_active_avg_exposure": float(mean_active_exposure_value),
        "mean_active_avg_position_fraction": float(mean_active_position_fraction_value),
        "mean_notional_turnover": float(mean_turnover_value),
        "total_long_trades": float(total_long_trades),
        "total_short_trades": float(total_short_trades),
        "interpretation": "Profit is evaluated after fees, slippage, funding buffer, walk-forward validation, and frozen holdout gates.",
    }


def summary_fold_count(summary: dict[str, Any] | None) -> int:
    if not isinstance(summary, dict):
        return 0
    fold_independence = summary.get("fold_independence", {})
    if isinstance(fold_independence, dict):
        try:
            return int(fold_independence.get("fold_count", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def threshold_optimization_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    best = payload.get("best", {}) if isinstance(payload.get("best"), dict) else {}
    return {
        "timed_out": bool(payload.get("timed_out", False)),
        "searched": int(payload.get("searched", 0) or 0),
        "max_threshold_evals": payload.get("max_threshold_evals"),
        "threshold_eval_limit_hit": bool(payload.get("threshold_eval_limit_hit", False)),
        "selection_dataset": payload.get("selection_dataset"),
        "gate_dataset": payload.get("gate_dataset"),
        "separate_gate_enabled": bool(payload.get("separate_gate_enabled", False)),
        "calibration_rows": int(payload.get("calibration_rows", 0) or 0),
        "gate_rows": int(payload.get("gate_rows", 0) or 0),
        "calibration_gate_passed_count": int(payload.get("calibration_gate_passed_count", 0) or 0),
        "validation_gate_evaluated_count": int(payload.get("validation_gate_evaluated_count", 0) or 0),
        "valid_gate_passed_count": int(payload.get("valid_gate_passed_count", 0) or 0),
        "selected_risk_profile": best.get("risk_profile"),
        "selected_side_policy": best.get("side_policy"),
        "selected_long_threshold": best.get("long_threshold"),
        "selected_short_threshold": best.get("short_threshold"),
        "selected_trade_signal_threshold": best.get("trade_signal_threshold"),
        "selected_score": best.get("score"),
        "selected_total_return": (best.get("backtest") or {}).get("total_return") if isinstance(best.get("backtest"), dict) else None,
        "selected_profit_factor": (best.get("backtest") or {}).get("profit_factor") if isinstance(best.get("backtest"), dict) else None,
        "selected_trades": (best.get("backtest") or {}).get("trades") if isinstance(best.get("backtest"), dict) else None,
        "risk_profiles_searched": payload.get("risk_profiles_searched", []),
        "threshold_pairs_searched": payload.get("threshold_pairs_searched", []),
        "trade_thresholds_searched": payload.get("trade_thresholds_searched", []),
        "side_policies_searched": payload.get("side_policies_searched", []),
        "side_policies_requested": payload.get("side_policies_requested", []),
        "short_candidate_allowed": bool(payload.get("short_candidate_allowed", False)),
        "short_candidate_blockers": payload.get("short_candidate_blockers", []),
        "short_side_preflight_gate": payload.get("short_side_preflight_gate", {}),
    }


def multi_horizon_specs_for_interval(interval: str) -> list[dict[str, str]]:
    steps_by_interval = {
        "5m": [(1, 5), (2, 10), (3, 15), (6, 30), (12, 60)],
        "15m": [(1, 15), (2, 30), (4, 60), (16, 240)],
        "1h": [(1, 60), (3, 180), (4, 240), (6, 360), (12, 720)],
    }
    return [
        {
            "key": f"next_{minutes}m",
            "label": f"Next {minutes} minutes",
            "minutes": str(minutes),
            "steps": str(steps),
            "target_col": f"edge_long_target_h{steps}",
            "return_col": f"future_return_h{steps}",
            "target_semantics": "net_long_edge_after_cost",
        }
        for steps, minutes in steps_by_interval.get(interval.lower(), [])
    ]


def cost_stress_profiles(base_cfg: BacktestConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": "base",
            "overrides": {},
        },
        {
            "name": "high_slippage",
            "overrides": {
                "slippage_rate": float(base_cfg.slippage_rate) * 2.0,
                "funding_rate_buffer": float(base_cfg.funding_rate_buffer) * 1.5,
            },
        },
        {
            "name": "high_fee_plus_slippage",
            "overrides": {
                "fee_rate": float(base_cfg.fee_rate) * 1.5,
                "slippage_rate": float(base_cfg.slippage_rate) * 2.0,
                "funding_rate_buffer": float(base_cfg.funding_rate_buffer) * 2.0,
                "min_atr_cost_multiplier": max(float(base_cfg.min_atr_cost_multiplier), 2.5),
            },
        },
    ]


def run_cost_stress_tests(
    test_df: pd.DataFrame,
    bundle,
    cfg_payload: dict[str, Any],
    *,
    min_profit_factor: float,
    max_drawdown_limit: float,
) -> dict[str, Any]:
    base_cfg = BacktestConfig(**cfg_payload)
    items: list[dict[str, Any]] = []
    for profile in cost_stress_profiles(base_cfg):
        cfg = BacktestConfig(**{**asdict(base_cfg), **profile["overrides"]})
        result, _ = run_backtest(test_df, bundle, cfg)
        passed = bool(
            result.total_return > 0.0
            and result.profit_factor >= min_profit_factor
            and result.max_drawdown >= -abs(max_drawdown_limit)
        )
        items.append(
            {
                "name": profile["name"],
                "passed": passed,
                "overrides": profile["overrides"],
                "backtest": asdict(result),
            }
        )
    pass_rate = float(sum(1 for item in items if item["passed"]) / len(items)) if items else 0.0
    return {
        "items": items,
        "pass_rate": pass_rate,
        "all_passed": bool(items and all(item["passed"] for item in items)),
    }


def audit_walk_forward_configs_on_holdout(
    holdout_df: pd.DataFrame,
    bundle,
    fold_reports: list[dict[str, Any]],
    *,
    min_profit_factor: float,
    max_drawdown_limit: float,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for fold in fold_reports:
        cfg_payload = fold.get("optimized_backtest_config", {})
        if not isinstance(cfg_payload, dict) or not cfg_payload:
            continue
        try:
            cfg = BacktestConfig(**cfg_payload)
            result, detail = run_backtest(holdout_df, bundle, cfg)
            stress = run_cost_stress_tests(
                holdout_df,
                bundle,
                asdict(cfg),
                min_profit_factor=min_profit_factor,
                max_drawdown_limit=max_drawdown_limit,
            )
            volatility = volatility_regime_backtest_report(
                holdout_df,
                detail,
                min_profit_factor=min_profit_factor,
                max_drawdown_floor=-abs(max_drawdown_limit),
                min_regime_trades=3,
            )
            passed = bool(
                result.total_return > 0.0
                and result.profit_factor >= min_profit_factor
                and result.max_drawdown >= -abs(max_drawdown_limit)
                and result.trades > 0
            )
            items.append(
                {
                    "fold": fold.get("fold"),
                    "source": "walk_forward_fold_optimized_config",
                    "used_for_selection": False,
                    "selected_risk_profile": fold.get("selected_risk_profile"),
                    "selected_side_policy": fold.get("selected_side_policy"),
                    "strategy_archetype_policy": cfg_payload.get("strategy_archetype_policy"),
                    "passed_basic_holdout_audit": passed,
                    "backtest": asdict(result),
                    "cost_stress": stress,
                    "volatility_regime": volatility,
                    "config": cfg_payload,
                }
            )
        except Exception as exc:
            items.append(
                {
                    "fold": fold.get("fold"),
                    "source": "walk_forward_fold_optimized_config",
                    "used_for_selection": False,
                    "error": str(exc),
                }
            )
    ranked = sorted(
        items,
        key=lambda item: (
            bool(item.get("passed_basic_holdout_audit", False)),
            float(item.get("backtest", {}).get("total_return", 0.0)) if isinstance(item.get("backtest"), dict) else 0.0,
            float(item.get("backtest", {}).get("profit_factor", 0.0)) if isinstance(item.get("backtest"), dict) else 0.0,
        ),
        reverse=True,
    )
    return {
        "enabled": bool(items),
        "used_for_selection": False,
        "note": "Applies already-selected walk-forward fold configs to frozen holdout as audit only; it must not select thresholds or promote a model.",
        "items": items,
        "ranked": ranked,
        "any_basic_passed": any(bool(item.get("passed_basic_holdout_audit", False)) for item in items),
    }


def walk_forward_windows(
    rows: int,
    *,
    folds: int,
    purge_rows: int = 0,
    train_rows: int | None = None,
    valid_rows: int | None = None,
    test_rows: int | None = None,
) -> list[dict[str, int]]:
    if rows < 600:
        return []

    folds = max(1, folds)
    train_rows = train_rows or max(400, int(rows * 0.50))
    valid_rows = valid_rows or max(120, int(rows * 0.15))
    test_rows = test_rows or max(120, int(rows * 0.15))
    purge_rows = max(0, int(purge_rows))
    total = train_rows + valid_rows + test_rows + 2 * purge_rows
    if total > rows:
        scale = rows / total
        train_rows = max(300, int(train_rows * scale))
        valid_rows = max(80, int(valid_rows * scale))
        test_rows = max(80, rows - train_rows - valid_rows - 2 * purge_rows)
    total = train_rows + valid_rows + test_rows + 2 * purge_rows
    if total > rows or min(train_rows, valid_rows, test_rows) <= 0:
        return []

    max_start = rows - total
    if folds == 1 or max_start <= 0:
        starts = [0]
    else:
        step = max(1, max_start // (folds - 1))
        starts = [min(max_start, idx * step) for idx in range(folds)]
    windows = []
    seen = set()
    for start in starts:
        window = {
            "train_start": start,
            "train_end": start + train_rows,
            "valid_start": start + train_rows + purge_rows,
            "valid_end": start + train_rows + purge_rows + valid_rows,
            "test_start": start + train_rows + purge_rows + valid_rows + purge_rows,
            "test_end": start + total,
            "purge_rows_after_train": purge_rows,
            "purge_rows_after_valid": purge_rows,
        }
        key = tuple(window.values())
        if key not in seen:
            windows.append(window)
            seen.add(key)
    return windows


def fold_independence_report(
    windows: list[dict[str, int]],
    *,
    purge_rows: int,
    holdout_rows: int,
) -> dict[str, Any]:
    max_window_overlap = 0
    max_test_overlap = 0
    max_window_rows = 1
    max_test_rows = 1
    for window in windows:
        max_window_rows = max(max_window_rows, int(window["test_end"] - window["train_start"]))
        max_test_rows = max(max_test_rows, int(window["test_end"] - window["test_start"]))
    for left_index, left in enumerate(windows):
        for right in windows[left_index + 1 :]:
            window_overlap = max(
                0,
                min(int(left["test_end"]), int(right["test_end"]))
                - max(int(left["train_start"]), int(right["train_start"])),
            )
            test_overlap = max(
                0,
                min(int(left["test_end"]), int(right["test_end"]))
                - max(int(left["test_start"]), int(right["test_start"])),
            )
            max_window_overlap = max(max_window_overlap, window_overlap)
            max_test_overlap = max(max_test_overlap, test_overlap)
    return {
        "fold_overlap": max_window_overlap > 0,
        "cross_fold_test_overlap": max_test_overlap > 0,
        "fold_count": len(windows),
        "max_cross_fold_overlap_rows": int(max_window_overlap),
        "max_cross_fold_test_overlap_rows": int(max_test_overlap),
        "overlap_ratio": float(max_window_overlap / max_window_rows),
        "test_overlap_ratio": float(max_test_overlap / max_test_rows),
        "purge_gap_rows": int(purge_rows),
        "non_overlapping_holdout_used": bool(holdout_rows > 0),
        "holdout_used_for_selection": False,
        "test_used_for_model_selection": False,
        "test_used_for_threshold_selection": False,
        "test_used_for_publish_gate": True,
        "note": (
            "Rolling folds can overlap across folds, but each fold keeps purge gaps between "
            "train/valid/test; frozen holdout is excluded from selection and used only as a final gate."
        ),
    }


def capped_holdout_training_split(
    frame: pd.DataFrame,
    *,
    purge_rows: int,
    train_rows: int | None,
    valid_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = len(frame)
    if rows <= 0:
        return pd.DataFrame(), pd.DataFrame(), {"mode": "empty"}
    purge_rows = max(0, int(purge_rows))
    if train_rows is None and valid_rows is None:
        final_train_end = max(300, int(rows * 0.75))
        final_valid_start = min(rows, final_train_end + purge_rows)
        return (
            frame.iloc[:final_train_end].reset_index(drop=True),
            frame.iloc[final_valid_start:].reset_index(drop=True),
            {
                "mode": "expanding_legacy",
                "train_start": 0,
                "train_end": int(final_train_end),
                "valid_start": int(final_valid_start),
                "valid_end": int(rows),
                "purge_rows": int(purge_rows),
            },
        )

    requested_valid_rows = max(80, int(valid_rows or max(120, rows * 0.15)))
    requested_train_rows = max(300, int(train_rows or max(400, rows * 0.50)))
    total = requested_train_rows + requested_valid_rows + purge_rows
    if total > rows:
        scale = rows / max(float(total), 1.0)
        requested_train_rows = max(300, int(requested_train_rows * scale))
        requested_valid_rows = max(80, rows - requested_train_rows - purge_rows)
    total = requested_train_rows + requested_valid_rows + purge_rows
    if total > rows or requested_valid_rows <= 0 or requested_train_rows <= 0:
        return pd.DataFrame(), pd.DataFrame(), {
            "mode": "capped_recent",
            "error": "not_enough_rows_for_capped_holdout_training",
            "rows": int(rows),
            "requested_train_rows": int(train_rows or 0),
            "requested_valid_rows": int(valid_rows or 0),
            "purge_rows": int(purge_rows),
        }

    train_start = rows - total
    train_end = train_start + requested_train_rows
    valid_start = train_end + purge_rows
    valid_end = valid_start + requested_valid_rows
    return (
        frame.iloc[train_start:train_end].reset_index(drop=True),
        frame.iloc[valid_start:valid_end].reset_index(drop=True),
        {
            "mode": "capped_recent",
            "train_start": int(train_start),
            "train_end": int(train_end),
            "valid_start": int(valid_start),
            "valid_end": int(valid_end),
            "train_rows": int(requested_train_rows),
            "valid_rows": int(requested_valid_rows),
            "purge_rows": int(purge_rows),
        },
    )


def validate_symbol_interval(
    symbol: str,
    interval: str,
    cfg: TraderConfig,
    *,
    include_realtime: bool,
    folds: int,
    max_model_trials: int,
    deadline: float,
    initial_balance: float,
    min_trades: int,
    max_drawdown_limit: float,
    min_profit_factor: float,
    complexity: str,
    rolling_folds: int,
    purge_rows: int | None,
    holdout_fraction: float,
    max_training_rows: int | None = None,
    validation_profile: str = "standard",
    wf_train_rows: int | None = None,
    wf_valid_rows: int | None = None,
    wf_test_rows: int | None = None,
    max_threshold_evals: int | None = None,
    force_compact_threshold_search: bool = False,
    promotion_allowed: bool = True,
) -> dict[str, Any]:
    symbol = symbol.upper()
    raw = load_symbol_interval(cfg.data_dir, symbol, interval, include_realtime=include_realtime)
    data_validation = dict(raw.attrs.get("data_validation", {}))
    market_context = dict(raw.attrs.get("market_context", {}))
    raw_rows_before_window = len(raw)
    window_rows = int(max_training_rows or 0)
    training_window_applied = False
    if window_rows > 0 and len(raw) > window_rows:
        raw = raw.tail(window_rows).reset_index(drop=True)
        training_window_applied = True
    label_horizon = primary_label_horizon(interval, cfg)
    label_min_return = primary_label_min_return(interval, cfg)
    frame = make_features(raw, label_horizon=label_horizon, label_min_return=label_min_return)
    effective_purge_rows = max(int(purge_rows) if purge_rows is not None else label_horizon, label_horizon, 0)
    holdout_fraction = max(0.0, min(float(holdout_fraction), 0.35))
    holdout_rows = 0
    if holdout_fraction > 0:
        holdout_rows = max(120, int(len(frame) * holdout_fraction))
        if len(frame) - holdout_rows - effective_purge_rows < 600:
            holdout_rows = 0
    holdout_start = len(frame) - holdout_rows if holdout_rows > 0 else len(frame)
    trainable_end = max(0, holdout_start - effective_purge_rows) if holdout_rows > 0 else len(frame)
    validation_frame = frame.iloc[:trainable_end].reset_index(drop=True)
    frozen_holdout_df = frame.iloc[holdout_start:].reset_index(drop=True) if holdout_rows > 0 else pd.DataFrame()
    effective_wf_train_rows = positive_int_or_none(wf_train_rows)
    effective_wf_valid_rows = positive_int_or_none(wf_valid_rows)
    effective_wf_test_rows = positive_int_or_none(wf_test_rows)
    effective_max_threshold_evals = positive_int_or_none(max_threshold_evals)
    windows = walk_forward_windows(
        len(validation_frame),
        folds=folds,
        purge_rows=effective_purge_rows,
        train_rows=effective_wf_train_rows,
        valid_rows=effective_wf_valid_rows,
        test_rows=effective_wf_test_rows,
    )
    if not windows:
        raise ValueError(f"not_enough_rows_for_walk_forward: {symbol} {interval} rows={len(frame)}")
    fold_independence = fold_independence_report(
        windows,
        purge_rows=effective_purge_rows,
        holdout_rows=holdout_rows,
    )

    _, _, feature_columns = feature_matrix(validation_frame.iloc[: windows[0]["train_end"]].reset_index(drop=True))
    exchange_rules = resolve_exchange_rule_values(
        cfg.data_dir,
        symbol,
        min_notional_usdt=cfg.exchange_min_notional_usdt,
        min_quantity=cfg.exchange_min_quantity,
        max_quantity=cfg.exchange_max_quantity,
        quantity_step=cfg.exchange_quantity_step,
        price_tick_size=cfg.exchange_price_tick_size,
    )
    base_backtest_cfg = BacktestConfig(
        initial_balance=initial_balance,
        leverage=cfg.default_leverage,
        max_allowed_leverage=cfg.max_leverage,
        fee_rate=cfg.fee_rate,
        maker_fee_rate=cfg.maker_fee_rate,
        maker_fill_fraction=cfg.maker_fill_fraction,
        slippage_rate=cfg.slippage_rate,
        partial_fill_ratio=cfg.partial_fill_ratio,
        execution_latency_bars=cfg.execution_latency_bars,
        min_order_notional_fraction=cfg.min_order_notional_fraction,
        **exchange_rules,
        exchange_downtime_guard_enabled=cfg.exchange_downtime_guard_enabled,
        exchange_gap_recovery_bars=cfg.exchange_gap_recovery_bars,
        liquidity_execution_enabled=cfg.liquidity_execution_enabled,
        max_bar_participation_rate=cfg.max_bar_participation_rate,
        liquidity_lookback_bars=cfg.liquidity_lookback_bars,
        slippage_impact_coefficient=cfg.slippage_impact_coefficient,
        max_dynamic_slippage_rate=cfg.max_dynamic_slippage_rate,
        liquidation_guard_enabled=cfg.liquidation_guard_enabled,
        maintenance_margin_rate=cfg.maintenance_margin_rate,
        liquidation_buffer=cfg.liquidation_buffer,
        liquidation_fee_rate=cfg.liquidation_fee_rate,
        ewma_volatility_enabled=cfg.ewma_volatility_enabled,
        ewma_volatility_span=cfg.ewma_volatility_span,
        ewma_daily_volatility_target=cfg.ewma_daily_volatility_target,
        funding_crowding_guard_enabled=cfg.funding_crowding_guard_enabled,
        funding_crowding_max_rate=cfg.funding_crowding_max_rate,
        regime_risk_guard_enabled=cfg.regime_risk_guard_enabled,
        regime_detection_method=cfg.regime_detection_method,
        regime_statistical_clusters=cfg.regime_statistical_clusters,
        regime_statistical_min_history=cfg.regime_statistical_min_history,
        regime_statistical_lookback=cfg.regime_statistical_lookback,
        regime_statistical_refit_interval=cfg.regime_statistical_refit_interval,
        regime_statistical_random_seed=cfg.regime_statistical_random_seed,
        long_threshold=cfg.long_threshold,
        short_threshold=cfg.short_threshold,
        risk_per_trade=cfg.risk_per_trade,
        min_confidence_gap=cfg.min_confidence_gap,
    )
    effective_force_compact_threshold_search = bool(
        force_compact_threshold_search
        or max_model_trials <= 1
        or (max_training_rows is not None and int(max_training_rows or 0) >= 6000)
        or len(validation_frame) >= 6000
        or effective_max_threshold_evals is not None
    )

    fold_reports: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []
    budget_allocation: list[dict[str, Any]] = []
    for fold_index, window in enumerate(windows, start=1):
        now = time.monotonic()
        remaining_seconds = max(0.0, float(deadline - now))
        if remaining_seconds <= 0.0:
            skipped_folds.append({"fold": fold_index, "reason": "time_budget_exhausted"})
            break
        remaining_folds = max(1, len(windows) - fold_index + 1)
        holdout_reserve_seconds = 0.0
        if holdout_rows > 0:
            holdout_reserve_seconds = min(420.0, max(180.0, remaining_seconds * 0.22))
        usable_fold_seconds = max(60.0, remaining_seconds - holdout_reserve_seconds)
        fold_budget_seconds = max(90.0, usable_fold_seconds / float(remaining_folds))
        fold_deadline = min(deadline, now + fold_budget_seconds)
        if holdout_rows > 0 and remaining_folds == 1:
            fold_deadline = min(fold_deadline, max(now + 60.0, deadline - holdout_reserve_seconds))
        budget_allocation.append(
            {
                "fold": fold_index,
                "remaining_seconds_at_start": remaining_seconds,
                "allocated_fold_seconds": max(0.0, fold_deadline - now),
                "reserved_holdout_seconds": holdout_reserve_seconds if remaining_folds == 1 else 0.0,
                "remaining_folds": remaining_folds,
            }
        )
        train_df = validation_frame.iloc[window["train_start"] : window["train_end"]].reset_index(drop=True)
        valid_df = validation_frame.iloc[window["valid_start"] : window["valid_end"]].reset_index(drop=True)
        test_df = validation_frame.iloc[window["test_start"] : window["test_end"]].reset_index(drop=True)
        try:
            bundle, model_report, test_detail = train_accuracy_first_candidates(
                train_df,
                valid_df,
                test_df,
                feature_columns,
                seed=cfg.random_seed + fold_index,
                max_model_trials=max_model_trials,
                deadline=fold_deadline,
                min_trades=min_trades,
                max_drawdown_floor=-abs(max_drawdown_limit),
                min_profit_factor=min_profit_factor,
                base_backtest_cfg=base_backtest_cfg,
                label_min_return=label_min_return,
                complexity=complexity,
                rolling_folds=rolling_folds,
                multi_horizon_specs=multi_horizon_specs_for_interval(interval),
                objective="return",
                force_compact_threshold_search=effective_force_compact_threshold_search,
                max_threshold_evals=effective_max_threshold_evals,
                validation_split_purge_rows=effective_purge_rows,
            )
            test_backtest = model_report.get("test_backtest", {})
            test_metrics = model_report.get("test_metrics", {})
            optimized_cfg = model_report.get("optimized_backtest_config", {}) or asdict(base_backtest_cfg)
            min_regime_trades = max(3, int((max(min_trades, 1) + 2) // 3))
            volatility_regime = volatility_regime_backtest_report(
                train_df,
                test_detail,
                min_profit_factor=min_profit_factor,
                max_drawdown_floor=-abs(max_drawdown_limit),
                min_regime_trades=min_regime_trades,
            )
            volatility_regime["required_for_eligible"] = bool(interval_requires_extra_publish_gates(interval))
            cost_stress = run_cost_stress_tests(
                test_df,
                bundle,
                optimized_cfg,
                min_profit_factor=min_profit_factor,
                max_drawdown_limit=max_drawdown_limit,
            )
            fold_reports.append(
                {
                    "fold": fold_index,
                    "window": window,
                    "rows": {
                        "train": len(train_df),
                        "valid": len(valid_df),
                        "test": len(test_df),
                    },
                    "date_range": {
                        "train_start": to_beijing_iso(train_df["open_datetime"].iloc[0]),
                        "train_end": to_beijing_iso(train_df["open_datetime"].iloc[-1]),
                        "valid_start": to_beijing_iso(valid_df["open_datetime"].iloc[0]),
                        "valid_end": to_beijing_iso(valid_df["open_datetime"].iloc[-1]),
                        "test_start": to_beijing_iso(test_df["open_datetime"].iloc[0]),
                        "test_end": to_beijing_iso(test_df["open_datetime"].iloc[-1]),
                    },
                    "model_name": bundle.model_name,
                    "test_metrics": test_metrics,
                    "test_backtest": test_backtest,
                    "cost_stress": cost_stress,
                    "volatility_regime": volatility_regime,
                    "selected_risk_profile": model_report.get("small_account_strategy", {}).get("selected_risk_profile"),
                    "selected_side_policy": model_report.get("small_account_strategy", {}).get("selected_side_policy"),
                    "model_selection_gate": model_report.get("model_selection_gate", {}),
                    "threshold_optimization_summary": threshold_optimization_summary(model_report.get("threshold_optimization", {})),
                    "optimized_backtest_config": optimized_cfg,
                    "anti_overfit": model_report.get("anti_overfit", {}),
                    "winning_weight_variant": model_report.get("winning_weight_variant"),
                    "ranking_top": model_report.get("ranking", [])[:5],
                }
            )
        except Exception as exc:
            skipped_folds.append({"fold": fold_index, "reason": str(exc)})

    if not fold_reports:
        raise RuntimeError(f"no_walk_forward_folds_completed: {symbol} {interval} skipped={skipped_folds}")

    frozen_holdout: dict[str, Any] = {
        "enabled": bool(holdout_rows > 0),
        "required_for_eligible": bool(holdout_rows > 0),
        "status": "not_configured" if holdout_rows <= 0 else "pending",
        "rows": int(len(frozen_holdout_df)) if holdout_rows > 0 else 0,
        "holdout_fraction": holdout_fraction,
        "purge_rows_before_holdout": effective_purge_rows if holdout_rows > 0 else 0,
        "gate_passed": False,
        "note": "Frozen holdout is excluded from walk-forward model/threshold selection and used only as a final audit gate.",
    }
    if holdout_rows > 0:
        frozen_holdout["date_range"] = {
            "holdout_start": to_beijing_iso(frozen_holdout_df["open_datetime"].iloc[0]) if len(frozen_holdout_df) else "",
            "holdout_end": to_beijing_iso(frozen_holdout_df["open_datetime"].iloc[-1]) if len(frozen_holdout_df) else "",
        }
        if time.monotonic() >= deadline:
            frozen_holdout["status"] = "skipped_time_budget_exhausted"
        else:
            holdout_started = time.monotonic()
            holdout_remaining_seconds = max(0.0, float(deadline - holdout_started))
            holdout_budget_seconds = max(90.0, holdout_remaining_seconds - 15.0)
            holdout_deadline = min(deadline, holdout_started + holdout_budget_seconds)
            frozen_holdout["budget"] = {
                "remaining_seconds_at_start": holdout_remaining_seconds,
                "allocated_holdout_seconds": max(0.0, holdout_deadline - holdout_started),
                "search_mode_hint": "forced_compact",
            }
            final_train_df, final_valid_df, holdout_training_window = capped_holdout_training_split(
                validation_frame,
                purge_rows=effective_purge_rows,
                train_rows=effective_wf_train_rows,
                valid_rows=effective_wf_valid_rows,
            )
            frozen_holdout["training_window"] = holdout_training_window
            if len(final_train_df) < 300 or len(final_valid_df) < 80 or len(frozen_holdout_df) < 80:
                frozen_holdout["status"] = "skipped_not_enough_rows"
                frozen_holdout["rows_used"] = {
                    "train": len(final_train_df),
                    "valid": len(final_valid_df),
                    "holdout": len(frozen_holdout_df),
                }
            else:
                try:
                    bundle, model_report, holdout_detail = train_accuracy_first_candidates(
                        final_train_df,
                        final_valid_df,
                        frozen_holdout_df,
                        feature_columns,
                        seed=cfg.random_seed + 10_000,
                        max_model_trials=max(1, min(max_model_trials, 1)),
                        deadline=holdout_deadline,
                        min_trades=min_trades,
                        max_drawdown_floor=-abs(max_drawdown_limit),
                        min_profit_factor=min_profit_factor,
                        base_backtest_cfg=base_backtest_cfg,
                        label_min_return=label_min_return,
                        complexity=complexity,
                        rolling_folds=0,
                        multi_horizon_specs=multi_horizon_specs_for_interval(interval),
                        objective="return",
                        force_compact_threshold_search=True,
                        max_threshold_evals=effective_max_threshold_evals,
                        validation_split_purge_rows=effective_purge_rows,
                    )
                    holdout_backtest = model_report.get("test_backtest", {})
                    holdout_metrics = model_report.get("test_metrics", {})
                    optimized_cfg = model_report.get("optimized_backtest_config", {}) or asdict(base_backtest_cfg)
                    cost_stress = run_cost_stress_tests(
                        frozen_holdout_df,
                        bundle,
                        optimized_cfg,
                        min_profit_factor=min_profit_factor,
                        max_drawdown_limit=max_drawdown_limit,
                    )
                    high_slippage_passed = bool(
                        next(
                            (
                                item.get("passed", False)
                                for item in cost_stress.get("items", [])
                                if item.get("name") == "high_slippage"
                            ),
                            False,
                        )
                    )
                    min_regime_trades = max(3, int((max(min_trades, 1) + 2) // 3))
                    volatility_regime = volatility_regime_backtest_report(
                        final_train_df,
                        holdout_detail,
                        min_profit_factor=min_profit_factor,
                        max_drawdown_floor=-abs(max_drawdown_limit),
                        min_regime_trades=min_regime_trades,
                    )
                    volatility_required = bool(interval_requires_extra_publish_gates(interval))
                    volatility_regime["required_for_eligible"] = volatility_required
                    walk_forward_config_audit = audit_walk_forward_configs_on_holdout(
                        frozen_holdout_df,
                        bundle,
                        fold_reports,
                        min_profit_factor=min_profit_factor,
                        max_drawdown_limit=max_drawdown_limit,
                    )
                    model_selection_gate = model_report.get("model_selection_gate", {})
                    threshold_optimization = model_report.get("threshold_optimization", {})
                    threshold_best_gate = (
                        threshold_optimization.get("best", {})
                        .get("validation_trading_gate", {})
                        if isinstance(threshold_optimization, dict)
                        else {}
                    )
                    model_gate_passed = bool(model_selection_gate.get("passed", False)) if isinstance(model_selection_gate, dict) else False
                    threshold_gate_passed = bool(threshold_best_gate.get("passed", False)) if isinstance(threshold_best_gate, dict) else False
                    threshold_search_timed_out = (
                        bool(threshold_optimization.get("timed_out", False))
                        if isinstance(threshold_optimization, dict)
                        else True
                    )
                    holdout_threshold_summary = threshold_optimization_summary(threshold_optimization)
                    profit_quality_gate = profit_quality_gate_report(holdout_backtest)
                    holdout_degradation_gate = holdout_degradation_gate_report(
                        holdout_backtest,
                        fold_reports,
                        min_trades=min_trades,
                        selected_validation=holdout_threshold_summary,
                    )
                    gate_components = holdout_gate_components(
                        holdout_backtest,
                        cost_stress=cost_stress,
                        high_slippage_passed=high_slippage_passed,
                        model_gate_passed=model_gate_passed,
                        threshold_gate_passed=threshold_gate_passed,
                        threshold_search_timed_out=threshold_search_timed_out,
                        volatility_required=volatility_required,
                        volatility_regime=volatility_regime,
                        min_profit_factor=min_profit_factor,
                        min_trades=min_trades,
                        max_drawdown_limit=max_drawdown_limit,
                    )
                    gate_components["profit_quality_gate"] = bool(profit_quality_gate.get("passed", False))
                    gate_components["holdout_degradation_gate"] = bool(holdout_degradation_gate.get("passed", False))
                    gate_passed = promotion_gate_from_components(gate_components)
                    shadow_candidate_eligible, shadow_candidate_reason = shadow_candidate_from_components(gate_components)
                    promotion_blockers = gate_blockers_from_components(gate_components)
                    frozen_holdout.update(
                        {
                            "status": "completed",
                            "gate_passed": gate_passed,
                            "promotion_gate_passed": gate_passed,
                            "paper_candidate_gate_passed": gate_passed,
                            "shadow_candidate_eligible": shadow_candidate_eligible,
                            "shadow_observation_gate_passed": shadow_candidate_eligible,
                            "shadow_candidate_reason": shadow_candidate_reason,
                            "shadow_reasons": [shadow_candidate_reason] if shadow_candidate_reason else [],
                            "promotion_blockers": promotion_blockers,
                            "gate_components": gate_components,
                            "profit_quality_gate": profit_quality_gate,
                            "holdout_degradation_gate": holdout_degradation_gate,
                            "observe_only_hard_reason": (
                                "validation_holdout_sign_flip"
                                if bool(holdout_degradation_gate.get("validation_holdout_sign_flip", False))
                                else ("profit_quality_gate_failed" if not bool(profit_quality_gate.get("passed", False)) else "")
                            ),
                            "model_selection_required_for_promotion": True,
                            "model_selection_required_for_shadow": False,
                            "candidate_usage": "paper_candidate" if gate_passed else ("shadow_observation_only" if shadow_candidate_eligible else "observe_only"),
                            "model_name": bundle.model_name,
                            "rows_used": {
                                "train": len(final_train_df),
                                "valid": len(final_valid_df),
                                "holdout": len(frozen_holdout_df),
                            },
                            "metrics": holdout_metrics,
                            "backtest": holdout_backtest,
                            "cost_stress": cost_stress,
                            "high_slippage_gate_passed": high_slippage_passed,
                            "model_selection_gate_passed": model_gate_passed,
                            "threshold_validation_gate_passed": threshold_gate_passed,
                            "threshold_search_timed_out": threshold_search_timed_out,
                            "volatility_regime": volatility_regime,
                            "walk_forward_config_holdout_audit": walk_forward_config_audit,
                            "optimized_backtest_config": optimized_cfg,
                            "threshold_optimization": threshold_optimization,
                            "threshold_optimization_summary": holdout_threshold_summary,
                            "directional_auxiliary_gate": model_report.get("directional_auxiliary_gate", {}),
                            "directional_signal_report": model_report.get("directional_signal_report", {}),
                            "small_account_strategy": model_report.get("small_account_strategy", {}),
                            "test_large_move_metrics": model_report.get("test_large_move_metrics", {}),
                            "model_selection_gate": model_selection_gate,
                            "anti_overfit": model_report.get("anti_overfit", {}),
                            "winning_weight_variant": model_report.get("winning_weight_variant"),
                            "selected_risk_profile": model_report.get("small_account_strategy", {}).get("selected_risk_profile"),
                            "selected_side_policy": model_report.get("small_account_strategy", {}).get("selected_side_policy"),
                            "ranking_top": model_report.get("ranking", [])[:5],
                        }
                    )
                except Exception as exc:
                    frozen_holdout["status"] = "failed"
                    frozen_holdout["reason"] = str(exc)

    returns = [float(item["test_backtest"].get("total_return", 0.0)) for item in fold_reports]
    drawdowns = [float(item["test_backtest"].get("max_drawdown", 0.0)) for item in fold_reports]
    raw_profit_factors = [float(item["test_backtest"].get("profit_factor", 0.0)) for item in fold_reports]
    profit_factors = [capped_profit_factor(value) for value in raw_profit_factors]
    realized_rr = [float(item["test_backtest"].get("realized_avg_win_loss_ratio", 0.0)) for item in fold_reports]
    balanced = [float(item["test_metrics"].get("balanced_accuracy", 0.0)) for item in fold_reports]
    fee_drags = [
        float(
            item["test_backtest"].get(
                "total_cost_drag",
                float(item["test_backtest"].get("fee_drag", 0.0))
                + float(item["test_backtest"].get("funding_drag", 0.0)),
            )
        )
        for item in fold_reports
    ]
    avg_exposures = [float(item["test_backtest"].get("avg_exposure", 0.0)) for item in fold_reports]
    active_avg_exposures = [float(item["test_backtest"].get("active_avg_exposure", 0.0)) for item in fold_reports]
    active_avg_position_fractions = [
        float(item["test_backtest"].get("active_avg_position_fraction", 0.0))
        for item in fold_reports
    ]
    notional_turnovers = [float(item["test_backtest"].get("notional_turnover", 0.0)) for item in fold_reports]
    long_trade_counts = [float(item["test_backtest"].get("long_trades", 0.0)) for item in fold_reports]
    short_trade_counts = [float(item["test_backtest"].get("short_trades", 0.0)) for item in fold_reports]
    expectancies = [float(item["test_backtest"].get("expectancy_after_cost", 0.0)) for item in fold_reports]
    per_trade_expectancies = [
        float(item["test_backtest"].get("expectancy_per_trade_after_cost", 0.0))
        for item in fold_reports
    ]
    trades = [float(item["test_backtest"].get("trades", 0.0)) for item in fold_reports]
    stress_pass_rates = [float(item.get("cost_stress", {}).get("pass_rate", 0.0)) for item in fold_reports]
    volatility_regime_required = bool(interval_requires_extra_publish_gates(interval))
    volatility_regime_passes = [
        bool(item.get("volatility_regime", {}).get("gate_passed", False))
        for item in fold_reports
    ]
    volatility_regime_pass_rate = (
        float(sum(1 for value in volatility_regime_passes if value) / len(volatility_regime_passes))
        if volatility_regime_passes
        else 0.0
    )
    required_volatility_regime_pass_rate = 0.60 if volatility_regime_required else 0.0
    per_regime_passes: dict[str, list[bool]] = {}
    regime_profit_factors: list[float] = []
    regime_drawdowns: list[float] = []
    regime_insufficient_trade_count = 0
    for item in fold_reports:
        regime_report = item.get("volatility_regime", {}) if isinstance(item.get("volatility_regime"), dict) else {}
        for regime_item in regime_report.get("items", []) or []:
            if not isinstance(regime_item, dict):
                continue
            name = str(regime_item.get("name") or "unknown")
            per_regime_passes.setdefault(name, []).append(bool(regime_item.get("passed", False)))
            if not bool(regime_item.get("enough_trades", False)):
                regime_insufficient_trade_count += 1
            metrics = regime_item.get("metrics", {}) if isinstance(regime_item.get("metrics"), dict) else {}
            regime_profit_factors.append(float(metrics.get("profit_factor", 0.0)))
            regime_drawdowns.append(float(metrics.get("max_drawdown", 0.0)))
    per_regime_pass_rate = {
        name: float(sum(1 for value in values if value) / len(values)) if values else 0.0
        for name, values in per_regime_passes.items()
    }
    high_slippage_passes = [
        bool(
            next(
                (
                    stress_item.get("passed", False)
                    for stress_item in item.get("cost_stress", {}).get("items", [])
                    if stress_item.get("name") == "high_slippage"
                ),
                False,
            )
        )
        for item in fold_reports
    ]
    high_slippage_pass_rate = float(sum(1 for value in high_slippage_passes if value) / len(high_slippage_passes)) if high_slippage_passes else 0.0
    model_selection_passes = [
        bool(item.get("model_selection_gate", {}).get("passed", False))
        for item in fold_reports
    ]
    model_selection_pass_rate = (
        float(sum(1 for value in model_selection_passes if value) / len(model_selection_passes))
        if model_selection_passes
        else 0.0
    )
    profitable_fold_rate = float(sum(1 for value in returns if value > 0.0) / len(returns))
    acceptable_pf_rate = float(sum(1 for value in raw_profit_factors if value >= min_profit_factor) / len(raw_profit_factors))
    acceptable_rr_rate = float(sum(1 for value in realized_rr if value >= 0.90) / len(realized_rr)) if realized_rr else 0.0
    worst_drawdown = min(drawdowns) if drawdowns else 0.0
    min_fold_trades = min(trades) if trades else 0.0
    frozen_holdout_required = bool(frozen_holdout.get("required_for_eligible", False))
    frozen_holdout_gate_passed = bool(frozen_holdout.get("gate_passed", False)) if frozen_holdout_required else True
    frozen_shadow_candidate_eligible = bool(frozen_holdout.get("shadow_candidate_eligible", False))
    shadow_candidate_eligible = bool(
        frozen_holdout_required
        and not frozen_holdout_gate_passed
        and frozen_shadow_candidate_eligible
        and profitable_fold_rate >= 0.50
        and acceptable_pf_rate >= 0.50
        and acceptable_rr_rate >= 0.50
        and mean(stress_pass_rates) >= 0.50
        and high_slippage_pass_rate >= 0.50
        and min_fold_trades >= min_trades
        and worst_drawdown >= -abs(max_drawdown_limit)
        and median(returns) > 0.0
        and median(per_trade_expectancies) > 0.0
    )
    stability_score = (
        1.8 * median(returns)
        + 12.0 * median(expectancies)
        + 35.0 * median(per_trade_expectancies)
        + 0.03 * profitable_fold_rate
        + 0.02 * min(mean(profit_factors), 2.0)
        + 0.005 * (mean(balanced) - 0.5)
        - 0.20 * abs(worst_drawdown)
        - 0.20 * mean(fee_drags)
    )
    if frozen_holdout_required and not frozen_holdout_gate_passed:
        stability_score -= 0.05
    eligible = bool(
        profitable_fold_rate >= 0.50
        and acceptable_pf_rate >= 0.50
        and acceptable_rr_rate >= 0.50
        and mean(stress_pass_rates) >= 0.50
        and high_slippage_pass_rate >= 0.50
        and model_selection_pass_rate >= 0.50
        and (not volatility_regime_required or volatility_regime_pass_rate >= required_volatility_regime_pass_rate)
        and frozen_holdout_gate_passed
        and min_fold_trades >= min_trades
        and worst_drawdown >= -abs(max_drawdown_limit)
        and median(returns) > 0.0
        and median(per_trade_expectancies) > 0.0
    )
    profile_promotion_allowed = bool(promotion_allowed)
    promotion_disabled_reason = "" if profile_promotion_allowed else "validation_profile_research_only"
    if not profile_promotion_allowed:
        eligible = False
        shadow_candidate_eligible = False
        frozen_holdout["profile_promotion_allowed"] = False
        frozen_holdout["profile_promotion_disabled_reason"] = promotion_disabled_reason
        frozen_holdout["promotion_gate_passed"] = False
        frozen_holdout["paper_candidate_gate_passed"] = False
        frozen_holdout["shadow_candidate_eligible"] = False
        frozen_holdout["shadow_observation_gate_passed"] = False
        frozen_holdout["candidate_usage"] = "research_only_profile_screen"
    if eligible:
        recommendation = "continue_paper_candidate"
        candidate_tier = "paper_candidate"
    elif shadow_candidate_eligible:
        recommendation = "shadow_observation_holdout_positive_not_promoted"
        candidate_tier = "shadow_observation"
    elif frozen_holdout_required and not frozen_holdout_gate_passed:
        recommendation = "observe_only_frozen_holdout_not_passed"
        candidate_tier = "observe_only"
    elif profitable_fold_rate == 0.0 or mean(profit_factors) < min_profit_factor:
        recommendation = "do_not_promote_needs_feature_or_label_work"
        candidate_tier = "failed_candidate"
    else:
        recommendation = "observe_only_needs_more_stability"
        candidate_tier = "observe_only"
    if promotion_disabled_reason:
        recommendation = "observe_only_validation_profile_research_only"
        candidate_tier = "observe_only"
    observe_only_hard_reason = ""
    if promotion_disabled_reason:
        observe_only_hard_reason = promotion_disabled_reason
    if candidate_tier == "observe_only" and frozen_holdout_required and not frozen_holdout_gate_passed:
        observe_only_hard_reason = observe_only_hard_reason or str(
            frozen_holdout.get("observe_only_hard_reason") or "frozen_holdout_not_passed"
        )
    effective_rank_score = float(stability_score)
    if candidate_tier == "paper_candidate":
        effective_rank_score += 1.0
    elif candidate_tier == "shadow_observation":
        effective_rank_score += 0.25
    elif frozen_holdout_required and not frozen_holdout_gate_passed:
        effective_rank_score = min(effective_rank_score, 0.0)
    profit_diagnostics = profit_bottleneck_report(
        mean_return_value=mean(returns),
        mean_fee_drag_value=mean(fee_drags),
        mean_balanced_accuracy_value=mean(balanced),
        mean_trades_value=mean(trades),
        min_required_trades=min_trades,
        mean_exposure_value=mean(avg_exposures),
        mean_active_exposure_value=mean(active_avg_exposures),
        mean_active_position_fraction_value=mean(active_avg_position_fractions),
        total_long_trades=sum(long_trade_counts),
        total_short_trades=sum(short_trade_counts),
        mean_turnover_value=mean(notional_turnovers),
    )
    no_trade_fold_present = any(float(item["test_backtest"].get("trades", 0.0)) <= 0.0 for item in fold_reports)
    research_candidate = research_candidate_report(
        frozen_holdout,
        profitable_fold_rate=profitable_fold_rate,
        acceptable_profit_factor_rate=acceptable_pf_rate,
        median_return_value=median(returns),
        mean_profit_factor_value=mean(profit_factors),
        worst_drawdown_value=worst_drawdown,
        mean_cost_stress_pass_rate=mean(stress_pass_rates),
        high_slippage_pass_rate=high_slippage_pass_rate,
        model_selection_pass_rate=model_selection_pass_rate,
        min_fold_trades=min_fold_trades,
        no_trade_fold_present=no_trade_fold_present,
        fee_drag_to_return_ratio=_payload_float(profit_diagnostics, "fee_drag_to_return_ratio"),
        max_drawdown_limit=max_drawdown_limit,
    )
    research_status = str(research_candidate.get("research_status") or "none")
    research_usage = str(research_candidate.get("usage") or "offline_diagnostics_only")
    research_eligible = research_status == "eligible"
    research_watchlist_allowed = bool(research_candidate.get("watchlist_allowed", False))
    research_reason = (
        "research_quality_passed_but_promotion_blocked"
        if research_eligible
        else (
            "research_watchlist_offline_optimization_only"
            if research_status == "watch"
            else ("research_blocked_offline_diagnostics_only" if research_status == "blocked" else "")
        )
    )

    return {
        "created_beijing": beijing_now_iso(),
        "symbol": symbol,
        "interval": interval,
        "feature_version": FEATURE_VERSION,
        "include_realtime": include_realtime,
        "rows": len(frame),
        "raw_rows": len(raw),
        "raw_rows_before_window": raw_rows_before_window,
        "data_validation": data_validation,
        "market_context": market_context,
        "max_training_rows": window_rows if window_rows > 0 else None,
        "training_window_applied": training_window_applied,
        "validation_profile": validation_profile,
        "validation_profile_promotion_allowed": profile_promotion_allowed,
        "validation_profile_promotion_disabled_reason": promotion_disabled_reason,
        "walk_forward_row_caps": {
            "train_rows": effective_wf_train_rows,
            "valid_rows": effective_wf_valid_rows,
            "test_rows": effective_wf_test_rows,
            "applied": any(value is not None for value in [effective_wf_train_rows, effective_wf_valid_rows, effective_wf_test_rows]),
        },
        "threshold_search_controls": {
            "force_compact": effective_force_compact_threshold_search,
            "max_threshold_evals": effective_max_threshold_evals,
        },
        "primary_label_horizon": label_horizon,
        "primary_label_min_return": label_min_return,
        "configured_label_min_return": cfg.label_min_return,
        "purged_validation": {
            "enabled": effective_purge_rows > 0,
            "purge_rows": effective_purge_rows,
            "purge_source": "label_horizon_or_user_override",
            "trainable_rows_before_frozen_holdout": len(validation_frame),
            "frozen_holdout_rows": len(frozen_holdout_df),
        },
        "folds_requested": folds,
        "folds_completed": len(fold_reports),
        "skipped_folds": skipped_folds,
        "complexity": complexity,
        "rolling_folds": rolling_folds,
        "max_model_trials": max_model_trials,
        "objective": "walk_forward_strategy_validation",
        "budget_allocation": budget_allocation,
        "small_account_strategy": {
            "enabled": True,
            "margin_type": "ISOLATED",
            "position_sizing": "confidence_atr_liquidity_scaled",
            "live_trading_enabled": False,
        },
        "summary": {
            "eligible_for_paper_candidate": eligible,
            "candidate_tier": candidate_tier,
            "recommendation": recommendation,
            "shadow_candidate_eligible": shadow_candidate_eligible,
            "paper_candidate_gate_passed": eligible,
            "shadow_observation_gate_passed": shadow_candidate_eligible,
            "shadow_candidate_reason": frozen_holdout.get("shadow_candidate_reason", ""),
            "shadow_reasons": frozen_holdout.get("shadow_reasons", []),
            "research_candidate_eligible": research_eligible,
            "research_status": research_status,
            "research_candidate_usage": research_usage,
            "research_usage": research_usage,
            "research_candidate_reason": research_reason,
            "research_watchlist_allowed": research_watchlist_allowed,
            "research_promotion_allowed": False,
            "research_shadow_allowed": False,
            "research_blockers": research_candidate.get("blockers", []),
            "research_reasons": research_candidate.get("reasons", []),
            "research_candidate": research_candidate,
            "promotion_blockers": frozen_holdout.get("promotion_blockers", []),
            "gate_components": frozen_holdout.get("gate_components", {}),
            "stability_score": float(stability_score),
            "effective_rank_score": float(effective_rank_score),
            "observe_only_hard_reason": observe_only_hard_reason,
            "profitable_fold_rate": profitable_fold_rate,
            "acceptable_profit_factor_rate": acceptable_pf_rate,
            "acceptable_realized_risk_reward_rate": acceptable_rr_rate,
            "median_return": median(returns),
            "mean_return": mean(returns),
            "median_expectancy_after_cost": median(expectancies),
            "mean_expectancy_after_cost": mean(expectancies),
            "median_expectancy_per_trade_after_cost": median(per_trade_expectancies),
            "mean_expectancy_per_trade_after_cost": mean(per_trade_expectancies),
            "worst_drawdown": float(worst_drawdown),
            "mean_profit_factor": mean(profit_factors),
            "mean_realized_avg_win_loss_ratio": mean(realized_rr),
            "required_min_realized_avg_win_loss_ratio": 0.90,
            "mean_balanced_accuracy": mean(balanced),
            "mean_trades": mean(trades),
            "min_trades": min_fold_trades,
            "required_min_trades_per_fold": min_trades,
            "mean_fee_drag": mean(fee_drags),
            "mean_total_cost_drag": mean(fee_drags),
            "mean_avg_exposure": mean(avg_exposures),
            "mean_active_avg_exposure": mean(active_avg_exposures),
            "mean_active_avg_position_fraction": mean(active_avg_position_fractions),
            "mean_notional_turnover": mean(notional_turnovers),
            "total_long_trades": sum(long_trade_counts),
            "total_short_trades": sum(short_trade_counts),
            "profit_diagnostics": profit_diagnostics,
            "mean_cost_stress_pass_rate": mean(stress_pass_rates),
            "high_slippage_pass_rate": high_slippage_pass_rate,
            "model_selection_pass_rate": model_selection_pass_rate,
            "frozen_holdout_required": frozen_holdout_required,
            "frozen_holdout_status": frozen_holdout.get("status"),
            "frozen_holdout_gate_passed": frozen_holdout_gate_passed,
            "frozen_holdout_promotion_gate_passed": bool(frozen_holdout.get("promotion_gate_passed", frozen_holdout_gate_passed)),
            "frozen_holdout_shadow_candidate_eligible": frozen_shadow_candidate_eligible,
            "frozen_holdout_gate_components": frozen_holdout.get("gate_components", {}),
            "frozen_holdout_profit_quality_gate": frozen_holdout.get("profit_quality_gate", {}),
            "frozen_holdout_degradation_gate": frozen_holdout.get("holdout_degradation_gate", {}),
            "volatility_regime_pass_rate": volatility_regime_pass_rate,
            "volatility_regime_per_bucket_pass_rate": per_regime_pass_rate,
            "worst_regime_profit_factor": min(regime_profit_factors) if regime_profit_factors else 0.0,
            "worst_regime_drawdown": min(regime_drawdowns) if regime_drawdowns else 0.0,
            "regime_insufficient_trade_count": regime_insufficient_trade_count,
            "volatility_regime_hard_gate_required": volatility_regime_required,
            "required_volatility_regime_pass_rate": required_volatility_regime_pass_rate,
            "cost_stress_required": True,
            "model_selection_hard_gate": True,
            "min_trades_hard_gate": True,
            "high_slippage_hard_gate": True,
            "volatility_regime_hard_gate": volatility_regime_required,
            "fold_independence": {**fold_independence, "fold_count": len(fold_reports)},
        },
        "folds": fold_reports,
        "frozen_holdout": frozen_holdout,
        "live_trading_enabled": False,
    }


def run_strategy_validation(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    intervals: list[str],
    include_realtime: bool,
    folds: int | None,
    max_model_trials: int | None,
    time_budget_minutes: float,
    initial_balance: float,
    min_trades: int,
    max_drawdown_limit: float,
    min_profit_factor: float,
    complexity: str,
    rolling_folds: int | None,
    purge_rows: int | None = None,
    holdout_fraction: float = 0.15,
    max_training_rows: int | None = None,
    validation_profile: str = "standard",
    wf_train_rows: int | None = None,
    wf_valid_rows: int | None = None,
    wf_test_rows: int | None = None,
    max_threshold_evals: int | None = None,
    per_target_budget_minutes: float | None = None,
    state_dir: str | Path = "state",
) -> dict[str, Any]:
    cfg.ensure_dirs()
    profile = validation_profile_or_default(validation_profile)
    legacy_default_folds = 3
    legacy_default_model_trials = 4
    legacy_default_rolling_folds = 1

    effective_folds = int(folds if folds is not None else legacy_default_folds)
    effective_max_model_trials = int(max_model_trials if max_model_trials is not None else legacy_default_model_trials)
    effective_rolling_folds = int(rolling_folds if rolling_folds is not None else legacy_default_rolling_folds)
    if profile.name != "standard":
        if profile.folds is not None and (folds is None or int(folds) == legacy_default_folds):
            effective_folds = int(profile.folds)
        if profile.max_model_trials is not None and (
            max_model_trials is None or int(max_model_trials) == legacy_default_model_trials
        ):
            effective_max_model_trials = int(profile.max_model_trials)
        if profile.rolling_folds is not None and (
            rolling_folds is None or int(rolling_folds) == legacy_default_rolling_folds
        ):
            effective_rolling_folds = int(profile.rolling_folds)
    effective_wf_train_rows = positive_int_or_none(wf_train_rows) or profile.wf_train_rows
    effective_wf_valid_rows = positive_int_or_none(wf_valid_rows) or profile.wf_valid_rows
    effective_wf_test_rows = positive_int_or_none(wf_test_rows) or profile.wf_test_rows
    effective_max_threshold_evals = positive_int_or_none(max_threshold_evals) or profile.max_threshold_evals
    effective_per_target_budget_minutes = (
        float(per_target_budget_minutes)
        if per_target_budget_minutes is not None and float(per_target_budget_minutes) > 0
        else profile.per_target_budget_minutes
    )
    effective_force_compact_threshold_search = bool(profile.force_compact_threshold_search)
    effective_promotion_allowed = bool(profile.promotion_allowed)

    deadline = time.monotonic() + max(1.0, time_budget_minutes * 60.0)
    run_stamp = beijing_stamp()
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for symbol in [item.upper() for item in symbols]:
        for interval in intervals:
            if time.monotonic() >= deadline:
                skipped.append({"symbol": symbol, "interval": interval, "reason": "time_budget_exhausted"})
                continue
            target_deadline = deadline
            if effective_per_target_budget_minutes is not None:
                target_deadline = min(deadline, time.monotonic() + max(1.0, effective_per_target_budget_minutes * 60.0))
            try:
                report = validate_symbol_interval(
                    symbol,
                    interval,
                    cfg,
                    include_realtime=include_realtime,
                    folds=effective_folds,
                    max_model_trials=effective_max_model_trials,
                    deadline=target_deadline,
                    initial_balance=initial_balance,
                    min_trades=min_trades,
                    max_drawdown_limit=max_drawdown_limit,
                    min_profit_factor=min_profit_factor,
                    complexity=complexity,
                    rolling_folds=effective_rolling_folds,
                    purge_rows=purge_rows,
                    holdout_fraction=holdout_fraction,
                    max_training_rows=max_training_rows,
                    validation_profile=profile.name,
                    wf_train_rows=effective_wf_train_rows,
                    wf_valid_rows=effective_wf_valid_rows,
                    wf_test_rows=effective_wf_test_rows,
                    max_threshold_evals=effective_max_threshold_evals,
                    force_compact_threshold_search=effective_force_compact_threshold_search,
                    promotion_allowed=effective_promotion_allowed,
                )
                per_target_latest = cfg.reports_dir / f"{symbol}_{interval}_strategy_validation.json"
                per_target_snapshot = cfg.reports_dir / f"{symbol}_{interval}_strategy_validation_{run_stamp}.json"
                report_payload = json.dumps(report, indent=2, ensure_ascii=False)
                safe_replace_text(per_target_latest, report_payload)
                safe_replace_text(per_target_snapshot, report_payload)
                report["report_path"] = str(per_target_snapshot)
                report["latest_target_report_path"] = str(per_target_latest)
                items.append(report)
            except Exception as exc:
                skipped.append({"symbol": symbol, "interval": interval, "reason": str(exc)})

    tier_rank = {"paper_candidate": 3, "shadow_observation": 2, "observe_only": 1, "failed_candidate": 0}

    def target_rank_key(item: dict[str, Any]) -> tuple[int, bool, bool, float]:
        summary = item.get("summary", {}) if isinstance(item, dict) else {}
        tier = str(summary.get("candidate_tier", "observe_only"))
        return (
            tier_rank.get(tier, 0),
            bool(summary.get("frozen_holdout_gate_passed", False)),
            bool(summary.get("shadow_observation_gate_passed", False)),
            float(summary.get("effective_rank_score", summary.get("stability_score", 0.0)) or 0.0),
        )

    ranked = sorted(items, key=target_rank_key, reverse=True)
    eligible = [item for item in ranked if item["summary"]["eligible_for_paper_candidate"]]
    shadow_candidates = [
        item
        for item in ranked
        if item["summary"].get("shadow_candidate_eligible", False)
        and not item["summary"].get("eligible_for_paper_candidate", False)
    ]
    research_candidates = [
        item
        for item in ranked
        if item["summary"].get("research_status") == "eligible"
        and not item["summary"].get("eligible_for_paper_candidate", False)
        and not item["summary"].get("shadow_candidate_eligible", False)
    ]
    research_watchlist = [
        item
        for item in ranked
        if item["summary"].get("research_status") == "watch"
        and not item["summary"].get("eligible_for_paper_candidate", False)
        and not item["summary"].get("shadow_candidate_eligible", False)
    ]
    research_blocked = [
        item
        for item in ranked
        if item["summary"].get("research_status") == "blocked"
        and not item["summary"].get("eligible_for_paper_candidate", False)
        and not item["summary"].get("shadow_candidate_eligible", False)
    ]
    failed_candidates = [
        item
        for item in ranked
        if not item["summary"].get("eligible_for_paper_candidate", False)
        and not item["summary"].get("shadow_candidate_eligible", False)
        and not item["summary"].get("research_candidate_eligible", False)
        and item["summary"].get("research_status") not in {"watch", "blocked"}
    ]
    aggregate = {
        "created_beijing": beijing_now_iso(),
        "objective": "walk_forward_strategy_validation",
        "feature_version": FEATURE_VERSION,
        "symbols": [item.upper() for item in symbols],
        "intervals": intervals,
        "include_realtime": include_realtime,
        "validation_profile": profile.name,
        "validation_profile_config": asdict(profile),
        "folds": effective_folds,
        "max_model_trials": effective_max_model_trials,
        "time_budget_minutes": time_budget_minutes,
        "per_target_budget_minutes": effective_per_target_budget_minutes,
        "complexity": complexity,
        "rolling_folds": effective_rolling_folds,
        "purge_rows": purge_rows,
        "holdout_fraction": holdout_fraction,
        "max_training_rows": int(max_training_rows or 0) or None,
        "walk_forward_row_caps": {
            "train_rows": effective_wf_train_rows,
            "valid_rows": effective_wf_valid_rows,
            "test_rows": effective_wf_test_rows,
        },
        "threshold_search_controls": {
            "force_compact": effective_force_compact_threshold_search,
            "max_threshold_evals": effective_max_threshold_evals,
        },
        "promotion_allowed": effective_promotion_allowed,
        "items": items,
        "ranked": ranked,
        "eligible": eligible,
        "shadow_candidates": shadow_candidates,
        "research_candidates": research_candidates,
        "research_watchlist": research_watchlist,
        "research_blocked": research_blocked,
        "failed_candidates": failed_candidates,
        "skipped_targets": skipped,
        "live_trading_enabled": False,
    }
    output = cfg.reports_dir / f"strategy_validation_{run_stamp}.json"
    latest = cfg.reports_dir / "strategy_validation_latest.json"
    safe_replace_text(output, json.dumps(aggregate, indent=2, ensure_ascii=False))
    safe_replace_text(latest, json.dumps(aggregate, indent=2, ensure_ascii=False))
    aggregate["report_path"] = str(output)
    aggregate["latest_report_path"] = str(latest)

    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    strategy_state = {
        "updated_beijing": beijing_now_iso(),
        "feature_version": FEATURE_VERSION,
        "strategy": "v4_profit_first_shadow_observation",
        "anti_overfit_validation": {
            "purge_rows": purge_rows,
            "holdout_fraction": holdout_fraction,
            "frozen_holdout_required": True,
        },
        "promotion_status": "paper_candidate" if eligible else ("shadow_observation" if shadow_candidates else "observe_only"),
        "candidate_counts": {
            "paper": len(eligible),
            "shadow": len(shadow_candidates),
            "research": len(research_candidates),
            "research_watch": len(research_watchlist),
            "research_blocked": len(research_blocked),
            "failed": len(failed_candidates),
        },
        "eligible_targets": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in eligible
        ],
        "paper_targets": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in eligible
        ],
        "shadow_targets": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in shadow_candidates
        ],
        "research_targets": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in research_candidates
        ],
        "research_watchlist": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in research_watchlist
        ],
        "research_blocked": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in research_blocked
        ],
        "failed_targets": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "summary": item["summary"],
                "report_path": item.get("report_path"),
            }
            for item in failed_candidates
        ],
        "best_shadow_target": (
            {
                "symbol": shadow_candidates[0]["symbol"],
                "interval": shadow_candidates[0]["interval"],
                "summary": shadow_candidates[0]["summary"],
                "report_path": shadow_candidates[0].get("report_path"),
            }
            if shadow_candidates
            else None
        ),
        "best_research_target": (
            {
                "symbol": research_candidates[0]["symbol"],
                "interval": research_candidates[0]["interval"],
                "summary": research_candidates[0]["summary"],
                "report_path": research_candidates[0].get("report_path"),
            }
            if research_candidates
            else None
        ),
        "best_research_watch": (
            {
                "symbol": research_watchlist[0]["symbol"],
                "interval": research_watchlist[0]["interval"],
                "summary": research_watchlist[0]["summary"],
                "report_path": research_watchlist[0].get("report_path"),
            }
            if research_watchlist
            else None
        ),
        "best_target": (
            {
                "symbol": ranked[0]["symbol"],
                "interval": ranked[0]["interval"],
                "summary": ranked[0]["summary"],
                "report_path": ranked[0].get("report_path"),
            }
            if ranked
            else None
        ),
        "live_trading_enabled": False,
        "source_report": str(output),
    }
    state_file = state_path / "local_strategy_state.json"
    should_write_state = True
    current_best_status = (
        ((strategy_state.get("best_target") or {}).get("summary") or {}).get("frozen_holdout_status")
        if isinstance(strategy_state.get("best_target"), dict)
        else None
    )
    current_best_summary = (
        ((strategy_state.get("best_target") or {}).get("summary") or {})
        if isinstance(strategy_state.get("best_target"), dict)
        else {}
    )
    current_fold_count = summary_fold_count(current_best_summary)
    if state_file.exists():
        try:
            previous_state = json.loads(state_file.read_text(encoding="utf-8"))
            previous_best_summary = (
                ((previous_state.get("best_target") or {}).get("summary") or {})
                if isinstance(previous_state.get("best_target"), dict)
                else {}
            )
            previous_best_status = (
                previous_best_summary.get("frozen_holdout_status")
                if isinstance(previous_best_summary, dict)
                else None
            )
            if previous_best_status == "completed":
                previous_fold_count = summary_fold_count(previous_best_summary)
                preserve_reason = ""
                if current_best_status != "completed":
                    preserve_reason = "preserved_previous_completed_frozen_holdout_state"
                elif current_fold_count < previous_fold_count:
                    preserve_reason = "preserved_previous_stronger_fold_count_state"
                if preserve_reason:
                    previous_state["last_lower_strength_validation"] = {
                        "updated_beijing": beijing_now_iso(),
                        "source_report": str(output),
                        "best_frozen_holdout_status": current_best_status,
                        "current_fold_count": current_fold_count,
                        "previous_fold_count": previous_fold_count,
                        "reason": preserve_reason,
                    }
                    safe_replace_text(state_file, json.dumps(previous_state, indent=2, ensure_ascii=False))
                    should_write_state = False
        except Exception:
            should_write_state = True
    if should_write_state:
        safe_replace_text(state_file, json.dumps(strategy_state, indent=2, ensure_ascii=False))
    aggregate["local_strategy_state_path"] = str(state_path / "local_strategy_state.json")
    return aggregate
