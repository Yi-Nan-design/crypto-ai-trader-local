from __future__ import annotations

from dataclasses import asdict
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alpha_models import (
    ALPHA_MODEL_INTERFACE_VERSION,
    build_lightgbm_classifier_alpha,
    lightgbm_ranker_availability_report,
    train_lightgbm_expected_return,
)
from .backtest import BacktestConfig, BacktestResult, auxiliary_signal_threshold, effective_thresholds, run_backtest
from .binance_data import load_symbol_interval, resolve_exchange_rule_values
from .config import TraderConfig
from .features import FEATURE_VERSION, MULTI_HORIZON_STEPS, feature_matrix, make_features, time_split
from .models import (
    EnsembleProbabilityModel,
    LogisticRegressionNumpy,
    MLPClassifierNumpy,
    ModelBundle,
    SklearnModelAdapter,
    StandardScaler,
    classification_metrics,
    optional_neural_candidates,
)
from .model_selection import rank_model_candidates
from .progress import safe_replace_text
from .shadow_learning import select_shadow_threshold_candidate
from .strategy_calibration import (
    archetype_matches_side,
    archetype_side_robustness,
    build_strategy_calibration_search_space,
    calibrate_directional_thresholds,
    cost_efficiency_ratio,
    directional_side_preflight_gate,
    directional_signal_capture_metrics,
    fee_drag_to_abs_return_ratio,
    long_side_preflight_gate,
    large_move_cutoff,
    normalize_trade_side_policy,
    profit_quality_penalty,
    short_side_preflight_gate,
    side_contribution_gate,
    side_policy_allows,
    small_account_risk_profiles,
    split_validation_for_strategy_calibration,
    total_cost_drag_value,
    trade_side_policy_grid,
    trading_filter_penalty,
    validation_trading_gate,
)
from .strategy_config import primary_label_horizon, primary_label_min_return
from .time_utils import beijing_now_iso, beijing_stamp


def accuracy_first_score(metrics: dict[str, float]) -> float:
    return float(
        metrics.get("balanced_accuracy", 0.0)
        + 0.10 * metrics.get("auc", 0.5)
        - 0.05 * metrics.get("log_loss", 1.0)
    )


def return_first_score(
    metrics: dict[str, float],
    result: BacktestResult,
    large_metrics: dict[str, float],
) -> float:
    expectancy_per_trade = float(getattr(result, "expectancy_per_trade_after_cost", 0.0))
    expectancy_active_bar = float(getattr(result, "expectancy_after_cost", 0.0))
    profit_factor_bonus = 0.025 * float(np.log1p(min(max(float(result.profit_factor), 0.0), 4.0)))
    return float(
        4.75 * float(result.total_return)
        + 65.0 * expectancy_per_trade
        + 18.0 * expectancy_active_bar
        + profit_factor_bonus
        + 0.060 * float(large_metrics.get("large_move_capture", 0.0))
        + 0.010 * float(metrics.get("balanced_accuracy", 0.0))
        - 0.25 * abs(float(result.max_drawdown))
        - 0.35 * min(total_cost_drag_value(result), 0.25)
        - profit_quality_penalty(result)
    )


def choose_primary_target_col(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    objective: str,
    requested: str | None = None,
) -> str:
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
    if objective == "return":
        candidates.extend(["edge_trade_target", "edge_long_target", "big_move_target", "big_up_target", "long_target", "target"])
    else:
        candidates.append("target")
    for col in candidates:
        if col not in train_df.columns or col not in valid_df.columns:
            continue
        train_y = train_df[col].to_numpy(dtype=int)
        valid_y = valid_df[col].to_numpy(dtype=int)
        if int(train_y.sum()) >= 8 and int((1 - train_y).sum()) >= 8 and int(valid_y.sum()) >= 3 and int((1 - valid_y).sum()) >= 3:
            return col
    return "target"




def large_move_sample_weight(frame: pd.DataFrame, label_min_return: float) -> np.ndarray:
    return large_move_sample_weight_for_returns(frame["future_return"], label_min_return)


def event_balanced_sample_weight(frame: pd.DataFrame, label_min_return: float) -> tuple[np.ndarray, dict[str, Any]]:
    weights = large_move_sample_weight(frame, label_min_return)
    returns = pd.to_numeric(frame["future_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    actionable = np.abs(returns) >= (2.0 * abs(float(label_min_return)))
    neutral = np.abs(returns) < abs(float(label_min_return))
    actionable_rate = float(actionable.mean()) if len(actionable) else 0.0
    neutral_rate = float(neutral.mean()) if len(neutral) else 0.0
    target_actionable_rate = 0.12
    event_multiplier = 1.0
    neutral_multiplier = 1.0
    enabled = bool(0.0 < actionable_rate < target_actionable_rate)
    if enabled:
        event_multiplier = min(5.0, max(1.0, target_actionable_rate / max(actionable_rate, 1e-9)))
        neutral_multiplier = 0.75
        weights = weights.copy()
        weights[actionable] *= event_multiplier
        weights[neutral] *= neutral_multiplier
        mean = float(weights.mean()) or 1.0
        weights = np.clip(weights / mean, 0.15, 12.0)
    report = {
        "enabled": enabled,
        "method": "actionable_event_weight_boost",
        "selection_data": "train_only",
        "label_min_return": float(label_min_return),
        "actionable_threshold": float(2.0 * abs(float(label_min_return))),
        "actionable_rate": actionable_rate,
        "neutral_rate": neutral_rate,
        "target_actionable_rate": target_actionable_rate,
        "event_multiplier": float(event_multiplier),
        "neutral_multiplier": float(neutral_multiplier),
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_max": float(np.max(weights)) if len(weights) else 0.0,
    }
    return weights, report


def volatility_regime_event_sample_weight(frame: pd.DataFrame, label_min_return: float) -> tuple[np.ndarray, dict[str, Any]]:
    weights = large_move_sample_weight(frame, label_min_return)
    returns = pd.to_numeric(frame["future_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    threshold = 2.0 * abs(float(label_min_return))
    actionable = np.abs(returns) >= threshold
    neutral = np.abs(returns) < abs(float(label_min_return))
    actionable_rate = float(actionable.mean()) if len(actionable) else 0.0
    target_actionable_rate = 0.14
    min_regime_rows = 80
    max_overall_actionable_rate = 0.30
    volatility_col = "atr_14" if "atr_14" in frame.columns else "volatility_24" if "volatility_24" in frame.columns else ""
    regimes: list[dict[str, Any]] = []
    enabled = False
    if volatility_col and len(frame) >= min_regime_rows * 3 and actionable_rate < max_overall_actionable_rate:
        volatility = pd.to_numeric(frame[volatility_col], errors="coerce")
        if volatility.notna().sum() >= min_regime_rows * 3:
            filled = volatility.fillna(float(volatility.median()))
            try:
                regime_ids = pd.qcut(filled, q=3, labels=False, duplicates="drop").to_numpy()
            except ValueError:
                regime_ids = np.zeros(len(frame), dtype=int)
            unique_regimes = [int(value) for value in sorted(pd.Series(regime_ids).dropna().unique())]
            if len(unique_regimes) >= 2:
                weights = weights.copy()
                for regime_id in unique_regimes:
                    mask = regime_ids == regime_id
                    rows = int(mask.sum())
                    if rows < min_regime_rows:
                        continue
                    regime_actionable_rate = float(actionable[mask].mean()) if rows else 0.0
                    regime_neutral_rate = float(neutral[mask].mean()) if rows else 0.0
                    event_multiplier = 1.0
                    neutral_multiplier = 1.0
                    boosted = bool(0.0 < regime_actionable_rate < target_actionable_rate)
                    if boosted:
                        event_multiplier = min(4.0, max(1.0, target_actionable_rate / max(regime_actionable_rate, 1e-9)))
                        neutral_multiplier = 0.82 if regime_neutral_rate > 0.25 else 0.90
                        weights[mask & actionable] *= event_multiplier
                        weights[mask & neutral] *= neutral_multiplier
                        enabled = True
                    regimes.append(
                        {
                            "regime": regime_id,
                            "rows": rows,
                            "actionable_rate": regime_actionable_rate,
                            "neutral_rate": regime_neutral_rate,
                            "event_multiplier": float(event_multiplier),
                            "neutral_multiplier": float(neutral_multiplier),
                            "boosted": boosted,
                        }
                    )
                if enabled:
                    mean = float(weights.mean()) or 1.0
                    weights = np.clip(weights / mean, 0.15, 10.0)
    report = {
        "enabled": enabled,
        "method": "volatility_regime_actionable_event_weight_boost",
        "selection_data": "train_only",
        "volatility_column": volatility_col,
        "label_min_return": float(label_min_return),
        "actionable_threshold": float(threshold),
        "actionable_rate": actionable_rate,
        "target_actionable_rate": target_actionable_rate,
        "max_overall_actionable_rate": max_overall_actionable_rate,
        "min_regime_rows": min_regime_rows,
        "regimes": regimes,
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_max": float(np.max(weights)) if len(weights) else 0.0,
    }
    return weights, report


def cost_edge_balanced_sample_weight(frame: pd.DataFrame, label_min_return: float) -> tuple[np.ndarray, dict[str, Any]]:
    returns = pd.to_numeric(frame["future_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "edge_return_threshold" in frame.columns:
        threshold = pd.to_numeric(frame["edge_return_threshold"], errors="coerce").fillna(abs(float(label_min_return))).to_numpy(dtype=float)
    else:
        threshold = np.full(len(returns), abs(float(label_min_return)), dtype=float)
    threshold = np.maximum(threshold, abs(float(label_min_return)))
    long_edge = returns - threshold
    short_edge = -returns - threshold
    edge = np.maximum.reduce([long_edge, short_edge, np.zeros(len(returns), dtype=float)])
    neutral = edge <= 0.0
    actionable = edge > 0.0
    base = np.full(len(returns), 0.22, dtype=float)
    near_edge = np.abs(np.abs(returns) - threshold) <= (0.35 * np.maximum(threshold, 1e-6))
    base[near_edge] = 0.55
    base[actionable] = 1.25 + np.minimum(5.0, edge[actionable] / np.maximum(threshold[actionable], 1e-6))
    long_positive = long_edge > 0.0
    short_positive = short_edge > 0.0
    for mask in [long_positive, short_positive]:
        count = int(mask.sum())
        if count:
            base[mask] *= len(returns) / max(2.0 * count, 1.0)
    mean = float(base.mean()) or 1.0
    weights = np.clip(base / mean, 0.15, 12.0)
    report = {
        "enabled": bool(int(actionable.sum()) >= 8 and int((~actionable).sum()) >= 8),
        "method": "cost_edge_balanced_weighting",
        "selection_data": "train_only",
        "label_min_return": float(label_min_return),
        "mean_edge_threshold": float(np.mean(threshold)) if len(threshold) else 0.0,
        "median_edge_threshold": float(np.median(threshold)) if len(threshold) else 0.0,
        "edge_actionable_rate": float(actionable.mean()) if len(actionable) else 0.0,
        "edge_neutral_rate": float(neutral.mean()) if len(neutral) else 0.0,
        "edge_long_rate": float(long_positive.mean()) if len(long_positive) else 0.0,
        "edge_short_rate": float(short_positive.mean()) if len(short_positive) else 0.0,
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_max": float(np.max(weights)) if len(weights) else 0.0,
    }
    return weights, report


def sample_weight_for_variant(frame: pd.DataFrame, label_min_return: float, variant: str) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "cost_edge_balanced":
        return cost_edge_balanced_sample_weight(frame, label_min_return)
    if variant == "event_balanced":
        return event_balanced_sample_weight(frame, label_min_return)
    if variant == "volatility_regime_event_balanced":
        return volatility_regime_event_sample_weight(frame, label_min_return)
    weights = large_move_sample_weight(frame, label_min_return)
    return weights, {
        "enabled": True,
        "method": "large_move_weighting",
        "selection_data": "train_only",
        "label_min_return": float(label_min_return),
        "weight_mean": float(np.mean(weights)) if len(weights) else 0.0,
        "weight_max": float(np.max(weights)) if len(weights) else 0.0,
    }


def large_move_sample_weight_for_returns(returns: pd.Series, label_min_return: float) -> np.ndarray:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base = np.full(len(returns), 0.35, dtype=float)
    scale = max(abs(label_min_return), 1e-6)
    magnitude = np.abs(returns) / scale
    tradable = magnitude >= 1.0
    base[tradable] = 1.0 + np.minimum(4.0, magnitude[tradable] * 0.65)

    up = returns > abs(label_min_return)
    down = returns < -abs(label_min_return)
    for mask in [up, down]:
        count = int(mask.sum())
        if count:
            base[mask] *= len(returns) / max(2.0 * count, 1.0)

    mean = float(base.mean()) or 1.0
    return np.clip(base / mean, 0.25, 8.0)


def large_move_capture_metrics(
    frame: pd.DataFrame,
    prob: np.ndarray,
    *,
    long_threshold: float,
    short_threshold: float,
    cutoff: float,
) -> dict[str, float]:
    returns = pd.to_numeric(frame["future_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prob = np.asarray(prob, dtype=float)
    big_up = returns >= cutoff
    big_down = returns <= -cutoff
    up_count = int(big_up.sum())
    down_count = int(big_down.sum())
    up_capture = float((prob[big_up] >= long_threshold).mean()) if up_count else 0.0
    down_capture = float((prob[big_down] <= short_threshold).mean()) if down_count else 0.0
    return {
        "large_move_cutoff": float(cutoff),
        "large_up_count": float(up_count),
        "large_down_count": float(down_count),
        "large_up_capture": up_capture,
        "large_down_capture": down_capture,
        "large_move_capture": (up_capture + down_capture) / 2.0 if up_count and down_count else max(up_capture, down_capture),
    }


def _regime_detail_metrics(detail: pd.DataFrame) -> dict[str, float | int]:
    if detail.empty or "strategy_return" not in detail.columns:
        return {
            "rows": int(len(detail)),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "profit_factor": 0.0,
            "fee_drag": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    returns = pd.to_numeric(detail["strategy_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak.replace(0, np.nan) - 1.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    position = pd.to_numeric(detail.get("position", pd.Series(0, index=detail.index)), errors="coerce").fillna(0)
    turnover = pd.to_numeric(detail.get("turnover", pd.Series(0, index=detail.index)), errors="coerce").fillna(0)
    long_entries = int(((position > 0) & (turnover > 0)).sum())
    short_entries = int(((position < 0) & (turnover > 0)).sum())
    return {
        "rows": int(len(detail)),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "trades": int((turnover > 0).sum()),
        "profit_factor": float(profit_factor),
        "fee_drag": float(pd.to_numeric(detail.get("trade_cost", pd.Series(0, index=detail.index)), errors="coerce").fillna(0).sum()),
        "long_trades": long_entries,
        "short_trades": short_entries,
    }


def volatility_regime_backtest_report(
    train_df: pd.DataFrame,
    test_detail: pd.DataFrame,
    *,
    min_profit_factor: float,
    max_drawdown_floor: float,
    min_regime_trades: int,
) -> dict[str, Any]:
    source_col = "atr_14" if "atr_14" in train_df.columns and "atr_14" in test_detail.columns else ""
    if not source_col:
        return {
            "enabled": False,
            "reason": "missing_atr_14",
            "gate_passed": False,
            "required_for_publish": False,
            "regime_definition_source": "unavailable",
            "items": [],
        }
    train_vol = pd.to_numeric(train_df[source_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(train_vol) < 120:
        return {
            "enabled": False,
            "reason": "not_enough_train_rows_for_regime_quantiles",
            "gate_passed": False,
            "required_for_publish": False,
            "regime_definition_source": "train_quantiles",
            "items": [],
        }
    low_cut = float(train_vol.quantile(1.0 / 3.0))
    high_cut = float(train_vol.quantile(2.0 / 3.0))
    test_vol = pd.to_numeric(test_detail[source_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    items: list[dict[str, Any]] = []
    definitions = [
        ("low_vol", test_vol <= low_cut),
        ("mid_vol", (test_vol > low_cut) & (test_vol <= high_cut)),
        ("high_vol", test_vol > high_cut),
    ]
    gate_passed = True
    for name, mask in definitions:
        bucket = test_detail.loc[mask.fillna(False)].copy()
        metrics = _regime_detail_metrics(bucket)
        enough_trades = int(metrics["trades"]) >= int(min_regime_trades)
        passed = bool(
            enough_trades
            and float(metrics["profit_factor"]) >= min_profit_factor
            and float(metrics["max_drawdown"]) >= max_drawdown_floor
        )
        if not passed:
            gate_passed = False
        items.append(
            {
                "name": name,
                "passed": passed,
                "enough_trades": enough_trades,
                "min_regime_trades": int(min_regime_trades),
                "metrics": metrics,
            }
        )
    return {
        "enabled": True,
        "gate_passed": gate_passed,
        "regime_definition_source": "train_quantiles",
        "source_column": source_col,
        "low_cutoff": low_cut,
        "high_cutoff": high_cut,
        "min_regime_trades": int(min_regime_trades),
        "required_profit_factor": float(min_profit_factor),
        "required_max_drawdown_floor": float(max_drawdown_floor),
        "items": items,
    }


def interval_requires_extra_publish_gates(interval: str) -> bool:
    return interval.lower() in {"1m", "3m", "5m", "15m"}


def signal_expectancy_after_cost(
    returns: np.ndarray,
    signal: np.ndarray,
    direction: str,
    cost_buffer: float,
) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    signal = np.asarray(signal, dtype=bool)
    if not int(signal.sum()):
        return {
            "signal_expectancy_after_cost": 0.0,
            "signal_total_return_after_cost": 0.0,
            "signal_profit_factor_after_cost": 0.0,
            "signal_win_rate_after_cost": 0.0,
            "signal_count": 0.0,
            "signal_cost_buffer": float(cost_buffer),
        }
    if direction == "short":
        net = -returns[signal] - float(cost_buffer)
    elif direction == "trade":
        net = np.abs(returns[signal]) - float(cost_buffer)
    else:
        net = returns[signal] - float(cost_buffer)
    wins = net[net > 0]
    losses = net[net < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 999.0
    else:
        profit_factor = 0.0
    return {
        "signal_expectancy_after_cost": float(net.mean()),
        "signal_total_return_after_cost": float(net.sum()),
        "signal_profit_factor_after_cost": float(profit_factor),
        "signal_win_rate_after_cost": float(len(wins) / max(len(net), 1)),
        "signal_count": float(len(net)),
        "signal_cost_buffer": float(cost_buffer),
    }


def overfit_guard_metrics(y_train: np.ndarray, train_prob: np.ndarray, valid_metrics: dict[str, float]) -> dict[str, float]:
    train_metrics = classification_metrics(y_train, train_prob)
    balanced_gap = max(0.0, train_metrics.get("balanced_accuracy", 0.0) - valid_metrics.get("balanced_accuracy", 0.0))
    accuracy_gap = max(0.0, train_metrics.get("accuracy", 0.0) - valid_metrics.get("accuracy", 0.0))
    log_loss_gap = max(0.0, valid_metrics.get("log_loss", 1.0) - train_metrics.get("log_loss", 1.0))
    penalty = max(0.0, balanced_gap - 0.08) * 0.35 + max(0.0, accuracy_gap - 0.08) * 0.15 + max(0.0, log_loss_gap - 0.08) * 0.05
    return {
        "train_accuracy": train_metrics.get("accuracy", 0.0),
        "train_balanced_accuracy": train_metrics.get("balanced_accuracy", 0.0),
        "train_auc": train_metrics.get("auc", 0.5),
        "train_log_loss": train_metrics.get("log_loss", 1.0),
        "balanced_accuracy_gap": balanced_gap,
        "accuracy_gap": accuracy_gap,
        "log_loss_gap": log_loss_gap,
        "penalty": float(min(0.25, penalty)),
    }


def rolling_validation_probe(
    model: object,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    folds: int,
    deadline: float,
    train_df: pd.DataFrame | None = None,
    label_min_return: float | None = None,
    weight_variant: str = "base_large_move",
) -> dict[str, Any]:
    if folds <= 0:
        return {"enabled": False, "folds": []}
    n_rows = len(x_train)
    min_train = max(240, int(n_rows * 0.35))
    validation_rows = max(80, int(n_rows * 0.10))
    if n_rows < min_train + validation_rows:
        return {"enabled": False, "reason": "not_enough_rows", "folds": []}

    fold_entries: list[dict[str, Any]] = []
    max_train_end = n_rows - validation_rows
    for idx in range(folds):
        if time.monotonic() >= deadline:
            break
        ratio = 0.55 if folds == 1 else 0.50 + idx * (0.35 / max(folds - 1, 1))
        train_end = int(n_rows * ratio)
        train_end = max(min_train, min(train_end, max_train_end))
        valid_end = min(n_rows, train_end + validation_rows)
        if valid_end <= train_end:
            continue
        try:
            fold_model = copy.deepcopy(model)
            fold_weights = sample_weight[:train_end]
            fold_weight_report: dict[str, Any] | None = None
            if train_df is not None and label_min_return is not None:
                fold_weights, fold_weight_report = sample_weight_for_variant(
                    train_df.iloc[:train_end].reset_index(drop=True),
                    label_min_return,
                    weight_variant,
                )
            fold_model.fit(x_train[:train_end], y_train[:train_end], sample_weight=fold_weights)
            prob = fold_model.predict_proba(x_train[train_end:valid_end])[:, 1]
            metrics = classification_metrics(y_train[train_end:valid_end], prob)
            entry = {
                "train_start": 0,
                "train_end": train_end,
                "valid_start": train_end,
                "valid_end": valid_end,
                "metrics": metrics,
                "weight_variant": weight_variant,
            }
            if fold_weight_report is not None:
                entry["sample_weight_report"] = fold_weight_report
            fold_entries.append(entry)
        except Exception as exc:
            fold_entries.append(
                {
                    "train_start": 0,
                    "train_end": train_end,
                    "valid_start": train_end,
                    "valid_end": valid_end,
                    "error": str(exc),
                }
            )
    good = [item["metrics"] for item in fold_entries if isinstance(item.get("metrics"), dict)]
    if not good:
        return {"enabled": True, "folds": fold_entries, "reason": "no_successful_folds"}
    summary = {
        key: float(np.mean([metrics.get(key, 0.0) for metrics in good]))
        for key in ["accuracy", "balanced_accuracy", "auc", "log_loss"]
    }
    return {
        "enabled": True,
        "folds": fold_entries,
        "summary": summary,
        "successful_folds": len(good),
    }


def directional_sample_weight_for_returns(returns: pd.Series, direction: str, label_min_return: float) -> np.ndarray:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    base = np.full(len(values), 0.30, dtype=float)
    threshold = abs(float(label_min_return))
    tradable = np.abs(values) >= threshold
    if direction == "trade":
        positive = np.abs(values) >= (2.0 * threshold)
    elif direction == "long":
        positive = values > threshold
    else:
        positive = values < -threshold
    base[tradable] = 0.80
    base[positive] = 1.75
    count = int(positive.sum())
    if count:
        base[positive] *= len(values) / max(2.0 * count, 1.0)
    mean = float(base.mean()) or 1.0
    return np.clip(base / mean, 0.20, 8.0)


def directional_sample_weight_for_frame(frame: pd.DataFrame, direction: str, label_min_return: float) -> np.ndarray:
    if direction == "long" and "future_return_net_long" in frame.columns:
        edge_values = pd.to_numeric(frame["future_return_net_long"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif direction == "short" and "future_return_net_short" in frame.columns:
        edge_values = pd.to_numeric(frame["future_return_net_short"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif direction == "trade" and "future_return_net_edge" in frame.columns:
        edge_values = pd.to_numeric(frame["future_return_net_edge"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        return directional_sample_weight_for_returns(frame["future_return"], direction, label_min_return)

    base = np.full(len(edge_values), 0.25, dtype=float)
    positive = edge_values > 0.0
    actionable = edge_values >= abs(float(label_min_return))
    near_edge = np.abs(edge_values) <= max(abs(float(label_min_return)), 1e-6)
    base[near_edge] = 0.50
    base[positive] = 1.15
    base[actionable] = 1.85 + np.minimum(4.0, edge_values[actionable] / max(abs(float(label_min_return)), 1e-6))
    count = int(positive.sum())
    if count:
        base[positive] *= len(edge_values) / max(2.0 * count, 1.0)
    mean = float(base.mean()) or 1.0
    return np.clip(base / mean, 0.20, 9.0)


def select_directional_model(
    direction: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    xs_train: np.ndarray,
    xs_valid: np.ndarray,
    *,
    seed: int,
    label_min_return: float,
    deadline: float,
    max_trials: int = 3,
    cost_buffer: float = 0.001,
) -> tuple[object | None, object | None, dict[str, Any]]:
    if direction == "trade":
        if "edge_trade_target" in train_df.columns and "edge_trade_target" in valid_df.columns:
            target_col = "edge_trade_target"
        elif "big_move_target" in train_df.columns and "big_move_target" in valid_df.columns:
            target_col = "big_move_target"
        else:
            target_col = "actionable_label" if "actionable_label" in train_df.columns else "tradable_label"
    elif direction == "long":
        if "edge_long_target" in train_df.columns and "edge_long_target" in valid_df.columns:
            target_col = "edge_long_target"
        else:
            target_col = "big_up_target" if "big_up_target" in train_df.columns and "big_up_target" in valid_df.columns else "long_target"
    elif direction == "short":
        if "edge_short_target" in train_df.columns and "edge_short_target" in valid_df.columns:
            target_col = "edge_short_target"
        else:
            target_col = "big_down_target" if "big_down_target" in train_df.columns and "big_down_target" in valid_df.columns else "short_target"
    else:
        target_col = f"{direction}_target"
    if target_col not in train_df.columns or target_col not in valid_df.columns:
        return None, None, {
            "direction": direction,
            "status": "skipped",
            "reason": f"missing_{target_col}",
        }
    y_train = train_df[target_col].to_numpy(dtype=int)
    y_valid = valid_df[target_col].to_numpy(dtype=int)
    if int(y_train.sum()) < 8 or int((1 - y_train).sum()) < 8 or int(y_valid.sum()) < 3 or int((1 - y_valid).sum()) < 3:
        return None, None, {
            "direction": direction,
            "status": "skipped",
            "reason": "insufficient_class_balance",
            "train_positive": int(y_train.sum()),
            "valid_positive": int(y_valid.sum()),
        }
    candidates, skipped = collect_accuracy_model_candidates(seed, complexity="standard")
    candidates = select_model_candidates(candidates, max_model_trials=max_trials, seed=seed)
    weights = directional_sample_weight_for_frame(train_df, direction, label_min_return)
    threshold_candidates = [0.50, 0.54, 0.57, 0.60, 0.63, 0.66, 0.70, 0.74, 0.78, 0.82]
    valid_returns = pd.to_numeric(valid_df["future_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ranked: list[dict[str, Any]] = []
    best_model: object | None = None
    best_score = -999.0
    shadow_model: object | None = None
    shadow_candidate: dict[str, Any] | None = None
    shadow_score = -999.0

    def threshold_report(prob: np.ndarray) -> dict[str, Any]:
        rows: list[dict[str, float]] = []
        positives = y_valid == 1
        negatives = y_valid == 0
        for threshold in threshold_candidates:
            signal = prob >= threshold
            signal_count = int(signal.sum())
            precision = float((y_valid[signal] == 1).mean()) if signal_count else 0.0
            capture = float(signal[positives].mean()) if int(positives.sum()) else 0.0
            rejection = float((~signal[negatives]).mean()) if int(negatives.sum()) else 0.0
            false_signal = float(signal[negatives].mean()) if int(negatives.sum()) else 0.0
            signal_rate = float(signal.mean()) if len(signal) else 0.0
            expectancy = signal_expectancy_after_cost(valid_returns, signal, direction, cost_buffer)
            signal_expectancy = float(expectancy["signal_expectancy_after_cost"])
            signal_total_return = float(expectancy["signal_total_return_after_cost"])
            signal_profit_factor = float(expectancy["signal_profit_factor_after_cost"])
            expectancy_score = (
                70.0 * signal_expectancy
                + 1.5 * signal_total_return
                + 0.035 * min(signal_profit_factor, 4.0)
            )
            if direction == "trade":
                score = (
                    expectancy_score
                    + 0.20 * precision
                    + 0.25 * rejection
                    + 0.10 * capture
                    - 0.12 * max(0.0, signal_rate - 0.45)
                )
            else:
                score = (
                    expectancy_score
                    + 0.22 * precision
                    + 0.16 * capture
                    + 0.10 * rejection
                    - 0.10 * max(0.0, signal_rate - 0.35)
                )
            rows.append(
                {
                    "threshold": float(threshold),
                    "score": float(score),
                    "signal_count": float(signal_count),
                    "signal_rate": signal_rate,
                    "signal_precision": precision,
                    "positive_capture": capture,
                    "negative_rejection_rate": rejection,
                    "false_signal_on_negative_rate": false_signal,
                    **expectancy,
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        return {"best": rows[0] if rows else {}, "ranking": rows[:10], "selection_dataset": "valid"}

    for model in candidates:
        name = str(getattr(model, "name", type(model).__name__))
        if time.monotonic() >= deadline:
            skipped.append({"name": name, "reason": "time_budget_exhausted"})
            break
        try:
            model.fit(xs_train, y_train, sample_weight=weights)
            prob = model.predict_proba(xs_valid)[:, 1]
            metrics = classification_metrics(y_valid, prob)
            thresholds = threshold_report(prob)
            best_threshold = thresholds.get("best", {})
            threshold_score = float(best_threshold.get("score", 0.0))
            score = accuracy_first_score(metrics) + 0.20 * threshold_score
            if best_threshold:
                score = 0.25 * accuracy_first_score(metrics) + threshold_score
            entry = {
                "name": name,
                "direction": direction,
                "status": "trained",
                "selection_score": float(score),
                "metrics": metrics,
                "signal_threshold": float(best_threshold.get("threshold", 0.57)),
                "signal_count": int(best_threshold.get("signal_count", 0.0)),
                "signal_rate": float(best_threshold.get("signal_rate", 0.0)),
                "signal_precision": float(best_threshold.get("signal_precision", 0.0)),
                "positive_capture": float(best_threshold.get("positive_capture", 0.0)),
                "negative_rejection_rate": float(best_threshold.get("negative_rejection_rate", 0.0)),
                "false_signal_on_negative_rate": float(best_threshold.get("false_signal_on_negative_rate", 0.0)),
                "signal_expectancy_after_cost": float(best_threshold.get("signal_expectancy_after_cost", 0.0)),
                "signal_total_return_after_cost": float(best_threshold.get("signal_total_return_after_cost", 0.0)),
                "signal_profit_factor_after_cost": float(best_threshold.get("signal_profit_factor_after_cost", 0.0)),
                "signal_win_rate_after_cost": float(best_threshold.get("signal_win_rate_after_cost", 0.0)),
                "signal_cost_buffer": float(best_threshold.get("signal_cost_buffer", cost_buffer)),
                "threshold_optimization": thresholds,
            }
            ranked.append(entry)
            if score > best_score:
                best_score = score
                best_model = model
            candidate = select_shadow_threshold_candidate(
                list(thresholds.get("ranking") or []),
            )
            if candidate is not None:
                candidate_score = (
                    float(candidate["signal_total_return_after_cost"])
                    + 0.001
                    * min(
                        float(
                            candidate[
                                "signal_profit_factor_after_cost"
                            ]
                        ),
                        10.0,
                    )
                )
                if candidate_score > shadow_score:
                    shadow_score = candidate_score
                    shadow_model = model
                    shadow_candidate = {
                        **candidate,
                        "model_name": name,
                        "direction": direction,
                        "selection_dataset": "validation_calibration",
                        "test_used_for_selection": False,
                    }
        except Exception as exc:
            skipped.append({"name": name, "reason": f"fit_failed: {exc}"})
    ranked.sort(key=lambda item: item["selection_score"], reverse=True)
    return best_model, shadow_model, {
        "direction": direction,
        "status": "trained" if best_model is not None else "skipped",
        "best_model": str(getattr(best_model, "name", type(best_model).__name__)) if best_model is not None else "",
        "ranking": ranked,
        "skipped": skipped,
        "target_col": target_col,
        "selection_data": "train_valid_only",
        "model_selection_dataset": "validation_calibration",
        "threshold_selection_dataset": "validation_calibration",
        "test_used_for_selection": False,
        "threshold_candidates": threshold_candidates,
        "threshold_selection_objective": "signal_expectancy_after_cost",
        "cost_buffer": float(cost_buffer),
        "valid_rows": len(valid_df),
        "valid_positive_count": int(y_valid.sum()),
        "valid_negative_count": int((1 - y_valid).sum()),
        "valid_positive_rate": float(y_valid.mean()) if len(y_valid) else 0.0,
        "shadow_candidate": shadow_candidate,
        "shadow_policy": {
            "min_signal_count": 8,
            "min_profit_factor": 1.20,
            "min_total_return": 0.0,
            "requires_positive_expectancy": True,
            "selection_dataset": "validation_calibration",
            "test_used_for_selection": False,
        },
    }


def train_directional_auxiliary_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    xs_train: np.ndarray,
    xs_valid: np.ndarray,
    xs_test: np.ndarray,
    seed: int,
    label_min_return: float,
    deadline: float,
    base_backtest_cfg: BacktestConfig,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, Any]]:
    auxiliary_models: dict[str, object] = {}
    auxiliary_metadata: dict[str, dict[str, object]] = {}
    report: dict[str, Any] = {
        "enabled": True,
        "selection_data": "train_valid_only",
        "test_data_usage": "final_metrics_only_after_auxiliary_selection",
        "directions": {},
    }
    for offset, direction in enumerate(["trade", "long", "short"], start=1):
        cost_buffer = float(base_backtest_cfg.fee_rate + base_backtest_cfg.slippage_rate + base_backtest_cfg.funding_rate_buffer)
        model, shadow_model, direction_report = select_directional_model(
            direction,
            train_df,
            valid_df,
            xs_train,
            xs_valid,
            seed=seed + 100 + offset,
            label_min_return=label_min_return,
            deadline=deadline,
            cost_buffer=cost_buffer,
        )
        if model is not None:
            key = f"direction_{direction}"
            auxiliary_models[key] = model
            auxiliary_metadata[key] = {
                "direction": direction,
                "target": str(direction_report.get("target_col", "tradable_label")),
                "label_min_return": float(label_min_return),
                "cost_buffer": float(cost_buffer),
                "selected_model": str(getattr(model, "name", type(model).__name__)),
                "selected_signal_threshold": float(
                    (direction_report.get("ranking", [{}])[0] if direction_report.get("ranking") else {}).get("signal_threshold", 0.57)
                ),
                "threshold_selection_dataset": "validation_calibration",
            }
            y_test_col = str(direction_report.get("target_col", "tradable_label"))
            y_test = test_df[y_test_col].to_numpy(dtype=int)
            test_prob = model.predict_proba(xs_test)[:, 1]
            direction_report["test_metrics"] = classification_metrics(y_test, test_prob)
            direction_report["test_data_usage"] = "final_metrics_only_after_auxiliary_selection"
            direction_report["test_used_for_selection"] = False
        shadow_candidate = direction_report.get("shadow_candidate")
        if shadow_model is not None and isinstance(shadow_candidate, dict):
            shadow_key = f"shadow_direction_{direction}"
            auxiliary_models[shadow_key] = shadow_model
            auxiliary_metadata[shadow_key] = {
                "direction": direction,
                "target": str(
                    direction_report.get(
                        "target_col",
                        "tradable_label",
                    )
                ),
                "selected_model": str(
                    shadow_candidate.get("model_name")
                    or getattr(
                        shadow_model,
                        "name",
                        type(shadow_model).__name__,
                    )
                ),
                "selected_signal_threshold": float(
                    shadow_candidate.get("threshold", 1.0)
                ),
                "signal_count": int(
                    shadow_candidate.get("signal_count", 0)
                ),
                "signal_total_return_after_cost": float(
                    shadow_candidate.get(
                        "signal_total_return_after_cost",
                        0.0,
                    )
                ),
                "signal_profit_factor_after_cost": float(
                    shadow_candidate.get(
                        "signal_profit_factor_after_cost",
                        0.0,
                    )
                ),
                "selection_dataset": "validation_calibration",
                "test_used_for_selection": False,
                "mode": "shadow_paper_only",
            }
        report["directions"][direction] = direction_report
    report["status"] = (
        "trained" if {"direction_trade", "direction_long", "direction_short"}.issubset(auxiliary_models) else "partial_or_skipped"
    )
    return auxiliary_models, auxiliary_metadata, report


def label_distribution_report(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    def summarize(frame: pd.DataFrame) -> dict[str, float]:
        rows = max(len(frame), 1)
        payload: dict[str, float] = {"rows": float(len(frame))}
        for col in [
            "long_target",
            "short_target",
            "tradable_label",
            "actionable_label",
            "edge_long_target",
            "edge_short_target",
            "edge_trade_target",
            "big_up_target",
            "big_down_target",
            "big_move_target",
        ]:
            if col in frame.columns:
                payload[f"{col}_rate"] = float(pd.to_numeric(frame[col], errors="coerce").fillna(0).mean())
        if "tradable_label" in frame.columns:
            payload["neutral_rate"] = 1.0 - float(pd.to_numeric(frame["tradable_label"], errors="coerce").fillna(0).mean())
        if "actionable_label" in frame.columns:
            payload["non_actionable_rate"] = 1.0 - float(pd.to_numeric(frame["actionable_label"], errors="coerce").fillna(0).mean())
        return payload

    return {
        "train": summarize(train_df),
        "valid": summarize(valid_df),
        "test": summarize(test_df),
    }


def train_multi_horizon_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    xs_train: np.ndarray,
    xs_valid: np.ndarray,
    xs_test: np.ndarray,
    seed: int,
    label_min_return: float,
    specs: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, Any]]:
    auxiliary_models: dict[str, object] = {}
    auxiliary_metadata: dict[str, dict[str, object]] = {}
    report: dict[str, Any] = {"enabled": bool(specs), "items": {}, "skipped": []}
    for spec in specs:
        key = spec["key"]
        target_col = spec["target_col"]
        return_col = spec["return_col"]
        if target_col not in train_df.columns or return_col not in train_df.columns:
            report["skipped"].append({"key": key, "reason": "missing_columns"})
            continue
        y_train = train_df[target_col].to_numpy(dtype=int)
        y_valid = valid_df[target_col].to_numpy(dtype=int)
        y_test = test_df[target_col].to_numpy(dtype=int)
        weights = large_move_sample_weight_for_returns(train_df[return_col], label_min_return)
        candidates = [
            _name_numpy_model(
                LogisticRegressionNumpy(learning_rate=0.02, epochs=900, l2=3e-3, seed=seed + int(spec.get("steps", "0"))),
                f"{key}_logistic_regression_numpy",
            ),
            _name_numpy_model(
                MLPClassifierNumpy(hidden_size=32, learning_rate=0.0015, epochs=320, batch_size=256, l2=5e-4, seed=seed + 11 + int(spec.get("steps", "0"))),
                f"{key}_mlp_numpy_h32",
            ),
        ]
        scored: list[tuple[float, object, dict[str, Any]]] = []
        for model in candidates:
            try:
                model.fit(xs_train, y_train, sample_weight=weights)
                valid_prob = model.predict_proba(xs_valid)[:, 1]
                valid_metrics = classification_metrics(y_valid, valid_prob)
                train_probe_rows = min(len(xs_train), 3000)
                train_prob = model.predict_proba(xs_train[-train_probe_rows:])[:, 1]
                overfit_guard = overfit_guard_metrics(y_train[-train_probe_rows:], train_prob, valid_metrics)
                score = accuracy_first_score(valid_metrics) - overfit_guard["penalty"]
                scored.append(
                    (
                        score,
                        model,
                        {
                            "name": getattr(model, "name", type(model).__name__),
                            "score": score,
                            "valid_metrics": valid_metrics,
                            "overfit_guard": overfit_guard,
                        },
                    )
                )
            except Exception as exc:
                report["skipped"].append({"key": key, "model": getattr(model, "name", type(model).__name__), "reason": str(exc)})
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        best_model = scored[0][1]
        test_prob = best_model.predict_proba(xs_test)[:, 1]
        test_metrics = classification_metrics(y_test, test_prob)
        returns = pd.to_numeric(test_df[return_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        predicted_direction = np.where(test_prob >= 0.5, "up", "down")
        actual_direction = np.where(returns > 0, "up", "down")
        matched = predicted_direction == actual_direction
        sample_rows = []
        sample_source = test_df.tail(30).reset_index(drop=True)
        sample_prob = test_prob[-len(sample_source) :]
        sample_returns = returns[-len(sample_source) :]
        for idx, row in sample_source.iterrows():
            pred = "up" if sample_prob[idx] >= 0.5 else "down"
            actual = "up" if sample_returns[idx] > 0 else "down"
            sample_rows.append(
                {
                    "open_time": int(row["open_time"]),
                    "open_datetime": str(row.get("open_datetime", "")),
                    "close": float(row["close"]),
                    "up_probability": float(sample_prob[idx]),
                    "predicted_direction": pred,
                    "actual_return": float(sample_returns[idx]),
                    "actual_direction": actual,
                    "matched": pred == actual,
                }
            )
        auxiliary_models[key] = best_model
        auxiliary_metadata[key] = {
            **spec,
            "model_name": getattr(best_model, "name", type(best_model).__name__),
            "test_direction_match_rate": float(matched.mean()) if len(matched) else 0.0,
        }
        report["items"][key] = {
            **spec,
            "best_model": getattr(best_model, "name", type(best_model).__name__),
            "ranking": [item[2] for item in scored],
            "test_metrics": test_metrics,
            "test_direction_match_rate": float(matched.mean()) if len(matched) else 0.0,
            "actual_match_sample": sample_rows,
        }
    return auxiliary_models, auxiliary_metadata, report






def _name_numpy_model(model: object, name: str) -> object:
    setattr(model, "name", name)
    return model


def collect_accuracy_model_candidates(seed: int = 42, complexity: str = "standard") -> tuple[list[object], list[dict[str, str]]]:
    expanded = complexity in {"expanded", "deep", "blackbox"}
    neural_first = complexity in {"deep", "blackbox"}
    candidates: list[object] = [
        _name_numpy_model(
            LogisticRegressionNumpy(learning_rate=0.03, epochs=800, l2=1e-3, seed=seed),
            "logistic_regression_numpy_lr003_l2_1e3",
        ),
        _name_numpy_model(
            LogisticRegressionNumpy(learning_rate=0.015, epochs=1000, l2=3e-3, seed=seed),
            "logistic_regression_numpy_lr0015_l2_3e3",
        ),
        _name_numpy_model(
            MLPClassifierNumpy(hidden_size=32, learning_rate=0.002, epochs=450, batch_size=256, l2=1e-4, seed=seed),
            "mlp_numpy_h32_lr002",
        ),
        _name_numpy_model(
            MLPClassifierNumpy(hidden_size=64, learning_rate=0.0015, epochs=420, batch_size=256, l2=3e-4, seed=seed),
            "mlp_numpy_h64_lr0015",
        ),
    ]
    if expanded:
        candidates.extend(
            [
                _name_numpy_model(
                    LogisticRegressionNumpy(learning_rate=0.006, epochs=1400, l2=1e-2, seed=seed),
                    "logistic_regression_numpy_lr006_l2_1e2",
                ),
                _name_numpy_model(
                    MLPClassifierNumpy(hidden_size=96, learning_rate=0.001, epochs=520, batch_size=384, l2=5e-4, seed=seed),
                    "mlp_numpy_h96_lr001",
                ),
            ]
        )
    if neural_first:
        candidates.extend(
            [
                _name_numpy_model(
                    MLPClassifierNumpy(hidden_size=128, learning_rate=0.0008, epochs=620, batch_size=384, l2=7e-4, seed=seed + 21),
                    "neural_mlp_numpy_h128_lr0008_l2_7e4",
                ),
                _name_numpy_model(
                    MLPClassifierNumpy(hidden_size=64, learning_rate=0.001, epochs=760, batch_size=256, l2=1e-3, seed=seed + 22),
                    "neural_mlp_numpy_h64_lr001_l2_1e3",
                ),
                _name_numpy_model(
                    MLPClassifierNumpy(hidden_size=160, learning_rate=0.0006, epochs=560, batch_size=512, l2=1e-3, seed=seed + 23),
                    "neural_mlp_numpy_h160_lr0006_l2_1e3",
                ),
            ]
        )
    skipped: list[dict[str, str]] = []
    neural_candidates, neural_skipped = optional_neural_candidates(seed, complexity=complexity)
    candidates.extend(neural_candidates)
    skipped.extend(neural_skipped)

    if importlib.util.find_spec("sklearn") is None:
        skipped.append({"name": "sklearn_models", "reason": "missing_dependency: sklearn"})
    else:
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        for max_iter, learning_rate, max_leaf_nodes, l2 in [
            (180, 0.03, 31, 0.01),
            (260, 0.025, 45, 0.03),
            (320, 0.02, 63, 0.05),
        ]:
            candidates.append(
                SklearnModelAdapter(
                    f"sklearn_hist_gradient_boosting_i{max_iter}_lr{learning_rate}_leaf{max_leaf_nodes}",
                    HistGradientBoostingClassifier(
                        max_iter=max_iter,
                        learning_rate=learning_rate,
                        max_leaf_nodes=max_leaf_nodes,
                        l2_regularization=l2,
                        random_state=seed,
                    ),
                )
            )
        for depth, leaf in [(6, 16), (8, 20), (10, 30)]:
            candidates.append(
                SklearnModelAdapter(
                    f"sklearn_extra_trees_d{depth}_leaf{leaf}",
                    ExtraTreesClassifier(
                        n_estimators=260,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        n_jobs=1,
                        random_state=seed,
                        class_weight="balanced",
                    ),
                )
            )
        for depth, leaf in [(6, 18), (9, 24)]:
            candidates.append(
                SklearnModelAdapter(
                    f"sklearn_random_forest_d{depth}_leaf{leaf}",
                    RandomForestClassifier(
                        n_estimators=260,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        n_jobs=1,
                        random_state=seed,
                        class_weight="balanced_subsample",
                    ),
                )
            )
        for c_value in [0.25, 0.8, 1.5]:
            candidates.append(
                SklearnModelAdapter(
                    f"sklearn_logistic_regression_c{c_value}",
                    LogisticRegression(max_iter=1500, C=c_value, class_weight="balanced", random_state=seed),
                )
            )
        for layers, alpha, learning_rate in [
            ((32,), 3e-4, 0.001),
            ((48, 16), 1e-4, 0.001),
            ((64, 24), 3e-4, 0.0007),
        ]:
            layer_name = "x".join(str(item) for item in layers)
            candidates.append(
                SklearnModelAdapter(
                    f"sklearn_mlp_{layer_name}_a{alpha}",
                    MLPClassifier(
                        hidden_layer_sizes=layers,
                        alpha=alpha,
                        learning_rate_init=learning_rate,
                        max_iter=500,
                        random_state=seed,
                        early_stopping=True,
                    ),
                )
            )
        if expanded:
            for max_iter, learning_rate, max_leaf_nodes, l2, min_samples in [
                (420, 0.014, 31, 0.08, 24),
                (520, 0.010, 63, 0.12, 30),
                (360, 0.018, 95, 0.06, 18),
            ]:
                candidates.append(
                    SklearnModelAdapter(
                        f"sklearn_hist_gradient_boosting_exp_i{max_iter}_lr{learning_rate}_leaf{max_leaf_nodes}",
                        HistGradientBoostingClassifier(
                            max_iter=max_iter,
                            learning_rate=learning_rate,
                            max_leaf_nodes=max_leaf_nodes,
                            l2_regularization=l2,
                            min_samples_leaf=min_samples,
                            random_state=seed,
                        ),
                    )
                )
            for estimators, depth, leaf, features in [
                (360, None, 24, "sqrt"),
                (420, 12, 10, 0.75),
                (520, 16, 6, None),
            ]:
                candidates.append(
                    SklearnModelAdapter(
                        f"sklearn_extra_trees_exp_n{estimators}_d{depth}_leaf{leaf}",
                        ExtraTreesClassifier(
                            n_estimators=estimators,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            max_features=features,
                            n_jobs=1,
                            random_state=seed,
                            class_weight="balanced",
                        ),
                    )
                )
            for estimators, depth, leaf, split in [(360, 12, 10, 24), (460, None, 12, 30), (520, 16, 6, 18)]:
                candidates.append(
                    SklearnModelAdapter(
                        f"sklearn_random_forest_exp_n{estimators}_d{depth}_leaf{leaf}",
                        RandomForestClassifier(
                            n_estimators=estimators,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            min_samples_split=split,
                            n_jobs=1,
                            random_state=seed,
                            class_weight="balanced_subsample",
                        ),
                    )
                )
            for c_value in [0.08, 3.0, 8.0]:
                candidates.append(
                    SklearnModelAdapter(
                        f"sklearn_logistic_regression_exp_c{c_value}",
                        LogisticRegression(max_iter=2200, C=c_value, class_weight="balanced", random_state=seed),
                    )
                )
            for layers, alpha, learning_rate in [
                ((96, 32), 5e-4, 0.0007),
                ((128, 48, 16), 8e-4, 0.0005),
                ((64, 64), 2e-4, 0.0008),
            ]:
                layer_name = "x".join(str(item) for item in layers)
                candidates.append(
                    SklearnModelAdapter(
                        f"sklearn_mlp_exp_{layer_name}_a{alpha}",
                        MLPClassifier(
                            hidden_layer_sizes=layers,
                            alpha=alpha,
                            learning_rate_init=learning_rate,
                            max_iter=700,
                            random_state=seed,
                            early_stopping=True,
                        ),
                    )
                )
        if neural_first:
            for layers, alpha, learning_rate, max_iter in [
                ((128, 64), 6e-4, 0.0006, 850),
                ((96, 48, 16), 9e-4, 0.0005, 900),
                ((160, 64, 24), 1.2e-3, 0.00045, 820),
            ]:
                layer_name = "x".join(str(item) for item in layers)
                candidates.append(
                    SklearnModelAdapter(
                        f"neural_mlp_sklearn_{layer_name}_a{alpha}",
                        MLPClassifier(
                            hidden_layer_sizes=layers,
                            alpha=alpha,
                            learning_rate_init=learning_rate,
                            max_iter=max_iter,
                            random_state=seed,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=18,
                        ),
                    )
                )

    if importlib.util.find_spec("lightgbm") is None:
        skipped.append({"name": "lightgbm_lgbm_classifier", "reason": "missing_dependency: lightgbm"})
    else:
        for n_estimators, learning_rate, num_leaves, max_depth, subsample, colsample, reg_lambda, min_child in [
            (180, 0.035, 31, -1, 0.85, 0.85, 1.0, 20),
            (260, 0.025, 45, 8, 0.80, 0.80, 2.0, 30),
            (320, 0.018, 63, 10, 0.75, 0.85, 3.0, 40),
        ]:
            candidates.append(
                build_lightgbm_classifier_alpha(
                    f"lightgbm_lgbm_classifier_n{n_estimators}_leaf{num_leaves}",
                    seed=seed,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample,
                    reg_lambda=reg_lambda,
                    min_child_samples=min_child,
                )
            )
        if expanded:
            for n_estimators, learning_rate, num_leaves, max_depth, subsample, colsample, reg_lambda, min_child in [
                (420, 0.014, 95, 12, 0.72, 0.72, 5.0, 50),
                (520, 0.010, 127, -1, 0.70, 0.80, 8.0, 65),
                (360, 0.020, 31, 6, 0.90, 0.70, 4.0, 18),
            ]:
                candidates.append(
                    build_lightgbm_classifier_alpha(
                        f"lightgbm_lgbm_classifier_exp_n{n_estimators}_leaf{num_leaves}",
                        seed=seed,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        max_depth=max_depth,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_lambda=reg_lambda,
                        min_child_samples=min_child,
                    )
                )

    if importlib.util.find_spec("xgboost") is None:
        skipped.append({"name": "xgboost_xgb_classifier", "reason": "missing_dependency: xgboost"})
    else:
        from xgboost import XGBClassifier

        for n_estimators, depth, learning_rate, subsample, colsample, reg_lambda, min_child_weight in [
            (180, 3, 0.035, 0.85, 0.85, 1.0, 2.0),
            (260, 4, 0.025, 0.80, 0.80, 2.0, 4.0),
            (320, 5, 0.018, 0.75, 0.85, 3.0, 6.0),
        ]:
            candidates.append(
                SklearnModelAdapter(
                    f"xgboost_xgb_classifier_n{n_estimators}_d{depth}",
                    XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample,
                        reg_lambda=reg_lambda,
                        min_child_weight=min_child_weight,
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=1,
                        verbosity=0,
                    ),
                )
            )
        if expanded:
            for n_estimators, depth, learning_rate, subsample, colsample, reg_lambda, min_child_weight in [
                (420, 3, 0.014, 0.70, 0.78, 5.0, 8.0),
                (520, 4, 0.010, 0.72, 0.72, 8.0, 10.0),
                (360, 6, 0.020, 0.80, 0.70, 4.0, 5.0),
            ]:
                candidates.append(
                    SklearnModelAdapter(
                        f"xgboost_xgb_classifier_exp_n{n_estimators}_d{depth}",
                        XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample,
                            reg_lambda=reg_lambda,
                            min_child_weight=min_child_weight,
                            eval_metric="logloss",
                            random_state=seed,
                            n_jobs=1,
                            verbosity=0,
                        ),
                    )
                )

    if importlib.util.find_spec("catboost") is None:
        skipped.append({"name": "catboost_classifier", "reason": "missing_dependency: catboost"})
    else:
        from catboost import CatBoostClassifier

        for iterations, depth, learning_rate, l2_leaf_reg in [
            (180, 4, 0.04, 3.0),
            (260, 5, 0.03, 5.0),
            (320, 6, 0.02, 7.0),
        ]:
            candidates.append(
                SklearnModelAdapter(
                    f"catboost_classifier_i{iterations}_d{depth}",
                    CatBoostClassifier(
                        iterations=iterations,
                        depth=depth,
                        learning_rate=learning_rate,
                        l2_leaf_reg=l2_leaf_reg,
                        loss_function="Logloss",
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                )
            )
        if expanded:
            for iterations, depth, learning_rate, l2_leaf_reg in [
                (420, 5, 0.016, 9.0),
                (520, 6, 0.012, 11.0),
                (360, 7, 0.020, 6.0),
            ]:
                candidates.append(
                    SklearnModelAdapter(
                        f"catboost_classifier_exp_i{iterations}_d{depth}",
                        CatBoostClassifier(
                            iterations=iterations,
                            depth=depth,
                            learning_rate=learning_rate,
                            l2_leaf_reg=l2_leaf_reg,
                            loss_function="Logloss",
                            random_seed=seed,
                            verbose=False,
                            allow_writing_files=False,
                            thread_count=1,
                        ),
                    )
                )

    return candidates, skipped


def select_model_candidates(candidates: list[object], max_model_trials: int, seed: int) -> list[object]:
    if max_model_trials <= 0 or max_model_trials >= len(candidates):
        return candidates

    def name_of(model: object) -> str:
        return str(getattr(model, "name", "")).lower()

    def add_unique(selected: list[object], bucket: list[object], limit: int) -> None:
        for model in bucket:
            if len(selected) >= limit:
                return
            if model not in selected:
                selected.append(model)

    torch_models = [model for model in candidates if name_of(model).startswith("torch_")]
    explicit_neural_mlp = [
        model
        for model in candidates
        if name_of(model).startswith("neural_mlp") and not name_of(model).startswith("torch_")
    ]
    standard_mlp = [
        model
        for model in candidates
        if "mlp" in name_of(model)
        and not name_of(model).startswith("neural_mlp")
        and not name_of(model).startswith("torch_")
    ]
    neural_mlp = explicit_neural_mlp + standard_mlp
    baseline = [model for model in candidates if name_of(model).startswith("logistic_regression_numpy")]
    non_torch_rest = [model for model in candidates if model not in torch_models]

    selected: list[object] = []
    reserve_torch = 1 if torch_models and max_model_trials >= 8 else 0
    non_torch_limit = max_model_trials - reserve_torch
    baseline_count = 1 if max_model_trials <= 5 else 2
    add_unique(selected, baseline, min(non_torch_limit, baseline_count))
    neural_target = min(non_torch_limit, len(selected) + max(1, min(4, max_model_trials // 2)))
    add_unique(selected, neural_mlp, neural_target)
    remaining_slots = max_model_trials - len(selected)
    if reserve_torch and remaining_slots > 0:
        add_unique(selected, torch_models, max_model_trials)
        remaining_slots = max_model_trials - len(selected)
    if remaining_slots <= 0:
        return selected
    rest = [model for model in non_torch_rest if model not in selected]
    if rest:
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(len(rest), size=min(remaining_slots, len(rest)), replace=False)
        selected.extend(rest[int(idx)] for idx in selected_idx)
    if len(selected) < max_model_trials:
        add_unique(selected, [model for model in candidates if model not in selected], max_model_trials)
    return selected


def train_accuracy_first_candidates(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    max_model_trials: int,
    deadline: float,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    base_backtest_cfg: BacktestConfig,
    label_min_return: float,
    complexity: str = "standard",
    rolling_folds: int = 0,
    multi_horizon_specs: list[dict[str, str]] | None = None,
    objective: str = "return",
    primary_target_col: str | None = None,
    force_compact_threshold_search: bool = False,
    max_threshold_evals: int | None = None,
    validation_split_purge_rows: int = 0,
) -> tuple[ModelBundle, dict[str, Any], pd.DataFrame]:
    validation_split = split_validation_for_strategy_calibration(
        valid_df,
        purge_rows=max(int(validation_split_purge_rows or 0), int(max(MULTI_HORIZON_STEPS)), 0),
    )
    validation_calibration_df = validation_split.calibration
    validation_gate_df = validation_split.gate
    validation_split_report = validation_split.report
    selected_primary_target_col = choose_primary_target_col(
        train_df,
        validation_calibration_df,
        objective,
        primary_target_col,
    )
    x_train, y_train, _ = feature_matrix(train_df, target_col=selected_primary_target_col, feature_columns=feature_columns)
    x_valid, y_valid, _ = feature_matrix(
        validation_calibration_df,
        target_col=selected_primary_target_col,
        feature_columns=feature_columns,
    )
    x_test, y_test, _ = feature_matrix(test_df, target_col=selected_primary_target_col, feature_columns=feature_columns)
    scaler = StandardScaler().fit(x_train)
    xs_train = scaler.transform(x_train)
    xs_valid = scaler.transform(x_valid)
    xs_test = scaler.transform(x_test)
    base_train_weights, base_weight_report = sample_weight_for_variant(train_df, label_min_return, "base_large_move")
    event_train_weights, event_balance_report = sample_weight_for_variant(train_df, label_min_return, "event_balanced")
    regime_train_weights, regime_event_balance_report = sample_weight_for_variant(
        train_df,
        label_min_return,
        "volatility_regime_event_balanced",
    )
    edge_train_weights, edge_balance_report = sample_weight_for_variant(train_df, label_min_return, "cost_edge_balanced")
    weight_variant_reports: dict[str, dict[str, Any]] = {
        "base_large_move": base_weight_report,
        "event_balanced": event_balance_report,
        "volatility_regime_event_balanced": regime_event_balance_report,
        "cost_edge_balanced": edge_balance_report,
    }
    weight_variants: list[tuple[str, np.ndarray]] = []
    if edge_balance_report.get("enabled"):
        weight_variants.append(("cost_edge_balanced", edge_train_weights))
    weight_variants.append(("base_large_move", base_train_weights))
    if event_balance_report.get("enabled"):
        weight_variants.append(("event_balanced", event_train_weights))
    if regime_event_balance_report.get("enabled"):
        weight_variants.append(("volatility_regime_event_balanced", regime_train_weights))
    if max_model_trials <= 1 and edge_balance_report.get("enabled"):
        weight_variants = [("cost_edge_balanced", edge_train_weights)]
    elif deadline - time.monotonic() < 8.0 * 60.0 and edge_balance_report.get("enabled"):
        weight_variants = [
            item
            for item in weight_variants
            if item[0] in {"cost_edge_balanced", "base_large_move"}
        ]
    valid_cutoff = large_move_cutoff(validation_calibration_df, label_min_return)
    candidates, skipped = collect_accuracy_model_candidates(seed, complexity=complexity)
    candidates = select_model_candidates(candidates, max_model_trials=max_model_trials, seed=seed)
    model_selection_backtest_cfg = base_backtest_cfg
    if objective == "return" and selected_primary_target_col in {"edge_long_target", "big_up_target", "long_target"}:
        model_selection_backtest_cfg = BacktestConfig(
            **{
                **asdict(base_backtest_cfg),
                "trade_side_policy": "long_only",
            }
        )

    scored: list[tuple[float, ModelBundle, dict[str, Any]]] = []
    fitted_for_ensemble: list[tuple[float, ModelBundle, dict[str, Any]]] = []
    for base_model in candidates:
        base_model_name = getattr(base_model, "name", type(base_model).__name__)
        for weight_variant, train_weights in weight_variants:
            model = copy.deepcopy(base_model)
            model_name = str(base_model_name) if weight_variant == "base_large_move" else f"{base_model_name}_{weight_variant}"
            setattr(model, "name", model_name)
            if time.monotonic() >= deadline:
                skipped.append({"name": model_name, "reason": "time_budget_exhausted"})
                continue
            try:
                model.fit(xs_train, y_train, sample_weight=train_weights)
                prob = model.predict_proba(xs_valid)[:, 1]
                metrics = classification_metrics(y_valid, prob)
                train_probe_rows = min(len(xs_train), 3000)
                train_prob = model.predict_proba(xs_train[-train_probe_rows:])[:, 1]
                overfit_guard = overfit_guard_metrics(y_train[-train_probe_rows:], train_prob, metrics)
                rolling_probe = rolling_validation_probe(
                    model,
                    xs_train,
                    y_train,
                    train_weights,
                    folds=rolling_folds,
                    deadline=deadline,
                    train_df=train_df,
                    label_min_return=label_min_return,
                    weight_variant=weight_variant,
                )
                base_effective_long, base_effective_short = effective_thresholds(model_selection_backtest_cfg)
                large_metrics = large_move_capture_metrics(
                    validation_calibration_df,
                    prob,
                    long_threshold=base_effective_long,
                    short_threshold=base_effective_short,
                    cutoff=valid_cutoff,
                )
                bundle = ModelBundle(
                    model_name=model_name,
                    model=model,
                    scaler=scaler,
                    feature_columns=feature_columns,
                    metrics=metrics,
                    config={
                        "seed": seed,
                        "feature_version": FEATURE_VERSION,
                        "large_move_weighting": "enabled",
                        "weight_variant": weight_variant,
                        "sample_weight_method": weight_variant_reports.get(weight_variant, {}).get("method", ""),
                        "primary_target_col": selected_primary_target_col,
                        "target_family": "cost_edge_profit_first",
                        "cost_edge_label_version": "v1",
                        "objective": objective,
                        "complexity": complexity,
                        "neural_network_priority": "enabled" if complexity in {"deep", "blackbox"} else "disabled",
                        "transformer_policy": "blackbox_only",
                    },
                )
                valid_backtest, _ = run_backtest(validation_calibration_df, bundle, model_selection_backtest_cfg)
                if objective == "return":
                    base_score = return_first_score(metrics, valid_backtest, large_metrics)
                else:
                    base_score = accuracy_first_score(metrics) + 0.08 * large_metrics["large_move_capture"]
                rolling_penalty = 0.0
                if rolling_probe.get("enabled") and isinstance(rolling_probe.get("summary"), dict):
                    rolling_balanced = float(rolling_probe["summary"].get("balanced_accuracy", 0.0))
                    rolling_penalty = max(0.0, metrics.get("balanced_accuracy", 0.0) - rolling_balanced - 0.05) * 0.25
                penalty, reasons = trading_filter_penalty(
                    valid_backtest,
                    min_trades=min_trades,
                    max_drawdown_floor=max_drawdown_floor,
                    min_profit_factor=min_profit_factor,
                )
                archetype_gate = archetype_side_robustness(valid_backtest, min_trades=min_trades)
                hard_gate = validation_trading_gate(
                    valid_backtest,
                    min_trades=min_trades,
                    max_drawdown_floor=max_drawdown_floor,
                    min_profit_factor=min_profit_factor,
                    archetype_gate=archetype_gate,
                    min_total_return=0.0010,
                    max_fee_drag_to_abs_return=0.60,
                    min_expectancy_per_trade_after_cost=0.00002,
                )
                hard_gate = {**hard_gate, "decision_dataset": "validation_calibration"}
                predictive_score = (
                    accuracy_first_score(metrics)
                    + 0.08 * float(large_metrics["large_move_capture"])
                    - overfit_guard["penalty"]
                    - rolling_penalty
                )
                selection_score = base_score - penalty - overfit_guard["penalty"] - rolling_penalty
                entry = {
                    "name": model_name,
                    "base_model_name": str(base_model_name),
                    "weight_variant": weight_variant,
                    "sample_weight_report": weight_variant_reports.get(weight_variant, {}),
                    "status": "trained",
                    "base_score": base_score,
                    "predictive_score": predictive_score,
                    "strategy_selection_score": selection_score,
                    "selection_score": selection_score,
                    "trading_penalty": penalty,
                    "overfit_penalty": overfit_guard["penalty"],
                    "rolling_penalty": rolling_penalty,
                    "archetype_side_gate": archetype_gate,
                    "validation_trading_gate": hard_gate,
                    "filter_reasons": reasons,
                    "metrics": metrics,
                    "overfit_guard": overfit_guard,
                    "rolling_validation": rolling_probe,
                    "large_move_metrics": large_metrics,
                    "valid_backtest": asdict(valid_backtest),
                }
                scored.append((selection_score, bundle, entry))
                fitted_for_ensemble.append((base_score, bundle, entry))
            except Exception as exc:
                skipped.append({"name": model_name, "reason": f"fit_failed: {exc}"})
    for size in [2, 3, 4]:
        top = sorted(fitted_for_ensemble, key=lambda item: item[0], reverse=True)[:size]
        if len(top) < size or time.monotonic() >= deadline:
            continue
        ensemble_name = "ensemble_probability_average" if size == 4 else f"ensemble_probability_average_top{size}"
        ensemble = EnsembleProbabilityModel(
            name=ensemble_name,
            models=[item[1].model for item in top],
            model_names=[item[1].model_name for item in top],
        )
        prob = ensemble.predict_proba(xs_valid)[:, 1]
        metrics = classification_metrics(y_valid, prob)
        train_probe_rows = min(len(xs_train), 3000)
        train_prob = ensemble.predict_proba(xs_train[-train_probe_rows:])[:, 1]
        overfit_guard = overfit_guard_metrics(y_train[-train_probe_rows:], train_prob, metrics)
        base_effective_long, base_effective_short = effective_thresholds(model_selection_backtest_cfg)
        large_metrics = large_move_capture_metrics(
            validation_calibration_df,
            prob,
            long_threshold=base_effective_long,
            short_threshold=base_effective_short,
            cutoff=valid_cutoff,
        )
        bundle = ModelBundle(
            model_name=ensemble.name,
            model=ensemble,
            scaler=scaler,
            feature_columns=feature_columns,
            metrics=metrics,
            config={
                "seed": seed,
                "feature_version": FEATURE_VERSION,
                "members": ",".join(ensemble.model_names),
                "large_move_weighting": "enabled",
                "complexity": complexity,
                "neural_network_priority": "enabled" if complexity in {"deep", "blackbox"} else "disabled",
                "transformer_policy": "blackbox_only",
                "primary_target_col": selected_primary_target_col,
                "target_family": "cost_edge_profit_first",
                "cost_edge_label_version": "v1",
                "objective": objective,
            },
        )
        valid_backtest, _ = run_backtest(validation_calibration_df, bundle, model_selection_backtest_cfg)
        if objective == "return":
            base_score = return_first_score(metrics, valid_backtest, large_metrics) + 0.002
        else:
            base_score = accuracy_first_score(metrics) + 0.08 * large_metrics["large_move_capture"] + 0.002
        penalty, reasons = trading_filter_penalty(
            valid_backtest,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
        )
        archetype_gate = archetype_side_robustness(valid_backtest, min_trades=min_trades)
        hard_gate = validation_trading_gate(
            valid_backtest,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
            archetype_gate=archetype_gate,
            min_total_return=0.0010,
            max_fee_drag_to_abs_return=0.60,
            min_expectancy_per_trade_after_cost=0.00002,
        )
        hard_gate = {**hard_gate, "decision_dataset": "validation_calibration"}
        predictive_score = (
            accuracy_first_score(metrics)
            + 0.08 * float(large_metrics["large_move_capture"])
            + 0.002
            - overfit_guard["penalty"]
        )
        selection_score = base_score - penalty - overfit_guard["penalty"]
        entry = {
            "name": ensemble.name,
            "status": "trained",
            "base_score": base_score,
            "predictive_score": predictive_score,
            "strategy_selection_score": selection_score,
            "selection_score": selection_score,
            "trading_penalty": penalty,
            "overfit_penalty": overfit_guard["penalty"],
            "rolling_penalty": 0.0,
            "archetype_side_gate": archetype_gate,
            "validation_trading_gate": hard_gate,
            "filter_reasons": reasons,
            "metrics": metrics,
            "overfit_guard": overfit_guard,
            "rolling_validation": {"enabled": False, "reason": "ensemble_uses_full_fit_members"},
            "large_move_metrics": large_metrics,
            "valid_backtest": asdict(valid_backtest),
            "members": ensemble.model_names,
        }
        scored.append((selection_score, bundle, entry))

    if not scored:
        raise RuntimeError(f"No accuracy optimization candidates trained successfully: {skipped}")

    selection_ranking = rank_model_candidates(scored)
    scored = list(selection_ranking.strategy_gated)
    valid_scored = list(selection_ranking.strategy_eligible)
    best_bundle = selection_ranking.selected[1]
    selected_model_gate = selection_ranking.selected[2].get(
        "validation_trading_gate",
        {},
    )
    model_selection_gate = {
        "decision_dataset": "validation_calibration",
        "gate_dataset": "model_candidate_selection_only",
        "separate_threshold_gate_enabled": bool(validation_split_report.get("enabled", False)),
        "passed": bool(valid_scored),
        "selected_model_gate_passed": bool(selected_model_gate.get("passed", False)),
        "rejected_by_validation_trading_gate": bool(not valid_scored),
        "research_model_selected": bool(not valid_scored),
        "reason": "" if valid_scored else "no_model_candidate_passed_validation_trading_gate",
        "policy": "Models that fail validation trading gates may be saved for research only; threshold/side gates must still pass before any paper candidate.",
    }
    directional_models, directional_metadata, directional_report = train_directional_auxiliary_models(
        train_df,
        validation_calibration_df,
        test_df,
        xs_train=xs_train,
        xs_valid=xs_valid,
        xs_test=xs_test,
        seed=seed,
        label_min_return=label_min_return,
        deadline=deadline,
        base_backtest_cfg=base_backtest_cfg,
    )
    horizon_models, horizon_metadata, multi_horizon_report = train_multi_horizon_models(
        train_df,
        validation_calibration_df,
        test_df,
        xs_train=xs_train,
        xs_valid=xs_valid,
        xs_test=xs_test,
        seed=seed,
        label_min_return=label_min_return,
        specs=multi_horizon_specs or [],
    )
    auxiliary_models = {**directional_models, **horizon_models}
    auxiliary_metadata = {**directional_metadata, **horizon_metadata}
    best_bundle.auxiliary_models = auxiliary_models
    best_bundle.auxiliary_metadata = auxiliary_metadata
    long_preflight = long_side_preflight_gate(directional_report)
    short_preflight = short_side_preflight_gate(directional_report)
    threshold_report = calibrate_directional_thresholds(
        validation_calibration_df,
        best_bundle,
        base_backtest_cfg,
        label_min_return=label_min_return,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
        deadline=deadline,
        force_compact=force_compact_threshold_search,
        long_preflight_gate=long_preflight,
        short_preflight_gate=short_preflight,
        max_threshold_evals=max_threshold_evals,
        validation_gate_df=validation_gate_df,
    )
    directional_gate = {
        "enabled": bool({"direction_trade", "direction_long", "direction_short"}.issubset(best_bundle.auxiliary_models)),
        "decision_dataset": str(threshold_report.get("gate_dataset", "validation")),
        "status": "not_applicable",
        "reason": "",
    }
    threshold_best_backtest = threshold_report.get("best", {}).get("backtest", {})
    if any(str(key).startswith("direction_") for key in best_bundle.auxiliary_models) and not directional_gate["enabled"]:
        best_bundle.auxiliary_models = {
            key: value for key, value in best_bundle.auxiliary_models.items() if not str(key).startswith("direction_")
        }
        best_bundle.auxiliary_metadata = {
            key: value for key, value in best_bundle.auxiliary_metadata.items() if not str(key).startswith("direction_")
        }
        directional_gate.update(
            {
                "status": "disabled_partial_auxiliary_set",
                "reason": "trade_long_short_auxiliary_models_not_all_available",
            }
        )
        directional_report["status"] = "disabled_partial_auxiliary_set"
        directional_report["validation_gate"] = directional_gate
        threshold_report = calibrate_directional_thresholds(
            validation_calibration_df,
            best_bundle,
            base_backtest_cfg,
            label_min_return=label_min_return,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
            deadline=deadline,
            force_compact=force_compact_threshold_search,
            long_preflight_gate=long_preflight,
            short_preflight_gate=short_preflight,
            max_threshold_evals=max_threshold_evals,
            validation_gate_df=validation_gate_df,
        )
    elif directional_gate["enabled"]:
        valid_return = float(threshold_best_backtest.get("total_return", 0.0))
        valid_profit_factor = float(threshold_best_backtest.get("profit_factor", 0.0))
        if valid_return <= 0.0 or valid_profit_factor < min_profit_factor:
            best_bundle.auxiliary_models = {
                key: value for key, value in best_bundle.auxiliary_models.items() if not str(key).startswith("direction_")
            }
            best_bundle.auxiliary_metadata = {
                key: value for key, value in best_bundle.auxiliary_metadata.items() if not str(key).startswith("direction_")
            }
            directional_gate.update(
                {
                    "status": "disabled_by_validation_gate",
                    "reason": "directional_auxiliary_validation_backtest_failed",
                    "validation_return": valid_return,
                    "validation_profit_factor": valid_profit_factor,
                }
            )
            directional_report["status"] = "disabled_by_validation_gate"
            directional_report["validation_gate"] = directional_gate
            threshold_report = calibrate_directional_thresholds(
                validation_calibration_df,
                best_bundle,
                base_backtest_cfg,
                label_min_return=label_min_return,
                min_trades=min_trades,
                max_drawdown_floor=max_drawdown_floor,
                min_profit_factor=min_profit_factor,
                deadline=deadline,
                force_compact=force_compact_threshold_search,
                long_preflight_gate=long_preflight,
                short_preflight_gate=short_preflight,
                max_threshold_evals=max_threshold_evals,
                validation_gate_df=validation_gate_df,
            )
        else:
            directional_gate.update(
                {
                    "status": "enabled",
                    "reason": "directional_auxiliary_validation_backtest_passed",
                    "validation_return": valid_return,
                    "validation_profit_factor": valid_profit_factor,
                }
            )
            directional_report["validation_gate"] = directional_gate
    directional_gate["decision_dataset"] = str(threshold_report.get("gate_dataset", "validation"))
    best_thresholds = threshold_report.get("best") or {}
    optimized_backtest_cfg = BacktestConfig(
        **(
            best_thresholds.get("backtest_config")
            or {
                **asdict(base_backtest_cfg),
                "long_threshold": float(best_thresholds.get("long_threshold", base_backtest_cfg.long_threshold)),
                "short_threshold": float(best_thresholds.get("short_threshold", base_backtest_cfg.short_threshold)),
            }
        )
    )
    test_prob = best_bundle.model.predict_proba(xs_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_prob)
    best_bundle.metrics = {
        **best_bundle.metrics,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    test_backtest, test_detail = run_backtest(test_df, best_bundle, optimized_backtest_cfg)
    optimized_effective_long, optimized_effective_short = effective_thresholds(optimized_backtest_cfg)
    test_direction_prob = best_bundle.predict_direction_probabilities(x_test)
    optimized_short_signal_threshold = max(
        1.0 - float(optimized_effective_short),
        0.5 + float(optimized_backtest_cfg.min_confidence_gap),
    )
    test_uses_directional = bool(np.asarray(test_direction_prob.get("uses_directional_models", [])).any())
    test_capture_long_threshold = (
        auxiliary_signal_threshold(best_bundle, "direction_long", optimized_effective_long)
        if test_uses_directional
        else optimized_effective_long
    )
    test_capture_short_threshold = (
        auxiliary_signal_threshold(best_bundle, "direction_short", optimized_short_signal_threshold)
        if test_uses_directional
        else optimized_short_signal_threshold
    )
    test_capture_trade_threshold = (
        auxiliary_signal_threshold(best_bundle, "direction_trade", optimized_backtest_cfg.trade_signal_threshold)
        if test_uses_directional
        else optimized_backtest_cfg.trade_signal_threshold
    )
    test_direction_prob_for_report = {
        **test_direction_prob,
        "trade_threshold": np.full(len(test_df), float(test_capture_trade_threshold), dtype=float),
        "long_threshold": np.full(len(test_df), float(test_capture_long_threshold), dtype=float),
        "short_signal_threshold": np.full(len(test_df), float(test_capture_short_threshold), dtype=float),
    }
    test_large_metrics = directional_signal_capture_metrics(
        test_df,
        test_direction_prob_for_report,
        long_threshold=test_capture_long_threshold,
        short_threshold=optimized_effective_short,
        cutoff=large_move_cutoff(test_df, label_min_return),
        neutral_cutoff=label_min_return,
    )
    if time.monotonic() < deadline - 5.0:
        expected_return_model, expected_return_report = train_lightgbm_expected_return(
            xs_train,
            pd.to_numeric(
                train_df.get(
                    "future_return",
                    pd.Series(np.nan, index=train_df.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=float),
            xs_valid,
            pd.to_numeric(
                validation_calibration_df.get(
                    "future_return",
                    pd.Series(np.nan, index=validation_calibration_df.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=float),
            seed=seed,
            sample_weight=base_train_weights,
        )
    else:
        expected_return_model = None
        expected_return_report = {
            "status": "skipped",
            "reason": "no_remaining_auxiliary_budget",
            "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
            "model_kind": "regressor",
            "target": "future_return",
            "test_used_for_training_or_selection": False,
        }
    if expected_return_model is not None:
        best_bundle.auxiliary_models["alpha_expected_return"] = (
            expected_return_model
        )
        best_bundle.auxiliary_metadata["alpha_expected_return"] = (
            expected_return_report
        )
    lightgbm_available = importlib.util.find_spec("lightgbm") is not None
    alpha_model_report = {
        "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
        "training_stage": "after_model_threshold_and_test_evaluation_frozen",
        "classifier": {
            "status": (
                "candidate_pool_enabled"
                if lightgbm_available
                else "skipped"
            ),
            "reason": (
                ""
                if lightgbm_available
                else "missing_dependency: lightgbm"
            ),
            "lightgbm_dependency_available": lightgbm_available,
            "selected_model": best_bundle.model_name,
            "selected_model_is_lightgbm": best_bundle.model_name.startswith(
                "lightgbm_"
            ),
            "probability_output": "p_up_p_down",
        },
        "regressor": expected_return_report,
        "ranker": lightgbm_ranker_availability_report(),
        "test_used_for_auxiliary_training_or_selection": False,
        "strategy_or_threshold_changed_by_regressor": False,
    }
    ranking = [item[2] for item in scored]
    predictive_ranking = [item[2] for item in selection_ranking.predictive]
    validation_trading_gate_summary = {
        "hard_gate_enabled": True,
        "model_candidate_selection_dataset": "validation_calibration",
        "threshold_selection_dataset": str(threshold_report.get("selection_dataset", "validation_full")),
        "threshold_gate_dataset": str(threshold_report.get("gate_dataset", "same_as_selection")),
        "separate_threshold_gate_enabled": bool(threshold_report.get("separate_gate_enabled", False)),
        "passed_candidates": int(len(valid_scored)),
        "candidate_count": int(len(scored)),
        "fallback_to_best_accuracy_when_none_pass": False,
        "research_model_selected_when_none_pass": bool(not valid_scored),
        "rejected_by_validation_trading_gate": bool(not valid_scored),
        "policy": "Model candidates that pass validation trading gates are selected before accuracy-only candidates; failing candidates are research-only.",
    }
    winning_weight_variant = str(best_bundle.config.get("weight_variant", "ensemble_or_unknown"))
    report = {
        "objective": objective,
        "feature_version": FEATURE_VERSION,
        "primary_target_col": selected_primary_target_col,
        "ranking": ranking,
        "predictive_ranking": predictive_ranking,
        "strategy_gated_ranking": ranking,
        "model_selection_contract": selection_ranking.audit(),
        "valid_ranked_for_selection": [
            item[2] for item in valid_scored
        ],
        "validation_split": validation_split_report,
        "validation_calibration_split": validation_split_report,
        "validation_trading_gate_summary": validation_trading_gate_summary,
        "model_selection_gate": model_selection_gate,
        "skipped": skipped,
        "best_model": best_bundle.model_name,
        "winning_weight_variant": winning_weight_variant,
        "test_metrics": test_metrics,
        "test_evaluation_audit": {
            "test_evaluation_stage": "after_model_threshold_gate_finalized",
            "test_evaluation_count": 1,
            "test_used_for_model_selection": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_publish_gate": True,
            "test_policy": "test split is audit-only for model/threshold selection, but is a final hard safety gate before publishing a candidate model",
        },
        "label_distribution": label_distribution_report(train_df, valid_df, test_df),
        "test_large_move_metrics": test_large_metrics,
        "directional_signal_report": directional_report,
        "long_side_preflight_gate": long_preflight,
        "short_side_preflight_gate": short_preflight,
        "directional_auxiliary_gate": directional_gate,
        "tradeability_model_status": str(directional_gate.get("status", "not_applicable")),
        "test_backtest": asdict(test_backtest),
        "optimized_backtest_config": asdict(optimized_backtest_cfg),
        "threshold_optimization": threshold_report,
        "small_account_strategy": {
            "enabled": True,
            "margin_type": "ISOLATED",
            "position_sizing": "confidence_atr_liquidity_scaled",
            "strategy_calibration_engine_contract_version": (
                threshold_report.get(
                    "strategy_calibration_engine_contract_version"
                )
            ),
            "risk_profile_catalog_schema_version": threshold_report.get(
                "risk_profile_catalog_schema_version"
            ),
            "risk_profile_catalog_version": threshold_report.get(
                "risk_profile_catalog_version"
            ),
            "risk_profile_catalog_path": threshold_report.get(
                "risk_profile_catalog_path"
            ),
            "risk_profiles_searched": threshold_report.get(
                "risk_profiles_searched",
                [item["name"] for item in small_account_risk_profiles(base_backtest_cfg)],
            ),
            "side_policies_requested": threshold_report.get("side_policies_requested", trade_side_policy_grid(base_backtest_cfg)),
            "side_policies_searched": threshold_report.get("side_policies_searched", trade_side_policy_grid(base_backtest_cfg)),
            "long_candidate_allowed": threshold_report.get("long_candidate_allowed", False),
            "long_candidate_blockers": threshold_report.get("long_candidate_blockers", []),
            "short_candidate_allowed": threshold_report.get("short_candidate_allowed", False),
            "short_candidate_blockers": threshold_report.get("short_candidate_blockers", []),
            "selected_risk_profile": best_thresholds.get("risk_profile"),
            "selected_side_policy": best_thresholds.get("side_policy") or optimized_backtest_cfg.trade_side_policy,
            "cost_penalty": "enabled",
        },
        "candidate_count": len(scored),
        "skipped_count": len(skipped),
        "event_balanced_sampling": event_balance_report,
        "cost_edge_balanced_sampling": edge_balance_report,
        "event_sampling_variants": weight_variant_reports,
        "volatility_regime_event_sampling": regime_event_balance_report,
        "complexity": complexity,
        "neural_network_priority": "enabled" if complexity in {"deep", "blackbox"} else "disabled",
        "transformer_policy": "blackbox_only",
        "rolling_folds": rolling_folds,
        "multi_horizon": multi_horizon_report,
        "alpha_models": alpha_model_report,
        "anti_overfit": {
            "overfit_gap_penalty": "enabled",
            "rolling_validation_penalty": "enabled" if rolling_folds > 0 else "disabled",
            "early_stopping_for_neural_models": "enabled",
        },
    }
    best_bundle.config = {
        **best_bundle.config,
        "objective": objective,
        "feature_version": FEATURE_VERSION,
        "candidate_count": len(scored),
        "skipped_count": len(skipped),
        "long_threshold": optimized_backtest_cfg.long_threshold,
        "short_threshold": optimized_backtest_cfg.short_threshold,
        "trade_side_policy": optimized_backtest_cfg.trade_side_policy,
        "small_account_risk_profile": best_thresholds.get("risk_profile", "base_small_account"),
        "model_selection_rejected_by_validation_trading_gate": bool(model_selection_gate["rejected_by_validation_trading_gate"]),
        "threshold_valid_gate_passed_count": int(threshold_report.get("valid_gate_passed_count", 0)),
        "threshold_selection_dataset": str(threshold_report.get("selection_dataset", "")),
        "threshold_gate_dataset": str(threshold_report.get("gate_dataset", "")),
        "threshold_calibration_rows": int(threshold_report.get("calibration_rows", 0)),
        "threshold_gate_rows": int(threshold_report.get("gate_rows", 0)),
        "large_move_weighting": "enabled",
        "event_balanced_sampling": "enabled" if event_balance_report.get("enabled") else "disabled",
        "volatility_regime_event_sampling": "enabled" if regime_event_balance_report.get("enabled") else "disabled",
        "winning_weight_variant": winning_weight_variant,
        "target_family": "cost_edge_profit_first",
        "cost_edge_label_version": "v1",
        "complexity": complexity,
        "neural_network_priority": "enabled" if complexity in {"deep", "blackbox"} else "disabled",
        "transformer_policy": "blackbox_only",
        "rolling_folds": rolling_folds,
        "anti_overfit": "enabled",
        "multi_horizon_enabled": bool(
            [
                key
                for key in best_bundle.auxiliary_models.keys()
                if not str(key).startswith(("direction_", "alpha_"))
            ]
        ),
        "multi_horizon_keys": ",".join(
            key
            for key in best_bundle.auxiliary_models.keys()
            if not str(key).startswith(("direction_", "alpha_"))
        ),
        "alpha_model_interface_version": ALPHA_MODEL_INTERFACE_VERSION,
        "alpha_expected_return_model": str(
            expected_return_report.get("model_name", "")
        ),
        "alpha_expected_return_status": str(
            expected_return_report.get("status", "skipped")
        ),
        "directional_auxiliary_status": str(directional_gate.get("status", "not_applicable")),
        "tradeability_model_status": str(directional_gate.get("status", "not_applicable")),
        "tradeability_model_present": bool("direction_trade" in best_bundle.auxiliary_models),
        "tradeability_model_target_col": str(
            best_bundle.auxiliary_metadata.get("direction_trade", {}).get("target", "actionable_label")
            if "direction_trade" in best_bundle.auxiliary_metadata
            else "actionable_label"
        ),
        "tradeability_model_removed_from_bundle": not bool("direction_trade" in best_bundle.auxiliary_models),
    }
    return best_bundle, report, test_detail


def optimize_symbol_interval(
    symbol: str,
    interval: str,
    cfg: TraderConfig,
    *,
    include_realtime: bool,
    max_model_trials: int,
    deadline: float,
    initial_balance: float,
    min_trades: int,
    max_drawdown_limit: float,
    min_profit_factor: float,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_training_rows: int = 60_000,
    objective: str = "return",
) -> dict[str, Any]:
    symbol = symbol.upper()
    raw = load_symbol_interval(cfg.data_dir, symbol, interval, include_realtime=include_realtime)
    data_validation = dict(raw.attrs.get("data_validation", {}))
    market_context = dict(raw.attrs.get("market_context", {}))
    raw_rows_before_window = len(raw)
    max_training_rows = int(max_training_rows or 0)
    if max_training_rows > 0 and len(raw) > max_training_rows:
        raw = raw.tail(max_training_rows).reset_index(drop=True)
    label_horizon = primary_label_horizon(interval, cfg)
    label_min_return = primary_label_min_return(interval, cfg)
    frame = make_features(raw, label_horizon=label_horizon, label_min_return=label_min_return)
    split_purge_rows = max(int(label_horizon), int(max(MULTI_HORIZON_STEPS)))
    train_df, valid_df, test_df = time_split(
        frame,
        cfg.train_fraction,
        cfg.validation_fraction,
        purge_rows=split_purge_rows,
    )
    _, _, feature_columns = feature_matrix(train_df)
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
    steps_by_interval = {
        "5m": [(1, 5), (2, 10), (3, 15), (6, 30), (12, 60)],
        "15m": [(1, 15), (2, 30), (4, 60), (16, 240)],
        "1h": [(1, 60), (3, 180), (4, 240), (6, 360), (12, 720)],
    }
    multi_horizon_specs = [
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
    bundle, model_report, test_detail = train_accuracy_first_candidates(
        train_df,
        valid_df,
        test_df,
        feature_columns,
        seed=cfg.random_seed,
        max_model_trials=max_model_trials,
        deadline=deadline,
        min_trades=min_trades,
        max_drawdown_floor=-abs(max_drawdown_limit),
        min_profit_factor=min_profit_factor,
        base_backtest_cfg=base_backtest_cfg,
        label_min_return=label_min_return,
        complexity=complexity,
        rolling_folds=rolling_folds,
        multi_horizon_specs=multi_horizon_specs,
        objective=objective,
        validation_split_purge_rows=split_purge_rows,
    )
    model_label = "return_ai" if objective == "return" else "accuracy_ai"
    model_path = cfg.model_dir / f"{symbol}_{interval}_{model_label}.pkl"
    candidate_model_path = cfg.model_dir / f"{symbol}_{interval}_{model_label}_candidate.pkl"
    bundle.save(candidate_model_path)
    detail_path = cfg.reports_dir / f"{symbol}_{interval}_model_optimization_backtest.csv"
    test_detail.to_csv(detail_path, index=False)
    test_backtest = model_report.get("test_backtest", {}) if isinstance(model_report.get("test_backtest"), dict) else {}
    test_total_return = float(test_backtest.get("total_return", 0.0))
    test_profit_factor = float(test_backtest.get("profit_factor", 0.0))
    test_trades = int(test_backtest.get("trades", 0))
    test_drawdown = float(test_backtest.get("max_drawdown", 0.0))
    test_realized_rr = float(test_backtest.get("realized_avg_win_loss_ratio", 0.0))
    model_selection_gate = model_report.get("model_selection_gate", {}) if isinstance(model_report.get("model_selection_gate"), dict) else {}
    model_rejected_by_validation_gate = bool(model_selection_gate.get("rejected_by_validation_trading_gate", False))
    threshold_best_gate = (
        (model_report.get("threshold_optimization") or {})
        .get("best", {})
        .get("validation_trading_gate", {})
    )
    threshold_timed_out = bool((model_report.get("threshold_optimization") or {}).get("timed_out", False))
    final_validation_gate_passed = bool(threshold_best_gate.get("passed", False))
    min_realized_rr = 0.90
    single_test_gate_passed = bool(
        test_total_return > 0.0
        and test_profit_factor >= min_profit_factor
        and test_trades >= min_trades
        and test_drawdown >= -abs(max_drawdown_limit)
        and test_realized_rr >= min_realized_rr
    )
    short_interval_extra_required = interval_requires_extra_publish_gates(interval)
    min_regime_trades = max(3, int(np.ceil(max(min_trades, 1) / 3.0)))
    volatility_regime_report = volatility_regime_backtest_report(
        train_df,
        test_detail,
        min_profit_factor=min_profit_factor,
        max_drawdown_floor=-abs(max_drawdown_limit),
        min_regime_trades=min_regime_trades,
    )
    volatility_regime_report["required_for_publish"] = bool(short_interval_extra_required)
    strategy_validation_gate = {
        "required_for_publish": bool(short_interval_extra_required),
        "passed": not short_interval_extra_required,
        "source": "not_required_for_interval" if not short_interval_extra_required else "strategy_validate_required_after_candidate",
        "reason": "" if not short_interval_extra_required else "short_interval_model_optimize_reports_candidate_only",
    }
    publish_allowed = bool(
        single_test_gate_passed
        and not model_rejected_by_validation_gate
        and final_validation_gate_passed
        and not threshold_timed_out
        and (not short_interval_extra_required)
        and (not volatility_regime_report.get("required_for_publish") or volatility_regime_report.get("gate_passed"))
    )
    if publish_allowed:
        publish_reason = "final_threshold_and_test_hard_gates_passed"
    elif threshold_timed_out:
        publish_reason = "threshold_search_timed_out_candidate_only"
    elif model_rejected_by_validation_gate:
        publish_reason = "model_rejected_by_validation_trading_gate"
    elif short_interval_extra_required:
        publish_reason = "short_interval_requires_walk_forward_strategy_validation"
    else:
        publish_reason = "test_hard_gates_failed"
    if publish_allowed:
        bundle.save(model_path)
    report = {
        "created_utc": beijing_now_iso(),
        "created_beijing": beijing_now_iso(),
        "symbol": symbol,
        "interval": interval,
        "objective": objective,
        "include_realtime": include_realtime,
        "feature_version": FEATURE_VERSION,
        "primary_label_horizon": label_horizon,
        "primary_label_min_return": label_min_return,
        "configured_label_min_return": cfg.label_min_return,
        "split_purge_rows": split_purge_rows,
        "split_purge_policy": "drop_tail_rows_from_train_and_validation_to_prevent_future_label_overlap",
        "complexity": complexity,
        "rolling_folds": rolling_folds,
        "rows": len(frame),
        "raw_rows": len(raw),
        "raw_rows_before_window": raw_rows_before_window,
        "data_validation": data_validation,
        "market_context": market_context,
        "max_training_rows": max_training_rows,
        "training_window_applied": bool(max_training_rows > 0 and raw_rows_before_window > max_training_rows),
        "model_name": bundle.model_name,
        "model_path": str(model_path),
        "candidate_model_path": str(candidate_model_path),
        "model_publish": {
            "status": "published" if publish_allowed else ("candidate_only_rejected" if model_rejected_by_validation_gate else "candidate_only"),
            "reason": publish_reason,
            "source": "model_optimize_candidate_publish_guard",
            "model_rejected_by_validation_trading_gate": model_rejected_by_validation_gate,
            "final_threshold_validation_gate_passed": final_validation_gate_passed,
            "threshold_search_timed_out": threshold_timed_out,
            "threshold_selection_dataset": str((model_report.get("threshold_optimization") or {}).get("selection_dataset", "")),
            "threshold_gate_dataset": str((model_report.get("threshold_optimization") or {}).get("gate_dataset", "")),
            "threshold_calibration_rows": int((model_report.get("threshold_optimization") or {}).get("calibration_rows", 0)),
            "threshold_gate_rows": int((model_report.get("threshold_optimization") or {}).get("gate_rows", 0)),
            "threshold_gate_evaluated_count": int((model_report.get("threshold_optimization") or {}).get("validation_gate_evaluated_count", 0)),
            "single_test_gate_passed": single_test_gate_passed,
            "short_interval_extra_gates_required": short_interval_extra_required,
            "strategy_validation_gate_passed": bool(strategy_validation_gate["passed"]),
            "volatility_regime_gate_passed": bool(volatility_regime_report.get("gate_passed", False)),
            "test_total_return": test_total_return,
            "test_profit_factor": test_profit_factor,
            "test_trades": test_trades,
            "test_max_drawdown": test_drawdown,
            "test_realized_avg_win_loss_ratio": test_realized_rr,
            "required_min_realized_avg_win_loss_ratio": min_realized_rr,
            "required_min_profit_factor": min_profit_factor,
            "required_min_trades": min_trades,
            "required_max_drawdown_limit": -abs(max_drawdown_limit),
        },
        "publish_gate": {
            "source": "final_test_plus_interval_safety_gates",
            "final_threshold_validation_gate_passed": final_validation_gate_passed,
            "threshold_search_timed_out": threshold_timed_out,
            "raw_model_selection_gate": model_selection_gate,
            "threshold_best_validation_gate": threshold_best_gate,
            "single_test_gate_passed": single_test_gate_passed,
            "strategy_validation_gate": strategy_validation_gate,
            "volatility_regime_gate": volatility_regime_report,
            "test_used_for_selection": False,
            "live_trading_enabled": False,
        },
        "detail_path": str(detail_path),
        "model_metrics": bundle.metrics,
        "model_report": model_report,
        "optimized_backtest_config": model_report.get("optimized_backtest_config", {}),
        "threshold_optimization": model_report.get("threshold_optimization", {}),
        "small_account_strategy": model_report.get("small_account_strategy", {}),
        "live_trading_enabled": False,
    }
    report_path = cfg.reports_dir / f"{symbol}_{interval}_model_optimization.json"
    safe_replace_text(report_path, json.dumps(report, indent=2))
    report["report_path"] = str(report_path)
    return report


def run_model_optimization(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    intervals: list[str],
    target_pairs: list[tuple[str, str]] | None = None,
    include_realtime: bool,
    time_budget_minutes: float,
    max_model_trials: int,
    initial_balance: float,
    min_trades: int,
    max_drawdown_limit: float,
    min_profit_factor: float,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_training_rows: int = 60_000,
    objective: str = "return",
) -> dict[str, Any]:
    cfg.ensure_dirs()
    deadline = time.monotonic() + max(1.0, time_budget_minutes * 60.0)
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    raw_targets = target_pairs or [(symbol.upper(), interval) for symbol in symbols for interval in intervals]
    targets: list[tuple[str, str]] = []
    for symbol, interval in raw_targets:
        target = (symbol.upper(), interval)
        if target not in targets:
            targets.append(target)
    for symbol, interval in targets:
        if time.monotonic() >= deadline:
            skipped.append({"symbol": symbol, "interval": interval, "reason": "time_budget_exhausted"})
            continue
        try:
            report = optimize_symbol_interval(
                symbol,
                interval,
                cfg,
                include_realtime=include_realtime,
                max_model_trials=max_model_trials,
                deadline=deadline,
                initial_balance=initial_balance,
                min_trades=min_trades,
                max_drawdown_limit=max_drawdown_limit,
                min_profit_factor=min_profit_factor,
                complexity=complexity,
                rolling_folds=rolling_folds,
                max_training_rows=max_training_rows,
                objective=objective,
            )
            items.append(report)
        except Exception as exc:
            skipped.append({"symbol": symbol, "interval": interval, "reason": str(exc)})

    ranked_for_selection = sorted(
        items,
        key=lambda item: (
            item["model_report"].get("ranking", [{}])[0].get("selection_score", 0.0)
            if item["model_report"].get("ranking")
            else 0.0,
            item["model_report"].get("ranking", [{}])[0].get("base_score", 0.0)
            if item["model_report"].get("ranking")
            else 0.0,
        ),
        reverse=True,
    )
    valid_ranked_for_selection = [
        item
        for item in ranked_for_selection
        if bool(
            item.get("model_report", {})
            .get("model_selection_gate", {})
            .get("passed", False)
        )
    ]
    test_final_audit_ranked = sorted(
        items,
        key=lambda item: (
            item["model_report"]["test_backtest"].get("total_return", 0.0),
            item["model_report"]["test_backtest"].get("profit_factor", 0.0),
            -abs(item["model_report"]["test_backtest"].get("max_drawdown", 0.0)),
        ),
        reverse=True,
    )
    aggregate = {
        "created_utc": beijing_now_iso(),
        "created_beijing": beijing_now_iso(),
        "objective": objective,
        "feature_version": FEATURE_VERSION,
        "complexity": complexity,
        "rolling_folds": rolling_folds,
        "max_training_rows": int(max_training_rows or 0),
        "include_realtime": include_realtime,
        "time_budget_minutes": time_budget_minutes,
        "max_model_trials": max_model_trials,
        "target_pairs": [{"symbol": symbol, "interval": interval} for symbol, interval in targets],
        "items": items,
        "ranked": ranked_for_selection,
        "valid_ranked_for_selection": valid_ranked_for_selection,
        "test_final_audit_ranked": test_final_audit_ranked,
        "ranking_policy": {
            "ranked_uses": "validation_selection_score",
            "test_used_for_model_selection": False,
            "test_used_for_publish_gate": True,
            "test_policy": "test ranking is audit-only for selection; per-target publish guards may still require final test hard gates",
            "test_used_for_selection": False,
        },
        "skipped_targets": skipped,
        "live_trading_enabled": False,
    }
    output = cfg.reports_dir / f"model_optimization_{beijing_stamp()}.json"
    safe_replace_text(output, json.dumps(aggregate, indent=2))
    aggregate["report_path"] = str(output)
    return aggregate
