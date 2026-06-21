from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .performance_report import evaluate_backtest_performance


MONITORING_SCHEMA_VERSION = 2
MONITORING_ALGORITHM_VERSION = "2026-06-20-v3"


class MonitoringThresholds(Protocol):
    """Configuration required by model and strategy monitoring."""

    monitoring_recent_rows: int
    monitoring_psi_threshold: float
    monitoring_ks_threshold: float
    monitoring_min_confidence: float
    monitoring_max_ece: float
    monitoring_min_rolling_sharpe: float
    monitoring_max_drawdown: float
    monitoring_return_deviation: float
    monitoring_regime_shift_threshold: float


@dataclass(frozen=True)
class FeatureDrift:
    """Distribution shift for one model feature."""

    feature: str
    psi: float
    ks: float
    reference_rows: int
    current_rows: int


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability calibration and confidence summary."""

    rows: int
    brier_score: float
    expected_calibration_error: float
    mean_confidence: float
    low_confidence_rate: float


@dataclass(frozen=True)
class RollingPerformance:
    """Recent realized strategy behavior from a replay detail frame."""

    rows: int
    total_return: float
    rolling_sharpe: float
    max_drawdown: float
    profit_factor: float
    trades: int


@dataclass(frozen=True)
class RetrainingDecision:
    """Explainable retraining recommendation; never an order instruction."""

    triggered: bool
    severity: str
    reasons: tuple[str, ...]


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def population_stability_index(
    reference: Any,
    current: Any,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins."""

    ref = _finite(reference)
    cur = _finite(current)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, max(int(bins), 2) + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        center = float(ref[0])
        scale = max(abs(center), 1.0) * 1e-9
        edges = np.array(
            [-np.inf, center - scale, center + scale, np.inf],
            dtype=float,
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    epsilon = 1e-6
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1.0), epsilon, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1.0), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def ks_statistic(reference: Any, current: Any) -> float:
    """Compute the two-sample empirical KS statistic without SciPy."""

    ref = np.sort(_finite(reference))
    cur = np.sort(_finite(current))
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, values, side="right") / len(ref)
    cur_cdf = np.searchsorted(cur, values, side="right") / len(cur)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def expected_calibration_error(
    targets: Any,
    probabilities: Any,
    *,
    bins: int = 10,
) -> float:
    """Return weighted absolute calibration error."""

    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(int(bins), 2) + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    error = 0.0
    for idx in range(len(edges) - 1):
        mask = bucket == idx
        if not bool(mask.any()):
            continue
        error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(error)


def calibration_metrics(targets: Any, probabilities: Any) -> CalibrationMetrics:
    """Summarize calibration and prediction confidence."""

    y = np.asarray(targets, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = np.clip(p[valid], 0.0, 1.0)
    if len(y) == 0:
        return CalibrationMetrics(0, 0.0, 0.0, 0.0, 1.0)
    confidence = np.abs(p - 0.5) * 2.0
    return CalibrationMetrics(
        rows=int(len(y)),
        brier_score=float(np.mean((p - y) ** 2)),
        expected_calibration_error=expected_calibration_error(y, p),
        mean_confidence=float(confidence.mean()),
        low_confidence_rate=float((confidence < 0.10).mean()),
    )


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
) -> list[FeatureDrift]:
    """Calculate PSI and KS for shared model features."""

    report: list[FeatureDrift] = []
    for feature in feature_columns:
        if feature not in reference.columns or feature not in current.columns:
            continue
        ref = _finite(pd.to_numeric(reference[feature], errors="coerce"))
        cur = _finite(pd.to_numeric(current[feature], errors="coerce"))
        report.append(
            FeatureDrift(
                feature=feature,
                psi=population_stability_index(ref, cur),
                ks=ks_statistic(ref, cur),
                reference_rows=int(len(ref)),
                current_rows=int(len(cur)),
            )
        )
    return sorted(report, key=lambda item: max(item.psi, item.ks), reverse=True)


def build_feature_reference(
    reference: pd.DataFrame,
    feature_columns: list[str],
    *,
    psi_bins: int = 10,
    quantile_points: int = 101,
) -> dict[str, Any]:
    """Compress a reference population into PSI bins and KS quantiles."""

    profiles: dict[str, Any] = {}
    probabilities = np.linspace(0.0, 1.0, max(int(quantile_points), 3))
    for feature in feature_columns:
        if feature not in reference.columns:
            continue
        values = _finite(pd.to_numeric(reference[feature], errors="coerce"))
        if len(values) < 2:
            continue
        raw_edges = np.unique(
            np.quantile(values, np.linspace(0.0, 1.0, max(int(psi_bins), 2) + 1))
        )
        if len(raw_edges) < 2:
            center = float(values[0])
            scale = max(abs(center), 1.0) * 1e-9
            edges = np.array(
                [-np.inf, center - scale, center + scale, np.inf],
                dtype=float,
            )
        else:
            edges = raw_edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
        counts = np.histogram(values, bins=edges)[0].astype(float)
        shares = counts / max(counts.sum(), 1.0)
        profiles[feature] = {
            "reference_rows": int(len(values)),
            "psi_edges": [
                None if not np.isfinite(value) else float(value)
                for value in edges
            ],
            "psi_shares": shares.tolist(),
            "ks_probabilities": probabilities.tolist(),
            "ks_quantiles": np.quantile(values, probabilities).astype(float).tolist(),
        }
    return profiles


def feature_drift_from_reference(
    reference_profiles: dict[str, Any],
    current: pd.DataFrame,
) -> list[FeatureDrift]:
    """Calculate drift against a persisted compressed reference profile."""

    report: list[FeatureDrift] = []
    epsilon = 1e-6
    for feature, profile in reference_profiles.items():
        if feature not in current.columns or not isinstance(profile, dict):
            continue
        values = _finite(pd.to_numeric(current[feature], errors="coerce"))
        if len(values) < 2:
            continue
        edges = np.asarray(
            [
                -np.inf if value is None and idx == 0 else (
                    np.inf if value is None else float(value)
                )
                for idx, value in enumerate(profile.get("psi_edges") or [])
            ],
            dtype=float,
        )
        reference_shares = np.asarray(profile.get("psi_shares") or [], dtype=float)
        if len(edges) < 2 or len(reference_shares) != len(edges) - 1:
            continue
        current_counts = np.histogram(values, bins=edges)[0].astype(float)
        current_shares = np.clip(
            current_counts / max(current_counts.sum(), 1.0),
            epsilon,
            None,
        )
        reference_shares = np.clip(reference_shares, epsilon, None)
        psi = float(
            np.sum(
                (current_shares - reference_shares)
                * np.log(current_shares / reference_shares)
            )
        )
        probabilities = np.asarray(
            profile.get("ks_probabilities") or [],
            dtype=float,
        )
        quantiles = np.asarray(profile.get("ks_quantiles") or [], dtype=float)
        if len(probabilities) and len(probabilities) == len(quantiles):
            current_cdf = np.searchsorted(
                np.sort(values),
                quantiles,
                side="right",
            ) / len(values)
            ks = float(np.max(np.abs(current_cdf - probabilities)))
        else:
            ks = 0.0
        report.append(
            FeatureDrift(
                feature=str(feature),
                psi=psi,
                ks=ks,
                reference_rows=int(profile.get("reference_rows", 0) or 0),
                current_rows=int(len(values)),
            )
        )
    return sorted(report, key=lambda item: max(item.psi, item.ks), reverse=True)


def categorical_total_variation(reference: Any, current: Any) -> float:
    """Measure categorical distribution shift on a zero-to-one scale."""

    ref = pd.Series(reference).dropna().astype(str)
    cur = pd.Series(current).dropna().astype(str)
    if ref.empty or cur.empty:
        return 0.0
    categories = sorted(set(ref.unique()).union(cur.unique()))
    ref_share = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_share = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(ref_share.to_numpy() - cur_share.to_numpy()).sum())


def micro_regime_shift(reference: Any, current: Any) -> float:
    """Compare stable down/neutral/up buckets for the micro-trend score."""

    def bucket(values: Any) -> np.ndarray:
        numeric = np.asarray(values, dtype=float)
        return np.where(
            numeric < -0.05,
            "trend_down",
            np.where(numeric > 0.05, "trend_up", "neutral"),
        )

    return categorical_total_variation(bucket(reference), bucket(current))


def micro_regime_distribution(values: Any) -> dict[str, float]:
    """Return stable down/neutral/up shares for persistence."""

    numeric = np.asarray(values, dtype=float)
    bucket = np.where(
        numeric < -0.05,
        "trend_down",
        np.where(numeric > 0.05, "trend_up", "neutral"),
    )
    counts = pd.Series(bucket).value_counts(normalize=True)
    return {
        key: float(counts.get(key, 0.0))
        for key in ("trend_down", "neutral", "trend_up")
    }


def distribution_total_variation(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> float:
    """Compare two persisted categorical share mappings."""

    keys = sorted(set(reference).union(current))
    return float(
        0.5
        * sum(
            abs(float(reference.get(key, 0.0)) - float(current.get(key, 0.0)))
            for key in keys
        )
    )


def rolling_performance(
    detail: pd.DataFrame,
    *,
    window: int,
) -> RollingPerformance:
    """Summarize the most recent replay rows with interval-aware Sharpe."""

    recent = detail.tail(max(int(window), 1)).copy()
    if recent.empty or "strategy_return" not in recent.columns:
        return RollingPerformance(0, 0.0, 0.0, 0.0, 0.0, 0)
    evaluation_detail = recent.drop(columns=["equity"], errors="ignore")
    performance = evaluate_backtest_performance(
        evaluation_detail,
        initial_balance=1.0,
    )
    returns = (
        pd.to_numeric(recent["strategy_return"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else (999.0 if gross_profit > 0.0 else 0.0)
    )
    trades = (
        int((pd.to_numeric(recent["notional_turnover"], errors="coerce").fillna(0.0) > 0.0).sum())
        if "notional_turnover" in recent.columns
        else 0
    )
    return RollingPerformance(
        rows=int(len(recent)),
        total_return=performance.total_return,
        rolling_sharpe=performance.sharpe_like,
        max_drawdown=performance.max_drawdown,
        profit_factor=float(profit_factor),
        trades=trades,
    )


def backtest_deviation(
    baseline: dict[str, Any],
    recent: dict[str, Any],
) -> dict[str, float]:
    """Compare recent replay behavior with the frozen test backtest."""

    keys = ("total_return", "max_drawdown", "profit_factor", "win_rate")
    return {
        f"{key}_delta": float(recent.get(key, 0.0) or 0.0)
        - float(baseline.get(key, 0.0) or 0.0)
        for key in keys
    }


def evaluate_retraining(
    *,
    feature_drift: list[FeatureDrift],
    calibration: CalibrationMetrics,
    rolling: RollingPerformance,
    recent_deviation: dict[str, float] | None,
    paper_deviation: dict[str, float] | None,
    regime_shift: float,
    cfg: MonitoringThresholds,
) -> RetrainingDecision:
    """Apply explicit monitoring thresholds to produce retraining reasons."""

    reasons: list[str] = []
    max_psi = max((item.psi for item in feature_drift), default=0.0)
    max_ks = max((item.ks for item in feature_drift), default=0.0)
    if max_psi >= float(cfg.monitoring_psi_threshold):
        reasons.append("feature_psi_exceeded")
    if max_ks >= float(cfg.monitoring_ks_threshold):
        reasons.append("feature_ks_exceeded")
    if calibration.mean_confidence < float(cfg.monitoring_min_confidence):
        reasons.append("prediction_confidence_deteriorated")
    if calibration.expected_calibration_error > float(cfg.monitoring_max_ece):
        reasons.append("probability_calibration_deteriorated")
    if rolling.rows and rolling.rolling_sharpe < float(cfg.monitoring_min_rolling_sharpe):
        reasons.append("rolling_sharpe_below_threshold")
    if rolling.max_drawdown < -abs(float(cfg.monitoring_max_drawdown)):
        reasons.append("rolling_drawdown_exceeded")
    if recent_deviation and recent_deviation.get(
        "total_return_delta",
        0.0,
    ) < -abs(float(cfg.monitoring_return_deviation)):
        reasons.append("recent_return_below_equal_window_baseline")
    if paper_deviation and paper_deviation.get("total_return_delta", 0.0) < -abs(
        float(cfg.monitoring_return_deviation)
    ):
        reasons.append("paper_return_below_frozen_test")
    if regime_shift >= float(cfg.monitoring_regime_shift_threshold):
        reasons.append("market_regime_distribution_shift")
    severity = "none"
    if reasons:
        severity = "high" if len(reasons) >= 3 or "rolling_drawdown_exceeded" in reasons else "medium"
    return RetrainingDecision(bool(reasons), severity, tuple(reasons))


def build_monitoring_snapshot(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_columns: list[str],
    calibration_targets: Any,
    calibration_probabilities: Any,
    recent_detail: pd.DataFrame,
    baseline_backtest: dict[str, Any],
    recent_backtest: dict[str, Any],
    cfg: MonitoringThresholds,
    reference_profile: dict[str, Any] | None = None,
    frozen_backtest: dict[str, Any] | None = None,
    paper_metrics: dict[str, Any] | None = None,
    paper_status: str = "missing",
) -> dict[str, Any]:
    """Build one serializable model-monitoring snapshot."""

    drift = (
        feature_drift_from_reference(
            dict(reference_profile.get("features") or {}),
            current_frame,
        )
        if reference_profile
        else feature_drift_report(
            reference_frame,
            current_frame,
            feature_columns,
        )
    )
    calibration = calibration_metrics(
        calibration_targets,
        calibration_probabilities,
    )
    rolling = rolling_performance(
        recent_detail,
        window=int(cfg.monitoring_recent_rows),
    )
    deviation = backtest_deviation(baseline_backtest, recent_backtest)
    paper_deviation = (
        backtest_deviation(frozen_backtest or {}, paper_metrics)
        if paper_status == "comparable" and paper_metrics
        else None
    )
    if "micro_trend_regime" in current_frame.columns and reference_profile:
        regime_shift = distribution_total_variation(
            dict(reference_profile.get("micro_regime_distribution") or {}),
            micro_regime_distribution(current_frame["micro_trend_regime"]),
        )
    elif (
        "micro_trend_regime" in reference_frame.columns
        and "micro_trend_regime" in current_frame.columns
    ):
        regime_shift = micro_regime_shift(
            reference_frame["micro_trend_regime"],
            current_frame["micro_trend_regime"],
        )
    else:
        regime_shift = 0.0
    retraining = evaluate_retraining(
        feature_drift=drift,
        calibration=calibration,
        rolling=rolling,
        recent_deviation=(
            deviation
            if "total_return" in baseline_backtest
            and "total_return" in recent_backtest
            else None
        ),
        paper_deviation=paper_deviation,
        regime_shift=regime_shift,
        cfg=cfg,
    )
    return {
        "schema_version": MONITORING_SCHEMA_VERSION,
        "algorithm_version": MONITORING_ALGORITHM_VERSION,
        "feature_drift": {
            "max_psi": max((item.psi for item in drift), default=0.0),
            "max_ks": max((item.ks for item in drift), default=0.0),
            "drifted_feature_count": int(
                sum(
                    item.psi >= float(cfg.monitoring_psi_threshold)
                    or item.ks >= float(cfg.monitoring_ks_threshold)
                    for item in drift
                )
            ),
            "top_features": [asdict(item) for item in drift[:12]],
        },
        "calibration": asdict(calibration),
        "rolling_performance": asdict(rolling),
        "recent_replay_vs_equal_window_baseline": deviation,
        "paper_vs_frozen_test": {
            "status": paper_status,
            "deviation": paper_deviation,
        },
        "regime_distribution_shift": regime_shift,
        "retraining": asdict(retraining),
        "selection_note": (
            "Monitoring is diagnostic. Retraining may be prioritized, but test "
            "results remain evaluation-only and live trading stays disabled."
        ),
    }
