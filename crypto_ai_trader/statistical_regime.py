from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .contracts import MarketRegime

try:
    from sklearn.cluster import KMeans as _SklearnKMeans
except Exception:  # pragma: no cover - exercised through the explicit fallback test
    _SklearnKMeans = None


STATISTICAL_REGIME_VERSION = "2026-06-20-walk-forward-kmeans-v1"
STATISTICAL_REGIME_FEATURES = (
    "return_24",
    "trend_strength_12_48",
    "efficiency_ratio_48",
    "atr_14",
    "quote_volume_z",
    "liquidity_quality_score",
)


class StatisticalRegimeConfig(Protocol):
    """Configuration fields consumed by the optional statistical detector."""

    trend_gate_min_efficiency: float
    regime_statistical_clusters: int
    regime_statistical_min_history: int
    regime_statistical_lookback: int
    regime_statistical_refit_interval: int
    regime_statistical_random_seed: int


@dataclass(frozen=True)
class StatisticalRegimeResult:
    """Walk-forward statistical assignments plus an aggregate availability flag."""

    frame: pd.DataFrame
    model_available: bool
    fallback_reason: str


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    defaults = {
        "return_24": 0.0,
        "trend_strength_12_48": 0.0,
        "efficiency_ratio_48": 0.0,
        "atr_14": 0.0,
        "quote_volume_z": 0.0,
        "liquidity_quality_score": 0.5,
    }
    columns: list[np.ndarray] = []
    for name in STATISTICAL_REGIME_FEATURES:
        source = (
            frame[name]
            if name in frame.columns
            else pd.Series(defaults[name], index=frame.index)
        )
        values = (
            pd.to_numeric(source, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(defaults[name])
            .to_numpy(dtype=float)
        )
        columns.append(values)
    return np.column_stack(columns)


def _cluster_regime_map(
    centers: np.ndarray,
    training_values: np.ndarray,
    *,
    trend_efficiency_threshold: float,
) -> dict[int, str]:
    trend_index = STATISTICAL_REGIME_FEATURES.index("trend_strength_12_48")
    efficiency_index = STATISTICAL_REGIME_FEATURES.index("efficiency_ratio_48")
    atr_index = STATISTICAL_REGIME_FEATURES.index("atr_14")
    atr_high_cut = float(np.quantile(training_values[:, atr_index], 0.67))
    mapping: dict[int, str] = {}
    for cluster_id, center in enumerate(centers):
        trend = float(center[trend_index])
        efficiency = float(center[efficiency_index])
        atr = float(center[atr_index])
        if atr > atr_high_cut and atr > 0.0:
            regime = MarketRegime.HIGH_VOL.value
        elif efficiency >= trend_efficiency_threshold and trend > 0.0:
            regime = MarketRegime.TREND_UP.value
        elif efficiency >= trend_efficiency_threshold and trend < 0.0:
            regime = MarketRegime.TREND_DOWN.value
        else:
            regime = MarketRegime.RANGE.value
        mapping[cluster_id] = regime
    return mapping


def _distance_confidence(model: object, values: np.ndarray) -> np.ndarray:
    distances = np.asarray(model.transform(values), dtype=float)
    ordered = np.sort(distances, axis=1)
    nearest = ordered[:, 0]
    second = ordered[:, 1]
    return np.clip(1.0 - nearest / np.maximum(second, 1e-12), 0.0, 1.0)


def walk_forward_kmeans_regimes(
    frame: pd.DataFrame,
    cfg: StatisticalRegimeConfig,
) -> StatisticalRegimeResult:
    """Assign KMeans regimes using only observations preceding each prediction block."""

    output = pd.DataFrame(
        {
            "statistical_regime": pd.Series(pd.NA, index=frame.index, dtype="object"),
            "statistical_regime_confidence": np.zeros(len(frame), dtype=float),
            "statistical_regime_cluster": pd.Series(pd.NA, index=frame.index, dtype="Int64"),
            "statistical_regime_fallback_reason": np.full(
                len(frame),
                "insufficient_history",
                dtype=object,
            ),
        },
        index=frame.index,
    )
    if frame.empty:
        return StatisticalRegimeResult(
            frame=output,
            model_available=_SklearnKMeans is not None,
            fallback_reason="empty_frame",
        )
    if _SklearnKMeans is None:
        output["statistical_regime_fallback_reason"] = "sklearn_unavailable"
        return StatisticalRegimeResult(
            frame=output,
            model_available=False,
            fallback_reason="sklearn_unavailable",
        )

    values = _feature_matrix(frame)
    clusters = int(getattr(cfg, "regime_statistical_clusters", 4))
    min_history = int(getattr(cfg, "regime_statistical_min_history", 240))
    lookback = int(getattr(cfg, "regime_statistical_lookback", 720))
    refit_interval = int(getattr(cfg, "regime_statistical_refit_interval", 24))
    random_seed = int(getattr(cfg, "regime_statistical_random_seed", 42))
    trend_threshold = max(
        float(getattr(cfg, "trend_gate_min_efficiency", 0.18)),
        1e-9,
    )
    successful_blocks = 0

    for block_start in range(min_history, len(frame), refit_interval):
        block_end = min(block_start + refit_interval, len(frame))
        training_start = max(0, block_start - lookback)
        training_values = values[training_start:block_start]
        if (
            len(training_values) < min_history
            or len(np.unique(training_values, axis=0)) < clusters
        ):
            output.iloc[
                block_start:block_end,
                output.columns.get_loc("statistical_regime_fallback_reason"),
            ] = "insufficient_distinct_history"
            continue

        mean = training_values.mean(axis=0)
        scale = training_values.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        scaled_training = (training_values - mean) / scale
        scaled_prediction = (values[block_start:block_end] - mean) / scale
        try:
            model = _SklearnKMeans(
                n_clusters=clusters,
                random_state=random_seed,
                n_init=10,
            )
            model.fit(scaled_training)
            cluster_ids = model.predict(scaled_prediction).astype(int)
            centers = np.asarray(model.cluster_centers_, dtype=float) * scale + mean
            regime_map = _cluster_regime_map(
                centers,
                training_values,
                trend_efficiency_threshold=trend_threshold,
            )
            confidence = _distance_confidence(model, scaled_prediction)
        except Exception:
            output.iloc[
                block_start:block_end,
                output.columns.get_loc("statistical_regime_fallback_reason"),
            ] = "kmeans_fit_failed"
            continue

        block_index = output.index[block_start:block_end]
        output.loc[block_index, "statistical_regime"] = [
            regime_map[int(cluster_id)] for cluster_id in cluster_ids
        ]
        output.loc[block_index, "statistical_regime_confidence"] = confidence
        output.loc[block_index, "statistical_regime_cluster"] = cluster_ids
        output.loc[block_index, "statistical_regime_fallback_reason"] = ""
        successful_blocks += 1

    return StatisticalRegimeResult(
        frame=output,
        model_available=True,
        fallback_reason="" if successful_blocks else "no_successful_fit",
    )
