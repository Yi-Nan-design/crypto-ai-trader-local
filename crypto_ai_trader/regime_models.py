from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from typing import Protocol

import numpy as np


REGIME_MODEL_INTERFACE_VERSION = "2026-06-21-v1"


class RegimeModel(Protocol):
    """Optional fitted regime-model interface for future walk-forward adapters."""

    name: str

    def fit(self, x: np.ndarray) -> "RegimeModel": ...

    def predict(self, x: np.ndarray) -> np.ndarray: ...

    def predict_confidence(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RegimeModelSpec:
    """Availability and lifecycle metadata for one regime method."""

    name: str
    method: str
    status: str
    dependency: str | None
    dependency_available: bool
    causal_fit_required: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["interface_version"] = REGIME_MODEL_INTERFACE_VERSION
        return payload


def regime_model_registry() -> list[dict[str, object]]:
    """List implemented and reserved regime models with optional dependencies."""

    sklearn_available = importlib.util.find_spec("sklearn") is not None
    return [
        RegimeModelSpec(
            "rule_based_crash_detector",
            "rule_based",
            "implemented",
            None,
            True,
            True,
            "Causal return thresholds with highest-priority risk-off output.",
        ).to_dict(),
        RegimeModelSpec(
            "liquidity_state_detector",
            "rule_based",
            "implemented",
            None,
            True,
            True,
            "Causal volume and liquidity-quality thresholds.",
        ).to_dict(),
        RegimeModelSpec(
            "volatility_regime_classifier",
            "rule_based_quantiles",
            "implemented",
            None,
            True,
            True,
            "Trailing ATR quantiles only.",
        ).to_dict(),
        RegimeModelSpec(
            "walk_forward_kmeans",
            "kmeans",
            "implemented" if sklearn_available else "optional_unavailable",
            "sklearn",
            sklearn_available,
            True,
            "Normalization and fit window must end before prediction blocks.",
        ).to_dict(),
        RegimeModelSpec(
            "gaussian_mixture_regime",
            "gaussian_mixture",
            "interface_reserved" if sklearn_available else "optional_unavailable",
            "sklearn",
            sklearn_available,
            True,
            "Requires walk-forward component-to-regime mapping.",
        ).to_dict(),
        RegimeModelSpec(
            "hidden_markov_regime",
            "hmm",
            "interface_reserved"
            if importlib.util.find_spec("hmmlearn") is not None
            else "optional_unavailable",
            "hmmlearn",
            importlib.util.find_spec("hmmlearn") is not None,
            True,
            "Must use filtered state probabilities, never smoothed future states.",
        ).to_dict(),
        RegimeModelSpec(
            "lightgbm_regime_classifier",
            "lightgbm_classifier",
            "interface_reserved"
            if importlib.util.find_spec("lightgbm") is not None
            else "optional_unavailable",
            "lightgbm",
            importlib.util.find_spec("lightgbm") is not None,
            True,
            "Training labels require a causal prior-window regime definition.",
        ).to_dict(),
    ]
