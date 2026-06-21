from __future__ import annotations

import importlib.util
from typing import Protocol
import warnings

import numpy as np


ALPHA_MODEL_INTERFACE_VERSION = "2026-06-21-v1"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class _LightGBMPortableState:
    """Serialize fitted LightGBM estimators without importing LightGBM on load."""

    model: object | None
    _lightgbm_model_string: str | None

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        model = state.get("model")
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "model_to_string"):
            state["_lightgbm_model_string"] = booster.model_to_string()
            state["model"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if "_lightgbm_model_string" not in self.__dict__:
            self._lightgbm_model_string = None

    def _prediction_model(self) -> object:
        if self.model is not None:
            return self.model
        model_string = getattr(self, "_lightgbm_model_string", None)
        if not model_string:
            raise RuntimeError("Alpha model is not fitted")
        if importlib.util.find_spec("lightgbm") is None:
            raise RuntimeError("missing_dependency: lightgbm")
        from lightgbm import Booster

        self.model = Booster(model_str=model_string)
        return self.model


class AlphaModel(Protocol):
    """Common fitted-model interface used by the Alpha layer."""

    name: str
    target_kind: str

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "AlphaModel": ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...

    def predict_expected_return(self, x: np.ndarray) -> np.ndarray | None: ...


class ClassifierAlphaAdapter(_LightGBMPortableState):
    """Expose a probability classifier through the shared Alpha interface."""

    target_kind = "binary_direction"

    def __init__(self, name: str, model: object):
        self.name = name
        self.model = model
        self._lightgbm_model_string: str | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ClassifierAlphaAdapter":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weight is None:
                self.model.fit(x, y)
            else:
                try:
                    self.model.fit(x, y, sample_weight=sample_weight)
                except TypeError:
                    self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self._prediction_model()
            if hasattr(model, "predict_proba"):
                probability = np.asarray(model.predict_proba(x), dtype=float)
                if probability.ndim != 2 or probability.shape[1] < 2:
                    raise ValueError(
                        "classifier predict_proba must return two columns"
                    )
                up = probability[:, -1]
            else:
                up = np.asarray(model.predict(x), dtype=float).reshape(-1)
        return np.column_stack([1.0 - up, up])

    def predict_expected_return(self, x: np.ndarray) -> None:
        return None


class RegressorAlphaAdapter(_LightGBMPortableState):
    """Expose a continuous-return model plus a bounded direction proxy."""

    target_kind = "expected_return"

    def __init__(self, name: str, model: object):
        self.name = name
        self.model = model
        self._lightgbm_model_string: str | None = None
        self.probability_scale_: float | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "RegressorAlphaAdapter":
        target = np.asarray(y, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weight is None:
                self.model.fit(x, target)
            else:
                try:
                    self.model.fit(x, target, sample_weight=sample_weight)
                except TypeError:
                    self.model.fit(x, target)
        finite = np.abs(target[np.isfinite(target)])
        scale = float(np.quantile(finite, 0.75)) if len(finite) else 0.0
        self.probability_scale_ = max(scale, 1e-6)
        return self

    def predict_expected_return(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self._prediction_model().predict(x)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.probability_scale_ is None:
            raise RuntimeError("Regressor Alpha model is not fitted")
        expected_return = self.predict_expected_return(x)
        up = _sigmoid(expected_return / self.probability_scale_)
        return np.column_stack([1.0 - up, up])


class RankerAlphaAdapter(RegressorAlphaAdapter):
    """LightGBM-style cross-sectional ranker requiring explicit group sizes."""

    target_kind = "cross_sectional_rank"

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        group: list[int] | np.ndarray | None = None,
    ) -> "RankerAlphaAdapter":
        if group is None:
            raise ValueError("ranker requires explicit cross-sectional group sizes")
        group_sizes = np.asarray(group, dtype=int).reshape(-1)
        if len(group_sizes) == 0 or bool((group_sizes < 2).any()):
            raise ValueError("each ranker group must contain at least two symbols")
        if int(group_sizes.sum()) != len(x):
            raise ValueError("ranker group sizes must sum to the number of rows")
        target = np.asarray(y, dtype=float)
        if (
            not bool(np.isfinite(target).all())
            or bool((target < 0).any())
            or not bool(np.equal(target, np.floor(target)).all())
        ):
            raise ValueError(
                "ranker relevance labels must be finite non-negative integers"
            )
        kwargs: dict[str, object] = {"group": group_sizes.tolist()}
        if sample_weight is not None:
            kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(x, target, **kwargs)
        return self

    def predict_rank_score(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self._prediction_model().predict(x)
        return np.asarray(prediction, dtype=float).reshape(-1)

    def predict_expected_return(self, x: np.ndarray) -> None:
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "rank scores are not probabilities; fit a separate calibration model"
        )


def build_lightgbm_classifier_alpha(
    name: str,
    *,
    seed: int,
    **params: object,
) -> ClassifierAlphaAdapter:
    """Build the optional LightGBM direction classifier adapter."""

    from lightgbm import LGBMClassifier

    model_params = {
        "objective": "binary",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
        **params,
    }
    return ClassifierAlphaAdapter(name, LGBMClassifier(**model_params))


def build_lightgbm_regressor_alpha(
    name: str,
    *,
    seed: int,
    **params: object,
) -> RegressorAlphaAdapter:
    """Build the optional LightGBM expected-return regressor adapter."""

    from lightgbm import LGBMRegressor

    model_params = {
        "objective": "huber",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
        **params,
    }
    return RegressorAlphaAdapter(name, LGBMRegressor(**model_params))


def build_lightgbm_ranker_alpha(
    name: str,
    *,
    seed: int,
    **params: object,
) -> RankerAlphaAdapter:
    """Build the optional LightGBM cross-sectional ranker adapter."""

    from lightgbm import LGBMRanker

    model_params = {
        "objective": "lambdarank",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
        **params,
    }
    return RankerAlphaAdapter(name, LGBMRanker(**model_params))


def build_xgboost_regressor_alpha(
    name: str,
    *,
    seed: int,
    **params: object,
) -> RegressorAlphaAdapter:
    """Build the optional XGBoost expected-return regressor adapter."""

    from xgboost import XGBRegressor

    model_params = {
        "objective": "reg:pseudohubererror",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
        **params,
    }
    return RegressorAlphaAdapter(name, XGBRegressor(**model_params))


def build_catboost_regressor_alpha(
    name: str,
    *,
    seed: int,
    **params: object,
) -> RegressorAlphaAdapter:
    """Build the optional CatBoost expected-return regressor adapter."""

    from catboost import CatBoostRegressor

    model_params = {
        "loss_function": "Huber:delta=1.0",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
        **params,
    }
    return RegressorAlphaAdapter(name, CatBoostRegressor(**model_params))


def optional_regressor_availability() -> list[dict[str, object]]:
    """Report optional continuous Alpha adapters without importing them."""

    return [
        {
            "name": "lightgbm_regressor",
            "dependency": "lightgbm",
            "available": importlib.util.find_spec("lightgbm") is not None,
            "target": "future_return",
        },
        {
            "name": "xgboost_regressor",
            "dependency": "xgboost",
            "available": importlib.util.find_spec("xgboost") is not None,
            "target": "future_return",
        },
        {
            "name": "catboost_regressor",
            "dependency": "catboost",
            "available": importlib.util.find_spec("catboost") is not None,
            "target": "future_return",
        },
    ]


def train_lightgbm_expected_return(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[RegressorAlphaAdapter | None, dict[str, object]]:
    """Fit an optional expected-return model without using test data."""

    if importlib.util.find_spec("lightgbm") is None:
        return None, {
            "status": "skipped",
            "reason": "missing_dependency: lightgbm",
            "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
            "model_kind": "regressor",
            "target": "future_return",
        }
    train_target = np.asarray(y_train, dtype=float)
    valid_target = np.asarray(y_valid, dtype=float)
    train_mask = np.isfinite(train_target)
    valid_mask = np.isfinite(valid_target)
    if int(train_mask.sum()) < 80 or int(valid_mask.sum()) < 20:
        return None, {
            "status": "skipped",
            "reason": "insufficient_finite_return_labels",
            "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
            "model_kind": "regressor",
            "target": "future_return",
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(valid_mask.sum()),
        }

    model = build_lightgbm_regressor_alpha(
        "lightgbm_expected_return_regressor",
        seed=seed,
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_samples=30,
    )
    weights = (
        np.asarray(sample_weight, dtype=float)[train_mask]
        if sample_weight is not None
        else None
    )
    try:
        model.fit(
            np.asarray(x_train, dtype=float)[train_mask],
            train_target[train_mask],
            sample_weight=weights,
        )
        prediction = model.predict_expected_return(
            np.asarray(x_valid, dtype=float)[valid_mask]
        )
    except Exception as exc:
        return None, {
            "status": "skipped",
            "reason": f"fit_failed: {type(exc).__name__}",
            "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
            "model_kind": "regressor",
            "target": "future_return",
        }

    actual = valid_target[valid_mask]
    error = prediction - actual
    correlation = (
        float(np.corrcoef(prediction, actual)[0, 1])
        if len(actual) > 1
        and float(np.std(prediction)) > 1e-12
        and float(np.std(actual)) > 1e-12
        else 0.0
    )
    return model, {
        "status": "trained",
        "reason": "",
        "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
        "model_kind": "regressor",
        "model_name": model.name,
        "target": "future_return",
        "selection_dataset": "validation_calibration",
        "test_used_for_training_or_selection": False,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "metrics": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "directional_accuracy": float(
                (np.sign(prediction) == np.sign(actual)).mean()
            ),
            "correlation": correlation,
        },
    }


def lightgbm_ranker_availability_report() -> dict[str, object]:
    """Describe why per-symbol training does not fit a cross-sectional ranker."""

    return {
        "status": "interface_ready_not_trained",
        "reason": "cross_sectional_symbol_groups_required",
        "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
        "model_kind": "ranker",
        "target": "cross_sectional_rank_label",
        "output": "rank_score",
        "dependency_available": importlib.util.find_spec("lightgbm") is not None,
        "test_used_for_training_or_selection": False,
    }
