from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crypto_ai_trader.alpha_models import (
    ALPHA_MODEL_INTERFACE_VERSION,
    ClassifierAlphaAdapter,
    RankerAlphaAdapter,
    RegressorAlphaAdapter,
    lightgbm_ranker_availability_report,
    train_lightgbm_expected_return,
)
from crypto_ai_trader.live_training import _alpha_model_version
from crypto_ai_trader.models import ModelBundle, StandardScaler


class _Classifier:
    def __init__(self) -> None:
        self.sample_weight: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "_Classifier":
        self.sample_weight = sample_weight
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        up = np.clip(0.5 + 0.1 * x[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - up, up])


class _Regressor:
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "_Regressor":
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 0.01 * x[:, 0]


class _Ranker(_Regressor):
    def __init__(self) -> None:
        self.group: list[int] | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        group: list[int],
        sample_weight: np.ndarray | None = None,
    ) -> "_Ranker":
        self.group = group
        return self


class AlphaModelTests(unittest.TestCase):
    def test_classifier_adapter_preserves_probability_contract(self) -> None:
        model = _Classifier()
        adapter = ClassifierAlphaAdapter("classifier", model)
        x = np.array([[-1.0], [1.0]])
        weights = np.array([0.5, 2.0])

        adapter.fit(x, np.array([0, 1]), sample_weight=weights)
        probability = adapter.predict_proba(x)

        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        np.testing.assert_allclose(model.sample_weight, weights)
        self.assertIsNone(adapter.predict_expected_return(x))
        self.assertEqual(adapter.target_kind, "binary_direction")

    def test_regressor_adapter_exposes_expected_return_and_probability_proxy(self) -> None:
        adapter = RegressorAlphaAdapter("regressor", _Regressor())
        x = np.array([[-1.0], [0.0], [1.0]])
        adapter.fit(x, np.array([-0.02, 0.0, 0.02]))

        expected_return = adapter.predict_expected_return(x)
        probability = adapter.predict_proba(x)

        np.testing.assert_allclose(expected_return, [-0.01, 0.0, 0.01])
        self.assertLess(probability[0, 1], 0.5)
        self.assertAlmostEqual(float(probability[1, 1]), 0.5)
        self.assertGreater(probability[2, 1], 0.5)

    def test_ranker_requires_explicit_cross_sectional_groups(self) -> None:
        adapter = RankerAlphaAdapter("ranker", _Ranker())
        x = np.arange(8, dtype=float).reshape(4, 2)
        y = np.array([0, 1, 0, 1], dtype=float)

        with self.assertRaises(ValueError):
            adapter.fit(x, y)
        with self.assertRaises(ValueError):
            adapter.fit(x, y, group=[1, 3])
        with self.assertRaises(ValueError):
            adapter.fit(x, y, group=[2, 3])
        with self.assertRaises(ValueError):
            adapter.fit(x, np.array([0.0, 0.5, 1.0, 2.0]), group=[2, 2])

        adapter.fit(x, y, group=[2, 2])
        self.assertEqual(adapter.model.group, [2, 2])
        self.assertEqual(adapter.target_kind, "cross_sectional_rank")
        self.assertEqual(len(adapter.predict_rank_score(x)), len(x))
        self.assertIsNone(adapter.predict_expected_return(x))
        with self.assertRaises(RuntimeError):
            adapter.predict_proba(x)

    def test_model_bundle_separates_expected_return_from_horizon_probabilities(self) -> None:
        x = np.array([[-1.0], [0.0], [1.0]])
        scaler = StandardScaler().fit(x)
        classifier = ClassifierAlphaAdapter("classifier", _Classifier()).fit(
            scaler.transform(x),
            np.array([0, 0, 1]),
        )
        regressor = RegressorAlphaAdapter("regressor", _Regressor()).fit(
            scaler.transform(x),
            np.array([-0.02, 0.0, 0.02]),
        )
        bundle = ModelBundle(
            model_name="classifier",
            model=classifier,
            scaler=scaler,
            feature_columns=["x"],
            metrics={},
            config={},
            auxiliary_models={"alpha_expected_return": regressor},
            auxiliary_metadata={},
        )

        expected_return = bundle.predict_expected_return(x)
        horizon = bundle.predict_horizon_probabilities(x)

        self.assertEqual(set(horizon), {"main"})
        self.assertEqual(len(expected_return), len(x))
        self.assertTrue(np.isfinite(expected_return).all())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.pkl"
            bundle.save(path)
            loaded = ModelBundle.load(path)
            np.testing.assert_allclose(
                loaded.predict_expected_return(x),
                expected_return,
            )

    def test_alpha_model_version_records_classifier_and_regressor(self) -> None:
        class Bundle:
            model_name = "classifier_v1"
            auxiliary_metadata = {
                "alpha_expected_return": {
                    "model_name": "return_v2",
                    "interface_version": ALPHA_MODEL_INTERFACE_VERSION,
                }
            }

        self.assertEqual(
            _alpha_model_version(Bundle()),
            (
                "classifier:classifier_v1"
                "|expected_return:return_v2"
                f"|interface:{ALPHA_MODEL_INTERFACE_VERSION}"
            ),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("lightgbm") is not None,
        "LightGBM is optional",
    )
    def test_lightgbm_expected_return_uses_validation_without_test_data(self) -> None:
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(120, 4))
        y_train = 0.002 * x_train[:, 0] - 0.001 * x_train[:, 1]
        x_valid = rng.normal(size=(40, 4))
        y_valid = 0.002 * x_valid[:, 0] - 0.001 * x_valid[:, 1]

        model, report = train_lightgbm_expected_return(
            x_train,
            y_train,
            x_valid,
            y_valid,
            seed=42,
        )

        self.assertIsNotNone(model)
        self.assertEqual(report["status"], "trained")
        self.assertEqual(
            report["interface_version"],
            ALPHA_MODEL_INTERFACE_VERSION,
        )
        self.assertFalse(report["test_used_for_training_or_selection"])
        self.assertIn("mae", report["metrics"])

        payload = pickle.dumps(model)
        restored = pickle.loads(payload)
        self.assertIsNone(restored.model)
        self.assertTrue(restored._lightgbm_model_string)
        self.assertTrue(
            np.isfinite(restored.predict_expected_return(x_valid[:3])).all()
        )

        unavailable = pickle.loads(payload)
        with patch(
            "crypto_ai_trader.alpha_models.importlib.util.find_spec",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "missing_dependency: lightgbm",
            ):
                unavailable.predict_expected_return(x_valid[:1])

    def test_ranker_report_is_explicitly_not_trained_per_symbol(self) -> None:
        report = lightgbm_ranker_availability_report()

        self.assertEqual(report["status"], "interface_ready_not_trained")
        self.assertEqual(
            report["reason"],
            "cross_sectional_symbol_groups_required",
        )
        self.assertFalse(report["test_used_for_training_or_selection"])


if __name__ == "__main__":
    unittest.main()
