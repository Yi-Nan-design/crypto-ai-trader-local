from __future__ import annotations

import unittest

from crypto_ai_trader.alpha_models import optional_regressor_availability
from crypto_ai_trader.regime_models import regime_model_registry


class OptionalModelRegistryTests(unittest.TestCase):
    def test_regime_registry_covers_required_methods(self) -> None:
        registry = {item["method"]: item for item in regime_model_registry()}

        self.assertTrue(
            {
                "rule_based",
                "rule_based_quantiles",
                "kmeans",
                "gaussian_mixture",
                "hmm",
                "lightgbm_classifier",
            }.issubset(registry)
        )
        self.assertEqual(registry["kmeans"]["causal_fit_required"], True)
        self.assertIn(
            registry["hmm"]["status"],
            {"interface_reserved", "optional_unavailable"},
        )

    def test_optional_regressor_registry_is_non_blocking(self) -> None:
        registry = {
            item["name"]: item for item in optional_regressor_availability()
        }

        self.assertEqual(
            set(registry),
            {
                "lightgbm_regressor",
                "xgboost_regressor",
                "catboost_regressor",
            },
        )
        self.assertTrue(
            all(isinstance(item["available"], bool) for item in registry.values())
        )


if __name__ == "__main__":
    unittest.main()
