from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from crypto_ai_trader.backtest import BacktestConfig
from crypto_ai_trader.portable import iter_portable_files
from crypto_ai_trader.risk_profile_catalog import (
    DEFAULT_RISK_PROFILE_CATALOG_PATH,
    resolve_risk_profile_catalog,
)


class RiskProfileCatalogTests(unittest.TestCase):
    def test_default_catalog_has_versioned_profile_contract(self) -> None:
        catalog = resolve_risk_profile_catalog(BacktestConfig())
        compact = resolve_risk_profile_catalog(
            BacktestConfig(),
            compact=True,
        )

        self.assertEqual(catalog.schema_version, 1)
        self.assertEqual(catalog.catalog_version, "2026-06-20-v1")
        self.assertEqual(catalog.source_path, DEFAULT_RISK_PROFILE_CATALOG_PATH)
        self.assertEqual(len(catalog.profiles), 19)
        self.assertEqual(
            [item["name"] for item in compact.profiles],
            list(catalog.compact_order),
        )

    def test_catalog_applies_min_max_and_base_rules(self) -> None:
        cfg = BacktestConfig(
            max_position_fraction=0.10,
            risk_per_trade=0.002,
            min_position_scale=0.60,
            range_gate_max_efficiency=0.30,
            range_reversion_min_score=0.20,
        )
        profiles = {
            item["name"]: item["params"]
            for item in resolve_risk_profile_catalog(cfg).profiles
        }

        preservation = profiles["micro_capital_preservation"]
        self.assertEqual(preservation["max_position_fraction"], 0.10)
        self.assertEqual(preservation["risk_per_trade"], 0.002)
        boosted = profiles["micro_range_grid_long_boost"]
        self.assertEqual(boosted["min_position_scale"], 0.60)
        self.assertEqual(boosted["range_gate_max_efficiency"], 0.30)
        self.assertEqual(boosted["range_reversion_min_score"], 0.20)
        base = profiles["base_small_account"]
        self.assertEqual(
            base["max_position_fraction"],
            cfg.max_position_fraction,
        )
        self.assertEqual(base["risk_per_trade"], cfg.risk_per_trade)

    def test_catalog_rejects_unknown_backtest_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.toml"
            path.write_text(
                "\n".join(
                    [
                        "schema_version = 1",
                        'catalog_version = "test"',
                        'compact_order = ["invalid"]',
                        "[[profiles]]",
                        'name = "invalid"',
                        "[profiles.values]",
                        "not_a_backtest_field = 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "unknown BacktestConfig field",
            ):
                resolve_risk_profile_catalog(
                    BacktestConfig(),
                    path=path,
                )

    def test_catalog_missing_file_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.toml"
            with self.assertRaises(FileNotFoundError):
                resolve_risk_profile_catalog(
                    BacktestConfig(),
                    path=missing,
                )

    def test_portable_file_set_contains_profile_catalog(self) -> None:
        root = Path(__file__).resolve().parent.parent
        relative_paths = {
            path.relative_to(root).as_posix()
            for path in iter_portable_files(root)
        }

        self.assertIn(
            "config/strategy_calibration_profiles.toml",
            relative_paths,
        )


if __name__ == "__main__":
    unittest.main()
