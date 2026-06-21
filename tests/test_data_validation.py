from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from crypto_ai_trader.binance_data import (
    exchange_rules_path,
    funding_history_path,
    load_symbol_interval,
    merge_funding_history,
    parse_funding_history_rows,
    resolve_exchange_rule_values,
    sync_recent_futures_klines,
)
from crypto_ai_trader.data_validation import (
    cross_exchange_price_deviation,
    isolation_forest_anomalies,
    interval_to_milliseconds,
    normalize_kline_frame,
    validate_kline_frame,
)
from crypto_ai_trader.exchange_rules import parse_futures_symbol_rules


def kline_row(
    open_time: int,
    *,
    open_price: float = 100.0,
    high: float = 102.0,
    low: float = 99.0,
    close: float = 101.0,
    volume: float = 10.0,
) -> dict[str, float | int]:
    return {
        "open_time": open_time,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": open_time + 59_999,
        "quote_volume": volume * close,
        "trades": 20,
        "taker_buy_base_volume": volume * 0.5,
        "taker_buy_quote_volume": volume * close * 0.5,
    }


class DataValidationTests(unittest.TestCase):
    def test_cross_exchange_deviation_flags_only_aligned_large_gaps(self) -> None:
        primary = pd.DataFrame(
            {"open_time": [1, 2, 3], "close": [100.0, 103.0, 101.0]}
        )
        reference = pd.DataFrame(
            {"open_time": [1, 2, 4], "close": [100.0, 100.0, 100.0]}
        )

        report = cross_exchange_price_deviation(
            primary,
            reference,
            threshold=0.02,
        )

        self.assertEqual(report["open_time"].tolist(), [1, 2])
        self.assertEqual(report["deviation_flag"].tolist(), [False, True])

    def test_isolation_forest_is_optional_and_non_destructive(self) -> None:
        frame = pd.DataFrame(
            {
                "return_1": [0.0] * 49 + [1.0],
                "quote_volume": [100.0] * 50,
            }
        )

        mask, report = isolation_forest_anomalies(
            frame,
            ["return_1", "quote_volume"],
            contamination=0.02,
        )

        self.assertEqual(len(mask), len(frame))
        self.assertIn(report["status"], {"completed", "skipped"})
        if report["status"] == "completed":
            self.assertGreaterEqual(int(report["anomaly_count"]), 1)
            self.assertFalse(report["destructive_filtering"])

    def test_interval_conversion(self) -> None:
        self.assertEqual(interval_to_milliseconds("5m"), 300_000)
        self.assertEqual(interval_to_milliseconds("1h"), 3_600_000)
        self.assertIsNone(interval_to_milliseconds("1M"))

    def test_validation_removes_duplicates_and_structural_errors(self) -> None:
        rows = [
            kline_row(0),
            kline_row(0, close=100.5),
            kline_row(60_000, high=100.0, close=101.0),
            kline_row(180_000),
        ]
        cleaned, report = validate_kline_frame(pd.DataFrame(rows), "1m")

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report.duplicate_rows_removed, 1)
        self.assertEqual(report.invalid_rows_removed, 1)
        self.assertEqual(report.missing_bar_count, 2)
        self.assertEqual(report.gap_recovery_rows, 1)
        self.assertEqual(cleaned["exchange_gap_before_bars"].tolist(), [0, 2])
        self.assertEqual(cleaned["exchange_available"].tolist(), [True, True])
        self.assertIn("duplicate_open_time_removed", report.issues)
        self.assertIn("structurally_invalid_rows_removed", report.issues)
        self.assertTrue(report.passed)

    def test_normalization_converts_numeric_and_datetime_columns(self) -> None:
        raw = pd.DataFrame([kline_row(0)])
        raw["close"] = raw["close"].astype(str)
        normalized = normalize_kline_frame(raw)

        self.assertEqual(float(normalized.loc[0, "close"]), 101.0)
        self.assertIn("open_datetime", normalized.columns)
        self.assertIn("close_datetime", normalized.columns)

    def test_unknown_legacy_availability_defaults_to_available(self) -> None:
        rows = [kline_row(0), kline_row(60_000)]
        rows[0]["exchange_available"] = None
        rows[1]["exchange_available"] = "False"

        cleaned, _ = validate_kline_frame(pd.DataFrame(rows), "1m")

        self.assertEqual(
            cleaned["exchange_available"].tolist(),
            [True, False],
        )

    def test_missing_required_columns_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_kline_frame(pd.DataFrame({"open_time": [0]}), "1m")

    def test_market_context_parsers_and_causal_funding_merge(self) -> None:
        exchange_info = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "filters": [
                        {
                            "filterType": "LOT_SIZE",
                            "minQty": "0.001",
                            "stepSize": "0.001",
                        },
                        {
                            "filterType": "MARKET_LOT_SIZE",
                            "minQty": "0.002",
                            "maxQty": "25",
                            "stepSize": "0.002",
                        },
                        {
                            "filterType": "PRICE_FILTER",
                            "tickSize": "0.10",
                        },
                        {
                            "filterType": "MIN_NOTIONAL",
                            "notional": "5",
                        },
                    ],
                }
            ]
        }
        rules = parse_futures_symbol_rules(exchange_info, "btcusdt")
        self.assertEqual(rules.quantity_filter_type, "MARKET_LOT_SIZE")
        self.assertEqual(rules.min_notional_usdt, 5.0)
        self.assertEqual(rules.min_quantity, 0.002)
        self.assertEqual(rules.max_quantity, 25.0)
        self.assertEqual(rules.quantity_step, 0.002)
        self.assertEqual(rules.price_tick_size, 0.10)

        funding = parse_funding_history_rows(
            [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": 3_600_000,
                    "fundingRate": "0.001",
                    "markPrice": "100",
                },
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": 32_400_000,
                    "fundingRate": "-0.002",
                    "markPrice": "101",
                },
            ]
        )
        klines = pd.DataFrame(
            {
                "open_time": [0, 14_400_000, 28_800_000, 43_200_000],
                "close": [100.0, 100.0, 101.0, 102.0],
            }
        )
        merged = merge_funding_history(klines, funding)

        self.assertEqual(merged.loc[0, "funding_payment_rate"], 0.001)
        self.assertTrue(pd.isna(merged.loc[1, "funding_payment_rate"]))
        self.assertEqual(merged.loc[2, "funding_payment_rate"], -0.002)
        self.assertTrue(pd.isna(merged.loc[3, "funding_payment_rate"]))

    def test_cached_market_context_is_loaded_without_overriding_explicit_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            symbol = "BTCUSDT"
            interval = "1h"
            raw_dir = base / "raw" / symbol / interval
            raw_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    kline_row(0),
                    kline_row(3_600_000),
                    kline_row(7_200_000),
                ]
            ).to_csv(raw_dir / "sample.csv", index=False)
            funding_file = funding_history_path(base, symbol)
            funding_file.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "funding_time": 3_600_000,
                        "funding_rate_8h": 0.0002,
                        "mark_price": 100.0,
                    }
                ]
            ).to_csv(funding_file, index=False)
            rules_file = exchange_rules_path(base, symbol)
            rules_file.write_text(
                json.dumps(
                    {
                        "symbol": symbol,
                        "min_notional_usdt": 5.0,
                        "min_quantity": 0.002,
                        "max_quantity": 25.0,
                        "quantity_step": 0.001,
                        "price_tick_size": 0.1,
                        "quantity_filter_type": "MARKET_LOT_SIZE",
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_symbol_interval(base, symbol, interval)
            cached = resolve_exchange_rule_values(base, symbol)
            explicit = resolve_exchange_rule_values(
                base,
                symbol,
                min_notional_usdt=12.0,
                min_quantity=0.004,
                quantity_step=0.003,
            )

            self.assertIn("funding_rate_8h", loaded.columns)
            self.assertTrue(pd.isna(loaded.loc[0, "funding_rate_8h"]))
            self.assertEqual(loaded.loc[1, "funding_rate_8h"], 0.0002)
            self.assertEqual(loaded.loc[1, "funding_payment_rate"], 0.0002)
            self.assertEqual(
                loaded.attrs["market_context"]["funding_events_applied"],
                1,
            )
            self.assertEqual(cached["exchange_min_notional_usdt"], 5.0)
            self.assertEqual(cached["exchange_max_quantity"], 25.0)
            self.assertEqual(cached["exchange_price_tick_size"], 0.1)
            self.assertEqual(explicit["exchange_min_notional_usdt"], 12.0)
            self.assertEqual(explicit["exchange_min_quantity"], 0.004)
            self.assertEqual(explicit["exchange_quantity_step"], 0.003)

    def test_realtime_sync_marks_cached_latest_unavailable_then_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            output = (
                base
                / "realtime"
                / "BTCUSDT"
                / "1m"
                / "BTCUSDT-1m-realtime.csv"
            )
            output.parent.mkdir(parents=True)
            pd.DataFrame([kline_row(0), kline_row(60_000)]).to_csv(
                output,
                index=False,
            )

            with patch(
                "crypto_ai_trader.binance_data.fetch_recent_futures_klines",
                side_effect=RuntimeError("network unavailable"),
            ):
                failed = sync_recent_futures_klines(
                    ["BTCUSDT"],
                    "1m",
                    data_dir=base,
                )
            stale = pd.read_csv(output)
            self.assertTrue(failed[0].skipped)
            self.assertFalse(bool(stale.iloc[-1]["exchange_available"]))

            recovered_rows = pd.DataFrame(
                [
                    kline_row(60_000),
                    kline_row(120_000),
                ]
            )
            with patch(
                "crypto_ai_trader.binance_data.fetch_recent_futures_klines",
                return_value=recovered_rows,
            ):
                sync_recent_futures_klines(
                    ["BTCUSDT"],
                    "1m",
                    data_dir=base,
                )
            recovered = pd.read_csv(output)
            self.assertTrue(bool(recovered.iloc[-1]["exchange_available"]))


if __name__ == "__main__":
    unittest.main()
