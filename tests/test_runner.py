from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from crypto_ai_trader.portfolio_paper import new_portfolio_paper_state
from crypto_ai_trader.runner import runner_once


def config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=root / "data",
        reports_dir=root / "reports",
        portfolio_correlation_lookback=2,
        portfolio_paper_initial_balance=10_000.0,
        portfolio_paper_max_history=10,
        maker_fill_fraction=0.0,
        maker_fee_rate=0.0002,
        fee_rate=0.0004,
        slippage_rate=0.0001,
        max_daily_loss=0.03,
        portfolio_max_drawdown=0.10,
        ensure_dirs=lambda: None,
    )


def live_report(symbol: str, close: float) -> dict:
    return {
        "symbol": symbol,
        "latest_open_time": 1_782_030_300_000,
        "latest_datetime": "2026-06-21T17:05:00+08:00",
        "latest_close": close,
        "model_name": "test_model",
        "latest_up_probability": 0.5,
        "backtest": {
            "total_return": 0.0,
            "max_drawdown": 0.0,
        },
    }


class RunnerTests(unittest.TestCase):
    def test_runner_accepts_typed_aligned_portfolio_mark(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = config(root)
            state = new_portfolio_paper_state("5m", cfg)
            reports = {
                "ETHUSDT": live_report("ETHUSDT", 1_800.0),
                "BNBUSDT": live_report("BNBUSDT", 600.0),
            }
            frame = pd.DataFrame(
                {
                    "open_time": [
                        1_782_029_700_000,
                        1_782_030_000_000,
                        1_782_030_300_000,
                    ],
                    "close": [100.0, 101.0, 102.0],
                }
            )
            snapshot = {
                "decision": {
                    "weights": {
                        "ETHUSDT": 0.0,
                        "BNBUSDT": 0.0,
                    }
                }
            }

            with (
                patch(
                    "crypto_ai_trader.runner.load_config",
                    return_value=cfg,
                ),
                patch(
                    "crypto_ai_trader.runner.sync_recent_futures_klines",
                    return_value=[],
                ),
                patch(
                    "crypto_ai_trader.runner.train_live_symbol",
                    side_effect=lambda symbol, *_args, **_kwargs: reports[symbol],
                ),
                patch(
                    "crypto_ai_trader.runner.load_symbol_interval",
                    return_value=frame,
                ),
                patch(
                    "crypto_ai_trader.runner.load_portfolio_paper_state",
                    return_value=state,
                ),
                patch(
                    "crypto_ai_trader.runner.build_portfolio_snapshot",
                    return_value=snapshot,
                ) as build_snapshot,
                patch(
                    "crypto_ai_trader.runner.persist_portfolio_paper_state",
                    return_value={
                        "state": state.to_dict(),
                        "safety": {"live_trading_enabled": False},
                    },
                ),
                patch("crypto_ai_trader.runner.safe_replace_text"),
                patch("crypto_ai_trader.runner.write_state"),
            ):
                payload = runner_once(
                    symbols=["ETHUSDT", "BNBUSDT"],
                    interval="5m",
                    limit=800,
                    base_url="https://example.invalid",
                    model_suffix="runner_test",
                    state_dir=root / "state",
                )

            self.assertEqual(
                build_snapshot.call_args.kwargs["expected_open_time"],
                1_782_030_300_000,
            )
            self.assertEqual(
                payload["portfolio"]["decision"]["weights"],
                {"ETHUSDT": 0.0, "BNBUSDT": 0.0},
            )
            self.assertFalse(
                payload["portfolio"]["paper"]["safety"][
                    "live_trading_enabled"
                ]
            )


if __name__ == "__main__":
    unittest.main()
