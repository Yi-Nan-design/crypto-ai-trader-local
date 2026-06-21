from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from crypto_ai_trader.portfolio_paper import (
    PortfolioPaperMark,
    extract_aligned_mark,
    load_portfolio_paper_state,
    mark_portfolio_paper_state,
    new_portfolio_paper_state,
    persist_portfolio_paper_state,
    portfolio_paper_risk_state,
    rebalance_portfolio_paper_state,
)


def config() -> SimpleNamespace:
    return SimpleNamespace(
        portfolio_paper_initial_balance=10_000.0,
        portfolio_paper_max_history=3,
        maker_fill_fraction=0.0,
        maker_fee_rate=0.0002,
        fee_rate=0.0004,
        slippage_rate=0.0001,
        max_daily_loss=0.03,
        portfolio_max_drawdown=0.10,
    )


def mark(open_time: int, eth: float, bnb: float) -> PortfolioPaperMark:
    return PortfolioPaperMark(
        open_time=open_time,
        datetime_beijing="2026-06-20T12:00:00+08:00",
        prices={"ETHUSDT": eth, "BNBUSDT": bnb},
    )


class PortfolioPaperTests(unittest.TestCase):
    def test_previous_weights_are_marked_before_new_rebalance(self) -> None:
        cfg = config()
        state = new_portfolio_paper_state("5m", cfg)
        first = mark_portfolio_paper_state(
            state,
            mark(1_781_930_000_000, 100.0, 50.0),
            mark_reason="aligned_closed_bar",
        )
        rebalance_portfolio_paper_state(
            state,
            {"ETHUSDT": 0.20, "BNBUSDT": -0.10},
            cfg,
            first,
        )
        equity_after_first_cost = state.equity

        second = mark_portfolio_paper_state(
            state,
            mark(1_781_930_300_000, 110.0, 45.0),
            mark_reason="aligned_closed_bar",
        )

        self.assertTrue(second["advanced"])
        self.assertAlmostEqual(second["market_return"], 0.03)
        self.assertAlmostEqual(
            state.equity,
            equity_after_first_cost * 1.03,
        )

    def test_duplicate_bar_does_not_repeat_pnl_or_cost(self) -> None:
        cfg = config()
        state = new_portfolio_paper_state("5m", cfg)
        first_mark = mark(1_781_930_000_000, 100.0, 50.0)
        event = mark_portfolio_paper_state(
            state,
            first_mark,
            mark_reason="aligned_closed_bar",
        )
        rebalance_portfolio_paper_state(
            state,
            {"ETHUSDT": 0.20},
            cfg,
            event,
        )
        equity = state.equity
        fees = state.commission_fees

        duplicate = mark_portfolio_paper_state(
            state,
            first_mark,
            mark_reason="aligned_closed_bar",
        )
        rebalance_portfolio_paper_state(
            state,
            {"ETHUSDT": 0.40},
            cfg,
            duplicate,
        )

        self.assertFalse(duplicate["advanced"])
        self.assertEqual(duplicate["action"], "DUPLICATE_BAR")
        self.assertAlmostEqual(state.equity, equity)
        self.assertAlmostEqual(state.commission_fees, fees)
        self.assertEqual(state.weights, {"ETHUSDT": 0.20})

    def test_daily_and_drawdown_circuit_breakers_use_equity(self) -> None:
        cfg = config()
        state = new_portfolio_paper_state("5m", cfg)
        state.equity = 8_900.0
        state.peak_equity = 10_000.0
        state.day_start_equity = 9_500.0

        risk = portfolio_paper_risk_state(state, cfg)

        self.assertFalse(risk["allow_portfolio"])
        self.assertEqual(risk["reason"], "portfolio_blocked_drawdown")
        self.assertAlmostEqual(risk["drawdown"], -0.11)
        self.assertLess(risk["daily_return"], -0.03)

    def test_unaligned_reports_are_not_marked(self) -> None:
        extracted, reason = extract_aligned_mark(
            [
                {
                    "symbol": "ETHUSDT",
                    "latest_open_time": 1000,
                    "latest_close": 100.0,
                },
                {
                    "symbol": "BNBUSDT",
                    "latest_open_time": 2000,
                    "latest_close": 50.0,
                },
            ]
        )

        self.assertIsNone(extracted)
        self.assertEqual(reason, "unaligned_latest_open_time")

    def test_persistence_compacts_and_corrupt_state_is_not_reset(self) -> None:
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_path = root / "state.json"
            ledger_path = root / "ledger.jsonl"
            latest_path = root / "latest.json"
            state = new_portfolio_paper_state("5m", cfg)
            for index in range(5):
                event = mark_portfolio_paper_state(
                    state,
                    mark(
                        1_781_930_000_000 + index * 300_000,
                        100.0 + index,
                        50.0,
                    ),
                    mark_reason="aligned_closed_bar",
                )
                rebalance_portfolio_paper_state(
                    state,
                    {"ETHUSDT": 0.10},
                    cfg,
                    event,
                )
                persist_portfolio_paper_state(
                    state,
                    event,
                    state_path=state_path,
                    ledger_path=ledger_path,
                    latest_path=latest_path,
                    max_history=cfg.portfolio_paper_max_history,
                )

            lines = ledger_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            loaded = load_portfolio_paper_state(
                state_path,
                interval="5m",
                cfg=cfg,
            )
            self.assertEqual(loaded.steps, 5)
            self.assertFalse(
                json.loads(latest_path.read_text(encoding="utf-8"))["safety"][
                    "live_trading_enabled"
                ]
            )

            state_path.write_text("{bad", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_portfolio_paper_state(
                    state_path,
                    interval="5m",
                    cfg=cfg,
                )


if __name__ == "__main__":
    unittest.main()
