from __future__ import annotations

import unittest

from crypto_ai_trader.model_selection import rank_model_candidates


def candidate(
    name: str,
    *,
    predictive_score: float,
    strategy_score: float,
    gate_passed: bool,
) -> tuple[float, object, dict[str, object]]:
    return (
        strategy_score,
        object(),
        {
            "name": name,
            "predictive_score": predictive_score,
            "strategy_selection_score": strategy_score,
            "validation_trading_gate": {"passed": gate_passed},
        },
    )


class ModelSelectionTests(unittest.TestCase):
    def test_predictive_and_strategy_rankings_are_independent(self) -> None:
        predictive_winner = candidate(
            "predictive_winner",
            predictive_score=0.90,
            strategy_score=0.10,
            gate_passed=False,
        )
        strategy_winner = candidate(
            "strategy_winner",
            predictive_score=0.65,
            strategy_score=0.80,
            gate_passed=True,
        )

        ranking = rank_model_candidates(
            [predictive_winner, strategy_winner]
        )

        self.assertEqual(
            ranking.predictive[0][2]["name"],
            "predictive_winner",
        )
        self.assertEqual(
            ranking.strategy_gated[0][2]["name"],
            "strategy_winner",
        )
        self.assertEqual(
            ranking.selected[2]["name"],
            "strategy_winner",
        )
        self.assertEqual(
            ranking.selected_via,
            "strategy_gate_then_strategy_score",
        )
        self.assertFalse(
            ranking.audit()["predictive_and_selected_match"]
        )

    def test_no_gate_pass_uses_explicit_research_fallback(self) -> None:
        ranking = rank_model_candidates(
            [
                candidate(
                    "lower",
                    predictive_score=0.90,
                    strategy_score=0.10,
                    gate_passed=False,
                ),
                candidate(
                    "higher",
                    predictive_score=0.60,
                    strategy_score=0.30,
                    gate_passed=False,
                ),
            ]
        )

        self.assertEqual(ranking.selected[2]["name"], "higher")
        self.assertEqual(
            ranking.selected_via,
            "research_fallback_strategy_score",
        )
        self.assertFalse(ranking.strategy_eligible)
        self.assertFalse(
            ranking.audit()["test_used_for_predictive_ranking"]
        )

    def test_empty_candidates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_model_candidates([])


if __name__ == "__main__":
    unittest.main()
