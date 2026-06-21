from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ModelCandidate = tuple[float, Any, dict[str, Any]]


@dataclass(frozen=True)
class ModelSelectionRanking:
    """Separate predictive ranking from strategy-gated compatibility ranking."""

    predictive: tuple[ModelCandidate, ...]
    strategy_gated: tuple[ModelCandidate, ...]
    strategy_eligible: tuple[ModelCandidate, ...]
    selected: ModelCandidate
    selected_via: str

    def audit(self) -> dict[str, Any]:
        """Return a serializable selection contract for optimization reports."""

        predictive_best = self.predictive[0][2]
        selected_entry = self.selected[2]
        return {
            "predictive_ranking_dataset": "validation_calibration",
            "strategy_gate_dataset": "validation_calibration",
            "test_used_for_predictive_ranking": False,
            "test_used_for_strategy_gate": False,
            "selected_via": self.selected_via,
            "predictive_candidate_count": len(self.predictive),
            "strategy_eligible_candidate_count": len(
                self.strategy_eligible
            ),
            "predictive_best_model": str(
                predictive_best.get("name", "")
            ),
            "selected_model": str(selected_entry.get("name", "")),
            "predictive_and_selected_match": bool(
                predictive_best.get("name") == selected_entry.get("name")
            ),
            "policy": (
                "Predictive quality and strategy compatibility are ranked "
                "separately. Compatibility selection remains the publishing "
                "default until independent calibration validation is complete."
            ),
        }


def _predictive_score(candidate: ModelCandidate) -> float:
    return float(candidate[2].get("predictive_score", candidate[0]))


def _strategy_score(candidate: ModelCandidate) -> float:
    return float(
        candidate[2].get("strategy_selection_score", candidate[0])
    )


def _strategy_gate_passed(candidate: ModelCandidate) -> bool:
    gate = candidate[2].get("validation_trading_gate")
    return bool(isinstance(gate, dict) and gate.get("passed", False))


def rank_model_candidates(
    candidates: list[ModelCandidate],
) -> ModelSelectionRanking:
    """Rank candidates without accepting test-split inputs."""

    if not candidates:
        raise ValueError("candidates must not be empty")

    predictive = tuple(
        sorted(candidates, key=_predictive_score, reverse=True)
    )
    strategy_gated = tuple(
        sorted(
            candidates,
            key=lambda item: (
                _strategy_gate_passed(item),
                _strategy_score(item),
            ),
            reverse=True,
        )
    )
    strategy_eligible = tuple(
        item for item in strategy_gated if _strategy_gate_passed(item)
    )
    if strategy_eligible:
        selected = strategy_eligible[0]
        selected_via = "strategy_gate_then_strategy_score"
    else:
        selected = strategy_gated[0]
        selected_via = "research_fallback_strategy_score"
    return ModelSelectionRanking(
        predictive=predictive,
        strategy_gated=strategy_gated,
        strategy_eligible=strategy_eligible,
        selected=selected,
        selected_via=selected_via,
    )
