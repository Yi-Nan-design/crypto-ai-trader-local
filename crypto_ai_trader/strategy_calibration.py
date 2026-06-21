from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    auxiliary_signal_threshold,
    effective_thresholds,
    run_backtest,
)
from .features import feature_matrix
from .models import ModelBundle
from .risk_profile_catalog import resolve_risk_profile_catalog


@dataclass(frozen=True)
class ValidationCalibrationSplit:
    """Chronological validation partitions used for policy selection and gating."""

    calibration: pd.DataFrame
    gate: pd.DataFrame | None
    report: dict[str, Any]


@dataclass(frozen=True)
class StrategyCalibrationSearchSpace:
    """Explicit threshold and side-policy candidates for strategy calibration."""

    threshold_pairs: tuple[tuple[float, float], ...]
    trade_thresholds: tuple[float, ...]
    side_policies: tuple[str, ...]
    compact: bool


@dataclass(frozen=True)
class StrategyCalibrationCandidate:
    """One immutable strategy-policy candidate before validation replay."""

    risk_profile: str
    risk_params: dict[str, Any]
    long_threshold: float
    short_threshold: float
    trade_signal_threshold: float
    side_policy: str

    def backtest_config(
        self,
        base_cfg: BacktestConfig,
    ) -> BacktestConfig:
        """Merge this candidate with the frozen base backtest configuration."""

        return BacktestConfig(
            **{
                **asdict(base_cfg),
                **self.risk_params,
                "long_threshold": self.long_threshold,
                "short_threshold": self.short_threshold,
                "trade_signal_threshold": self.trade_signal_threshold,
                "trade_side_policy": self.side_policy,
            }
        )


@dataclass(frozen=True)
class StrategyCalibrationScoreBreakdown:
    """Auditable components of the frozen strategy-calibration score."""

    total: float
    expectancy_per_trade_after_cost: float
    expectancy_after_cost: float
    penalty: float
    side_policy_penalty: float
    leverage_penalty: float
    overtrade_penalty: float
    trade_count_stability_penalty: float
    cost_efficiency_ratio: float
    cost_efficiency_penalty: float
    low_absolute_return_penalty: float
    neutral_penalty: float
    false_neutral_penalty: float
    large_move_capture_penalty: float
    threshold_distance: float
    relaxed_threshold: bool
    threshold_stability_penalty: float

    def report_fields(self) -> dict[str, float | bool]:
        """Return the existing flat report fields without changing names."""

        return {
            "score": float(self.total),
            "expectancy_per_trade_after_cost": float(
                self.expectancy_per_trade_after_cost
            ),
            "expectancy_after_cost": float(self.expectancy_after_cost),
            "penalty": float(self.penalty),
            "side_policy_penalty": float(self.side_policy_penalty),
            "leverage_penalty": float(self.leverage_penalty),
            "overtrade_penalty": float(self.overtrade_penalty),
            "trade_count_stability_penalty": float(
                self.trade_count_stability_penalty
            ),
            "cost_efficiency_ratio": float(
                self.cost_efficiency_ratio
            ),
            "cost_efficiency_penalty": float(
                self.cost_efficiency_penalty
            ),
            "low_absolute_return_penalty": float(
                self.low_absolute_return_penalty
            ),
            "neutral_penalty": float(self.neutral_penalty),
            "false_neutral_penalty": float(
                self.false_neutral_penalty
            ),
            "large_move_capture_penalty": float(
                self.large_move_capture_penalty
            ),
            "threshold_distance": float(self.threshold_distance),
            "relaxed_threshold": bool(self.relaxed_threshold),
            "threshold_stability_penalty": float(
                self.threshold_stability_penalty
            ),
        }


@dataclass(frozen=True)
class StrategyCalibrationEvaluation:
    """Typed validation replay result for one strategy-policy candidate."""

    candidate: StrategyCalibrationCandidate
    config: BacktestConfig
    result: BacktestResult
    effective_long_threshold: float
    effective_short_threshold: float
    effective_direction_long_threshold: float
    effective_direction_short_threshold: float
    effective_direction_trade_threshold: float
    large_move: dict[str, float]
    archetype_side_gate: dict[str, Any]
    side_contribution_gate: dict[str, Any]
    validation_trading_gate: dict[str, Any]
    filter_reasons: tuple[str, ...]
    score: StrategyCalibrationScoreBreakdown
    long_side_preflight_gate: dict[str, Any]
    short_side_preflight_gate: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return whether this candidate passed all validation gates."""

        return bool(self.validation_trading_gate.get("passed", False))

    def to_report(self) -> dict[str, Any]:
        """Serialize to the legacy-compatible ranking entry contract."""

        candidate = self.candidate
        report = {
            "risk_profile": candidate.risk_profile,
            "side_policy": candidate.side_policy,
            "long_threshold": candidate.long_threshold,
            "short_threshold": candidate.short_threshold,
            "trade_signal_threshold": candidate.trade_signal_threshold,
            "effective_long_threshold": self.effective_long_threshold,
            "effective_short_threshold": self.effective_short_threshold,
            "effective_trade_signal_threshold": float(
                self.effective_direction_trade_threshold
            ),
            "effective_direction_long_threshold": float(
                self.effective_direction_long_threshold
            ),
            "effective_direction_short_threshold": float(
                self.effective_direction_short_threshold
            ),
            "effective_direction_trade_threshold": float(
                self.effective_direction_trade_threshold
            ),
            "backtest": asdict(self.result),
            "calibration_backtest": asdict(self.result),
            "gate_backtest": {},
            "backtest_config": asdict(self.config),
            "large_move": self.large_move,
            "calibration_large_move": self.large_move,
            "gate_large_move": {},
            "archetype_side_gate": self.archetype_side_gate,
            "gate_archetype_side_gate": {},
            "side_contribution_gate": self.side_contribution_gate,
            "gate_side_contribution_gate": {},
            "long_side_preflight_gate": (
                self.long_side_preflight_gate
            ),
            "short_side_preflight_gate": (
                self.short_side_preflight_gate
            ),
            "calibration_validation_trading_gate": (
                self.validation_trading_gate
            ),
            "validation_trading_gate": self.validation_trading_gate,
            "filter_reasons": list(self.filter_reasons),
        }
        report.update(self.score.report_fields())
        return report


@dataclass(frozen=True)
class StrategyCalibrationFinalization:
    """Final selected report entry and independent-gate audit counters."""

    best: dict[str, Any]
    validation_gate_evaluated_count: int
    final_gate_passed_count: int


def normalize_trade_side_policy(value: str | None) -> str:
    """Normalize external side-policy aliases to the backtest contract."""

    policy = str(value or "both").strip().lower()
    if policy in {"long", "long_only"}:
        return "long_only"
    if policy in {"short", "short_only"}:
        return "short_only"
    if policy in {"none", "no_trade"}:
        return "none"
    return "both"


def total_cost_drag_value(result: BacktestResult) -> float:
    """Return the canonical non-negative execution-cost drag."""

    total = float(getattr(result, "total_cost_drag", 0.0) or 0.0)
    if total > 0.0:
        return max(total, 0.0)
    return max(
        float(getattr(result, "fee_drag", 0.0))
        + float(getattr(result, "funding_drag", 0.0)),
        0.0,
    )


def cost_efficiency_ratio(result: BacktestResult) -> float:
    """Measure modeled costs as a share of absolute net edge plus costs."""

    fee_drag = total_cost_drag_value(result)
    net_return = abs(float(getattr(result, "total_return", 0.0)))
    denominator = net_return + fee_drag
    if denominator <= 1e-12:
        return 0.0
    return float(fee_drag / denominator)


def fee_drag_to_abs_return_ratio(result: BacktestResult) -> float:
    """Measure modeled costs relative to the absolute net return."""

    fee_drag = total_cost_drag_value(result)
    total_return = abs(float(getattr(result, "total_return", 0.0)))
    if total_return <= 1e-12:
        return 999.0 if fee_drag > 0 else 0.0
    return float(fee_drag / total_return)


def profit_quality_penalty(
    result: BacktestResult,
    *,
    min_net_return: float = 0.0020,
) -> float:
    """Penalize low absolute return and cost-dominated validation results."""

    total_return = float(getattr(result, "total_return", 0.0))
    low_return_penalty = max(0.0, float(min_net_return) - total_return) * 8.0
    cost_ratio = cost_efficiency_ratio(result)
    cost_penalty = max(0.0, cost_ratio - 0.35) * 0.18
    return float(
        min(0.08, low_return_penalty) + min(0.08, cost_penalty)
    )


def side_policy_allows(side: str, policy: str) -> bool:
    """Return whether a normalized side policy permits the requested side."""

    if policy == "none":
        return False
    if policy == "long_only":
        return side == "long"
    if policy == "short_only":
        return side == "short"
    return True


def archetype_matches_side(name: str, policy: str) -> bool:
    """Return whether a strategy-archetype name belongs to the policy side."""

    if policy == "none":
        return False
    lowered = str(name).lower()
    if policy == "long_only":
        return "long" in lowered
    if policy == "short_only":
        return "short" in lowered
    return True


def archetype_side_robustness(
    result: BacktestResult,
    *,
    min_trades: int,
    trade_side_policy: str | None = None,
) -> dict[str, Any]:
    """Evaluate side and strategy-archetype contribution on validation data."""

    policy = normalize_trade_side_policy(
        trade_side_policy
        or getattr(result, "trade_side_policy", "both")
    )
    min_archetype_trades = max(
        3,
        int(np.ceil(max(min_trades, 1) / 3.0)),
    )
    bad_items: list[dict[str, Any]] = []
    insufficient_items: list[dict[str, Any]] = []
    penalty = 0.0

    side_specs = [
        (
            "long",
            int(result.long_trades),
            float(result.long_total_return),
            float(result.long_profit_factor),
            1.0,
        ),
        (
            "short",
            int(result.short_trades),
            float(result.short_total_return),
            float(result.short_profit_factor),
            1.25,
        ),
    ]
    for name, trades, total_return, profit_factor, weight in side_specs:
        if not side_policy_allows(name, policy):
            continue
        if trades <= 0:
            continue
        if trades < min_archetype_trades:
            insufficient_items.append(
                {"name": name, "trades": trades, "scope": "side"}
            )
            continue
        if total_return < 0.0 and profit_factor < 1.0:
            item_penalty = min(
                0.14 * weight,
                abs(total_return) * 3.0 * weight
                + (1.0 - profit_factor) * 0.05 * weight,
            )
            penalty += item_penalty
            bad_items.append(
                {
                    "name": name,
                    "scope": "side",
                    "trades": trades,
                    "total_return": total_return,
                    "profit_factor": profit_factor,
                    "penalty": float(item_penalty),
                }
            )

    archetypes = result.strategy_archetype_summary or {}
    enough_archetype_count = 0
    for name, metrics in archetypes.items():
        if not archetype_matches_side(name, policy):
            continue
        trades = int(metrics.get("trades", 0))
        if trades <= 0:
            continue
        if trades < min_archetype_trades:
            insufficient_items.append(
                {"name": name, "trades": trades, "scope": "archetype"}
            )
            continue
        enough_archetype_count += 1
        total_return = float(metrics.get("total_return", 0.0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        if total_return < 0.0 and profit_factor < 1.0:
            item_penalty = min(
                0.10,
                abs(total_return) * 3.0
                + (1.0 - profit_factor) * 0.04,
            )
            penalty += item_penalty
            bad_items.append(
                {
                    "name": name,
                    "scope": "archetype",
                    "trades": trades,
                    "total_return": total_return,
                    "profit_factor": profit_factor,
                    "penalty": float(item_penalty),
                }
            )

    reasons = [
        f"bad_{item['scope']}_{item['name']}" for item in bad_items
    ]
    if (
        result.trades >= min_trades
        and archetypes
        and enough_archetype_count == 0
    ):
        penalty += 0.03
        reasons.append("insufficient_archetype_trades")
    return {
        "passed": not bad_items and penalty <= 0.0,
        "evaluated_side_policy": policy,
        "evaluated_sides": [
            side
            for side in ["long", "short"]
            if side_policy_allows(side, policy)
        ],
        "penalty": float(min(penalty, 0.30)),
        "min_archetype_trades": min_archetype_trades,
        "bad_archetype_sides": bad_items,
        "insufficient_archetype_sides": insufficient_items,
        "reasons": reasons,
    }


def trading_filter_penalty(
    result: BacktestResult,
    *,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> tuple[float, list[str]]:
    """Calculate soft validation penalties used for strategy ranking."""

    penalty = 0.0
    reasons: list[str] = []
    if result.trades < min_trades:
        penalty += 0.12
        reasons.append(f"trades_below_{min_trades}")
    if result.total_return < 0.0:
        penalty += min(0.25, abs(result.total_return) * 2.0 + 0.04)
        reasons.append("validation_return_negative")
    if result.max_drawdown < max_drawdown_floor:
        penalty += min(
            0.20,
            abs(result.max_drawdown - max_drawdown_floor),
        )
        reasons.append("drawdown_too_deep")
    if result.profit_factor < min_profit_factor:
        penalty += min(
            0.20,
            (min_profit_factor - result.profit_factor) * 0.30,
        )
        reasons.append(f"profit_factor_below_{min_profit_factor}")
    expectancy_per_trade = float(
        getattr(result, "expectancy_per_trade_after_cost", 0.0)
    )
    if expectancy_per_trade < 0.0:
        penalty += min(
            0.18,
            abs(expectancy_per_trade) * 45.0 + 0.03,
        )
        reasons.append("negative_expectancy_after_cost")
    archetype_gate = archetype_side_robustness(
        result,
        min_trades=min_trades,
    )
    if float(archetype_gate["penalty"]) > 0:
        penalty += float(archetype_gate["penalty"])
        reasons.extend(
            str(item) for item in archetype_gate.get("reasons", [])
        )
    return penalty, reasons


def validation_trading_gate(
    result: BacktestResult,
    *,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    archetype_gate: dict[str, Any] | None = None,
    min_total_return: float = 0.0,
    max_fee_drag_to_abs_return: float | None = None,
    min_expectancy_per_trade_after_cost: float = 0.0,
) -> dict[str, Any]:
    """Apply hard validation gates before a strategy may be selected."""

    reasons: list[str] = []
    if result.trades < min_trades:
        reasons.append(f"trades_below_{min_trades}")
    if result.total_return <= 0.0:
        reasons.append("validation_return_not_positive")
    if result.total_return < float(min_total_return):
        reasons.append(f"validation_return_below_{min_total_return}")
    if result.profit_factor < min_profit_factor:
        reasons.append(f"profit_factor_below_{min_profit_factor}")
    expectancy_per_trade = float(
        getattr(result, "expectancy_per_trade_after_cost", 0.0)
    )
    if expectancy_per_trade <= 0.0:
        reasons.append("expectancy_per_trade_not_positive")
    if expectancy_per_trade < float(
        min_expectancy_per_trade_after_cost
    ):
        reasons.append("expectancy_per_trade_below_quality_min")
    fee_drag_ratio = fee_drag_to_abs_return_ratio(result)
    if (
        max_fee_drag_to_abs_return is not None
        and fee_drag_ratio > float(max_fee_drag_to_abs_return)
    ):
        reasons.append("fee_drag_ratio_above_validation_max")
    if result.max_drawdown < max_drawdown_floor:
        reasons.append("drawdown_too_deep")
    leverage = max(1, int(getattr(result, "leverage", 1) or 1))
    if leverage > 1:
        leveraged_min_return = float(min_total_return) * (
            1.0 + 0.65 * float(leverage - 1)
        )
        leveraged_min_trades = int(
            np.ceil(
                float(min_trades)
                * (1.0 + 0.25 * float(leverage - 1))
            )
        )
        leveraged_min_expectancy = float(
            min_expectancy_per_trade_after_cost
        ) * (1.0 + 0.50 * float(leverage - 1))
        leveraged_max_fee_ratio = (
            min(float(max_fee_drag_to_abs_return), 0.35)
            if max_fee_drag_to_abs_return is not None
            else 0.35
        )
        if result.total_return < leveraged_min_return:
            reasons.append("leveraged_return_quality_below_min")
        if result.trades < leveraged_min_trades:
            reasons.append("leveraged_trade_count_below_min")
        if expectancy_per_trade < leveraged_min_expectancy:
            reasons.append("leveraged_expectancy_quality_below_min")
        if fee_drag_ratio > leveraged_max_fee_ratio:
            reasons.append("leveraged_fee_drag_ratio_too_high")
    if not result.rr_gate_passed:
        reasons.append("realized_risk_reward_gate_failed")
    if (
        archetype_gate is not None
        and not bool(archetype_gate.get("passed", False))
    ):
        reasons.append("archetype_side_gate_failed")
        reasons.extend(
            str(item) for item in archetype_gate.get("reasons", [])
        )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "evaluated_side_policy": (
            str(archetype_gate.get("evaluated_side_policy"))
            if (
                isinstance(archetype_gate, dict)
                and archetype_gate.get("evaluated_side_policy")
            )
            else str(
                getattr(result, "trade_side_policy", "both") or "both"
            )
        ),
        "required_min_trades": int(min_trades),
        "required_min_profit_factor": float(min_profit_factor),
        "required_max_drawdown_floor": float(max_drawdown_floor),
        "required_min_total_return": float(min_total_return),
        "required_max_fee_drag_to_abs_return": (
            float(max_fee_drag_to_abs_return)
            if max_fee_drag_to_abs_return is not None
            else None
        ),
        "required_min_expectancy_per_trade_after_cost": float(
            min_expectancy_per_trade_after_cost
        ),
        "selected_leverage": leverage,
        "fee_drag_to_abs_return_ratio": fee_drag_ratio,
        "required_positive_return": True,
        "required_positive_expectancy_per_trade_after_cost": True,
        "required_realized_risk_reward_gate": True,
    }


def directional_signal_capture_metrics(
    frame: pd.DataFrame,
    direction_prob: dict[str, np.ndarray],
    *,
    long_threshold: float,
    short_threshold: float,
    cutoff: float,
    neutral_cutoff: float,
) -> dict[str, float]:
    """Measure long, short, large-move, and neutral signal behavior."""

    returns = (
        pd.to_numeric(frame["future_return"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    prob_long = np.asarray(
        direction_prob.get("long", direction_prob.get("up", [])),
        dtype=float,
    )
    fallback_short = 1.0 - np.asarray(
        direction_prob.get("up", np.zeros(len(returns))),
        dtype=float,
    )
    prob_short = np.asarray(
        direction_prob.get("short", fallback_short),
        dtype=float,
    )
    prob_trade = np.asarray(
        direction_prob.get("trade", np.ones(len(returns))),
        dtype=float,
    )
    short_signal_threshold = max(
        1.0 - float(short_threshold),
        0.0,
    )
    trade_threshold = (
        float(direction_prob["trade_threshold"][0])
        if "trade_threshold" in direction_prob
        else 0.0
    )
    if "long_threshold" in direction_prob:
        long_threshold = float(
            np.asarray(
                direction_prob["long_threshold"],
                dtype=float,
            )[0]
        )
    if "short_signal_threshold" in direction_prob:
        short_signal_threshold = float(
            np.asarray(
                direction_prob["short_signal_threshold"],
                dtype=float,
            )[0]
        )
    big_up = returns >= cutoff
    big_down = returns <= -cutoff
    trade_signal = prob_trade >= trade_threshold
    long_signal = (
        trade_signal
        & (prob_long >= long_threshold)
        & (prob_long >= prob_short)
    )
    short_signal = (
        trade_signal
        & (prob_short >= short_signal_threshold)
        & (prob_short > prob_long)
    )
    neutral = np.abs(returns) < abs(float(neutral_cutoff))
    false_trade_on_neutral = (
        float(
            (
                long_signal[neutral]
                | short_signal[neutral]
            ).mean()
        )
        if int(neutral.sum())
        else 0.0
    )
    long_precision = (
        float((returns[long_signal] > 0).mean())
        if int(long_signal.sum())
        else 0.0
    )
    short_precision = (
        float((returns[short_signal] < 0).mean())
        if int(short_signal.sum())
        else 0.0
    )
    neutral_rate = (
        float((~long_signal & ~short_signal).mean())
        if len(returns)
        else 0.0
    )
    up_capture = (
        float(long_signal[big_up].mean())
        if int(big_up.sum())
        else 0.0
    )
    down_capture = (
        float(short_signal[big_down].mean())
        if int(big_down.sum())
        else 0.0
    )
    return {
        "large_move_cutoff": float(cutoff),
        "large_up_count": float(int(big_up.sum())),
        "large_down_count": float(int(big_down.sum())),
        "large_up_capture": up_capture,
        "large_down_capture": down_capture,
        "large_move_capture": (
            (up_capture + down_capture) / 2.0
            if int(big_up.sum()) and int(big_down.sum())
            else max(up_capture, down_capture)
        ),
        "long_signal_count": float(int(long_signal.sum())),
        "short_signal_count": float(int(short_signal.sum())),
        "long_signal_precision": long_precision,
        "short_signal_precision": short_precision,
        "neutral_no_trade_rate": neutral_rate,
        "false_trade_on_neutral_rate": false_trade_on_neutral,
        "trade_signal_rate": (
            float(trade_signal.mean()) if len(trade_signal) else 1.0
        ),
        "long_signal_threshold": float(long_threshold),
        "short_signal_threshold": float(short_signal_threshold),
        "trade_signal_threshold": trade_threshold,
    }


def directional_side_preflight_gate(
    directional_report: dict[str, Any] | None,
    side: str,
) -> dict[str, Any]:
    """Reject weak directional auxiliary models before policy calibration."""

    side = (
        "short"
        if str(side).strip().lower() == "short"
        else "long"
    )
    directions = (
        directional_report.get("directions", {})
        if isinstance(directional_report, dict)
        else {}
    )
    side_report = (
        directions.get(side, {})
        if isinstance(directions, dict)
        else {}
    )
    ranking = (
        side_report.get("ranking", [])
        if isinstance(side_report, dict)
        else []
    )
    best = (
        ranking[0]
        if ranking and isinstance(ranking[0], dict)
        else {}
    )
    valid_rows = (
        int(side_report.get("valid_rows", 0) or 0)
        if isinstance(side_report, dict)
        else 0
    )
    valid_positive_count = (
        int(side_report.get("valid_positive_count", 0) or 0)
        if isinstance(side_report, dict)
        else 0
    )
    valid_positive_rate = (
        float(side_report.get("valid_positive_rate", 0.0) or 0.0)
        if isinstance(side_report, dict)
        else 0.0
    )
    signal_count = int(best.get("signal_count", 0) or 0)
    signal_rate = float(best.get("signal_rate", 0.0) or 0.0)
    signal_precision = float(best.get("signal_precision", 0.0) or 0.0)
    false_signal_rate = float(
        best.get("false_signal_on_negative_rate", 0.0) or 0.0
    )
    signal_expectancy = float(
        best.get("signal_expectancy_after_cost", 0.0) or 0.0
    )
    signal_total_return = float(
        best.get("signal_total_return_after_cost", 0.0) or 0.0
    )
    signal_profit_factor = float(
        best.get("signal_profit_factor_after_cost", 0.0) or 0.0
    )
    min_signal_count = max(
        12,
        int(np.ceil(float(valid_rows) * 0.003)),
    )
    max_signal_rate = 0.24 if side == "long" else 0.20
    reasons: list[str] = []
    prefix = f"{side}_preflight"
    if (
        not isinstance(side_report, dict)
        or side_report.get("status") != "trained"
    ):
        reasons.append(f"{prefix}_not_trained")
    if valid_positive_count < 30:
        reasons.append(f"{prefix}_insufficient_positive_count")
    if signal_count < min_signal_count:
        reasons.append(f"{prefix}_sparse_signal")
    if signal_rate < 0.003 or signal_rate > max_signal_rate:
        reasons.append(f"{prefix}_signal_rate_out_of_bounds")
    if signal_expectancy <= 0.0:
        reasons.append(f"{prefix}_expectancy_not_positive")
    if signal_total_return <= 0.0:
        reasons.append(f"{prefix}_return_not_positive")
    if signal_profit_factor < 1.20:
        reasons.append(f"{prefix}_profit_factor_below_1_2")
    if signal_precision < valid_positive_rate + 0.03:
        reasons.append(f"{prefix}_precision_edge_too_small")
    if false_signal_rate > 0.35:
        reasons.append(f"{prefix}_false_signal_too_high")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "side": side,
        "status": (
            side_report.get("status", "missing")
            if isinstance(side_report, dict)
            else "missing"
        ),
        "valid_rows": valid_rows,
        "valid_positive_count": valid_positive_count,
        "valid_positive_rate": valid_positive_rate,
        "min_signal_count": min_signal_count,
        "max_signal_rate": max_signal_rate,
        "signal_count": signal_count,
        "signal_rate": signal_rate,
        "signal_precision": signal_precision,
        "false_signal_on_negative_rate": false_signal_rate,
        "signal_expectancy_after_cost": signal_expectancy,
        "signal_total_return_after_cost": signal_total_return,
        "signal_profit_factor_after_cost": signal_profit_factor,
    }


def short_side_preflight_gate(
    directional_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the short auxiliary-model preflight contract."""

    return directional_side_preflight_gate(directional_report, "short")


def long_side_preflight_gate(
    directional_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the long auxiliary-model preflight contract."""

    return directional_side_preflight_gate(directional_report, "long")


def side_contribution_gate(
    result: BacktestResult,
    *,
    side_policy: str,
    min_trades: int,
    min_profit_factor: float,
    long_preflight_gate: dict[str, Any] | None = None,
    short_preflight_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require every selected side to contribute positive validation edge."""

    policy = normalize_trade_side_policy(side_policy)
    if policy == "none":
        return {
            "passed": True,
            "reasons": [],
            "policy": policy,
            "long_candidate_allowed": False,
            "short_candidate_allowed": False,
        }

    def evaluate_side(
        side: str,
        preflight_gate: dict[str, Any] | None,
        required_trades: int,
        required_pf: float,
    ) -> dict[str, Any]:
        side_reasons: list[str] = []
        preflight_passed = bool(
            (preflight_gate or {}).get("passed", False)
        )
        trades = int(getattr(result, f"{side}_trades", 0))
        total_return = float(
            getattr(result, f"{side}_total_return", 0.0)
        )
        profit_factor = float(
            getattr(result, f"{side}_profit_factor", 0.0)
        )
        if trades < required_trades:
            side_reasons.append(f"{side}_trades_below_min")
        if total_return <= 0.0:
            side_reasons.append(f"{side}_return_not_positive")
        if profit_factor < required_pf:
            side_reasons.append(f"{side}_profit_factor_below_min")
        validation_edge_passed = not side_reasons
        if not preflight_passed and not validation_edge_passed:
            side_reasons.insert(0, f"{side}_preflight_failed")
            side_reasons.extend(
                str(item)
                for item in (preflight_gate or {}).get("reasons", [])
            )
        return {
            "passed": not side_reasons,
            "reasons": side_reasons,
            "preflight_passed": preflight_passed,
            "preflight_overridden_by_validation_edge": bool(
                (not preflight_passed) and validation_edge_passed
            ),
            "required_trades": int(required_trades),
            "required_profit_factor": float(required_pf),
            "trades": trades,
            "total_return": total_return,
            "profit_factor": profit_factor,
            "preflight_gate": preflight_gate or {},
        }

    reasons: list[str] = []
    long_result: dict[str, Any] | None = None
    short_result: dict[str, Any] | None = None
    if policy in {"long_only", "both"}:
        if policy == "long_only":
            required_long_trades = int(min_trades)
            required_long_pf = max(float(min_profit_factor), 1.15)
        else:
            required_long_trades = max(
                6,
                int(np.ceil(float(min_trades) / 3.0)),
                int(np.ceil(float(result.trades) * 0.25)),
            )
            required_long_pf = max(float(min_profit_factor), 1.08)
        long_result = evaluate_side(
            "long",
            long_preflight_gate,
            required_long_trades,
            required_long_pf,
        )
        reasons.extend(long_result["reasons"])
    if policy in {"short_only", "both"}:
        if policy == "short_only":
            required_short_trades = int(min_trades)
            required_short_pf = max(float(min_profit_factor), 1.20)
        else:
            required_short_trades = max(
                6,
                int(np.ceil(float(min_trades) / 3.0)),
                int(np.ceil(float(result.trades) * 0.25)),
            )
            required_short_pf = max(float(min_profit_factor), 1.10)
        short_result = evaluate_side(
            "short",
            short_preflight_gate,
            required_short_trades,
            required_short_pf,
        )
        reasons.extend(short_result["reasons"])
    return {
        "passed": not reasons,
        "reasons": reasons,
        "policy": policy,
        "long_candidate_allowed": bool(
            long_result and long_result.get("passed", False)
        ),
        "short_candidate_allowed": bool(
            short_result and short_result.get("passed", False)
        ),
        "long_preflight_overridden_by_validation_edge": bool(
            long_result
            and long_result.get(
                "preflight_overridden_by_validation_edge",
                False,
            )
        ),
        "short_preflight_overridden_by_validation_edge": bool(
            short_result
            and short_result.get(
                "preflight_overridden_by_validation_edge",
                False,
            )
        ),
        "required_long_trades": int(
            (long_result or {}).get("required_trades", 0)
        ),
        "required_short_trades": int(
            (short_result or {}).get("required_trades", 0)
        ),
        "required_long_profit_factor": float(
            (long_result or {}).get("required_profit_factor", 0.0)
        ),
        "required_short_profit_factor": float(
            (short_result or {}).get("required_profit_factor", 0.0)
        ),
        "long_trades": int(getattr(result, "long_trades", 0)),
        "short_trades": int(getattr(result, "short_trades", 0)),
        "long_total_return": float(
            getattr(result, "long_total_return", 0.0)
        ),
        "short_total_return": float(
            getattr(result, "short_total_return", 0.0)
        ),
        "long_profit_factor": float(
            getattr(result, "long_profit_factor", 0.0)
        ),
        "short_profit_factor": float(
            getattr(result, "short_profit_factor", 0.0)
        ),
        "long_side_gate": long_result or {},
        "short_side_gate": short_result or {},
        "long_preflight_gate": long_preflight_gate or {},
        "short_preflight_gate": short_preflight_gate or {},
    }


def small_account_risk_profiles(
    base_cfg: BacktestConfig,
    *,
    compact: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the versioned TOML risk-profile catalog."""

    return list(
        resolve_risk_profile_catalog(
            base_cfg,
            compact=compact,
        ).profiles
    )


def threshold_grid(
    base_cfg: BacktestConfig,
    *,
    compact: bool = False,
) -> list[tuple[float, float]]:
    """Build deterministic long/short probability threshold candidates."""

    if compact:
        values = [
            (
                round(float(base_cfg.long_threshold), 4),
                round(float(base_cfg.short_threshold), 4),
            ),
            (0.54, 0.46),
            (0.58, 0.42),
            (0.63, 0.37),
        ]
        ordered: list[tuple[float, float]] = []
        for pair in values:
            long_threshold, short_threshold = pair
            if short_threshold < long_threshold and pair not in ordered:
                ordered.append(pair)
        return ordered

    long_values = [
        0.50,
        0.52,
        0.54,
        0.58,
        0.63,
        0.66,
        0.70,
        float(base_cfg.long_threshold),
    ]
    short_values = [
        0.50,
        0.48,
        0.46,
        0.42,
        0.37,
        0.34,
        0.30,
        float(base_cfg.short_threshold),
    ]
    pairs = {
        (round(long_threshold, 4), round(short_threshold, 4))
        for long_threshold in long_values
        for short_threshold in short_values
        if short_threshold < long_threshold
    }
    return sorted(pairs)


def trade_threshold_grid(
    base_cfg: BacktestConfig,
    *,
    compact: bool = False,
) -> list[float]:
    """Build deterministic tradeability probability threshold candidates."""

    if compact:
        values = [float(base_cfg.trade_signal_threshold), 0.54, 0.70]
        ordered: list[float] = []
        for value in values:
            rounded = round(float(value), 4)
            if rounded not in ordered:
                ordered.append(rounded)
        return ordered
    values = [
        0.50,
        0.54,
        0.57,
        0.63,
        0.70,
        0.78,
        0.82,
        float(base_cfg.trade_signal_threshold),
    ]
    return sorted({round(float(value), 4) for value in values})


def trade_side_policy_grid(base_cfg: BacktestConfig) -> list[str]:
    """Build side-policy candidates while preserving the configured policy."""

    configured = normalize_trade_side_policy(
        str(getattr(base_cfg, "trade_side_policy", "both") or "both")
    )
    values = ["long_only", configured, "both", "short_only"]
    unique: list[str] = []
    for value in values:
        if value not in {"both", "long_only", "short_only"}:
            value = "both"
        if value not in unique:
            unique.append(value)
    return unique


def build_strategy_calibration_search_space(
    base_cfg: BacktestConfig,
    *,
    compact: bool = False,
) -> StrategyCalibrationSearchSpace:
    """Create the complete strategy-policy search space without market data."""

    return StrategyCalibrationSearchSpace(
        threshold_pairs=tuple(threshold_grid(base_cfg, compact=compact)),
        trade_thresholds=tuple(
            trade_threshold_grid(base_cfg, compact=compact)
        ),
        side_policies=tuple(trade_side_policy_grid(base_cfg)),
        compact=bool(compact),
    )


def split_validation_for_strategy_calibration(
    valid_df: pd.DataFrame,
    *,
    purge_rows: int,
    calibration_fraction: float = 0.55,
    min_rows: int = 80,
) -> ValidationCalibrationSplit:
    """Split validation chronologically; the test split is never accepted."""

    rows = len(valid_df)
    purge_rows = max(0, int(purge_rows or 0))
    min_required = max(2 * int(min_rows) + purge_rows, 220)
    if rows < min_required:
        return ValidationCalibrationSplit(
            calibration=valid_df.reset_index(drop=True),
            gate=None,
            report={
                "enabled": False,
                "reason": (
                    "insufficient_validation_rows_for_calibration_gate_split"
                ),
                "rows": int(rows),
                "required_rows": int(min_required),
                "purge_rows": int(purge_rows),
                "selection_dataset": "validation_full",
                "gate_dataset": "not_available",
                "test_used_for_selection": False,
            },
        )

    split_at = int(rows * float(calibration_fraction))
    split_at = max(
        int(min_rows) + purge_rows,
        min(rows - int(min_rows), split_at),
    )
    calibration_end = max(int(min_rows), split_at - purge_rows)
    gate_start = min(rows, split_at)
    calibration_df = valid_df.iloc[:calibration_end].reset_index(drop=True)
    gate_df = valid_df.iloc[gate_start:].reset_index(drop=True)
    if len(calibration_df) < min_rows or len(gate_df) < min_rows:
        return ValidationCalibrationSplit(
            calibration=valid_df.reset_index(drop=True),
            gate=None,
            report={
                "enabled": False,
                "reason": "calibration_or_gate_slice_too_small",
                "rows": int(rows),
                "calibration_rows": int(len(calibration_df)),
                "gate_rows": int(len(gate_df)),
                "purge_rows": int(purge_rows),
                "selection_dataset": "validation_full",
                "gate_dataset": "not_available",
                "test_used_for_selection": False,
            },
        )

    return ValidationCalibrationSplit(
        calibration=calibration_df,
        gate=gate_df,
        report={
            "enabled": True,
            "reason": "",
            "rows": int(rows),
            "calibration_rows": int(len(calibration_df)),
            "gate_rows": int(len(gate_df)),
            "purge_rows": int(purge_rows),
            "calibration_start": 0,
            "calibration_end": int(calibration_end),
            "gate_start": int(gate_start),
            "gate_end": int(rows),
            "selection_dataset": "validation_calibration",
            "gate_dataset": "validation_gate",
            "test_used_for_selection": False,
        },
    )


def large_move_cutoff(frame: pd.DataFrame, label_min_return: float) -> float:
    returns = frame.get("future_return")
    if returns is None or returns.empty:
        return max(abs(label_min_return) * 2.0, 0.001)
    abs_returns = pd.to_numeric(returns, errors="coerce").abs().dropna()
    if abs_returns.empty:
        return max(abs(label_min_return) * 2.0, 0.001)
    quantile_cutoff = float(abs_returns.quantile(0.70))
    return max(abs(label_min_return) * 2.0, quantile_cutoff, 0.001)


def iter_strategy_calibration_candidates(
    risk_profiles: list[dict[str, Any]],
    threshold_pairs: list[tuple[float, float]],
    trade_thresholds: list[float],
    side_policies: list[str],
) -> Iterator[StrategyCalibrationCandidate]:
    """Yield candidates in the frozen legacy nested-loop order."""

    for risk_profile in risk_profiles:
        for long_threshold, short_threshold in threshold_pairs:
            for trade_threshold in trade_thresholds:
                for side_policy in side_policies:
                    yield StrategyCalibrationCandidate(
                        risk_profile=str(risk_profile["name"]),
                        risk_params=dict(risk_profile["params"]),
                        long_threshold=float(long_threshold),
                        short_threshold=float(short_threshold),
                        trade_signal_threshold=float(trade_threshold),
                        side_policy=str(side_policy),
                    )


def score_strategy_calibration_candidate(
    *,
    candidate: StrategyCalibrationCandidate,
    config: BacktestConfig,
    base_cfg: BacktestConfig,
    result: BacktestResult,
    capture: dict[str, float],
    penalty: float,
    validation_rows: int,
    min_trades: int,
) -> StrategyCalibrationScoreBreakdown:
    """Apply the frozen return-first strategy calibration score."""

    trade_rate = float(result.trades) / max(
        float(validation_rows),
        1.0,
    )
    overtrade_penalty = min(
        0.30,
        max(0.0, trade_rate - 0.10) * 1.10,
    )
    trade_count_stability_penalty = min(
        0.10,
        max(
            0.0,
            (
                2.0 * float(min_trades)
                - float(result.trades)
            )
            / max(2.0 * float(min_trades), 1.0),
        )
        * 0.12,
    )
    neutral_penalty = min(
        0.12,
        max(
            0.0,
            0.20 - float(capture.get("neutral_no_trade_rate", 0.0)),
        )
        * 0.50,
    )
    false_neutral_penalty = min(
        0.12,
        float(capture.get("false_trade_on_neutral_rate", 0.0)) * 0.70,
    )
    large_move_capture_penalty = min(
        0.12,
        max(
            0.0,
            0.08 - float(capture["large_move_capture"]),
        )
        * 0.85,
    )
    threshold_distance = (
        abs(candidate.long_threshold - float(base_cfg.long_threshold))
        + abs(candidate.short_threshold - float(base_cfg.short_threshold))
        + 0.5
        * abs(
            candidate.trade_signal_threshold
            - float(base_cfg.trade_signal_threshold)
        )
    )
    relaxed_threshold = bool(
        candidate.long_threshold < float(base_cfg.long_threshold)
        or candidate.short_threshold > float(base_cfg.short_threshold)
        or candidate.trade_signal_threshold
        < float(base_cfg.trade_signal_threshold)
    )
    cost_ratio = cost_efficiency_ratio(result)
    threshold_stability_penalty = 0.02 * threshold_distance
    if relaxed_threshold and (
        trade_rate > 0.10 or cost_ratio > 0.35
    ):
        threshold_stability_penalty += (
            0.12 * threshold_distance
            + max(0.0, trade_rate - 0.10) * 0.60
            + max(0.0, cost_ratio - 0.35) * 0.08
        )
    cost_efficiency_penalty = min(
        0.12,
        max(0.0, cost_ratio - 0.35) * 0.25,
    )
    low_absolute_return_penalty = min(
        0.10,
        max(0.0, 0.0020 - float(result.total_return)) * 8.0,
    )
    side_policy_penalty = {
        "long_only": 0.0,
        "both": 0.0015,
        "short_only": 0.0025,
    }.get(candidate.side_policy, 0.004)
    leverage_penalty = 0.0
    if int(getattr(config, "leverage", 1) or 1) > 1:
        leverage_penalty = (
            0.010 * float(int(config.leverage) - 1)
            + min(0.060, max(0.0, cost_ratio - 0.25) * 0.18)
            + min(
                0.050,
                max(0.0, 0.0040 - float(result.total_return)) * 7.0,
            )
        )
    expectancy_per_trade = float(
        getattr(result, "expectancy_per_trade_after_cost", 0.0)
    )
    expectancy_active_bar = float(
        getattr(result, "expectancy_after_cost", 0.0)
    )
    profit_factor_bonus = 0.025 * float(
        np.log1p(
            min(max(float(result.profit_factor), 0.0), 3.0)
        )
    )
    total = (
        3.80 * result.total_return
        + 70.0 * expectancy_per_trade
        + 16.0 * expectancy_active_bar
        + profit_factor_bonus
        + 0.08 * capture["large_move_capture"]
        - 0.16 * abs(result.max_drawdown)
        - 0.35 * min(total_cost_drag_value(result), 0.25)
        - cost_efficiency_penalty
        - low_absolute_return_penalty
        - overtrade_penalty
        - trade_count_stability_penalty
        - neutral_penalty
        - false_neutral_penalty
        - large_move_capture_penalty
        - threshold_stability_penalty
        - side_policy_penalty
        - leverage_penalty
        - penalty
    )
    return StrategyCalibrationScoreBreakdown(
        total=float(total),
        expectancy_per_trade_after_cost=expectancy_per_trade,
        expectancy_after_cost=expectancy_active_bar,
        penalty=float(penalty),
        side_policy_penalty=float(side_policy_penalty),
        leverage_penalty=float(leverage_penalty),
        overtrade_penalty=float(overtrade_penalty),
        trade_count_stability_penalty=float(
            trade_count_stability_penalty
        ),
        cost_efficiency_ratio=float(cost_ratio),
        cost_efficiency_penalty=float(cost_efficiency_penalty),
        low_absolute_return_penalty=float(
            low_absolute_return_penalty
        ),
        neutral_penalty=float(neutral_penalty),
        false_neutral_penalty=float(false_neutral_penalty),
        large_move_capture_penalty=float(
            large_move_capture_penalty
        ),
        threshold_distance=float(threshold_distance),
        relaxed_threshold=relaxed_threshold,
        threshold_stability_penalty=float(
            threshold_stability_penalty
        ),
    )


def evaluate_strategy_calibration_candidate(
    valid_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    candidate: StrategyCalibrationCandidate,
    direction_prob: dict[str, np.ndarray],
    *,
    cutoff: float,
    label_min_return: float,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    long_preflight_gate: dict[str, Any],
    short_preflight_gate: dict[str, Any],
) -> StrategyCalibrationEvaluation:
    """Backtest, gate, and score one immutable strategy candidate."""

    config = candidate.backtest_config(base_cfg)
    result, _ = run_backtest(valid_df, bundle, config)
    effective_long, effective_short = effective_thresholds(config)
    short_signal_threshold = max(
        1.0 - float(effective_short),
        0.5 + float(config.min_confidence_gap),
    )
    uses_directional = bool(
        np.asarray(
            direction_prob.get("uses_directional_models", [])
        ).any()
    )
    capture_long_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_long",
            effective_long,
        )
        if uses_directional
        else effective_long
    )
    capture_short_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_short",
            short_signal_threshold,
        )
        if uses_directional
        else short_signal_threshold
    )
    capture_trade_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_trade",
            candidate.trade_signal_threshold,
        )
        if uses_directional
        else candidate.trade_signal_threshold
    )
    capture_prob = {
        **direction_prob,
        "trade_threshold": np.full(
            len(valid_df),
            float(capture_trade_threshold),
            dtype=float,
        ),
        "long_threshold": np.full(
            len(valid_df),
            float(capture_long_threshold),
            dtype=float,
        ),
        "short_signal_threshold": np.full(
            len(valid_df),
            float(capture_short_threshold),
            dtype=float,
        ),
    }
    capture = directional_signal_capture_metrics(
        valid_df,
        capture_prob,
        long_threshold=capture_long_threshold,
        short_threshold=effective_short,
        cutoff=cutoff,
        neutral_cutoff=label_min_return,
    )
    penalty, reasons = trading_filter_penalty(
        result,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
    )
    archetype_gate = archetype_side_robustness(
        result,
        min_trades=min_trades,
    )
    side_gate = side_contribution_gate(
        result,
        side_policy=candidate.side_policy,
        min_trades=min_trades,
        min_profit_factor=min_profit_factor,
        long_preflight_gate=long_preflight_gate,
        short_preflight_gate=short_preflight_gate,
    )
    hard_gate = validation_trading_gate(
        result,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
        archetype_gate=archetype_gate,
        min_total_return=0.0020,
        max_fee_drag_to_abs_return=0.45,
        min_expectancy_per_trade_after_cost=0.00004,
    )
    if not bool(side_gate.get("passed", False)):
        hard_gate = {
            **hard_gate,
            "passed": False,
            "reasons": (
                list(hard_gate.get("reasons", []))
                + list(side_gate.get("reasons", []))
            ),
        }
    calibration_hard_gate = {
        **hard_gate,
        "decision_dataset": "validation_calibration",
    }
    score = score_strategy_calibration_candidate(
        candidate=candidate,
        config=config,
        base_cfg=base_cfg,
        result=result,
        capture=capture,
        penalty=penalty,
        validation_rows=len(valid_df),
        min_trades=min_trades,
    )
    return StrategyCalibrationEvaluation(
        candidate=candidate,
        config=config,
        result=result,
        effective_long_threshold=float(effective_long),
        effective_short_threshold=float(effective_short),
        effective_direction_long_threshold=float(
            capture_long_threshold
        ),
        effective_direction_short_threshold=float(
            capture_short_threshold
        ),
        effective_direction_trade_threshold=float(
            capture_trade_threshold
        ),
        large_move=capture,
        archetype_side_gate=archetype_gate,
        side_contribution_gate=side_gate,
        validation_trading_gate=calibration_hard_gate,
        filter_reasons=tuple(str(item) for item in reasons),
        score=score,
        long_side_preflight_gate=long_preflight_gate,
        short_side_preflight_gate=short_preflight_gate,
    )


def build_no_trade_calibration_fallback(
    valid_df: pd.DataFrame,
    validation_gate_df: pd.DataFrame | None,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    *,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> StrategyCalibrationFinalization:
    """Build the explicit no-trade result when no candidate passes."""

    no_trade_cfg = BacktestConfig(
        **{
            **asdict(base_cfg),
            "long_threshold": 0.99,
            "short_threshold": 0.01,
            "min_confidence_gap": 0.49,
            "trade_signal_threshold": 0.99,
            "trade_side_policy": "none",
        }
    )
    no_trade_eval_df = (
        validation_gate_df
        if validation_gate_df is not None
        else valid_df
    )
    no_trade_result, _ = run_backtest(
        no_trade_eval_df,
        bundle,
        no_trade_cfg,
    )
    archetype_gate = archetype_side_robustness(
        no_trade_result,
        min_trades=min_trades,
    )
    no_trade_gate = validation_trading_gate(
        no_trade_result,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
        archetype_gate=archetype_gate,
    )
    best = {
        "risk_profile": "no_trade_recommended",
        "long_threshold": no_trade_cfg.long_threshold,
        "short_threshold": no_trade_cfg.short_threshold,
        "trade_signal_threshold": no_trade_cfg.trade_signal_threshold,
        "side_policy": no_trade_cfg.trade_side_policy,
        "effective_long_threshold": 0.99,
        "effective_short_threshold": 0.01,
        "effective_trade_signal_threshold": 0.99,
        "score": -999.0,
        "backtest": asdict(no_trade_result),
        "calibration_backtest": {},
        "gate_backtest": (
            asdict(no_trade_result)
            if validation_gate_df is not None
            else {}
        ),
        "backtest_config": asdict(no_trade_cfg),
        "large_move": {},
        "calibration_large_move": {},
        "gate_large_move": {},
        "penalty": 999.0,
        "overtrade_penalty": 0.0,
        "neutral_penalty": 0.0,
        "false_neutral_penalty": 0.0,
        "threshold_stability_penalty": 0.0,
        "archetype_side_gate": archetype_gate,
        "validation_trading_gate": {
            **no_trade_gate,
            "passed": False,
            "reasons": ["no_validation_threshold_config_passed"],
            "decision_dataset": (
                "validation_gate"
                if validation_gate_df is not None
                else "validation_full"
            ),
        },
        "filter_reasons": [
            "no_validation_threshold_config_passed"
        ],
        "fallback_no_trade_recommended": True,
    }
    return StrategyCalibrationFinalization(
        best=best,
        validation_gate_evaluated_count=(
            1 if validation_gate_df is not None else 0
        ),
        final_gate_passed_count=0,
    )


def replay_independent_validation_gate(
    best: dict[str, Any],
    gate_df: pd.DataFrame,
    bundle: ModelBundle,
    gate_direction_prob: dict[str, np.ndarray],
    *,
    gate_cutoff: float,
    label_min_return: float,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    long_preflight_gate: dict[str, Any],
    short_preflight_gate: dict[str, Any],
) -> StrategyCalibrationFinalization:
    """Replay the selected candidate on the independent validation gate."""

    config = BacktestConfig(**best["backtest_config"])
    gate_result, _ = run_backtest(gate_df, bundle, config)
    effective_long, effective_short = effective_thresholds(config)
    short_signal_threshold = max(
        1.0 - float(effective_short),
        0.5 + float(config.min_confidence_gap),
    )
    uses_directional = bool(
        np.asarray(
            gate_direction_prob.get("uses_directional_models", [])
        ).any()
    )
    capture_long_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_long",
            effective_long,
        )
        if uses_directional
        else effective_long
    )
    capture_short_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_short",
            short_signal_threshold,
        )
        if uses_directional
        else short_signal_threshold
    )
    capture_trade_threshold = (
        auxiliary_signal_threshold(
            bundle,
            "direction_trade",
            config.trade_signal_threshold,
        )
        if uses_directional
        else config.trade_signal_threshold
    )
    capture_prob = {
        **gate_direction_prob,
        "trade_threshold": np.full(
            len(gate_df),
            float(capture_trade_threshold),
            dtype=float,
        ),
        "long_threshold": np.full(
            len(gate_df),
            float(capture_long_threshold),
            dtype=float,
        ),
        "short_signal_threshold": np.full(
            len(gate_df),
            float(capture_short_threshold),
            dtype=float,
        ),
    }
    capture = directional_signal_capture_metrics(
        gate_df,
        capture_prob,
        long_threshold=capture_long_threshold,
        short_threshold=effective_short,
        cutoff=gate_cutoff,
        neutral_cutoff=label_min_return,
    )
    gate_penalty, filter_reasons = trading_filter_penalty(
        gate_result,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
    )
    archetype_gate = archetype_side_robustness(
        gate_result,
        min_trades=min_trades,
    )
    side_gate = side_contribution_gate(
        gate_result,
        side_policy=str(
            best.get("side_policy", config.trade_side_policy)
        ),
        min_trades=min_trades,
        min_profit_factor=min_profit_factor,
        long_preflight_gate=long_preflight_gate,
        short_preflight_gate=short_preflight_gate,
    )
    hard_gate = validation_trading_gate(
        gate_result,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
        archetype_gate=archetype_gate,
        min_total_return=0.0020,
        max_fee_drag_to_abs_return=0.45,
        min_expectancy_per_trade_after_cost=0.00004,
    )
    if not bool(side_gate.get("passed", False)):
        hard_gate = {
            **hard_gate,
            "passed": False,
            "reasons": (
                list(hard_gate.get("reasons", []))
                + list(side_gate.get("reasons", []))
            ),
        }
    calibration_gate = best.get(
        "calibration_validation_trading_gate",
        {},
    )
    hard_gate = {
        **hard_gate,
        "decision_dataset": "validation_gate",
        "calibration_gate_passed": bool(
            calibration_gate.get("passed", False)
        ),
        "calibration_gate_reasons": list(
            calibration_gate.get("reasons", [])
        ),
    }
    finalized = {
        **best,
        "backtest": asdict(gate_result),
        "gate_backtest": asdict(gate_result),
        "large_move": capture,
        "gate_large_move": capture,
        "gate_penalty": gate_penalty,
        "gate_archetype_side_gate": archetype_gate,
        "gate_side_contribution_gate": side_gate,
        "validation_trading_gate": hard_gate,
        "filter_reasons": filter_reasons,
    }
    return StrategyCalibrationFinalization(
        best=finalized,
        validation_gate_evaluated_count=1,
        final_gate_passed_count=(
            1 if hard_gate.get("passed") else 0
        ),
    )


def finalize_strategy_calibration_selection(
    best: dict[str, Any] | None,
    valid_df: pd.DataFrame,
    validation_gate_df: pd.DataFrame | None,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    gate_direction_prob: dict[str, np.ndarray] | None,
    *,
    gate_cutoff: float,
    label_min_return: float,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    long_preflight_gate: dict[str, Any],
    short_preflight_gate: dict[str, Any],
    calibration_gate_passed_count: int,
) -> StrategyCalibrationFinalization:
    """Finalize no-trade fallback or independent validation-gate replay."""

    if best is None:
        return build_no_trade_calibration_fallback(
            valid_df,
            validation_gate_df,
            bundle,
            base_cfg,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
        )
    if (
        validation_gate_df is not None
        and gate_direction_prob is not None
    ):
        return replay_independent_validation_gate(
            best,
            validation_gate_df,
            bundle,
            gate_direction_prob,
            gate_cutoff=gate_cutoff,
            label_min_return=label_min_return,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
            long_preflight_gate=long_preflight_gate,
            short_preflight_gate=short_preflight_gate,
        )
    return StrategyCalibrationFinalization(
        best=best,
        validation_gate_evaluated_count=0,
        final_gate_passed_count=int(
            calibration_gate_passed_count
        ),
    )


def calibrate_directional_thresholds(
    valid_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    *,
    label_min_return: float,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
    deadline: float | None = None,
    force_compact: bool = False,
    validation_gate_df: pd.DataFrame | None = None,
    long_preflight_gate: dict[str, Any] | None = None,
    short_preflight_gate: dict[str, Any] | None = None,
    max_threshold_evals: int | None = None,
) -> dict[str, Any]:
    x_valid, _, _ = feature_matrix(valid_df)
    direction_prob = bundle.predict_direction_probabilities(x_valid)
    cutoff = large_move_cutoff(valid_df, label_min_return)
    gate_df = validation_gate_df if validation_gate_df is not None and len(validation_gate_df) > 0 else None
    gate_direction_prob: dict[str, np.ndarray] | None = None
    gate_cutoff = 0.0
    if gate_df is not None:
        x_gate, _, _ = feature_matrix(gate_df)
        gate_direction_prob = bundle.predict_direction_probabilities(x_gate)
        gate_cutoff = large_move_cutoff(gate_df, label_min_return)
    best: dict[str, Any] | None = None
    ranking: list[dict[str, Any]] = []
    timed_out = False
    compact_search = bool(force_compact or (deadline is not None and (deadline - time.monotonic()) < 20.0 * 60.0))
    risk_profile_catalog = resolve_risk_profile_catalog(
        base_cfg,
        compact=compact_search,
    )
    risk_profiles = list(risk_profile_catalog.profiles)
    search_space = build_strategy_calibration_search_space(
        base_cfg,
        compact=compact_search,
    )
    threshold_pairs = list(search_space.threshold_pairs)
    trade_thresholds = list(search_space.trade_thresholds)
    requested_side_policies = list(search_space.side_policies)
    long_gate = long_preflight_gate or {"passed": False, "reasons": ["long_preflight_not_available"]}
    short_gate = short_preflight_gate or {"passed": False, "reasons": ["short_preflight_not_available"]}
    side_policies = requested_side_policies
    remaining_seconds = float(deadline - time.monotonic()) if deadline is not None else 999999.0
    eval_limit = int(max_threshold_evals or 0)
    eval_limit = eval_limit if eval_limit > 0 else None
    eval_limit_hit = False
    if compact_search:
        if remaining_seconds < 120.0:
            risk_profiles = risk_profiles[:2]
            threshold_pairs = threshold_pairs[:2]
            trade_thresholds = trade_thresholds[:2]
            side_policies = [policy for policy in side_policies if policy in {"long_only", "short_only"}][:2]
        elif remaining_seconds < 300.0:
            risk_profiles = risk_profiles[:3]
            threshold_pairs = threshold_pairs[:3]
            trade_thresholds = trade_thresholds[:2]
            side_policies = [policy for policy in side_policies if policy in {"long_only", "short_only", "both"}][:3]
    for candidate in iter_strategy_calibration_candidates(
        risk_profiles,
        threshold_pairs,
        trade_thresholds,
        side_policies,
    ):
        if eval_limit is not None and len(ranking) >= eval_limit:
            eval_limit_hit = True
            break
        if (
            ranking
            and deadline is not None
            and time.monotonic() >= deadline
        ):
            timed_out = True
            break
        evaluation = evaluate_strategy_calibration_candidate(
            valid_df,
            bundle,
            base_cfg,
            candidate,
            direction_prob,
            cutoff=cutoff,
            label_min_return=label_min_return,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
            long_preflight_gate=long_gate,
            short_preflight_gate=short_gate,
        )
        entry = evaluation.to_report()
        ranking.append(entry)
        if evaluation.passed and (
            best is None or entry["score"] > best["score"]
        ):
            best = entry
    ranking.sort(key=lambda item: (bool(item.get("validation_trading_gate", {}).get("passed")), item["score"]), reverse=True)
    calibration_gate_passed_count = int(
        sum(1 for item in ranking if item.get("calibration_validation_trading_gate", {}).get("passed"))
    )
    finalization = finalize_strategy_calibration_selection(
        best,
        valid_df,
        gate_df,
        bundle,
        base_cfg,
        gate_direction_prob,
        gate_cutoff=gate_cutoff,
        label_min_return=label_min_return,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
        long_preflight_gate=long_gate,
        short_preflight_gate=short_gate,
        calibration_gate_passed_count=(
            calibration_gate_passed_count
        ),
    )
    best = finalization.best
    validation_gate_evaluated_count = (
        finalization.validation_gate_evaluated_count
    )
    final_gate_passed_count = (
        finalization.final_gate_passed_count
    )
    return {
        "strategy_calibration_contract_version": "2026-06-20-v1",
        "strategy_calibration_engine_contract_version": (
            "2026-06-20-typed-v1"
        ),
        "risk_profile_catalog_schema_version": (
            risk_profile_catalog.schema_version
        ),
        "risk_profile_catalog_version": (
            risk_profile_catalog.catalog_version
        ),
        "risk_profile_catalog_path": str(
            risk_profile_catalog.source_path
        ),
        "best": best or {},
        "ranking": ranking[:20],
        "searched": len(ranking),
        "timed_out": timed_out,
        "max_threshold_evals": eval_limit,
        "threshold_eval_limit_hit": eval_limit_hit,
        "deadline_applied": deadline is not None,
        "selection_dataset": "validation_calibration" if gate_df is not None else "validation_full",
        "gate_dataset": "validation_gate" if gate_df is not None else "same_as_selection",
        "test_used_for_selection": False,
        "calibration_rows": int(len(valid_df)),
        "gate_rows": int(len(gate_df)) if gate_df is not None else 0,
        "separate_gate_enabled": bool(gate_df is not None),
        "calibration_gate_passed_count": calibration_gate_passed_count,
        "validation_gate_evaluated_count": validation_gate_evaluated_count,
        "hard_gate_policy": "validation thresholds and side policy must pass return, profit factor, drawdown, trade count, realized risk reward, fee quality, and archetype side gates",
        "compact_search": compact_search,
        "risk_profiles_searched": [item["name"] for item in risk_profiles],
        "threshold_pairs_searched": threshold_pairs,
        "trade_thresholds_searched": trade_thresholds,
        "side_policies_requested": requested_side_policies,
        "side_policies_searched": side_policies,
        "long_side_preflight_gate": long_gate,
        "long_candidate_allowed": bool(long_gate.get("passed", False)),
        "long_candidate_blockers": list(long_gate.get("reasons", [])),
        "short_side_preflight_gate": short_gate,
        "short_candidate_allowed": bool(short_gate.get("passed", False)),
        "short_candidate_blockers": list(short_gate.get("reasons", [])),
        "valid_gate_passed_count": int(final_gate_passed_count),
    }
