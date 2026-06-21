from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os
import statistics
import time

from .config import load_config
from .time_utils import add_beijing_aliases, beijing_now_iso, beijing_stamp, to_beijing_iso


VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
MEMORY_VERSION = "1.0"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        add_beijing_aliases(payload)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    last_error: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
    try:
        tmp.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def avg(values: list[float], default: float = 0.0) -> float:
    filtered = [item for item in values if math.isfinite(item)]
    return float(sum(filtered) / len(filtered)) if filtered else default


def median(values: list[float], default: float = 0.0) -> float:
    filtered = [item for item in values if math.isfinite(item)]
    return float(statistics.median(filtered)) if filtered else default


def file_time_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return beijing_now_iso()


def infer_symbol_interval(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_")
    symbol = ""
    interval = ""
    for idx, part in enumerate(parts):
        upper = part.upper()
        if upper.endswith("USDT") and not symbol:
            symbol = upper
            if idx + 1 < len(parts) and parts[idx + 1] in VALID_INTERVALS:
                interval = parts[idx + 1]
            break
    return symbol, interval


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "auc",
        "log_loss",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_auc",
        "test_log_loss",
    ]
    return {key: as_float(metrics.get(key)) for key in keys if key in metrics}


def extract_metrics(item: dict[str, Any]) -> dict[str, float]:
    summary = as_dict(item.get("summary"))
    if summary:
        payload = {}
        if "mean_balanced_accuracy" in summary:
            payload["balanced_accuracy"] = as_float(summary.get("mean_balanced_accuracy"))
        if "profitable_fold_rate" in summary:
            payload["profitable_fold_rate"] = as_float(summary.get("profitable_fold_rate"))
        if "acceptable_profit_factor_rate" in summary:
            payload["acceptable_profit_factor_rate"] = as_float(summary.get("acceptable_profit_factor_rate"))
        if payload:
            return payload
    metrics = as_dict(item.get("model_metrics")) or as_dict(item.get("metrics"))
    if metrics:
        return compact_metrics(metrics)
    report = as_dict(item.get("model_report"))
    ranking = as_list(report.get("ranking"))
    if ranking and isinstance(ranking[0], dict):
        return compact_metrics(as_dict(ranking[0].get("metrics")))
    return {}


def compact_backtest(backtest: dict[str, Any]) -> dict[str, float]:
    keys = [
        "final_balance",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "trades",
        "win_rate",
        "profit_factor",
        "sharpe_like",
        "sortino_ratio",
        "calmar_ratio",
        "fee_ratio",
        "gross_return_before_cost",
        "total_cost_drag",
        "notional_turnover",
        "avg_exposure",
        "average_exposure",
        "execution_events",
        "duration_days",
    ]
    payload: dict[str, float] = {}
    for key in keys:
        if key in backtest:
            payload[key] = as_float(backtest.get(key))
    return payload


def extract_backtest(item: dict[str, Any]) -> dict[str, float]:
    summary = as_dict(item.get("summary"))
    if summary:
        return {
            "total_return": as_float(summary.get("median_return")),
            "max_drawdown": as_float(summary.get("worst_drawdown")),
            "profit_factor": as_float(summary.get("mean_profit_factor")),
            "trades": as_float(summary.get("mean_trades"), float(as_int(summary.get("folds_completed", item.get("folds_completed", 0))))),
            "win_rate": as_float(summary.get("profitable_fold_rate")),
            "fee_drag": as_float(summary.get("mean_fee_drag")),
        }
    direct = as_dict(item.get("backtest"))
    if direct:
        return compact_backtest(direct)
    paper_metrics = as_dict(item.get("metrics"))
    if "total_return" in paper_metrics:
        return compact_backtest(paper_metrics)

    threshold = as_dict(item.get("threshold_optimization"))
    best_threshold = as_dict(threshold.get("best"))
    if as_dict(best_threshold.get("backtest")):
        return compact_backtest(as_dict(best_threshold.get("backtest")))

    parameter = as_dict(item.get("parameter_optimization"))
    best_parameter = as_dict(parameter.get("best"))
    if as_dict(best_parameter.get("test")):
        return compact_backtest(as_dict(best_parameter.get("test")))
    if as_dict(best_parameter.get("backtest")):
        return compact_backtest(as_dict(best_parameter.get("backtest")))

    if "total_return" in item or "max_drawdown" in item or "profit_factor" in item:
        return compact_backtest(item)

    state = as_dict(item.get("state"))
    if state:
        equity = as_float(state.get("equity"), as_float(state.get("balance"), 10_000.0))
        initial = as_float(item.get("initial_balance"), 10_000.0)
        total_return = (equity - initial) / initial if initial else 0.0
        return {
            "final_balance": equity,
            "total_return": total_return,
            "max_drawdown": 0.0,
            "trades": float(as_int(state.get("trades"))),
            "profit_factor": 0.0,
            "win_rate": 0.0,
        }
    return {}


def extract_thresholds(item: dict[str, Any]) -> dict[str, float]:
    cfg = as_dict(item.get("optimized_backtest_config"))
    threshold = as_dict(item.get("threshold_optimization"))
    best = as_dict(threshold.get("best"))
    payload: dict[str, float] = {}
    for source in (cfg, best):
        for key in ["long_threshold", "short_threshold", "leverage", "stop_loss", "take_profit", "max_position_fraction"]:
            if key in source and key not in payload:
                payload[key] = as_float(source.get(key))
    folds = as_list(item.get("folds"))
    if folds and isinstance(folds[0], dict):
        fold_cfg = as_dict(folds[0].get("optimized_backtest_config"))
        for key in ["long_threshold", "short_threshold", "leverage", "stop_loss", "take_profit", "max_position_fraction"]:
            if key in fold_cfg and key not in payload:
                payload[key] = as_float(fold_cfg.get(key))
    return payload


def extract_strategy_flags(item: dict[str, Any]) -> dict[str, Any]:
    cfg = as_dict(item.get("optimized_backtest_config"))
    threshold = as_dict(item.get("threshold_optimization"))
    best = as_dict(threshold.get("best"))
    report = as_dict(item.get("model_report"))
    model_gate = as_dict(report.get("model_selection_gate"))
    publish = as_dict(item.get("model_publish"))
    side_policy = clean_text(
        cfg.get("trade_side_policy")
        or best.get("side_policy")
        or best.get("trade_side_policy")
        or "both",
        "both",
    )
    selected_risk_profile = clean_text(
        as_dict(item.get("small_account_strategy")).get("selected_risk_profile")
        or best.get("risk_profile")
        or "",
    )
    return {
        "trade_side_policy": side_policy,
        "selected_risk_profile": selected_risk_profile,
        "no_trade_recommended": bool(best.get("fallback_no_trade_recommended") or selected_risk_profile == "no_trade_recommended"),
        "rejected_by_validation_trading_gate": bool(model_gate.get("rejected_by_validation_trading_gate", False)),
        "model_publish_status": clean_text(publish.get("status")),
        "model_publish_reason": clean_text(publish.get("reason")),
        "valid_gate_passed_count": as_int(threshold.get("valid_gate_passed_count")),
    }


def extract_large_move(item: dict[str, Any]) -> dict[str, float]:
    threshold = as_dict(item.get("threshold_optimization"))
    best = as_dict(threshold.get("best"))
    large = as_dict(best.get("large_move"))
    if large:
        return {key: as_float(value) for key, value in large.items()}
    report = as_dict(item.get("model_report"))
    ranking = as_list(report.get("ranking"))
    if ranking and isinstance(ranking[0], dict):
        large = as_dict(ranking[0].get("large_move_metrics"))
        return {key: as_float(value) for key, value in large.items()}
    return {}


def observation_score(metrics: dict[str, float], backtest: dict[str, float], large_move: dict[str, float]) -> float:
    total_return = as_float(backtest.get("total_return"))
    max_drawdown = as_float(backtest.get("max_drawdown"))
    profit_factor = as_float(backtest.get("profit_factor"), 1.0)
    trades = as_float(backtest.get("trades"))
    balanced = as_float(metrics.get("test_balanced_accuracy", metrics.get("balanced_accuracy", 0.5)), 0.5)
    auc = as_float(metrics.get("test_auc", metrics.get("auc", 0.5)), 0.5)
    large_capture = as_float(large_move.get("large_move_capture"), 0.0)
    low_trade_penalty = 0.04 if trades and trades < 8 else 0.0
    return float(
        total_return
        + max_drawdown * 0.45
        + max(0.0, profit_factor - 1.0) * 0.025
        + (balanced - 0.5) * 0.06
        + (auc - 0.5) * 0.03
        + large_capture * 0.015
        - low_trade_penalty
    )


def extract_selection_payload(
    source: str,
    item: dict[str, Any],
    metrics: dict[str, float],
    backtest: dict[str, float],
    large_move: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], str]:
    if source in {"model_optimization", "model_optimization_batch", "ai_optimization"}:
        report = as_dict(item.get("model_report"))
        ranking = as_list(report.get("ranking"))
        if ranking and isinstance(ranking[0], dict):
            top = ranking[0]
            selection_metrics = compact_metrics(as_dict(top.get("metrics")))
            selection_backtest = compact_backtest(as_dict(top.get("valid_backtest")))
            selection_large = as_dict(top.get("large_move_metrics"))
            return selection_metrics, selection_backtest, {key: as_float(value) for key, value in selection_large.items()}, "validation_ranking"
        return {}, {}, {}, "test_metrics_excluded_no_validation_payload"
    if source == "historical_backtest":
        return {}, {}, large_move, "historical_test_excluded"
    return metrics, backtest, large_move, "walk_forward_or_paper_observation"


def make_observation(source: str, path: Path, item: dict[str, Any], index: int = 0) -> dict[str, Any] | None:
    inferred_symbol, inferred_interval = infer_symbol_interval(path)
    symbol = clean_text(item.get("symbol"), inferred_symbol).upper()
    interval = clean_text(item.get("interval"), inferred_interval)
    if not symbol or not interval:
        return None

    metrics = extract_metrics(item)
    backtest = extract_backtest(item)
    thresholds = extract_thresholds(item)
    strategy_flags = extract_strategy_flags(item)
    large_move = extract_large_move(item)
    selection_metrics, selection_backtest, selection_large_move, selection_basis = extract_selection_payload(
        source,
        item,
        metrics,
        backtest,
        large_move,
    )
    created = clean_text(item.get("created_beijing") or item.get("created_utc"), file_time_iso(path))
    created_beijing = to_beijing_iso(created) or created
    model_name = clean_text(item.get("model_name") or item.get("model"), "strategy_validation" if source.startswith("strategy_validation") else "unknown")
    summary = as_dict(item.get("summary"))
    research_candidate = as_dict(summary.get("research_candidate"))
    research_status = clean_text(
        summary.get("research_status") or research_candidate.get("research_status"),
        "none",
    )
    research_usage = clean_text(
        summary.get("research_usage") or summary.get("research_candidate_usage") or research_candidate.get("usage"),
        "offline_diagnostics_only",
    )
    research_blockers = as_list(summary.get("research_blockers")) or as_list(research_candidate.get("blockers"))
    research_reasons = as_list(summary.get("research_reasons")) or as_list(research_candidate.get("reasons"))
    score = observation_score(selection_metrics, selection_backtest, selection_large_move)
    if strategy_flags.get("rejected_by_validation_trading_gate") or strategy_flags.get("no_trade_recommended"):
        score = min(score, -0.05)
    hash_input = json.dumps(
        {
            "source": source,
            "file": path.name,
            "index": index,
            "created": created_beijing,
            "symbol": symbol,
            "interval": interval,
            "model": model_name,
            "return": backtest.get("total_return"),
            "trades": backtest.get("trades"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    observation_id = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:16]
    return {
        "id": observation_id,
        "source": source,
        "report_file": path.name,
        "report_path": str(path),
        "created_beijing": created_beijing,
        "symbol": symbol,
        "interval": interval,
        "target": f"{symbol}:{interval}",
        "model_name": model_name,
        "rows": as_int(item.get("rows", item.get("raw_rows", 0))),
        "latest_up_probability": item.get("latest_up_probability"),
        "metrics": metrics,
        "backtest": backtest,
        "thresholds": thresholds,
        "strategy_flags": strategy_flags,
        "large_move": large_move,
        "eligible_for_paper_candidate": bool(summary.get("eligible_for_paper_candidate", item.get("eligible_for_paper_candidate", False))),
        "research_candidate_eligible": research_status == "eligible",
        "research_status": research_status,
        "research_usage": research_usage,
        "research_watchlist_allowed": bool(summary.get("research_watchlist_allowed") or research_candidate.get("watchlist_allowed", False)),
        "research_promotion_allowed": False,
        "research_shadow_allowed": False,
        "research_blockers": research_blockers,
        "research_reasons": research_reasons,
        "research_candidate": research_candidate,
        "recommendation": clean_text(summary.get("recommendation", item.get("recommendation", ""))),
        "folds_completed": as_int(item.get("folds_completed")),
        "latest_walk_forward_summary": {
            "eligible_for_paper_candidate": bool(summary.get("eligible_for_paper_candidate", False)),
            "candidate_tier": clean_text(summary.get("candidate_tier", "")),
            "recommendation": clean_text(summary.get("recommendation", "")),
            "research_candidate_eligible": research_status == "eligible",
            "research_status": research_status,
            "research_usage": research_usage,
            "research_candidate_reason": clean_text(summary.get("research_candidate_reason", "")),
            "research_watchlist_allowed": bool(summary.get("research_watchlist_allowed") or research_candidate.get("watchlist_allowed", False)),
            "research_blockers": research_blockers,
            "research_reasons": research_reasons,
            "promotion_blockers": as_list(summary.get("promotion_blockers")),
            "high_slippage_pass_rate": as_float(summary.get("high_slippage_pass_rate")),
            "volatility_regime_pass_rate": as_float(summary.get("volatility_regime_pass_rate")),
            "model_selection_pass_rate": as_float(summary.get("model_selection_pass_rate")),
            "min_trades": as_float(summary.get("min_trades")),
        }
        if source.startswith("strategy_validation")
        else {},
        "selection_metrics": selection_metrics,
        "selection_backtest": selection_backtest,
        "selection_large_move": selection_large_move,
        "selection_basis": selection_basis,
        "selection_score": score,
        "score": score,
    }


def collect_from_report(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return []

    name = path.name
    observations: list[dict[str, Any]] = []
    if name.startswith("runner_live_"):
        for idx, item in enumerate(as_list(payload.get("items"))):
            if isinstance(item, dict):
                obs = make_observation("runner_live", path, item, idx)
                if obs:
                    observations.append(obs)
        return observations

    if name.startswith("live_training_"):
        for run_idx, run in enumerate(as_list(payload.get("runs"))):
            for item_idx, item in enumerate(as_list(as_dict(run).get("items"))):
                if isinstance(item, dict):
                    obs = make_observation("live_training", path, item, run_idx * 1000 + item_idx)
                    if obs:
                        observations.append(obs)
        return observations

    if name.startswith("model_optimization_"):
        for idx, item in enumerate(as_list(payload.get("items")) or as_list(payload.get("ranked"))):
            if isinstance(item, dict):
                obs = make_observation("model_optimization_batch", path, item, idx)
                if obs:
                    observations.append(obs)
        return observations

    if name.startswith("strategy_validation_"):
        for idx, item in enumerate(as_list(payload.get("items")) or as_list(payload.get("ranked"))):
            if isinstance(item, dict):
                obs = make_observation("strategy_validation_batch", path, item, idx)
                if obs:
                    observations.append(obs)
        if observations:
            return observations

    source = "report"
    if name.endswith("_model_optimization.json"):
        source = "model_optimization"
    elif name.endswith("_strategy_validation.json"):
        source = "strategy_validation"
    elif name.endswith("_runner_live_train_metrics.json") or name.endswith("_live_train_metrics.json"):
        source = "live_train_metrics"
    elif name.endswith("_ai_optimization.json"):
        source = "ai_optimization"
    elif name.endswith("_backtest_summary.json"):
        source = "historical_backtest"
    elif name.endswith("_paper_summary.json"):
        source = "paper_replay"
    else:
        return []

    obs = make_observation(source, path, payload)
    return [obs] if obs else []


def collect_observations(reports_dir: Path) -> list[dict[str, Any]]:
    patterns = [
        "runner_live_*.json",
        "live_training_*.json",
        "model_optimization_*.json",
        "*_model_optimization.json",
        "strategy_validation_*.json",
        "*_strategy_validation.json",
        "*_runner_live_train_metrics.json",
        "*_live_train_metrics.json",
        "*_ai_optimization.json",
        "*_backtest_summary.json",
        "*_paper_summary.json",
    ]
    seen_files: set[Path] = set()
    observations: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(reports_dir.glob(pattern), key=lambda item: item.stat().st_mtime):
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            observations.extend(collect_from_report(path))
    return observations


def summarize_target(target: str, observations: list[dict[str, Any]], max_items: int) -> dict[str, Any]:
    ordered = sorted(observations, key=lambda item: clean_text(item.get("created_beijing")))
    kept = ordered[-max_items:]
    recent = kept[-20:]
    returns = [as_float(as_dict(item.get("selection_backtest")).get("total_return")) for item in recent]
    drawdowns = [as_float(as_dict(item.get("selection_backtest")).get("max_drawdown")) for item in recent]
    profit_factors = [as_float(as_dict(item.get("selection_backtest")).get("profit_factor"), 1.0) for item in recent]
    trades = [as_float(as_dict(item.get("selection_backtest")).get("trades")) for item in recent]
    scores = [as_float(item.get("selection_score", item.get("score"))) for item in recent]
    balanced = [
        as_float(as_dict(item.get("selection_metrics")).get("balanced_accuracy", 0.5), 0.5)
        for item in recent
    ]
    latest = kept[-1] if kept else {}
    best = max(kept, key=lambda item: as_float(item.get("selection_score", item.get("score"))), default={})
    recent_positive_rate = sum(1 for item in returns if item > 0) / len(returns) if returns else 0.0
    avg_return = avg(returns)
    avg_pf = avg(profit_factors, 1.0)
    worst_drawdown = min(drawdowns) if drawdowns else 0.0
    avg_trades = avg(trades)
    avg_balanced = avg(balanced, 0.5)
    recent_score = avg(scores)
    previous_scores = [as_float(item.get("selection_score", item.get("score"))) for item in kept[-40:-20]]
    trend_delta = recent_score - avg(previous_scores) if previous_scores else 0.0
    stability_score = max(0.0, min(1.0, 0.35 * recent_positive_rate + 0.25 * max(0.0, avg_pf - 0.8) + 0.25 * max(0.0, avg_balanced - 0.45) + 0.15 * max(0.0, avg_return * 20)))
    def walk_forward_observation_passes(item: dict[str, Any]) -> bool:
        selection_backtest = as_dict(item.get("selection_backtest"))
        selection_metrics = as_dict(item.get("selection_metrics"))
        return bool(
            clean_text(item.get("source")).startswith("strategy_validation")
            and item.get("eligible_for_paper_candidate")
            and as_float(as_dict(item.get("latest_walk_forward_summary")).get("model_selection_pass_rate")) >= 0.50
            and as_float(selection_backtest.get("total_return")) > 0.0
            and as_float(selection_backtest.get("profit_factor"), 1.0) >= 1.0
            and as_float(selection_backtest.get("trades")) >= 8
            and as_float(selection_backtest.get("max_drawdown")) >= -0.15
            and as_float(selection_metrics.get("balanced_accuracy"), 0.5) >= 0.52
        )

    latest_walk_forward = next(
        (
            item
            for item in reversed(kept)
            if clean_text(item.get("source")).startswith("strategy_validation")
        ),
        {},
    )
    latest_walk_forward_blocks_paper = bool(
        latest_walk_forward
        and not bool(latest_walk_forward.get("eligible_for_paper_candidate"))
    )
    latest_walk_forward_eligible = bool(
        latest_walk_forward
        and walk_forward_observation_passes(latest_walk_forward)
    )
    latest_walk_forward_research_status = clean_text(latest_walk_forward.get("research_status"), "none") if latest_walk_forward else "none"
    latest_walk_forward_research = bool(
        latest_walk_forward
        and latest_walk_forward_research_status == "eligible"
        and not latest_walk_forward.get("eligible_for_paper_candidate")
    )
    latest_walk_forward_research_watch = bool(
        latest_walk_forward
        and latest_walk_forward_research_status == "watch"
        and not latest_walk_forward.get("eligible_for_paper_candidate")
    )
    latest_walk_forward_research_blocked = bool(
        latest_walk_forward
        and latest_walk_forward_research_status == "blocked"
        and not latest_walk_forward.get("eligible_for_paper_candidate")
    )
    if latest_walk_forward_blocks_paper:
        if latest_walk_forward_research:
            stability_score = min(stability_score, 0.35)
        elif latest_walk_forward_research_watch:
            stability_score = min(stability_score, 0.30)
        else:
            stability_score = min(stability_score, 0.20 if latest_walk_forward_research_blocked else 0.25)
    aggregate_eligible_for_paper = (
        len(recent) >= 2
        and avg_return > 0
        and recent_positive_rate >= 0.5
        and avg_pf >= 1.0
        and worst_drawdown >= -0.15
        and avg_trades >= 8
    )
    eligible_for_paper = bool(
        not latest_walk_forward_blocks_paper
        and (aggregate_eligible_for_paper or latest_walk_forward_eligible)
    )

    if eligible_for_paper:
        next_action = "continue_paper_simulation"
        if latest_walk_forward_eligible and not aggregate_eligible_for_paper:
            reason = "Latest walk-forward validation passed; continue paper observation while older memory remains under review."
        else:
            reason = "Recent memory is positive enough for continued paper/testnet observation only."
    elif latest_walk_forward_blocks_paper and (latest_walk_forward_research or latest_walk_forward_research_watch):
        next_action = "research_optimize_cost_edge"
        research = as_dict(latest_walk_forward.get("research_candidate"))
        reason = (
            "Latest walk-forward is research-only and can feed offline cost-edge optimization, but promotion gates still block paper/testnet. "
            f"Research notes: {', '.join(as_list(research.get('reasons'))[:4]) or 'needs fold stability'}."
        )
    elif latest_walk_forward_blocks_paper and latest_walk_forward_research_blocked:
        next_action = "feature_and_label_optimization"
        research = as_dict(latest_walk_forward.get("research_candidate"))
        blockers = as_list(latest_walk_forward.get("research_blockers")) or as_list(research.get("blockers"))
        reason = (
            "Latest walk-forward research status is blocked; keep it as offline diagnostics and improve features, labels, "
            f"cost filters, and short-side rules before more paper simulation. Blockers: {', '.join(blockers[:4]) or 'not specified'}."
        )
    elif latest_walk_forward_blocks_paper:
        next_action = "observe"
        latest_summary = as_dict(latest_walk_forward.get("latest_walk_forward_summary"))
        recommendation = clean_text(latest_summary.get("recommendation"), "latest walk-forward validation did not pass")
        reason = f"Latest walk-forward validation blocks paper/testnet promotion: {recommendation}."
    elif worst_drawdown < -0.15:
        next_action = "reduce_risk_and_recalibrate"
        reason = "Worst recent drawdown is too large; lower leverage/position fraction before more simulation."
    elif avg_pf < 1.0 and avg_trades >= 8:
        next_action = "raise_confidence_thresholds"
        reason = "Trades are frequent but profit factor is weak; reduce noisy entries."
    elif avg_balanced < 0.52:
        next_action = "feature_and_label_optimization"
        reason = "Direction quality is weak; prioritize feature/label changes over leverage changes."
    elif avg_trades < 8:
        next_action = "collect_more_closed_klines"
        reason = "Too few trades in memory; keep observing before trusting the result."
    else:
        next_action = "observe"
        reason = "No strong promotion or rejection signal yet."

    latest_prob = latest.get("latest_up_probability")
    long_threshold = as_float(as_dict(latest.get("thresholds")).get("long_threshold"), 0.57)
    short_threshold = as_float(as_dict(latest.get("thresholds")).get("short_threshold"), 0.43)
    latest_direction = "wait"
    if latest_prob is not None:
        prob = as_float(latest_prob)
        if prob >= long_threshold:
            latest_direction = "long_candidate"
        elif prob <= short_threshold:
            latest_direction = "short_candidate"

    symbol, _, interval = target.partition(":")
    return {
        "symbol": symbol,
        "interval": interval,
        "target": target,
        "observation_count": len(kept),
        "latest": latest,
        "best": best,
        "summary": {
            "avg_return": avg_return,
            "median_return": median(returns),
            "positive_rate": recent_positive_rate,
            "worst_drawdown": worst_drawdown,
            "avg_profit_factor": avg_pf,
            "avg_trades": avg_trades,
            "avg_balanced_accuracy": avg_balanced,
            "recent_score": recent_score,
            "trend_delta": trend_delta,
            "stability_score": stability_score,
            "eligible_for_paper": eligible_for_paper,
            "latest_walk_forward_eligible": latest_walk_forward_eligible,
            "latest_walk_forward_blocks_paper": latest_walk_forward_blocks_paper,
            "latest_walk_forward_research_candidate": latest_walk_forward_research,
            "latest_walk_forward_research_status": latest_walk_forward_research_status,
            "latest_walk_forward_research_watch": latest_walk_forward_research_watch,
            "latest_walk_forward_research_blocked": latest_walk_forward_research_blocked,
            "latest_direction": latest_direction,
            "next_action": next_action,
            "reason": reason,
        },
        "observations": kept,
    }


def build_global_summary(targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summaries = list(targets.values())
    action_priority = {
        "continue_paper_simulation": 5,
        "observe": 3,
        "raise_confidence_thresholds": 2,
        "feature_and_label_optimization": 1,
        "research_optimize_cost_edge": 2,
        "collect_more_closed_klines": 0,
        "reduce_risk_and_recalibrate": -2,
    }
    ranked = sorted(
        summaries,
        key=lambda item: (
            bool(as_dict(item.get("summary")).get("eligible_for_paper")),
            action_priority.get(str(as_dict(item.get("summary")).get("next_action")), 0),
            as_float(as_dict(item.get("summary")).get("stability_score")),
            as_float(as_dict(item.get("summary")).get("recent_score")),
        ),
        reverse=True,
    )
    top_candidates = [
        {
            "target": item["target"],
            "symbol": item["symbol"],
            "interval": item["interval"],
            "eligible_for_paper": as_dict(item.get("summary")).get("eligible_for_paper", False),
            "research_candidate": as_dict(item.get("summary")).get("latest_walk_forward_research_candidate", False),
            "research_status": clean_text(as_dict(item.get("summary")).get("latest_walk_forward_research_status"), "none"),
            "research_watch": bool(as_dict(item.get("summary")).get("latest_walk_forward_research_watch", False)),
            "research_blocked": bool(as_dict(item.get("summary")).get("latest_walk_forward_research_blocked", False)),
            "stability_score": as_float(as_dict(item.get("summary")).get("stability_score")),
            "avg_return": as_float(as_dict(item.get("summary")).get("avg_return")),
            "worst_drawdown": as_float(as_dict(item.get("summary")).get("worst_drawdown")),
            "next_action": as_dict(item.get("summary")).get("next_action", "observe"),
            "best_model": as_dict(item.get("best")).get("model_name", "unknown"),
        }
        for item in ranked[:8]
    ]
    weak_targets = [
        item
        for item in top_candidates
        if item["next_action"] in {"research_optimize_cost_edge", "feature_and_label_optimization", "raise_confidence_thresholds", "reduce_risk_and_recalibrate"}
    ]
    research_candidates = [
        item
        for item in top_candidates
        if item["research_status"] in {"eligible", "watch"}
    ]
    research_blocked = [
        item
        for item in top_candidates
        if item["research_status"] == "blocked"
    ]
    next_actions = [
        "Update memory after every model-optimize/live-train/runner cycle before deciding the next simulation step.",
        "Keep live_trading_enabled=false; memory recommendations are paper/testnet-only.",
    ]
    if top_candidates:
        best = top_candidates[0]
        if best["eligible_for_paper"]:
            next_actions.append(f"Continue paper simulation for {best['target']} using the current best remembered model.")
        else:
            next_actions.append(f"Prioritize {best['target']} for observation, but keep it out of testnet until gates improve.")
    if weak_targets:
        next_actions.append("Run return-first/cost-edge optimization only on weak/watch targets; blocked targets need feature, label, cost, or side-policy repair first.")
    return {
        "top_candidates": top_candidates,
        "research_candidates": research_candidates,
        "research_blocked": research_blocked,
        "weak_targets": weak_targets,
        "next_actions": next_actions,
    }


def collect_monitoring_alerts(reports_dir: Path) -> list[dict[str, Any]]:
    """Collect active monitoring triggers separately from return observations."""

    alerts: list[dict[str, Any]] = []
    for path in reports_dir.glob("*_*_monitoring.json"):
        payload = read_json(path, {})
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            continue
        retraining = as_dict(payload.get("retraining"))
        if not retraining.get("active") or retraining.get("acknowledged"):
            continue
        symbol = clean_text(payload.get("symbol")).upper()
        interval = clean_text(payload.get("interval"))
        alerts.append(
            {
                "symbol": symbol,
                "interval": interval,
                "target": f"{symbol}:{interval}",
                "severity": clean_text(retraining.get("severity"), "medium"),
                "reasons": as_list(retraining.get("reasons")),
                "trigger_id": clean_text(retraining.get("trigger_id")),
                "valid_until_beijing": clean_text(
                    retraining.get("valid_until_beijing")
                ),
                "report_path": str(path),
            }
        )
    return sorted(
        alerts,
        key=lambda item: item["severity"] == "high",
        reverse=True,
    )


def update_simulation_memory(
    *,
    config_path: str | Path | None = None,
    state_dir: str | Path = "state",
    max_observations_per_target: int = 120,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.ensure_dirs()
    state_path = Path(state_dir)
    existing = read_json(state_path / "simulation_memory.json", {})
    observations = collect_observations(cfg.reports_dir)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        grouped.setdefault(clean_text(obs.get("target")), []).append(obs)

    targets = {
        target: summarize_target(target, items, max(20, max_observations_per_target))
        for target, items in grouped.items()
        if target
    }
    global_summary = build_global_summary(targets)
    global_summary["monitoring_alerts"] = collect_monitoring_alerts(
        cfg.reports_dir
    )
    payload = {
        "version": MEMORY_VERSION,
        "created_beijing": clean_text(existing.get("created_beijing")) or beijing_now_iso(),
        "updated_beijing": beijing_now_iso(),
        "update_count": as_int(existing.get("update_count")) + 1,
        "mode": "simulation_memory_only",
        "safety": {
            "live_trading_enabled": bool(cfg.live_trading_enabled),
            "real_orders_allowed": False,
            "api_keys_used": False,
        },
        "observation_count": sum(len(item.get("observations", [])) for item in targets.values()),
        "target_count": len(targets),
        "targets": targets,
        "global": global_summary,
    }
    state_output = state_path / "simulation_memory.json"
    latest_output = cfg.reports_dir / "simulation_memory_latest.json"
    stamped_output = cfg.reports_dir / f"simulation_memory_{beijing_stamp()}.json"
    write_json(state_output, payload)
    write_json(latest_output, payload)
    write_json(stamped_output, payload)
    payload["state_path"] = str(state_output)
    payload["report_path"] = str(latest_output)
    payload["stamped_report_path"] = str(stamped_output)
    return payload
