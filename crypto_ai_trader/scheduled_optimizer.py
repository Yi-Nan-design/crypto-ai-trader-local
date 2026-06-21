from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import os
import time

from .config import load_config
from .data_maintenance import run_data_maintenance
from .model_optimization import run_model_optimization
from .monitoring import MONITORING_ALGORITHM_VERSION, MONITORING_SCHEMA_VERSION
from .progress import safe_replace_text, tracker_for_reports
from .simulation_memory import update_simulation_memory
from .time_utils import beijing_now_iso, beijing_stamp


WEAK_ACTIONS = {
    "research_optimize_cost_edge",
    "raise_confidence_thresholds",
    "feature_and_label_optimization",
    "reduce_risk_and_recalibrate",
    "collect_more_closed_klines",
}
LOCK_STALE_SECONDS = 2 * 60 * 60


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def process_is_running(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
        return True
    except OSError:
        return False


def acquire_lock(path: Path) -> tuple[int | None, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("pid") and not process_is_running(existing.get("pid")):
            path.unlink(missing_ok=True)
        elif not existing.get("pid"):
            existing["lock_path"] = str(path)
            return None, existing
        else:
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                age = 0.0
            if age > LOCK_STALE_SECONDS:
                path.unlink(missing_ok=True)
            else:
                existing["lock_path"] = str(path)
                existing["lock_age_seconds"] = age
                return None, existing

    payload = {
        "pid": os.getpid(),
        "created_beijing": beijing_now_iso(),
        "lock_path": str(path),
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing["lock_path"] = str(path)
        return None, existing
    os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return fd, payload


def release_lock(fd: int | None, path: Path) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    path.unlink(missing_ok=True)


def normalize_targets(symbols: list[str] | None, intervals: list[str] | None) -> list[tuple[str, str]]:
    if not symbols or not intervals:
        return []
    return [(symbol.upper(), interval) for interval in intervals for symbol in symbols]


def monitoring_triggered_targets(
    reports_dir: Path,
    *,
    allowed_symbols: list[str] | None = None,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Return retraining-triggered targets ordered by report freshness."""

    allowed = {symbol.upper() for symbol in (allowed_symbols or [])}
    now = datetime.now().astimezone()
    rows: list[tuple[float, tuple[str, str], dict[str, Any]]] = []
    for path in reports_dir.glob("*_*_monitoring.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("schema_version") != MONITORING_SCHEMA_VERSION
            or payload.get("algorithm_version") != MONITORING_ALGORITHM_VERSION
        ):
            continue
        retraining = payload.get("retraining")
        if (
            not isinstance(retraining, dict)
            or not retraining.get("active")
            or retraining.get("acknowledged")
        ):
            continue
        symbol = str(payload.get("symbol") or "").upper()
        interval = str(payload.get("interval") or "")
        target = (symbol, interval)
        if not symbol or not interval or (allowed and symbol not in allowed):
            continue
        valid_until = str(retraining.get("valid_until_beijing") or "")
        try:
            valid_until_dt = datetime.fromisoformat(valid_until)
        except ValueError:
            continue
        if valid_until_dt.tzinfo is None:
            valid_until_dt = valid_until_dt.astimezone()
        if valid_until_dt < now:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        rows.append(
            (
                modified,
                target,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "severity": retraining.get("severity", "medium"),
                    "reasons": list(retraining.get("reasons") or []),
                    "trigger_id": retraining.get("trigger_id"),
                    "model_version": payload.get("model_version"),
                    "valid_until_beijing": valid_until,
                    "report_path": str(path),
                },
            )
        )
    rows.sort(key=lambda item: (item[2]["severity"] == "high", item[0]), reverse=True)
    targets: list[tuple[str, str]] = []
    details: list[dict[str, Any]] = []
    for _, target, detail in rows:
        if target in targets:
            continue
        targets.append(target)
        details.append(detail)
    return targets, details


def acknowledge_monitoring_triggers(
    reports_dir: Path,
    completed_targets: set[tuple[str, str]],
    *,
    optimization_report: str,
) -> list[dict[str, Any]]:
    """Acknowledge active triggers only after their target optimized successfully."""

    acknowledgements: list[dict[str, Any]] = []
    for symbol, interval in sorted(completed_targets):
        path = reports_dir / f"{symbol}_{interval}_monitoring.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        retraining = payload.get("retraining")
        if (
            not isinstance(retraining, dict)
            or not retraining.get("active")
            or retraining.get("acknowledged")
        ):
            continue
        acknowledged_at = beijing_now_iso()
        retraining.update(
            {
                "active": False,
                "acknowledged": True,
                "acknowledged_beijing": acknowledged_at,
                "acknowledged_by_report": optimization_report,
            }
        )
        safe_write_json(path, payload)
        acknowledgement = {
            "event": "monitoring_trigger_acknowledged",
            "created_beijing": acknowledged_at,
            "symbol": symbol,
            "interval": interval,
            "trigger_id": retraining.get("trigger_id"),
            "optimization_report": optimization_report,
            "live_trading_enabled": False,
        }
        history_path = reports_dir / f"{symbol}_{interval}_monitoring_history.jsonl"
        existing: list[str] = []
        if history_path.exists():
            try:
                existing = [
                    line
                    for line in history_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except OSError:
                existing = []
        existing.append(
            json.dumps(acknowledgement, ensure_ascii=False, separators=(",", ":"))
        )
        safe_replace_text(history_path, "\n".join(existing[-500:]) + "\n")
        acknowledgements.append(acknowledgement)
    return acknowledgements


def targets_from_memory(memory: dict[str, Any], max_targets: int) -> list[tuple[str, str]]:
    global_summary = memory.get("global") if isinstance(memory.get("global"), dict) else {}
    candidates = []
    for key in ["weak_targets", "top_candidates"]:
        for item in global_summary.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            action = str(item.get("next_action") or "")
            if key == "weak_targets" or action in WEAK_ACTIONS or item.get("eligible_for_paper"):
                symbol = str(item.get("symbol") or "").upper()
                interval = str(item.get("interval") or "")
                if symbol and interval:
                    candidates.append((symbol, interval))
    unique: list[tuple[str, str]] = []
    for target in candidates:
        if target not in unique:
            unique.append(target)
    return unique[:max_targets]


def target_lists(targets: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    symbols: list[str] = []
    intervals: list[str] = []
    for symbol, interval in targets:
        if symbol not in symbols:
            symbols.append(symbol)
        if interval not in intervals:
            intervals.append(interval)
    return symbols, intervals


def rotate_requested_targets(targets: list[tuple[str, str]], max_targets: int) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    if not targets:
        return [], {"enabled": False, "reason": "no_requested_targets"}
    limit = max(1, min(int(max_targets), len(targets)))
    if limit >= len(targets):
        return targets[:limit], {"enabled": False, "reason": "max_targets_covers_all_requested"}
    slot = int(time.time() // 3600)
    start = slot % len(targets)
    rotated = targets[start:] + targets[:start]
    selected = rotated[:limit]
    return selected, {
        "enabled": True,
        "slot": slot,
        "start_index": start,
        "requested_count": len(targets),
        "max_targets": limit,
        "policy": "rotate_requested_targets_hourly",
    }


def run_scheduled_optimization_once(
    *,
    config_path: str | Path | None = None,
    symbols: list[str] | None = None,
    intervals: list[str] | None = None,
    include_realtime: bool = True,
    time_budget_minutes: float = 12.0,
    max_model_trials: int = 1,
    max_training_rows: int = 12_000,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_targets: int = 1,
    maintenance: bool = True,
    maintenance_dry_run: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.ensure_dirs()
    progress = tracker_for_reports(cfg.reports_dir)
    progress.reset("Scheduled model optimization", 5 if maintenance else 4)
    lock_path = Path("state") / "scheduled_optimizer.lock"
    lock_fd, lock_info = acquire_lock(lock_path)
    if lock_fd is None:
        progress.finish("Scheduled model optimization skipped: another optimizer is running", metrics={"lock_path": str(lock_path)})
        payload = {
            "created_beijing": beijing_now_iso(),
            "mode": "scheduled_optimization",
            "status": "skipped",
            "reason": "scheduled_optimizer_lock_active",
            "lock": lock_info,
            "safety": {
                "live_trading_enabled": bool(cfg.live_trading_enabled),
                "real_orders_allowed": False,
                "api_keys_used": False,
            },
        }
        latest = cfg.reports_dir / "scheduled_optimization_latest.json"
        stamped = cfg.reports_dir / f"scheduled_optimization_{beijing_stamp()}.json"
        safe_write_json(latest, payload)
        safe_write_json(stamped, payload)
        payload["report_path"] = str(latest)
        payload["stamped_report_path"] = str(stamped)
        return payload

    try:
        return _run_scheduled_optimization_once_locked(
            cfg=cfg,
            progress=progress,
            config_path=config_path,
            symbols=symbols,
            intervals=intervals,
            include_realtime=include_realtime,
            time_budget_minutes=time_budget_minutes,
            max_model_trials=max_model_trials,
            max_training_rows=max_training_rows,
            complexity=complexity,
            rolling_folds=rolling_folds,
            max_targets=max_targets,
            maintenance=maintenance,
            maintenance_dry_run=maintenance_dry_run,
        )
    finally:
        release_lock(lock_fd, lock_path)


def _run_scheduled_optimization_once_locked(
    *,
    cfg: Any,
    progress: Any,
    config_path: str | Path | None,
    symbols: list[str] | None,
    intervals: list[str] | None,
    include_realtime: bool,
    time_budget_minutes: float,
    max_model_trials: int,
    max_training_rows: int,
    complexity: str,
    rolling_folds: int,
    max_targets: int,
    maintenance: bool,
    maintenance_dry_run: bool,
) -> dict[str, Any]:
    progress.update("Refreshing strategy memory before optimization")
    memory_before = update_simulation_memory(config_path=config_path)
    requested = normalize_targets(symbols, intervals)
    monitoring_targets, monitoring_details = monitoring_triggered_targets(
        cfg.reports_dir,
        allowed_symbols=symbols,
    )
    rotation_report: dict[str, Any] = {"enabled": False}
    if requested:
        ordinary_targets = [
            target for target in requested if target not in monitoring_targets
        ]
        triggered_selected = monitoring_targets[:max_targets]
        remaining_slots = max(max_targets - len(triggered_selected), 0)
        rotated_ordinary, rotation_report = rotate_requested_targets(
            ordinary_targets,
            remaining_slots,
        ) if remaining_slots else ([], {"enabled": False, "reason": "monitoring_targets_fill_budget"})
        selected_targets = triggered_selected + rotated_ordinary
        if monitoring_targets:
            rotation_report["monitoring_prioritized"] = True
            rotation_report["monitoring_selected_count"] = len(
                triggered_selected
            )
    else:
        selected_targets = monitoring_targets[:max_targets]
        if not selected_targets:
            selected_targets = targets_from_memory(memory_before, max_targets)
    if not selected_targets:
        selected_targets = [(symbol.upper(), cfg.realtime_interval) for symbol in list(cfg.symbols)[:2]]
    selected_symbols, selected_intervals = target_lists(selected_targets)

    maintenance_before = None
    if maintenance:
        progress.update("Compacting realtime training data")
        maintenance_before = run_data_maintenance(config_path=config_path, dry_run=maintenance_dry_run)

    progress.update("Running return-first model optimization", metrics={"targets": len(selected_targets)})
    optimization = run_model_optimization(
        cfg,
        symbols=selected_symbols,
        intervals=selected_intervals,
        target_pairs=selected_targets,
        include_realtime=include_realtime,
        time_budget_minutes=time_budget_minutes,
        max_model_trials=max_model_trials,
        max_training_rows=max_training_rows,
        initial_balance=10_000,
        min_trades=12,
        max_drawdown_limit=0.10,
        min_profit_factor=1.0,
        complexity=complexity,
        rolling_folds=rolling_folds,
        objective="return",
    )
    completed_targets = {
        (
            str(item.get("symbol") or "").upper(),
            str(item.get("interval") or ""),
        )
        for item in optimization.get("items", [])
        if isinstance(item, dict)
        and item.get("symbol")
        and item.get("interval")
    }
    monitoring_acknowledgements = acknowledge_monitoring_triggers(
        cfg.reports_dir,
        completed_targets,
        optimization_report=str(optimization.get("report_path") or ""),
    )
    progress.update("Refreshing strategy memory after optimization")
    memory_after = update_simulation_memory(config_path=config_path)

    payload = {
        "created_beijing": beijing_now_iso(),
        "mode": "scheduled_optimization",
        "requested_targets": [{"symbol": symbol, "interval": interval} for symbol, interval in requested],
        "target_rotation": rotation_report,
        "selected_targets": [{"symbol": symbol, "interval": interval} for symbol, interval in selected_targets],
        "monitoring_retraining_triggers": monitoring_details,
        "monitoring_acknowledgements": monitoring_acknowledgements,
        "complexity": complexity,
        "rolling_folds": rolling_folds,
        "include_realtime": include_realtime,
        "objective": "return",
        "time_budget_minutes": time_budget_minutes,
        "max_model_trials": max_model_trials,
        "max_training_rows": int(max_training_rows or 0),
        "maintenance": maintenance_before,
        "optimization_report": optimization.get("report_path"),
        "optimization_ranked": [
            {
                "symbol": item.get("symbol"),
                "interval": item.get("interval"),
                "model_name": item.get("model_name"),
                "balanced_accuracy": item.get("model_report", {}).get("test_metrics", {}).get("balanced_accuracy"),
                "accuracy": item.get("model_report", {}).get("test_metrics", {}).get("accuracy"),
                "auc": item.get("model_report", {}).get("test_metrics", {}).get("auc"),
                "total_return": item.get("model_report", {}).get("test_backtest", {}).get("total_return"),
                "max_drawdown": item.get("model_report", {}).get("test_backtest", {}).get("max_drawdown"),
                "profit_factor": item.get("model_report", {}).get("test_backtest", {}).get("profit_factor"),
            }
            for item in optimization.get("ranked", [])[:8]
        ],
        "memory_before": {
            "observation_count": memory_before.get("observation_count"),
            "top_candidates": memory_before.get("global", {}).get("top_candidates", [])[:5],
        },
        "memory_after": {
            "observation_count": memory_after.get("observation_count"),
            "top_candidates": memory_after.get("global", {}).get("top_candidates", [])[:5],
            "next_actions": memory_after.get("global", {}).get("next_actions", []),
        },
        "safety": {
            "live_trading_enabled": bool(cfg.live_trading_enabled),
            "real_orders_allowed": False,
            "api_keys_used": False,
        },
    }
    latest = cfg.reports_dir / "scheduled_optimization_latest.json"
    stamped = cfg.reports_dir / f"scheduled_optimization_{beijing_stamp()}.json"
    safe_write_json(latest, payload)
    safe_write_json(stamped, payload)
    payload["report_path"] = str(latest)
    payload["stamped_report_path"] = str(stamped)
    progress.finish("Scheduled model optimization complete", metrics={"scheduled_report": str(latest)})
    return payload


def run_scheduled_optimization_loop(
    *,
    iterations: int,
    sleep_minutes: float,
    **kwargs: Any,
) -> None:
    count = 0
    while True:
        count += 1
        payload = run_scheduled_optimization_once(**kwargs)
        print(
            json.dumps(
                {
                    "iteration": count,
                    "selected_targets": payload.get("selected_targets", []),
                    "report_path": payload.get("report_path"),
                    "live_trading_enabled": payload.get("safety", {}).get("live_trading_enabled", False),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if iterations and count >= iterations:
            return
        time.sleep(max(60.0, sleep_minutes * 60.0))
