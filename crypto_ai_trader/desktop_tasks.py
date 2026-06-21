from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any

from .config import load_config
from .progress import safe_replace_text
from .time_utils import add_beijing_aliases, beijing_now_iso, beijing_task_stamp, to_beijing_iso


VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h"}
TASK_KINDS = {
    "runner",
    "model-optimize",
    "live-train",
    "ai-optimize",
    "cycle",
    "doctor",
    "export-portable",
    "download-cache",
    "memory-update",
    "scheduled-optimize",
    "strategy-validate",
    "data-maintenance",
}
PROCESS_REGISTRY: dict[str, subprocess.Popen] = {}


def utc_now() -> str:
    return beijing_now_iso()


def windows_creationflags() -> int:
    if os.name != "nt":
        return 0
    return (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    )


def project_python(root: Path) -> str:
    if os.name == "nt":
        candidate = root / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = root / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def task_store_path(state_dir: Path) -> Path:
    return state_dir / "desktop_tasks.json"


def task_log_dir(logs_dir: Path) -> Path:
    return logs_dir / "desktop_tasks"


def load_tasks(state_dir: Path) -> list[dict[str, Any]]:
    tasks = read_json(task_store_path(state_dir), [])
    return tasks if isinstance(tasks, list) else []


def save_tasks(state_dir: Path, tasks: list[dict[str, Any]]) -> None:
    write_json(task_store_path(state_dir), tasks[-200:])


def sanitize_symbol_list(value: Any, allowed_symbols: set[str], default: list[str]) -> list[str]:
    items = value if isinstance(value, list) else default
    parsed = [str(item).strip().upper() for item in items if str(item or "").strip()]
    parsed = [item for item in parsed if item in allowed_symbols]
    return parsed or default


def sanitize_interval_list(value: Any, default: list[str]) -> list[str]:
    items = value if isinstance(value, list) else default
    parsed = [str(item) for item in items if str(item) in VALID_INTERVALS]
    return parsed or default


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def sanitize_complexity(value: Any, default: str = "standard") -> str:
    parsed = str(value or default).strip().lower()
    return parsed if parsed in {"standard", "expanded", "deep", "blackbox"} else default


def sanitize_validation_profile(value: Any, default: str = "standard") -> str:
    parsed = str(value or default).strip().lower()
    return parsed if parsed in {"standard", "large-sample-light", "fast-screen", "deep-audit"} else default


def build_task_command(root: Path, kind: str, params: dict[str, Any]) -> list[str]:
    cfg = load_config(root / "config.default.json")
    allowed_symbols = {item.upper() for item in cfg.symbols}
    python = project_python(root)
    base = [python, "-m"]
    if kind == "model-optimize":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["SOLUSDT", "ETHUSDT", "BNBUSDT"])
        intervals = sanitize_interval_list(params.get("intervals"), ["1h"])
        command = [
            *base,
            "crypto_ai_trader.cli",
            "model-optimize",
            "--complexity",
            sanitize_complexity(params.get("complexity"), "standard"),
            "--rolling-folds",
            str(clamp_int(params.get("rolling_folds"), 0, 0, 5)),
            "--symbols",
            *symbols,
            "--intervals",
            *intervals,
            "--time-budget-minutes",
            str(clamp_float(params.get("time_budget_minutes"), 15.0, 1.0, 180.0)),
            "--max-model-trials",
            str(clamp_int(params.get("max_model_trials"), 1, 1, 200)),
            "--max-training-rows",
            str(clamp_int(params.get("max_training_rows"), 12_000, 0, 1_000_000)),
            "--initial-balance",
            str(clamp_float(params.get("initial_balance"), 10_000.0, 100.0, 1_000_000_000.0)),
            "--min-trades",
            str(clamp_int(params.get("min_trades"), 8, 1, 10_000)),
            "--max-drawdown-limit",
            str(clamp_float(params.get("max_drawdown_limit"), 0.10, 0.01, 1.0)),
            "--min-profit-factor",
            str(clamp_float(params.get("min_profit_factor"), 1.0, 0.0, 10.0)),
        ]
        if bool_value(params.get("include_realtime"), True):
            command.append("--include-realtime")
        return command
    if kind == "strategy-validate":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
        intervals = sanitize_interval_list(params.get("intervals"), ["1h"])
        command = [
            *base,
            "crypto_ai_trader.cli",
            "strategy-validate",
            "--complexity",
            sanitize_complexity(params.get("complexity"), "standard"),
            "--validation-profile",
            sanitize_validation_profile(params.get("validation_profile"), "standard"),
            "--rolling-folds",
            str(clamp_int(params.get("rolling_folds"), 1, 0, 5)),
            "--symbols",
            *symbols,
            "--intervals",
            *intervals,
            "--folds",
            str(clamp_int(params.get("folds"), 3, 1, 10)),
            "--time-budget-minutes",
            str(clamp_float(params.get("time_budget_minutes"), 30.0, 1.0, 240.0)),
            "--per-target-budget-minutes",
            str(clamp_float(params.get("per_target_budget_minutes"), 0.0, 0.0, 240.0)),
            "--max-model-trials",
            str(clamp_int(params.get("max_model_trials"), 4, 1, 100)),
            "--max-training-rows",
            str(clamp_int(params.get("max_training_rows"), 0, 0, 1_000_000)),
            "--wf-train-rows",
            str(clamp_int(params.get("wf_train_rows"), 0, 0, 1_000_000)),
            "--wf-valid-rows",
            str(clamp_int(params.get("wf_valid_rows"), 0, 0, 1_000_000)),
            "--wf-test-rows",
            str(clamp_int(params.get("wf_test_rows"), 0, 0, 1_000_000)),
            "--max-threshold-evals",
            str(clamp_int(params.get("max_threshold_evals"), 0, 0, 100_000)),
            "--initial-balance",
            str(clamp_float(params.get("initial_balance"), 10_000.0, 100.0, 1_000_000_000.0)),
            "--min-trades",
            str(clamp_int(params.get("min_trades"), 8, 1, 10_000)),
            "--max-drawdown-limit",
            str(clamp_float(params.get("max_drawdown_limit"), 0.10, 0.01, 1.0)),
            "--min-profit-factor",
            str(clamp_float(params.get("min_profit_factor"), 0.8, 0.0, 10.0)),
        ]
        if bool_value(params.get("include_realtime"), False):
            command.append("--include-realtime")
        return command
    if kind == "scheduled-optimize":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["SOLUSDT", "ETHUSDT", "BNBUSDT"])
        intervals = sanitize_interval_list(params.get("intervals"), ["1h"])
        command = [
            *base,
            "crypto_ai_trader.cli",
            "scheduled-optimize",
            "--complexity",
            sanitize_complexity(params.get("complexity"), "standard"),
            "--rolling-folds",
            str(clamp_int(params.get("rolling_folds"), 0, 0, 5)),
            "--symbols",
            *symbols,
            "--intervals",
            *intervals,
            "--time-budget-minutes",
            str(clamp_float(params.get("time_budget_minutes"), 12.0, 1.0, 180.0)),
            "--max-model-trials",
            str(clamp_int(params.get("max_model_trials"), 1, 1, 300)),
            "--max-training-rows",
            str(clamp_int(params.get("max_training_rows"), 12_000, 0, 1_000_000)),
            "--max-targets",
            str(clamp_int(params.get("max_targets"), 1, 1, 12)),
        ]
        if bool_value(params.get("include_realtime"), True):
            command.append("--include-realtime")
        else:
            command.append("--no-include-realtime")
        if bool_value(params.get("maintenance"), True):
            command.append("--maintenance")
        else:
            command.append("--no-maintenance")
        if bool_value(params.get("maintenance_dry_run"), False):
            command.append("--maintenance-dry-run")
        return command
    if kind == "data-maintenance":
        command = [
            *base,
            "crypto_ai_trader.cli",
            "data-maintenance",
            "--cleanup-tmp-older-than-hours",
            str(clamp_float(params.get("cleanup_tmp_older_than_hours"), 6.0, 0.0, 720.0)),
            "--max-realtime-rows",
            str(clamp_int(params.get("max_realtime_rows"), 6000, 500, 200000)),
        ]
        if bool_value(params.get("dry_run"), False):
            command.append("--dry-run")
        if not bool_value(params.get("archive_old_realtime"), True):
            command.append("--no-archive-realtime")
        return command
    if kind == "download-cache":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["ETHUSDT", "BNBUSDT"])
        interval = sanitize_interval_list([params.get("interval")], [cfg.realtime_interval])[0]
        start = str(params.get("start") or "2024-01")
        end = str(params.get("end") or "2025-12")
        return [
            *base,
            "crypto_ai_trader.cli",
            "download",
            "--symbols",
            *symbols,
            "--interval",
            interval,
            "--start",
            start,
            "--end",
            end,
            "--cache-only",
            "--allow-partial-cache",
        ]
    if kind == "memory-update":
        return [
            *base,
            "crypto_ai_trader.cli",
            "memory-update",
            "--max-observations-per-target",
            str(clamp_int(params.get("max_observations_per_target"), 120, 20, 1000)),
        ]
    if kind == "live-train":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["ETHUSDT"])
        interval = sanitize_interval_list([params.get("interval")], [cfg.realtime_interval])[0]
        return [
            *base,
            "crypto_ai_trader.cli",
            "live-train",
            "--symbols",
            *symbols,
            "--interval",
            interval,
            "--limit",
            str(clamp_int(params.get("limit"), min(cfg.realtime_limit, 800), 100, 2000)),
            "--iterations",
            str(clamp_int(params.get("iterations"), 1, 1, 100)),
            "--sleep-seconds",
            str(clamp_int(params.get("sleep_seconds"), 60, 1, 3600)),
        ]
    if kind == "ai-optimize":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
        interval = sanitize_interval_list([params.get("interval")], [cfg.interval])[0]
        return [
            *base,
            "crypto_ai_trader.cli",
            "ai-optimize",
            "--symbols",
            *symbols,
            "--interval",
            interval,
            "--trials",
            str(clamp_int(params.get("trials"), 40, 1, 500)),
            "--max-leverage",
            str(clamp_int(params.get("max_leverage"), min(cfg.max_leverage, 3), 1, cfg.max_leverage)),
            "--min-trades",
            str(clamp_int(params.get("min_trades"), 12, 1, 10_000)),
            "--max-drawdown-limit",
            str(clamp_float(params.get("max_drawdown_limit"), 0.35, 0.01, 1.0)),
            "--quiet",
        ]
    if kind == "cycle":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
        interval = sanitize_interval_list([params.get("interval")], [cfg.interval])[0]
        start = str(params.get("start") or "2024-01")
        end = str(params.get("end") or "2025-12")
        command = [
            *base,
            "crypto_ai_trader.cli",
            "cycle",
            "--symbols",
            *symbols,
            "--interval",
            interval,
            "--start",
            start,
            "--end",
            end,
        ]
        if bool_value(params.get("skip_download"), False):
            command.append("--skip-download")
        return command
    if kind == "runner":
        symbols = sanitize_symbol_list(params.get("symbols"), allowed_symbols, ["ETHUSDT", "BNBUSDT"])
        interval = sanitize_interval_list([params.get("interval")], [cfg.realtime_interval])[0]
        return [
            *base,
            "crypto_ai_trader.runner",
            "once",
            "--symbols",
            *symbols,
            "--interval",
            interval,
            "--limit",
            str(clamp_int(params.get("limit"), min(cfg.realtime_limit, 800), 100, 2000)),
        ]
    if kind == "doctor":
        return [*base, "crypto_ai_trader.cli", "doctor"]
    if kind == "export-portable":
        name = str(params.get("name") or "")
        command = [*base, "crypto_ai_trader.portable"]
        if name:
            command.extend(["--name", Path(name).name])
        return command
    raise ValueError(f"Unsupported task kind: {kind}")


def refresh_task_record(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id", ""))
    process = PROCESS_REGISTRY.get(task_id)
    if task.get("status") == "running" and process is not None:
        code = process.poll()
        if code is not None:
            task["status"] = "completed" if code == 0 else "failed"
            task["exit_code"] = code
            task["ended_utc"] = utc_now()
            task["ended_beijing"] = task["ended_utc"]
    return task


def refresh_tasks(state_dir: Path) -> list[dict[str, Any]]:
    tasks = [refresh_task_record(task) for task in load_tasks(state_dir)]
    for task in tasks:
        add_beijing_aliases(task)
    save_tasks(state_dir, tasks)
    return tasks


def active_same_kind(tasks: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for task in tasks:
        if task.get("kind") == kind and task.get("status") == "running":
            return task
    return None


def start_task(root: Path, state_dir: Path, logs_dir: Path, kind: str, params: dict[str, Any]) -> dict[str, Any]:
    if kind not in TASK_KINDS:
        raise ValueError(f"Unsupported task kind: {kind}")
    tasks = refresh_tasks(state_dir)
    existing = active_same_kind(tasks, kind)
    if existing:
        return {"ok": False, "error": f"{kind} is already running", "task": existing}

    task_id = f"{beijing_task_stamp()}_{kind.replace('-', '_')}_{uuid.uuid4().hex[:8]}"
    log_dir = task_log_dir(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{task_id}.out.log"
    stderr_path = log_dir / f"{task_id}.err.log"
    command = build_task_command(root, kind, params)
    with stdout_path.open("ab") as out_log, stderr_path.open("ab") as err_log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=out_log,
            stderr=err_log,
            creationflags=windows_creationflags(),
        )
    PROCESS_REGISTRY[task_id] = process
    task = {
        "task_id": task_id,
        "kind": kind,
        "status": "running",
        "pid": process.pid,
        "command": command,
        "params": params,
        "started_utc": utc_now(),
        "started_beijing": utc_now(),
        "ended_utc": "",
        "ended_beijing": "",
        "exit_code": None,
        "logs": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
        "live_trading_enabled": False,
    }
    tasks.append(task)
    save_tasks(state_dir, tasks)
    return {"ok": True, "task": task}


def get_task(state_dir: Path, task_id: str) -> dict[str, Any] | None:
    for task in refresh_tasks(state_dir):
        if task.get("task_id") == task_id:
            return task
    return None


def cancel_task(state_dir: Path, task_id: str, force: bool = False) -> dict[str, Any]:
    tasks = refresh_tasks(state_dir)
    target = None
    for task in tasks:
        if task.get("task_id") == task_id:
            target = task
            break
    if not target:
        return {"ok": False, "error": "task_not_found"}
    if target.get("status") != "running":
        return {"ok": True, "task": target}
    process = PROCESS_REGISTRY.get(task_id)
    if process is None:
        target["status"] = "unknown"
        target["ended_utc"] = utc_now()
    else:
        if force:
            process.kill()
        else:
            process.terminate()
        target["status"] = "cancelled"
        target["ended_utc"] = utc_now()
        target["ended_beijing"] = target["ended_utc"]
        target["exit_code"] = process.poll()
    add_beijing_aliases(target)
    save_tasks(state_dir, tasks)
    return {"ok": True, "task": target}


def read_task_log(state_dir: Path, logs_dir: Path, task_id: str, stream: str, tail: int) -> dict[str, Any]:
    task = get_task(state_dir, task_id)
    if not task:
        return {"ok": False, "error": "task_not_found"}
    logs = task.get("logs") or {}
    path = Path(str(logs.get(stream if stream in {"stdout", "stderr"} else "stdout", ""))).resolve()
    if not path.exists():
        return {"ok": True, "task_id": task_id, "stream": stream, "lines": []}
    log_root = task_log_dir(logs_dir).resolve()
    if path.parent.resolve() != log_root:
        return {"ok": False, "error": "invalid_log_path"}
    max_lines = clamp_int(tail, 200, 1, 1000)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    return {"ok": True, "task_id": task_id, "stream": stream, "lines": lines}


def latest_model_optimization(reports_dir: Path) -> dict[str, Any]:
    files = sorted(reports_dir.glob("model_optimization_*.json"), key=lambda item: item.stat().st_mtime)
    if not files:
        return {"ok": True, "empty": True, "message": "No model optimization reports found."}
    path = files[-1]
    payload = read_json(path, {})
    return {
        "ok": True,
        "empty": False,
        "file": path.name,
        "path": str(path),
        "created_utc": payload.get("created_utc", ""),
        "created_beijing": payload.get("created_beijing") or to_beijing_iso(payload.get("created_utc", "")),
        "objective": payload.get("objective", ""),
        "feature_version": payload.get("feature_version", ""),
        "ranked": payload.get("ranked", []),
        "skipped_targets": payload.get("skipped_targets", []),
        "live_trading_enabled": payload.get("live_trading_enabled", False),
    }


def list_model_optimization_reports(reports_dir: Path, limit: int = 20) -> dict[str, Any]:
    max_items = clamp_int(limit, 20, 1, 100)
    files = sorted(reports_dir.glob("model_optimization_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    items = []
    for path in files[:max_items]:
        payload = read_json(path, {})
        items.append(
            {
                "file": path.name,
                "path": str(path),
                "created_utc": payload.get("created_utc", ""),
                "created_beijing": payload.get("created_beijing") or to_beijing_iso(payload.get("created_utc", "")),
                "objective": payload.get("objective", ""),
                "feature_version": payload.get("feature_version", ""),
                "targets": len(payload.get("items", [])),
            }
        )
    return {"ok": True, "items": items}


def open_workspace_path(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "root": root,
        "data": root / "data",
        "reports": root / "reports",
        "models": root / "models",
        "logs": root / "logs",
        "state": root / "state",
        "exports": root / "exports",
    }
    kind = str(payload.get("kind") or "root")
    base = allowed.get(kind)
    if base is None:
        return {"ok": False, "error": "unsupported_path_kind"}
    target = base.resolve()
    select = str(payload.get("select") or "")
    if select:
        candidate = (base / select).resolve()
        if base.resolve() not in candidate.parents and candidate != base.resolve():
            return {"ok": False, "error": "invalid_selected_path"}
        target = candidate
    if os.name == "nt":
        subprocess.Popen(["explorer.exe", str(target)], cwd=root)
    else:
        subprocess.Popen(["xdg-open", str(target)], cwd=root)
    return {"ok": True, "opened": str(target)}
