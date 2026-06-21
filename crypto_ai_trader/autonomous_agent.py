from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

from .ai_optimization import run_ai_optimization_for_symbol
from .binance_data import load_symbol_interval
from .config import TraderConfig, load_config
from .live_training import download_result_payload, sync_and_train_live
from .progress import safe_replace_text, tracker_for_reports
from .time_utils import add_beijing_aliases, beijing_now_iso, beijing_stamp


def utc_now() -> str:
    return beijing_now_iso()


def stamp_now() -> str:
    return beijing_stamp()


@dataclass(frozen=True)
class AutonomousPolicy:
    min_total_return: float = 0.0
    max_drawdown_limit: float = 0.08
    min_profit_factor: float = 1.0
    min_trades: int = 12
    min_rows: int = 300
    max_report_age_minutes: int = 90
    optimization_trials: int = 40
    optimization_cooldown_minutes: int = 360


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    add_beijing_aliases(payload)
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def parse_utc(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def minutes_since(value: object) -> float | None:
    parsed = parse_utc(value)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).total_seconds() / 60.0


def as_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def backtest_from_report(item: dict[str, Any]) -> dict[str, Any]:
    if isinstance(item.get("backtest"), dict):
        return dict(item["backtest"])
    optimization = item.get("parameter_optimization")
    if isinstance(optimization, dict):
        best = optimization.get("best")
        if isinstance(best, dict) and isinstance(best.get("test"), dict):
            return dict(best["test"])
    return {}


def score_backtest_payload(backtest: dict[str, Any], policy: AutonomousPolicy) -> float:
    total_return = as_float(backtest.get("total_return"))
    max_drawdown = as_float(backtest.get("max_drawdown"))
    profit_factor = as_float(backtest.get("profit_factor"))
    trades = as_int(backtest.get("trades"))
    score = total_return
    if max_drawdown < -abs(policy.max_drawdown_limit):
        score -= abs(max_drawdown + abs(policy.max_drawdown_limit)) * 2.0
    if profit_factor < policy.min_profit_factor:
        score -= (policy.min_profit_factor - profit_factor) * 0.05
    if trades < policy.min_trades:
        score -= 0.10
    return float(score)


def evaluate_report_item(
    item: dict[str, Any],
    *,
    source: str,
    report_path: Path | None,
    policy: AutonomousPolicy,
) -> dict[str, Any]:
    backtest = backtest_from_report(item)
    total_return = as_float(backtest.get("total_return"))
    max_drawdown = as_float(backtest.get("max_drawdown"))
    profit_factor = as_float(backtest.get("profit_factor"))
    trades = as_int(backtest.get("trades"))
    rows = as_int(item.get("rows", item.get("raw_rows", 0)))

    checks = {
        "return_ok": total_return >= policy.min_total_return,
        "drawdown_ok": max_drawdown >= -abs(policy.max_drawdown_limit),
        "profit_factor_ok": profit_factor >= policy.min_profit_factor,
        "trades_ok": trades >= policy.min_trades,
        "rows_ok": rows >= policy.min_rows,
    }
    passed = all(checks.values())
    if passed:
        status = "paper_candidate"
        recommendation = "Continue paper observation; do not promote to real trading."
    elif rows < policy.min_rows:
        status = "needs_more_data"
        recommendation = "Collect more closed klines before trusting this model."
    elif trades < policy.min_trades:
        status = "needs_more_trades"
        recommendation = "Keep observing or lower frequency sensitivity only after validation."
    elif total_return < policy.min_total_return or profit_factor < policy.min_profit_factor:
        status = "needs_optimization"
        recommendation = "Run bounded local optimization and keep this out of live trading."
    elif max_drawdown < -abs(policy.max_drawdown_limit):
        status = "risk_blocked"
        recommendation = "Do not advance; drawdown exceeds the safety gate."
    else:
        status = "observe"
        recommendation = "Keep paper observation until checks are stable."

    return {
        "source": source,
        "report_path": str(report_path) if report_path else item.get("report_path", ""),
        "symbol": str(item.get("symbol", "")).upper(),
        "interval": str(item.get("interval", "")),
        "model_name": item.get("model_name", ""),
        "created_utc": item.get("created_utc", ""),
        "latest_datetime": item.get("latest_datetime", ""),
        "latest_up_probability": item.get("latest_up_probability"),
        "rows": rows,
        "backtest": backtest,
        "checks": checks,
        "status": status,
        "score": score_backtest_payload(backtest, policy),
        "recommendation": recommendation,
    }


def collect_runner_evaluations(
    cfg: TraderConfig,
    *,
    state_dir: Path,
    policy: AutonomousPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = read_json(state_dir / "runner_state.json", {"status": "not_started"})
    report_path = Path(str(state.get("latest_report") or cfg.reports_dir / "runner_live_latest.json"))
    payload = read_json(report_path, {})
    evaluations: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if isinstance(item, dict):
            evaluations.append(evaluate_report_item(item, source="runner_live", report_path=report_path, policy=policy))
    runner_age_minutes = minutes_since(state.get("last_run_utc"))
    runner_summary = {
        "status": state.get("status", "not_started"),
        "control_command": read_json(state_dir / "runner_control.json", {"command": "run"}).get("command", "run"),
        "last_run_utc": state.get("last_run_utc"),
        "age_minutes": runner_age_minutes,
        "stale": runner_age_minutes is None or runner_age_minutes > policy.max_report_age_minutes,
        "latest_report": str(report_path),
        "live_trading_enabled": bool(state.get("live_trading_enabled", False)),
    }
    return evaluations, runner_summary


def collect_ai_evaluations(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    interval: str,
    policy: AutonomousPolicy,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for symbol in symbols:
        report_path = cfg.reports_dir / f"{symbol}_{interval}_ai_optimization.json"
        payload = read_json(report_path, {})
        if payload:
            evaluations.append(evaluate_report_item(payload, source="ai_optimization", report_path=report_path, policy=policy))
    return evaluations


def collect_summary_evaluations(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    interval: str,
    policy: AutonomousPolicy,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for symbol in symbols:
        report_path = cfg.reports_dir / f"{symbol}_{interval}_backtest_summary.json"
        payload = read_json(report_path, {})
        if not payload:
            continue
        item = {
            "symbol": symbol,
            "interval": interval,
            "model_name": "baseline",
            "rows": policy.min_rows,
            "backtest": payload,
        }
        evaluations.append(evaluate_report_item(item, source="historical_backtest", report_path=report_path, policy=policy))
    return evaluations


def build_decision(
    *,
    cfg: TraderConfig,
    runner_summary: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]], list[str]]:
    blockers: list[dict[str, str]] = []
    next_actions: list[str] = []
    if cfg.live_trading_enabled:
        blockers.append(
            {
                "level": "critical",
                "message": "config live_trading_enabled is true; disable it before any autonomous cycle.",
            }
        )
    if runner_summary.get("live_trading_enabled"):
        blockers.append(
            {
                "level": "critical",
                "message": "runner state reports live trading enabled; stop and inspect before continuing.",
            }
        )
    if runner_summary.get("stale"):
        next_actions.append("Refresh the local runner or run one live-train cycle before trusting realtime decisions.")
    if not evaluations:
        next_actions.append("Run historical training/backtest or realtime runner first; no usable reports were found.")
        if blockers:
            return "blocked", blockers, next_actions
        return "needs_data", blockers, next_actions

    best = sorted(evaluations, key=lambda item: item["score"], reverse=True)[0]
    paper_candidates = sorted(
        [item for item in evaluations if item["status"] == "paper_candidate"],
        key=lambda row: row["score"],
        reverse=True,
    )
    need_optimization = [item for item in evaluations if item["status"] == "needs_optimization"]
    risk_blocked = [item for item in evaluations if item["status"] == "risk_blocked"]

    if blockers:
        decision = "blocked"
    elif paper_candidates:
        decision = "paper_observation"
        next_actions.append(
            f"Keep observing {paper_candidates[0]['symbol']} {paper_candidates[0]['interval']} in paper mode only."
        )
    elif risk_blocked:
        decision = "risk_blocked"
        next_actions.append("Reduce leverage/position fraction and require fresh validation before any promotion.")
    elif need_optimization:
        decision = "optimize_locally"
        next_actions.append("Run bounded local AI optimization for weak symbols; keep all actions simulated.")
    else:
        decision = "observe"
        next_actions.append(f"Best current candidate is {best['symbol']} {best['interval']}, but gates are not stable enough.")
    if not any("live trading" in item.lower() for item in next_actions):
        next_actions.append("Do not place real orders; this agent is review-and-simulation only.")
    return decision, blockers, next_actions


def select_optimization_symbols(
    *,
    requested_symbols: list[str],
    evaluations: list[dict[str, Any]],
    max_symbols: int = 4,
) -> list[str]:
    failing = [
        item["symbol"]
        for item in sorted(evaluations, key=lambda row: row["score"])
        if item.get("symbol") and item.get("status") in {"needs_optimization", "risk_blocked", "needs_more_trades"}
    ]
    ordered: list[str] = []
    for symbol in failing + requested_symbols:
        symbol = symbol.upper()
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered[:max_symbols]


def cooldown_allows(state: dict[str, Any], policy: AutonomousPolicy) -> bool:
    age = minutes_since(state.get("last_optimization_utc"))
    return age is None or age >= policy.optimization_cooldown_minutes


def execute_local_optimization(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    interval: str,
    policy: AutonomousPolicy,
    state_dir: Path,
) -> list[dict[str, Any]]:
    state = read_json(state_dir / "autonomous_agent_state.json", {})
    if not cooldown_allows(state, policy):
        return [
            {
                "action": "ai_optimization",
                "status": "skipped",
                "reason": "cooldown_active",
                "last_optimization_utc": state.get("last_optimization_utc"),
            }
        ]

    progress = tracker_for_reports(cfg.reports_dir)
    progress.reset("Autonomous local optimization", len(symbols))
    executed: list[dict[str, Any]] = []
    for symbol in symbols:
        progress.update(f"Optimizing {symbol} {interval}", current_symbol=symbol)
        try:
            raw = load_symbol_interval(cfg.data_dir, symbol, interval)
            report = run_ai_optimization_for_symbol(
                symbol,
                interval,
                raw,
                cfg,
                initial_balance=10_000,
                max_leverage=min(cfg.max_leverage, 3),
                n_trials=policy.optimization_trials,
                min_trades=policy.min_trades,
                max_drawdown_limit=policy.max_drawdown_limit,
                min_profit_factor=policy.min_profit_factor,
            )
            executed.append(
                {
                    "action": "ai_optimization",
                    "status": "completed",
                    "symbol": symbol,
                    "report_path": report.get("report_path", ""),
                }
            )
            progress.advance(f"{symbol} optimization complete", current_symbol=symbol)
        except Exception as exc:
            executed.append({"action": "ai_optimization", "status": "failed", "symbol": symbol, "error": str(exc)})
            progress.advance(f"{symbol} optimization failed: {exc}", current_symbol=symbol)
    progress.finish("Autonomous local optimization complete")
    state["last_optimization_utc"] = utc_now()
    write_json(state_dir / "autonomous_agent_state.json", state)
    return executed


def execute_live_training(
    cfg: TraderConfig,
    *,
    symbols: list[str],
    interval: str,
    limit: int,
    base_url: str,
) -> dict[str, Any]:
    result = sync_and_train_live(
        symbols=symbols,
        interval=interval,
        cfg=cfg,
        limit=limit,
        base_url=base_url,
        model_suffix="autonomous_live",
    )
    return {
        "action": "live_training",
        "status": "completed",
        "output": str(result.get("output", "")),
        "sync": [download_result_payload(item) for item in result.get("sync", [])],
    }


def run_autonomous_review(
    *,
    config_path: str | Path | None = None,
    symbols: list[str] | None = None,
    interval: str | None = None,
    runner_interval: str | None = None,
    state_dir: str | Path = "state",
    policy: AutonomousPolicy | None = None,
    execute_optimization: bool = False,
    execute_live_train: bool = False,
    live_limit: int | None = None,
    live_base_url: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    cfg.ensure_dirs()
    active_policy = policy or AutonomousPolicy()
    active_symbols = [item.upper() for item in (symbols or list(cfg.symbols))]
    active_interval = interval or cfg.interval
    active_runner_interval = runner_interval or cfg.realtime_interval
    state_path = Path(state_dir)

    runner_evaluations, runner_summary = collect_runner_evaluations(
        cfg,
        state_dir=state_path,
        policy=active_policy,
    )
    runner_evaluations = [
        item for item in runner_evaluations if not active_runner_interval or item.get("interval") == active_runner_interval
    ]
    evaluations = (
        runner_evaluations
        + collect_ai_evaluations(cfg, symbols=active_symbols, interval=active_interval, policy=active_policy)
        + collect_summary_evaluations(cfg, symbols=active_symbols, interval=active_interval, policy=active_policy)
    )
    try:
        from .simulation_memory import update_simulation_memory

        memory_payload = update_simulation_memory(config_path=config_path, state_dir=state_path)
        memory_summary = {
            "updated_beijing": memory_payload.get("updated_beijing"),
            "observation_count": memory_payload.get("observation_count"),
            "target_count": memory_payload.get("target_count"),
            "top_candidates": memory_payload.get("global", {}).get("top_candidates", [])[:5],
            "next_actions": memory_payload.get("global", {}).get("next_actions", []),
            "report_path": memory_payload.get("report_path"),
            "state_path": memory_payload.get("state_path"),
        }
    except Exception as exc:
        memory_summary = {"status": "failed", "error": str(exc)}
    ranked = sorted(evaluations, key=lambda item: item["score"], reverse=True)
    decision, blockers, next_actions = build_decision(cfg=cfg, runner_summary=runner_summary, evaluations=evaluations)

    executed_actions: list[dict[str, Any]] = []
    if execute_live_train:
        try:
            executed_actions.append(
                execute_live_training(
                    cfg,
                    symbols=active_symbols,
                    interval=active_runner_interval,
                    limit=live_limit or cfg.realtime_limit,
                    base_url=live_base_url or cfg.realtime_base_url,
                )
            )
        except Exception as exc:
            executed_actions.append({"action": "live_training", "status": "failed", "error": str(exc)})

    if execute_optimization:
        optimization_symbols = select_optimization_symbols(requested_symbols=active_symbols, evaluations=evaluations)
        executed_actions.extend(
            execute_local_optimization(
                cfg,
                symbols=optimization_symbols,
                interval=active_interval,
                policy=active_policy,
                state_dir=state_path,
            )
        )

    payload = {
        "created_utc": utc_now(),
        "mode": "review_and_simulation",
        "safety": {
            "live_trading_enabled": cfg.live_trading_enabled,
            "real_orders_allowed": False,
            "blockers": blockers,
        },
        "policy": asdict(active_policy),
        "symbols": active_symbols,
        "historical_interval": active_interval,
        "runner_interval": active_runner_interval,
        "runner": runner_summary,
        "simulation_memory": memory_summary,
        "decision": decision,
        "next_actions": next_actions,
        "ranked": ranked,
        "executed_actions": executed_actions,
    }
    latest_path = cfg.reports_dir / "autonomous_review_latest.json"
    stamped_path = cfg.reports_dir / f"autonomous_review_{stamp_now()}.json"
    write_json(latest_path, payload)
    write_json(stamped_path, payload)

    state = read_json(state_path / "autonomous_agent_state.json", {})
    state.update(
        {
            "last_review_utc": payload["created_utc"],
            "latest_report": str(latest_path),
            "latest_stamped_report": str(stamped_path),
            "decision": decision,
            "blockers": blockers,
        }
    )
    write_json(state_path / "autonomous_agent_state.json", state)
    return payload


def run_autonomous_loop(
    *,
    review_every_seconds: int,
    iterations: int,
    **kwargs: Any,
) -> None:
    count = 0
    while True:
        count += 1
        payload = run_autonomous_review(**kwargs)
        print(json.dumps({"iteration": count, "decision": payload["decision"], "report": "reports/autonomous_review_latest.json"}, indent=2))
        if iterations and count >= iterations:
            return
        time.sleep(max(review_every_seconds, 60))
