from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import pandas as pd

from .binance_data import load_symbol_interval, sync_recent_futures_klines
from .config import load_config
from .live_training import download_result_payload, train_live_symbol
from .portfolio import build_portfolio_snapshot
from .portfolio_paper import (
    extract_aligned_mark,
    load_portfolio_paper_state,
    mark_portfolio_paper_state,
    persist_portfolio_paper_state,
    portfolio_paper_risk_state,
    rebalance_portfolio_paper_state,
)
from .progress import safe_replace_text, tracker_for_reports
from .time_utils import add_beijing_aliases, beijing_now_iso, beijing_stamp


def utc_now() -> str:
    return beijing_now_iso()


def detect_local_proxy() -> str:
    ports = [7890, 7891, 7897, 1080, 1087, 10808, 10809, 20170, 2080, 8080, 8888]
    for port in ports:
        sock = socket.socket()
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
            return f"http://127.0.0.1:{port}"
        except OSError:
            continue
        finally:
            sock.close()
    return ""


def apply_proxy(proxy: str | None, auto_detect: bool) -> str:
    chosen = proxy or ""
    if not chosen and auto_detect:
        chosen = detect_local_proxy()
    if chosen:
        os.environ["HTTPS_PROXY"] = chosen
        os.environ["HTTP_PROXY"] = chosen
    return chosen


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def control_path(state_dir: Path) -> Path:
    return state_dir / "runner_control.json"


def state_path(state_dir: Path) -> Path:
    return state_dir / "runner_state.json"


def set_control(state_dir: Path, command: str) -> None:
    write_json(control_path(state_dir), {"command": command, "updated_utc": utc_now()})


def read_control(state_dir: Path) -> str:
    payload = read_json(control_path(state_dir), {"command": "run"})
    return str(payload.get("command", "run")).lower()


def write_state(state_dir: Path, **updates: Any) -> None:
    path = state_path(state_dir)
    payload = read_json(path, {})
    payload.update(updates)
    payload["updated_utc"] = utc_now()
    payload["updated_beijing"] = payload["updated_utc"]
    add_beijing_aliases(payload, overwrite=True)
    write_json(path, payload)


def runner_once(
    *,
    symbols: list[str],
    interval: str,
    limit: int,
    base_url: str,
    model_suffix: str,
    state_dir: Path,
    max_model_trials: int = 1,
    time_budget_minutes: float = 3.0,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_training_rows: int = 3000,
    progress: Any | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    cfg.ensure_dirs()
    sync_results = sync_recent_futures_klines(
        symbols=symbols,
        interval=interval,
        data_dir=cfg.data_dir,
        limit=limit,
        base_url=base_url,
    )
    if progress is not None:
        progress.advance(
            f"Live kline sync complete: {', '.join(f'{item.symbol}:{item.rows}' for item in sync_results)}",
            metrics={"synced_symbols": len(sync_results)},
            event_type="live_sync",
        )
    reports = []
    for symbol in symbols:
        if progress is not None:
            progress.update(
                f"Training lightweight runner model for {symbol}",
                current_symbol=symbol,
                metrics={
                    "complexity": complexity,
                    "max_model_trials": max_model_trials,
                    "time_budget_minutes": time_budget_minutes,
                    "max_training_rows": max_training_rows,
                },
            )
        report = train_live_symbol(
            symbol,
            interval,
            cfg,
            include_realtime=True,
            model_suffix=model_suffix,
            max_model_trials=max_model_trials,
            time_budget_minutes=time_budget_minutes,
            complexity=complexity,
            rolling_folds=rolling_folds,
            max_training_rows=max_training_rows,
        )
        reports.append(report)
        if progress is not None:
            progress.advance(
                f"{symbol} runner train complete: prob_up {report['latest_up_probability']:.4f}",
                current_symbol=symbol,
                metrics={
                    "model_name": report["model_name"],
                    "latest_up_probability": report["latest_up_probability"],
                    "risk_level": (report.get("latest_risk_decision") or {}).get("risk_level"),
                    "risk_reason": (report.get("latest_risk_decision") or {}).get("reason"),
                    "risk_allow_trade": (report.get("latest_risk_decision") or {}).get("allow_trade"),
                    "execution_allowed": (report.get("latest_execution_decision") or {}).get("allow_execution"),
                    "execution_reason": (report.get("latest_execution_decision") or {}).get("reason"),
                    "total_return": report["backtest"]["total_return"],
                    "max_drawdown": report["backtest"]["max_drawdown"],
                },
                event_type="runner_train",
            )
    ranked = sorted(reports, key=lambda item: item["backtest"]["total_return"], reverse=True)
    return_series: dict[str, pd.Series] = {}
    for symbol in symbols:
        frame = load_symbol_interval(
            cfg.data_dir,
            symbol,
            interval,
            include_realtime=True,
        )
        if frame.empty or "close" not in frame.columns:
            continue
        recent = frame.tail(cfg.portfolio_correlation_lookback + 1).copy()
        close = pd.to_numeric(recent["close"], errors="coerce")
        series = close.pct_change()
        series.index = pd.to_numeric(recent["open_time"], errors="coerce")
        return_series[symbol] = series
    return_matrix = (
        pd.concat(return_series, axis=1).sort_index()
        if return_series
        else pd.DataFrame()
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    portfolio_paper_state_path = (
        state_dir / f"portfolio_paper_{interval}.json"
    )
    portfolio_paper_ledger_path = (
        cfg.reports_dir / f"portfolio_paper_{interval}.jsonl"
    )
    portfolio_paper_latest_path = (
        cfg.reports_dir / "portfolio_paper_latest.json"
    )
    portfolio_paper_state = load_portfolio_paper_state(
        portfolio_paper_state_path,
        interval=interval,
        cfg=cfg,
    )
    portfolio_mark, portfolio_mark_reason = extract_aligned_mark(reports)
    portfolio_paper_event = mark_portfolio_paper_state(
        portfolio_paper_state,
        portfolio_mark,
        mark_reason=portfolio_mark_reason,
    )
    portfolio_risk_before = portfolio_paper_risk_state(
        portfolio_paper_state,
        cfg,
    )
    portfolio_snapshot = build_portfolio_snapshot(
        reports,
        return_matrix,
        cfg,
        interval=interval,
        current_drawdown=float(portfolio_risk_before["drawdown"]),
        current_daily_return=float(
            portfolio_risk_before["daily_return"]
        ),
        current_risk_source="portfolio_paper_ledger",
        input_available=portfolio_mark is not None,
        input_unavailable_reason=(
            "portfolio_blocked_unaligned_market_data"
            if portfolio_mark_reason == "unaligned_latest_open_time"
            else f"portfolio_blocked_{portfolio_mark_reason}"
        ),
        expected_open_time=(
            int(portfolio_mark.open_time)
            if portfolio_mark is not None
            else None
        ),
    )
    if (
        bool(portfolio_paper_event.get("advanced"))
        and not bool(portfolio_risk_before["allow_portfolio"])
        and any(
            abs(float(weight)) > 1e-15
            for weight in portfolio_paper_state.weights.values()
        )
    ):
        portfolio_paper_state.circuit_breaker_events += 1
    portfolio_paper_event = rebalance_portfolio_paper_state(
        portfolio_paper_state,
        dict(portfolio_snapshot["decision"]["weights"]),
        cfg,
        portfolio_paper_event,
    )
    portfolio_risk_after = portfolio_paper_risk_state(
        portfolio_paper_state,
        cfg,
    )
    portfolio_paper_latest = persist_portfolio_paper_state(
        portfolio_paper_state,
        portfolio_paper_event,
        state_path=portfolio_paper_state_path,
        ledger_path=portfolio_paper_ledger_path,
        latest_path=portfolio_paper_latest_path,
        max_history=cfg.portfolio_paper_max_history,
    )
    portfolio_snapshot["paper"] = {
        **portfolio_paper_latest,
        "risk_before_decision": portfolio_risk_before,
        "risk_after_rebalance": portfolio_risk_after,
    }
    payload = {
        "created_utc": utc_now(),
        "created_beijing": utc_now(),
        "symbols": symbols,
        "interval": interval,
        "sync": [download_result_payload(item) for item in sync_results],
        "items": reports,
        "ranked": ranked,
        "portfolio": portfolio_snapshot,
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    latest_path = cfg.reports_dir / "runner_live_latest.json"
    stamped_path = cfg.reports_dir / f"runner_live_{interval}_{beijing_stamp()}.json"
    portfolio_path = cfg.reports_dir / "portfolio_snapshot_latest.json"
    safe_replace_text(latest_path, json.dumps(payload, indent=2, ensure_ascii=False))
    safe_replace_text(stamped_path, json.dumps(payload, indent=2, ensure_ascii=False))
    safe_replace_text(
        portfolio_path,
        json.dumps(portfolio_snapshot, indent=2, ensure_ascii=False),
    )
    def selected_risk_profile(item: dict) -> str | None:
        strategy = item.get("small_account_strategy") or {}
        if not strategy:
            strategy = ((item.get("model_report") or {}).get("small_account_strategy") or {})
        value = strategy.get("selected_risk_profile")
        return value if value else None

    write_state(
        state_dir,
        status="running",
        last_run_utc=payload["created_utc"],
        latest_report=str(latest_path),
        latest_stamped_report=str(stamped_path),
        latest_portfolio_report=str(portfolio_path),
        latest_portfolio=portfolio_snapshot["decision"],
        latest_portfolio_paper=str(portfolio_paper_latest_path),
        latest_portfolio_paper_state=portfolio_paper_state.to_dict(),
        latest_ranked=[
            {
                "symbol": item["symbol"],
                "model_name": item["model_name"],
                "latest_up_probability": item["latest_up_probability"],
                "latest_risk_decision": item.get("latest_risk_decision"),
                "latest_execution_decision": item.get("latest_execution_decision"),
                "latest_horizon_probabilities": item.get("latest_horizon_probabilities", {}),
                "total_return": item["backtest"]["total_return"],
                "max_drawdown": item["backtest"]["max_drawdown"],
                "long_threshold": (item.get("optimized_backtest_config") or {}).get("long_threshold"),
                "short_threshold": (item.get("optimized_backtest_config") or {}).get("short_threshold"),
                "trade_side_policy": (item.get("optimized_backtest_config") or {}).get("trade_side_policy"),
                "model_rejected_by_validation_trading_gate": (
                    (item.get("model_report") or {}).get("model_selection_gate") or {}
                ).get("rejected_by_validation_trading_gate"),
                "selected_risk_profile": selected_risk_profile(item),
            }
            for item in ranked
        ],
    )
    return payload


def run_loop(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    state_dir = Path(args.state_dir)
    symbols = [item.upper() for item in args.symbols]
    proxy = apply_proxy(args.proxy or cfg.https_proxy, cfg.auto_detect_proxy)
    set_control(state_dir, "run")
    write_state(
        state_dir,
        status="starting",
        pid=os.getpid(),
        started_utc=utc_now(),
        symbols=symbols,
        interval=args.interval,
        limit=args.limit,
        train_every_seconds=args.train_every_seconds,
        live_max_training_rows=args.live_max_training_rows,
        poll_seconds=args.poll_seconds,
        proxy=proxy,
        live_trading_enabled=False,
    )
    progress = tracker_for_reports(cfg.reports_dir)
    progress.reset(f"Local runner {args.interval}", 1)

    iteration = 0
    next_train_at = 0.0
    while True:
        command = read_control(state_dir)
        if command == "stop":
            write_state(state_dir, status="stopped", stopped_utc=utc_now())
            progress.finish("Local runner stopped")
            return
        if command == "pause":
            write_state(state_dir, status="paused")
            progress.update("Local runner paused", status="paused")
            time.sleep(max(args.poll_seconds, 1))
            continue

        now = time.time()
        if now < next_train_at:
            write_state(state_dir, status="waiting", next_train_epoch=next_train_at)
            time.sleep(max(args.poll_seconds, 1))
            continue

        iteration += 1
        try:
            progress.reset(f"Local runner {args.interval} iteration {iteration}", len(symbols) + 1)
            progress.update("Syncing realtime closed klines")
            payload = runner_once(
                symbols=symbols,
                interval=args.interval,
                limit=args.limit,
                base_url=args.base_url,
                model_suffix=args.model_suffix,
                state_dir=state_dir,
                max_model_trials=args.live_max_model_trials,
                time_budget_minutes=args.live_time_budget_minutes,
                complexity=args.live_complexity,
                rolling_folds=args.live_rolling_folds,
                max_training_rows=args.live_max_training_rows,
                progress=progress,
            )
            best = payload["ranked"][0] if payload["ranked"] else {}
            if best:
                progress.finish(
                    f"Runner iteration complete: best {best['symbol']}",
                    metrics={
                        "model_name": best["model_name"],
                        "latest_up_probability": best["latest_up_probability"],
                        "risk_level": (best.get("latest_risk_decision") or {}).get("risk_level"),
                        "risk_reason": (best.get("latest_risk_decision") or {}).get("reason"),
                        "risk_allow_trade": (best.get("latest_risk_decision") or {}).get("allow_trade"),
                        "total_return": best["backtest"]["total_return"],
                        "max_drawdown": best["backtest"]["max_drawdown"],
                    },
                )
            write_state(state_dir, status="waiting", iteration=iteration, last_error="")
            next_train_at = time.time() + max(args.train_every_seconds, 60)
        except Exception as exc:
            write_state(state_dir, status="error", last_error=str(exc), last_error_utc=utc_now())
            progress.fail(f"Local runner error: {exc}")
            next_train_at = time.time() + max(args.error_sleep_seconds, 60)


def cmd_once(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_proxy(args.proxy or cfg.https_proxy, cfg.auto_detect_proxy)
    payload = runner_once(
        symbols=[item.upper() for item in args.symbols],
        interval=args.interval,
        limit=args.limit,
        base_url=args.base_url,
        model_suffix=args.model_suffix,
        state_dir=Path(args.state_dir),
        max_model_trials=args.live_max_model_trials,
        time_budget_minutes=args.live_time_budget_minutes,
        complexity=args.live_complexity,
        rolling_folds=args.live_rolling_folds,
        max_training_rows=args.live_max_training_rows,
    )
    print(json.dumps(payload, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    payload = read_json(state_path(Path(args.state_dir)), {"status": "not_started"})
    payload["control_command"] = read_control(Path(args.state_dir))
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local resumable realtime training runner")
    parser.add_argument("--config", default="config.default.json")
    parser.add_argument("--state-dir", default="state")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BNBUSDT"])
    run.add_argument("--interval", default="5m")
    run.add_argument("--limit", type=int, default=800)
    run.add_argument("--base-url", default="https://fapi.binance.com")
    run.add_argument("--model-suffix", default="runner_live")
    run.add_argument("--train-every-seconds", type=int, default=900)
    run.add_argument("--poll-seconds", type=int, default=30)
    run.add_argument("--error-sleep-seconds", type=int, default=120)
    run.add_argument("--proxy", default=None)
    run.add_argument("--live-max-model-trials", type=int, default=1)
    run.add_argument("--live-time-budget-minutes", type=float, default=3.0)
    run.add_argument("--live-complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    run.add_argument("--live-rolling-folds", type=int, default=0)
    run.add_argument("--live-max-training-rows", type=int, default=3000)
    run.set_defaults(func=run_loop)

    once = sub.add_parser("once")
    once.add_argument("--symbols", nargs="+", default=["ETHUSDT", "BNBUSDT"])
    once.add_argument("--interval", default="5m")
    once.add_argument("--limit", type=int, default=800)
    once.add_argument("--base-url", default="https://fapi.binance.com")
    once.add_argument("--model-suffix", default="runner_live")
    once.add_argument("--live-max-model-trials", type=int, default=1)
    once.add_argument("--live-time-budget-minutes", type=float, default=3.0)
    once.add_argument("--live-complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    once.add_argument("--live-rolling-folds", type=int, default=0)
    once.add_argument("--live-max-training-rows", type=int, default=3000)
    once.add_argument("--proxy", default=None)
    once.set_defaults(func=cmd_once)

    pause = sub.add_parser("pause")
    pause.set_defaults(func=lambda args: set_control(Path(args.state_dir), "pause"))
    resume = sub.add_parser("resume")
    resume.set_defaults(func=lambda args: set_control(Path(args.state_dir), "run"))
    stop = sub.add_parser("stop")
    stop.set_defaults(func=lambda args: set_control(Path(args.state_dir), "stop"))
    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
