from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import csv
import io
import json
import mimetypes
import os
import socket
import subprocess
import sys
import time
import urllib.parse
from typing import Any

from urllib.error import HTTPError

from .binance_data import (
    binance_public_kline_url,
    data_base_url_candidates,
    download_bytes,
    parse_kline_zip,
    request_json,
)
from .config import load_config
from .desktop_tasks import (
    cancel_task,
    latest_model_optimization,
    list_model_optimization_reports,
    open_workspace_path,
    project_python,
    read_task_log,
    refresh_tasks,
    start_task,
)
from .progress import safe_replace_text
from .runner import control_path, set_control, state_path
from .time_utils import add_beijing_aliases, beijing_now_iso, to_beijing_iso


ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
REPORTS_DIR = ROOT / "reports"
STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"
_DECISION_CHART_CACHE: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}


def _windows_creationflags() -> int:
    if os.name != "nt":
        return 0
    return (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    )


def _detect_local_proxy() -> str:
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


def _apply_dashboard_network_env(cfg: Any) -> str:
    proxy = cfg.https_proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""
    if not proxy and cfg.auto_detect_proxy:
        proxy = _detect_local_proxy()
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy

    data_base_urls = [cfg.data_base_url] if cfg.data_base_url else list(cfg.data_base_urls or ())
    data_base_urls = [item for item in data_base_urls if item]
    if data_base_urls:
        os.environ["BINANCE_DATA_BASE_URLS"] = ";".join(data_base_urls)
        os.environ["BINANCE_DATA_BASE_URL"] = data_base_urls[0]
    return proxy


class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path in {"/", "/dashboard.html"}:
                self._send_file(WEB_DIR / "dashboard.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/config":
                self._send_json(read_ui_config())
                return
            if parsed.path == "/api/app/health":
                self._send_json(read_app_health())
                return
            if parsed.path == "/api/app/paths":
                self._send_json(read_app_paths())
                return
            if parsed.path == "/api/binance/download-check":
                params = urllib.parse.parse_qs(parsed.query)
                self._send_json(read_binance_download_check(params))
                return
            if parsed.path == "/api/runner":
                self._send_json(read_runner_status())
                return
            if parsed.path == "/api/decision-chart":
                params = urllib.parse.parse_qs(parsed.query)
                self._send_json(read_decision_chart(params))
                return
            if parsed.path == "/api/progress":
                self._send_json(read_progress())
                return
            if parsed.path == "/api/reports":
                self._send_json(read_report_index())
                return
            if parsed.path == "/api/tasks":
                params = urllib.parse.parse_qs(parsed.query)
                active_only = _first_query_value(params, "active", "false").lower() in {"1", "true", "yes"}
                tasks = refresh_tasks(STATE_DIR)
                if active_only:
                    tasks = [task for task in tasks if task.get("status") == "running"]
                self._send_json({"ok": True, "items": tasks})
                return
            if parsed.path.startswith("/api/tasks/"):
                self._send_task_get(parsed)
                return
            if parsed.path == "/api/model-optimization/latest":
                self._send_json(latest_model_optimization(REPORTS_DIR))
                return
            if parsed.path == "/api/scheduled-optimization/latest":
                self._send_json(read_latest_report("scheduled_optimization_latest.json", "No scheduled optimization report has been generated yet."))
                return
            if parsed.path == "/api/data-maintenance/latest":
                self._send_json(read_latest_report("data_maintenance_latest.json", "No data maintenance report has been generated yet."))
                return
            if parsed.path == "/api/simulation-memory/latest":
                self._send_json(read_simulation_memory_latest())
                return
            if parsed.path == "/api/monitoring/latest":
                self._send_json(read_monitoring_latest())
                return
            if parsed.path == "/api/portfolio/latest":
                self._send_json(
                    read_latest_report(
                        "portfolio_snapshot_latest.json",
                        "No portfolio planning snapshot has been generated yet.",
                    )
                )
                return
            if parsed.path == "/api/portfolio-paper/latest":
                self._send_json(
                    read_latest_report(
                        "portfolio_paper_latest.json",
                        "No cross-symbol paper portfolio ledger has been generated yet.",
                    )
                )
                return
            if parsed.path == "/api/shadow-portfolio/latest":
                self._send_json(
                    read_latest_report(
                        "shadow_portfolio_snapshot_latest.json",
                        "No shadow learning portfolio snapshot has been generated yet.",
                    )
                )
                return
            if parsed.path == "/api/model-optimization/reports":
                params = urllib.parse.parse_qs(parsed.query)
                try:
                    limit = int(_first_query_value(params, "limit", "20"))
                except ValueError:
                    limit = 20
                self._send_json(list_model_optimization_reports(REPORTS_DIR, limit=limit))
                return
            if parsed.path.startswith("/reports/"):
                rel = parsed.path.removeprefix("/reports/")
                target = (REPORTS_DIR / rel).resolve()
                if REPORTS_DIR.resolve() in target.parents and target.exists() and target.is_file():
                    self._send_file(target, mimetypes.guess_type(target.name)[0] or "text/plain")
                    return
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}", "path": parsed.path}, status=500)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        payload = self._read_json_body()
        try:
            self._require_desktop_token()
            if parsed.path == "/api/runner/start":
                self._send_json(start_runner(payload))
                return
            if parsed.path == "/api/runner/pause":
                self._send_json(update_runner_control("pause"))
                return
            if parsed.path == "/api/runner/resume":
                self._send_json(update_runner_control("run"))
                return
            if parsed.path == "/api/runner/stop":
                self._send_json(update_runner_control("stop"))
                return
            if parsed.path == "/api/tasks/start":
                kind = str(payload.get("kind") or "")
                params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
                self._send_json(start_task(ROOT, STATE_DIR, LOGS_DIR, kind, params))
                return
            if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/cancel"):
                task_id = parsed.path.removeprefix("/api/tasks/").removesuffix("/cancel").strip("/")
                self._send_json(cancel_task(STATE_DIR, task_id, force=bool(payload.get("force"))))
                return
            if parsed.path == "/api/export/portable":
                self._send_json(start_task(ROOT, STATE_DIR, LOGS_DIR, "export-portable", payload))
                return
            if parsed.path == "/api/system/open-path":
                self._send_json(open_workspace_path(ROOT, payload))
                return
        except PermissionError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=403)
            return
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        self.send_error(404)

    def _send_task_get(self, parsed: urllib.parse.ParseResult) -> None:
        path = parsed.path.removeprefix("/api/tasks/").strip("/")
        params = urllib.parse.parse_qs(parsed.query)
        if path.endswith("/logs"):
            task_id = path.removesuffix("/logs").strip("/")
            stream = _first_query_value(params, "stream", "stdout")
            try:
                tail = int(_first_query_value(params, "tail", "200"))
            except ValueError:
                tail = 200
            self._send_json(read_task_log(STATE_DIR, LOGS_DIR, task_id, stream, tail))
            return
        task_id = path
        for task in refresh_tasks(STATE_DIR):
            if task.get("task_id") == task_id:
                self._send_json({"ok": True, "task": task})
                return
        self._send_json({"ok": False, "error": "task_not_found"}, status=404)

    def _require_desktop_token(self) -> None:
        expected = os.environ.get("CRYPTO_AI_DESKTOP_TOKEN", "")
        if not expected:
            return
        actual = self.headers.get("X-Desktop-Token", "")
        if actual != expected:
            raise PermissionError("Missing or invalid desktop token")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 64_000:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return add_beijing_aliases(payload) if isinstance(payload, dict) else payload


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def process_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return False
        if result.returncode != 0:
            return False
        stdout = result.stdout or ""
        rows = list(csv.reader(io.StringIO(stdout.strip())))
        if not rows:
            return False
        if len(rows[0]) < 2:
            return False
        image_name = clean_text(rows[0][0]).lower()
        reported_pid = clean_text(rows[0][1])
        return reported_pid == str(pid) and image_name in {"python.exe", "pythonw.exe"}
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_app_health() -> dict[str, Any]:
    return {
        "ok": True,
        "name": "Crypto AI Trader",
        "root": str(ROOT),
        "python": project_python(ROOT),
        "server_python": sys.executable,
        "process_id": os.getpid(),
        "desktop_token_required": bool(os.environ.get("CRYPTO_AI_DESKTOP_TOKEN")),
        "live_trading_enabled": False,
    }


def read_app_paths() -> dict[str, Any]:
    paths = {
        "root": ROOT,
        "data": ROOT / "data",
        "reports": REPORTS_DIR,
        "models": ROOT / "models",
        "logs": LOGS_DIR,
        "state": STATE_DIR,
        "exports": ROOT / "exports",
    }
    return {"ok": True, "paths": {key: str(path) for key, path in paths.items()}}


def _error_summary(exc: Exception) -> str:
    text = str(exc)
    if len(text) > 280:
        text = text[:277] + "..."
    return f"{type(exc).__name__}: {text}"


def read_binance_download_check(params: dict[str, list[str]]) -> dict[str, Any]:
    cfg = load_config(ROOT / "config.default.json")
    proxy = _apply_dashboard_network_env(cfg)
    symbol = clean_text(_first_query_value(params, "symbol", "BNBUSDT"), "BNBUSDT").upper()
    if symbol not in {item.upper() for item in cfg.symbols}:
        symbol = "BNBUSDT"
    archive_interval = clean_text(_first_query_value(params, "interval", "1h"), "1h") or "1h"
    if archive_interval not in {"1m", "3m", "5m", "15m", "30m", "1h"}:
        archive_interval = "1h"
    period = clean_text(_first_query_value(params, "period", "2024-01"), "2024-01") or "2024-01"
    realtime_interval = clean_text(cfg.realtime_interval, "5m") or "5m"

    source_text = cfg.data_base_url or ";".join(cfg.data_base_urls or ())
    archive_checks: list[dict[str, Any]] = []
    archive_ok = False
    archive_rows = 0
    archive_bytes = 0
    for candidate in data_base_url_candidates(source_text):
        url = binance_public_kline_url(
            symbol,
            archive_interval,
            period,
            market=cfg.market,
            frequency="monthly",
            base_url=candidate,
        )
        started = time.time()
        item: dict[str, Any] = {"url": url, "source": candidate or "default"}
        try:
            payload = download_bytes(url, retries=1, timeout=12)
            frame = parse_kline_zip(payload)
            archive_rows = int(len(frame))
            archive_bytes = int(len(payload))
            item.update(
                {
                    "ok": True,
                    "elapsed_ms": round((time.time() - started) * 1000, 2),
                    "bytes": archive_bytes,
                    "rows": archive_rows,
                }
            )
            archive_checks.append(item)
            archive_ok = True
            break
        except HTTPError as exc:
            item.update(
                {
                    "ok": False,
                    "status": exc.code,
                    "elapsed_ms": round((time.time() - started) * 1000, 2),
                    "error": _error_summary(exc),
                }
            )
        except Exception as exc:
            item.update(
                {
                    "ok": False,
                    "elapsed_ms": round((time.time() - started) * 1000, 2),
                    "error": _error_summary(exc),
                }
            )
        archive_checks.append(item)

    realtime_base_url = clean_text(cfg.realtime_base_url, "https://fapi.binance.com") or "https://fapi.binance.com"
    rest_url = f"{realtime_base_url.rstrip('/')}/fapi/v1/klines"
    rest_started = time.time()
    rest_check: dict[str, Any] = {
        "url": rest_url,
        "params": {"symbol": symbol, "interval": realtime_interval, "limit": 2},
    }
    try:
        payload = request_json(
            rest_url,
            {"symbol": symbol, "interval": realtime_interval, "limit": 2},
            retries=1,
            timeout=10,
        )
        rest_rows = len(payload) if isinstance(payload, list) else 0
        rest_check.update(
            {
                "ok": rest_rows > 0,
                "rows": rest_rows,
                "elapsed_ms": round((time.time() - rest_started) * 1000, 2),
            }
        )
    except Exception as exc:
        rest_check.update(
            {
                "ok": False,
                "elapsed_ms": round((time.time() - rest_started) * 1000, 2),
                "error": _error_summary(exc),
            }
        )

    can_download = archive_ok or bool(rest_check.get("ok"))
    message = "可以从 Binance 下载数据。" if can_download else "当前不能从 Binance 下载数据，请检查 VPN、代理、防火墙或 data_base_urls 镜像。"
    if archive_ok:
        message = f"历史归档 K 线可下载：{symbol} {archive_interval} {period}，{archive_rows} 行。"
    elif rest_check.get("ok"):
        message = f"实时 REST K 线可下载：{symbol} {realtime_interval}。历史归档源暂不可用。"

    payload: dict[str, Any] = {
        "ok": True,
        "can_download": can_download,
        "historical_archive_ok": archive_ok,
        "realtime_rest_ok": bool(rest_check.get("ok")),
        "message": message,
        "created_beijing": beijing_now_iso(),
        "symbol": symbol,
        "archive_interval": archive_interval,
        "realtime_interval": realtime_interval,
        "period": period,
        "proxy": proxy,
        "data_base_urls": [item for item in data_base_url_candidates(source_text) if item],
        "archive_bytes": archive_bytes,
        "archive_rows": archive_rows,
        "archive_checks": archive_checks,
        "rest_check": rest_check,
        "live_trading_enabled": False,
    }
    write_json(REPORTS_DIR / "binance_download_check_latest.json", payload)
    return payload


def read_ui_config() -> dict[str, Any]:
    cfg = load_config()
    return {
        "symbols": cfg.symbols,
        "default_runner_symbols": ["ETHUSDT", "BNBUSDT"],
        "intervals": ["1m", "3m", "5m", "15m", "30m", "1h"],
        "default_interval": cfg.realtime_interval,
        "default_limit": min(cfg.realtime_limit, 800),
        "default_train_every_seconds": 900,
        "runner_max_model_trials": cfg.runner_max_model_trials,
        "runner_time_budget_minutes": cfg.runner_time_budget_minutes,
        "runner_rolling_folds": cfg.runner_rolling_folds,
        "runner_max_training_rows": cfg.runner_max_training_rows,
        "shadow_learning_enabled": cfg.shadow_learning_enabled,
        "shadow_max_position_fraction": (
            cfg.shadow_max_position_fraction
        ),
        "live_trading_enabled": False,
    }


def read_runner_status() -> dict[str, Any]:
    payload = read_json(state_path(STATE_DIR))
    if not payload:
        payload = {"status": "not_started"}
    control = read_json(control_path(STATE_DIR)) or {"command": "run"}
    pid = payload.get("pid")
    try:
        pid_int = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid_int = None
    running = process_is_running(pid_int)
    status = str(payload.get("status", "not_started"))
    effective_status = status
    if status not in {"not_started", "stopped"} and not running:
        effective_status = "offline"
    return {
        **add_beijing_aliases(payload),
        "control_command": control.get("command", "run"),
        "process_running": running,
        "effective_status": effective_status,
        "live_trading_enabled": False,
    }


def _first_query_value(params: dict[str, list[str]], key: str, default: str = "") -> str:
    values = params.get(key) or []
    value = values[0] if values else default
    return default if value is None else str(value)


def _resolve_workspace_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _choose_chart_item(runner: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    items = runner.get("items") or []
    ranked = runner.get("ranked") or []
    items = [item for item in items if isinstance(item, dict)]
    ranked = [item for item in ranked if isinstance(item, dict)]
    candidates = items or ranked
    if not candidates:
        return None
    if symbol and symbol.lower() != "auto":
        symbol = symbol.upper()
        for item in candidates:
            if str(item.get("symbol", "")).upper() == symbol:
                return item
    return ranked[0] if ranked else candidates[0]


def _signal_for_position(position: int, previous: int) -> str:
    if position == 1 and previous <= 0:
        return "buy"
    if position == -1 and previous >= 0:
        return "sell"
    if position == 0 and previous != 0:
        return "exit"
    return ""


def read_decision_chart(params: dict[str, list[str]]) -> dict[str, Any]:
    cfg = load_config()
    runner_path = REPORTS_DIR / "runner_live_latest.json"
    runner = as_dict(read_json(runner_path))
    symbol = clean_text(_first_query_value(params, "symbol", "auto"), "auto")
    try:
        points = int(_first_query_value(params, "points", "180"))
    except ValueError:
        points = 180
    points = max(40, min(points, 500))

    item = _choose_chart_item(runner, symbol)
    if not item:
        return {
            "ok": True,
            "empty": True,
            "message": "暂无 runner 决策报告。请先在控制台点击开始。",
            "points": [],
            "signals": [],
        }

    item_symbol = clean_text(item.get("symbol")).upper()
    interval = clean_text(item.get("interval") or runner.get("interval") or cfg.realtime_interval, "5m")
    detail_path = _resolve_workspace_path(clean_text(item.get("detail_path")))
    if detail_path is None or not detail_path.exists():
        fallback = REPORTS_DIR / f"{item_symbol}_{interval}_runner_live_backtest.csv"
        detail_path = fallback.resolve()

    reports_root = REPORTS_DIR.resolve()
    if reports_root not in detail_path.parents or not detail_path.exists():
        return {
            "ok": True,
            "empty": True,
            "symbol": item_symbol,
            "interval": interval,
            "message": f"未找到 {item_symbol} {interval} 的决策明细 CSV。",
            "points": [],
            "signals": [],
        }

    runner_mtime = runner_path.stat().st_mtime_ns if runner_path.exists() else 0
    detail_mtime = detail_path.stat().st_mtime_ns
    cache_key = (item_symbol, points, str(detail_path), detail_mtime, runner_mtime)
    cached = _DECISION_CHART_CACHE.get(cache_key)
    if cached is not None:
        return cached

    import pandas as pd

    data = pd.read_csv(detail_path).tail(points).reset_index(drop=True)
    if data.empty:
        return {
            "ok": True,
            "empty": True,
            "symbol": item_symbol,
            "interval": interval,
            "message": "决策明细为空。",
            "points": [],
            "signals": [],
        }

    chart_points: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    previous_position = 0
    for idx, row in data.iterrows():
        position = int(row.get("position", 0))
        signal = _signal_for_position(position, previous_position)
        point_time = to_beijing_iso(row.get("open_datetime", ""))
        raw_risk_allow = row.get("risk_allow_trade")
        raw_risk_level = row.get("risk_level")
        raw_risk_size = row.get("risk_max_position_size")
        raw_risk_reason = row.get("risk_reason")
        point = {
            "index": int(idx),
            "time": point_time,
            "time_beijing": point_time,
            "close": float(row.get("close", 0.0)),
            "prob_up": float(row.get("prob_up", 0.0)),
            "position": position,
            "equity": float(row.get("equity", 0.0)),
            "signal": signal,
            "decision_reason_code": (
                None
                if pd.isna(row.get("decision_reason_code"))
                else str(row.get("decision_reason_code"))
            ),
            "risk_allow_trade": (
                None if pd.isna(raw_risk_allow) else bool(raw_risk_allow)
            ),
            "risk_level": (
                None if pd.isna(raw_risk_level) else str(raw_risk_level)
            ),
            "risk_max_position_size": (
                None if pd.isna(raw_risk_size) else float(raw_risk_size)
            ),
            "risk_reason": (
                None if pd.isna(raw_risk_reason) else str(raw_risk_reason)
            ),
            "execution_available": (
                None
                if pd.isna(row.get("execution_available"))
                else bool(row.get("execution_available"))
            ),
            "execution_reason_code": (
                None
                if pd.isna(row.get("execution_reason_code"))
                else str(row.get("execution_reason_code"))
            ),
        }
        chart_points.append(point)
        if signal:
            signals.append(point)
        previous_position = position

    latest = chart_points[-1]
    decision = "观望"
    if latest["position"] > 0:
        decision = "做多/持多"
    elif latest["position"] < 0:
        decision = "做空/持空"
    optimized_cfg = item.get("optimized_backtest_config") if isinstance(item.get("optimized_backtest_config"), dict) else {}
    long_threshold = float(optimized_cfg.get("long_threshold", cfg.long_threshold))
    short_threshold = float(optimized_cfg.get("short_threshold", cfg.short_threshold))
    latest_risk_decision = (
        item.get("latest_risk_decision")
        if isinstance(item.get("latest_risk_decision"), dict)
        else None
    )
    if latest_risk_decision is None and latest.get("risk_reason"):
        latest_risk_decision = {
            "allow_trade": latest.get("risk_allow_trade"),
            "risk_level": latest.get("risk_level"),
            "max_position_size": latest.get("risk_max_position_size"),
            "reason": latest.get("risk_reason"),
        }

    response = {
        "ok": True,
        "empty": False,
        "symbol": item_symbol,
        "interval": interval,
        "model_name": item.get("model_name", ""),
        "created_utc": item.get("created_utc") or runner.get("created_utc", ""),
        "created_beijing": item.get("created_beijing") or runner.get("created_beijing") or to_beijing_iso(item.get("created_utc") or runner.get("created_utc", "")),
        "latest_datetime": item.get("latest_datetime") or latest["time"],
        "latest_datetime_beijing": to_beijing_iso(item.get("latest_datetime") or latest["time"]),
        "latest_close": item.get("latest_close", latest["close"]),
        "latest_up_probability": item.get("latest_up_probability", latest["prob_up"]),
        "latest_horizon_probabilities": item.get("latest_horizon_probabilities", {}),
        "recent_horizon_matches": item.get("recent_horizon_matches", {}),
        "latest_risk_decision": latest_risk_decision,
        "latest_execution_decision": item.get("latest_execution_decision"),
        "decision": decision,
        "thresholds": {
            "long": long_threshold,
            "short": short_threshold,
            "side_policy": optimized_cfg.get("trade_side_policy", "both"),
        },
        "backtest": item.get("backtest", {}),
        "detail_path": str(detail_path),
        "points": chart_points,
        "signals": signals[-80:],
    }
    _DECISION_CHART_CACHE[cache_key] = response
    if len(_DECISION_CHART_CACHE) > 12:
        oldest_key = next(iter(_DECISION_CHART_CACHE))
        _DECISION_CHART_CACHE.pop(oldest_key, None)
    return response


def _validated_runner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    config = read_ui_config()
    configured_symbols = config.get("symbols") or []
    if not isinstance(configured_symbols, (list, tuple, set)):
        configured_symbols = [configured_symbols]
    valid_symbols = {clean_text(item).upper() for item in configured_symbols if clean_text(item)}

    raw_symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if raw_symbols is None:
        raw_symbols = []
    elif not isinstance(raw_symbols, (list, tuple, set)):
        raw_symbols = [raw_symbols]
    symbols = [clean_text(item).upper() for item in raw_symbols if clean_text(item)]
    symbols = [item for item in symbols if item in valid_symbols]
    if not symbols:
        symbols = list(config.get("default_runner_symbols") or ["ETHUSDT", "BNBUSDT"])

    interval = str(payload.get("interval") or config.get("default_interval") or "5m")
    valid_intervals = config.get("intervals") or ["1m", "3m", "5m", "15m", "30m", "1h"]
    if interval not in valid_intervals:
        interval = config.get("default_interval") or "5m"

    def clamp_int(value: object, default: int, low: int, high: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(high, parsed))

    return {
        "symbols": symbols,
        "interval": interval,
        "limit": clamp_int(payload.get("limit"), config.get("default_limit", 800), 100, 2000),
        "train_every_seconds": clamp_int(payload.get("train_every_seconds"), config.get("default_train_every_seconds", 900), 60, 86_400),
        "live_max_model_trials": clamp_int(
            payload.get("live_max_model_trials"),
            config.get("runner_max_model_trials", 4),
            1,
            32,
        ),
        "live_time_budget_minutes": clamp_int(
            payload.get("live_time_budget_minutes"),
            int(config.get("runner_time_budget_minutes", 6)),
            1,
            60,
        ),
        "live_rolling_folds": clamp_int(
            payload.get("live_rolling_folds"),
            config.get("runner_rolling_folds", 1),
            0,
            5,
        ),
        "live_max_training_rows": clamp_int(
            payload.get("live_max_training_rows"),
            config.get("runner_max_training_rows", 8_000),
            3_000,
            250_000,
        ),
    }


def start_runner(payload: dict[str, Any]) -> dict[str, Any]:
    status = read_runner_status()
    if status.get("process_running"):
        set_control(STATE_DIR, "run")
        return {"ok": True, "message": "Runner is already running", "status": read_runner_status()}

    params = _validated_runner_payload(payload)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOGS_DIR / "runner.gui.out.log"
    err_path = LOGS_DIR / "runner.gui.err.log"
    command = [
        project_python(ROOT),
        "-m",
        "crypto_ai_trader.runner",
        "run",
        "--symbols",
        *params["symbols"],
        "--interval",
        params["interval"],
        "--limit",
        str(params["limit"]),
        "--train-every-seconds",
        str(params["train_every_seconds"]),
        "--live-max-model-trials",
        str(params["live_max_model_trials"]),
        "--live-time-budget-minutes",
        str(params["live_time_budget_minutes"]),
        "--live-rolling-folds",
        str(params["live_rolling_folds"]),
        "--live-max-training-rows",
        str(params["live_max_training_rows"]),
    ]
    set_control(STATE_DIR, "run")
    with out_path.open("ab") as out_log, err_path.open("ab") as err_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=out_log,
            stderr=err_log,
            creationflags=_windows_creationflags(),
        )
    return {
        "ok": True,
        "message": "Runner started",
        "pid": process.pid,
        "params": params,
        "logs": {"stdout": str(out_path), "stderr": str(err_path)},
    }


def update_runner_control(command: str) -> dict[str, Any]:
    if command not in {"pause", "run", "stop"}:
        raise ValueError(f"Unsupported runner command: {command}")
    set_control(STATE_DIR, command)
    return {"ok": True, "command": command, "status": read_runner_status()}


def read_progress() -> dict:
    payload = read_json(REPORTS_DIR / "progress.json")
    if payload:
        add_beijing_aliases(payload)
        return payload
    return {
        "title": "Crypto AI Trader",
        "status": "idle",
        "message": "waiting for local run",
        "current_symbol": "",
        "current_step": 0,
        "total_steps": 1,
        "percent": 0,
        "metrics": {},
        "events": [],
        "updated_utc": "",
    }


def read_report_index() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    training = []
    for path in sorted(REPORTS_DIR.glob("*_train_metrics.json"))[-20:]:
        payload = as_dict(read_json(path))
        metrics = as_dict(payload.get("metrics"))
        training.append(
            {
                "file": path.name,
                "symbol": clean_text(payload.get("symbol")),
                "interval": clean_text(payload.get("interval")),
                "model": clean_text(payload.get("model_name")),
                "rows": payload.get("rows", 0),
                "accuracy": metrics.get("test_accuracy", metrics.get("accuracy", 0)),
                "log_loss": metrics.get("test_log_loss", metrics.get("log_loss", 0)),
            }
        )

    backtests = []
    for path in sorted(REPORTS_DIR.glob("*_backtest_summary.json"))[-20:]:
        payload = as_dict(read_json(path))
        backtests.append(
            {
                "file": path.name,
                "name": path.name.replace("_backtest_summary.json", ""),
                "return": payload.get("total_return", 0),
                "max_drawdown": payload.get("max_drawdown", 0),
                "trades": payload.get("trades", 0),
                "win_rate": payload.get("win_rate", 0),
                "profit_factor": payload.get("profit_factor", 0),
                "detail_path": clean_text(payload.get("detail_path")),
            }
        )
    return {"training": training, "backtests": backtests}


def read_latest_report(file_name: str, empty_message: str) -> dict[str, Any]:
    path = REPORTS_DIR / file_name
    payload = as_dict(read_json(path))
    if not payload:
        return {
            "ok": True,
            "empty": True,
            "message": empty_message,
            "live_trading_enabled": False,
        }
    return {
        "ok": True,
        "empty": False,
        "file": path.name,
        "path": str(path),
        "created_beijing": payload.get("created_beijing") or payload.get("updated_beijing") or "",
        "payload": payload,
        "live_trading_enabled": bool(as_dict(payload.get("safety")).get("live_trading_enabled", False)),
    }


def read_simulation_memory_latest() -> dict[str, Any]:
    path = REPORTS_DIR / "simulation_memory_latest.json"
    payload = as_dict(read_json(path))
    if not payload:
        return {
            "ok": True,
            "empty": True,
            "message": "No simulation memory has been generated yet.",
            "live_trading_enabled": False,
        }
    global_summary = as_dict(payload.get("global"))
    return {
        "ok": True,
        "empty": False,
        "file": path.name,
        "path": str(path),
        "updated_beijing": payload.get("updated_beijing", ""),
        "observation_count": payload.get("observation_count", 0),
        "target_count": payload.get("target_count", 0),
        "top_candidates": global_summary.get("top_candidates", []),
        "weak_targets": global_summary.get("weak_targets", []),
        "next_actions": global_summary.get("next_actions", []),
        "monitoring_alerts": global_summary.get("monitoring_alerts", []),
        "live_trading_enabled": bool(as_dict(payload.get("safety")).get("live_trading_enabled", False)),
    }


def read_monitoring_latest() -> dict[str, Any]:
    """Return the latest per-target monitoring snapshots."""

    items: list[dict[str, Any]] = []
    for path in REPORTS_DIR.glob("*_*_monitoring.json"):
        payload = as_dict(read_json(path))
        if not payload:
            continue
        retraining = as_dict(payload.get("retraining"))
        drift = as_dict(payload.get("feature_drift"))
        calibration = as_dict(payload.get("calibration"))
        rolling = as_dict(payload.get("rolling_performance"))
        items.append(
            {
                "symbol": clean_text(payload.get("symbol")).upper(),
                "interval": clean_text(payload.get("interval")),
                "model_name": clean_text(payload.get("model_name")),
                "created_beijing": clean_text(payload.get("created_beijing")),
                "retraining": retraining,
                "feature_drift": drift,
                "calibration": calibration,
                "rolling_performance": rolling,
                "report_path": str(path),
                "modified_ns": path.stat().st_mtime_ns,
            }
        )
    items.sort(
        key=lambda item: (
            as_dict(item.get("retraining")).get("severity") == "high",
            bool(as_dict(item.get("retraining")).get("triggered")),
            int(item.get("modified_ns", 0)),
        ),
        reverse=True,
    )
    triggered = [
        item
        for item in items
        if bool(as_dict(item.get("retraining")).get("active"))
        and not bool(as_dict(item.get("retraining")).get("acknowledged"))
    ]
    acknowledged = [
        item
        for item in items
        if bool(as_dict(item.get("retraining")).get("acknowledged"))
    ]
    for item in items:
        item.pop("modified_ns", None)
    return {
        "ok": True,
        "empty": not items,
        "items": items,
        "triggered_count": len(triggered),
        "acknowledged_count": len(acknowledged),
        "high_severity_count": sum(
            as_dict(item.get("retraining")).get("severity") == "high"
            for item in triggered
        ),
        "live_trading_enabled": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local dashboard server for Crypto AI Trader")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
