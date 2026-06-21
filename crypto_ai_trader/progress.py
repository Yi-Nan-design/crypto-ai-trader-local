from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import time
from typing import Any

from .time_utils import beijing_now_iso


def now_text() -> str:
    return beijing_now_iso()


def safe_replace_text(path: Path, text: str, *, encoding: str = "utf-8", attempts: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{int(time.time() * 1000)}.tmp")
    tmp.write_text(text, encoding=encoding)
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error


@dataclass
class ProgressTracker:
    path: Path
    title: str = "Crypto AI Trader"
    total_steps: int = 1
    current_step: int = 0
    status: str = "idle"
    message: str = ""
    current_symbol: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def reset(self, title: str, total_steps: int) -> None:
        self.title = title
        self.total_steps = max(int(total_steps), 1)
        self.current_step = 0
        self.status = "running"
        self.message = "starting"
        self.current_symbol = ""
        self.metrics = {}
        self.events = []
        self.add_event("start", title)
        self.write()

    def update(
        self,
        message: str,
        *,
        status: str = "running",
        step: int | None = None,
        current_symbol: str | None = None,
        metrics: dict[str, Any] | None = None,
        event_type: str = "progress",
    ) -> None:
        if step is not None:
            self.current_step = max(0, min(int(step), self.total_steps))
        self.status = status
        self.message = message
        if current_symbol is not None:
            self.current_symbol = current_symbol
        if metrics:
            self.metrics.update(metrics)
        self.add_event(event_type, message, current_symbol=current_symbol, metrics=metrics)
        self.write()

    def advance(
        self,
        message: str,
        *,
        current_symbol: str | None = None,
        metrics: dict[str, Any] | None = None,
        event_type: str = "progress",
    ) -> None:
        self.update(
            message,
            step=self.current_step + 1,
            current_symbol=current_symbol,
            metrics=metrics,
            event_type=event_type,
        )

    def finish(self, message: str = "finished", metrics: dict[str, Any] | None = None) -> None:
        self.update(message, status="finished", step=self.total_steps, metrics=metrics, event_type="finish")

    def fail(self, message: str) -> None:
        self.update(message, status="failed", event_type="error")

    def add_event(
        self,
        event_type: str,
        message: str,
        *,
        current_symbol: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "time": now_text(),
                "time_beijing": now_text(),
                "type": event_type,
                "message": message,
                "symbol": current_symbol if current_symbol is not None else self.current_symbol,
                "metrics": metrics or {},
            }
        )
        self.events = self.events[-120:]

    def payload(self) -> dict[str, Any]:
        percent = round(self.current_step / max(self.total_steps, 1) * 100, 2)
        return {
            "title": self.title,
            "status": self.status,
            "message": self.message,
            "current_symbol": self.current_symbol,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "percent": percent,
            "metrics": self.metrics,
            "events": self.events,
            "updated_utc": now_text(),
            "updated_beijing": now_text(),
        }

    def write(self) -> None:
        payload = self.payload()
        safe_replace_text(self.path, json.dumps(payload, indent=2))
        write_static_dashboard(self.path.parent / "dashboard_live.html", payload)


def tracker_for_reports(reports_dir: str | Path) -> ProgressTracker:
    return ProgressTracker(Path(reports_dir) / "progress.json")


def write_static_dashboard(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    events = list(reversed(payload.get("events", [])[-50:]))
    total_return = metrics.get("total_return")
    max_drawdown = metrics.get("max_drawdown")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="2">
  <title>Crypto AI Trader Live</title>
  <style>
    body {{ margin:0; background:#f7f8fa; color:#18202a; font-family:Segoe UI, Arial, sans-serif; letter-spacing:0; }}
    header {{ background:#101820; color:white; padding:18px 28px; border-bottom:4px solid #0f8b8d; display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
    h1 {{ margin:0; font-size:24px; }}
    main {{ width:min(1180px, calc(100vw - 28px)); margin:22px auto; display:grid; gap:16px; }}
    section {{ background:white; border:1px solid #d9dee7; border-radius:8px; box-shadow:0 12px 30px rgba(24,32,42,.08); padding:18px; }}
    .top {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; }}
    .title {{ font-size:20px; font-weight:700; margin-bottom:6px; }}
    .muted {{ color:#667085; font-size:14px; }}
    .percent {{ color:#0f8b8d; font-size:34px; font-weight:800; text-align:right; }}
    .bar {{ height:14px; background:#e9edf3; border-radius:999px; overflow:hidden; border:1px solid #d9dee7; margin-top:14px; }}
    .fill {{ height:100%; width:{payload.get("percent", 0)}%; background:linear-gradient(90deg,#0f8b8d,#2f6fed,#e0a21a); }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:16px; }}
    .stat {{ border:1px solid #d9dee7; border-radius:8px; padding:13px; background:#fbfcfe; min-height:76px; }}
    .stat label {{ display:block; color:#667085; font-size:12px; margin-bottom:8px; }}
    .stat strong {{ display:block; font-size:20px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .event {{ border-left:4px solid #0f8b8d; padding:9px 10px; background:#fbfcfe; border-radius:6px; margin-bottom:10px; border-top:1px solid #d9dee7; border-right:1px solid #d9dee7; border-bottom:1px solid #d9dee7; }}
    .error {{ border-left-color:#c43d4b; }}
    .finish {{ border-left-color:#1d8a4d; }}
    .time {{ color:#667085; font-size:12px; margin-bottom:4px; }}
    .positive {{ color:#1d8a4d; }}
    .negative {{ color:#c43d4b; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ text-align:left; padding:10px 8px; border-bottom:1px solid #d9dee7; }}
    th {{ color:#667085; background:#fbfcfe; }}
    @media (max-width:800px) {{ .stats,.grid {{ grid-template-columns:1fr; }} .top {{ flex-direction:column; }} .percent {{ text-align:left; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Crypto AI Trader</h1>
    <div>{escape(payload.get("status", "idle"))} | {escape(payload.get("updated_beijing") or payload.get("updated_utc", ""))}</div>
  </header>
  <main>
    <section>
      <div class="top">
        <div>
          <div class="title">{escape(payload.get("title", ""))}</div>
          <div class="muted">{escape(payload.get("message", ""))}</div>
        </div>
        <div class="percent">{payload.get("percent", 0):.0f}%</div>
      </div>
      <div class="bar"><div class="fill"></div></div>
      <div class="stats">
        <div class="stat"><label>Symbol</label><strong>{escape(payload.get("current_symbol", "") or "-")}</strong></div>
        <div class="stat"><label>Step</label><strong>{payload.get("current_step", 0)} / {payload.get("total_steps", 1)}</strong></div>
        <div class="stat"><label>Model</label><strong>{escape(metrics.get("model_name", "-"))}</strong></div>
        <div class="stat"><label>Return / Drawdown</label><strong>{format_return_pair(total_return, max_drawdown)}</strong></div>
      </div>
    </section>
    <div class="grid">
      <section>
        <h2>Recent Events</h2>
        {render_events(events)}
      </section>
      <section>
        <h2>Metrics</h2>
        {render_metrics(metrics)}
      </section>
    </div>
  </main>
</body>
</html>
"""
    safe_replace_text(path, html)


def escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def format_return_pair(total_return: object, max_drawdown: object) -> str:
    if total_return is None:
        return "-"
    ret_class = "positive" if float(total_return) >= 0 else "negative"
    dd_class = "positive" if float(max_drawdown or 0) >= 0 else "negative"
    return f'<span class="{ret_class}">{format_pct(total_return)}</span> / <span class="{dd_class}">{format_pct(max_drawdown)}</span>'


def render_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<div class="muted">No events yet</div>'
    rows = []
    for item in events:
        cls = "event " + escape(item.get("type", ""))
        symbol = f" | {escape(item.get('symbol', ''))}" if item.get("symbol") else ""
        rows.append(
            f'<div class="{cls}"><div class="time">{escape(item.get("time_beijing") or item.get("time", ""))}{symbol}</div>'
            f'<div>{escape(item.get("message", ""))}</div></div>'
        )
    return "".join(rows)


def render_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<div class="muted">No metrics yet</div>'
    rows = []
    for key, value in sorted(metrics.items()):
        rows.append(f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>")
    return f"<table>{''.join(rows)}</table>"
