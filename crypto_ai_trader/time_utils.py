from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def beijing_now_iso() -> str:
    return beijing_now().isoformat(timespec="seconds")


def beijing_stamp() -> str:
    return beijing_now().strftime("%Y%m%d_%H%M%S")


def beijing_task_stamp() -> str:
    return beijing_now().strftime("%Y%m%d%H%M%S")


def to_beijing_iso(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).astimezone(BEIJING_TZ).isoformat(timespec="seconds")
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(BEIJING_TZ).isoformat(timespec="seconds")


def add_beijing_aliases(payload: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    aliases = {
        "created_utc": "created_beijing",
        "updated_utc": "updated_beijing",
        "started_utc": "started_beijing",
        "ended_utc": "ended_beijing",
        "last_run_utc": "last_run_beijing",
        "stopped_utc": "stopped_beijing",
        "last_error_utc": "last_error_beijing",
        "last_review_utc": "last_review_beijing",
        "last_optimization_utc": "last_optimization_beijing",
    }
    for source, target in aliases.items():
        if source in payload and (overwrite or target not in payload):
            payload[target] = to_beijing_iso(payload.get(source))
    return payload
