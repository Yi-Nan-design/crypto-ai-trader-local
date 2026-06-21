from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import shutil
import time

import pandas as pd

from .binance_data import load_klines, save_klines
from .config import load_config
from .progress import safe_replace_text
from .time_utils import beijing_now_iso, beijing_stamp


def report_path(path: Path, root: Path | None = None) -> str:
    """Return stable project-relative paths for reports, avoiding Windows codepage noise."""
    try:
        if root is not None:
            return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        pass
    return str(path)


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_replace_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def cleanup_tmp_files(root: Path, *, older_than_hours: float, dry_run: bool) -> list[dict[str, Any]]:
    cutoff = time.time() - max(0.0, older_than_hours) * 3600.0
    removed: list[dict[str, Any]] = []
    for base in [root / "reports", root / "state", root / "logs"]:
        if not base.exists():
            continue
        for path in base.rglob("*.tmp"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_mtime > cutoff:
                continue
            item = {
                "path": report_path(path, root),
                "bytes": stat.st_size,
                "action": "would_delete" if dry_run else "deleted",
            }
            if not dry_run:
                try:
                    path.unlink()
                except OSError as exc:
                    item["action"] = "failed"
                    item["error"] = str(exc)
            removed.append(item)
    return removed


def realtime_dirs(data_dir: Path) -> list[Path]:
    root = data_dir / "realtime"
    if not root.exists():
        return []
    return [path for path in root.glob("*/*") if path.is_dir()]


def compact_realtime_dir(
    root: Path,
    interval_dir: Path,
    *,
    dry_run: bool,
    archive_old: bool,
    max_rows: int = 6000,
) -> dict[str, Any]:
    symbol = interval_dir.parent.name.upper()
    interval = interval_dir.name
    output = interval_dir / f"{symbol}-{interval}-realtime_compacted.csv"
    source_files = sorted(list(interval_dir.glob("*.csv")) + list(interval_dir.glob("*.parquet")))
    source_files = [path for path in source_files if path.name != output.name]
    files = list(source_files)
    included_existing_compacted = False
    if output.exists():
        files.append(output)
        included_existing_compacted = True
    result: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "source_files": len(source_files),
        "included_existing_compacted": included_existing_compacted,
        "action": "skipped",
        "rows_before": 0,
        "rows_deduplicated": 0,
        "rows_after": 0,
        "rows_trimmed_as_redundant": 0,
        "max_realtime_rows": max_rows,
        "bytes_before": sum(path.stat().st_size for path in files if path.exists()),
    }
    if not files:
        return result

    frames = []
    for path in files:
        try:
            frames.append(load_klines(path))
        except Exception as exc:
            result.setdefault("errors", []).append({"path": report_path(path, root), "error": str(exc)})
    if not frames:
        result["action"] = "failed"
        return result

    merged = pd.concat(frames, ignore_index=True)
    result["rows_before"] = int(len(merged))
    merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    result["rows_deduplicated"] = int(len(merged))
    if max_rows > 0 and len(merged) > max_rows:
        result["rows_trimmed_as_redundant"] = int(len(merged) - max_rows)
        merged = merged.tail(max_rows).reset_index(drop=True)
    result["rows_after"] = int(len(merged))
    result["output"] = report_path(output, root)
    if not merged.empty:
        result["retained_first_open_time"] = int(merged["open_time"].iloc[0])
        result["retained_last_open_time"] = int(merged["open_time"].iloc[-1])
    should_compact = bool(source_files) or result["rows_before"] != result["rows_after"] or not output.exists()
    if not should_compact:
        result["action"] = "already_compact"
        return result
    if dry_run:
        result["action"] = "would_compact"
        return result

    saved = save_klines(merged, output)
    validation = load_klines(saved)
    if len(validation) != len(merged):
        result["action"] = "failed"
        result["error"] = "validation_row_count_mismatch"
        return result

    moved = []
    if archive_old:
        archive_dir = root / "data" / "maintenance_archive" / "realtime" / symbol / interval / beijing_stamp()
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in source_files:
            if path.resolve() == saved.resolve():
                continue
            destination = archive_dir / path.name
            try:
                shutil.move(str(path), str(destination))
                moved.append(report_path(destination, root))
            except OSError as exc:
                result.setdefault("errors", []).append({"path": report_path(path, root), "error": str(exc)})
    result["action"] = "compacted"
    result["archived_files"] = moved
    result["bytes_after"] = saved.stat().st_size if saved.exists() else 0
    return result


def run_data_maintenance(
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
    archive_old_realtime: bool = True,
    cleanup_tmp_older_than_hours: float = 6.0,
    max_realtime_rows: int = 6000,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(".").resolve()
    data_dir = cfg.data_dir
    before_bytes = directory_size(data_dir)
    realtime_results = [
        compact_realtime_dir(root, path, dry_run=dry_run, archive_old=archive_old_realtime, max_rows=max_realtime_rows)
        for path in realtime_dirs(data_dir)
    ]
    tmp_results = cleanup_tmp_files(root, older_than_hours=cleanup_tmp_older_than_hours, dry_run=dry_run)
    after_bytes = directory_size(data_dir)
    payload = {
        "created_beijing": beijing_now_iso(),
        "dry_run": dry_run,
        "max_realtime_rows": max_realtime_rows,
        "safety": {
            "raw_historical_data_deleted": False,
            "archive_cache_deleted": False,
            "live_trading_enabled": bool(cfg.live_trading_enabled),
            "real_orders_allowed": False,
        },
        "data_dir": str(data_dir),
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "realtime_compaction": realtime_results,
        "tmp_cleanup": tmp_results,
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    latest = cfg.reports_dir / "data_maintenance_latest.json"
    stamped = cfg.reports_dir / f"data_maintenance_{beijing_stamp()}.json"
    safe_write_json(latest, payload)
    safe_write_json(stamped, payload)
    payload["report_path"] = str(latest)
    payload["stamped_report_path"] = str(stamped)
    return payload
