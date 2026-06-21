from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import zipfile
from typing import Any

from .config import load_config
from .time_utils import beijing_now_iso, beijing_stamp


DEFAULT_INCLUDE_DIRS = ["crypto_ai_trader", "config", "web", "desktop", "docs", "scripts", ".vscode", "data", "models", "reports", "state"]
DEFAULT_INCLUDE_FILES = ["README.md", "requirements.txt", "config.default.json", "package.json"]
EXCLUDED_DIRS = {".venv", "__pycache__", ".cache", "downloads", "logs", "node_modules", "tools", "dist", "release", "exports"}
EXCLUDED_FILE_NAMES = {"runner_task.cmd"}
SENSITIVE_NAMES = {"config.json", ".env", "secrets.json"}


def utc_now() -> str:
    return beijing_now_iso()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_exclude(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(part in EXCLUDED_DIRS for part in rel_parts):
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.name.lower() in SENSITIVE_NAMES:
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def iter_portable_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in DEFAULT_INCLUDE_FILES:
        path = root / name
        if path.exists() and path.is_file() and not should_exclude(path, root):
            files.append(path)
    for name in DEFAULT_INCLUDE_DIRS:
        base = root / name
        if not base.exists() or not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not should_exclude(path, root):
                files.append(path)
    return sorted(set(files), key=lambda item: item.as_posix().lower())


def live_trading_is_disabled(root: Path) -> bool:
    config_path = root / "config.default.json"
    if not config_path.exists():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("live_trading_enabled") is False


def build_manifest(root: Path, files: list[Path], zip_name: str) -> dict[str, Any]:
    key_paths = [
        "README.md",
        "requirements.txt",
        "config.default.json",
        "config/strategy_calibration_profiles.toml",
        "crypto_ai_trader/cli.py",
        "crypto_ai_trader/dashboard_server.py",
        "web/dashboard.html",
    ]
    key_hashes = {}
    for rel in key_paths:
        path = root / rel
        if path.exists() and path.is_file():
            key_hashes[rel.replace("\\", "/")] = sha256_file(path)
    total_size = sum(path.stat().st_size for path in files)
    return {
        "created_utc": utc_now(),
        "created_beijing": utc_now(),
        "source_root": str(root),
        "zip_name": zip_name,
        "file_count": len(files),
        "total_bytes": total_size,
        "included_roots": DEFAULT_INCLUDE_DIRS,
        "included_files": DEFAULT_INCLUDE_FILES,
        "excluded_dirs": sorted(EXCLUDED_DIRS),
        "excluded_files": sorted(EXCLUDED_FILE_NAMES | SENSITIVE_NAMES),
        "live_trading_enabled_false": live_trading_is_disabled(root),
        "key_hashes": key_hashes,
        "restore_hint": "Run scripts\\restore_portable.ps1 after extracting on a new Windows machine.",
    }


def export_portable(root: Path | None = None, output_dir: Path | None = None, name: str | None = None) -> dict[str, Any]:
    root = (root or Path.cwd()).resolve()
    cfg = load_config(root / "config.default.json")
    reports_dir = (root / cfg.reports_dir).resolve()
    exports_dir = (output_dir or (root / "exports")).resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = beijing_stamp()
    zip_name = name or f"crypto_ai_trader_portable_{stamp}.zip"
    if not zip_name.lower().endswith(".zip"):
        zip_name = f"{zip_name}.zip"
    zip_path = exports_dir / zip_name
    files = iter_portable_files(root)
    manifest = build_manifest(root, files, zip_name)
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        for path in files:
            rel = path.relative_to(root).as_posix()
            archive.write(path, rel)
    result = {
        "ok": True,
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "size_bytes": zip_path.stat().st_size,
        "manifest": manifest,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    latest_path = reports_dir / "portable_export_latest.json"
    latest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Crypto AI Trader portable archive")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    result = export_portable(output_dir=Path(args.output_dir), name=args.name)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
