from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile
import calendar
import csv
import json
import os
import socket
import time

import numpy as np
import pandas as pd

from .data_validation import normalize_kline_frame, validate_kline_frame
from .exchange_rules import FuturesSymbolRules, parse_futures_symbol_rules
from .time_utils import beijing_now_iso


KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    interval: str
    files: list[Path]
    rows: int
    skipped: list[str] | None = None


@dataclass(frozen=True)
class MarketContextResult:
    """Files written while synchronizing public derivatives market context."""

    symbol: str
    funding_file: Path
    funding_rows: int
    exchange_rules_file: Path
    exchange_rules: FuturesSymbolRules


def month_range(start: str, end: str) -> list[str]:
    """Return inclusive YYYY-MM month strings."""
    start_year, start_month = [int(part) for part in start.split("-")]
    end_year, end_month = [int(part) for part in end.split("-")]
    cursor = date(start_year, start_month, 1)
    finish = date(end_year, end_month, 1)
    months: list[str] = []
    while cursor <= finish:
        months.append(f"{cursor.year:04d}-{cursor.month:02d}")
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def daily_range(start: str, end: str) -> list[str]:
    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    days = pd.date_range(start_date, end_date, freq="D")
    return [day.strftime("%Y-%m-%d") for day in days]


def days_for_month(period: str) -> list[str]:
    year, month = [int(part) for part in period.split("-")]
    first = date(year, month, 1)
    count = calendar.monthrange(year, month)[1]
    return [(first + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(count)]


def binance_public_kline_url(
    symbol: str,
    interval: str,
    period: str,
    market: str = "futures_um",
    frequency: str = "monthly",
    base_url: str | None = None,
) -> str:
    symbol = symbol.upper()
    custom_base = base_url or os.environ.get("BINANCE_DATA_BASE_URL", "")
    if custom_base:
        base = custom_base.rstrip("/")
    elif market == "futures_um":
        base = "https://data.binance.vision/data/futures/um"
    elif market == "spot":
        base = "https://data.binance.vision/data/spot"
    else:
        raise ValueError(f"Unsupported market: {market}")
    return f"{base}/{frequency}/klines/{symbol}/{interval}/{symbol}-{interval}-{period}.zip"


def split_base_urls(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace(",", ";")
    return [item.strip().rstrip("/") for item in normalized.split(";") if item.strip()]


def data_base_url_candidates(base_url: str | None = None) -> list[str | None]:
    candidates: list[str | None] = []
    for source in [
        base_url,
        os.environ.get("BINANCE_DATA_BASE_URLS", ""),
        os.environ.get("BINANCE_DATA_BASE_URL", ""),
    ]:
        candidates.extend(split_base_urls(source))
    candidates.append(None)
    unique: list[str | None] = []
    for item in candidates:
        key = item or ""
        if key not in {value or "" for value in unique}:
            unique.append(item)
    return unique


def kline_archive_cache_path(
    data_dir: str | Path,
    symbol: str,
    interval: str,
    period: str,
    *,
    market: str,
    frequency: str,
) -> Path:
    return (
        Path(data_dir)
        / "archive_cache"
        / market
        / frequency
        / "klines"
        / symbol.upper()
        / interval
        / f"{symbol.upper()}-{interval}-{period}.zip"
    )


def fetch_kline_archive(
    symbol: str,
    interval: str,
    period: str,
    data_dir: str | Path,
    *,
    market: str,
    frequency: str,
    base_url: str | None = None,
    cache_only: bool = False,
) -> bytes:
    cache_path = kline_archive_cache_path(
        data_dir,
        symbol,
        interval,
        period,
        market=market,
        frequency=frequency,
    )
    if cache_path.exists():
        return cache_path.read_bytes()
    if cache_only:
        raise FileNotFoundError(f"Kline archive is not cached: {cache_path}")

    errors: list[str] = []
    for candidate in data_base_url_candidates(base_url):
        url = binance_public_kline_url(symbol, interval, period, market=market, frequency=frequency, base_url=candidate)
        try:
            payload = download_bytes(url)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            return payload
        except HTTPError as exc:
            errors.append(f"{url}: HTTP {exc.code}")
            if exc.code == 404:
                continue
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Failed to fetch kline archive for {symbol} {interval} {period}. Tried: {' | '.join(errors)}")


def download_bytes(url: str, retries: int = 5, timeout: int = 30) -> bytes:
    last_error: Exception | None = None
    headers = {"User-Agent": "crypto-ai-trader/0.1"}
    for attempt in range(1, retries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                payload = response.read()
                if payload:
                    return payload
                last_error = RuntimeError("empty response body")
        except HTTPError as exc:
            if exc.code == 404:
                raise
            last_error = exc
        except (URLError, IncompleteRead, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2 * attempt, 8))
    proxy_hint = ""
    if not (os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")):
        proxy_hint = " No HTTP_PROXY/HTTPS_PROXY is configured."
    raise RuntimeError(f"Failed to download {url}: {last_error}.{proxy_hint}")


def request_json(url: str, params: dict[str, object] | None = None, retries: int = 5, timeout: int = 30) -> object:
    query = urlencode(params or {})
    full_url = f"{url}?{query}" if query else url
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            payload = download_bytes(full_url, retries=1, timeout=timeout)
            return json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Failed to load JSON from {full_url}: {last_error}")


def market_context_dir(data_dir: str | Path, symbol: str) -> Path:
    """Return the local public-market-context directory for a symbol."""

    return Path(data_dir) / "market_context" / symbol.upper()


def funding_history_path(data_dir: str | Path, symbol: str) -> Path:
    return market_context_dir(data_dir, symbol) / "funding_rates.csv"


def exchange_rules_path(data_dir: str | Path, symbol: str) -> Path:
    return market_context_dir(data_dir, symbol) / "exchange_rules.json"


def parse_funding_history_rows(payload: object) -> pd.DataFrame:
    """Normalize Binance funding-rate API rows into a stable local schema."""

    if not isinstance(payload, list):
        raise ValueError("Unexpected Binance funding-rate response")
    records: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "symbol": str(item.get("symbol", "")).upper(),
                "funding_time": pd.to_numeric(
                    pd.Series([item.get("fundingTime", item.get("funding_time"))]),
                    errors="coerce",
                ).iloc[0],
                "funding_rate_8h": pd.to_numeric(
                    pd.Series([item.get("fundingRate", item.get("funding_rate_8h"))]),
                    errors="coerce",
                ).iloc[0],
                "mark_price": pd.to_numeric(
                    pd.Series([item.get("markPrice", item.get("mark_price"))]),
                    errors="coerce",
                ).iloc[0],
            }
        )
    frame = pd.DataFrame(
        records,
        columns=["symbol", "funding_time", "funding_rate_8h", "mark_price"],
    )
    if frame.empty:
        return frame
    frame = frame[
        np.isfinite(frame["funding_time"])
        & np.isfinite(frame["funding_rate_8h"])
    ].copy()
    frame["funding_time"] = frame["funding_time"].astype("int64")
    frame["funding_datetime"] = pd.to_datetime(
        frame["funding_time"],
        unit="ms",
        utc=True,
        errors="coerce",
    )
    return (
        frame.sort_values("funding_time")
        .drop_duplicates(subset=["symbol", "funding_time"], keep="last")
        .reset_index(drop=True)
    )


def merge_funding_history(
    klines: pd.DataFrame,
    funding_history: pd.DataFrame,
) -> pd.DataFrame:
    """Map each funding settlement to the K-line containing its timestamp."""

    data = klines.copy().reset_index(drop=True)
    if funding_history.empty or "open_time" not in data.columns:
        return data
    funding_columns = ["funding_time", "funding_rate_8h"]
    if "mark_price" in funding_history.columns:
        funding_columns.append("mark_price")
    funding = funding_history[funding_columns].copy()
    funding["funding_time"] = pd.to_numeric(
        funding["funding_time"],
        errors="coerce",
    )
    funding["funding_rate_8h"] = pd.to_numeric(
        funding["funding_rate_8h"],
        errors="coerce",
    )
    funding = (
        funding.dropna()
        .sort_values("funding_time")
        .drop_duplicates("funding_time", keep="last")
    )
    if funding.empty:
        return data
    data = data.drop(
        columns=[
            column
            for column in (
                "funding_payment_rate",
                "funding_rate_8h",
                "funding_mark_price",
            )
            if column in data.columns
        ]
    )
    open_times = pd.to_numeric(data["open_time"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(open_times, kind="stable")
    sorted_open_times = open_times[order]
    event_times = funding["funding_time"].to_numpy(dtype=float)
    sorted_positions = np.searchsorted(
        sorted_open_times,
        event_times,
        side="right",
    ) - 1
    positive_diffs = np.diff(sorted_open_times)
    positive_diffs = positive_diffs[positive_diffs > 0]
    inferred_interval = (
        float(np.median(positive_diffs))
        if len(positive_diffs)
        else 0.0
    )
    final_boundary = (
        float(sorted_open_times[-1] + inferred_interval)
        if len(sorted_open_times)
        else float("-inf")
    )
    valid = (
        (sorted_positions >= 0)
        & (event_times < final_boundary)
        & np.isfinite(event_times)
        & np.isfinite(funding["funding_rate_8h"].to_numpy(dtype=float))
    )
    if not bool(valid.any()):
        return data

    original_positions = order[sorted_positions[valid]]
    event_frame = pd.DataFrame(
        {
            "row_index": original_positions,
            "funding_payment_rate": funding.loc[
                valid,
                "funding_rate_8h",
            ].to_numpy(dtype=float),
        }
    )
    if "mark_price" in funding.columns:
        event_frame["funding_mark_price"] = funding.loc[
            valid,
            "mark_price",
        ].to_numpy(dtype=float)
    grouped_rates = event_frame.groupby("row_index")["funding_payment_rate"].sum()
    data["funding_payment_rate"] = np.nan
    data["funding_rate_8h"] = np.nan
    data.loc[grouped_rates.index, "funding_payment_rate"] = grouped_rates.to_numpy()
    data.loc[grouped_rates.index, "funding_rate_8h"] = grouped_rates.to_numpy()
    if "funding_mark_price" in event_frame.columns:
        grouped_mark = event_frame.groupby("row_index")["funding_mark_price"].last()
        data["funding_mark_price"] = np.nan
        data.loc[grouped_mark.index, "funding_mark_price"] = grouped_mark.to_numpy()
    return data


def load_cached_exchange_rules(
    data_dir: str | Path,
    symbol: str,
) -> FuturesSymbolRules | None:
    """Read cached public symbol filters when available."""

    path = exchange_rules_path(data_dir, symbol)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FuturesSymbolRules(
        symbol=str(payload["symbol"]).upper(),
        min_notional_usdt=float(payload.get("min_notional_usdt", 0.0)),
        min_quantity=float(payload.get("min_quantity", 0.0)),
        max_quantity=float(payload.get("max_quantity", 0.0)),
        quantity_step=float(payload.get("quantity_step", 0.0)),
        price_tick_size=float(payload.get("price_tick_size", 0.0)),
        quantity_filter_type=str(payload.get("quantity_filter_type", "LOT_SIZE")),
        source=str(payload.get("source", "binance_futures_exchange_info")),
    )


def resolve_exchange_rule_values(
    data_dir: str | Path,
    symbol: str,
    *,
    min_notional_usdt: float = 0.0,
    min_quantity: float = 0.0,
    max_quantity: float = 0.0,
    quantity_step: float = 0.0,
    price_tick_size: float = 0.0,
) -> dict[str, float]:
    """Prefer explicit rules and otherwise fill zero values from local cache."""

    cached = load_cached_exchange_rules(data_dir, symbol)
    return {
        "exchange_min_notional_usdt": (
            float(min_notional_usdt)
            if float(min_notional_usdt) > 0.0 or cached is None
            else cached.min_notional_usdt
        ),
        "exchange_min_quantity": (
            float(min_quantity)
            if float(min_quantity) > 0.0 or cached is None
            else cached.min_quantity
        ),
        "exchange_max_quantity": (
            float(max_quantity)
            if float(max_quantity) > 0.0 or cached is None
            else cached.max_quantity
        ),
        "exchange_quantity_step": (
            float(quantity_step)
            if float(quantity_step) > 0.0 or cached is None
            else cached.quantity_step
        ),
        "exchange_price_tick_size": (
            float(price_tick_size)
            if float(price_tick_size) > 0.0 or cached is None
            else cached.price_tick_size
        ),
    }


def _utc_boundary_milliseconds(value: str, *, end: bool) -> int:
    text = str(value).strip()
    timestamp = pd.Timestamp(text)
    if end and len(text) == 7:
        timestamp = timestamp + pd.offsets.MonthBegin(1) - pd.Timedelta(milliseconds=1)
    elif end and len(text) == 10:
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def fetch_futures_funding_history(
    symbol: str,
    start: str,
    end: str,
    *,
    base_url: str = "https://fapi.binance.com",
    limit: int = 1000,
) -> pd.DataFrame:
    """Download public USDT-M funding history with deterministic pagination."""

    start_ms = _utc_boundary_milliseconds(start, end=False)
    end_ms = _utc_boundary_milliseconds(end, end=True)
    if start_ms > end_ms:
        raise ValueError("funding history start must not be after end")
    page_limit = max(1, min(int(limit), 1000))
    cursor = start_ms
    pages: list[pd.DataFrame] = []
    while cursor <= end_ms:
        payload = request_json(
            f"{base_url.rstrip('/')}/fapi/v1/fundingRate",
            params={
                "symbol": symbol.upper(),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": page_limit,
            },
        )
        page = parse_funding_history_rows(payload)
        if page.empty:
            break
        page = page[
            (page["funding_time"] >= cursor)
            & (page["funding_time"] <= end_ms)
        ].copy()
        if page.empty:
            break
        pages.append(page)
        last_time = int(page["funding_time"].max())
        if last_time < cursor or len(page) < page_limit:
            break
        cursor = last_time + 1
    if not pages:
        return parse_funding_history_rows([])
    return (
        pd.concat(pages, ignore_index=True)
        .sort_values("funding_time")
        .drop_duplicates(subset=["symbol", "funding_time"], keep="last")
        .reset_index(drop=True)
    )


def fetch_futures_exchange_rules(
    symbols: list[str],
    *,
    base_url: str = "https://fapi.binance.com",
) -> dict[str, FuturesSymbolRules]:
    """Download and parse public USDT-M exchange symbol filters."""

    payload = request_json(f"{base_url.rstrip('/')}/fapi/v1/exchangeInfo")
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Binance exchangeInfo response")
    return {
        symbol.upper(): parse_futures_symbol_rules(payload, symbol)
        for symbol in symbols
    }


def sync_futures_market_context(
    symbols: list[str],
    start: str,
    end: str,
    *,
    data_dir: str | Path = "data",
    base_url: str = "https://fapi.binance.com",
) -> list[MarketContextResult]:
    """Persist public funding history and symbol filters for offline research."""

    normalized_symbols = [str(symbol).upper() for symbol in symbols]
    rule_map = fetch_futures_exchange_rules(
        normalized_symbols,
        base_url=base_url,
    )
    results: list[MarketContextResult] = []
    for symbol in normalized_symbols:
        funding = fetch_futures_funding_history(
            symbol,
            start,
            end,
            base_url=base_url,
        )
        funding_file = funding_history_path(data_dir, symbol)
        funding_file.parent.mkdir(parents=True, exist_ok=True)
        if funding_file.exists():
            existing = parse_funding_history_rows(
                pd.read_csv(funding_file).to_dict(orient="records")
            )
            funding = (
                pd.concat([existing, funding], ignore_index=True)
                .sort_values("funding_time")
                .drop_duplicates(subset=["symbol", "funding_time"], keep="last")
                .reset_index(drop=True)
            )
        funding.to_csv(funding_file, index=False)

        rules = rule_map[symbol]
        rules_file = exchange_rules_path(data_dir, symbol)
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        rules_payload = {
            **rules.to_dict(),
            "updated_beijing": beijing_now_iso(),
        }
        rules_file.write_text(
            json.dumps(rules_payload, indent=2),
            encoding="utf-8",
        )
        results.append(
            MarketContextResult(
                symbol=symbol,
                funding_file=funding_file,
                funding_rows=len(funding),
                exchange_rules_file=rules_file,
                exchange_rules=rules,
            )
        )
    return results


def parse_kline_zip(payload: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found in Binance archive")
        with archive.open(csv_names[0]) as raw:
            sample = raw.read(256).decode("utf-8", errors="ignore")
            raw.seek(0)
            has_header = sample.lower().startswith("open_time")
            df = pd.read_csv(raw, header=0 if has_header else None)

    if len(df.columns) >= len(KLINE_COLUMNS):
        df = df.iloc[:, : len(KLINE_COLUMNS)]
        df.columns = KLINE_COLUMNS
    else:
        raise ValueError(f"Unexpected Binance kline columns: {list(df.columns)}")

    return normalize_kline_frame(df).sort_values("open_time").reset_index(drop=True)


def parse_kline_rows(rows: list[list[object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return normalize_kline_frame(df).sort_values("open_time").reset_index(drop=True)


def save_klines(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
            return output_path
        except Exception:
            output_path = output_path.with_suffix(".csv")
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_path


def import_kline_zip(zip_path: str | Path, output_path: str | Path) -> Path:
    df = parse_kline_zip(Path(zip_path).read_bytes())
    return save_klines(df, Path(output_path))


def load_klines(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if data_path.suffix == ".parquet":
        return normalize_kline_frame(pd.read_parquet(data_path))
    return normalize_kline_frame(pd.read_csv(data_path))


def load_symbol_interval(
    data_dir: str | Path,
    symbol: str,
    interval: str,
    include_realtime: bool = False,
) -> pd.DataFrame:
    base = Path(data_dir) / "raw" / symbol.upper() / interval
    files = sorted(list(base.glob("*.parquet")) + list(base.glob("*.csv")))
    if include_realtime:
        realtime_base = Path(data_dir) / "realtime" / symbol.upper() / interval
        files.extend(sorted(list(realtime_base.glob("*.parquet")) + list(realtime_base.glob("*.csv"))))
    if not files:
        raise FileNotFoundError(f"No kline files found in {base}")
    frames = [load_klines(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    validated, report = validate_kline_frame(df, interval)
    funding_file = funding_history_path(data_dir, symbol)
    if funding_file.exists():
        funding = parse_funding_history_rows(
            pd.read_csv(funding_file).to_dict(orient="records")
        )
        validated = merge_funding_history(validated, funding)
    rules_file = exchange_rules_path(data_dir, symbol)
    rules_payload = (
        json.loads(rules_file.read_text(encoding="utf-8"))
        if rules_file.exists()
        else {}
    )
    validated.attrs["data_validation"] = report.to_dict()
    validated.attrs["symbol"] = symbol.upper()
    validated.attrs["interval"] = interval
    validated.attrs["market_context"] = {
        "funding_history_path": str(funding_file) if funding_file.exists() else None,
        "funding_cache_rows": int(len(funding)) if funding_file.exists() else 0,
        "funding_events_applied": (
            int(validated["funding_payment_rate"].notna().sum())
            if "funding_payment_rate" in validated.columns
            else 0
        ),
        "funding_latest_utc": (
            str(funding["funding_datetime"].max())
            if funding_file.exists() and not funding.empty
            else None
        ),
        "exchange_rules_path": str(rules_file) if rules_file.exists() else None,
        "exchange_rules_updated_beijing": rules_payload.get("updated_beijing"),
    }
    return validated


def fetch_recent_futures_klines(
    symbol: str,
    interval: str,
    limit: int = 1500,
    base_url: str = "https://fapi.binance.com",
    closed_only: bool = True,
) -> pd.DataFrame:
    limit = max(1, min(int(limit), 1500))
    rows = request_json(
        f"{base_url.rstrip('/')}/fapi/v1/klines",
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
    )
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected Binance kline response: {rows}")
    df = parse_kline_rows(rows)
    if closed_only:
        now_ms = int(time.time() * 1000)
        df = df[df["close_time"] < now_ms - 1000].reset_index(drop=True)
    return df


def sync_recent_futures_klines(
    symbols: list[str],
    interval: str,
    data_dir: str | Path = "data",
    limit: int = 1500,
    base_url: str = "https://fapi.binance.com",
    allow_stale: bool = True,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for symbol in [item.upper() for item in symbols]:
        output = Path(data_dir) / "realtime" / symbol / interval / f"{symbol}-{interval}-realtime.csv"
        existing = load_klines(output) if output.exists() else pd.DataFrame()
        try:
            latest = fetch_recent_futures_klines(symbol, interval, limit=limit, base_url=base_url, closed_only=True)
        except Exception as exc:
            if allow_stale and not existing.empty:
                stale = existing.copy()
                if "exchange_available" not in stale.columns:
                    stale["exchange_available"] = True
                else:
                    stale["exchange_available"] = (
                        stale["exchange_available"].fillna(True).astype(bool)
                    )
                if "exchange_unavailable_reason" not in stale.columns:
                    stale["exchange_unavailable_reason"] = ""
                else:
                    stale["exchange_unavailable_reason"] = (
                        stale["exchange_unavailable_reason"].fillna("").astype(str)
                    )
                stale.loc[stale.index[-1], "exchange_available"] = False
                stale.loc[
                    stale.index[-1],
                    "exchange_unavailable_reason",
                ] = "realtime_sync_failed_using_cached_data"
                save_klines(stale, output)
                results.append(
                    DownloadResult(
                        symbol=symbol,
                        interval=interval,
                        files=[output],
                        rows=len(existing),
                        skipped=[f"realtime_sync_failed_using_cached_data: {exc}"],
                    )
                )
                continue
            raise
        latest = latest.copy()
        latest["exchange_available"] = True
        latest["exchange_unavailable_reason"] = ""
        if not existing.empty:
            if "exchange_available" not in existing.columns:
                existing["exchange_available"] = True
            else:
                existing["exchange_available"] = (
                    existing["exchange_available"].fillna(True).astype(bool)
                )
            if "exchange_unavailable_reason" not in existing.columns:
                existing["exchange_unavailable_reason"] = ""
            else:
                existing["exchange_unavailable_reason"] = (
                    existing["exchange_unavailable_reason"].fillna("").astype(str)
                )
        if existing.empty:
            merged = latest
        else:
            merged = pd.concat([existing, latest], ignore_index=True)
        merged, _ = validate_kline_frame(merged, interval)
        save_klines(merged, output)
        results.append(DownloadResult(symbol=symbol, interval=interval, files=[output], rows=len(merged)))
    return results


def download_monthly_klines(
    symbols: list[str],
    interval: str,
    start: str,
    end: str,
    data_dir: str | Path = "data",
    market: str = "futures_um",
    fallback_daily: bool = True,
    base_url: str | None = None,
    cache_only: bool = False,
    allow_partial_cache: bool = False,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for symbol in [item.upper() for item in symbols]:
        saved_files: list[Path] = []
        skipped: list[str] = []
        total_rows = 0
        for month in month_range(start, end):
            output = Path(data_dir) / "raw" / symbol / interval / f"{symbol}-{interval}-{month}.csv"
            if output.exists():
                df = load_klines(output)
            else:
                try:
                    payload = fetch_kline_archive(
                        symbol,
                        interval,
                        month,
                        data_dir,
                        market=market,
                        frequency="monthly",
                        base_url=base_url,
                        cache_only=cache_only,
                    )
                    df = parse_kline_zip(payload)
                except Exception:
                    if cache_only and not fallback_daily:
                        if allow_partial_cache:
                            skipped.append(month)
                            continue
                        raise
                    if not fallback_daily:
                        if allow_partial_cache:
                            skipped.append(month)
                            continue
                        raise
                    try:
                        df = download_daily_month(
                            symbol,
                            interval,
                            month,
                            data_dir,
                            market=market,
                            base_url=base_url,
                            cache_only=cache_only,
                            allow_partial_cache=allow_partial_cache,
                        )
                    except Exception:
                        if allow_partial_cache:
                            skipped.append(month)
                            continue
                        raise
                output = save_klines(df, output)
            saved_files.append(output)
            total_rows += len(df)
        if not saved_files and not allow_partial_cache:
            raise RuntimeError(f"No kline files available for {symbol} {interval} {start}..{end}")
        results.append(DownloadResult(symbol, interval, saved_files, total_rows, skipped=skipped or None))
    return results


def download_daily_month(
    symbol: str,
    interval: str,
    month: str,
    data_dir: str | Path = "data",
    market: str = "futures_um",
    base_url: str | None = None,
    cache_only: bool = False,
    allow_partial_cache: bool = False,
) -> pd.DataFrame:
    skipped: list[str] = []
    frames: list[pd.DataFrame] = []
    for day in days_for_month(month):
        output = Path(data_dir) / "raw_daily" / symbol / interval / f"{symbol}-{interval}-{day}.csv"
        if output.exists():
            frames.append(load_klines(output))
            continue
        try:
            payload = fetch_kline_archive(
                symbol,
                interval,
                day,
                data_dir,
                market=market,
                frequency="daily",
                base_url=base_url,
                cache_only=cache_only,
            )
            df = parse_kline_zip(payload)
            save_klines(df, output)
            frames.append(df)
        except Exception:
            if allow_partial_cache:
                skipped.append(day)
                continue
            raise
    if not frames:
        raise RuntimeError(f"No daily files available for {symbol} {interval} {month}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time")


def diagnose_network(output_path: str | Path | None = None) -> dict[str, object]:
    hosts = [
        ("data.binance.vision", 443),
        ("demo-fapi.binance.com", 443),
        ("fapi.binance.com", 443),
    ]
    checks: list[dict[str, object]] = []
    for host, port in hosts:
        item: dict[str, object] = {"host": host, "port": port}
        try:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            item["dns_ok"] = True
            item["addresses"] = sorted({addr[4][0] for addr in addresses})[:5]
        except Exception as exc:
            item["dns_ok"] = False
            item["dns_error"] = str(exc)
            checks.append(item)
            continue

        started = time.time()
        try:
            with socket.create_connection((host, port), timeout=8):
                item["connect_ok"] = True
                item["connect_ms"] = round((time.time() - started) * 1000, 2)
        except Exception as exc:
            item["connect_ok"] = False
            item["connect_error"] = str(exc)
        checks.append(item)

    env = {
        "HTTP_PROXY": bool(os.environ.get("HTTP_PROXY")),
        "HTTPS_PROXY": bool(os.environ.get("HTTPS_PROXY")),
        "HTTPS_PROXY_VALUE": os.environ.get("HTTPS_PROXY", ""),
        "BINANCE_DATA_BASE_URL": os.environ.get("BINANCE_DATA_BASE_URL", ""),
    }
    http_checks = diagnose_http_urls()
    payload: dict[str, object] = {
        "created_utc": beijing_now_iso(),
        "created_beijing": beijing_now_iso(),
        "environment": env,
        "checks": checks,
        "http_checks": http_checks,
        "recommendation": network_recommendation(checks, env, http_checks),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def diagnose_http_urls() -> list[dict[str, object]]:
    urls = [
        "https://data.binance.vision/",
        "https://demo-fapi.binance.com/fapi/v1/ping",
    ]
    results: list[dict[str, object]] = []
    for url in urls:
        item: dict[str, object] = {"url": url}
        started = time.time()
        try:
            with urlopen(Request(url, headers={"User-Agent": "crypto-ai-trader/0.1"}), timeout=12) as response:
                item["ok"] = True
                item["status"] = response.status
                item["elapsed_ms"] = round((time.time() - started) * 1000, 2)
        except HTTPError as exc:
            item["ok"] = True
            item["status"] = exc.code
            item["elapsed_ms"] = round((time.time() - started) * 1000, 2)
            item["note"] = "HTTP error response received, network path is reachable"
        except Exception as exc:
            item["ok"] = False
            item["error"] = str(exc)
        results.append(item)
    return results


def network_recommendation(
    checks: list[dict[str, object]],
    env: dict[str, object],
    http_checks: list[dict[str, object]] | None = None,
) -> str:
    if http_checks and all(item.get("ok") for item in http_checks):
        if env.get("HTTPS_PROXY"):
            return "Proxy path is reachable. Downloads should work through HTTPS_PROXY."
        return "HTTP path is reachable. Downloads should work."
    if all(item.get("connect_ok") for item in checks):
        return "Network connectivity looks available."
    dns_failed = any(not item.get("dns_ok") for item in checks)
    blocked = any("10013" in str(item.get("connect_error", "")) for item in checks)
    if blocked:
        return "Windows socket permission denied. Allow Python through firewall/security software or run with network approval."
    if dns_failed:
        return "DNS lookup failed. Check DNS, VPN, or proxy settings."
    if not env.get("HTTPS_PROXY"):
        return "Connection failed and HTTPS_PROXY is not set. If you use a proxy/VPN, set HTTPS_PROXY before running downloads."
    return "Connection failed. Check proxy/VPN/firewall and retry."


def expected_monthly_rows(interval: str, period: str) -> int | None:
    minutes = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
    }.get(interval)
    if minutes is None:
        return None
    year, month = [int(part) for part in period.split("-")]
    days = calendar.monthrange(year, month)[1]
    return days * 24 * 60 // minutes
