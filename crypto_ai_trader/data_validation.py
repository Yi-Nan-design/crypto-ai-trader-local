from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util

import numpy as np
import pandas as pd

from .exchange_availability import coerce_exchange_available

REQUIRED_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
)

NUMERIC_KLINE_COLUMNS = (
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
    "funding_payment_rate",
    "funding_rate_8h",
)


@dataclass(frozen=True)
class DataValidationConfig:
    """Rules used to clean and audit Binance kline frames."""

    price_outlier_z: float = 12.0
    volume_outlier_z: float = 12.0
    strict: bool = False


@dataclass
class DataValidationReport:
    """Summary of structural cleaning and non-destructive anomaly detection."""

    interval: str
    rows_before: int
    rows_after: int = 0
    duplicate_rows_removed: int = 0
    invalid_rows_removed: int = 0
    missing_bar_count: int = 0
    gap_recovery_rows: int = 0
    price_outlier_count: int = 0
    volume_outlier_count: int = 0
    expected_interval_ms: int | None = None
    blocking_issues: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blocking_issues

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def interval_to_milliseconds(interval: str) -> int | None:
    """Convert common Binance intervals to milliseconds."""

    value = str(interval or "").strip()
    if len(value) < 2:
        return None
    unit = value[-1]
    try:
        amount = int(value[:-1])
    except ValueError:
        return None
    multipliers = {
        "s": 1_000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }
    multiplier = multipliers.get(unit)
    return amount * multiplier if multiplier and amount > 0 else None


def normalize_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric and datetime columns without applying audit policy."""

    data = frame.copy()
    for column in NUMERIC_KLINE_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "open_time" in data.columns:
        data["open_datetime"] = pd.to_datetime(data["open_time"], unit="ms", utc=True, errors="coerce")
    if "close_time" in data.columns:
        data["close_datetime"] = pd.to_datetime(data["close_time"], unit="ms", utc=True, errors="coerce")
    if "ignore" in data.columns:
        data = data.drop(columns=["ignore"])
    return data


def robust_zscore(series: pd.Series) -> pd.Series:
    """Return a MAD-based robust z-score with stable zero-variance behavior."""

    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        return pd.Series(0.0, index=series.index)
    return 0.6745 * (numeric - median) / mad


def isolation_forest_anomalies(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    contamination: float = 0.01,
    random_seed: int = 42,
) -> tuple[pd.Series, dict[str, object]]:
    """Run an optional multivariate anomaly detector without blocking loading."""

    if importlib.util.find_spec("sklearn") is None:
        return pd.Series(False, index=frame.index), {
            "status": "skipped",
            "reason": "missing_dependency: sklearn",
            "anomaly_count": 0,
        }
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(False, index=frame.index), {
            "status": "skipped",
            "reason": "no_available_columns",
            "anomaly_count": 0,
        }
    values = (
        frame[available]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = values.notna().all(axis=1)
    if int(valid.sum()) < 20:
        return pd.Series(False, index=frame.index), {
            "status": "skipped",
            "reason": "insufficient_complete_rows",
            "anomaly_count": 0,
        }
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        contamination=float(np.clip(contamination, 1e-4, 0.25)),
        random_state=int(random_seed),
        n_jobs=1,
    )
    prediction = model.fit_predict(values.loc[valid].to_numpy(dtype=float))
    mask = pd.Series(False, index=frame.index)
    mask.loc[valid] = prediction == -1
    return mask, {
        "status": "completed",
        "reason": "",
        "columns": available,
        "rows": int(valid.sum()),
        "anomaly_count": int(mask.sum()),
        "destructive_filtering": False,
    }


def cross_exchange_price_deviation(
    primary: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    threshold: float = 0.01,
    timestamp_column: str = "open_time",
    price_column: str = "close",
) -> pd.DataFrame:
    """Compare aligned close prices without treating reference data as truth."""

    required = {timestamp_column, price_column}
    for name, frame in (("primary", primary), ("reference", reference)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{name} exchange frame missing columns: {missing}"
            )
    left = primary[[timestamp_column, price_column]].rename(
        columns={price_column: "primary_close"}
    )
    right = reference[[timestamp_column, price_column]].rename(
        columns={price_column: "reference_close"}
    )
    aligned = left.merge(right, on=timestamp_column, how="inner")
    denominator = pd.to_numeric(
        aligned["reference_close"],
        errors="coerce",
    ).replace(0.0, np.nan)
    aligned["price_deviation"] = (
        pd.to_numeric(aligned["primary_close"], errors="coerce")
        / denominator
        - 1.0
    )
    aligned["deviation_flag"] = (
        aligned["price_deviation"].abs() >= abs(float(threshold))
    )
    return aligned


def validate_kline_frame(
    frame: pd.DataFrame,
    interval: str,
    config: DataValidationConfig | None = None,
) -> tuple[pd.DataFrame, DataValidationReport]:
    """Clean structural kline errors and report non-destructive anomalies."""

    cfg = config or DataValidationConfig()
    data = normalize_kline_frame(frame)
    report = DataValidationReport(
        interval=interval,
        rows_before=len(data),
        expected_interval_ms=interval_to_milliseconds(interval),
    )

    missing_columns = sorted(set(REQUIRED_KLINE_COLUMNS).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required kline columns: {missing_columns}")

    duplicate_mask = data.duplicated(subset=["open_time"], keep="last")
    report.duplicate_rows_removed = int(duplicate_mask.sum())
    data = data.loc[~duplicate_mask].copy()

    finite_price = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
    positive_price = (data[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_high = data["high"] >= data[["open", "low", "close"]].max(axis=1)
    valid_low = data["low"] <= data[["open", "high", "close"]].min(axis=1)
    valid_volume = (
        data[["volume", "quote_volume", "trades"]].notna().all(axis=1)
        & (data[["volume", "quote_volume", "trades"]] >= 0).all(axis=1)
    )
    valid_time = data["open_time"].notna() & np.isfinite(data["open_time"])
    structurally_valid = finite_price & positive_price & valid_high & valid_low & valid_volume & valid_time
    report.invalid_rows_removed = int((~structurally_valid).sum())
    data = data.loc[structurally_valid].sort_values("open_time").reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid kline rows remain after structural validation")

    expected_ms = report.expected_interval_ms
    if expected_ms:
        time_diff = data["open_time"].diff()
        missing_before = np.maximum(
            np.rint(time_diff.fillna(expected_ms).to_numpy(dtype=float) / expected_ms).astype(int) - 1,
            0,
        )
        data["exchange_gap_before_bars"] = missing_before
        if "exchange_available" in data.columns:
            data["exchange_available"] = data["exchange_available"].map(
                coerce_exchange_available
            )
        else:
            data["exchange_available"] = True
        positive_gaps = time_diff[time_diff > expected_ms]
        report.missing_bar_count = int(
            sum(max(int(round(float(value) / expected_ms)) - 1, 0) for value in positive_gaps)
        )
        report.gap_recovery_rows = int((missing_before > 0).sum())
    else:
        data["exchange_gap_before_bars"] = 0
        if "exchange_available" in data.columns:
            data["exchange_available"] = data["exchange_available"].map(
                coerce_exchange_available
            )
        else:
            data["exchange_available"] = True

    close_return = data["close"].pct_change()
    log_volume = np.log1p(data["quote_volume"].clip(lower=0))
    report.price_outlier_count = int((robust_zscore(close_return).abs() > cfg.price_outlier_z).sum())
    report.volume_outlier_count = int((robust_zscore(log_volume).abs() > cfg.volume_outlier_z).sum())

    if report.duplicate_rows_removed:
        report.issues.append("duplicate_open_time_removed")
    if report.invalid_rows_removed:
        report.issues.append("structurally_invalid_rows_removed")
    if report.missing_bar_count:
        report.issues.append("missing_kline_intervals_detected")
    if report.price_outlier_count:
        report.issues.append("price_outliers_detected")
    if report.volume_outlier_count:
        report.issues.append("volume_outliers_detected")

    report.rows_after = len(data)
    if cfg.strict and report.issues:
        raise ValueError(f"Kline validation failed: {report.to_dict()}")
    return data, report
