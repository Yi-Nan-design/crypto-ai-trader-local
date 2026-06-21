from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class ExchangeAvailabilityConfig(Protocol):
    """Configuration required by the exchange availability guard."""

    exchange_downtime_guard_enabled: bool
    exchange_gap_recovery_bars: int


@dataclass(frozen=True)
class ExchangeAvailability:
    """Per-bar execution availability derived from causal market context."""

    available: np.ndarray
    blocked_reason: np.ndarray
    gap_recovery_blocked: np.ndarray
    explicit_unavailable: np.ndarray


def coerce_exchange_available(value: object, *, default: bool = True) -> bool:
    """Parse persisted availability values without treating "False" as true."""

    if value is None or bool(pd.isna(value)):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(float(value))
    text = str(value).strip().lower()
    if text in {"false", "0", "no", "off", "unavailable", "down"}:
        return False
    if text in {"true", "1", "yes", "on", "available", "up"}:
        return True
    return bool(default)


def resolve_exchange_availability(
    data: pd.DataFrame,
    cfg: ExchangeAvailabilityConfig,
) -> ExchangeAvailability:
    """Block execution on explicit outages and after observed K-line gaps."""

    rows = len(data)
    if not bool(getattr(cfg, "exchange_downtime_guard_enabled", True)):
        return ExchangeAvailability(
            available=np.ones(rows, dtype=bool),
            blocked_reason=np.full(rows, "", dtype=object),
            gap_recovery_blocked=np.zeros(rows, dtype=bool),
            explicit_unavailable=np.zeros(rows, dtype=bool),
        )

    if "exchange_available" in data.columns:
        explicit_available = np.asarray(
            [
                coerce_exchange_available(value)
                for value in data["exchange_available"].tolist()
            ],
            dtype=bool,
        )
    else:
        explicit_available = np.ones(rows, dtype=bool)
    explicit_unavailable = ~explicit_available
    if "exchange_unavailable_reason" in data.columns:
        explicit_reason = (
            data["exchange_unavailable_reason"]
            .fillna("")
            .astype(str)
            .to_numpy(dtype=object)
        )
    else:
        explicit_reason = np.full(rows, "", dtype=object)

    if "exchange_gap_before_bars" in data.columns:
        gap_before = (
            pd.to_numeric(data["exchange_gap_before_bars"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            > 0.0
        )
    else:
        gap_before = np.zeros(rows, dtype=bool)

    recovery_bars = max(int(getattr(cfg, "exchange_gap_recovery_bars", 1)), 0)
    gap_recovery_blocked = np.zeros(rows, dtype=bool)
    remaining = 0
    for idx in range(rows):
        if gap_before[idx]:
            remaining = max(remaining, recovery_bars)
        if remaining > 0:
            gap_recovery_blocked[idx] = True
            remaining -= 1

    available = explicit_available & ~gap_recovery_blocked
    reasons = np.full(rows, "", dtype=object)
    reasons[gap_recovery_blocked] = "kline_gap_recovery"
    reasons[explicit_unavailable] = np.where(
        explicit_reason[explicit_unavailable] != "",
        explicit_reason[explicit_unavailable],
        "exchange_explicitly_unavailable",
    )
    return ExchangeAvailability(
        available=available,
        blocked_reason=reasons,
        gap_recovery_blocked=gap_recovery_blocked,
        explicit_unavailable=explicit_unavailable,
    )
