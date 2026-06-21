from __future__ import annotations

from .config import TraderConfig


def primary_label_horizon(interval: str, cfg: TraderConfig) -> int:
    if interval in {"5m", "15m", "1h"}:
        return 1
    return int(cfg.label_horizon)


def primary_label_min_return(interval: str, cfg: TraderConfig) -> float:
    base = float(cfg.label_min_return)
    cost_edge = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate)) + 0.0001
    multiplier = 1.35 if interval.lower() in {"1m", "3m", "5m", "15m"} else 1.25
    return max(base, cost_edge * multiplier)
