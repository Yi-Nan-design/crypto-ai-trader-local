from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import infer_bar_hours


@dataclass(frozen=True)
class PerformanceEvaluation:
    """Canonical return, risk, cost, and calendar performance metrics."""

    total_return: float
    max_drawdown: float
    annualized_return: float
    sharpe_like: float
    sortino_ratio: float
    calmar_ratio: float
    fee_ratio: float
    gross_return_before_cost: float
    duration_days: float
    periods_per_year: float
    performance_by_year: dict[str, dict[str, float | int]]
    performance_by_month: dict[str, dict[str, float | int]]
    performance_by_symbol: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_ratio(
    numerator: float,
    denominator: float,
    *,
    positive_sentinel: float = 999.0,
) -> float:
    if denominator > 1e-15:
        return float(numerator / denominator)
    if numerator > 0.0:
        return float(positive_sentinel)
    return 0.0


def _geometric_return(returns: pd.Series) -> float:
    numeric = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if not len(numeric):
        return 0.0
    return float(np.prod(1.0 + np.clip(numeric, -0.999999999, None)) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    numeric = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if not len(numeric):
        return 0.0
    equity = np.concatenate(
        [
            np.array([1.0], dtype=float),
            np.cumprod(1.0 + np.clip(numeric, -0.999999999, None)),
        ]
    )
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / np.maximum(peak, 1e-12) - 1.0))


def annualized_geometric_return(
    total_return: float,
    *,
    duration_days: float,
) -> float:
    """Annualize a geometric total return over its observed duration."""

    if duration_days <= 0.0 or not np.isfinite(total_return):
        return 0.0
    if total_return <= -1.0:
        return -1.0
    exponent = np.log1p(total_return) * (365.0 / duration_days)
    if exponent >= np.log1p(999.0):
        return 999.0
    return float(max(np.expm1(exponent), -1.0))


def risk_adjusted_statistics(
    returns: pd.Series,
    *,
    total_return: float,
    max_drawdown: float,
    bar_hours: float,
    duration_days: float,
) -> dict[str, float]:
    """Calculate interval-aware annualized return, Sharpe, Sortino, and Calmar."""

    numeric = (
        pd.to_numeric(returns, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    periods_per_year = 365.0 * 24.0 / max(float(bar_hours), 1.0 / 60.0)
    mean_return = float(np.mean(numeric)) if len(numeric) else 0.0
    standard_deviation = float(np.std(numeric)) if len(numeric) else 0.0
    downside_deviation = (
        float(np.sqrt(np.mean(np.minimum(numeric, 0.0) ** 2)))
        if len(numeric)
        else 0.0
    )
    annualization = float(np.sqrt(periods_per_year))
    sharpe = _finite_ratio(
        mean_return * annualization,
        standard_deviation,
    )
    sortino = _finite_ratio(
        mean_return * annualization,
        downside_deviation,
    )
    annualized_return = annualized_geometric_return(
        total_return,
        duration_days=duration_days,
    )
    calmar = _finite_ratio(
        annualized_return,
        abs(float(max_drawdown)),
    )
    return {
        "annualized_return": annualized_return,
        "sharpe_like": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "periods_per_year": periods_per_year,
    }


def _beijing_timestamps(data: pd.DataFrame) -> pd.Series | None:
    if "open_time" in data.columns:
        numeric = pd.to_numeric(data["open_time"], errors="coerce")
        timestamps = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    elif "open_datetime" in data.columns:
        timestamps = pd.to_datetime(
            data["open_datetime"],
            utc=True,
            errors="coerce",
        )
    else:
        return None
    return timestamps.dt.tz_convert("Asia/Shanghai")


def _period_summary(rows: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(
        rows.get("strategy_return", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    costs = pd.to_numeric(
        rows.get("total_cost", pd.Series(0.0, index=rows.index)),
        errors="coerce",
    ).fillna(0.0)
    gross_returns = returns + costs
    positive_gross = float(gross_returns.clip(lower=0.0).sum())
    total_cost = float(costs.sum())
    active = (
        pd.to_numeric(
            rows.get(
                "executed_notional_position",
                rows.get("position", pd.Series(0.0, index=rows.index)),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .abs()
        > 1e-15
    )
    active_returns = returns[active]
    gross_profit = float(active_returns.clip(lower=0.0).sum())
    gross_loss = abs(float(active_returns.clip(upper=0.0).sum()))
    return {
        "rows": int(len(rows)),
        "total_return": _geometric_return(returns),
        "max_drawdown": _max_drawdown(returns),
        "win_rate": (
            float((active_returns > 0.0).mean())
            if len(active_returns)
            else 0.0
        ),
        "profit_factor": _finite_ratio(gross_profit, gross_loss),
        "execution_events": int(
            (
                pd.to_numeric(
                    rows.get(
                        "notional_turnover",
                        pd.Series(0.0, index=rows.index),
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                > 1e-15
            ).sum()
        ),
        "turnover": float(
            pd.to_numeric(
                rows.get(
                    "notional_turnover",
                    pd.Series(0.0, index=rows.index),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .sum()
        ),
        "total_cost_drag": total_cost,
        "fee_ratio": _finite_ratio(total_cost, positive_gross),
        "average_exposure": float(
            pd.to_numeric(
                rows.get(
                    "notional_exposure",
                    pd.Series(0.0, index=rows.index),
                ),
                errors="coerce",
            )
            .fillna(0.0)
            .mean()
        )
        if len(rows)
        else 0.0,
    }


def calendar_performance(
    data: pd.DataFrame,
    *,
    frequency: str,
) -> dict[str, dict[str, float | int]]:
    """Aggregate strategy results by Beijing calendar year or month."""

    timestamps = _beijing_timestamps(data)
    if timestamps is None:
        return {}
    working = data.copy()
    if frequency == "year":
        working["_performance_period"] = timestamps.dt.strftime("%Y")
    elif frequency == "month":
        working["_performance_period"] = timestamps.dt.strftime("%Y-%m")
    else:
        raise ValueError("frequency must be 'year' or 'month'")
    working = working[working["_performance_period"].notna()]
    return {
        str(period): _period_summary(rows)
        for period, rows in working.groupby(
            "_performance_period",
            sort=True,
        )
    }


def symbol_performance(
    data: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Aggregate by symbol when a multi-symbol detail frame supplies one."""

    if "symbol" in data.columns:
        symbols = data["symbol"].astype(str).str.upper()
        working = data.assign(_performance_symbol=symbols)
        return {
            str(symbol): _period_summary(rows)
            for symbol, rows in working.groupby(
                "_performance_symbol",
                sort=True,
            )
            if symbol
        }
    symbol = str(data.attrs.get("symbol") or "").upper()
    return {symbol: _period_summary(data)} if symbol else {}


def evaluate_backtest_performance(
    data: pd.DataFrame,
    *,
    initial_balance: float,
) -> PerformanceEvaluation:
    """Build canonical performance metrics from a completed backtest detail."""

    returns = pd.to_numeric(
        data.get("strategy_return", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    equity = pd.to_numeric(
        data.get("equity", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    total_return = (
        float(equity.iloc[-1] / max(float(initial_balance), 1e-12) - 1.0)
        if len(equity)
        else _geometric_return(returns)
    )
    max_drawdown = _max_drawdown(returns)
    bar_hours = max(float(infer_bar_hours(data)), 1.0 / 60.0)
    duration_days = max(len(data) * bar_hours / 24.0, bar_hours / 24.0)
    statistics = risk_adjusted_statistics(
        returns,
        total_return=total_return,
        max_drawdown=max_drawdown,
        bar_hours=bar_hours,
        duration_days=duration_days,
    )
    costs = pd.to_numeric(
        data.get("total_cost", pd.Series(0.0, index=data.index)),
        errors="coerce",
    ).fillna(0.0)
    gross_returns = returns + costs
    positive_gross = float(gross_returns.clip(lower=0.0).sum())
    return PerformanceEvaluation(
        total_return=total_return,
        max_drawdown=max_drawdown,
        annualized_return=float(statistics["annualized_return"]),
        sharpe_like=float(statistics["sharpe_like"]),
        sortino_ratio=float(statistics["sortino_ratio"]),
        calmar_ratio=float(statistics["calmar_ratio"]),
        fee_ratio=_finite_ratio(float(costs.sum()), positive_gross),
        gross_return_before_cost=_geometric_return(gross_returns),
        duration_days=float(duration_days),
        periods_per_year=float(statistics["periods_per_year"]),
        performance_by_year=calendar_performance(
            data,
            frequency="year",
        ),
        performance_by_month=calendar_performance(
            data,
            frequency="month",
        ),
        performance_by_symbol=symbol_performance(data),
    )
