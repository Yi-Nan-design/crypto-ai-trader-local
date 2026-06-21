from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_VERSION = "v3_forward_edge_path_rank_meta_optional"
MULTI_HORIZON_STEPS = (1, 2, 3, 4, 6, 12, 16)
META_LABEL_COLUMNS = (
    "meta_primary_side",
    "meta_signed_future_return",
    "meta_edge_after_cost",
    "meta_label",
    "meta_label_active",
)


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for forward-looking training labels."""

    horizon: int = 3
    min_return: float = 0.001
    multi_horizon_steps: tuple[int, ...] = MULTI_HORIZON_STEPS
    risk_adjusted_threshold: float = 0.5
    volatility_floor: float = 1e-8
    meta_side_column: str | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.min_return < 0:
            raise ValueError("min_return must be non-negative")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive")
        if any(step < 1 for step in self.multi_horizon_steps):
            raise ValueError("multi_horizon_steps must contain positive integers")
        if (
            self.meta_side_column is not None
            and not self.meta_side_column.strip()
        ):
            raise ValueError("meta_side_column must not be empty")


@dataclass(frozen=True)
class MetaLabelConfig:
    """Configuration for optional primary-signal execution labels."""

    side_column: str = "primary_side_signal"
    future_return_column: str = "future_return"
    edge_threshold_column: str = "edge_return_threshold"

    def __post_init__(self) -> None:
        for name in (
            self.side_column,
            self.future_return_column,
            self.edge_threshold_column,
        ):
            if not name.strip():
                raise ValueError("meta-label column names must not be empty")


def future_realized_volatility(close: pd.Series, horizon: int) -> pd.Series:
    """Return RMS volatility of the next `horizon` one-bar returns."""

    forward_returns = [
        close.shift(-step) / close.shift(-(step - 1)) - 1.0
        for step in range(1, horizon + 1)
    ]
    future_path = pd.concat(forward_returns, axis=1)
    return np.sqrt(future_path.pow(2).mean(axis=1, skipna=False))


def add_path_dependent_labels(
    data: pd.DataFrame,
    *,
    horizon: int,
    upper_barrier: float | pd.Series,
    lower_barrier: float | pd.Series,
) -> pd.DataFrame:
    """Add forward-path triple-barrier and favorable/adverse excursions."""

    required = {"close", "high", "low"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Cannot build path-dependent labels; missing columns: {missing}"
        )
    if int(horizon) < 1:
        raise ValueError("horizon must be at least 1")

    labeled = data.copy()
    close = pd.to_numeric(labeled["close"], errors="coerce")
    high = pd.to_numeric(labeled["high"], errors="coerce")
    low = pd.to_numeric(labeled["low"], errors="coerce")
    upper = pd.Series(upper_barrier, index=labeled.index, dtype=float).abs()
    lower = pd.Series(lower_barrier, index=labeled.index, dtype=float).abs()
    future_high_returns = pd.concat(
        [high.shift(-step) / close - 1.0 for step in range(1, horizon + 1)],
        axis=1,
    )
    future_low_returns = pd.concat(
        [low.shift(-step) / close - 1.0 for step in range(1, horizon + 1)],
        axis=1,
    )
    complete = (
        close.notna()
        & upper.notna()
        & lower.notna()
        & future_high_returns.notna().all(axis=1)
        & future_low_returns.notna().all(axis=1)
    )
    label = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    hit_step = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    unresolved = complete.copy()
    for step in range(1, horizon + 1):
        high_return = future_high_returns.iloc[:, step - 1]
        low_return = future_low_returns.iloc[:, step - 1]
        upper_hit = unresolved & (high_return >= upper)
        lower_hit = unresolved & (low_return <= -lower)
        both_hit = upper_hit & lower_hit
        upper_only = upper_hit & ~both_hit
        lower_only = lower_hit & ~both_hit
        label.loc[upper_only] = 1
        label.loc[lower_only] = -1
        hit_step.loc[upper_only | lower_only] = step
        if bool(both_hit.any()):
            step_return = close.shift(-step) / close - 1.0
            label.loc[both_hit] = np.sign(
                step_return.loc[both_hit]
            ).astype(int)
            hit_step.loc[both_hit] = step
        unresolved &= ~(upper_hit | lower_hit)
    label.loc[unresolved] = 0
    hit_step.loc[unresolved] = horizon
    labeled["max_favorable_excursion_label"] = (
        future_high_returns.max(axis=1, skipna=False).where(complete)
    )
    labeled["max_adverse_excursion_label"] = (
        future_low_returns.min(axis=1, skipna=False).where(complete)
    )
    labeled["triple_barrier_label"] = label
    labeled["triple_barrier_hit_step"] = hit_step
    return labeled


def add_cross_sectional_rank_labels(
    data: pd.DataFrame,
    *,
    group_column: str = "open_time",
    value_column: str = "future_risk_adjusted_return",
    bins: int = 5,
) -> pd.DataFrame:
    """Rank forward outcomes within a timestamp for cross-sectional training."""

    required = {group_column, value_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Cannot build cross-sectional labels; missing columns: {missing}"
        )
    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    labeled = data.copy()
    values = pd.to_numeric(labeled[value_column], errors="coerce")
    percentile = values.groupby(labeled[group_column]).rank(
        method="average",
        pct=True,
    )
    relevance = np.floor(
        percentile.clip(lower=0.0, upper=1.0 - 1e-12) * int(bins)
    )
    labeled["cross_sectional_rank_label"] = percentile
    labeled["cross_sectional_relevance_label"] = relevance.astype("Int64")
    return labeled


def add_meta_labels(
    data: pd.DataFrame,
    config: MetaLabelConfig | None = None,
) -> pd.DataFrame:
    """Label whether a causal primary side signal beats its cost hurdle."""

    config = config or MetaLabelConfig()
    required = {
        config.side_column,
        config.future_return_column,
        config.edge_threshold_column,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Cannot build meta labels; missing columns: {missing}"
        )

    labeled = data.copy()
    side = pd.to_numeric(
        labeled[config.side_column],
        errors="coerce",
    )
    invalid_side = side.notna() & ~side.isin([-1, 0, 1])
    if bool(invalid_side.any()):
        raise ValueError(
            "meta-label side values must be -1, 0, 1, or missing"
        )
    future_return = pd.to_numeric(
        labeled[config.future_return_column],
        errors="coerce",
    )
    edge_threshold = (
        pd.to_numeric(
            labeled[config.edge_threshold_column],
            errors="coerce",
        )
        .abs()
    )
    active = side.isin([-1, 1])
    complete = active & future_return.notna() & edge_threshold.notna()
    signed_return = (side * future_return).where(active)
    edge_after_cost = (signed_return - edge_threshold).where(active)
    meta_label = pd.Series(
        pd.NA,
        index=labeled.index,
        dtype="Int64",
    )
    meta_label.loc[complete] = (
        edge_after_cost.loc[complete] > 0.0
    ).astype(int)

    labeled["meta_primary_side"] = side.astype("Int64")
    labeled["meta_signed_future_return"] = signed_return
    labeled["meta_edge_after_cost"] = edge_after_cost
    labeled["meta_label"] = meta_label
    labeled["meta_label_active"] = active.astype(bool)
    return labeled


def add_forward_labels(data: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Add compatible binary, edge, event, and risk-adjusted forward labels."""

    required = {"close", "event_return_threshold", "edge_return_threshold"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Cannot build labels; missing columns: {missing}")

    labeled = data.copy()
    close = pd.to_numeric(labeled["close"], errors="coerce")
    horizon = int(config.horizon)
    min_return = float(config.min_return)

    labeled["future_return"] = close.shift(-horizon) / close - 1.0
    labeled["future_realized_volatility"] = future_realized_volatility(close, horizon)
    volatility_denominator = labeled["future_realized_volatility"].clip(lower=config.volatility_floor)
    labeled["future_risk_adjusted_return"] = labeled["future_return"] / volatility_denominator
    labeled["risk_adjusted_long_target"] = (
        labeled["future_risk_adjusted_return"] > config.risk_adjusted_threshold
    ).astype(int)
    labeled["risk_adjusted_short_target"] = (
        labeled["future_risk_adjusted_return"] < -config.risk_adjusted_threshold
    ).astype(int)

    labeled["long_target"] = (labeled["future_return"] > min_return).astype(int)
    labeled["short_target"] = (labeled["future_return"] < -min_return).astype(int)
    labeled["target"] = labeled["long_target"]
    labeled["tradable_label"] = labeled["future_return"].abs() >= min_return
    labeled["actionable_label"] = labeled["future_return"].abs() >= (2.0 * min_return)
    labeled["long_edge_after_cost"] = labeled["future_return"] - labeled["edge_return_threshold"]
    labeled["short_edge_after_cost"] = -labeled["future_return"] - labeled["edge_return_threshold"]
    labeled["absolute_edge_after_cost"] = pd.concat(
        [labeled["long_edge_after_cost"], labeled["short_edge_after_cost"]],
        axis=1,
    ).max(axis=1)
    labeled["future_return_net_long"] = labeled["long_edge_after_cost"]
    labeled["future_return_net_short"] = labeled["short_edge_after_cost"]
    labeled["future_return_net_edge"] = labeled["absolute_edge_after_cost"]
    labeled["edge_long_target"] = (labeled["long_edge_after_cost"] > 0).astype(int)
    labeled["edge_short_target"] = (labeled["short_edge_after_cost"] > 0).astype(int)
    labeled["edge_trade_target"] = (labeled["absolute_edge_after_cost"] > 0).astype(int)
    labeled["long_target_net"] = labeled["edge_long_target"]
    labeled["short_target_net"] = labeled["edge_short_target"]
    labeled["tradable_label_net"] = labeled["edge_trade_target"].astype(bool)
    labeled["actionable_label_net"] = (
        labeled["absolute_edge_after_cost"] >= abs(min_return)
    ).astype(bool)
    labeled["big_up_target"] = (
        labeled["future_return"] > labeled["event_return_threshold"]
    ).astype(int)
    labeled["big_down_target"] = (
        labeled["future_return"] < -labeled["event_return_threshold"]
    ).astype(int)
    labeled["big_move_target"] = (
        labeled["future_return"].abs() >= labeled["event_return_threshold"]
    ).astype(int)
    if config.meta_side_column is not None:
        labeled = add_meta_labels(
            labeled,
            MetaLabelConfig(side_column=config.meta_side_column),
        )

    horizon_columns: dict[str, pd.Series] = {}
    for step in config.multi_horizon_steps:
        future_return = close.shift(-step) / close - 1.0
        future_volatility = future_realized_volatility(close, step)
        risk_adjusted_return = future_return / future_volatility.clip(lower=config.volatility_floor)
        scaled_event_threshold = labeled["event_return_threshold"] * np.sqrt(float(step))
        scaled_edge_threshold = labeled["edge_return_threshold"] * np.sqrt(float(step))
        net_long = future_return - scaled_edge_threshold
        net_short = -future_return - scaled_edge_threshold
        net_edge = pd.concat([net_long, net_short], axis=1).max(axis=1)

        horizon_columns[f"future_return_h{step}"] = future_return
        horizon_columns[f"future_realized_volatility_h{step}"] = future_volatility
        horizon_columns[f"future_risk_adjusted_return_h{step}"] = risk_adjusted_return
        horizon_columns[f"risk_adjusted_long_target_h{step}"] = (
            risk_adjusted_return > config.risk_adjusted_threshold
        ).astype(int)
        horizon_columns[f"risk_adjusted_short_target_h{step}"] = (
            risk_adjusted_return < -config.risk_adjusted_threshold
        ).astype(int)
        horizon_columns[f"target_h{step}"] = (future_return > min_return).astype(int)
        horizon_columns[f"long_target_h{step}"] = horizon_columns[f"target_h{step}"]
        horizon_columns[f"short_target_h{step}"] = (future_return < -min_return).astype(int)
        horizon_columns[f"future_return_net_long_h{step}"] = net_long
        horizon_columns[f"future_return_net_short_h{step}"] = net_short
        horizon_columns[f"future_return_net_edge_h{step}"] = net_edge
        horizon_columns[f"target_net_h{step}"] = (net_long > 0).astype(int)
        horizon_columns[f"edge_long_target_h{step}"] = (net_long > 0).astype(int)
        horizon_columns[f"edge_short_target_h{step}"] = (net_short > 0).astype(int)
        horizon_columns[f"edge_trade_target_h{step}"] = (net_edge > 0).astype(int)
        horizon_columns[f"short_target_net_h{step}"] = (net_short > 0).astype(int)
        horizon_columns[f"tradable_label_net_h{step}"] = (net_edge > 0).astype(bool)
        horizon_columns[f"actionable_label_net_h{step}"] = (
            net_edge >= abs(min_return)
        ).astype(bool)
        horizon_columns[f"big_up_target_h{step}"] = (
            future_return > scaled_event_threshold
        ).astype(int)
        horizon_columns[f"big_down_target_h{step}"] = (
            future_return < -scaled_event_threshold
        ).astype(int)
        horizon_columns[f"big_move_target_h{step}"] = (
            future_return.abs() >= scaled_event_threshold
        ).astype(int)
        horizon_columns[f"tradable_label_h{step}"] = future_return.abs() >= min_return
        horizon_columns[f"actionable_label_h{step}"] = (
            future_return.abs() >= (2.0 * min_return)
        )

    if horizon_columns:
        labeled = pd.concat(
            [labeled, pd.DataFrame(horizon_columns, index=labeled.index)],
            axis=1,
        )
    return labeled


def label_columns(config: LabelConfig) -> list[str]:
    """Return label columns emitted by `add_forward_labels`."""

    columns = [
        "future_return",
        "future_realized_volatility",
        "future_risk_adjusted_return",
        "risk_adjusted_long_target",
        "risk_adjusted_short_target",
        "target",
        "long_target",
        "short_target",
        "tradable_label",
        "actionable_label",
        "event_return_threshold",
        "edge_return_threshold",
        "label_cost_buffer",
        "long_edge_after_cost",
        "short_edge_after_cost",
        "absolute_edge_after_cost",
        "future_return_net_long",
        "future_return_net_short",
        "future_return_net_edge",
        "edge_long_target",
        "edge_short_target",
        "edge_trade_target",
        "long_target_net",
        "short_target_net",
        "tradable_label_net",
        "actionable_label_net",
        "big_up_target",
        "big_down_target",
        "big_move_target",
    ]
    for step in config.multi_horizon_steps:
        columns.extend(
            [
                f"future_return_h{step}",
                f"future_realized_volatility_h{step}",
                f"future_risk_adjusted_return_h{step}",
                f"risk_adjusted_long_target_h{step}",
                f"risk_adjusted_short_target_h{step}",
                f"target_h{step}",
                f"long_target_h{step}",
                f"short_target_h{step}",
                f"future_return_net_long_h{step}",
                f"future_return_net_short_h{step}",
                f"future_return_net_edge_h{step}",
                f"target_net_h{step}",
                f"edge_long_target_h{step}",
                f"edge_short_target_h{step}",
                f"edge_trade_target_h{step}",
                f"short_target_net_h{step}",
                f"tradable_label_net_h{step}",
                f"actionable_label_net_h{step}",
                f"big_up_target_h{step}",
                f"big_down_target_h{step}",
                f"big_move_target_h{step}",
                f"tradable_label_h{step}",
                f"actionable_label_h{step}",
            ]
        )
    if config.meta_side_column is not None:
        columns.extend(META_LABEL_COLUMNS)
    return columns
