from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureGroup:
    """Named feature family used for audit and model-card reporting."""

    name: str
    columns: tuple[str, ...]
    optional: bool = False


FEATURE_GROUPS = (
    FeatureGroup(
        "technical_features",
        (
            "return_1",
            "return_3",
            "return_6",
            "return_12",
            "return_24",
            "return_48",
            "return_96",
            "ema_fast_gap",
            "ema_slow_gap",
            "rsi_14",
            "macd_line",
            "macd_signal_gap",
        ),
    ),
    FeatureGroup(
        "volatility_features",
        (
            "volatility_12",
            "volatility_24",
            "realized_volatility_24",
            "atr_14",
            "volatility_ratio_12_48",
            "volatility_ratio_24_96",
            "volatility_ratio_48_192",
        ),
    ),
    FeatureGroup(
        "volume_features",
        (
            "log_volume_change",
            "quote_volume_z",
            "trade_count_z",
            "volume_ma_ratio_24",
            "volume_ma_ratio_96",
            "price_volume_divergence_24",
        ),
    ),
    FeatureGroup(
        "funding_features",
        ("funding_rate_level", "funding_rate_change"),
    ),
    FeatureGroup(
        "open_interest_features",
        ("open_interest_change", "open_interest_z"),
        optional=True,
    ),
    FeatureGroup(
        "microstructure_features",
        (
            "taker_buy_ratio",
            "volume_pressure",
            "spread_proxy",
            "liquidity_quality_score",
        ),
    ),
    FeatureGroup(
        "cross_sectional_features",
        ("cross_sectional_momentum_rank", "btc_relative_strength"),
        optional=True,
    ),
    FeatureGroup(
        "regime_features",
        (
            "trend_strength_12_48",
            "trend_strength_48_192",
            "efficiency_ratio_48",
            "micro_trend_regime",
            "drawdown_from_high_96",
        ),
    ),
)


def feature_group_map() -> dict[str, tuple[str, ...]]:
    """Return immutable feature groups for reports and validation."""

    return {group.name: group.columns for group in FEATURE_GROUPS}


def add_optional_open_interest_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add causal open-interest features when the source column is available."""

    if "open_interest" not in data.columns:
        return data.copy()
    output = data.copy()
    open_interest = pd.to_numeric(
        output["open_interest"],
        errors="coerce",
    )
    output["open_interest_change"] = open_interest.pct_change()
    mean = open_interest.rolling(48).mean()
    std = open_interest.rolling(48).std(ddof=0).replace(0.0, np.nan)
    output["open_interest_z"] = (open_interest - mean) / std
    return output


def add_cross_sectional_features(
    data: pd.DataFrame,
    *,
    timestamp_column: str = "open_time",
    symbol_column: str = "symbol",
    momentum_column: str = "return_24",
    btc_symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Add same-timestamp momentum rank and BTC-relative strength."""

    required = {timestamp_column, symbol_column, momentum_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Cannot build cross-sectional features; missing columns: {missing}"
        )
    output = data.copy()
    momentum = pd.to_numeric(output[momentum_column], errors="coerce")
    output["cross_sectional_momentum_rank"] = momentum.groupby(
        output[timestamp_column]
    ).rank(method="average", pct=True)
    btc_rows = output[symbol_column].astype(str).str.upper() == btc_symbol.upper()
    btc_momentum = (
        pd.Series(
            np.where(btc_rows, momentum, np.nan),
            index=output.index,
        )
        .groupby(output[timestamp_column])
        .transform("max")
    )
    output["btc_relative_strength"] = momentum - btc_momentum
    return output
