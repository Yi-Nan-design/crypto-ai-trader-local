from __future__ import annotations

import numpy as np
import pandas as pd

from .labeling import LabelConfig, MULTI_HORIZON_STEPS, add_forward_labels, label_columns


FEATURE_VERSION = "v12_grouped_context_features"

EXECUTION_CONTEXT_COLUMNS = [
    "funding_payment_rate",
    "funding_rate_8h",
    "exchange_available",
    "exchange_gap_before_bars",
    "exchange_unavailable_reason",
]

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_48",
    "return_96",
    "log_volume_change",
    "quote_volume_z",
    "trade_count_z",
    "high_low_range",
    "close_open_range",
    "ema_fast_gap",
    "ema_slow_gap",
    "trend_strength_12_48",
    "trend_strength_48_192",
    "rsi_14",
    "volatility_12",
    "volatility_24",
    "realized_volatility_24",
    "taker_buy_ratio",
    "volume_pressure",
    "funding_rate_level",
    "funding_rate_change",
    "macd_line",
    "macd_signal_gap",
    "bollinger_z_20",
    "bollinger_bandwidth_20",
    "atr_14",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_ratio",
    "spread_proxy",
    "vwap_gap_24",
    "volume_ma_ratio_24",
    "price_volume_divergence_24",
    "volatility_ratio_12_48",
    "volatility_ratio_24_96",
    "volatility_ratio_48_192",
    "return_1_z_48",
    "efficiency_ratio_48",
    "efficiency_ratio_192",
    "micro_trend_regime",
    "range_position_20",
    "range_position_96",
    "drawdown_from_high_96",
    "breakout_gap_20",
    "breakdown_gap_20",
    "breakout_gap_96",
    "breakdown_gap_96",
    "mean_reversion_pressure_20",
    "range_compression_20_96",
    "range_bound_score_20",
    "grid_reversion_long_score_20",
    "grid_reversion_short_score_20",
    "trend_breakout_score_20",
    "trend_breakdown_score_20",
    "taker_pressure_3",
    "taker_pressure_change_3",
    "liquidity_shock_24",
    "atr_regime_14_96",
    "volume_ma_ratio_96",
    "higher_tf_trend_alignment_long",
    "higher_tf_trend_alignment_short",
    "platform_event_long_score",
    "platform_event_short_score",
    "platform_strategy_long_score",
    "platform_strategy_short_score",
    "crowding_long_risk",
    "crowding_short_risk",
    "liquidity_quality_score",
    "trend_quality_long",
    "trend_quality_short",
    "range_quality_long",
    "range_quality_short",
    "event_volatility_budget",
    "volume_price_impulse",
    "breakout_followthrough_long_3",
    "breakout_followthrough_short_3",
    "exhaustion_reversal_long_score",
    "exhaustion_reversal_short_score",
    "hour_sin",
    "hour_cos",
]


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(
    df: pd.DataFrame,
    label_horizon: int = 3,
    label_min_return: float = 0.001,
    *,
    drop_future_na: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    close = data["close"]
    data["return_1"] = close.pct_change(1)
    data["return_3"] = close.pct_change(3)
    data["return_6"] = close.pct_change(6)
    data["return_12"] = close.pct_change(12)
    data["return_24"] = close.pct_change(24)
    data["return_48"] = close.pct_change(48)
    data["return_96"] = close.pct_change(96)
    data["log_volume_change"] = np.log1p(data["volume"]).diff()
    data["quote_volume_z"] = _zscore(np.log1p(data["quote_volume"]), 48)
    data["trade_count_z"] = _zscore(np.log1p(data["trades"]), 48)
    data["high_low_range"] = (data["high"] - data["low"]) / close.replace(0, np.nan)
    data["close_open_range"] = (data["close"] - data["open"]) / data["open"].replace(0, np.nan)
    data["spread_proxy"] = data["high_low_range"]

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=48, adjust=False).mean()
    ema_higher = close.ewm(span=192, adjust=False).mean()
    data["ema_fast_gap"] = (close - ema_fast) / close.replace(0, np.nan)
    data["ema_slow_gap"] = (close - ema_slow) / close.replace(0, np.nan)
    data["trend_strength_12_48"] = (ema_fast - ema_slow) / close.replace(0, np.nan)
    data["trend_strength_48_192"] = (ema_slow - ema_higher) / close.replace(0, np.nan)
    data["rsi_14"] = _rsi(close, 14) / 100.0
    data["volatility_12"] = data["return_1"].rolling(12).std(ddof=0)
    data["volatility_24"] = data["return_1"].rolling(24).std(ddof=0)
    data["realized_volatility_24"] = np.sqrt(
        data["return_1"].pow(2).rolling(24).sum()
    )
    volatility_48 = data["return_1"].rolling(48).std(ddof=0)
    volatility_96 = data["return_1"].rolling(96).std(ddof=0)
    volatility_192 = data["return_1"].rolling(192).std(ddof=0)
    data["volatility_ratio_12_48"] = data["volatility_12"] / volatility_48.replace(0, np.nan)
    data["volatility_ratio_24_96"] = data["volatility_24"] / volatility_96.replace(0, np.nan)
    data["volatility_ratio_48_192"] = volatility_48 / volatility_192.replace(0, np.nan)
    data["return_1_z_48"] = data["return_1"] / volatility_48.replace(0, np.nan)
    path_return_48 = data["return_1"].abs().rolling(48).sum()
    path_return_192 = data["return_1"].abs().rolling(192).sum()
    data["efficiency_ratio_48"] = (data["return_48"].abs() / path_return_48.replace(0, np.nan)).clip(0.0, 1.0)
    data["efficiency_ratio_192"] = (close.pct_change(192).abs() / path_return_192.replace(0, np.nan)).clip(0.0, 1.0)
    data["micro_trend_regime"] = data["efficiency_ratio_48"] * np.sign(data["return_24"])

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_ratio = (ema_12 - ema_26) / close.replace(0, np.nan)
    data["macd_line"] = macd_ratio
    data["macd_signal_gap"] = macd_ratio - macd_ratio.ewm(span=9, adjust=False).mean()

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std(ddof=0)
    data["bollinger_z_20"] = (close - sma_20) / std_20.replace(0, np.nan)
    data["bollinger_bandwidth_20"] = (4.0 * std_20) / close.replace(0, np.nan)
    rolling_high_20 = data["high"].rolling(20).max()
    rolling_low_20 = data["low"].rolling(20).min()
    rolling_high_96 = data["high"].rolling(96).max()
    rolling_low_96 = data["low"].rolling(96).min()
    range_width_20 = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    range_width_96 = (rolling_high_96 - rolling_low_96).replace(0, np.nan)
    data["range_position_20"] = (close - rolling_low_20) / range_width_20
    data["range_position_96"] = (close - rolling_low_96) / range_width_96
    data["drawdown_from_high_96"] = close / rolling_high_96.replace(
        0,
        np.nan,
    ) - 1.0
    data["breakout_gap_20"] = close / rolling_high_20.shift(1).replace(0, np.nan) - 1.0
    data["breakdown_gap_20"] = close / rolling_low_20.shift(1).replace(0, np.nan) - 1.0
    data["breakout_gap_96"] = close / rolling_high_96.shift(1).replace(0, np.nan) - 1.0
    data["breakdown_gap_96"] = close / rolling_low_96.shift(1).replace(0, np.nan) - 1.0
    data["mean_reversion_pressure_20"] = -data["bollinger_z_20"] * (1.0 - data["efficiency_ratio_48"])
    bandwidth_mean_96 = data["bollinger_bandwidth_20"].rolling(96).mean().replace(0, np.nan)
    data["range_compression_20_96"] = (1.0 - data["bollinger_bandwidth_20"] / bandwidth_mean_96).clip(-2.0, 2.0)
    range_bound = (1.0 - data["efficiency_ratio_48"]).clip(0.0, 1.0)
    compressed_range = data["range_compression_20_96"].clip(0.0, 1.0)
    data["range_bound_score_20"] = (range_bound * (0.5 + 0.5 * compressed_range)).clip(0.0, 1.0)
    lower_extreme = ((0.45 - data["range_position_20"]) / 0.45).clip(0.0, 1.0)
    upper_extreme = ((data["range_position_20"] - 0.55) / 0.45).clip(0.0, 1.0)
    data["grid_reversion_long_score_20"] = (
        data["range_bound_score_20"] * lower_extreme * (-data["bollinger_z_20"] / 2.0).clip(0.0, 1.0)
    )
    data["grid_reversion_short_score_20"] = (
        data["range_bound_score_20"] * upper_extreme * (data["bollinger_z_20"] / 2.0).clip(0.0, 1.0)
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.rolling(14).mean() / close.replace(0, np.nan)
    data["atr_regime_14_96"] = data["atr_14"] / data["atr_14"].rolling(96).mean().replace(0, np.nan) - 1.0
    candle_range_base = close.replace(0, np.nan)
    data["upper_shadow_ratio"] = (data["high"] - pd.concat([data["open"], close], axis=1).max(axis=1)) / candle_range_base
    data["lower_shadow_ratio"] = (pd.concat([data["open"], close], axis=1).min(axis=1) - data["low"]) / candle_range_base
    data["body_ratio"] = (close - data["open"]).abs() / data["open"].replace(0, np.nan)

    taker_quote = pd.to_numeric(data.get("taker_buy_quote_volume", 0), errors="coerce")
    quote_volume = data["quote_volume"].replace(0, np.nan)
    data["taker_buy_ratio"] = taker_quote / quote_volume
    data["volume_pressure"] = (data["taker_buy_ratio"] - 0.5) * data["quote_volume_z"]
    funding_source = pd.to_numeric(
        data.get(
            "funding_rate_8h",
            pd.Series(np.nan, index=data.index),
        ),
        errors="coerce",
    )
    data["funding_rate_level"] = funding_source.ffill().fillna(0.0)
    data["funding_rate_change"] = data["funding_rate_level"].diff().fillna(0.0)
    data["taker_pressure_3"] = data["taker_buy_ratio"].rolling(3).mean() - 0.5
    data["taker_pressure_change_3"] = data["taker_pressure_3"] - data["taker_pressure_3"].shift(3)
    rolling_volume = data["volume"].rolling(24).sum()
    rolling_quote = data["quote_volume"].rolling(24).sum()
    rolling_vwap = rolling_quote / rolling_volume.replace(0, np.nan)
    data["vwap_gap_24"] = (close - rolling_vwap) / close.replace(0, np.nan)
    data["volume_ma_ratio_24"] = data["volume"] / data["volume"].rolling(24).mean().replace(0, np.nan) - 1.0
    data["price_volume_divergence_24"] = data["return_1"].rolling(24).corr(
        data["log_volume_change"]
    )
    data["volume_ma_ratio_96"] = data["volume"] / data["volume"].rolling(96).mean().replace(0, np.nan) - 1.0
    data["liquidity_shock_24"] = data["quote_volume_z"] * data["volume_ma_ratio_24"].clip(-3.0, 3.0)
    liquidity_boost = (1.0 + data["volume_ma_ratio_24"].clip(0.0, 3.0)).fillna(1.0)
    signed_volume_boost = (1.0 + data["quote_volume_z"].clip(0.0, 3.0)).fillna(1.0)
    data["volume_price_impulse"] = data["return_1_z_48"].clip(-5.0, 5.0) * signed_volume_boost
    data["trend_breakout_score_20"] = (
        data["breakout_gap_20"].clip(0.0, 0.05)
        * data["efficiency_ratio_48"].clip(0.0, 1.0)
        * liquidity_boost
        * (1.0 + data["taker_pressure_3"].clip(0.0, 0.5))
    )
    data["trend_breakdown_score_20"] = (
        (-data["breakdown_gap_20"]).clip(0.0, 0.05)
        * data["efficiency_ratio_48"].clip(0.0, 1.0)
        * liquidity_boost
        * (1.0 + (-data["taker_pressure_3"]).clip(0.0, 0.5))
    )
    data["breakout_followthrough_long_3"] = (
        data["return_3"].clip(0.0, 0.05)
        * data["efficiency_ratio_48"].clip(0.0, 1.0)
        * liquidity_boost
        * (1.0 + data["taker_pressure_change_3"].clip(0.0, 0.5))
    )
    data["breakout_followthrough_short_3"] = (
        (-data["return_3"]).clip(0.0, 0.05)
        * data["efficiency_ratio_48"].clip(0.0, 1.0)
        * liquidity_boost
        * (1.0 + (-data["taker_pressure_change_3"]).clip(0.0, 0.5))
    )
    data["exhaustion_reversal_long_score"] = (
        data["lower_shadow_ratio"].clip(0.0, 0.03)
        * lower_extreme
        * (1.0 - data["efficiency_ratio_48"]).clip(0.0, 1.0)
        * signed_volume_boost
    )
    data["exhaustion_reversal_short_score"] = (
        data["upper_shadow_ratio"].clip(0.0, 0.03)
        * upper_extreme
        * (1.0 - data["efficiency_ratio_48"]).clip(0.0, 1.0)
        * signed_volume_boost
    )
    higher_tf_long = (
        data["trend_strength_12_48"].clip(0.0, 0.05)
        * data["trend_strength_48_192"].clip(0.0, 0.05)
        * (0.5 + data["efficiency_ratio_192"].clip(0.0, 1.0))
    )
    higher_tf_short = (
        (-data["trend_strength_12_48"]).clip(0.0, 0.05)
        * (-data["trend_strength_48_192"]).clip(0.0, 0.05)
        * (0.5 + data["efficiency_ratio_192"].clip(0.0, 1.0))
    )
    data["higher_tf_trend_alignment_long"] = (higher_tf_long / 0.0004).clip(0.0, 1.0)
    data["higher_tf_trend_alignment_short"] = (higher_tf_short / 0.0004).clip(0.0, 1.0)
    liquidity_quality = (
        0.35
        + 0.35 * (data["volume_ma_ratio_24"].clip(-0.75, 2.0) + 0.75) / 2.75
        + 0.30 * (data["quote_volume_z"].clip(-1.5, 2.5) + 1.5) / 4.0
    ).clip(0.0, 1.0)
    data["liquidity_quality_score"] = liquidity_quality
    rsi_overbought = ((data["rsi_14"] - 0.68) / 0.22).clip(0.0, 1.0)
    rsi_oversold = ((0.32 - data["rsi_14"]) / 0.22).clip(0.0, 1.0)
    upper_band_extreme = ((data["bollinger_z_20"] - 1.2) / 1.8).clip(0.0, 1.0)
    lower_band_extreme = ((-data["bollinger_z_20"] - 1.2) / 1.8).clip(0.0, 1.0)
    euphoric_taker = ((data["taker_pressure_3"] - 0.12) / 0.28).clip(0.0, 1.0)
    panic_taker = ((-data["taker_pressure_3"] - 0.12) / 0.28).clip(0.0, 1.0)
    data["crowding_long_risk"] = (
        0.40 * rsi_overbought + 0.35 * upper_band_extreme + 0.25 * euphoric_taker
    ).clip(0.0, 1.0)
    data["crowding_short_risk"] = (
        0.40 * rsi_oversold + 0.35 * lower_band_extreme + 0.25 * panic_taker
    ).clip(0.0, 1.0)
    data["trend_quality_long"] = (
        0.35 * data["higher_tf_trend_alignment_long"].clip(0.0, 1.0)
        + 0.25 * (data["trend_breakout_score_20"] / 0.00035).clip(0.0, 1.0)
        + 0.20 * (data["breakout_followthrough_long_3"] / 0.003).clip(0.0, 1.0)
        + 0.20 * liquidity_quality
    ).clip(0.0, 1.0)
    data["trend_quality_short"] = (
        0.35 * data["higher_tf_trend_alignment_short"].clip(0.0, 1.0)
        + 0.25 * (data["trend_breakdown_score_20"] / 0.00035).clip(0.0, 1.0)
        + 0.20 * (data["breakout_followthrough_short_3"] / 0.003).clip(0.0, 1.0)
        + 0.20 * liquidity_quality
    ).clip(0.0, 1.0)
    data["range_quality_long"] = (
        0.55 * data["grid_reversion_long_score_20"].clip(0.0, 1.0)
        + 0.25 * (data["exhaustion_reversal_long_score"] / 0.012).clip(0.0, 1.0)
        + 0.20 * liquidity_quality
    ).clip(0.0, 1.0)
    data["range_quality_short"] = (
        0.55 * data["grid_reversion_short_score_20"].clip(0.0, 1.0)
        + 0.25 * (data["exhaustion_reversal_short_score"] / 0.012).clip(0.0, 1.0)
        + 0.20 * liquidity_quality
    ).clip(0.0, 1.0)
    data["platform_strategy_long_score"] = pd.concat(
        [data["trend_quality_long"], data["range_quality_long"]],
        axis=1,
    ).max(axis=1).mul(1.0 - 0.45 * data["crowding_long_risk"]).clip(0.0, 1.0)
    data["platform_strategy_short_score"] = pd.concat(
        [data["trend_quality_short"], data["range_quality_short"]],
        axis=1,
    ).max(axis=1).mul(1.0 - 0.45 * data["crowding_short_risk"]).clip(0.0, 1.0)
    data["platform_event_long_score"] = pd.concat(
        [
            (data["trend_breakout_score_20"] / 0.00035).clip(0.0, 1.0),
            (data["breakout_followthrough_long_3"] / 0.003).clip(0.0, 1.0),
            data["grid_reversion_long_score_20"].clip(0.0, 1.0),
            (data["exhaustion_reversal_long_score"] / 0.012).clip(0.0, 1.0),
            data["higher_tf_trend_alignment_long"].clip(0.0, 1.0),
            data["platform_strategy_long_score"],
        ],
        axis=1,
    ).max(axis=1)
    data["platform_event_short_score"] = pd.concat(
        [
            (data["trend_breakdown_score_20"] / 0.00035).clip(0.0, 1.0),
            (data["breakout_followthrough_short_3"] / 0.003).clip(0.0, 1.0),
            data["grid_reversion_short_score_20"].clip(0.0, 1.0),
            (data["exhaustion_reversal_short_score"] / 0.012).clip(0.0, 1.0),
            data["higher_tf_trend_alignment_short"].clip(0.0, 1.0),
            data["platform_strategy_short_score"],
        ],
        axis=1,
    ).max(axis=1)

    rolling_event_atr = data["atr_14"].rolling(96).median()
    event_threshold = np.maximum(
        1.5 * abs(float(label_min_return)),
        (0.85 * rolling_event_atr).fillna(abs(float(label_min_return))).to_numpy(dtype=float),
    )
    edge_threshold = np.minimum(
        event_threshold,
        np.maximum(
            1.15 * abs(float(label_min_return)),
            (0.45 * rolling_event_atr).fillna(abs(float(label_min_return))).to_numpy(dtype=float),
        ),
    )
    data["event_return_threshold"] = event_threshold
    data["edge_return_threshold"] = edge_threshold
    data["label_cost_buffer"] = edge_threshold
    data["event_volatility_budget"] = data["atr_14"] / pd.Series(event_threshold, index=data.index).replace(0, np.nan)

    if "open_datetime" in data.columns:
        hours = pd.to_datetime(data["open_datetime"], utc=True).dt.hour
        data["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    label_config = LabelConfig(
        horizon=label_horizon,
        min_return=label_min_return,
        multi_horizon_steps=MULTI_HORIZON_STEPS,
    )
    data = add_forward_labels(data, label_config)

    keep = [
        "open_time",
        "open_datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
    ]
    keep += label_columns(label_config)
    keep += FEATURE_COLUMNS
    keep += EXECUTION_CONTEXT_COLUMNS
    existing = [col for col in keep if col in data.columns]
    data = data[existing].replace([np.inf, -np.inf], np.nan)
    if drop_future_na:
        required = [
            col
            for col in existing
            if col not in EXECUTION_CONTEXT_COLUMNS
        ]
        data = data.dropna(subset=required)
    else:
        subset = [col for col in ["open_time", "open", "high", "low", "close"] + FEATURE_COLUMNS if col in data.columns]
        data = data.dropna(subset=subset)
    data = data.reset_index(drop=True)
    return data


def feature_matrix(
    feature_frame: pd.DataFrame,
    target_col: str = "target",
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    available = feature_columns or [col for col in FEATURE_COLUMNS if col in feature_frame.columns]
    x = feature_frame[available].to_numpy(dtype=float)
    y = feature_frame[target_col].to_numpy(dtype=int)
    return x, y, available


def feature_only_matrix(feature_frame: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    available = feature_columns or [col for col in FEATURE_COLUMNS if col in feature_frame.columns]
    x = feature_frame[available].to_numpy(dtype=float)
    return x, available


def time_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction + validation_fraction >= 0.95:
        raise ValueError("train_fraction + validation_fraction must leave room for test data")
    n = len(frame)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    purge_rows = max(0, int(purge_rows or 0))
    train_slice_end = max(0, train_end - purge_rows)
    valid_slice_end = max(train_end, validation_end - purge_rows)
    return (
        frame.iloc[:train_slice_end].reset_index(drop=True),
        frame.iloc[train_end:valid_slice_end].reset_index(drop=True),
        frame.iloc[validation_end:].reset_index(drop=True),
    )
