from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .contracts import (
    LiquidityState,
    MarketRegime,
    RegimeState,
    VolatilityState,
)
from .statistical_regime import (
    STATISTICAL_REGIME_VERSION,
    walk_forward_kmeans_regimes,
)


RULE_BASED_REGIME_VERSION = "2026-06-20-rule-based-v1"
SUPPORTED_REGIME_METHODS = {"rule_based", "walk_forward_kmeans"}


class RegimeConfig(Protocol):
    """Configuration fields used by rule-based and optional statistical regimes."""

    regime_detection_method: str
    trend_gate_min_efficiency: float
    volatility_regime_lookback: int
    volatility_regime_low_quantile: float
    volatility_regime_high_quantile: float
    regime_crash_return_3: float
    regime_crash_return_24: float
    regime_liquidity_z_threshold: float
    regime_liquidity_quality_threshold: float


def _numeric_column(
    frame: pd.DataFrame,
    name: str,
    default: float,
) -> pd.Series:
    source = frame[name] if name in frame.columns else pd.Series(default, index=frame.index)
    return (
        pd.to_numeric(source, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def causal_volatility_thresholds(
    atr: pd.Series,
    *,
    lookback: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.Series, pd.Series]:
    """Build trailing ATR thresholds without reading future observations."""

    values = pd.to_numeric(atr, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lookback = max(30, int(lookback))
    min_periods = max(20, min(lookback // 3, 80))
    low_quantile = float(np.clip(low_quantile, 0.05, 0.49))
    high_quantile = float(np.clip(high_quantile, 0.51, 0.95))

    rolling_low = values.rolling(lookback, min_periods=min_periods).quantile(low_quantile).shift(1)
    rolling_high = values.rolling(lookback, min_periods=min_periods).quantile(high_quantile).shift(1)
    expanding_low = values.expanding(min_periods=2).quantile(low_quantile).shift(1)
    expanding_high = values.expanding(min_periods=2).quantile(high_quantile).shift(1)
    historical_median = values.expanding(min_periods=1).median().shift(1)
    current_fallback = values.ffill().fillna(0.0)

    raw_low_cut = rolling_low.fillna(expanding_low).fillna(historical_median).fillna(current_fallback)
    raw_high_cut = rolling_high.fillna(expanding_high).fillna(historical_median).fillna(current_fallback)
    bounds = pd.concat([raw_low_cut, raw_high_cut], axis=1)
    low_cut = bounds.min(axis=1)
    high_cut = bounds.max(axis=1)
    return low_cut.astype(float), high_cut.astype(float)


def _detect_rule_based_regime_frame(
    frame: pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.DataFrame:
    """Classify each row with a causal, explainable market-state baseline."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "market_regime",
                "regime_confidence",
                "regime_risk_off",
                "volatility_state",
                "liquidity_state",
                "regime_reason",
            ],
            index=frame.index,
        )

    atr = _numeric_column(frame, "atr_14", 0.0)
    low_cut, high_cut = causal_volatility_thresholds(
        atr,
        lookback=int(getattr(cfg, "volatility_regime_lookback", 240)),
        low_quantile=float(getattr(cfg, "volatility_regime_low_quantile", 0.33)),
        high_quantile=float(getattr(cfg, "volatility_regime_high_quantile", 0.67)),
    )
    atr_values = atr.to_numpy(dtype=float)
    low_values = low_cut.to_numpy(dtype=float)
    high_values = high_cut.to_numpy(dtype=float)
    low_vol = atr_values < low_values
    high_vol = (atr_values > high_values) & (atr_values > 0.0)
    volatility_state = np.full(len(frame), VolatilityState.MID.value, dtype=object)
    volatility_state[low_vol] = VolatilityState.LOW.value
    volatility_state[high_vol] = VolatilityState.HIGH.value

    liquidity_z = _numeric_column(frame, "quote_volume_z", 0.0).to_numpy(dtype=float)
    liquidity_quality = _numeric_column(frame, "liquidity_quality_score", 0.5).to_numpy(dtype=float)
    crisis_z = float(getattr(cfg, "regime_liquidity_z_threshold", -2.0))
    crisis_quality = float(getattr(cfg, "regime_liquidity_quality_threshold", 0.15))
    liquidity_crisis = (liquidity_z <= crisis_z) | (liquidity_quality <= crisis_quality)
    thin_liquidity = (~liquidity_crisis) & ((liquidity_z < -0.8) | (liquidity_quality < 0.35))
    liquidity_state = np.full(len(frame), LiquidityState.NORMAL.value, dtype=object)
    liquidity_state[thin_liquidity] = LiquidityState.THIN.value
    liquidity_state[liquidity_crisis] = LiquidityState.CRISIS.value

    return_3 = _numeric_column(frame, "return_3", 0.0).to_numpy(dtype=float)
    return_24 = _numeric_column(frame, "return_24", 0.0).to_numpy(dtype=float)
    crash_3_limit = -abs(float(getattr(cfg, "regime_crash_return_3", -0.02)))
    crash_24_limit = -abs(float(getattr(cfg, "regime_crash_return_24", -0.04)))
    crash = (return_3 <= crash_3_limit) | (return_24 <= crash_24_limit)

    efficiency = _numeric_column(frame, "efficiency_ratio_48", 0.0).to_numpy(dtype=float)
    trend = _numeric_column(frame, "trend_strength_12_48", 0.0).to_numpy(dtype=float)
    trend_threshold = max(float(getattr(cfg, "trend_gate_min_efficiency", 0.18)), 1e-9)
    trending = efficiency >= trend_threshold
    trend_up = trending & (trend > 0.0)
    trend_down = trending & (trend < 0.0)

    regime = np.full(len(frame), MarketRegime.RANGE.value, dtype=object)
    reason = np.full(len(frame), "low_directional_efficiency", dtype=object)
    regime[trend_up] = MarketRegime.TREND_UP.value
    reason[trend_up] = "positive_trend_with_directional_efficiency"
    regime[trend_down] = MarketRegime.TREND_DOWN.value
    reason[trend_down] = "negative_trend_with_directional_efficiency"
    regime[high_vol] = MarketRegime.HIGH_VOL.value
    reason[high_vol] = "atr_above_trailing_high_quantile"
    regime[liquidity_crisis] = MarketRegime.LIQUIDITY_CRISIS.value
    reason[liquidity_crisis] = "liquidity_proxy_below_safety_threshold"
    regime[crash] = MarketRegime.CRASH.value
    reason[crash] = "negative_return_exceeded_crash_threshold"

    trend_confidence = np.clip(efficiency / trend_threshold, 0.0, 1.0)
    range_confidence = np.clip(1.0 - efficiency / trend_threshold, 0.0, 1.0)
    high_vol_confidence = np.clip(
        np.divide(
            atr_values,
            np.maximum(high_values, 1e-12),
            out=np.ones(len(frame), dtype=float),
            where=np.maximum(high_values, 1e-12) > 0,
        )
        - 1.0,
        0.0,
        1.0,
    )
    liquidity_confidence = np.maximum(
        np.clip((crisis_z - liquidity_z) / max(abs(crisis_z), 1e-9), 0.0, 1.0),
        np.clip((crisis_quality - liquidity_quality) / max(crisis_quality, 1e-9), 0.0, 1.0),
    )
    crash_confidence = np.maximum(
        np.clip(np.abs(np.minimum(return_3, 0.0)) / max(abs(crash_3_limit), 1e-9) - 1.0, 0.0, 1.0),
        np.clip(np.abs(np.minimum(return_24, 0.0)) / max(abs(crash_24_limit), 1e-9) - 1.0, 0.0, 1.0),
    )
    confidence = np.where(trending, trend_confidence, range_confidence)
    confidence = np.where(high_vol, np.maximum(0.5, high_vol_confidence), confidence)
    confidence = np.where(liquidity_crisis, np.maximum(0.5, liquidity_confidence), confidence)
    confidence = np.where(crash, np.maximum(0.5, crash_confidence), confidence)

    return pd.DataFrame(
        {
            "market_regime": regime,
            "regime_confidence": np.clip(confidence, 0.0, 1.0),
            "regime_risk_off": crash | liquidity_crisis,
            "volatility_state": volatility_state,
            "liquidity_state": liquidity_state,
            "regime_reason": reason,
        },
        index=frame.index,
    )


def detect_regime_frame(
    frame: pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.DataFrame:
    """Classify market state with a rule baseline and optional causal KMeans overlay."""

    requested_method = str(
        getattr(cfg, "regime_detection_method", "rule_based") or "rule_based"
    ).strip().lower()
    if requested_method not in SUPPORTED_REGIME_METHODS:
        raise ValueError(
            f"unsupported regime_detection_method: {requested_method}"
        )

    classified = _detect_rule_based_regime_frame(frame, cfg)
    classified["regime_method_requested"] = requested_method
    classified["regime_method_used"] = "rule_based"
    classified["regime_model_version"] = RULE_BASED_REGIME_VERSION
    classified["regime_fallback_reason"] = ""
    classified["regime_override_reason"] = ""
    classified["regime_cluster"] = pd.Series(
        pd.NA,
        index=classified.index,
        dtype="Int64",
    )
    if requested_method == "rule_based" or frame.empty:
        return classified

    statistical = walk_forward_kmeans_regimes(frame, cfg).frame
    available = statistical["statistical_regime"].notna()
    risk_override = available & classified["regime_risk_off"].astype(bool)
    use_statistical = available & ~risk_override

    classified.loc[use_statistical, "market_regime"] = statistical.loc[
        use_statistical,
        "statistical_regime",
    ]
    classified.loc[use_statistical, "regime_confidence"] = statistical.loc[
        use_statistical,
        "statistical_regime_confidence",
    ]
    classified.loc[use_statistical, "regime_reason"] = [
        f"walk_forward_kmeans_cluster_{int(cluster_id)}"
        for cluster_id in statistical.loc[
            use_statistical,
            "statistical_regime_cluster",
        ]
    ]
    classified.loc[use_statistical, "regime_method_used"] = "walk_forward_kmeans"
    classified.loc[use_statistical, "regime_model_version"] = (
        STATISTICAL_REGIME_VERSION
    )
    classified.loc[use_statistical, "regime_cluster"] = statistical.loc[
        use_statistical,
        "statistical_regime_cluster",
    ]

    fallback = ~available
    classified.loc[fallback, "regime_method_used"] = "rule_based_fallback"
    classified.loc[fallback, "regime_fallback_reason"] = statistical.loc[
        fallback,
        "statistical_regime_fallback_reason",
    ]
    classified.loc[risk_override, "regime_method_used"] = "rule_based_risk_override"
    classified.loc[risk_override, "regime_override_reason"] = "risk_override"
    classified.loc[risk_override, "regime_cluster"] = statistical.loc[
        risk_override,
        "statistical_regime_cluster",
    ]
    return classified


def latest_regime_state(frame: pd.DataFrame) -> RegimeState | None:
    """Convert the latest classified row into the shared regime contract."""

    if frame.empty or "market_regime" not in frame.columns:
        return None
    row = frame.iloc[-1]
    timestamp = row.get("open_datetime", row.get("open_time", ""))
    return RegimeState(
        timestamp=str(timestamp),
        regime=MarketRegime(str(row["market_regime"])),
        confidence=float(row.get("regime_confidence", 0.0)),
        volatility_state=VolatilityState(str(row.get("volatility_state", VolatilityState.MID.value))),
        liquidity_state=LiquidityState(str(row.get("liquidity_state", LiquidityState.NORMAL.value))),
        risk_off=bool(row.get("regime_risk_off", False)),
        reason=str(row.get("regime_reason", "unclassified")),
    )


def summarize_regime_performance(
    detail: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Summarize executed backtest performance by detected market regime."""

    required = {"market_regime", "strategy_return"}
    if detail.empty or not required.issubset(detail.columns):
        return {}
    summary: dict[str, dict[str, float | int]] = {}
    executed = (
        detail["executed_notional_position"].abs() > 1e-15
        if "executed_notional_position" in detail.columns
        else detail.get("position", pd.Series(0, index=detail.index)).abs() > 0
    )
    for label in sorted(str(value) for value in detail["market_regime"].dropna().unique()):
        mask = detail["market_regime"].astype(str) == label
        rows = detail.loc[mask]
        active_rows = rows.loc[executed.loc[mask]]
        returns = pd.to_numeric(rows["strategy_return"], errors="coerce").fillna(0.0)
        active_returns = pd.to_numeric(active_rows["strategy_return"], errors="coerce").fillna(0.0)
        wins = active_returns[active_returns > 0]
        losses = active_returns[active_returns < 0]
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (999.0 if gross_profit > 0 else 0.0)
        )
        equity = (1.0 + returns).cumprod().to_numpy(dtype=float)
        peak = np.maximum.accumulate(equity) if len(equity) else np.array([], dtype=float)
        drawdown = equity / peak - 1.0 if len(peak) else np.array([], dtype=float)
        turnover = (
            pd.to_numeric(rows["notional_turnover"], errors="coerce").fillna(0.0)
            if "notional_turnover" in rows.columns
            else pd.Series(0.0, index=rows.index)
        )
        costs = (
            pd.to_numeric(rows["total_cost"], errors="coerce").fillna(0.0)
            if "total_cost" in rows.columns
            else pd.Series(0.0, index=rows.index)
        )
        summary[label] = {
            "rows": int(len(rows)),
            "active_rows": int(len(active_rows)),
            "execution_events": int((turnover > 0).sum()),
            "total_return": float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0),
            "profit_factor": float(profit_factor),
            "win_rate": float(len(wins) / max(len(active_returns), 1)),
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "total_cost_drag": float(costs.sum()),
        }
    return summary
