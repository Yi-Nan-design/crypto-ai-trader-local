from __future__ import annotations

from enum import StrEnum
from typing import Protocol

import numpy as np
import pandas as pd

from .contracts import RiskDecision, RiskLevel, StrategyDecision
from .cost_model import infer_bar_hours


class RiskConfig(Protocol):
    """Configuration fields required by portfolio sizing and risk guards."""

    leverage: int
    max_allowed_leverage: int
    max_position_fraction: float
    max_notional_exposure: float
    stop_loss: float
    use_atr_exits: bool
    stop_loss_atr_multiplier: float
    min_exit_pct: float
    max_exit_pct: float
    risk_per_trade: float
    min_confidence_gap: float
    dynamic_position_sizing: bool
    min_position_scale: float
    volatility_target: float
    ewma_volatility_enabled: bool
    ewma_volatility_span: int
    ewma_daily_volatility_target: float
    min_volatility_scale: float
    max_volatility_scale: float
    event_position_boost_enabled: bool
    event_position_min_score: float
    event_position_boost_strength: float
    position_rebalance_band: float
    max_position_fraction_step: float
    drawdown_cooldown_enabled: bool
    cooldown_drawdown: float
    cooldown_loss_streak: int
    cooldown_bars: int
    funding_crowding_guard_enabled: bool
    funding_crowding_max_rate: float
    regime_risk_guard_enabled: bool


class RiskReason(StrEnum):
    """Stable reason codes emitted by the risk layer."""

    ALLOWED = "risk_allowed"
    NO_POSITION = "risk_no_position_requested"
    EXPOSURE_CAPPED = "risk_exposure_capped"
    LIQUIDITY_REDUCED = "risk_reduced_low_liquidity"
    VOLATILITY_REDUCED = "risk_reduced_high_volatility"
    FUNDING_CROWDING_BLOCKED = "risk_blocked_funding_crowding"
    REGIME_RISK_OFF_BLOCKED = "risk_blocked_regime_risk_off"
    DRAWDOWN_BLOCKED = "risk_blocked_drawdown_cooldown"
    ZERO_CAPACITY = "risk_blocked_zero_capacity"


def maximum_position_fraction(cfg: RiskConfig) -> float:
    """Return the strategy exposure cap after leverage/notional constraints."""

    leverage = max(float(cfg.leverage), 1.0)
    maximum = max(float(cfg.max_position_fraction), 0.0)
    max_notional = max(float(getattr(cfg, "max_notional_exposure", 0.0) or 0.0), 0.0)
    if max_notional > 0:
        maximum = min(maximum, max_notional / leverage)
    return maximum


def causal_atr_values(
    data: pd.DataFrame,
    *,
    fallback: float,
) -> np.ndarray:
    """Return ATR values using only current and previously observed rows."""

    if "atr_14" not in data.columns:
        return np.full(len(data), max(float(fallback), 1e-9), dtype=float)
    atr = pd.to_numeric(data["atr_14"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return (
        atr.ffill()
        .fillna(max(float(fallback), 1e-9))
        .to_numpy(dtype=float)
    )


def causal_ewma_volatility(
    data: pd.DataFrame,
    *,
    span: int,
    fallback: float,
) -> np.ndarray:
    """Return causal EWMA return volatility for each closed bar."""

    if "return_1" in data.columns:
        returns = pd.to_numeric(data["return_1"], errors="coerce")
    elif "close" in data.columns:
        close = pd.to_numeric(data["close"], errors="coerce")
        returns = close.pct_change()
    else:
        return np.full(len(data), max(float(fallback), 1e-9), dtype=float)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    effective_span = max(int(span), 2)
    mean = returns.ewm(
        span=effective_span,
        adjust=False,
        min_periods=2,
    ).mean()
    second_moment = returns.pow(2).ewm(
        span=effective_span,
        adjust=False,
        min_periods=2,
    ).mean()
    variance = (second_moment - mean.pow(2)).clip(lower=0.0)
    return (
        np.sqrt(variance)
        .ffill()
        .fillna(max(float(fallback), 1e-9))
        .to_numpy(dtype=float)
    )


def causal_ewma_daily_volatility(
    data: pd.DataFrame,
    *,
    span: int,
    fallback_daily: float,
) -> np.ndarray:
    """Return causal EWMA volatility scaled to a 24-hour horizon."""

    bar_hours = max(float(infer_bar_hours(data)), 1.0 / 60.0)
    scale = float(np.sqrt(24.0 / bar_hours))
    per_bar_fallback = max(float(fallback_daily) / scale, 1e-9)
    return causal_ewma_volatility(
        data,
        span=span,
        fallback=per_bar_fallback,
    ) * scale


def stop_distance_series(
    data: pd.DataFrame,
    cfg: RiskConfig,
) -> np.ndarray:
    """Return the row-level stop distance used by sizing and backtesting."""

    fixed_stop = max(float(cfg.stop_loss), 1e-9)
    if not bool(getattr(cfg, "use_atr_exits", False)) or "atr_14" not in data.columns:
        return np.full(len(data), fixed_stop, dtype=float)
    multiplier = max(float(getattr(cfg, "stop_loss_atr_multiplier", 1.0)), 1e-9)
    atr_values = causal_atr_values(data, fallback=fixed_stop / multiplier)
    min_exit = max(float(getattr(cfg, "min_exit_pct", 0.0)), 1e-6)
    max_exit = max(float(getattr(cfg, "max_exit_pct", fixed_stop)), min_exit)
    return np.clip(atr_values * multiplier, min_exit, max_exit)


def volatility_position_scale(
    data: pd.DataFrame,
    cfg: RiskConfig,
) -> np.ndarray:
    """Return the causal volatility multiplier applied to position size."""

    if bool(getattr(cfg, "ewma_volatility_enabled", False)):
        volatility_values = causal_ewma_daily_volatility(
            data,
            span=int(getattr(cfg, "ewma_volatility_span", 48)),
            fallback_daily=float(
                getattr(cfg, "ewma_daily_volatility_target", 0.03)
            ),
        )
        return np.clip(
            float(getattr(cfg, "ewma_daily_volatility_target", 0.03))
            / np.maximum(volatility_values, 1e-9),
            float(cfg.min_volatility_scale),
            float(cfg.max_volatility_scale),
        )
    if "atr_14" in data.columns and not bool(
        getattr(cfg, "use_atr_exits", False)
    ):
        atr_values = causal_atr_values(
            data,
            fallback=float(cfg.volatility_target),
        )
        return np.clip(
            float(cfg.volatility_target)
            / np.maximum(atr_values, 1e-9),
            float(cfg.min_volatility_scale),
            float(cfg.max_volatility_scale),
        )
    return np.ones(len(data), dtype=float)


def dynamic_position_fraction(
    data: pd.DataFrame,
    prob: np.ndarray,
    position: np.ndarray,
    cfg: RiskConfig,
    event_scores: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Size positions from risk budget, confidence, volatility and liquidity."""

    max_fraction = maximum_position_fraction(cfg)
    if not cfg.dynamic_position_sizing:
        return np.where(position != 0, max_fraction, 0.0)

    stop_distance = stop_distance_series(data, cfg)
    stop_risk = np.maximum(
        stop_distance * max(float(cfg.leverage), 1.0),
        1e-9,
    )
    base_fraction = np.minimum(
        max_fraction,
        float(cfg.risk_per_trade) / stop_risk,
    )
    confidence_room = max(0.5 - float(cfg.min_confidence_gap), 1e-9)
    confidence = np.clip(
        (np.abs(prob - 0.5) - float(cfg.min_confidence_gap)) / confidence_room,
        0.0,
        1.0,
    )
    confidence_scale = float(cfg.min_position_scale) + (
        1.0 - float(cfg.min_position_scale)
    ) * confidence

    volatility_scale = volatility_position_scale(data, cfg)

    liquidity_scale = np.ones(len(data), dtype=float)
    if "quote_volume_z" in data.columns:
        liquidity_z = (
            pd.to_numeric(data["quote_volume_z"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        liquidity_scale = np.where(
            liquidity_z < -1.0, 0.55, np.where(liquidity_z < -0.4, 0.80, 1.0)
        )

    event_scale = np.ones(len(data), dtype=float)
    if bool(getattr(cfg, "event_position_boost_enabled", False)):
        if event_scores is None:
            zero = np.zeros(len(data), dtype=float)
            long_score = (
                pd.to_numeric(data["platform_event_long_score"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                if "platform_event_long_score" in data.columns
                else zero
            )
            short_score = (
                pd.to_numeric(data["platform_event_short_score"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                if "platform_event_short_score" in data.columns
                else zero
            )
        else:
            long_score, short_score = event_scores
        side_score = np.where(
            position > 0, long_score, np.where(position < 0, short_score, 0.0)
        )
        min_score = float(getattr(cfg, "event_position_min_score", 0.55))
        score_room = max(1.0 - min_score, 1e-9)
        score_strength = np.clip((side_score - min_score) / score_room, 0.0, 1.0)
        event_scale = 1.0 + float(
            getattr(cfg, "event_position_boost_strength", 0.80)
        ) * score_strength

    fraction = (
        base_fraction
        * confidence_scale
        * volatility_scale
        * liquidity_scale
        * event_scale
    )
    fraction = np.where(position != 0, fraction, 0.0)
    return np.clip(fraction, 0.0, max_fraction)


def apply_position_rebalance_rules(
    position: np.ndarray,
    target_fraction: np.ndarray,
    cfg: RiskConfig,
) -> np.ndarray:
    """Limit small or abrupt exposure changes while a position stays open."""

    band = max(float(getattr(cfg, "position_rebalance_band", 0.0)), 0.0)
    max_step = max(float(getattr(cfg, "max_position_fraction_step", 1.0)), 0.0)
    if (band <= 0.0 and max_step >= 1.0) or not len(target_fraction):
        return target_fraction
    smoothed = np.asarray(target_fraction, dtype=float).copy()
    raw_position = np.asarray(position, dtype=float)
    for idx in range(1, len(smoothed)):
        if raw_position[idx] == 0.0:
            smoothed[idx] = 0.0
            continue
        if raw_position[idx] != raw_position[idx - 1] or raw_position[idx - 1] == 0.0:
            continue
        previous = float(smoothed[idx - 1])
        target = float(smoothed[idx])
        diff = target - previous
        if abs(diff) < band:
            smoothed[idx] = previous
            continue
        if max_step < 1.0:
            smoothed[idx] = previous + float(np.clip(diff, -max_step, max_step))
    return np.clip(smoothed, 0.0, maximum_position_fraction(cfg))


def drawdown_cooldown_gate(
    preliminary_returns: np.ndarray,
    cfg: RiskConfig,
) -> np.ndarray:
    """Block new exposure temporarily after drawdown or loss-streak triggers."""

    if not bool(getattr(cfg, "drawdown_cooldown_enabled", False)) or not len(
        preliminary_returns
    ):
        return np.ones(len(preliminary_returns), dtype=bool)
    gate = np.ones(len(preliminary_returns), dtype=bool)
    equity = 1.0
    peak = 1.0
    loss_streak = 0
    cooldown_remaining = 0
    drawdown_limit = -abs(float(getattr(cfg, "cooldown_drawdown", 0.006)))
    loss_streak_limit = max(
        1, int(getattr(cfg, "cooldown_loss_streak", 3))
    )
    cooldown_bars = max(1, int(getattr(cfg, "cooldown_bars", 12)))
    for idx, value in enumerate(np.asarray(preliminary_returns, dtype=float)):
        cooldown_finished = False
        if cooldown_remaining > 0:
            gate[idx] = False
            cooldown_remaining -= 1
            value = 0.0
            cooldown_finished = cooldown_remaining == 0
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if cooldown_finished:
            peak = equity
            loss_streak = 0
            continue
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        if value < 0:
            loss_streak += 1
        elif value > 0:
            loss_streak = 0
        if cooldown_remaining <= 0 and (
            drawdown <= drawdown_limit or loss_streak >= loss_streak_limit
        ):
            cooldown_remaining = cooldown_bars
            loss_streak = 0
    return gate


def funding_crowding_gate(
    data: pd.DataFrame,
    position: np.ndarray,
    cfg: RiskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Block positions paying an extreme same-side funding rate."""

    raw_position = np.asarray(position, dtype=int)
    if (
        not bool(getattr(cfg, "funding_crowding_guard_enabled", False))
        or "funding_rate_8h" not in data.columns
    ):
        return np.ones(len(raw_position), dtype=bool), np.zeros(
            len(raw_position),
            dtype=float,
        )
    funding = (
        pd.to_numeric(data["funding_rate_8h"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    limit = max(
        float(getattr(cfg, "funding_crowding_max_rate", 0.0005)),
        0.0,
    )
    blocked = ((raw_position > 0) & (funding > limit)) | (
        (raw_position < 0) & (funding < -limit)
    )
    return ~blocked, funding


def regime_risk_gate(
    data: pd.DataFrame,
    position: np.ndarray,
    cfg: RiskConfig,
) -> np.ndarray:
    """Veto active exposure during causal crash or liquidity-crisis regimes."""

    proposed = np.asarray(position, dtype=int)
    if not bool(getattr(cfg, "regime_risk_guard_enabled", False)):
        return np.ones(len(proposed), dtype=bool)
    if "regime_risk_off" in data.columns:
        risk_off = (
            data["regime_risk_off"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )
    elif "market_regime" in data.columns:
        risk_off = (
            data["market_regime"]
            .astype(str)
            .isin({"crash", "liquidity_crisis"})
            .to_numpy()
        )
    else:
        risk_off = np.zeros(len(proposed), dtype=bool)
    return np.where(proposed != 0, ~risk_off, True)


def evaluate_strategy_risk(
    decision: StrategyDecision,
    cfg: RiskConfig,
) -> RiskDecision:
    """Apply static exposure and leverage limits to one strategy decision."""

    maximum = maximum_position_fraction(cfg)
    if decision.target_direction == 0:
        return RiskDecision(
            allow_trade=True,
            risk_level=RiskLevel.LOW,
            max_position_size=0.0,
            reason=RiskReason.NO_POSITION.value,
        )
    if maximum <= 0:
        return RiskDecision(
            allow_trade=False,
            risk_level=RiskLevel.EXTREME,
            max_position_size=0.0,
            reason=RiskReason.ZERO_CAPACITY.value,
        )
    allowed_size = min(float(decision.target_exposure), maximum)
    capped = allowed_size + 1e-12 < float(decision.target_exposure)
    return RiskDecision(
        allow_trade=allowed_size > 0,
        risk_level=RiskLevel.MEDIUM if capped else RiskLevel.LOW,
        max_position_size=allowed_size,
        reason=RiskReason.EXPOSURE_CAPPED.value if capped else RiskReason.ALLOWED.value,
    )


def build_risk_decision_frame(
    data: pd.DataFrame,
    proposed_position: np.ndarray,
    target_fraction: np.ndarray,
    cooldown_gate: np.ndarray,
    cfg: RiskConfig,
    funding_gate: np.ndarray | None = None,
    regime_gate: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build row-level risk decisions without changing strategy or PnL output."""

    proposed = np.asarray(proposed_position, dtype=int)
    target = np.asarray(target_fraction, dtype=float)
    cooldown = np.asarray(cooldown_gate, dtype=bool)
    funding_allowed = (
        np.ones(len(proposed), dtype=bool)
        if funding_gate is None
        else np.asarray(funding_gate, dtype=bool)
    )
    regime_allowed = (
        np.ones(len(proposed), dtype=bool)
        if regime_gate is None
        else np.asarray(regime_gate, dtype=bool)
    )
    active = proposed != 0
    regime_blocked = active & ~regime_allowed
    drawdown_blocked = active & regime_allowed & ~cooldown
    funding_blocked = active & regime_allowed & cooldown & ~funding_allowed
    blocked = regime_blocked | drawdown_blocked | funding_blocked
    allow_trade = ~blocked
    max_size = np.where(
        active & regime_allowed & cooldown & funding_allowed,
        target,
        0.0,
    )

    reason = np.full(len(data), RiskReason.ALLOWED.value, dtype=object)
    level = np.full(len(data), RiskLevel.LOW.value, dtype=object)
    reason[~active] = RiskReason.NO_POSITION.value
    reason[regime_blocked] = RiskReason.REGIME_RISK_OFF_BLOCKED.value
    level[regime_blocked] = RiskLevel.EXTREME.value
    reason[drawdown_blocked] = RiskReason.DRAWDOWN_BLOCKED.value
    level[drawdown_blocked] = RiskLevel.EXTREME.value
    reason[funding_blocked] = RiskReason.FUNDING_CROWDING_BLOCKED.value
    level[funding_blocked] = RiskLevel.HIGH.value

    dynamic_sizing = bool(getattr(cfg, "dynamic_position_sizing", False))
    if dynamic_sizing and "quote_volume_z" in data.columns:
        liquidity = (
            pd.to_numeric(data["quote_volume_z"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        reduced = (
            active
            & regime_allowed
            & cooldown
            & funding_allowed
            & (liquidity < -0.4)
        )
        reason[reduced] = RiskReason.LIQUIDITY_REDUCED.value
        level[reduced] = np.where(
            liquidity[reduced] < -1.0, RiskLevel.HIGH.value, RiskLevel.MEDIUM.value
        )

    ewma_values = causal_ewma_volatility(
        data,
        span=int(getattr(cfg, "ewma_volatility_span", 48)),
        fallback=max(
            float(getattr(cfg, "ewma_daily_volatility_target", 0.03))
            / np.sqrt(
                24.0
                / max(float(infer_bar_hours(data)), 1.0 / 60.0)
            ),
            1e-9,
        ),
    )
    ewma_daily_values = causal_ewma_daily_volatility(
        data,
        span=int(getattr(cfg, "ewma_volatility_span", 48)),
        fallback_daily=float(
            getattr(cfg, "ewma_daily_volatility_target", 0.03)
        ),
    )
    if dynamic_sizing and bool(
        getattr(cfg, "ewma_volatility_enabled", False)
    ):
        reduced = (
            active
            & regime_allowed
            & cooldown
            & funding_allowed
            & (
                ewma_daily_values
                > float(
                    getattr(
                        cfg,
                        "ewma_daily_volatility_target",
                        0.03,
                    )
                )
            )
            & (reason == RiskReason.ALLOWED.value)
        )
        reason[reduced] = RiskReason.VOLATILITY_REDUCED.value
        level[reduced] = RiskLevel.MEDIUM.value
    elif dynamic_sizing and "atr_14" in data.columns:
        atr = (
            pd.to_numeric(data["atr_14"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(float(cfg.volatility_target))
            .to_numpy(dtype=float)
        )
        reduced = (
            active
            & regime_allowed
            & cooldown
            & funding_allowed
            & (atr > float(cfg.volatility_target))
            & (reason == RiskReason.ALLOWED.value)
        )
        reason[reduced] = RiskReason.VOLATILITY_REDUCED.value
        level[reduced] = RiskLevel.MEDIUM.value

    return pd.DataFrame(
        {
            "risk_allow_trade": allow_trade,
            "risk_level": level,
            "risk_max_position_size": max_size,
            "risk_reason": reason,
            "risk_ewma_volatility": ewma_values,
            "risk_ewma_daily_volatility": ewma_daily_values,
            "risk_position_volatility_measure": (
                "ewma_return"
                if bool(getattr(cfg, "ewma_volatility_enabled", False))
                else "atr_range_or_fallback"
            ),
        },
        index=data.index,
    )
