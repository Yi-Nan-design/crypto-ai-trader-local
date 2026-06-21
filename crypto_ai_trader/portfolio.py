from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .contracts import (
    PortfolioAssetInput,
    PortfolioDecision,
    RiskLevel,
)
from .time_utils import beijing_now_iso


class PortfolioConfig(Protocol):
    """Configuration required by cross-asset portfolio construction."""

    portfolio_target_gross_exposure: float
    portfolio_max_total_leverage: float
    portfolio_max_single_weight: float
    portfolio_max_sector_exposure: float
    portfolio_max_cluster_exposure: float
    portfolio_min_liquidity_score: float
    portfolio_max_drawdown: float
    portfolio_volatility_floor: float
    portfolio_volatility_target_enabled: bool
    portfolio_target_daily_volatility: float
    portfolio_min_volatility_observations: int
    portfolio_correlation_threshold: float
    portfolio_correlation_lookback: int
    portfolio_cvar_confidence: float
    portfolio_max_cvar_loss: float
    portfolio_min_cvar_observations: int
    max_daily_loss: float
    default_leverage: int
    portfolio_require_complete_inputs: bool
    portfolio_symbol_sectors: dict[str, str]


def correlation_clusters(
    returns: pd.DataFrame,
    *,
    threshold: float = 0.75,
    min_periods: int = 48,
) -> dict[str, str]:
    """Build deterministic connected correlation clusters from past returns."""

    symbols = [str(column).upper() for column in returns.columns]
    if not symbols:
        return {}
    numeric = returns.copy()
    numeric.columns = symbols
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(min_periods=max(int(min_periods), 2)).abs()
    graph: dict[str, set[str]] = {symbol: set() for symbol in symbols}
    for left_idx, left in enumerate(symbols):
        for right in symbols[left_idx + 1 :]:
            value = correlation.loc[left, right]
            if np.isfinite(value) and float(value) >= float(threshold):
                graph[left].add(right)
                graph[right].add(left)

    clusters: dict[str, str] = {}
    cluster_index = 0
    for symbol in symbols:
        if symbol in clusters:
            continue
        cluster_index += 1
        cluster_name = f"corr_{cluster_index:02d}"
        queue: deque[str] = deque([symbol])
        while queue:
            current = queue.popleft()
            if current in clusters:
                continue
            clusters[current] = cluster_name
            queue.extend(sorted(graph[current].difference(clusters)))
    return clusters


def _capped_proportional_weights(
    scores: dict[str, float],
    caps: dict[str, float],
    target_gross: float,
) -> dict[str, float]:
    """Allocate gross exposure proportionally while honoring asset caps."""

    weights = {symbol: 0.0 for symbol in scores}
    remaining = max(float(target_gross), 0.0)
    active = {
        symbol
        for symbol, score in scores.items()
        if abs(float(score)) > 1e-15 and float(caps.get(symbol, 0.0)) > 0.0
    }
    while active and remaining > 1e-12:
        score_sum = sum(abs(float(scores[symbol])) for symbol in active)
        if score_sum <= 1e-15:
            break
        allocated = 0.0
        capped: set[str] = set()
        for symbol in sorted(active):
            room = max(float(caps[symbol]) - abs(weights[symbol]), 0.0)
            desired = remaining * abs(float(scores[symbol])) / score_sum
            addition = min(desired, room)
            weights[symbol] += np.sign(scores[symbol]) * addition
            allocated += addition
            if room - addition <= 1e-12:
                capped.add(symbol)
        if allocated <= 1e-12:
            break
        remaining -= allocated
        active.difference_update(capped)
        if not capped:
            break
    return weights


def construct_portfolio(
    assets: list[PortfolioAssetInput],
    cfg: PortfolioConfig,
    *,
    current_drawdown: float = 0.0,
    current_daily_return: float = 0.0,
    input_available: bool = True,
    input_unavailable_reason: str = "portfolio_blocked_unavailable_input",
) -> PortfolioDecision:
    """Convert approved asset signals into constrained target weights."""

    unique: dict[str, PortfolioAssetInput] = {}
    for asset in assets:
        unique[asset.symbol.upper()] = asset
    asset_reasons = {
        symbol: "portfolio_no_signal"
        for symbol in unique
    }
    if not input_available:
        reason = (
            str(input_unavailable_reason).strip()
            or "portfolio_blocked_unavailable_input"
        )
        return PortfolioDecision(
            weights={symbol: 0.0 for symbol in unique},
            gross_exposure=0.0,
            net_exposure=0.0,
            cluster_exposure={},
            sector_exposure={},
            allow_portfolio=False,
            risk_level=RiskLevel.EXTREME,
            reason=reason,
            asset_reasons={symbol: reason for symbol in unique},
        )
    if current_drawdown <= -abs(float(cfg.portfolio_max_drawdown)):
        return PortfolioDecision(
            weights={symbol: 0.0 for symbol in unique},
            gross_exposure=0.0,
            net_exposure=0.0,
            cluster_exposure={},
            sector_exposure={},
            allow_portfolio=False,
            risk_level=RiskLevel.EXTREME,
            reason="portfolio_blocked_drawdown",
            asset_reasons={
                symbol: "portfolio_blocked_drawdown"
                for symbol in unique
            },
        )
    if current_daily_return <= -abs(float(cfg.max_daily_loss)):
        return PortfolioDecision(
            weights={symbol: 0.0 for symbol in unique},
            gross_exposure=0.0,
            net_exposure=0.0,
            cluster_exposure={},
            sector_exposure={},
            allow_portfolio=False,
            risk_level=RiskLevel.EXTREME,
            reason="portfolio_blocked_daily_loss",
            asset_reasons={
                symbol: "portfolio_blocked_daily_loss"
                for symbol in unique
            },
        )

    scores: dict[str, float] = {}
    caps: dict[str, float] = {}
    clusters: dict[str, str] = {}
    sectors: dict[str, str] = {}
    for symbol, asset in unique.items():
        clusters[symbol] = asset.correlation_cluster
        sectors[symbol] = asset.sector
        if not asset.input_available:
            scores[symbol] = 0.0
            caps[symbol] = 0.0
            asset_reasons[symbol] = asset.input_reason
            continue
        if asset.direction == 0:
            scores[symbol] = 0.0
            caps[symbol] = 0.0
            continue
        if asset.liquidity_score < float(cfg.portfolio_min_liquidity_score):
            scores[symbol] = 0.0
            caps[symbol] = 0.0
            asset_reasons[symbol] = "portfolio_blocked_low_liquidity"
            continue
        volatility = max(
            float(asset.volatility),
            float(cfg.portfolio_volatility_floor),
            1e-9,
        )
        scores[symbol] = (
            float(asset.direction)
            * float(asset.signal_strength)
            * float(asset.confidence)
            / volatility
        )
        caps[symbol] = min(
            float(asset.max_weight),
            float(cfg.portfolio_max_single_weight),
        )
        asset_reasons[symbol] = "portfolio_weight_allowed"

    gross_target = min(
        max(float(cfg.portfolio_target_gross_exposure), 0.0),
        max(float(cfg.portfolio_max_total_leverage), 0.0),
    )
    weights = _capped_proportional_weights(scores, caps, gross_target)

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for symbol, cluster in clusters.items():
        by_cluster[cluster].append(symbol)
    cluster_limit = max(float(cfg.portfolio_max_cluster_exposure), 0.0)
    for cluster, symbols in by_cluster.items():
        exposure = sum(abs(weights.get(symbol, 0.0)) for symbol in symbols)
        if cluster_limit > 0.0 and exposure > cluster_limit + 1e-12:
            scale = cluster_limit / exposure
            for symbol in symbols:
                weights[symbol] *= scale
                if abs(weights[symbol]) > 1e-15:
                    asset_reasons[symbol] = "portfolio_reduced_cluster_cap"

    by_sector: dict[str, list[str]] = defaultdict(list)
    for symbol, sector in sectors.items():
        by_sector[sector].append(symbol)
    sector_limit = max(float(cfg.portfolio_max_sector_exposure), 0.0)
    for sector, symbols in by_sector.items():
        exposure = sum(abs(weights.get(symbol, 0.0)) for symbol in symbols)
        if sector_limit > 0.0 and exposure > sector_limit + 1e-12:
            scale = sector_limit / exposure
            for symbol in symbols:
                weights[symbol] *= scale
                if abs(weights[symbol]) > 1e-15:
                    asset_reasons[symbol] = "portfolio_reduced_sector_cap"

    gross = sum(abs(weight) for weight in weights.values())
    max_total = max(float(cfg.portfolio_max_total_leverage), 0.0)
    if max_total > 0.0 and gross > max_total + 1e-12:
        scale = max_total / gross
        weights = {symbol: weight * scale for symbol, weight in weights.items()}
        gross = max_total
        for symbol, weight in weights.items():
            if abs(weight) > 1e-15:
                asset_reasons[symbol] = "portfolio_reduced_total_leverage"

    cluster_exposure = {
        cluster: float(
            sum(abs(weights.get(symbol, 0.0)) for symbol in symbols)
        )
        for cluster, symbols in by_cluster.items()
    }
    sector_exposure = {
        sector: float(
            sum(abs(weights.get(symbol, 0.0)) for symbol in symbols)
        )
        for sector, symbols in by_sector.items()
    }
    net = float(sum(weights.values()))
    active = any(abs(weight) > 1e-15 for weight in weights.values())
    constrained = any(
        reason.startswith("portfolio_reduced")
        for reason in asset_reasons.values()
    )
    return PortfolioDecision(
        weights={symbol: float(weight) for symbol, weight in weights.items()},
        gross_exposure=float(gross),
        net_exposure=net,
        cluster_exposure=cluster_exposure,
        sector_exposure=sector_exposure,
        allow_portfolio=active,
        risk_level=(
            RiskLevel.MEDIUM
            if constrained
            else (RiskLevel.LOW if active else RiskLevel.HIGH)
        ),
        reason=(
            "portfolio_constrained"
            if constrained
            else ("portfolio_allowed" if active else "portfolio_no_trade")
        ),
        asset_reasons=asset_reasons,
    )


def historical_expected_shortfall(
    returns: pd.DataFrame,
    weights: dict[str, float],
    *,
    confidence: float = 0.95,
    min_observations: int = 100,
) -> dict[str, Any]:
    """Estimate trailing portfolio VaR and Expected Shortfall without look-ahead."""

    active_weights = {
        str(symbol).upper(): float(weight)
        for symbol, weight in weights.items()
        if abs(float(weight)) > 1e-15
    }
    confidence = float(np.clip(confidence, 0.50, 0.999))
    minimum = max(int(min_observations), 2)
    available_symbols = [
        symbol
        for symbol in active_weights
        if symbol in returns.columns
    ]
    if not available_symbols:
        return {
            "available": False,
            "reason": "no_active_weight_history",
            "confidence": confidence,
            "observations": 0,
            "var_loss": 0.0,
            "expected_shortfall_loss": 0.0,
        }
    frame = (
        returns[available_symbols]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="any")
    )
    if len(frame) < minimum:
        return {
            "available": False,
            "reason": "insufficient_history",
            "confidence": confidence,
            "observations": int(len(frame)),
            "required_observations": minimum,
            "var_loss": 0.0,
            "expected_shortfall_loss": 0.0,
        }
    weight_vector = np.asarray(
        [active_weights[symbol] for symbol in available_symbols],
        dtype=float,
    )
    portfolio_returns = frame.to_numpy(dtype=float) @ weight_vector
    cutoff = float(np.quantile(portfolio_returns, 1.0 - confidence))
    tail = portfolio_returns[portfolio_returns <= cutoff]
    expected_shortfall = float(-tail.mean()) if len(tail) else 0.0
    var_loss = float(-cutoff)
    return {
        "available": True,
        "reason": "historical_closed_bar_estimate",
        "confidence": confidence,
        "observations": int(len(portfolio_returns)),
        "var_loss": var_loss if var_loss > 0.0 else 0.0,
        "expected_shortfall_loss": (
            expected_shortfall if expected_shortfall > 0.0 else 0.0
        ),
    }


def historical_portfolio_volatility(
    returns: pd.DataFrame,
    weights: dict[str, float],
    *,
    min_observations: int,
) -> dict[str, Any]:
    """Estimate causal trailing portfolio volatility on a 24-hour horizon."""

    active = {
        str(symbol).upper(): float(weight)
        for symbol, weight in weights.items()
        if abs(float(weight)) > 1e-15
    }
    symbols = [symbol for symbol in active if symbol in returns.columns]
    if not symbols:
        return {
            "available": False,
            "reason": "no_active_weight_history",
            "observations": 0,
            "daily_volatility": 0.0,
        }
    frame = returns[symbols].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna(how="any")
    minimum = max(int(min_observations), 2)
    if len(frame) < minimum:
        return {
            "available": False,
            "reason": "insufficient_history",
            "observations": int(len(frame)),
            "required_observations": minimum,
            "daily_volatility": 0.0,
        }
    vector = np.asarray([active[symbol] for symbol in symbols], dtype=float)
    portfolio_returns = frame.to_numpy(dtype=float) @ vector
    per_bar_volatility = float(np.std(portfolio_returns, ddof=0))
    numeric_index = pd.to_numeric(
        pd.Series(frame.index),
        errors="coerce",
    ).dropna()
    if len(numeric_index) >= 2:
        median_ms = float(np.median(np.diff(numeric_index.to_numpy(dtype=float))))
        bar_hours = median_ms / 3_600_000.0 if median_ms > 0.0 else 1.0
    else:
        bar_hours = 1.0
    daily_scale = float(np.sqrt(24.0 / max(bar_hours, 1.0 / 60.0)))
    return {
        "available": True,
        "reason": "historical_closed_bar_estimate",
        "observations": int(len(portfolio_returns)),
        "bar_hours": bar_hours,
        "per_bar_volatility": per_bar_volatility,
        "daily_volatility": per_bar_volatility * daily_scale,
    }


def apply_portfolio_volatility_target(
    decision: PortfolioDecision,
    returns: pd.DataFrame,
    cfg: PortfolioConfig,
) -> tuple[PortfolioDecision, dict[str, Any]]:
    """Reduce portfolio weights to a trailing daily volatility target."""

    risk = historical_portfolio_volatility(
        returns,
        decision.weights,
        min_observations=int(cfg.portfolio_min_volatility_observations),
    )
    target = max(float(cfg.portfolio_target_daily_volatility), 0.0)
    enabled = bool(cfg.portfolio_volatility_target_enabled)
    risk.update(
        {
            "enabled": enabled,
            "target_daily_volatility": target,
            "target_applied": False,
        }
    )
    if not enabled or not decision.allow_portfolio:
        return decision, risk
    if not bool(risk.get("available")):
        reason = "portfolio_blocked_volatility_history"
        blocked = PortfolioDecision(
            weights={symbol: 0.0 for symbol in decision.weights},
            gross_exposure=0.0,
            net_exposure=0.0,
            cluster_exposure={
                cluster: 0.0 for cluster in decision.cluster_exposure
            },
            sector_exposure={
                sector: 0.0 for sector in decision.sector_exposure
            },
            allow_portfolio=False,
            risk_level=RiskLevel.EXTREME,
            reason=reason,
            asset_reasons={
                symbol: reason for symbol in decision.weights
            },
        )
        return blocked, risk
    realized = float(risk["daily_volatility"])
    if target <= 0.0 or realized <= target:
        return decision, risk
    scale = float(np.clip(target / max(realized, 1e-12), 0.0, 1.0))
    weights = {
        symbol: float(weight) * scale
        for symbol, weight in decision.weights.items()
    }
    asset_reasons = dict(decision.asset_reasons)
    for symbol, weight in weights.items():
        if abs(weight) > 1e-15:
            asset_reasons[symbol] = "portfolio_reduced_volatility_target"
    adjusted = PortfolioDecision(
        weights=weights,
        gross_exposure=float(decision.gross_exposure) * scale,
        net_exposure=float(decision.net_exposure) * scale,
        cluster_exposure={
            key: float(value) * scale
            for key, value in decision.cluster_exposure.items()
        },
        sector_exposure={
            key: float(value) * scale
            for key, value in decision.sector_exposure.items()
        },
        allow_portfolio=True,
        risk_level=RiskLevel.HIGH,
        reason="portfolio_constrained_volatility_target",
        asset_reasons=asset_reasons,
    )
    risk.update(
        {
            "target_applied": True,
            "scale": scale,
            "pre_target_daily_volatility": realized,
            "post_target_daily_volatility": realized * scale,
        }
    )
    return adjusted, risk


def apply_expected_shortfall_limit(
    decision: PortfolioDecision,
    returns: pd.DataFrame,
    cfg: PortfolioConfig,
) -> tuple[PortfolioDecision, dict[str, Any]]:
    """Scale planning weights when trailing Expected Shortfall exceeds its cap."""

    risk = historical_expected_shortfall(
        returns,
        decision.weights,
        confidence=float(cfg.portfolio_cvar_confidence),
        min_observations=int(cfg.portfolio_min_cvar_observations),
    )
    limit = max(float(cfg.portfolio_max_cvar_loss), 0.0)
    risk["limit"] = limit
    risk["limit_applied"] = False
    if (
        not decision.allow_portfolio
        or not bool(risk.get("available"))
        or limit <= 0.0
        or float(risk["expected_shortfall_loss"]) <= limit
    ):
        return decision, risk

    scale = float(
        np.clip(
            limit / max(float(risk["expected_shortfall_loss"]), 1e-12),
            0.0,
            1.0,
        )
    )
    weights = {
        symbol: float(weight) * scale
        for symbol, weight in decision.weights.items()
    }
    cluster_exposure = {
        cluster: float(exposure) * scale
        for cluster, exposure in decision.cluster_exposure.items()
    }
    sector_exposure = {
        sector: float(exposure) * scale
        for sector, exposure in decision.sector_exposure.items()
    }
    asset_reasons = dict(decision.asset_reasons)
    for symbol, weight in weights.items():
        if abs(weight) > 1e-15:
            asset_reasons[symbol] = "portfolio_reduced_cvar"
    adjusted = PortfolioDecision(
        weights=weights,
        gross_exposure=float(decision.gross_exposure) * scale,
        net_exposure=float(decision.net_exposure) * scale,
        cluster_exposure=cluster_exposure,
        sector_exposure=sector_exposure,
        allow_portfolio=any(abs(weight) > 1e-15 for weight in weights.values()),
        risk_level=RiskLevel.HIGH,
        reason="portfolio_constrained_cvar",
        asset_reasons=asset_reasons,
    )
    risk.update(
        {
            "limit_applied": True,
            "scale": scale,
            "pre_limit_expected_shortfall_loss": float(
                risk["expected_shortfall_loss"]
            ),
            "post_limit_expected_shortfall_loss": float(
                risk["expected_shortfall_loss"]
            )
            * scale,
        }
    )
    return adjusted, risk


def portfolio_inputs_from_reports(
    reports: list[dict[str, Any]],
    *,
    clusters: dict[str, str],
    default_max_weight: float,
    leverage: float = 1.0,
    sectors: dict[str, str] | None = None,
    require_complete_inputs: bool = False,
    expected_open_time: int | None = None,
) -> list[PortfolioAssetInput]:
    """Convert runner reports into portfolio requests without placing orders."""

    inputs: list[PortfolioAssetInput] = []
    for report in reports:
        symbol = str(report.get("symbol") or "").upper()
        if not symbol:
            continue
        probabilities = report.get("latest_direction_probabilities")
        probabilities = probabilities if isinstance(probabilities, dict) else {}
        alpha = report.get("latest_alpha_prediction")
        alpha = alpha if isinstance(alpha, dict) else {}
        risk = report.get("latest_risk_decision")
        risk = risk if isinstance(risk, dict) else {}
        execution = report.get("latest_execution_decision")
        execution = execution if isinstance(execution, dict) else {}
        input_reason = "portfolio_input_available"
        if require_complete_inputs:
            required_checks = (
                (bool(probabilities), "portfolio_blocked_missing_direction_probabilities"),
                (bool(alpha), "portfolio_blocked_missing_alpha_prediction"),
                (bool(risk), "portfolio_blocked_missing_risk_decision"),
                (bool(execution), "portfolio_blocked_missing_execution_decision"),
                (
                    "latest_liquidity_score" in report,
                    "portfolio_blocked_missing_liquidity_score",
                ),
                (
                    "latest_open_time" in report,
                    "portfolio_blocked_missing_open_time",
                ),
            )
            for available, reason in required_checks:
                if not available:
                    input_reason = reason
                    break
            if input_reason == "portfolio_input_available":
                try:
                    report_open_time = int(report["latest_open_time"])
                except (TypeError, ValueError):
                    input_reason = "portfolio_blocked_invalid_open_time"
                else:
                    if (
                        expected_open_time is not None
                        and report_open_time != int(expected_open_time)
                    ):
                        input_reason = (
                            "portfolio_blocked_stale_or_unaligned_open_time"
                        )
            for value, reason in (
                (
                    alpha.get("volatility_forecast"),
                    "portfolio_blocked_invalid_volatility_forecast",
                ),
                (
                    report.get("latest_liquidity_score"),
                    "portfolio_blocked_invalid_liquidity_score",
                ),
                (
                    risk.get("max_position_size"),
                    "portfolio_blocked_invalid_risk_capacity",
                ),
            ):
                if input_reason != "portfolio_input_available":
                    break
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    input_reason = reason
                    break
                if not np.isfinite(numeric) or numeric < 0.0:
                    input_reason = reason
                    break
        input_available = input_reason == "portfolio_input_available"
        long_probability = float(
            probabilities.get("long", report.get("latest_up_probability", 0.5))
            or 0.5
        )
        short_probability = float(
            probabilities.get("short", 1.0 - long_probability)
            or (1.0 - long_probability)
        )
        trade_probability = float(probabilities.get("trade", 1.0) or 0.0)
        up_probability = float(
            report.get("latest_up_probability", long_probability) or 0.5
        )
        optimized = report.get("optimized_backtest_config")
        optimized = optimized if isinstance(optimized, dict) else {}
        confidence_gap = max(
            float(optimized.get("min_confidence_gap", 0.0) or 0.0),
            0.0,
        )
        long_threshold = max(
            float(optimized.get("long_threshold", 0.57) or 0.57),
            0.5 + confidence_gap,
        )
        short_threshold = min(
            float(optimized.get("short_threshold", 0.43) or 0.43),
            0.5 - confidence_gap,
        )
        trade_threshold = float(
            optimized.get("trade_signal_threshold", 0.57) or 0.57
        )
        direction = 0
        signal_strength = 0.0
        if trade_probability >= trade_threshold and up_probability >= long_threshold:
            direction = 1
            signal_strength = float(
                np.clip(
                    (up_probability - long_threshold)
                    / max(1.0 - long_threshold, 1e-9),
                    0.0,
                    1.0,
                )
            )
        elif trade_probability >= trade_threshold and up_probability <= short_threshold:
            direction = -1
            signal_strength = float(
                np.clip(
                    (short_threshold - up_probability)
                    / max(short_threshold, 1e-9),
                    0.0,
                    1.0,
                )
            )
        side_policy = str(
            optimized.get("trade_side_policy", "both") or "both"
        ).strip().lower()
        if side_policy in {"long", "long_only"} and direction < 0:
            direction = 0
        elif side_policy in {"short", "short_only"} and direction > 0:
            direction = 0
        elif side_policy in {"none", "no_trade"}:
            direction = 0
        confidence = float(
            np.clip(
                (report.get("latest_alpha_prediction") or {}).get(
                    "confidence",
                    signal_strength,
                ),
                0.0,
                1.0,
            )
        )
        if execution and not execution.get(
            "allow_execution",
            True,
        ):
            direction = 0
        if risk and not bool(risk.get("allow_trade", True)):
            direction = 0
        raw_risk_fraction = risk.get("max_position_size")
        if raw_risk_fraction is None:
            max_weight = float(default_max_weight)
        else:
            max_weight = min(
                float(default_max_weight),
                max(float(raw_risk_fraction), 0.0) * max(float(leverage), 1.0),
            )
        try:
            liquidity_score = float(report.get("latest_liquidity_score"))
        except (TypeError, ValueError):
            liquidity_score = 0.0
        if not np.isfinite(liquidity_score):
            liquidity_score = 0.0
        inputs.append(
            PortfolioAssetInput(
                symbol=symbol,
                direction=direction,
                signal_strength=signal_strength,
                confidence=confidence,
                volatility=max(
                    float(alpha.get("volatility_forecast") or 0.0),
                    0.0,
                ),
                liquidity_score=float(np.clip(liquidity_score, 0.0, 1.0)),
                max_weight=max(max_weight, 0.0),
                correlation_cluster=clusters.get(symbol, f"solo_{symbol}"),
                sector=(sectors or {}).get(symbol, "unclassified"),
                input_available=input_available,
                input_reason=input_reason,
            )
        )
    return inputs


def build_portfolio_snapshot(
    reports: list[dict[str, Any]],
    returns: pd.DataFrame,
    cfg: PortfolioConfig,
    *,
    interval: str,
    current_drawdown: float = 0.0,
    current_daily_return: float = 0.0,
    current_risk_source: str = "no_live_portfolio_equity_planning_default",
    input_available: bool = True,
    input_unavailable_reason: str = "portfolio_blocked_unavailable_input",
    expected_open_time: int | None = None,
) -> dict[str, Any]:
    """Build a report-only cross-asset target-weight snapshot."""

    lookback = max(int(cfg.portfolio_correlation_lookback), 2)
    recent_returns = returns.tail(lookback)
    clusters = correlation_clusters(
        recent_returns,
        threshold=float(cfg.portfolio_correlation_threshold),
        min_periods=min(48, max(len(recent_returns) // 2, 2)),
    )
    inputs = portfolio_inputs_from_reports(
        reports,
        clusters=clusters,
        default_max_weight=float(cfg.portfolio_max_single_weight),
        leverage=float(cfg.default_leverage),
        sectors={
            str(symbol).upper(): str(sector)
            for symbol, sector in cfg.portfolio_symbol_sectors.items()
        },
        require_complete_inputs=bool(cfg.portfolio_require_complete_inputs),
        expected_open_time=expected_open_time,
    )
    decision = construct_portfolio(
        inputs,
        cfg,
        current_drawdown=current_drawdown,
        current_daily_return=current_daily_return,
        input_available=input_available,
        input_unavailable_reason=input_unavailable_reason,
    )
    decision, volatility_target = apply_portfolio_volatility_target(
        decision,
        recent_returns,
        cfg,
    )
    decision, expected_shortfall = apply_expected_shortfall_limit(
        decision,
        recent_returns,
        cfg,
    )
    return {
        "schema_version": 1,
        "created_beijing": beijing_now_iso(),
        "interval": interval,
        "method": "signal_confidence_inverse_volatility",
        "status": "planning_only",
        "current_drawdown": float(current_drawdown),
        "current_daily_return": float(current_daily_return),
        "current_risk_source": current_risk_source,
        "current_drawdown_source": current_risk_source,
        "input_available": bool(input_available),
        "input_unavailable_reason": (
            None if input_available else input_unavailable_reason
        ),
        "correlation": {
            "threshold": float(cfg.portfolio_correlation_threshold),
            "lookback": lookback,
            "observations": int(len(recent_returns)),
            "clusters": clusters,
        },
        "constraints": {
            "target_gross_exposure": float(cfg.portfolio_target_gross_exposure),
            "max_total_leverage": float(cfg.portfolio_max_total_leverage),
            "max_single_weight": float(cfg.portfolio_max_single_weight),
            "max_sector_exposure": float(cfg.portfolio_max_sector_exposure),
            "max_cluster_exposure": float(cfg.portfolio_max_cluster_exposure),
            "min_liquidity_score": float(cfg.portfolio_min_liquidity_score),
            "max_drawdown": float(cfg.portfolio_max_drawdown),
            "volatility_target_enabled": bool(
                cfg.portfolio_volatility_target_enabled
            ),
            "target_daily_volatility": float(
                cfg.portfolio_target_daily_volatility
            ),
            "min_volatility_observations": int(
                cfg.portfolio_min_volatility_observations
            ),
            "cvar_confidence": float(cfg.portfolio_cvar_confidence),
            "max_cvar_loss": float(cfg.portfolio_max_cvar_loss),
            "min_cvar_observations": int(
                cfg.portfolio_min_cvar_observations
            ),
            "require_complete_inputs": bool(
                cfg.portfolio_require_complete_inputs
            ),
        },
        "volatility_target": volatility_target,
        "expected_shortfall": expected_shortfall,
        "inputs": [item.to_dict() for item in inputs],
        "decision": decision.to_dict(),
        "safety": {
            "live_trading_enabled": False,
            "real_orders_allowed": False,
            "api_keys_used": False,
        },
    }
