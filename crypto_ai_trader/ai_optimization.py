from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .binance_data import resolve_exchange_rule_values
from .config import TraderConfig
from .features import MULTI_HORIZON_STEPS, feature_matrix, make_features, time_split
from .models import ModelBundle, classification_metrics, train_candidate_models_with_report
from .strategy_config import primary_label_horizon, primary_label_min_return
from .time_utils import beijing_now_iso


def score_backtest(
    result: BacktestResult,
    *,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> float:
    score = result.total_return
    if result.trades < min_trades:
        score -= 0.25
    if result.max_drawdown < max_drawdown_floor:
        score -= abs(result.max_drawdown - max_drawdown_floor) * 2.0
    if result.profit_factor < min_profit_factor:
        score -= (min_profit_factor - result.profit_factor) * 0.05
    return float(score)


def build_backtest_config(base: BacktestConfig, params: dict[str, Any]) -> BacktestConfig:
    return BacktestConfig(
        initial_balance=base.initial_balance,
        leverage=int(params["leverage"]),
        max_allowed_leverage=base.max_allowed_leverage,
        fee_rate=base.fee_rate,
        maker_fee_rate=base.maker_fee_rate,
        maker_fill_fraction=base.maker_fill_fraction,
        slippage_rate=base.slippage_rate,
        partial_fill_ratio=base.partial_fill_ratio,
        execution_latency_bars=base.execution_latency_bars,
        min_order_notional_fraction=base.min_order_notional_fraction,
        exchange_min_notional_usdt=base.exchange_min_notional_usdt,
        exchange_min_quantity=base.exchange_min_quantity,
        exchange_max_quantity=base.exchange_max_quantity,
        exchange_quantity_step=base.exchange_quantity_step,
        exchange_price_tick_size=base.exchange_price_tick_size,
        exchange_downtime_guard_enabled=base.exchange_downtime_guard_enabled,
        exchange_gap_recovery_bars=base.exchange_gap_recovery_bars,
        liquidity_execution_enabled=base.liquidity_execution_enabled,
        max_bar_participation_rate=base.max_bar_participation_rate,
        liquidity_lookback_bars=base.liquidity_lookback_bars,
        slippage_impact_coefficient=base.slippage_impact_coefficient,
        max_dynamic_slippage_rate=base.max_dynamic_slippage_rate,
        ewma_volatility_enabled=base.ewma_volatility_enabled,
        ewma_volatility_span=base.ewma_volatility_span,
        ewma_daily_volatility_target=base.ewma_daily_volatility_target,
        funding_crowding_guard_enabled=base.funding_crowding_guard_enabled,
        funding_crowding_max_rate=base.funding_crowding_max_rate,
        regime_risk_guard_enabled=base.regime_risk_guard_enabled,
        regime_detection_method=base.regime_detection_method,
        regime_statistical_clusters=base.regime_statistical_clusters,
        regime_statistical_min_history=base.regime_statistical_min_history,
        regime_statistical_lookback=base.regime_statistical_lookback,
        regime_statistical_refit_interval=base.regime_statistical_refit_interval,
        regime_statistical_random_seed=base.regime_statistical_random_seed,
        liquidation_guard_enabled=base.liquidation_guard_enabled,
        maintenance_margin_rate=base.maintenance_margin_rate,
        liquidation_buffer=base.liquidation_buffer,
        liquidation_fee_rate=base.liquidation_fee_rate,
        long_threshold=float(params["long_threshold"]),
        short_threshold=float(params["short_threshold"]),
        max_position_fraction=float(params["max_position_fraction"]),
        stop_loss=float(params["stop_loss"]),
        take_profit=float(params["take_profit"]),
    )


def default_param_grid(max_leverage: int) -> list[dict[str, Any]]:
    thresholds = [0.53, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.72]
    stop_losses = [0.008, 0.012, 0.02, 0.03]
    take_profits = [0.012, 0.02, 0.035, 0.055]
    position_fractions = [0.15, 0.25, 0.35, 0.5]
    params: list[dict[str, Any]] = []
    for leverage in range(1, max(1, max_leverage) + 1):
        for threshold in thresholds:
            for stop_loss in stop_losses:
                for take_profit in take_profits:
                    for max_position_fraction in position_fractions:
                        params.append(
                            {
                                "leverage": leverage,
                                "long_threshold": threshold,
                                "short_threshold": 1.0 - threshold,
                                "stop_loss": stop_loss,
                                "take_profit": take_profit,
                                "max_position_fraction": max_position_fraction,
                            }
                        )
    return params


def optimize_backtest_params(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    *,
    max_leverage: int,
    n_trials: int,
    seed: int,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> dict[str, Any]:
    if importlib.util.find_spec("optuna") is not None:
        return optimize_with_optuna(
            valid_df,
            test_df,
            bundle,
            base_cfg,
            max_leverage=max_leverage,
            n_trials=n_trials,
            seed=seed,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
        )
    return optimize_with_random_search(
        valid_df,
        test_df,
        bundle,
        base_cfg,
        max_leverage=max_leverage,
        n_trials=n_trials,
        seed=seed,
        min_trades=min_trades,
        max_drawdown_floor=max_drawdown_floor,
        min_profit_factor=min_profit_factor,
    )


def optimize_with_random_search(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    *,
    max_leverage: int,
    n_trials: int,
    seed: int,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    grid = default_param_grid(max_leverage)
    if n_trials < len(grid):
        selected = rng.choice(len(grid), size=max(n_trials, 1), replace=False)
        candidates = [grid[int(idx)] for idx in selected]
    else:
        candidates = grid

    trials = []
    for params in candidates:
        cfg = build_backtest_config(base_cfg, params)
        valid_result, _ = run_backtest(valid_df, bundle, cfg)
        score = score_backtest(
            valid_result,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
        )
        trials.append({"params": params, "score": score, "valid": asdict(valid_result)})
    return finalize_param_search("random_search", trials, test_df, bundle, base_cfg)


def optimize_with_optuna(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    *,
    max_leverage: int,
    n_trials: int,
    seed: int,
    min_trades: int,
    max_drawdown_floor: float,
    min_profit_factor: float,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        threshold = trial.suggest_categorical("long_threshold", [0.53, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.72])
        params = {
            "leverage": trial.suggest_int("leverage", 1, max(1, max_leverage)),
            "long_threshold": threshold,
            "short_threshold": 1.0 - threshold,
            "stop_loss": trial.suggest_float("stop_loss", 0.006, 0.04),
            "take_profit": trial.suggest_float("take_profit", 0.01, 0.08),
            "max_position_fraction": trial.suggest_float("max_position_fraction", 0.1, 0.5),
        }
        cfg = build_backtest_config(base_cfg, params)
        valid_result, _ = run_backtest(valid_df, bundle, cfg)
        score = score_backtest(
            valid_result,
            min_trades=min_trades,
            max_drawdown_floor=max_drawdown_floor,
            min_profit_factor=min_profit_factor,
        )
        trials.append({"params": params, "score": score, "valid": asdict(valid_result)})
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, n_trials), show_progress_bar=False)
    return finalize_param_search("optuna", trials, test_df, bundle, base_cfg)


def finalize_param_search(
    search_method: str,
    trials: list[dict[str, Any]],
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
) -> dict[str, Any]:
    ranked = sorted(trials, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    best_cfg = build_backtest_config(base_cfg, best["params"])
    test_result, test_detail = run_backtest(test_df, bundle, best_cfg)
    return {
        "search_method": search_method,
        "best": {**best, "test": asdict(test_result)},
        "trials": ranked,
        "test_detail": test_detail,
    }


def run_ai_optimization_for_symbol(
    symbol: str,
    interval: str,
    raw: pd.DataFrame,
    cfg: TraderConfig,
    *,
    initial_balance: float,
    max_leverage: int,
    n_trials: int,
    min_trades: int,
    max_drawdown_limit: float,
    min_profit_factor: float,
) -> dict[str, Any]:
    symbol = symbol.upper()
    data_validation = dict(raw.attrs.get("data_validation", {}))
    market_context = dict(raw.attrs.get("market_context", {}))
    label_horizon = primary_label_horizon(interval, cfg)
    label_min_return = primary_label_min_return(interval, cfg)
    feature_frame = make_features(raw, label_horizon=label_horizon, label_min_return=label_min_return)
    split_purge_rows = max(int(label_horizon), int(max(MULTI_HORIZON_STEPS)))
    train_df, valid_df, test_df = time_split(
        feature_frame,
        cfg.train_fraction,
        cfg.validation_fraction,
        purge_rows=split_purge_rows,
    )
    x_train, y_train, cols = feature_matrix(train_df)
    x_valid, y_valid, _ = feature_matrix(valid_df)
    x_test, y_test, _ = feature_matrix(test_df)

    bundle, model_report = train_candidate_models_with_report(
        x_train,
        y_train,
        x_valid,
        y_valid,
        cols,
        seed=cfg.random_seed,
        enable_ensemble=True,
    )
    test_prob = bundle.predict_up_probability(x_test)
    test_metrics = classification_metrics(y_test, test_prob)
    bundle.metrics = {**bundle.metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}

    model_path = cfg.model_dir / f"{symbol}_{interval}_ai.pkl"
    bundle.save(model_path)

    exchange_rules = resolve_exchange_rule_values(
        cfg.data_dir,
        symbol,
        min_notional_usdt=cfg.exchange_min_notional_usdt,
        min_quantity=cfg.exchange_min_quantity,
        max_quantity=cfg.exchange_max_quantity,
        quantity_step=cfg.exchange_quantity_step,
        price_tick_size=cfg.exchange_price_tick_size,
    )
    base_backtest_cfg = BacktestConfig(
        initial_balance=initial_balance,
        leverage=cfg.default_leverage,
        max_allowed_leverage=cfg.max_leverage,
        fee_rate=cfg.fee_rate,
        maker_fee_rate=cfg.maker_fee_rate,
        maker_fill_fraction=cfg.maker_fill_fraction,
        slippage_rate=cfg.slippage_rate,
        partial_fill_ratio=cfg.partial_fill_ratio,
        execution_latency_bars=cfg.execution_latency_bars,
        min_order_notional_fraction=cfg.min_order_notional_fraction,
        **exchange_rules,
        exchange_downtime_guard_enabled=cfg.exchange_downtime_guard_enabled,
        exchange_gap_recovery_bars=cfg.exchange_gap_recovery_bars,
        liquidity_execution_enabled=cfg.liquidity_execution_enabled,
        max_bar_participation_rate=cfg.max_bar_participation_rate,
        liquidity_lookback_bars=cfg.liquidity_lookback_bars,
        slippage_impact_coefficient=cfg.slippage_impact_coefficient,
        max_dynamic_slippage_rate=cfg.max_dynamic_slippage_rate,
        liquidation_guard_enabled=cfg.liquidation_guard_enabled,
        maintenance_margin_rate=cfg.maintenance_margin_rate,
        liquidation_buffer=cfg.liquidation_buffer,
        liquidation_fee_rate=cfg.liquidation_fee_rate,
        ewma_volatility_enabled=cfg.ewma_volatility_enabled,
        ewma_volatility_span=cfg.ewma_volatility_span,
        ewma_daily_volatility_target=cfg.ewma_daily_volatility_target,
        funding_crowding_guard_enabled=cfg.funding_crowding_guard_enabled,
        funding_crowding_max_rate=cfg.funding_crowding_max_rate,
        regime_risk_guard_enabled=cfg.regime_risk_guard_enabled,
        regime_detection_method=cfg.regime_detection_method,
        regime_statistical_clusters=cfg.regime_statistical_clusters,
        regime_statistical_min_history=cfg.regime_statistical_min_history,
        regime_statistical_lookback=cfg.regime_statistical_lookback,
        regime_statistical_refit_interval=cfg.regime_statistical_refit_interval,
        regime_statistical_random_seed=cfg.regime_statistical_random_seed,
        long_threshold=cfg.long_threshold,
        short_threshold=cfg.short_threshold,
    )
    param_report = optimize_backtest_params(
        valid_df,
        test_df,
        bundle,
        base_backtest_cfg,
        max_leverage=max_leverage,
        n_trials=n_trials,
        seed=cfg.random_seed,
        min_trades=min_trades,
        max_drawdown_floor=-abs(max_drawdown_limit),
        min_profit_factor=min_profit_factor,
    )
    test_detail = param_report.pop("test_detail")
    detail_path = cfg.reports_dir / f"{symbol}_{interval}_ai_optimized_backtest.csv"
    test_detail.to_csv(detail_path, index=False)

    report = {
        "created_utc": beijing_now_iso(),
        "created_beijing": beijing_now_iso(),
        "symbol": symbol,
        "interval": interval,
        "primary_label_horizon": label_horizon,
        "primary_label_min_return": label_min_return,
        "configured_label_min_return": cfg.label_min_return,
        "split_purge_rows": split_purge_rows,
        "split_purge_policy": "drop_tail_rows_from_train_and_validation_to_prevent_future_label_overlap",
        "rows": len(feature_frame),
        "data_validation": data_validation,
        "market_context": market_context,
        "model_name": bundle.model_name,
        "model_path": str(model_path),
        "model_metrics": bundle.metrics,
        "model_report": model_report,
        "parameter_optimization": param_report,
        "detail_path": str(detail_path),
    }
    report_path = cfg.reports_dir / f"{symbol}_{interval}_ai_optimization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
