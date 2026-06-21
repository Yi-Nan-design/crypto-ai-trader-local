from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ai_optimization import run_ai_optimization_for_symbol
from .backtest import BacktestConfig, run_backtest, save_backtest_report
from .binance_data import (
    diagnose_network,
    download_monthly_klines,
    import_kline_zip,
    load_symbol_interval,
    resolve_exchange_rule_values,
    sync_futures_market_context,
    sync_recent_futures_klines,
)
from .config import load_config
from .data_maintenance import run_data_maintenance
from .features import make_features, feature_matrix, time_split
from .live_training import download_result_payload, train_live_symbol
from .model_optimization import run_model_optimization
from .models import ModelBundle, train_candidate_models, train_candidate_models_with_report, classification_metrics
from .paper import run_paper_replay
from .progress import tracker_for_reports
from .scheduled_optimizer import run_scheduled_optimization_loop, run_scheduled_optimization_once
from .strategy_config import primary_label_horizon, primary_label_min_return
from .strategy_validation import run_strategy_validation
from .time_utils import beijing_now_iso, beijing_stamp


def cmd_download(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    results = download_monthly_klines(
        symbols=args.symbols,
        interval=args.interval or cfg.interval,
        start=args.start,
        end=args.end,
        data_dir=cfg.data_dir,
        market=args.market or cfg.market,
        fallback_daily=not args.no_daily_fallback,
        base_url=args.data_base_url or cfg.data_base_url or None,
        cache_only=args.cache_only,
        allow_partial_cache=args.allow_partial_cache,
    )
    for result in results:
        skipped = f", skipped {len(result.skipped or [])}" if result.skipped else ""
        print(f"{result.symbol} {result.interval}: saved {len(result.files)} files, {result.rows} rows{skipped}")


def cmd_sync_market_context(args: argparse.Namespace) -> None:
    """Download public funding and exchange filters for local simulation."""

    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    results = sync_futures_market_context(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        data_dir=cfg.data_dir,
        base_url=args.base_url or cfg.realtime_base_url,
    )
    payload = {
        "created_beijing": beijing_now_iso(),
        "start": args.start,
        "end": args.end,
        "live_trading_enabled": False,
        "symbols": [
            {
                "symbol": result.symbol,
                "funding_file": str(result.funding_file),
                "funding_rows": result.funding_rows,
                "exchange_rules_file": str(result.exchange_rules_file),
                "exchange_rules": result.exchange_rules.to_dict(),
            }
            for result in results
        ],
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        cfg.reports_dir
        / f"market_context_sync_{beijing_stamp()}.json"
    )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "report_path": str(report_path)}, indent=2))


def train_one(symbol: str, interval: str, args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    raw = load_symbol_interval(cfg.data_dir, symbol, interval)
    feature_frame = make_primary_feature_frame(raw, interval, cfg)
    train_df, valid_df, test_df = time_split(feature_frame, cfg.train_fraction, cfg.validation_fraction)
    x_train, y_train, cols = feature_matrix(train_df)
    x_valid, y_valid, _ = feature_matrix(valid_df)
    x_test, y_test, _ = feature_matrix(test_df)

    bundle, model_report = train_candidate_models_with_report(x_train, y_train, x_valid, y_valid, cols, seed=cfg.random_seed)
    test_prob = bundle.predict_up_probability(x_test)
    test_metrics = classification_metrics(y_test, test_prob)
    bundle.metrics = {**bundle.metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}

    output = cfg.model_dir / f"{symbol.upper()}_{interval}.pkl"
    bundle.save(output)
    metrics_path = cfg.reports_dir / f"{symbol.upper()}_{interval}_train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "symbol": symbol.upper(),
                "interval": interval,
                "rows": len(feature_frame),
                "model_name": bundle.model_name,
                "features": cols,
                "metrics": bundle.metrics,
                "model_report": model_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"trained {symbol.upper()} {interval}: {bundle.model_name}")
    print(json.dumps(bundle.metrics, indent=2))
    print(f"model: {output}")
    print(f"metrics: {metrics_path}")
    return output


def apply_network_overrides(cfg, args: argparse.Namespace) -> None:
    proxy = getattr(args, "proxy", None) or cfg.https_proxy
    if not proxy and cfg.auto_detect_proxy:
        proxy = detect_local_proxy()
    if proxy:
        os.environ["HTTPS_PROXY"] = proxy
        os.environ["HTTP_PROXY"] = proxy
    data_base_url = getattr(args, "data_base_url", None) or cfg.data_base_url
    data_base_urls = [data_base_url] if data_base_url else list(getattr(cfg, "data_base_urls", ()) or ())
    data_base_urls = [item for item in data_base_urls if item]
    if data_base_urls:
        os.environ["BINANCE_DATA_BASE_URLS"] = ";".join(data_base_urls)
        os.environ["BINANCE_DATA_BASE_URL"] = data_base_urls[0]


def detect_local_proxy() -> str:
    ports = [7890, 7891, 7897, 1080, 1087, 10808, 10809, 20170, 2080, 8080, 8888]
    for port in ports:
        sock = socket.socket()
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", port))
            return f"http://127.0.0.1:{port}"
        except OSError:
            continue
        finally:
            sock.close()
    return ""


def make_primary_feature_frame(raw: pd.DataFrame, interval: str, cfg, *, drop_future_na: bool = True) -> pd.DataFrame:
    return make_features(
        raw,
        label_horizon=primary_label_horizon(interval, cfg),
        label_min_return=primary_label_min_return(interval, cfg),
        drop_future_na=drop_future_na,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    interval = args.interval or cfg.interval
    for symbol in args.symbols:
        train_one(symbol, interval, args)


def _backtest_cfg(
    args: argparse.Namespace,
    symbol: str | None = None,
) -> BacktestConfig:
    cfg = load_config(args.config)
    leverage = min(args.leverage or cfg.default_leverage, cfg.max_leverage)
    exchange_rules = (
        resolve_exchange_rule_values(
            cfg.data_dir,
            symbol,
            min_notional_usdt=cfg.exchange_min_notional_usdt,
            min_quantity=cfg.exchange_min_quantity,
            max_quantity=cfg.exchange_max_quantity,
            quantity_step=cfg.exchange_quantity_step,
            price_tick_size=cfg.exchange_price_tick_size,
        )
        if symbol
        else {
            "exchange_min_notional_usdt": cfg.exchange_min_notional_usdt,
            "exchange_min_quantity": cfg.exchange_min_quantity,
            "exchange_max_quantity": cfg.exchange_max_quantity,
            "exchange_quantity_step": cfg.exchange_quantity_step,
            "exchange_price_tick_size": cfg.exchange_price_tick_size,
        }
    )
    return BacktestConfig(
        initial_balance=args.initial_balance,
        leverage=leverage,
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
        long_threshold=args.long_threshold or cfg.long_threshold,
        short_threshold=args.short_threshold or cfg.short_threshold,
        risk_per_trade=cfg.risk_per_trade,
        min_confidence_gap=cfg.min_confidence_gap,
    )


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbol = args.symbol.upper()
    interval = args.interval or cfg.interval
    model_path = Path(args.model or (cfg.model_dir / f"{symbol}_{interval}.pkl"))
    bundle = ModelBundle.load(model_path)
    raw = load_symbol_interval(cfg.data_dir, symbol, interval)
    frame = make_primary_feature_frame(raw, interval, cfg)
    _, _, test_df = time_split(frame, cfg.train_fraction, cfg.validation_fraction)
    result, detail = run_backtest(test_df, bundle, _backtest_cfg(args, symbol))
    result = save_backtest_report(result, detail, cfg.reports_dir, f"{symbol}_{interval}")
    print(json.dumps(result.__dict__, indent=2))


def calibrate_thresholds(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bundle: ModelBundle,
    base_cfg: BacktestConfig,
    max_leverage: int = 3,
    min_trades: int = 12,
    max_drawdown_floor: float = -0.35,
) -> dict[str, object]:
    candidates = [0.53, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.72]
    rows: list[dict[str, object]] = []
    for leverage in range(1, max(1, max_leverage) + 1):
        for long_threshold in candidates:
            short_threshold = 1.0 - long_threshold
            cfg = BacktestConfig(
                **{
                    **base_cfg.__dict__,
                    "leverage": leverage,
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                }
            )
            valid_result, _ = run_backtest(valid_df, bundle, cfg)
            score = valid_result.total_return
            if valid_result.max_drawdown < max_drawdown_floor:
                score -= abs(valid_result.max_drawdown - max_drawdown_floor) * 2.0
            if valid_result.trades < min_trades:
                score -= 0.25
            if valid_result.profit_factor < 1.0:
                score -= 0.05
            rows.append(
                {
                    "leverage": leverage,
                    "long_threshold": long_threshold,
                    "short_threshold": short_threshold,
                    "score": score,
                    "valid": valid_result.__dict__,
                }
            )

    best = dict(sorted(rows, key=lambda item: item["score"], reverse=True)[0])
    best_cfg = BacktestConfig(
        **{
            **base_cfg.__dict__,
            "leverage": int(best["leverage"]),
            "long_threshold": float(best["long_threshold"]),
            "short_threshold": float(best["short_threshold"]),
        }
    )
    test_result, test_detail = run_backtest(test_df, bundle, best_cfg)
    best["test"] = test_result.__dict__
    best["test_detail"] = test_detail
    best["all_candidates"] = rows
    return best


def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbol = args.symbol.upper()
    interval = args.interval or cfg.interval
    model_path = Path(args.model or (cfg.model_dir / f"{symbol}_{interval}.pkl"))
    bundle = ModelBundle.load(model_path)
    raw = load_symbol_interval(cfg.data_dir, symbol, interval)
    frame = make_primary_feature_frame(raw, interval, cfg)
    _, valid_df, test_df = time_split(frame, cfg.train_fraction, cfg.validation_fraction)
    best = calibrate_thresholds(
        valid_df,
        test_df,
        bundle,
        _backtest_cfg(args, symbol),
        max_leverage=min(args.max_leverage, cfg.max_leverage),
        min_trades=args.min_trades,
        max_drawdown_floor=-abs(args.max_drawdown_limit),
    )
    detail = best.pop("test_detail")
    name = f"{symbol}_{interval}_calibrated"
    result_payload = {
        key: value
        for key, value in best.items()
        if key != "all_candidates"
    }
    report_path = cfg.reports_dir / f"{name}_thresholds.json"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(cfg.reports_dir / f"{name}_backtest.csv", index=False)
    report_path.write_text(
        json.dumps({**result_payload, "all_candidates": best["all_candidates"]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result_payload, indent=2))
    print(f"calibration report: {report_path}")


def cmd_optimize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    interval = args.interval or cfg.interval
    cfg.ensure_dirs()
    aggregate: list[dict[str, object]] = []
    for symbol in args.symbols:
        symbol = symbol.upper()
        model_path = Path(args.model_dir or cfg.model_dir) / f"{symbol}_{interval}.pkl"
        bundle = ModelBundle.load(model_path)
        raw = load_symbol_interval(cfg.data_dir, symbol, interval)
        frame = make_primary_feature_frame(raw, interval, cfg)
        _, valid_df, test_df = time_split(frame, cfg.train_fraction, cfg.validation_fraction)
        best = calibrate_thresholds(
            valid_df,
            test_df,
            bundle,
            _backtest_cfg(args, symbol),
            max_leverage=min(args.max_leverage, cfg.max_leverage),
            min_trades=args.min_trades,
            max_drawdown_floor=-abs(args.max_drawdown_limit),
        )
        detail = best.pop("test_detail")
        name = f"{symbol}_{interval}_optimized"
        detail.to_csv(cfg.reports_dir / f"{name}_backtest.csv", index=False)
        report_path = cfg.reports_dir / f"{name}_thresholds.json"
        report_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        item = {
            "symbol": symbol,
            "interval": interval,
            "model_name": bundle.model_name,
            "leverage": best["leverage"],
            "long_threshold": best["long_threshold"],
            "short_threshold": best["short_threshold"],
            "valid": best["valid"],
            "test": best["test"],
            "report_path": str(report_path),
        }
        aggregate.append(item)
        print(json.dumps(item, indent=2))

    ranked = sorted(aggregate, key=lambda item: item["test"]["total_return"], reverse=True)
    output = cfg.reports_dir / f"optimization_{interval}_{beijing_stamp()}.json"
    output.write_text(json.dumps({"items": aggregate, "ranked": ranked}, indent=2), encoding="utf-8")
    print(f"optimization report: {output}")


def cmd_ai_optimize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    interval = args.interval or cfg.interval
    symbols = [item.upper() for item in args.symbols]
    progress = tracker_for_reports(cfg.reports_dir)
    progress.reset(f"AI optimization {interval}", len(symbols))
    reports: list[dict[str, object]] = []
    try:
        for symbol in symbols:
            progress.update(f"AI optimizing {symbol}", current_symbol=symbol)
            raw = load_symbol_interval(cfg.data_dir, symbol, interval)
            report = run_ai_optimization_for_symbol(
                symbol,
                interval,
                raw,
                cfg,
                initial_balance=args.initial_balance,
                max_leverage=min(args.max_leverage, cfg.max_leverage),
                n_trials=args.trials,
                min_trades=args.min_trades,
                max_drawdown_limit=args.max_drawdown_limit,
                min_profit_factor=args.min_profit_factor,
            )
            reports.append(report)
            best = report["parameter_optimization"]["best"]
            progress.advance(
                f"{symbol} AI optimization complete: {report['model_name']} return {best['test']['total_return']:.2%}",
                current_symbol=symbol,
                metrics={
                    "model_name": report["model_name"],
                    "total_return": best["test"]["total_return"],
                    "max_drawdown": best["test"]["max_drawdown"],
                    "trades": best["test"]["trades"],
                },
                event_type="ai_optimize",
            )
            if not args.quiet:
                print(json.dumps(report, indent=2))

        ranked = sorted(
            reports,
            key=lambda item: item["parameter_optimization"]["best"]["test"]["total_return"],
            reverse=True,
        )
        output = cfg.reports_dir / f"ai_optimization_{interval}_{beijing_stamp()}.json"
        output.write_text(json.dumps({"items": reports, "ranked": ranked}, indent=2), encoding="utf-8")
        progress.finish(f"AI optimization complete: {output}", metrics={"ai_optimization_report": str(output)})
        print(f"ai optimization report: {output}")
    except Exception as exc:
        progress.fail(str(exc))
        raise


def cmd_model_optimize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    result = run_model_optimization(
        cfg,
        symbols=[item.upper() for item in args.symbols],
        intervals=args.intervals,
        include_realtime=args.include_realtime,
        time_budget_minutes=args.time_budget_minutes,
        max_model_trials=args.max_model_trials,
        initial_balance=args.initial_balance,
        min_trades=args.min_trades,
        max_drawdown_limit=args.max_drawdown_limit,
        min_profit_factor=args.min_profit_factor,
        complexity=args.complexity,
        rolling_folds=args.rolling_folds,
        max_training_rows=args.max_training_rows,
        objective=args.objective,
    )
    summary = {
        "report_path": result["report_path"],
        "objective": result["objective"],
        "feature_version": result["feature_version"],
        "complexity": result.get("complexity", "standard"),
        "rolling_folds": result.get("rolling_folds", 0),
        "max_training_rows": result.get("max_training_rows"),
        "include_realtime": result["include_realtime"],
        "ranked": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "model_name": item["model_name"],
                "accuracy": item["model_report"]["test_metrics"].get("accuracy"),
                "balanced_accuracy": item["model_report"]["test_metrics"].get("balanced_accuracy"),
                "auc": item["model_report"]["test_metrics"].get("auc"),
                "log_loss": item["model_report"]["test_metrics"].get("log_loss"),
                "test_return": item["model_report"]["test_backtest"].get("total_return"),
                "test_drawdown": item["model_report"]["test_backtest"].get("max_drawdown"),
                "test_trades": item["model_report"]["test_backtest"].get("trades"),
                "test_profit_factor": item["model_report"]["test_backtest"].get("profit_factor"),
                "long_threshold": item.get("optimized_backtest_config", {}).get("long_threshold"),
                "short_threshold": item.get("optimized_backtest_config", {}).get("short_threshold"),
                "large_move_capture": item["model_report"].get("test_large_move_metrics", {}).get("large_move_capture"),
                "large_up_capture": item["model_report"].get("test_large_move_metrics", {}).get("large_up_capture"),
                "large_down_capture": item["model_report"].get("test_large_move_metrics", {}).get("large_down_capture"),
                "report_path": item["report_path"],
                "model_publish_status": item.get("model_publish", {}).get("status"),
                "model_publish_reason": item.get("model_publish", {}).get("reason"),
                "active_model_path": item["model_path"] if item.get("model_publish", {}).get("status") == "published" else None,
                "candidate_model_path": item.get("candidate_model_path"),
                "saved_model_path": (
                    item["model_path"]
                    if item.get("model_publish", {}).get("status") == "published"
                    else item.get("candidate_model_path")
                ),
            }
            for item in result["ranked"]
        ],
        "skipped_targets": result["skipped_targets"],
        "live_trading_enabled": result["live_trading_enabled"],
    }
    print(json.dumps(summary, indent=2))


def cmd_strategy_validate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    result = run_strategy_validation(
        cfg,
        symbols=[item.upper() for item in args.symbols],
        intervals=args.intervals,
        include_realtime=args.include_realtime,
        folds=args.folds,
        max_model_trials=args.max_model_trials,
        time_budget_minutes=args.time_budget_minutes,
        initial_balance=args.initial_balance,
        min_trades=args.min_trades,
        max_drawdown_limit=args.max_drawdown_limit,
        min_profit_factor=args.min_profit_factor,
        complexity=args.complexity,
        rolling_folds=args.rolling_folds,
        purge_rows=args.purge_rows,
        holdout_fraction=args.holdout_fraction,
        max_training_rows=args.max_training_rows,
        validation_profile=args.validation_profile,
        wf_train_rows=args.wf_train_rows,
        wf_valid_rows=args.wf_valid_rows,
        wf_test_rows=args.wf_test_rows,
        max_threshold_evals=args.max_threshold_evals,
        per_target_budget_minutes=args.per_target_budget_minutes,
        state_dir=args.state_dir,
    )
    summary = {
        "report_path": result["report_path"],
        "latest_report_path": result["latest_report_path"],
        "local_strategy_state_path": result["local_strategy_state_path"],
        "feature_version": result["feature_version"],
        "validation_profile": result.get("validation_profile"),
        "complexity": result["complexity"],
        "rolling_folds": result["rolling_folds"],
        "purge_rows": result.get("purge_rows"),
        "holdout_fraction": result.get("holdout_fraction"),
        "max_training_rows": result.get("max_training_rows"),
        "walk_forward_row_caps": result.get("walk_forward_row_caps"),
        "threshold_search_controls": result.get("threshold_search_controls"),
        "per_target_budget_minutes": result.get("per_target_budget_minutes"),
        "promotion_allowed": result.get("promotion_allowed"),
        "include_realtime": result["include_realtime"],
        "ranked": [
            {
                "symbol": item["symbol"],
                "interval": item["interval"],
                "folds_completed": item["folds_completed"],
                "eligible": item["summary"]["eligible_for_paper_candidate"],
                "recommendation": item["summary"]["recommendation"],
                "stability_score": item["summary"]["stability_score"],
                "profitable_fold_rate": item["summary"]["profitable_fold_rate"],
                "median_return": item["summary"]["median_return"],
                "mean_return": item["summary"]["mean_return"],
                "worst_drawdown": item["summary"]["worst_drawdown"],
                "mean_profit_factor": item["summary"]["mean_profit_factor"],
                "mean_balanced_accuracy": item["summary"]["mean_balanced_accuracy"],
                "frozen_holdout_status": item["summary"].get("frozen_holdout_status"),
                "frozen_holdout_gate_passed": item["summary"].get("frozen_holdout_gate_passed"),
                "report_path": item.get("report_path"),
            }
            for item in result["ranked"]
        ],
        "skipped_targets": result["skipped_targets"],
        "live_trading_enabled": result["live_trading_enabled"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_data_maintenance(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    payload = run_data_maintenance(
        config_path=args.config,
        dry_run=args.dry_run,
        archive_old_realtime=not args.no_archive_realtime,
        cleanup_tmp_older_than_hours=args.cleanup_tmp_older_than_hours,
        max_realtime_rows=args.max_realtime_rows,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_scheduled_optimize(args: argparse.Namespace) -> None:
    payload_args = {
        "config_path": args.config,
        "symbols": [item.upper() for item in args.symbols] if args.symbols else None,
        "intervals": args.intervals,
        "include_realtime": args.include_realtime,
        "time_budget_minutes": args.time_budget_minutes,
        "max_model_trials": args.max_model_trials,
        "max_training_rows": args.max_training_rows,
        "complexity": args.complexity,
        "rolling_folds": args.rolling_folds,
        "max_targets": args.max_targets,
        "maintenance": args.maintenance,
        "maintenance_dry_run": args.maintenance_dry_run,
    }
    if args.iterations and args.iterations > 1:
        run_scheduled_optimization_loop(
            iterations=args.iterations,
            sleep_minutes=args.sleep_minutes,
            **payload_args,
        )
        return
    payload = run_scheduled_optimization_once(**payload_args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_paper(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbol = args.symbol.upper()
    interval = args.interval or cfg.interval
    model_path = Path(args.model or (cfg.model_dir / f"{symbol}_{interval}.pkl"))
    bundle = ModelBundle.load(model_path)
    raw = load_symbol_interval(cfg.data_dir, symbol, interval)
    frame = make_primary_feature_frame(raw, interval, cfg)
    _, _, test_df = time_split(frame, cfg.train_fraction, cfg.validation_fraction)
    state, summary_path = run_paper_replay(
        test_df,
        bundle,
        _backtest_cfg(args, symbol),
        cfg.reports_dir,
        f"{symbol}_{interval}",
    )
    print(json.dumps({"state": state.__dict__, "summary_path": str(summary_path)}, indent=2))


def cmd_cycle(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    cfg.ensure_dirs()
    interval = args.interval or cfg.interval
    progress = tracker_for_reports(cfg.reports_dir)
    per_symbol_steps = 3
    total_steps = (0 if args.skip_download else len(args.symbols)) + len(args.symbols) * per_symbol_steps
    progress.reset(f"Training cycle {interval}", total_steps)
    try:
        if not args.skip_download:
            for symbol in args.symbols:
                symbol = symbol.upper()
                progress.update(f"Downloading {symbol} {interval} history", current_symbol=symbol)
                results = download_monthly_klines(
                    symbols=[symbol],
                    interval=interval,
                    start=args.start,
                    end=args.end,
                    data_dir=cfg.data_dir,
                    market=args.market or cfg.market,
                    fallback_daily=not args.no_daily_fallback,
                    base_url=args.data_base_url or cfg.data_base_url or None,
                    cache_only=args.cache_only,
                    allow_partial_cache=args.allow_partial_cache,
                )
                result = results[0]
                progress.advance(
                    f"{symbol} download complete: {result.rows} rows",
                    current_symbol=symbol,
                    metrics={"rows": result.rows},
                    event_type="download",
                )
                print(f"{result.symbol} {result.interval}: saved {len(result.files)} files, {result.rows} rows")

        aggregate: list[dict[str, object]] = []
        for symbol in args.symbols:
            symbol = symbol.upper()
            progress.update(f"Training {symbol} model", current_symbol=symbol)
            model_path = train_one(symbol, interval, args)
            bundle = ModelBundle.load(model_path)
            progress.advance(
                f"{symbol} model trained: {bundle.model_name}",
                current_symbol=symbol,
                metrics={"model_name": bundle.model_name, **bundle.metrics},
                event_type="train",
            )

            raw = load_symbol_interval(cfg.data_dir, symbol, interval)
            frame = make_primary_feature_frame(raw, interval, cfg)
            _, _, test_df = time_split(frame, cfg.train_fraction, cfg.validation_fraction)

            backtest_cfg = _backtest_cfg(args, symbol)
            progress.update(f"Backtesting {symbol}", current_symbol=symbol, metrics={"model_name": bundle.model_name})
            result, detail = run_backtest(test_df, bundle, backtest_cfg)
            result = save_backtest_report(result, detail, cfg.reports_dir, f"{symbol}_{interval}")
            progress.advance(
                f"{symbol} backtest complete: return {result.total_return:.2%}, drawdown {result.max_drawdown:.2%}",
                current_symbol=symbol,
                metrics={
                    "model_name": bundle.model_name,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "trades": result.trades,
                    "win_rate": result.win_rate,
                },
                event_type="backtest",
            )

            progress.update(f"Replaying paper trading for {symbol}", current_symbol=symbol)
            paper_state, paper_summary = run_paper_replay(
                test_df,
                bundle,
                backtest_cfg,
                cfg.reports_dir,
                f"{symbol}_{interval}",
            )
            progress.advance(
                f"{symbol} paper replay complete: equity {paper_state.equity:.2f}",
                current_symbol=symbol,
                metrics={"paper_equity": paper_state.equity, "paper_trades": paper_state.trades},
                event_type="paper",
            )
            aggregate.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "model_path": str(model_path),
                    "model_name": bundle.model_name,
                    "metrics": bundle.metrics,
                    "backtest": result.__dict__,
                    "paper": paper_state.__dict__,
                    "paper_summary": str(paper_summary),
                }
            )

        stamp = beijing_stamp()
        output = cfg.reports_dir / f"cycle_{interval}_{stamp}.json"
        output.write_text(
            json.dumps({"created_utc": beijing_now_iso(), "created_beijing": beijing_now_iso(), "stamp_beijing": stamp, "items": aggregate}, indent=2),
            encoding="utf-8",
        )
        progress.finish(f"Training cycle complete: {output}", metrics={"cycle_report": str(output)})
        print(f"cycle report: {output}")
    except Exception as exc:
        progress.fail(str(exc))
        raise


def cmd_live_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    cfg.ensure_dirs()
    symbols = [item.upper() for item in (args.symbols or cfg.symbols)]
    interval = args.interval or cfg.realtime_interval
    limit = args.limit or cfg.realtime_limit
    base_url = args.base_url or cfg.realtime_base_url
    progress = tracker_for_reports(cfg.reports_dir)
    total_steps = args.iterations * (1 + len(symbols))
    progress.reset(f"Live training {interval}", total_steps)
    all_runs: list[dict[str, object]] = []
    try:
        for iteration in range(1, args.iterations + 1):
            progress.update(
                f"Syncing closed klines iteration {iteration}/{args.iterations}",
                metrics={"interval": interval, "limit": limit},
            )
            sync_results = sync_recent_futures_klines(
                symbols=symbols,
                interval=interval,
                data_dir=cfg.data_dir,
                limit=limit,
                base_url=base_url,
            )
            progress.advance(
                f"Live kline sync complete: {', '.join(f'{item.symbol}:{item.rows}' for item in sync_results)}",
                metrics={"synced_symbols": len(sync_results)},
                event_type="live_sync",
            )

            reports = []
            for symbol in symbols:
                progress.update(f"Training live model for {symbol}", current_symbol=symbol)
                report = train_live_symbol(
                    symbol,
                    interval,
                    cfg,
                    include_realtime=True,
                    model_suffix=args.model_suffix,
                    max_model_trials=args.max_model_trials,
                    time_budget_minutes=args.time_budget_minutes,
                    complexity=args.complexity,
                    rolling_folds=args.rolling_folds,
                    max_training_rows=args.max_training_rows,
                )
                reports.append(report)
                progress.advance(
                    f"{symbol} live train complete: prob_up {report['latest_up_probability']:.4f}",
                    current_symbol=symbol,
                    metrics={
                        "model_name": report["model_name"],
                        "latest_up_probability": report["latest_up_probability"],
                        "total_return": report["backtest"]["total_return"],
                        "max_drawdown": report["backtest"]["max_drawdown"],
                    },
                    event_type="live_train",
                )

            ranked = sorted(reports, key=lambda item: item["backtest"]["total_return"], reverse=True)
            run_payload = {
                "iteration": iteration,
                "created_utc": beijing_now_iso(),
                "created_beijing": beijing_now_iso(),
                "sync": [download_result_payload(item) for item in sync_results],
                "items": reports,
                "ranked": ranked,
            }
            all_runs.append(run_payload)
            latest_path = cfg.reports_dir / f"live_training_{interval}_latest.json"
            latest_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

            if iteration < args.iterations:
                time.sleep(max(args.sleep_seconds, 1))

        output = cfg.reports_dir / f"live_training_{interval}_{beijing_stamp()}.json"
        output.write_text(json.dumps({"runs": all_runs}, indent=2), encoding="utf-8")
        progress.finish(f"Live training complete: {output}", metrics={"live_report": str(output)})
        print(f"live training report: {output}")
    except Exception as exc:
        progress.fail(str(exc))
        raise


def autonomous_policy_from_args(args: argparse.Namespace):
    from .autonomous_agent import AutonomousPolicy

    return AutonomousPolicy(
        min_total_return=args.min_total_return,
        max_drawdown_limit=args.max_drawdown_limit,
        min_profit_factor=args.min_profit_factor,
        min_trades=args.min_trades,
        min_rows=args.min_rows,
        max_report_age_minutes=args.max_report_age_minutes,
        optimization_trials=args.optimization_trials,
        optimization_cooldown_minutes=args.optimization_cooldown_minutes,
    )


def cmd_autonomous_review(args: argparse.Namespace) -> None:
    from .autonomous_agent import run_autonomous_review

    payload = run_autonomous_review(
        config_path=args.config,
        symbols=args.symbols,
        interval=args.interval,
        runner_interval=args.runner_interval,
        state_dir=args.state_dir,
        policy=autonomous_policy_from_args(args),
        execute_optimization=args.execute_optimization,
        execute_live_train=args.execute_live_train,
        live_limit=args.live_limit,
        live_base_url=args.live_base_url,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_memory_update(args: argparse.Namespace) -> None:
    from .simulation_memory import update_simulation_memory

    payload = update_simulation_memory(
        config_path=args.config,
        state_dir=args.state_dir,
        max_observations_per_target=args.max_observations_per_target,
    )
    summary = {
        "updated_beijing": payload.get("updated_beijing"),
        "observation_count": payload.get("observation_count"),
        "target_count": payload.get("target_count"),
        "top_candidates": payload.get("global", {}).get("top_candidates", [])[:5],
        "next_actions": payload.get("global", {}).get("next_actions", []),
        "report_path": payload.get("report_path"),
        "state_path": payload.get("state_path"),
        "live_trading_enabled": payload.get("safety", {}).get("live_trading_enabled", False),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_autonomous_loop(args: argparse.Namespace) -> None:
    from .autonomous_agent import run_autonomous_loop

    run_autonomous_loop(
        review_every_seconds=args.review_every_seconds,
        iterations=args.iterations,
        config_path=args.config,
        symbols=args.symbols,
        interval=args.interval,
        runner_interval=args.runner_interval,
        state_dir=args.state_dir,
        policy=autonomous_policy_from_args(args),
        execute_optimization=args.execute_optimization,
        execute_live_train=args.execute_live_train,
        live_limit=args.live_limit,
        live_base_url=args.live_base_url,
    )


def cmd_summarize(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    reports = sorted(cfg.reports_dir.glob("*_backtest_summary.json"))
    metrics = sorted(cfg.reports_dir.glob("*_train_metrics.json"))
    if not reports and not metrics:
        print(f"No reports found in {cfg.reports_dir}")
        return

    print("TRAINING")
    for path in metrics[-20:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model_name", "")
        rows = payload.get("rows", 0)
        item = payload.get("metrics", {})
        print(
            f"{payload.get('symbol')} {payload.get('interval')} model={model} rows={rows} "
            f"valid_acc={item.get('accuracy', 0):.4f} test_acc={item.get('test_accuracy', 0):.4f} "
            f"test_loss={item.get('test_log_loss', 0):.4f}"
        )

    print("BACKTEST")
    for path in reports[-20:]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = path.name.replace("_backtest_summary.json", "")
        print(
            f"{name} return={payload.get('total_return', 0):.4f} "
            f"max_dd={payload.get('max_drawdown', 0):.4f} trades={payload.get('trades', 0)} "
            f"win_rate={payload.get('win_rate', 0):.4f} pf={payload.get('profit_factor', 0):.4f}"
        )


def cmd_doctor(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_network_overrides(cfg, args)
    cfg.ensure_dirs()
    payload = diagnose_network(cfg.reports_dir / "network_diagnostic.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_import_zip(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    symbol = args.symbol.upper()
    output = cfg.data_dir / "raw" / symbol / args.interval / f"{symbol}-{args.interval}-{args.period}.csv"
    saved = import_kline_zip(args.zip, output)
    print(f"imported: {saved}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    from .dashboard_server import main as dashboard_main

    dashboard_main(["--host", args.host, "--port", str(args.port)])


def cmd_smoke(_: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    n = 2500
    drift = rng.normal(0, 0.002, size=n)
    signal = np.sin(np.arange(n) / 35) * 0.002
    close = 100 * np.exp(np.cumsum(drift + signal))
    open_ = close * (1 + rng.normal(0, 0.0008, size=n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0001, 0.004, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0001, 0.004, size=n))
    volume = rng.lognormal(8, 0.4, size=n)
    df = pd.DataFrame(
        {
            "open_time": np.arange(n) * 3600_000,
            "open_datetime": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(300, 4000, size=n),
            "taker_buy_quote_volume": volume * close * rng.uniform(0.35, 0.65, size=n),
        }
    )
    frame = make_features(df)
    train_df, valid_df, test_df = time_split(frame)
    x_train, y_train, cols = feature_matrix(train_df)
    x_valid, y_valid, _ = feature_matrix(valid_df)
    bundle = train_candidate_models(x_train, y_train, x_valid, y_valid, cols)
    result, _ = run_backtest(test_df, bundle, BacktestConfig())
    print(
        json.dumps(
            {
                "rows": len(frame),
                "model": bundle.model_name,
                "validation_metrics": bundle.metrics,
                "backtest": result.__dict__,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance crypto AI training, backtest and paper trading")
    parser.add_argument("--config", default="config.default.json")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--interval", default=None)
    download.add_argument("--start", required=True, help="YYYY-MM")
    download.add_argument("--end", required=True, help="YYYY-MM")
    download.add_argument("--market", default=None, choices=["futures_um", "spot"])
    download.add_argument("--proxy", default=None)
    download.add_argument("--data-base-url", default=None)
    download.add_argument("--no-daily-fallback", action="store_true")
    download.add_argument("--cache-only", action="store_true", help="Use local CSV/zip cache only; do not contact remote data sources.")
    download.add_argument("--allow-partial-cache", action="store_true", help="Continue when some requested months are missing from local/remote data.")
    download.set_defaults(func=cmd_download)

    market_context = sub.add_parser("sync-market-context")
    market_context.add_argument("--symbols", nargs="+", required=True)
    market_context.add_argument("--start", required=True, help="YYYY-MM or YYYY-MM-DD")
    market_context.add_argument("--end", required=True, help="YYYY-MM or YYYY-MM-DD")
    market_context.add_argument("--proxy", default=None)
    market_context.add_argument("--base-url", default=None)
    market_context.set_defaults(func=cmd_sync_market_context)

    train = sub.add_parser("train")
    train.add_argument("--symbols", nargs="+", required=True)
    train.add_argument("--interval", default=None)
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--interval", default=None)
    backtest.add_argument("--model", default=None)
    backtest.add_argument("--initial-balance", type=float, default=10_000)
    backtest.add_argument("--leverage", type=int, default=None)
    backtest.add_argument("--long-threshold", type=float, default=None)
    backtest.add_argument("--short-threshold", type=float, default=None)
    backtest.set_defaults(func=cmd_backtest)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--symbol", required=True)
    calibrate.add_argument("--interval", default=None)
    calibrate.add_argument("--model", default=None)
    calibrate.add_argument("--initial-balance", type=float, default=10_000)
    calibrate.add_argument("--leverage", type=int, default=None)
    calibrate.add_argument("--max-leverage", type=int, default=3)
    calibrate.add_argument("--min-trades", type=int, default=12)
    calibrate.add_argument("--max-drawdown-limit", type=float, default=0.35)
    calibrate.add_argument("--long-threshold", type=float, default=None)
    calibrate.add_argument("--short-threshold", type=float, default=None)
    calibrate.set_defaults(func=cmd_calibrate)

    paper = sub.add_parser("paper")
    paper.add_argument("--symbol", required=True)
    paper.add_argument("--interval", default=None)
    paper.add_argument("--model", default=None)
    paper.add_argument("--initial-balance", type=float, default=10_000)
    paper.add_argument("--leverage", type=int, default=None)
    paper.add_argument("--long-threshold", type=float, default=None)
    paper.add_argument("--short-threshold", type=float, default=None)
    paper.set_defaults(func=cmd_paper)

    cycle = sub.add_parser("cycle")
    cycle.add_argument("--symbols", nargs="+", required=True)
    cycle.add_argument("--interval", default=None)
    cycle.add_argument("--start", required=True, help="YYYY-MM")
    cycle.add_argument("--end", required=True, help="YYYY-MM")
    cycle.add_argument("--market", default=None, choices=["futures_um", "spot"])
    cycle.add_argument("--proxy", default=None)
    cycle.add_argument("--data-base-url", default=None)
    cycle.add_argument("--no-daily-fallback", action="store_true")
    cycle.add_argument("--skip-download", action="store_true")
    cycle.add_argument("--cache-only", action="store_true", help="Use local CSV/zip cache only during the download phase.")
    cycle.add_argument("--allow-partial-cache", action="store_true", help="Continue cycle if some requested months are unavailable.")
    cycle.add_argument("--initial-balance", type=float, default=10_000)
    cycle.add_argument("--leverage", type=int, default=None)
    cycle.add_argument("--long-threshold", type=float, default=None)
    cycle.add_argument("--short-threshold", type=float, default=None)
    cycle.set_defaults(func=cmd_cycle)

    optimize = sub.add_parser("optimize")
    optimize.add_argument("--symbols", nargs="+", required=True)
    optimize.add_argument("--interval", default=None)
    optimize.add_argument("--model-dir", default=None)
    optimize.add_argument("--initial-balance", type=float, default=10_000)
    optimize.add_argument("--leverage", type=int, default=None)
    optimize.add_argument("--max-leverage", type=int, default=3)
    optimize.add_argument("--min-trades", type=int, default=12)
    optimize.add_argument("--max-drawdown-limit", type=float, default=0.35)
    optimize.add_argument("--long-threshold", type=float, default=None)
    optimize.add_argument("--short-threshold", type=float, default=None)
    optimize.set_defaults(func=cmd_optimize)

    ai_optimize = sub.add_parser("ai-optimize")
    ai_optimize.add_argument("--symbols", nargs="+", required=True)
    ai_optimize.add_argument("--interval", default=None)
    ai_optimize.add_argument("--initial-balance", type=float, default=10_000)
    ai_optimize.add_argument("--max-leverage", type=int, default=3)
    ai_optimize.add_argument("--trials", type=int, default=80)
    ai_optimize.add_argument("--min-trades", type=int, default=12)
    ai_optimize.add_argument("--max-drawdown-limit", type=float, default=0.35)
    ai_optimize.add_argument("--min-profit-factor", type=float, default=1.0)
    ai_optimize.add_argument("--quiet", action="store_true")
    ai_optimize.set_defaults(func=cmd_ai_optimize)

    model_optimize = sub.add_parser("model-optimize")
    model_optimize.add_argument("--symbols", nargs="+", default=["SOLUSDT", "ETHUSDT", "BNBUSDT"])
    model_optimize.add_argument("--intervals", nargs="+", default=["1h"])
    model_optimize.add_argument("--objective", default="return", choices=["return", "accuracy"])
    model_optimize.add_argument("--complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    model_optimize.add_argument("--rolling-folds", type=int, default=0)
    model_optimize.add_argument("--include-realtime", action="store_true")
    model_optimize.add_argument("--time-budget-minutes", type=float, default=15)
    model_optimize.add_argument("--max-model-trials", type=int, default=1)
    model_optimize.add_argument("--max-training-rows", type=int, default=12_000, help="Use the most recent N rows per target; 0 means full history.")
    model_optimize.add_argument("--initial-balance", type=float, default=10_000)
    model_optimize.add_argument("--min-trades", type=int, default=8)
    model_optimize.add_argument("--max-drawdown-limit", type=float, default=0.10)
    model_optimize.add_argument("--min-profit-factor", type=float, default=1.0)
    model_optimize.add_argument("--proxy", default=None)
    model_optimize.add_argument("--data-base-url", default=None)
    model_optimize.set_defaults(func=cmd_model_optimize)

    strategy_validate = sub.add_parser("strategy-validate")
    strategy_validate.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    strategy_validate.add_argument("--intervals", nargs="+", default=["1h"])
    strategy_validate.add_argument("--complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    strategy_validate.add_argument(
        "--validation-profile",
        default="standard",
        choices=["standard", "large-sample-light", "fast-screen", "deep-audit"],
        help="Validation budget profile. Non-standard profiles bound walk-forward windows and threshold search while keeping holdout gates.",
    )
    strategy_validate.add_argument("--rolling-folds", type=int, default=1)
    strategy_validate.add_argument("--purge-rows", type=int, default=None)
    strategy_validate.add_argument("--holdout-fraction", type=float, default=0.15)
    strategy_validate.add_argument("--include-realtime", action="store_true")
    strategy_validate.add_argument("--folds", type=int, default=3)
    strategy_validate.add_argument("--time-budget-minutes", type=float, default=30)
    strategy_validate.add_argument(
        "--per-target-budget-minutes",
        type=float,
        default=None,
        help="Optional per symbol/interval budget so early targets cannot consume the whole run.",
    )
    strategy_validate.add_argument("--max-model-trials", type=int, default=4)
    strategy_validate.add_argument("--max-training-rows", type=int, default=0, help="Use the most recent N rows per target; 0 means full history.")
    strategy_validate.add_argument("--wf-train-rows", type=int, default=None, help="Optional train rows per walk-forward fold.")
    strategy_validate.add_argument("--wf-valid-rows", type=int, default=None, help="Optional validation rows per walk-forward fold.")
    strategy_validate.add_argument("--wf-test-rows", type=int, default=None, help="Optional test rows per walk-forward fold.")
    strategy_validate.add_argument(
        "--max-threshold-evals",
        type=int,
        default=None,
        help="Optional cap on threshold/risk/side combinations evaluated per calibration pass.",
    )
    strategy_validate.add_argument("--initial-balance", type=float, default=10_000)
    strategy_validate.add_argument("--min-trades", type=int, default=8)
    strategy_validate.add_argument("--max-drawdown-limit", type=float, default=0.10)
    strategy_validate.add_argument("--min-profit-factor", type=float, default=0.8)
    strategy_validate.add_argument("--state-dir", default="state")
    strategy_validate.add_argument("--proxy", default=None)
    strategy_validate.add_argument("--data-base-url", default=None)
    strategy_validate.set_defaults(func=cmd_strategy_validate)

    scheduled_optimize = sub.add_parser("scheduled-optimize")
    scheduled_optimize.add_argument("--symbols", nargs="+", default=None)
    scheduled_optimize.add_argument("--intervals", nargs="+", default=None)
    scheduled_optimize.add_argument("--include-realtime", dest="include_realtime", action="store_true", default=True)
    scheduled_optimize.add_argument("--no-include-realtime", dest="include_realtime", action="store_false")
    scheduled_optimize.add_argument("--time-budget-minutes", type=float, default=12)
    scheduled_optimize.add_argument("--max-model-trials", type=int, default=1)
    scheduled_optimize.add_argument("--max-training-rows", type=int, default=12_000)
    scheduled_optimize.add_argument("--complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    scheduled_optimize.add_argument("--rolling-folds", type=int, default=0)
    scheduled_optimize.add_argument("--max-targets", type=int, default=1)
    scheduled_optimize.add_argument("--maintenance", dest="maintenance", action="store_true", default=True)
    scheduled_optimize.add_argument("--no-maintenance", dest="maintenance", action="store_false")
    scheduled_optimize.add_argument("--maintenance-dry-run", action="store_true")
    scheduled_optimize.add_argument("--iterations", type=int, default=1)
    scheduled_optimize.add_argument("--sleep-minutes", type=float, default=60)
    scheduled_optimize.set_defaults(func=cmd_scheduled_optimize)

    data_maintenance = sub.add_parser("data-maintenance")
    data_maintenance.add_argument("--dry-run", action="store_true")
    data_maintenance.add_argument("--no-archive-realtime", action="store_true")
    data_maintenance.add_argument("--cleanup-tmp-older-than-hours", type=float, default=6.0)
    data_maintenance.add_argument("--max-realtime-rows", type=int, default=6000)
    data_maintenance.set_defaults(func=cmd_data_maintenance)

    live_train = sub.add_parser("live-train")
    live_train.add_argument("--symbols", nargs="+", default=None)
    live_train.add_argument("--interval", default=None)
    live_train.add_argument("--limit", type=int, default=None)
    live_train.add_argument("--iterations", type=int, default=1)
    live_train.add_argument("--sleep-seconds", type=int, default=60)
    live_train.add_argument("--base-url", default=None)
    live_train.add_argument("--proxy", default=None)
    live_train.add_argument("--data-base-url", default=None)
    live_train.add_argument("--model-suffix", default="live")
    live_train.add_argument("--max-model-trials", type=int, default=1)
    live_train.add_argument("--time-budget-minutes", type=float, default=3.0)
    live_train.add_argument("--complexity", default="standard", choices=["standard", "expanded", "deep", "blackbox"])
    live_train.add_argument("--rolling-folds", type=int, default=0)
    live_train.add_argument("--max-training-rows", type=int, default=3000)
    live_train.set_defaults(func=cmd_live_train)

    autonomous_review = sub.add_parser("autonomous-review")
    autonomous_review.add_argument("--symbols", nargs="+", default=None)
    autonomous_review.add_argument("--interval", default=None)
    autonomous_review.add_argument("--runner-interval", default=None)
    autonomous_review.add_argument("--state-dir", default="state")
    autonomous_review.add_argument("--min-total-return", type=float, default=0.0)
    autonomous_review.add_argument("--max-drawdown-limit", type=float, default=0.08)
    autonomous_review.add_argument("--min-profit-factor", type=float, default=1.0)
    autonomous_review.add_argument("--min-trades", type=int, default=12)
    autonomous_review.add_argument("--min-rows", type=int, default=300)
    autonomous_review.add_argument("--max-report-age-minutes", type=int, default=90)
    autonomous_review.add_argument("--optimization-trials", type=int, default=40)
    autonomous_review.add_argument("--optimization-cooldown-minutes", type=int, default=360)
    autonomous_review.add_argument("--execute-optimization", action="store_true")
    autonomous_review.add_argument("--execute-live-train", action="store_true")
    autonomous_review.add_argument("--live-limit", type=int, default=None)
    autonomous_review.add_argument("--live-base-url", default=None)
    autonomous_review.set_defaults(func=cmd_autonomous_review)

    memory_update = sub.add_parser("memory-update")
    memory_update.add_argument("--state-dir", default="state")
    memory_update.add_argument("--max-observations-per-target", type=int, default=120)
    memory_update.set_defaults(func=cmd_memory_update)

    autonomous_loop = sub.add_parser("autonomous-loop")
    autonomous_loop.add_argument("--symbols", nargs="+", default=None)
    autonomous_loop.add_argument("--interval", default=None)
    autonomous_loop.add_argument("--runner-interval", default=None)
    autonomous_loop.add_argument("--state-dir", default="state")
    autonomous_loop.add_argument("--review-every-seconds", type=int, default=3600)
    autonomous_loop.add_argument("--iterations", type=int, default=0)
    autonomous_loop.add_argument("--min-total-return", type=float, default=0.0)
    autonomous_loop.add_argument("--max-drawdown-limit", type=float, default=0.08)
    autonomous_loop.add_argument("--min-profit-factor", type=float, default=1.0)
    autonomous_loop.add_argument("--min-trades", type=int, default=12)
    autonomous_loop.add_argument("--min-rows", type=int, default=300)
    autonomous_loop.add_argument("--max-report-age-minutes", type=int, default=90)
    autonomous_loop.add_argument("--optimization-trials", type=int, default=40)
    autonomous_loop.add_argument("--optimization-cooldown-minutes", type=int, default=360)
    autonomous_loop.add_argument("--execute-optimization", action="store_true")
    autonomous_loop.add_argument("--execute-live-train", action="store_true")
    autonomous_loop.add_argument("--live-limit", type=int, default=None)
    autonomous_loop.add_argument("--live-base-url", default=None)
    autonomous_loop.set_defaults(func=cmd_autonomous_loop)

    summarize = sub.add_parser("summarize")
    summarize.set_defaults(func=cmd_summarize)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--proxy", default=None)
    doctor.add_argument("--data-base-url", default=None)
    doctor.set_defaults(func=cmd_doctor)

    import_zip = sub.add_parser("import-zip")
    import_zip.add_argument("--symbol", required=True)
    import_zip.add_argument("--interval", required=True)
    import_zip.add_argument("--period", required=True, help="YYYY-MM or YYYY-MM-DD")
    import_zip.add_argument("--zip", required=True)
    import_zip.set_defaults(func=cmd_import_zip)

    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.set_defaults(func=cmd_dashboard)

    smoke = sub.add_parser("smoke")
    smoke.set_defaults(func=cmd_smoke)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
