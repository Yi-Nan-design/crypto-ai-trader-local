from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .binance_data import (
    DownloadResult,
    load_symbol_interval,
    resolve_exchange_rule_values,
    sync_recent_futures_klines,
)
from .config import TraderConfig
from .contracts import AlphaPrediction, RiskDecision, RiskLevel
from .exchange_availability import coerce_exchange_available
from .features import MULTI_HORIZON_STEPS, feature_only_matrix, feature_matrix, make_features, time_split
from .meta_signal import (
    MetaSignalInputs,
    WeightedMetaSignal,
    funding_alpha_scores,
)
from .model_optimization import train_accuracy_first_candidates
from .monitoring import (
    MONITORING_ALGORITHM_VERSION,
    MONITORING_SCHEMA_VERSION,
    build_feature_reference,
    build_monitoring_snapshot,
    micro_regime_distribution,
    rolling_performance,
)
from .progress import safe_replace_text
from .regime import latest_regime_state
from .strategy import StrategyContext, StrategyOrchestrator
from .strategy_config import primary_label_horizon, primary_label_min_return
from .time_utils import beijing_now_iso, beijing_stamp


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _model_version(
    *,
    model_name: str,
    feature_columns: list[str],
    model_config: dict[str, object],
    latest_training_open_time: int,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> str:
    """Return a stable fingerprint for the trained data/config contract."""

    payload = {
        "model_name": model_name,
        "feature_columns": feature_columns,
        "model_config": model_config,
        "latest_training_open_time": int(latest_training_open_time),
        "train_rows": int(train_rows),
        "valid_rows": int(valid_rows),
        "test_rows": int(test_rows),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:20]


def _alpha_model_version(bundle: object) -> str:
    """Describe every fitted model contributing to the Alpha contract."""

    classifier_name = str(getattr(bundle, "model_name", "unknown"))
    metadata = getattr(bundle, "auxiliary_metadata", {}) or {}
    expected_return_metadata = dict(
        metadata.get("alpha_expected_return", {}) or {}
    )
    version = f"classifier:{classifier_name}"
    if expected_return_metadata.get("model_name"):
        version += (
            f"|expected_return:{expected_return_metadata['model_name']}"
            f"|interface:{expected_return_metadata.get('interface_version', 'unknown')}"
        )
    return version


def _prune_versioned_references(
    reports_dir: Path,
    symbol: str,
    interval: str,
    *,
    keep: int = 5,
) -> None:
    candidates = sorted(
        reports_dir.glob(
            f"{symbol}_{interval}_monitoring_reference_*.json"
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates[max(int(keep), 1) :]:
        path.unlink(missing_ok=True)


def _append_compacted_jsonl(
    path: Path,
    payload: dict[str, object],
    *,
    max_entries: int = 500,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if path.exists():
        try:
            existing = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            existing = []
    existing.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    safe_replace_text(path, "\n".join(existing[-max_entries:]) + "\n")


def _apply_trigger_lifecycle(
    monitoring: dict[str, object],
    previous: dict[str, object],
    *,
    symbol: str,
    interval: str,
    model_version: str,
    cfg: TraderConfig,
) -> None:
    retraining = monitoring.get("retraining")
    if not isinstance(retraining, dict):
        return
    previous_retraining = previous.get("retraining")
    if not isinstance(previous_retraining, dict):
        previous_retraining = {}
    raw_triggered = bool(retraining.get("triggered"))
    previous_compatible = (
        previous.get("schema_version") == MONITORING_SCHEMA_VERSION
        and previous.get("algorithm_version") == MONITORING_ALGORITHM_VERSION
    )
    consecutive = (
        int(previous_retraining.get("consecutive_breaches", 0) or 0) + 1
        if raw_triggered
        and previous_compatible
        and bool(previous_retraining.get("raw_triggered"))
        else (1 if raw_triggered else 0)
    )
    minimum = max(int(cfg.monitoring_min_consecutive_breaches), 1)
    now_text = str(monitoring.get("created_beijing") or beijing_now_iso())
    now = datetime.fromisoformat(now_text)
    reasons = tuple(str(item) for item in retraining.get("reasons") or ())
    trigger_digest = hashlib.sha1(
        "|".join(
            [
                symbol,
                interval,
                MONITORING_ALGORITHM_VERSION,
                model_version,
                now_text,
                *reasons,
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    active = bool(raw_triggered and consecutive >= minimum)
    retraining.update(
        {
            "raw_triggered": raw_triggered,
            "consecutive_breaches": consecutive,
            "required_consecutive_breaches": minimum,
            "active": active,
            "trigger_id": trigger_digest if active else None,
            "valid_until_beijing": (
                now + timedelta(hours=max(int(cfg.monitoring_trigger_max_age_hours), 1))
            ).isoformat(timespec="seconds"),
            "acknowledged": False,
            "acknowledged_beijing": None,
            "acknowledged_by_report": None,
        }
    )
    if (
        previous_compatible
        and bool(previous_retraining.get("active"))
        and previous.get("model_version") != model_version
    ):
        monitoring["previous_trigger_acknowledged_by_model_version"] = {
            "trigger_id": previous_retraining.get("trigger_id"),
            "model_version": model_version,
            "acknowledged_beijing": now_text,
        }


def train_live_symbol(
    symbol: str,
    interval: str,
    cfg: TraderConfig,
    *,
    include_realtime: bool = True,
    model_suffix: str = "live",
    max_model_trials: int = 1,
    time_budget_minutes: float = 3.0,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_training_rows: int = 3000,
) -> dict[str, object]:
    symbol = symbol.upper()
    raw = load_symbol_interval(cfg.data_dir, symbol, interval, include_realtime=include_realtime)
    data_validation = dict(raw.attrs.get("data_validation", {}))
    market_context = dict(raw.attrs.get("market_context", {}))
    raw_rows_before_window = len(raw)
    if max_training_rows and max_training_rows > 0 and len(raw) > max_training_rows:
        raw = raw.tail(max_training_rows).reset_index(drop=True)
    label_horizon = primary_label_horizon(interval, cfg)
    label_min_return = primary_label_min_return(interval, cfg)
    feature_frame = make_features(raw, label_horizon=label_horizon, label_min_return=label_min_return)
    prediction_frame = make_features(raw, label_horizon=label_horizon, label_min_return=label_min_return, drop_future_na=False)
    split_purge_rows = max(int(label_horizon), int(max(MULTI_HORIZON_STEPS)))
    train_df, valid_df, test_df = time_split(
        feature_frame,
        cfg.train_fraction,
        cfg.validation_fraction,
        purge_rows=split_purge_rows,
    )
    _, _, cols = feature_matrix(train_df)
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
        initial_balance=10_000.0,
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
        risk_per_trade=cfg.risk_per_trade,
        min_confidence_gap=cfg.min_confidence_gap,
    )
    bundle, model_report, detail = train_accuracy_first_candidates(
        train_df,
        valid_df,
        test_df,
        cols,
        seed=cfg.random_seed,
        max_model_trials=max_model_trials,
        deadline=time.monotonic() + max(float(time_budget_minutes), 0.5) * 60,
        min_trades=8,
        max_drawdown_floor=-0.18,
        min_profit_factor=0.75,
        base_backtest_cfg=base_backtest_cfg,
        label_min_return=label_min_return,
        complexity=complexity,
        rolling_folds=rolling_folds,
        multi_horizon_specs=multi_horizon_specs_for_interval(interval),
        objective="return",
        validation_split_purge_rows=split_purge_rows,
    )

    latest_row = prediction_frame.tail(1)
    latest_x, _ = feature_only_matrix(latest_row, cols)
    latest_prob = float(bundle.predict_up_probability(latest_x)[0])
    latest_expected_return_values = bundle.predict_expected_return(latest_x)
    latest_expected_return = (
        float(latest_expected_return_values[0])
        if len(latest_expected_return_values)
        and np.isfinite(latest_expected_return_values[0])
        else None
    )
    latest_direction_raw = bundle.predict_direction_probabilities(latest_x)
    latest_direction_probabilities = {
        "long": float(latest_direction_raw["long"][0]),
        "short": float(latest_direction_raw["short"][0]),
        "trade": float(latest_direction_raw["trade"][0]),
        "up": float(latest_direction_raw["up"][0]),
    }
    latest_horizon_probabilities = {
        key: float(value[0]) if hasattr(value, "__len__") else float(value)
        for key, value in bundle.predict_horizon_probabilities(latest_x).items()
    }
    latest_alpha_prediction = AlphaPrediction(
        timestamp=str(latest_row["open_datetime"].iloc[-1]),
        symbol=symbol,
        horizon=interval,
        expected_return=latest_expected_return,
        p_up=latest_prob,
        p_down=1.0 - latest_prob,
        volatility_forecast=float(latest_row["volatility_24"].iloc[-1]),
        confidence=abs(latest_prob - 0.5) * 2.0,
        model_version=_alpha_model_version(bundle),
    )
    latest_exchange_available = coerce_exchange_available(
        latest_row["exchange_available"].iloc[-1]
        if "exchange_available" in latest_row.columns
        else True
    )
    latest_exchange_reason = str(
        latest_row["exchange_unavailable_reason"].iloc[-1]
        if "exchange_unavailable_reason" in latest_row.columns
        else ""
    )
    latest_execution_decision = {
        "allow_execution": latest_exchange_available,
        "reason": (
            "exchange_available"
            if latest_exchange_available
            else (latest_exchange_reason or "exchange_unavailable")
        ),
        "gap_before_bars": int(
            latest_row["exchange_gap_before_bars"].iloc[-1]
            if "exchange_gap_before_bars" in latest_row.columns
            else 0
        ),
    }
    latest_liquidity_score = float(
        latest_row["liquidity_quality_score"].iloc[-1]
        if "liquidity_quality_score" in latest_row.columns
        else 1.0
    )
    horizon_matches = summarize_horizon_matches(feature_frame, bundle, cols)
    model_path = cfg.model_dir / f"{symbol}_{interval}_{model_suffix}.pkl"
    bundle.save(model_path)
    model_version = _model_version(
        model_name=bundle.model_name,
        feature_columns=cols,
        model_config=dict(bundle.config),
        latest_training_open_time=int(feature_frame["open_time"].iloc[-1]),
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        test_rows=len(test_df),
    )

    optimized_config = model_report.get("optimized_backtest_config") or asdict(base_backtest_cfg)
    backtest_cfg = BacktestConfig(**optimized_config)
    backtest_result, detail = run_backtest(test_df, bundle, backtest_cfg)
    monitoring_recent_rows = max(int(cfg.monitoring_recent_rows), 40)
    recent_replay_frame = prediction_frame.tail(
        monitoring_recent_rows
    ).reset_index(drop=True)
    _, recent_detail = run_backtest(
        recent_replay_frame,
        bundle,
        backtest_cfg,
    )
    calibration_frame = feature_frame.tail(
        monitoring_recent_rows
    ).reset_index(drop=True)
    calibration_x, calibration_y, _ = feature_matrix(
        calibration_frame,
        feature_columns=cols,
    )
    calibration_prob = bundle.predict_up_probability(calibration_x)
    monitoring_path = cfg.reports_dir / f"{symbol}_{interval}_monitoring.json"
    monitoring_reference_path = (
        cfg.reports_dir / f"{symbol}_{interval}_monitoring_reference.json"
    )
    monitoring_history_path = (
        cfg.reports_dir / f"{symbol}_{interval}_monitoring_history.jsonl"
    )
    previous_monitoring = _read_json(monitoring_path)
    reference_profile = _read_json(monitoring_reference_path)
    if (
        reference_profile.get("schema_version") != MONITORING_SCHEMA_VERSION
        or reference_profile.get("algorithm_version")
        != MONITORING_ALGORITHM_VERSION
        or reference_profile.get("model_version") != model_version
    ):
        baseline_window = asdict(
            rolling_performance(
                detail,
                window=monitoring_recent_rows,
            )
        )
        reference_profile = {
            "schema_version": MONITORING_SCHEMA_VERSION,
            "algorithm_version": MONITORING_ALGORITHM_VERSION,
            "created_beijing": beijing_now_iso(),
            "symbol": symbol,
            "interval": interval,
            "model_name": bundle.model_name,
            "model_version": model_version,
            "feature_columns": cols,
            "features": build_feature_reference(train_df, cols),
            "micro_regime_distribution": (
                micro_regime_distribution(train_df["micro_trend_regime"])
                if "micro_trend_regime" in train_df.columns
                else {}
            ),
            "baseline_equal_window": baseline_window,
            "reference_rows": int(len(train_df)),
            "compression": {
                "psi_bins": 10,
                "ks_quantile_points": 101,
                "raw_training_rows_persisted": False,
            },
            "safety": {
                "live_trading_enabled": False,
                "real_orders_allowed": False,
                "api_keys_used": False,
            },
        }
        safe_replace_text(
            monitoring_reference_path,
            json.dumps(reference_profile, indent=2, ensure_ascii=False),
        )
        immutable_reference_path = (
            cfg.reports_dir
            / f"{symbol}_{interval}_monitoring_reference_{model_version}.json"
        )
        if not immutable_reference_path.exists():
            safe_replace_text(
                immutable_reference_path,
                json.dumps(reference_profile, indent=2, ensure_ascii=False),
            )
        _prune_versioned_references(
            cfg.reports_dir,
            symbol,
            interval,
        )
    baseline_equal_window = dict(
        reference_profile.get("baseline_equal_window") or {}
    )
    recent_equal_window = asdict(
        rolling_performance(
            recent_detail,
            window=monitoring_recent_rows,
        )
    )
    paper_path = cfg.reports_dir / f"{symbol}_{interval}_paper_summary.json"
    paper_payload = _read_json(paper_path)
    paper_metrics = (
        dict(paper_payload.get("metrics") or {})
        if paper_payload.get("model_name") == bundle.model_name
        and isinstance(paper_payload.get("metrics"), dict)
        else None
    )
    if not paper_payload:
        paper_status = "missing"
    elif not isinstance(paper_payload.get("metrics"), dict):
        paper_status = "legacy_metrics_missing"
    elif paper_payload.get("model_name") != bundle.model_name:
        paper_status = "model_mismatch"
    else:
        paper_status = "comparable"
    monitoring = build_monitoring_snapshot(
        reference_frame=train_df,
        current_frame=recent_replay_frame,
        feature_columns=cols,
        calibration_targets=calibration_y,
        calibration_probabilities=calibration_prob,
        recent_detail=recent_detail,
        baseline_backtest=baseline_equal_window,
        recent_backtest=recent_equal_window,
        cfg=cfg,
        reference_profile=reference_profile,
        frozen_backtest=asdict(backtest_result),
        paper_metrics=paper_metrics,
        paper_status=paper_status,
    )
    monitoring.update(
        {
            "created_beijing": beijing_now_iso(),
            "symbol": symbol,
            "interval": interval,
            "model_name": bundle.model_name,
            "model_version": model_version,
            "reference_path": str(monitoring_reference_path),
            "history_path": str(monitoring_history_path),
            "reference_rows": int(len(train_df)),
            "current_rows": int(len(recent_replay_frame)),
            "calibration_rows": int(len(calibration_frame)),
            "safety": {
                "live_trading_enabled": False,
                "real_orders_allowed": False,
                "api_keys_used": False,
            },
        }
    )
    _apply_trigger_lifecycle(
        monitoring,
        previous_monitoring,
        symbol=symbol,
        interval=interval,
        model_version=model_version,
        cfg=cfg,
    )
    latest_risk_decision = None
    latest_risk_contract = None
    risk_reason_counts: dict[str, int] = {}
    latest_regime = latest_regime_state(recent_detail)
    if not recent_detail.empty and "risk_reason" in recent_detail.columns:
        latest_risk = recent_detail.iloc[-1]
        latest_risk_contract = RiskDecision(
            allow_trade=bool(latest_risk["risk_allow_trade"]),
            risk_level=RiskLevel(str(latest_risk["risk_level"])),
            max_position_size=float(latest_risk["risk_max_position_size"]),
            reason=str(latest_risk["risk_reason"]),
        )
        latest_risk_decision = latest_risk_contract.to_dict()
        risk_reason_counts = {
            str(key): int(value)
            for key, value in recent_detail["risk_reason"]
            .value_counts()
            .to_dict()
            .items()
        }
    backtest_risk_reason_counts = (
        {
            str(key): int(value)
            for key, value in detail["risk_reason"].value_counts().to_dict().items()
        }
        if not detail.empty and "risk_reason" in detail.columns
        else {}
    )
    latest_meta_signal = None
    latest_strategy_decision = None
    if latest_regime is not None:
        latest_feature = latest_row.iloc[-1]
        def latest_numeric(name: str, default: float = 0.0) -> float:
            value = pd.to_numeric(
                pd.Series([latest_feature.get(name, default)]),
                errors="coerce",
            ).iloc[0]
            return float(value) if np.isfinite(value) else float(default)

        funding_long, funding_short = funding_alpha_scores(
            latest_numeric("funding_rate_8h")
        )
        transaction_cost_estimate_raw = float(
            recent_detail.iloc[-1].get("required_edge", 0.0)
            if not recent_detail.empty
            else 0.0
        )
        transaction_cost_estimate = (
            transaction_cost_estimate_raw
            if np.isfinite(transaction_cost_estimate_raw)
            else 0.0
        )
        latest_meta_signal = WeightedMetaSignal().fuse(
            MetaSignalInputs(
                alpha=latest_alpha_prediction,
                regime=latest_regime,
                risk=latest_risk_contract,
                transaction_cost_estimate=transaction_cost_estimate,
                trend_long=latest_numeric("trend_quality_long", 0.5),
                trend_short=latest_numeric("trend_quality_short", 0.5),
                mean_reversion_long=latest_numeric(
                    "range_quality_long",
                    0.5,
                ),
                mean_reversion_short=latest_numeric(
                    "range_quality_short",
                    0.5,
                ),
                funding_long=funding_long,
                funding_short=funding_short,
            )
        )
        latest_strategy_decision = StrategyOrchestrator().decide(
            StrategyContext(
                meta_signal=latest_meta_signal,
                regime=latest_regime,
                current_position=int(
                    recent_detail.iloc[-1].get("executed_position", 0)
                    if not recent_detail.empty
                    else 0
                ),
                risk_state=latest_risk_contract,
                transaction_cost_estimate=transaction_cost_estimate,
                max_exposure=float(
                    latest_risk_contract.max_position_size
                    if latest_risk_contract is not None
                    else 0.0
                ),
                stop_loss=float(
                    recent_detail.iloc[-1].get("stop_loss_pct", 0.0)
                    if not recent_detail.empty
                    else 0.0
                )
                or None,
                take_profit=float(
                    recent_detail.iloc[-1].get("take_profit_pct", 0.0)
                    if not recent_detail.empty
                    else 0.0
                )
                or None,
            )
        )

    report = {
        "created_utc": beijing_now_iso(),
        "created_beijing": beijing_now_iso(),
        "symbol": symbol,
        "interval": interval,
        "rows": len(feature_frame),
        "raw_rows": len(raw),
        "raw_rows_before_window": raw_rows_before_window,
        "data_validation": data_validation,
        "market_context": market_context,
        "max_training_rows": max_training_rows,
        "training_window_applied": raw_rows_before_window != len(raw),
        "primary_label_horizon": label_horizon,
        "primary_label_min_return": label_min_return,
        "configured_label_min_return": cfg.label_min_return,
        "split_purge_rows": split_purge_rows,
        "split_purge_policy": "drop_tail_rows_from_train_and_validation_to_prevent_future_label_overlap",
        "history_usage": {
            "mode": "all_available_historical_plus_closed_realtime",
            "raw_rows": len(raw),
            "raw_rows_before_window": raw_rows_before_window,
            "max_training_rows": max_training_rows,
            "feature_rows": len(feature_frame),
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
            "split_purge_rows": split_purge_rows,
            "include_realtime": include_realtime,
            "sample_growth_policy": "expand_history_then_compact_redundant_realtime",
        },
        "training_strategy": {
            "complexity": complexity,
            "max_model_trials": max_model_trials,
            "time_budget_minutes": time_budget_minutes,
            "neural_network_priority": "mlp_and_tabular_neural_models" if complexity in {"deep", "blackbox"} else "lightweight_tabular_baseline",
            "transformer_policy": "blackbox_only",
            "rolling_folds": rolling_folds,
            "objective": "return",
            "small_account_strategy": "enabled",
            "position_sizing": "confidence_atr_liquidity_scaled",
        },
        "model_name": bundle.model_name,
        "model_version": model_version,
        "model_path": str(model_path),
        "latest_open_time": int(latest_row["open_time"].iloc[-1]),
        "latest_datetime": str(latest_row["open_datetime"].iloc[-1]),
        "latest_close": float(latest_row["close"].iloc[-1]),
        "latest_up_probability": latest_prob,
        "latest_alpha_prediction": latest_alpha_prediction.to_dict(),
        "latest_execution_decision": latest_execution_decision,
        "latest_liquidity_score": latest_liquidity_score,
        "latest_risk_decision": latest_risk_decision,
        "latest_risk_source": "recent_closed_bar_replay",
        "latest_regime_state": (
            latest_regime.to_dict() if latest_regime is not None else None
        ),
        "latest_meta_signal": (
            latest_meta_signal.to_dict()
            if latest_meta_signal is not None
            else None
        ),
        "latest_strategy_decision": (
            asdict(latest_strategy_decision)
            if latest_strategy_decision is not None
            else None
        ),
        "latest_decision_status": "planning_only",
        "risk_reason_counts": risk_reason_counts,
        "backtest_risk_reason_counts": backtest_risk_reason_counts,
        "latest_direction_probabilities": latest_direction_probabilities,
        "latest_horizon_probabilities": latest_horizon_probabilities,
        "recent_horizon_matches": horizon_matches,
        "metrics": bundle.metrics,
        "model_report": model_report,
        "optimized_backtest_config": optimized_config,
        "backtest": asdict(backtest_result),
        "monitoring": monitoring,
    }
    report_path = cfg.reports_dir / f"{symbol}_{interval}_{model_suffix}_train_metrics.json"
    detail_path = cfg.reports_dir / f"{symbol}_{interval}_{model_suffix}_backtest.csv"
    report["report_path"] = str(report_path)
    report["detail_path"] = str(detail_path)
    report["monitoring_path"] = str(monitoring_path)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    safe_replace_text(
        report_path,
        json.dumps(report, indent=2, ensure_ascii=False),
    )
    detail.to_csv(detail_path, index=False)
    safe_replace_text(
        monitoring_path,
        json.dumps(monitoring, indent=2, ensure_ascii=False),
    )
    _append_compacted_jsonl(
        monitoring_history_path,
        monitoring,
    )
    return report


def multi_horizon_specs_for_interval(interval: str) -> list[dict[str, str]]:
    interval = interval.lower()
    steps_by_interval = {
        "5m": [(1, 5), (2, 10), (3, 15), (6, 30), (12, 60)],
        "15m": [(1, 15), (2, 30), (4, 60), (16, 240)],
        "1h": [(1, 60), (3, 180), (4, 240), (6, 360), (12, 720)],
    }
    specs: list[dict[str, str]] = []
    for steps, minutes in steps_by_interval.get(interval, []):
        specs.append(
            {
                "key": f"next_{minutes}m",
                "label": f"Next {minutes} minutes",
                "minutes": str(minutes),
                "steps": str(steps),
                "target_col": f"target_h{steps}",
                "return_col": f"future_return_h{steps}",
            }
        )
    return specs


def summarize_horizon_matches(
    frame: pd.DataFrame,
    bundle,
    feature_columns: list[str],
    *,
    tail: int = 120,
) -> dict[str, object]:
    if not getattr(bundle, "auxiliary_metadata", None):
        return {"enabled": False, "items": {}}
    source = frame.tail(tail).reset_index(drop=True)
    if source.empty:
        return {"enabled": False, "items": {}}
    x, _ = feature_only_matrix(source, feature_columns)
    probabilities = bundle.predict_horizon_probabilities(x)
    items: dict[str, object] = {}
    for key, meta in bundle.auxiliary_metadata.items():
        return_col = str(meta.get("return_col") or "")
        if key not in probabilities or return_col not in source.columns:
            continue
        prob = probabilities[key]
        returns = pd.to_numeric(source[return_col], errors="coerce").fillna(0.0)
        predicted_up = prob >= 0.5
        actual_up = returns.to_numpy(dtype=float) > 0
        matched = predicted_up == actual_up
        samples = []
        sample_source = source.tail(20).reset_index(drop=True)
        sample_prob = prob[-len(sample_source) :]
        sample_returns = returns.tail(len(sample_source)).reset_index(drop=True)
        for idx, row in sample_source.iterrows():
            pred = "up" if sample_prob[idx] >= 0.5 else "down"
            actual_return = float(sample_returns.iloc[idx])
            actual = "up" if actual_return > 0 else "down"
            samples.append(
                {
                    "open_time": int(row["open_time"]),
                    "open_datetime": str(row.get("open_datetime", "")),
                    "close": float(row["close"]),
                    "up_probability": float(sample_prob[idx]),
                    "predicted_direction": pred,
                    "actual_return": actual_return,
                    "actual_direction": actual,
                    "matched": pred == actual,
                }
            )
        items[key] = {
            "metadata": meta,
            "rows": int(len(source)),
            "direction_match_rate": float(matched.mean()) if len(matched) else 0.0,
            "up_probability_latest_matched_row": float(prob[-1]) if len(prob) else 0.0,
            "actual_return_latest_matched_row": float(returns.iloc[-1]) if len(returns) else 0.0,
            "samples": samples,
        }
    return {"enabled": bool(items), "items": items}


def sync_and_train_live(
    symbols: list[str],
    interval: str,
    cfg: TraderConfig,
    *,
    limit: int = 1500,
    base_url: str = "https://fapi.binance.com",
    model_suffix: str = "live",
    max_model_trials: int = 1,
    time_budget_minutes: float = 3.0,
    complexity: str = "standard",
    rolling_folds: int = 0,
    max_training_rows: int = 3000,
) -> dict[str, object]:
    sync_results = sync_recent_futures_klines(
        symbols=symbols,
        interval=interval,
        data_dir=cfg.data_dir,
        limit=limit,
        base_url=base_url,
    )
    reports = [
        train_live_symbol(
            item.upper(),
            interval,
            cfg,
            model_suffix=model_suffix,
            max_model_trials=max_model_trials,
            time_budget_minutes=time_budget_minutes,
            complexity=complexity,
            rolling_folds=rolling_folds,
            max_training_rows=max_training_rows,
        )
        for item in symbols
    ]
    ranked = sorted(reports, key=lambda item: item["backtest"]["total_return"], reverse=True)
    output = cfg.reports_dir / f"live_training_{interval}_{beijing_stamp()}.json"
    output.write_text(
        json.dumps(
            {
                "created_utc": beijing_now_iso(),
                "created_beijing": beijing_now_iso(),
                "sync": [download_result_payload(item) for item in sync_results],
                "items": reports,
                "ranked": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"sync": sync_results, "items": reports, "ranked": ranked, "output": output}


def download_result_payload(item: DownloadResult) -> dict[str, object]:
    return {
        "symbol": item.symbol,
        "interval": item.interval,
        "files": [str(path) for path in item.files],
        "rows": item.rows,
        "skipped": item.skipped or [],
    }
