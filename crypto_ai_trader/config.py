from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraderConfig:
    market: str = "futures_um"
    interval: str = "1h"
    realtime_interval: str = "5m"
    realtime_limit: int = 1500
    realtime_base_url: str = "https://fapi.binance.com"
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
    data_base_url: str = ""
    data_base_urls: tuple[str, ...] = ()
    https_proxy: str = ""
    auto_detect_proxy: bool = True
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    fee_rate: float = 0.00045
    maker_fee_rate: float = 0.0002
    maker_fill_fraction: float = 0.0
    slippage_rate: float = 0.0002
    partial_fill_ratio: float = 1.0
    execution_latency_bars: int = 0
    min_order_notional_fraction: float = 0.0
    exchange_min_notional_usdt: float = 0.0
    exchange_min_quantity: float = 0.0
    exchange_max_quantity: float = 0.0
    exchange_quantity_step: float = 0.0
    exchange_price_tick_size: float = 0.0
    exchange_downtime_guard_enabled: bool = True
    exchange_gap_recovery_bars: int = 1
    liquidity_execution_enabled: bool = True
    max_bar_participation_rate: float = 0.01
    liquidity_lookback_bars: int = 48
    slippage_impact_coefficient: float = 1.0
    max_dynamic_slippage_rate: float = 0.02
    liquidation_guard_enabled: bool = True
    maintenance_margin_rate: float = 0.005
    liquidation_buffer: float = 0.01
    liquidation_fee_rate: float = 0.005
    max_leverage: int = 3
    default_leverage: int = 1
    margin_type: str = "ISOLATED"
    risk_per_trade: float = 0.005
    ewma_volatility_enabled: bool = True
    ewma_volatility_span: int = 48
    ewma_daily_volatility_target: float = 0.03
    funding_crowding_guard_enabled: bool = True
    funding_crowding_max_rate: float = 0.0005
    regime_risk_guard_enabled: bool = True
    regime_detection_method: str = "rule_based"
    regime_statistical_clusters: int = 4
    regime_statistical_min_history: int = 240
    regime_statistical_lookback: int = 720
    regime_statistical_refit_interval: int = 24
    regime_statistical_random_seed: int = 42
    max_daily_loss: float = 0.03
    long_threshold: float = 0.57
    short_threshold: float = 0.43
    min_confidence_gap: float = 0.07
    label_horizon: int = 3
    label_min_return: float = 0.001
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    random_seed: int = 42
    monitoring_recent_rows: int = 240
    monitoring_psi_threshold: float = 0.25
    monitoring_ks_threshold: float = 0.20
    monitoring_min_confidence: float = 0.12
    monitoring_max_ece: float = 0.15
    monitoring_min_rolling_sharpe: float = 0.0
    monitoring_max_drawdown: float = 0.08
    monitoring_return_deviation: float = 0.03
    monitoring_regime_shift_threshold: float = 0.35
    monitoring_trigger_max_age_hours: int = 6
    monitoring_min_consecutive_breaches: int = 2
    portfolio_target_gross_exposure: float = 0.45
    portfolio_max_total_leverage: float = 1.0
    portfolio_max_single_weight: float = 0.25
    portfolio_max_sector_exposure: float = 0.35
    portfolio_max_cluster_exposure: float = 0.35
    portfolio_min_liquidity_score: float = 0.20
    portfolio_max_drawdown: float = 0.10
    portfolio_volatility_floor: float = 0.001
    portfolio_volatility_target_enabled: bool = True
    portfolio_target_daily_volatility: float = 0.02
    portfolio_min_volatility_observations: int = 100
    portfolio_correlation_threshold: float = 0.75
    portfolio_correlation_lookback: int = 500
    portfolio_cvar_confidence: float = 0.95
    portfolio_max_cvar_loss: float = 0.01
    portfolio_min_cvar_observations: int = 100
    portfolio_paper_initial_balance: float = 10_000.0
    portfolio_paper_max_history: int = 2_000
    portfolio_require_complete_inputs: bool = True
    shadow_learning_enabled: bool = True
    shadow_min_signal_count: int = 8
    shadow_min_profit_factor: float = 1.20
    shadow_max_position_fraction: float = 0.05
    shadow_leverage: int = 1
    shadow_target_gross_exposure: float = 0.20
    runner_max_model_trials: int = 4
    runner_time_budget_minutes: float = 6.0
    runner_rolling_folds: int = 1
    runner_max_training_rows: int = 8_000
    portfolio_symbol_sectors: dict[str, str] = field(
        default_factory=lambda: {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "layer1",
            "SOLUSDT": "layer1",
            "BNBUSDT": "exchange",
        }
    )
    live_trading_enabled: bool = False

    @classmethod
    def from_file(cls, path: str | Path = "config.default.json") -> "TraderConfig":
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        values = {}
        for field in cls.__dataclass_fields__.values():
            if field.name not in raw:
                continue
            value = raw[field.name]
            if field.name in {"data_dir", "model_dir", "reports_dir"}:
                value = Path(value)
            elif field.name == "symbols":
                value = tuple(value)
            elif field.name == "data_base_urls":
                if isinstance(value, str):
                    value = tuple(item.strip() for item in value.replace(",", ";").split(";") if item.strip())
                else:
                    value = tuple(str(item).strip() for item in (value or ()) if str(item or "").strip())
            elif field.name == "portfolio_symbol_sectors":
                value = {
                    str(symbol).strip().upper(): str(sector).strip().lower()
                    for symbol, sector in dict(value or {}).items()
                    if str(symbol).strip() and str(sector).strip()
                }
            values[field.name] = value
        return cls(**values)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.model_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> TraderConfig:
    return TraderConfig.from_file(path or "config.default.json")
