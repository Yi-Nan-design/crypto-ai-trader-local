# Quant Refactor Requirements Traceability

This matrix maps `量化模型优化.txt` to current code and tests. "Reserved"
means a typed optional interface exists and missing dependencies do not block
the core system.

## Layer Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| Data collection/storage | Implemented | `binance_data.py`, `data_maintenance.py` |
| Rule validation, MAD robust z-score | Implemented | `data_validation.py`, `tests/test_data_validation.py` |
| Isolation Forest | Optional implemented | `isolation_forest_anomalies()` |
| Cross-exchange deviation | Interface implemented | `cross_exchange_price_deviation()` |
| Feature families | Implemented | `feature_catalog.py`, `features.py` |
| Funding/open-interest/cross-sectional feature boundaries | Implemented/optional | `feature_catalog.py`, `tests/test_feature_catalog.py` |
| Future return and risk-adjusted labels | Implemented | `labeling.py` |
| Triple barrier, meta, rank, MAE/MFE labels | Implemented interfaces | `labeling.py`, `tests/test_labeling.py` |
| Rule crash/liquidity/volatility regime | Implemented | `regime.py` |
| Walk-forward KMeans | Implemented | `statistical_regime.py` |
| HMM/GMM/LightGBM regime model | Reserved optional interfaces | `regime_models.py` |
| LightGBM classifier/regressor/ranker Alpha | Implemented | `alpha_models.py` |
| CatBoost/XGBoost classifier and regressor boundaries | Implemented/optional | `models.py`, `alpha_models.py` |
| Transformer candidate | Optional implemented | `TorchSequenceClassifierAdapter` |
| Weighted and regime-gated Meta Signal | Implemented | `meta_signal.py` |
| Base/trend/mean-reversion/funding/cross-sectional strategies | Implemented | `strategy.py` |
| Strategy orchestrator and reason codes | Implemented | `strategy.py`, `tests/test_meta_strategy.py` |
| Inverse-volatility target weights | Implemented | `portfolio.py` |
| Single, sector, cluster, leverage, drawdown, liquidity caps | Implemented | `portfolio.py`, `tests/test_portfolio.py` |
| Portfolio daily volatility target | Implemented | `apply_portfolio_volatility_target()` |
| EWMA volatility, ES/CVaR, leverage, liquidity, funding guards | Implemented | `risk.py`, `portfolio.py` |
| Circuit breaker and exchange failure guard | Implemented | `portfolio_paper.py`, `exchange_availability.py`, `binance_data.py` |
| Limit-first, TWAP, VWAP | Simulated planning interfaces implemented | `execution_algorithms.py` |
| Simple and optional learned slippage | Implemented/reserved | `liquidity_execution.py`, `execution_algorithms.py` |
| Vectorized backtest | Implemented | `backtest.py` |
| Event-driven paper backtest | Implemented | `paper.py` |
| Walk-forward validation | Implemented | `strategy_validation.py` |
| Fees, funding, latency, fills, minimum size, leverage, liquidation, downtime | Implemented | `cost_model.py`, `paper.py`, `liquidation.py`, `exchange_rules.py` |
| Required performance metrics and regime/calendar/symbol views | Implemented | `performance_report.py`, `regime.py` |
| PSI, KS, confidence, calibration, rolling risk and deviation monitors | Implemented | `monitoring.py` |
| Persistent retraining trigger lifecycle | Implemented | `live_training.py`, `scheduled_optimizer.py` |

## Engineering Rules

| Rule | Evidence |
| --- | --- |
| Typed contracts/dataclasses | `contracts.py`, model/portfolio/risk configs |
| Config outside strategy code | `config.default.json`, `config/strategy_calibration_profiles.toml` |
| Secrets not stored | Account credentials are read only from environment by the dormant dry-run exchange adapter |
| Chronological purged splits | `features.time_split`, `strategy_calibration.py`, `strategy_validation.py` |
| No label columns in model matrix | Explicit `FEATURE_COLUMNS` and labeling tests |
| Explicit fees/slippage/funding | `cost_model.py` and backtest detail columns |
| Traceable decisions | strategy, risk, execution, and portfolio reason codes |
| Optional dependencies | sklearn/LightGBM/XGBoost/CatBoost/PyTorch/HMM registries and skip reports |
| Unit and integration validation | `tests/`, CLI `smoke`, CLI `doctor` |
| Live trading locked | `config.default.json`, report safety blocks, desktop task whitelist |

## Deliberately Not Activated

- Real Binance orders and account API keys.
- Reinforcement learning for trade direction.
- Per-symbol LightGBM ranker training without aligned cross-sectional groups.
- HMM/GMM classification without causal walk-forward calibration.
- Learned slippage without order-book/spread training labels.

These are extension boundaries, not silent fallbacks.
