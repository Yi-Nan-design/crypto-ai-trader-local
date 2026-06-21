# Crypto AI Trader System Guide

## Safety Scope

The project is an offline research, backtest, paper-simulation, and testnet
candidate system. `config.default.json` must keep
`live_trading_enabled=false`. The pipeline does not need account API keys for
training, market-context downloads, backtests, monitoring, or portfolio paper
simulation.

## Incremental Module Structure

The existing package was kept intact and split by responsibility instead of
being rewritten into a second package:

| Layer | Main modules |
| --- | --- |
| Data | `binance_data.py`, `data_maintenance.py` |
| Validation | `data_validation.py` |
| Features | `features.py`, `feature_catalog.py` |
| Labels | `labeling.py` |
| Regime | `regime.py`, `statistical_regime.py`, `regime_models.py` |
| Alpha | `alpha_models.py`, `models.py`, `model_selection.py` |
| Meta signal | `meta_signal.py` |
| Strategy | `strategy.py`, `strategy_calibration.py` |
| Portfolio | `portfolio.py`, `portfolio_paper.py` |
| Risk | `risk.py`, `liquidation.py`, `exchange_availability.py` |
| Execution | `execution_algorithms.py`, `cost_model.py`, `exchange_rules.py`, `liquidity_execution.py` |
| Evaluation | `backtest.py`, `performance_report.py`, `strategy_validation.py` |
| Monitoring | `monitoring.py`, `scheduled_optimizer.py`, `simulation_memory.py` |
| Local runtime | `runner.py`, `dashboard_server.py`, `desktop_tasks.py` |

## Key Contracts

- `AlphaPrediction`: forecast only; it cannot place an order.
- `RegimeState`: causal market state, confidence, liquidity/volatility state,
  and risk-off flag.
- `MetaSignal`: weighted fusion of Alpha, trend, mean-reversion, funding,
  cross-sectional, regime, cost, and risk context.
- `StrategyDecision`: requested direction, exposure, exits, holding period,
  and reason code.
- `RiskDecision`: final allow/veto decision and maximum position size.
- `PortfolioAssetInput` / `PortfolioDecision`: fail-closed cross-symbol inputs
  and constrained planning weights.
- `ExecutionPlan`: simulated limit-first, TWAP, or VWAP child-order plan. It
  always records `live_orders_allowed=false`.

## Training Flow

1. Load cached historical and closed realtime K-lines.
2. Normalize and validate duplicates, gaps, OHLC consistency, volume, outliers,
   and exchange availability.
3. Build causal feature groups.
4. Build forward labels separately from model features.
5. Apply chronological train/validation/test splits with purge rows.
6. Rank Alpha models on validation data.
7. Freeze model and strategy calibration before evaluating the test split.
8. Save model, backtest detail, monitoring snapshot, and report.

Run baseline training:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli train --symbols BTCUSDT ETHUSDT --interval 1h
```

Run bounded AI optimization:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli model-optimize --symbols ETHUSDT BNBUSDT --intervals 5m 15m --time-budget-minutes 15 --max-model-trials 12 --include-realtime
```

Run walk-forward strategy validation:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli strategy-validate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --intervals 1h 5m --include-realtime --folds 3
```

## Backtest Flow

The vectorized backtest converts probabilities into strategy requests, applies
regime and risk gates, sizes exposure, simulates latency and fills, applies
exchange filters, funding, liquidation, fees, and slippage, then evaluates the
executed path. Trade quality metrics use executed notional turnover and actual
filled exposure rather than rejected signals.

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli backtest --symbol ETHUSDT --interval 1h --leverage 1
```

## Risk Flow

Risk is applied after strategy interpretation and may veto any request:

1. Regime crash/liquidity risk-off veto.
2. Drawdown cooldown and portfolio daily-loss circuit breaker.
3. Funding crowding and liquidity guards.
4. EWMA daily volatility scaling.
5. Per-trade risk budget, hard leverage cap, and notional cap.
6. Portfolio single-coin, sector, correlation-cluster, total leverage,
   volatility-target, and Expected Shortfall limits.
7. Exchange availability and liquidation override.

Missing, stale, or malformed portfolio risk inputs fail closed to zero weight.

## Paper Flow

Single-symbol paper replay is event-driven through `PaperBroker`. It supports
funding, exchange filters, partial fills, liquidity capacity, liquidation,
exchange downtime, and configured bar latency.

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli paper --symbol ETHUSDT --interval 1h --leverage 1
```

The local multi-symbol runner also maintains a persistent close-to-close
portfolio paper ledger:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_runner.ps1"
```

## Monitoring And Retraining

Each live-training cycle records PSI, KS, calibration error, prediction
confidence, rolling Sharpe, rolling drawdown, recent replay deviation, paper
deviation, and regime-distribution shift. A trigger becomes active only after
the configured number of consecutive breaches. `scheduled-optimize` gives
active triggers priority and acknowledges them only after a target completes.

## Extension Boundaries

- Train the LightGBM ranker only after building aligned multi-symbol groups.
- HMM, GMM, and LightGBM regime classifiers must use walk-forward fitting.
- Open-interest and cross-sectional features activate only when source data is
  present.
- Learned slippage, TCN/TFT, DeepLOB, and execution RL remain optional research
  adapters; none can bypass risk or enable live trading.
