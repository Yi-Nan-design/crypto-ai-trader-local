# Small Account Crypto Strategy Layer

This project stays in training, backtest, paper simulation, and reporting mode. `live_trading_enabled` must remain `false`.

## Strategy Ideas Used

- Momentum and trend filter: EMA gaps, MACD, efficiency ratio, breakout distance, and micro trend regime.
- Mean reversion and grid/range filter: Bollinger z-score, range position, range compression, VWAP gap, and grid-style reversion pressure.
- Liquidity and cost filter: quote volume z-score, volume ratio, taker pressure, and liquidity shock.
- Volatility filter: ATR, volatility ratios, and ATR regime.
- Small-account futures risk: isolated-style assumptions, low leverage, confidence-scaled position size, ATR/stop-loss risk cap, fee-drag penalty, and a cost-edge filter.

## Implementation

- `crypto_ai_trader/features.py`
  - Feature version is `v4_platform_regime`.
  - Adds trend, breakout, mean-reversion, liquidity, taker-pressure, ATR regime, range compression, grid-style reversion, and trend breakout/breakdown score features.
- `crypto_ai_trader/backtest.py`
  - Adds dynamic position sizing based on confidence, ATR volatility, and liquidity.
  - Blocks trades when ATR does not cover fees, slippage, and a funding-rate buffer.
  - Adds trend/range regime gates so low-confidence shorts and range entries are filtered by market context.
  - Splits executed rows into strategy archetypes such as `range_grid_long`, `range_grid_short`, `trend_breakout_long`, and `trend_breakout_short`.
  - Supports optional ATR-multiple stop-loss/take-profit exits inside candidate backtests while keeping fixed-percent exits as comparable risk profiles.
  - Applies exit clipping symmetrically for long and short positions.
  - Adds report metrics: average exposure, notional turnover, fee drag, effective thresholds, cost-edge pass rate, and regime-gate pass rate.
  - Splits final backtest diagnostics into long/short and strategy-archetype trades, returns, profit factor, win rate, drawdown, and fee drag.
- `crypto_ai_trader/model_optimization.py`
  - Searches small-account risk profiles during validation threshold calibration.
  - Penalizes fee drag and records the selected risk profile in reports.
  - Trains the normal large-move weighting branch, an optional event-balanced branch, and an optional volatility-regime event-balanced branch when actionable moves are sparse, then lets validation ranking choose between them.
  - Recomputes event-sampling weights inside rolling-validation folds from each fold's training prefix only.
  - Keeps short-interval `model-optimize` results as candidates only; 1m/3m/5m/15m models need walk-forward strategy validation before they can replace an active model.
  - Writes `valid_ranked_for_selection` separately from `test_final_audit_ranked`; test metrics are not used for model or threshold selection, but final model publishing still requires test hard gates as a safety guard.
  - Adds volatility-regime publish diagnostics using train-set ATR quantiles, with low/mid/high volatility return, profit factor, trades, drawdown, and long/short trade counts.
  - Trains optional independent long/short auxiliary signal models, but keeps them only when validation backtest passes return and profit-factor gates.
  - Trains an optional tradeability/no-trade auxiliary model on `actionable_label = abs(future_return) >= 2 * primary_label_min_return`.
  - Selects auxiliary signal thresholds on the validation split only, with tradeability scoring weighted toward precision and neutral rejection.
  - Penalizes extreme threshold choices so validation tuning does not overfit tiny differences between similar configurations.
  - Saves new models as candidates first and publishes them to `models/{SYMBOL}_{INTERVAL}_accuracy_ai.pkl` only if final test hard gates pass.
  - Records directional signal diagnostics such as long/short signal precision, large up/down capture, neutral no-trade rate, false trade on neutral rate, and trade signal rate.
- `crypto_ai_trader/strategy_config.py`
  - Centralizes primary label horizon and primary label minimum return.
  - Uses `max(configured_label_min_return, 2 * (fee + slippage) + funding_buffer)` so labels only reward economically tradable moves.
- `crypto_ai_trader/strategy_validation.py`
  - Runs walk-forward validation across historical windows.
  - Uses a purge gap between train/validation/test splits. By default the purge gap is at least the primary label horizon so adjacent labels cannot bleed directly across split boundaries.
  - Reserves a frozen holdout tail when `--holdout-fraction` is greater than zero. This frozen holdout is not used for model or threshold selection; it is only a final promotion gate.
  - Reports cross-fold overlap explicitly with `max_cross_fold_overlap_rows`, `max_cross_fold_test_overlap_rows`, `holdout_used_for_selection=false`, `test_used_for_model_selection=false`, and `test_used_for_threshold_selection=false`.
  - Requires minimum trades per fold and high-slippage stress pass rate before a target becomes eligible for paper observation.
  - Records volatility-regime validation for each fold using train-set ATR quantiles, then applies it as a hard eligibility gate for short intervals (`1m`, `3m`, `5m`, `15m`).
  - Keeps volatility-regime validation as an audit field for `1h` targets so stable higher-timeframe candidates are not rejected only because one volatility bucket has too few trades.
  - Writes `reports/strategy_validation_latest.json`, `reports/{SYMBOL}_{INTERVAL}_strategy_validation.json`, and `state/local_strategy_state.json`.
  - Promotes targets only when out-of-window folds pass return, drawdown, profit-factor, trade-count, and cost-stress gates.
- `crypto_ai_trader/simulation_memory.py`
  - Stores accumulated local training, validation, runner, and paper observations.
  - Uses validation/walk-forward/paper observations for next-target selection; final test reports are retained for audit but excluded from automatic target selection.
- `crypto_ai_trader/live_training.py`
  - Reuses the same small-account risk settings for realtime closed-kline training reports.

## Default Risk Bias

- Keep default leverage at `1x`.
- Prefer isolated-margin interpretation.
- Keep `risk_per_trade` around `0.5%` or lower.
- Reduce trade count when confidence is weak or expected edge may be eaten by fees.
- Promote a model only after validation, test backtest, and paper simulation agree.

## Current Validation Gate

Use this command before treating a strategy update as a local candidate:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli strategy-validate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --intervals 1h 5m --include-realtime --folds 3 --time-budget-minutes 55 --max-model-trials 1 --complexity standard --rolling-folds 0 --min-trades 12 --max-drawdown-limit 0.10 --min-profit-factor 1.0 --holdout-fraction 0.15
```

For broad historical screening without letting the first symbol consume the whole run, use a bounded validation profile. `fast-screen` reads the full available history as the source, but caps each walk-forward and frozen-holdout training window; it is research-only and cannot promote a paper candidate:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli strategy-validate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --intervals 1h --validation-profile fast-screen --time-budget-minutes 10 --per-target-budget-minutes 2.5 --max-training-rows 0 --folds 1 --max-model-trials 1 --wf-train-rows 1200 --wf-valid-rows 300 --wf-test-rows 300 --max-threshold-evals 12 --holdout-fraction 0.15 --min-trades 2 --min-profit-factor 0.5
```

Use `large-sample-light` for a slower but still bounded validation pass. It keeps purge gaps, walk-forward validation, frozen holdout, cost stress, and the no-live-trading boundary while adding row caps and threshold-search limits to reduce overfitting and runtime spikes.

Data coverage updated on 2026-06-04:

- `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, and `BNBUSDT` now have complete Binance USDT-M Futures monthly history for `5m`, `15m`, and `1h` from `2024-01` through `2025-12`.
- Each `5m` symbol has `24` monthly files and `210528` K-line rows. Each `15m` symbol has `24` monthly files and `70176` K-line rows. Each `1h` symbol has `24` monthly files and `17544` K-line rows.
- Hourly scheduled optimization now prioritizes `BNBUSDT/ETHUSDT` on `15m/1h` with a small rolling window (`--max-training-rows 8000`) so it can finish locally. Manual deep runs can use larger windows, and `--max-training-rows 0` means full history.

Latest 2026-06-04 `ETHUSDT 5m` v4 platform-regime model check:

- Command used a bounded 20,000-row window: `model-optimize --symbols ETHUSDT --intervals 5m --time-budget-minutes 8 --max-model-trials 2 --max-training-rows 20000 --complexity standard --rolling-folds 0 --min-trades 12 --max-drawdown-limit 0.10 --min-profit-factor 1.0`.
- Result remains candidate-only and is not published: `test_return=-0.03235`, `profit_factor=0.6277`, `trades=309`, `balanced_accuracy=0.5041`, `large_down_capture=0.75`, `large_up_capture=0`.
- Interpretation: the high raw accuracy is mostly class imbalance/no-trade bias. The model sees some downside events but still loses after costs, so it stays out of paper/testnet promotion.

Latest 2026-06-04 targeted verification after adding purge-gap reporting and frozen-holdout gates:

- `BNBUSDT 1h` remains observe only.
  - Command used: `strategy-validate --symbols BNBUSDT --intervals 1h --folds 1 --time-budget-minutes 12 --max-model-trials 1 --complexity standard --rolling-folds 0 --min-trades 12 --max-drawdown-limit 0.10 --min-profit-factor 1.0 --holdout-fraction 0.10`.
  - Fold result: `median_return=-0.005787`, `mean_profit_factor=0.5694`, `mean_balanced_accuracy=0.5157`, `mean_trades=40`, `high_slippage_pass_rate=0`.
  - Purge/holdout result: `purge_rows=3`, frozen holdout reserved `1743` rows from the final 10% of the dataset, and the holdout was excluded from model/threshold selection.
  - The frozen holdout audit was skipped by the 12-minute verification budget, so `frozen_holdout_gate_passed=false`. A skipped or failed frozen holdout blocks promotion by design.
  - Simulation memory was refreshed afterward. All top targets remain `observe`, and no target is eligible for paper/testnet promotion.

Latest 2026-06-03 validation result after cost-consistent labels, memory leakage guard, event-balanced candidate testing, short-interval publish protection, and cost-stress walk-forward checks:

- No target is currently eligible for continued paper/testnet promotion. All current targets stay in observe/research mode.
- `BNBUSDT 1h`: downgraded from paper candidate to observe only after the full 3-fold recheck.
  - Walk-forward summary: `profitable_fold_rate=0.6667`, `median_return=0.00390`, `mean_return=0.00347`, `mean_profit_factor=2.0280`, `mean_trades=92.7`, `worst_drawdown=-0.00911`.
  - Blocking reason: `high_slippage_pass_rate=0.3333`; one fold turned negative and cost stress did not stay robust enough.
  - Volatility-regime diagnostics are recorded but not hard-gated for `1h` because low/mid buckets can have too few trades.
- `BNBUSDT 15m`: observe only. A bounded 1-fold run was slightly positive (`return=0.000034`, `profit_factor=3.4830`), but it produced only `4` trades, failed high-slippage stress, and failed the short-interval volatility-regime hard gate. The second fold was skipped by the time budget.
- `ETHUSDT 15m`: observe only. The latest 2-fold check had `median_return=-0.04047`, `worst_drawdown=-0.08117`, `mean_profit_factor=1.0987`, and `mean_balanced_accuracy=0.5233`, so it is not stable enough for paper/testnet promotion.
- `ETHUSDT 15m` and `BNBUSDT 15m` now have 24 months of Binance futures `15m` historical data downloaded for future feature/label experiments.
- Simulation memory now gives the latest walk-forward validation downgrade authority: if the newest walk-forward report fails, older positive reports cannot keep a target marked as `continue_paper_simulation`.
- Candidate publishing guard:
  - `published` only when final test return is positive, profit factor meets the configured floor, trades meet the configured minimum, and drawdown stays within the configured limit.
  - Failed candidates are saved to `models/{SYMBOL}_{INTERVAL}_accuracy_ai_candidate.pkl` and written to reports, but do not replace the active model.
- `ETHUSDT 1h`: observe only after the stricter cost label; it lost stability across folds.
- `SOLUSDT 1h`: observe only; older memory looked promising, but stricter current validation and optimization are not strong enough.
- `BTCUSDT 1h`: not promoted.
- `BNBUSDT 5m`: still not promoted. The stricter no-trade label shows why: latest split has `actionable_label_rate` around `1.9%` train, `5.5%` valid, and `17.6%` test, so actionable moves are sparse and unstable.
  - Event-balanced threshold scanning reduced noisy 5m trading from 88 trades to 2 trades and reduced the loss from about `-1.2%` to about `-0.07%`, but this is still not enough because the final test trade count and profit factor fail hard gates.
  - The latest event-balanced candidate (`logistic_regression_numpy_lr003_l2_1e3_event_balanced`) improved some large-down capture, but final test results still failed hard gates: `test_return=-0.01378`, `profit_factor=0.6769`, `trades=109`, `max_drawdown=-0.01347`. It remains `candidate_only` and does not replace the active model.
  - After adding volatility-regime event sampling and ATR exit candidates, the latest 5m check still failed: `test_return=-0.00798`, `profit_factor=0.8347`, `trades=93`. All trades landed in the high-volatility bucket, where PF was still below 1, so the volatility-regime gate failed. The model remains `candidate_only`.
  - Latest walk-forward validation with volatility-regime hard gate also failed: `profitable_fold_rate=0`, `median_return=-0.00580`, `mean_profit_factor=0.8037`, `volatility_regime_pass_rate=0`, `high_slippage_pass_rate=0`. This confirms the 5m target should stay out of paper/testnet promotion.
- All `5m` targets: not promoted after label horizon alignment, cost-consistent labels, and no-trade filtering. They need more closed 5m history or a different event-sampling scheme before paper/testnet consideration.
- Next research step: focus on feature/label work before more leverage or threshold tuning. Priorities are walk-forward-aware event balancing, volatility-regime-specific event models, and higher-timeframe confirmation filters so single-split improvements cannot replace the active model without robust test behavior.
- Runtime-control note: scheduled optimization now uses very light hourly defaults (`8` minutes, `2` model trials, latest `8000` rows, `1` target) and the Windows launcher applies a hard timeout (`time_budget_minutes + 3` minutes) with process-tree termination, so a slow optimization cannot keep consuming CPU indefinitely.
- 2026-06-06 runner/scheduler verification:
  - Hourly scheduled optimization was verified from Windows Task Scheduler. The 15:04 Beijing run completed with exit code `0`, wrote `reports/scheduled_optimization_20260606_151317.json`, and used `BNBUSDT/ETHUSDT` on `15m/1h`, latest `8000` rows, `2` trials, and `1` selected target.
  - The scheduled run selected `BNBUSDT 15m`, but it remains `candidate_only`: `test_return=-0.01240`, `profit_factor=0.74896`, `test_balanced_accuracy=0.53333`, and realized win/loss ratio `0.88113` versus the required `0.90`.
  - The realtime runner was changed to a lightweight continuous path: `standard` complexity, `1` model trial, latest `3000` training rows, and per-stage progress updates. Heavy MLP/tree/blackbox searches stay in hourly or manual `model-optimize`.
  - The verified runner round synced `ETHUSDT` and `BNBUSDT` 5m closed klines, trained both symbols, wrote `reports/runner_live_5m_20260606_155159.json`, and returned to `waiting`. Both 5m runner backtests remained negative, so no symbol is promoted.
  - Latest 5m/10m horizon probabilities are present in runner reports. In the verified round, `BNBUSDT` had `next_5m=0.4083`, `next_10m=0.3846`; `ETHUSDT` had `next_5m=0.4687`, `next_10m=0.2873`.
- 2026-06-06 strategy hardening update:
  - The `next_5m` and `next_10m` auxiliary horizon models are now used by `run_backtest`, not only displayed in reports. They can assist near-threshold entries when both short-horizon models agree with the main probability, and they can block entries when the short-horizon signal contradicts the trade direction.
  - Validation-set trading gates are now hard gates during threshold calibration: return must be positive, profit factor must meet the configured floor, trades must meet the minimum, drawdown must stay within the configured floor, realized win/loss ratio must pass, and side/archetype gates must pass.
  - If no validation threshold configuration passes, the optimizer returns `no_trade_recommended` instead of forcing a losing strategy. The 2026-06-06 16:21 Beijing scheduled run verified this behavior on `BNBUSDT 15m`: `valid_gate_passed_count=0`, selected risk profile `no_trade_recommended`, test trades `0`, and model remains `candidate_only`.
  - This follows the exchange-grid-strategy lesson from OKX, Bybit, and Binance Academy material: grid/range bots are meant for bounded volatile markets, while stop-loss/range boundaries, fees, funding, and liquidation risk must be treated as first-class gates rather than afterthoughts.

## Strategy References

- Binance Academy stop-loss/take-profit and position-sizing material: risk/reward, ATR/volatility exits, and fee/slippage-aware sizing.
- Binance Futures funding-rate material: funding is treated as a stress buffer and not ignored in candidate gates.
- Bybit grid-bot material: grid logic is treated as a range-bound archetype, not as a universal strategy.
- OKX/Bybit bot material: AI/manual grid parameters inspired the local split between range/grid and trend-breakout regimes, but live bot execution remains disabled.
