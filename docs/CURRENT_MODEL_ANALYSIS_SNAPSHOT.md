# Current Model Analysis Snapshot

Generated for external review on 2026-06-25, Beijing time.

This repository contains the model, training, backtest, risk, portfolio, and desktop control source code. Local runtime artifacts are intentionally excluded from Git:

- Excluded: `data/`, `models/`, `reports/`, `state/`, `.venv/`, logs, and credential files.
- Not uploaded: trained `.pkl` files, raw Binance kline files, local report CSVs, and API credentials.
- Safety status from `config.default.json`: `live_trading_enabled=false`, `margin_type=ISOLATED`, `default_leverage=10`, `max_leverage=10`, `shadow_leverage=10`.

## Source Areas To Review

Primary model and training code:

- `crypto_ai_trader/model_optimization.py`
- `crypto_ai_trader/model_selection.py`
- `crypto_ai_trader/models.py`
- `crypto_ai_trader/live_training.py`
- `crypto_ai_trader/runner.py`

Feature and label code:

- `crypto_ai_trader/features.py`
- `crypto_ai_trader/feature_catalog.py`
- `crypto_ai_trader/labeling.py`

Strategy, risk, and backtest code:

- `crypto_ai_trader/strategy.py`
- `crypto_ai_trader/strategy_calibration.py`
- `crypto_ai_trader/shadow_learning.py`
- `crypto_ai_trader/backtest.py`
- `crypto_ai_trader/cost_model.py`
- `crypto_ai_trader/risk.py`
- `crypto_ai_trader/portfolio.py`
- `crypto_ai_trader/portfolio_paper.py`

Relevant docs:

- `docs/MODEL_SELECTION_CALIBRATION.md`
- `docs/PERFORMANCE_EVALUATION.md`
- `docs/RISK_LAYER.md`
- `docs/PORTFOLIO_LAYER.md`
- `docs/LOCAL_RUNNER.md`
- `docs/small_account_crypto_strategy.md`

## Latest Local Optimization Snapshot

The latest scheduled run completed at `2026-06-25T13:17:02+08:00`.

Run settings:

- Mode: `scheduled_optimization`
- Objective: `return`
- Complexity: `standard`
- Time budget: `12` minutes
- Max model trials: `1`
- Max training rows: `12000`
- Include realtime closed K-lines: `true`
- Selected target: `BNBUSDT 1h`

### BNBUSDT 1h

Report timestamp: `2026-06-25T13:16:23+08:00`

Model:

- Model name: `logistic_regression_numpy_lr003_l2_1e3_cost_edge_balanced`
- Feature version: `v12_grouped_context_features`
- Raw rows: `12000`
- Rows after processing: `11786`

Classification metrics:

- Validation accuracy: `0.4171`
- Validation balanced accuracy: `0.5000`
- Validation recall: `1.0000`
- Validation specificity: `0.0000`
- Validation AUC: `0.5935`
- Validation log loss: `1.2368`
- Test accuracy: `0.3959`
- Test balanced accuracy: `0.5000`
- Test recall: `1.0000`
- Test specificity: `0.0000`
- Test AUC: `0.6548`
- Test log loss: `1.2336`

Threshold and strategy result:

- Best risk profile: `no_trade_recommended`
- Side policy: `none`
- Long threshold: `0.99`
- Short threshold: `0.01`
- Score: `-999.0`
- Test return: `0.0`
- Test trades: `0`
- Test execution events: `0`
- Profit factor: `0.0`
- Leverage: `10`
- Average exposure: `0.0`

CSV backtest summary:

- Rows: `1768`
- Nonzero `position` rows: `0`
- Nonzero `executed_position` rows: `0`
- Sum `strategy_return`: `0.0`
- `trade_side_policy` was `none` on all rows.
- Top decision reasons: `no_alpha_signal` `1763`, `blocked_horizon_confirmation` `3`, `blocked_side_policy` `2`.
- `side_policy_gate_pass=false` on all rows.

Publish gate:

- `model_publish.status`: `candidate_only_rejected`
- Reason: `model_rejected_by_validation_trading_gate`
- Final threshold validation gate: failed
- Raw model selection gate: failed
- Threshold validation reason: `no_validation_threshold_config_passed`
- Volatility regime gate: failed because each regime had zero trades.

### ETHUSDT 1h

Report timestamp: `2026-06-25T12:42:22+08:00`

Model:

- Model name: `logistic_regression_numpy_lr003_l2_1e3_cost_edge_balanced`
- Feature version: `v12_grouped_context_features`
- Raw rows: `12000`
- Rows after processing: `11786`

Classification metrics:

- Validation accuracy: `0.3918`
- Validation balanced accuracy: `0.5000`
- Validation recall: `1.0000`
- Validation specificity: `0.0000`
- Validation AUC: `0.6213`
- Validation log loss: `1.2525`
- Test accuracy: `0.3462`
- Test balanced accuracy: `0.5009`
- Test recall: `1.0000`
- Test specificity: `0.0017`
- Test AUC: `0.6944`
- Test log loss: `1.2855`

Threshold and strategy result:

- Best risk profile: `no_trade_recommended`
- Side policy: `none`
- Long threshold: `0.99`
- Short threshold: `0.01`
- Score: `-999.0`
- Test return: `0.0`
- Test trades: `0`
- Test execution events: `0`
- Profit factor: `0.0`
- Leverage: `10`
- Average exposure: `0.0`

CSV backtest summary:

- Rows: `1768`
- Nonzero `position` rows: `0`
- Nonzero `executed_position` rows: `0`
- Sum `strategy_return`: `0.0`
- `trade_side_policy` was `none` on all rows.
- Top decision reason: `no_alpha_signal` on all `1768` rows.
- `side_policy_gate_pass=false` on all rows.

Publish gate:

- `model_publish.status`: `candidate_only_rejected`
- Reason: `threshold_search_timed_out_candidate_only`
- Final threshold validation gate: failed
- Raw model selection gate: failed
- Threshold validation reason: `no_validation_threshold_config_passed`
- Volatility regime gate: failed because each regime had zero trades.

## Latest 5m Runner Snapshot

The latest 5m live-training reports are older than the latest 1h scheduled optimization reports.

- BTCUSDT 5m: `2026-06-22T16:00:27+08:00`, model `ensemble_probability_average_top3`, latest up probability `0.1711`, shadow not eligible, blocker `no_validation_qualified_shadow_side`.
- ETHUSDT 5m: `2026-06-22T16:06:34+08:00`, model `logistic_regression_numpy_lr003_l2_1e3_cost_edge_balanced`, latest up probability `0.6845`, shadow long eligible with model `mlp_numpy_h32_lr002`, threshold `0.54`, latest shadow probability `0.0011`, blocker `shadow_probability_below_threshold`.
- SOLUSDT 5m: `2026-06-22T16:12:38+08:00`, model `sklearn_logistic_regression_c0.25_cost_edge_balanced`, latest up probability `0.4690`, shadow not eligible, blocker `no_validation_qualified_shadow_side`.
- BNBUSDT 5m: `2026-06-22T15:40:54+08:00`, model `logistic_regression_numpy_lr003_l2_1e3_cost_edge_balanced`, latest up probability `0.7434`, shadow short eligible, threshold `0.54`, latest shadow probability `0.4605`, blockers `shadow_probability_below_threshold` and `shadow_strategy_filter_blocked`.

## Main Issues For Review

1. The 1h classifier is directionally biased. Validation and test recall are near `1.0`, while specificity is near `0.0`. This means the model is effectively predicting the positive/up class almost always, so balanced accuracy is near random even when AUC is above `0.5`.

2. The strict publish path is inactive. Threshold optimization falls back to `no_trade_recommended`, sets side policy to `none`, and produces zero positions. Current zero return is inactivity, not strategy quality.

3. Low-threshold trading and strict publish gates are not yet aligned. The project has a shadow low-threshold path, but the scheduled 1h optimization path still rejects all candidates before producing trades.

4. The hourly scheduled run uses only `max_model_trials=1`, so current snapshots mostly test the Numpy logistic baseline. This is not enough to evaluate LightGBM, XGBoost, CatBoost, sklearn MLP, tree ensembles, or probability ensembles.

5. The 5m runner reports are stale relative to the latest scheduled 1h optimization. Live-training refresh and scheduled optimization are not yet producing a unified current model view.

6. The strategy gate rejects all rows through `side_policy_gate_pass=false`. The next useful debug target is the transition from model probabilities to side policy, not leverage. Leverage is already `10`, but there is no exposure because the strategy emits no trade.

7. The report-level model evidence and execution evidence should be separated more clearly. AUC above `0.6` may indicate some ranking signal, but zero executed trades means there is no validated trading edge.

## Suggested Review Questions

Ask ChatGPT to inspect this repository and answer:

1. Why does the current 1h model collapse into high recall and near-zero specificity?
2. Which part of the threshold/side-policy/publish gate causes all rows to become `no_alpha_signal` or `side_policy=none`?
3. How should low-threshold long/short signals be integrated into scheduled optimization without leaking test data?
4. Which model families and hyperparameter ranges should be prioritized for small-account crypto futures after this failure mode?
5. How should the 5m live-training path and 1h scheduled optimization path be reconciled so reports stay current and comparable?
6. What code paths can be removed or simplified if they only produce inactive `no_trade_recommended` candidates?

## Safety Boundary

This snapshot is for offline analysis only. It must not be interpreted as permission to enable real trading, add API keys, or place orders.
