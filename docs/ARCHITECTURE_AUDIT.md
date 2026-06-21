# Architecture Audit

## Scope

This audit reflects the repository state on 2026-06-21. The system remains
research, backtest, and paper-simulation only. `live_trading_enabled` must stay
`false`.

## Current Structure

The project already separates exchange access, data download, features, models,
backtesting, optimization, runner control, reporting, and the Windows desktop
shell. The main structural issue is responsibility concentration:

- `model_optimization.py` owns model candidates, sample weighting, auxiliary
  models, publishing, and optimization report assembly.
- `strategy_calibration.py` owns validation policy search, cost and side
  gates, risk-profile candidates, and the calibration backtest evaluator.
- `strategy_validation.py` owns walk-forward windows, holdout evaluation, cost
  stress, research-candidate rules, promotion rules, and report assembly.
- `backtest.py` owns signal interpretation, strategy filters, sizing, risk
  gates, execution costs, PnL, and performance metrics.
- `features.py` previously owned both historical feature construction and
  future-label construction.

## Findings

### Model Chain

Live-training reports now expose alpha predictions through the shared
`AlphaPrediction` contract. Predictive and strategy-gated rankings are
separate, while compatibility selection remains in the optimization
orchestrator for backward-compatible publishing behavior.

### Leakage Risk

The system uses chronological splits and purged validation boundaries. Feature
selection also uses an explicit `FEATURE_COLUMNS` allowlist. These controls
reduce direct look-ahead leakage. Future labels are now built in `labeling.py`,
separate from historical feature construction. Unit tests also verify that
future and target columns do not enter the model feature matrix.

### Strategy Coupling

Probability interpretation, direction choice, strategy archetype selection,
and some market filters remain in the backtest module. Shared
`AlphaPrediction`, `StrategyDecision`, and `RiskDecision` contracts now exist.
Position sizing, leverage/notional exposure caps, position-rebalance limits,
and drawdown cooldown logic are implemented in `risk.py`. Backtest, paper
trading, runner reports, and the dashboard now expose separate strategy and
risk reason codes.

### Backtest Realism

The current backtest includes leverage, fees, slippage, funding drag, exposure,
turnover, drawdown controls, maker/taker cost splitting, fixed execution
latency, deterministic partial fills, minimum/maximum order handling, K-line
gap recovery blocking, and isolated-margin liquidation simulation. Missing or
incomplete execution effects include scheduled market-context refresh,
order-book/queue-aware fills, incident-timeline reconstruction, and
exchange-specific maintenance-margin tiers.

### Tests

The repository now contains focused unit tests for labeling, kline validation,
typed contracts, and decision reason codes. CLI smoke and bounded optimization
runs remain as integration checks.

## Incremental Target Structure

Keep the existing `crypto_ai_trader` package and add boundaries gradually:

1. `labeling.py`: forward, edge, event, risk-adjusted, and optional meta labels.
2. `data_validation.py`: kline schema, duplicates, gaps, price/volume anomalies.
3. `contracts.py`: alpha, regime, strategy, portfolio, and risk dataclasses.
4. `regime.py`: rule-based baseline followed by optional statistical models.
5. `strategy.py`: convert alpha and regime inputs into explainable decisions.
6. `risk.py`: final veto and exposure limits.
7. `cost_model.py`: execution-cost assumptions shared by backtest and paper.
8. Split optimization and validation reporting only after these interfaces are
   stable.

## Completed Increments

### Labeling Boundary

Future-label construction was extracted from `features.py` into `labeling.py`.
It preserves existing label columns and adds:

- future realized volatility;
- future risk-adjusted return;
- long and short risk-adjusted targets;
- equivalent multi-horizon risk-adjusted labels;
- optional primary-signal meta labels after transaction-cost hurdles.

The existing default training target is unchanged. This isolates architecture
work from strategy-performance changes. Meta-label generation requires an
explicit causal side column, leaves no-signal and incomplete-forward rows as
nullable labels, and is excluded from `FEATURE_COLUMNS`.

### Data Validation Boundary

`data_validation.py` now normalizes and validates kline data at load boundaries.
It removes structurally invalid rows and reports duplicates, missing intervals,
price outliers, and volume outliers. Outliers are reported rather than silently
deleted, preserving legitimate market shocks for model research.

Training and optimization reports now include the corresponding validation
summary.

### Decision Contracts

`contracts.py` defines shared alpha and strategy decision records.
`strategy.py` defines stable reason codes for no-signal, filter rejection,
risk rejection, and allowed long or short decisions. Backtests record these
codes without changing the existing PnL calculation.

### Risk Boundary

`risk.py` now owns:

- leverage-aware maximum position fraction;
- confidence, volatility, and liquidity-based position sizing;
- position rebalance limits;
- drawdown cooldown decisions;
- row-level and single-decision risk outputs.

The simulation layer now honors `StrategyDecision.target_exposure` and records
`RiskDecision` separately from the strategy reason. The runner and dashboard
transport the latest risk decision additively, so older reports remain readable.

Invalid zero, negative, fractional, or non-finite leverage is rejected when a
`BacktestConfig` is created. A pre-existing cooldown retrigger issue was also
fixed: completing a cooldown resets its drawdown reference instead of
immediately starting another cooldown from the same historical loss.

This is not yet a complete institutional risk engine. Operational daily loss
limits and exchange maintenance-margin tiers remain future work.

Risk budgeting now uses the final row-level ATR stop distance whenever ATR
exits are enabled. ATR values use forward-fill plus a configured fallback, not
a full-sample median, so later rows cannot change earlier position sizes. Fixed
stop configurations retain their existing sizing path.

The risk layer also exposes a causal EWMA estimator built from closed-bar
returns and normalized to a 24-hour horizon. Project configuration enables it
for training, optimization, strategy validation, and live runner backtests.
The default daily target is 3%. High EWMA volatility scales positions toward
the configured minimum. Direct `BacktestConfig()` callers keep the
compatibility default disabled so existing smoke baselines remain comparable.
Reports include the EWMA span, per-bar and daily volatility, target, position
scale, and row-level `risk_position_volatility_measure`.

Directional funding crowding is now a risk veto rather than only a strategy
quality signal. A long is blocked when positive 8-hour funding exceeds the
configured limit; a short is blocked when negative funding breaches the
opposite limit. The non-paying direction remains available. Reports include the
causal funding input, gate result, pass rate, and blocked-row count.

### Execution Cost Boundary

`cost_model.py` now calculates notional turnover, commission, slippage, and
funding costs. It also converts target notional into an executed path with
configurable bar latency, partial fills, minimum-order rejection, and a fixed
maker/taker turnover split. `backtest.py` consumes the executed path and cost
breakdown and remains responsible for PnL orchestration and performance
metrics.

The legacy `trade_cost` and `fee_drag` fields remain available for existing
optimization gates. New reports expose commission and slippage separately, and
the dashboard shows both the cost split and execution quality while still
reading older reports.

### Liquidation Boundary

`liquidation.py` now owns isolated-margin liquidation assumptions and event
detection. Backtest and paper replay share the same leverage, maintenance
margin, safety buffer, and liquidation fee configuration.

The execution path supports forced-flat events. A liquidation resets the
position to zero, adds forced taker turnover and liquidation fees, and requires
a continuing signal to pay re-entry costs. Protective stops closer than the
liquidation boundary are assumed to execute first unless the next bar opens
beyond the liquidation price.

### Funding And Exchange-Rule Boundary

`funding_payment_rate` and `funding_rate_8h` are retained through feature
construction as execution context, not as model features. Cached Binance
history is mapped to the K-line containing each settlement timestamp and
charged once. Positive rates debit longs and credit shorts. A continuous
`funding_rate_8h` input remains available as a prorated compatibility path; if
both inputs are absent, the legacy absolute funding buffer preserves old
behavior.

`exchange_rules.py` applies configurable quantity steps, minimum quantities,
and minimum USDT notionals. Backtest and paper opening orders share this
normalizer and record rejection reasons and rounding loss. Defaults disable the
rules, so existing configurations and smoke results remain compatible.

`sync-market-context` now downloads public Binance funding history and
`exchangeInfo` filters into `data/market_context/<SYMBOL>/`. K-line loading
causally merges the funding cache. Backtest, optimization, live training, and
strategy validation use cached symbol rules whenever explicit configuration
remains zero. Refresh remains an explicit local command, and no account API is
used.

### Liquidity Execution Boundary

`liquidity_execution.py` now owns causal OHLCV-based execution capacity and
dynamic slippage. It uses:

- current closed-bar quote volume for available capacity;
- a trailing quote-volume median built from prior bars for liquidity stress;
- closed-bar high/low range as the volatility/impact proxy;
- actual taker turnover divided by quote volume as market participation.

Backtest and paper replay share the same scalar fill and slippage formulas.
Paper replay additionally records partial closes, funding settlements,
commission, slippage, and liquidity-limited orders in its ledger. The project
configuration enables this path by default, while direct `BacktestConfig()`
keeps compatibility mode for controlled tests and old callers.

### Exchange Availability Boundary

`exchange_availability.py` converts explicit availability and causal missing
K-line recovery markers into an execution permission. It defers opens, closes,
and rebalances while preserving the existing position. Liquidation remains a
risk override. Backtest and paper replay share this rule and expose separate
execution reasons instead of rewriting strategy decisions.

Cached exchange filters now include maximum market quantity and price tick.
Oversized target changes are divided into exchange-valid child orders across
bars in both backtest and paper replay. Price tick remains cached metadata until
a price-bearing limit/stop order contract exists; it is not incorrectly applied
to market-order fills.

### Monitoring And Retraining Boundary

`monitoring.py` now owns feature PSI/KS, calibration, confidence, rolling
performance, recent-replay deviation, regime-distribution shift, and
explainable retraining triggers. Live training writes one latest snapshot per
symbol/interval. Scheduled optimization prioritizes triggered targets within
its existing bounded time and model-trial limits.

Rolling return, Sharpe, and drawdown reuse the canonical performance evaluator,
including the initial-equity peak. Material recent-replay underperformance
against the equal-window frozen baseline now produces its own reason code
instead of remaining report-only.

Monitoring is diagnostic: it cannot place orders, publish a model by itself, or
override frozen-test and walk-forward promotion gates.

### Regime Boundary

`regime.py` now owns the first causal, rule-based market-state baseline. It
emits a shared `RegimeState` contract and row-level fields for:

- trend up and trend down;
- range;
- high volatility;
- crash;
- liquidity crisis;
- volatility and liquidity substates;
- confidence, risk-off status, and an explainable reason.

ATR regime thresholds use only trailing observations. The previous early-window
fallback to a full-sample median was removed because it could leak future
distribution information. Backtest reports now include performance by regime
using executed exposure and execution events. The existing strategy gate is
still preserved, so this increment adds observability without silently changing
the current trading policy.

`statistical_regime.py` adds an optional walk-forward KMeans baseline behind
`regime_detection_method=walk_forward_kmeans`. Each fit and normalization
window ends before its prediction block, and appending future rows cannot
change earlier assignments. The method falls back explicitly when sklearn is
unavailable, history is insufficient, observations are not distinct enough,
or fitting fails. Rule-based crash and liquidity-crisis detection retains
highest priority. Reports record requested/used method, model version, cluster,
and fallback reason. The project default remains `rule_based`.

### Alpha Model Boundary

`alpha_models.py` defines classifier, regressor, and ranker adapters behind a
versioned interface. Existing LightGBM classifiers now enter the candidate pool
through the classifier adapter. A separate LightGBM regressor fits
`future_return` on the training split and reports validation-calibration MAE,
RMSE, directional accuracy, and correlation without reading test rows.

The fitted regressor is stored as `alpha_expected_return` in `ModelBundle`, and
live-training reports populate `AlphaPrediction.expected_return` when it is
available. Strategy, risk, and execution do not consume the field in this
increment. Auxiliary fitting occurs only after model, threshold, and test
evaluation are frozen, using remaining time, so it cannot change selection.
The Alpha prediction version records both contributing model names.

The LightGBM ranker adapter requires explicit cross-sectional symbol groups.
Per-symbol optimization therefore reports the interface as ready but not
trained instead of misusing time rows as ranking groups. Rankers emit a
`rank_score`, not an expected return or probability.

New LightGBM adapters serialize fitted Booster text, allowing the parent bundle
to load even when the optional dependency is absent. Inference from that
optional model still requires LightGBM and fails explicitly when unavailable.

### Portfolio Construction Boundary

`portfolio.py` now converts per-symbol runner reports into constrained target
notional weights. It uses optimized direction thresholds and side policy,
risk-approved capacity, current liquidity quality, inverse volatility, and
causal trailing return correlations.

The allocator enforces single-symbol, total gross, total notional leverage, and
correlation-cluster caps. It also estimates trailing historical Expected
Shortfall from aligned closed-bar returns and proportionally scales all active
weights when the configured tail-loss limit is exceeded. It writes
`portfolio_snapshot_latest.json` and feeds a local-refresh dashboard view. The
output is explicitly `planning_only`; it is not connected to order creation.

Portfolio drawdown and Beijing-day loss now come from the persistent local paper
equity path. Frozen or recent backtest drawdown is deliberately not presented
as current portfolio drawdown.

### Cross-Symbol Paper Ledger Boundary

`portfolio_paper.py` now persists one state per interval and marks previous
signed portfolio weights on each new aligned closed K-line. It tracks equity,
peak drawdown, Beijing-day return, turnover, commission, and slippage.

The runner reads this state before constructing the next portfolio. Maximum
drawdown and daily-loss breaches produce extreme-risk zero-weight decisions.
Duplicate, stale, unaligned, or missing-price updates cannot repeat or invent
portfolio PnL. Corrupt state raises instead of silently resetting the risk
history.

The portfolio ledger is not an exchange execution engine. Funding,
symbol-specific partial fills, and liquidation events still live in the
single-symbol paper broker and require future reconciliation.

### Backtest And Evaluation Boundary

`performance_report.py` now owns the canonical total return, maximum drawdown,
interval-aware annualized return, Sharpe-like, Sortino, Calmar, cost ratio,
gross-before-cost return, and Beijing calendar performance formulas.

Historical backtest and single-symbol paper replay both consume this evaluator.
Paper replay records one close-of-bar equity/cost/turnover snapshot per source
K-line and folds its final forced close into the final bar, avoiding a fake
extra period. Reports include performance by year, month, and symbol. Compact
simulation memory retains the new scalar metrics but excludes calendar
dictionaries to bound long-running state growth.

This boundary is evaluative only. It cannot select a model, approve a strategy,
or bypass strategy-validation promotion gates. Short samples can produce
unstable annualized ratios, so reports also expose duration and inferred
periods per year.

### Model Selection And Strategy Calibration Boundary

`model_selection.py` now separates predictive Alpha ranking from the existing
strategy-gated compatibility ranking. Reports preserve the compatibility
selection behavior while exposing both rankings, their candidate counts, and
whether their winners agree.

`strategy_calibration.py` owns the typed chronological validation calibration
and gate split, purge audit, threshold search space, side-policy normalization,
cost-quality metrics, directional preflight gates, strategy-archetype gates,
and the calibration backtest evaluator. `risk_profile_catalog.py` loads and
validates small-account risk profiles from
`config/strategy_calibration_profiles.toml`.

The evaluator now represents every search combination as an immutable
`StrategyCalibrationCandidate`, every score as
`StrategyCalibrationScoreBreakdown`, and every validation replay as
`StrategyCalibrationEvaluation`. A deterministic candidate iterator replaces
the previous four nested loops while preserving risk-profile, threshold, and
side-policy order. `StrategyCalibrationFinalization` now owns both the
no-trade fallback and independent validation-gate replay.
The test split is not accepted by this boundary. Main test metrics are
calculated only after the model and optimized strategy configuration are
frozen.

The calibration report now includes
`strategy_calibration_contract_version=2026-06-20-v1` and an explicit
`test_used_for_selection=false` audit field. `model_optimization.py` re-exports
the moved functions for compatibility while retaining only one implementation.
Risk-profile reports also record the TOML schema, catalog version, and source
path. Unknown fields, duplicate names, missing compact profiles, and invalid
`BacktestConfig` combinations fail explicitly.
Reports identify the typed evaluator with
`strategy_calibration_engine_contract_version=2026-06-20-typed-v1`. Existing
ranking keys and the return-first score formula are unchanged.

### Meta Signal And Strategy Rule Boundary

`meta_signal.py` now owns an explainable weighted fusion of directional Alpha,
trend, mean-reversion, funding, cross-sectional, regime, transaction-cost, and
risk inputs. Crash, liquidity-crisis, explicit risk veto, and insufficient
expected return after cost can reduce the signal to zero.

`strategy.py` now exposes `BaseStrategy`, `TrendStrategy`,
`MeanReversionStrategy`, `FundingStrategy`, `CrossSectionalStrategy`, and
`StrategyOrchestrator`. These classes only produce `StrategyDecision`
requests. Live reports expose the latest fused signal and strategy request as
`planning_only`; they are not connected to order creation.

### Expanded Label And Feature Interfaces

`labeling.py` now includes separate triple-barrier, maximum favorable/adverse
excursion, and cross-sectional rank-label builders in addition to the existing
return, risk-adjusted, and meta labels. The new forward labels remain excluded
from `FEATURE_COLUMNS`.

`feature_catalog.py` records technical, volatility, volume, funding,
open-interest, microstructure, cross-sectional, and regime feature families.
Default features now include causal funding level/change, realized volatility,
price-volume divergence, a spread proxy, and drawdown from a recent high.
Open-interest and multi-symbol features remain explicit optional transforms.

### Portfolio Risk Completion

Portfolio construction now adds sector caps, a 24-hour portfolio volatility
target, and fail-closed input validation. Missing or stale risk, execution,
volatility, liquidity, or bar-time context produces zero asset capacity.
Regime crash/liquidity risk-off is an actual risk veto in backtest and recent
closed-bar reporting. `max_leverage` is now a hard `BacktestConfig` limit, not
only a search-space convention.

### Execution Planning Interfaces

`execution_algorithms.py` adds simulation-only limit-first, TWAP, and VWAP
plans plus an optional learned-slippage protocol. Paper replay now applies
configured whole-bar decision latency. Backtest trade counts, win rate, profit
factor, expectancy, and side attribution use executed exposure and turnover,
so rejected signals no longer appear as trades.

### Optional Model Registries

`regime_models.py` records implemented KMeans/rule methods and causal optional
boundaries for GMM, HMM, and LightGBM regime classifiers. `alpha_models.py`
adds optional XGBoost and CatBoost expected-return regressor builders. Missing
dependencies remain explicit skip states.

## Preservation Rules

- Preserve CLI commands and model bundle compatibility.
- Preserve Binance historical and realtime data paths.
- Preserve purged chronological splitting.
- Preserve desktop and browser control surfaces.
- Preserve report formats unless a versioned additive field is introduced.
- Do not enable API-key use or real orders.

## Next Increments

1. Add exchange-context freshness monitoring and scheduled refresh health.
2. Add order-book/spread inputs and optional learned slippage prediction.
3. Build a causal cross-sectional dataset before enabling the LightGBM ranker.
4. Reconcile the cross-symbol paper ledger with symbol-level funding, partial
   fills, liquidation events, and matching frozen model versions.
