# Regime Layer

## Purpose

The regime layer describes market context. It does not predict orders and does
not bypass strategy or risk controls.

## Detection Methods

`crypto_ai_trader.regime.detect_regime_frame` classifies each bar as:

- `trend_up`
- `trend_down`
- `range`
- `high_vol`
- `crash`
- `liquidity_crisis`

It also emits volatility state, liquidity state, confidence, risk-off status,
and a reason code. Crash and liquidity-crisis states are marked risk-off for
later strategy and risk integration.

The configured default is `regime_detection_method=rule_based`. An optional
`walk_forward_kmeans` method is implemented in
`crypto_ai_trader.statistical_regime`:

- every fit uses only rows strictly before the prediction block;
- model refits occur at a configured bar interval;
- normalization statistics are fitted on the same historical window;
- cluster centroids map to trend, range, and high-volatility states;
- crash and liquidity-crisis rules always override statistical assignments;
- missing sklearn, insufficient history, insufficient distinct observations,
  or a failed fit explicitly falls back to the rule model.

The statistical method is research-only and does not change the default
strategy or risk policy.

## Leakage Control

ATR quantiles are calculated from rolling or expanding history and shifted by
one bar. KMeans training windows end before the first row in each prediction
block. Appending future rows must not change any prior rule or statistical
regime classification. These invariants are covered by `tests/test_regime.py`.

## Backtest Output

Backtest detail files include:

- `market_regime`
- `regime_confidence`
- `regime_risk_off`
- `volatility_state`
- `liquidity_state`
- `regime_reason`
- `regime_method_requested`
- `regime_method_used`
- `regime_model_version`
- `regime_fallback_reason`
- `regime_override_reason`
- `regime_cluster`

Backtest summaries include `regime_summary`, calculated from executed exposure,
execution events, returns, drawdown, profit factor, win rate, and cost drag.
They also include counts by method, model version, fallback reason, and
rule-based override reason. Safety overrides are not reported as model
fallbacks.

## Current Boundary

The existing strategy regime gate remains unchanged. Regime classification is
context consumed by existing filters and reports; it cannot place an order or
override risk. Statistical classification remains disabled by default.
