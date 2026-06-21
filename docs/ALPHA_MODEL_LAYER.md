# Alpha Model Layer

## Purpose

The Alpha layer forecasts direction and return. It does not choose leverage,
approve risk, create orders, or grant paper/live eligibility.

## Shared Interface

`crypto_ai_trader.alpha_models` defines one interface for:

- probability classifiers;
- continuous expected-return regressors;
- cross-sectional rankers.

The interface version is `2026-06-21-v1`. All adapters expose a stable name,
target kind, fit method, and probability output. Regressors and rankers also
expose `predict_expected_return`.

## LightGBM Models

### Classifier

LightGBM direction classifiers participate in the existing validation candidate
pool through `ClassifierAlphaAdapter`. They still compete under the same
chronological validation, overfit, and strategy-compatibility gates.

### Regressor

`lightgbm_expected_return_regressor` is an auxiliary model trained on
`future_return` using:

- the training split for fitting;
- the validation-calibration split for MAE, RMSE, direction accuracy, and
  correlation;
- no test rows for training or model selection.

It is stored as `alpha_expected_return` inside `ModelBundle`. Live-training
reports can therefore populate `AlphaPrediction.expected_return`. The current
strategy and backtest do not consume this field, so adding the forecast does
not silently change position decisions.

The regressor is trained only after the primary model, threshold policy, and
test evaluation are frozen. It uses only remaining optimization time, so it
cannot reduce the threshold search budget or change candidate selection.

### Ranker

`RankerAlphaAdapter` requires explicit groups with at least two symbols per
group and finite non-negative integer relevance labels. It emits a
`rank_score`, not a probability or expected return. Current optimization runs are per symbol, so reports use
`interface_ready_not_trained` with reason
`cross_sectional_symbol_groups_required`.

Training a ranker on rows from one symbol across time would not be a valid
cross-sectional ranking problem. A later dataset builder must align multiple
symbols at each closed timestamp and provide a causal
`cross_sectional_rank_label`.

## Compatibility

- Existing `ModelBundle` pickle files still load.
- Missing LightGBM skips the optional adapters without blocking baseline
  models.
- New LightGBM adapters store fitted Booster text rather than a direct Python
  estimator. A bundle can load without LightGBM; optional inference then raises
  an explicit missing-dependency error.
- `alpha_*` auxiliary models are excluded from multi-horizon probability
  output.
- Direction probabilities remain the only Alpha values used by the current
  strategy path.
- `AlphaPrediction.model_version` records both the classifier and expected
  return regressor when both contribute.
- Live trading remains disabled.
