# Model Selection And Strategy Calibration

The optimization pipeline now exposes two distinct validation decisions.

## Model Selection

`crypto_ai_trader/model_selection.py` ranks trained Alpha candidates in two
views:

- `predictive_ranking`: classification quality, large-move capture, overfit
  penalty, and rolling-validation penalty;
- `strategy_gated_ranking`: the existing validation backtest score with cost,
  trade-count, drawdown, profit-factor, and risk/reward gates.

The compatibility publishing path still selects a strategy-gate-passing model
first. If none passes, the selected model is explicitly marked as a research
fallback. The report records whether the predictive winner and compatibility
winner are the same.

Neither ranking accepts the test split. Test metrics are calculated only after
the selected model and strategy configuration are frozen.

## Strategy Calibration

`crypto_ai_trader/strategy_calibration.py` owns:

- chronological validation-calibration and validation-gate partitioning;
- purge rows between both validation slices;
- long/short probability threshold candidates;
- tradeability threshold candidates;
- normalized long-only, short-only, and both-side policy candidates;
- cost-quality, validation, directional preflight, side-contribution, and
  strategy-archetype gates;
- versioned small-account risk-profile candidates loaded from
  `config/strategy_calibration_profiles.toml`;
- immutable typed candidate, score-breakdown, and evaluation contracts;
- typed no-trade and independent validation-gate finalization;
- the bounded calibration backtest evaluator.

Its public split and calibration functions do not accept a test frame. Small
validation sets fall back to one validation selection set and report that no
independent gate was available. `model_optimization.py` imports and re-exports
the moved functions so existing CLI and module callers remain compatible.

Risk-profile TOML uses four explicit rule groups:

- `values`: fixed profile override;
- `min`: cap a value at the configured profile maximum;
- `max`: raise a value to the configured profile minimum;
- `base`: copy the current `BacktestConfig` value.

The loader rejects unknown fields and validates every resolved profile by
constructing a `BacktestConfig`. Missing configuration fails explicitly rather
than silently substituting stale defaults.

The typed evaluator preserves the legacy candidate enumeration order and flat
JSON ranking fields. The return-first score still uses the same return,
expectancy, profit-factor, large-move, drawdown, cost, activity, threshold,
side-policy, and leverage components.

## Report Contract

Per-target model optimization reports include:

- `ranking`: compatibility strategy-gated ranking;
- `predictive_ranking`: Alpha predictive ranking;
- `strategy_gated_ranking`: explicit alias for the compatibility ranking;
- `valid_ranked_for_selection`: only validation-gate-passing candidates;
- `model_selection_contract`: datasets, candidate counts, selected path, and
  whether predictive and selected winners match;
- `validation_calibration_split`: chronological split and purge audit;
- `strategy_calibration_contract_version`: versioned calibration payload
  contract;
- `strategy_calibration_engine_contract_version`: typed evaluator
  implementation contract;
- `risk_profile_catalog_schema_version`, `risk_profile_catalog_version`, and
  `risk_profile_catalog_path`: exact profile configuration provenance;
- `test_used_for_selection=false`: direct leakage audit on the calibration
  result;
- `test_evaluation_audit`: confirms test evaluation occurs after model and
  policy selection.

Aggregate `valid_ranked_for_selection` likewise excludes targets whose model
selection gate failed.

## Safety

- Test data is not used for model or threshold ranking.
- Test results remain a final publish safety gate.
- A research fallback is not a paper candidate.
- Strategy validation remains the authoritative promotion gate.
- Live trading remains disabled.
