# Monitoring And Retraining

## Purpose

`monitoring.py` separates model/strategy health diagnostics from training,
strategy decisions, risk approval, and execution. It never places orders.

Each `live-train` or runner training cycle writes:

```text
reports/<SYMBOL>_<INTERVAL>_monitoring.json
```

The same snapshot is embedded in the corresponding training report.

## Monitors

- feature PSI using reference-training quantile bins;
- two-sample empirical KS statistic;
- Brier score and expected calibration error;
- mean prediction confidence and low-confidence rate;
- interval-aware rolling Sharpe, rolling drawdown, profit factor, and trades;
- recent replay versus an equal-row frozen-test baseline;
- actual paper-ledger versus frozen-test return, drawdown, profit factor, and
  win-rate deltas when a matching model report exists;
- down/neutral/up micro-regime distribution shift.

The recent replay is diagnostic and may overlap the latest labeled window. It
does not replace chronological split evaluation or strategy validation and is
not labeled as live performance. Paper reconciliation is `missing`, `legacy`,
or `model_mismatch` unless the report contains comparable metrics from the same
model name.

## Retraining Trigger

Default thresholds are in `config.default.json`:

```text
monitoring_psi_threshold
monitoring_ks_threshold
monitoring_min_confidence
monitoring_max_ece
monitoring_min_rolling_sharpe
monitoring_max_drawdown
monitoring_return_deviation
monitoring_regime_shift_threshold
```

The trigger records explicit reason codes such as:

- `feature_psi_exceeded`;
- `feature_ks_exceeded`;
- `prediction_confidence_deteriorated`;
- `probability_calibration_deteriorated`;
- `rolling_sharpe_below_threshold`;
- `rolling_drawdown_exceeded`;
- `recent_return_below_equal_window_baseline`;
- `paper_return_below_frozen_test`;
- `market_regime_distribution_shift`.

A trigger prioritizes the target during the next bounded
`scheduled-optimize` run. It does not start an unbounded process and does not
publish a model automatically.

The trigger lifecycle requires consecutive breaches, has an explicit expiry,
and is acknowledged only after that exact symbol/interval completes a scheduled
optimization. Monitoring files include schema, algorithm, model, trigger, and
acknowledgement versions so old reports cannot silently remain active.
Acknowledgement means the optimization job handled the alert; it does not mean
the resulting model is profitable or promoted. The next monitoring snapshot
must reassess drift, calibration, and paper deviation.

Reference populations are compressed into PSI histograms and KS quantiles.
Raw training rows are not duplicated. At most five immutable reference
versions are retained per target, and monitoring history is compacted to the
latest 500 events.

## Dashboard

The desktop/browser control panel reads `/api/monitoring/latest` on its slow
status refresh cadence. The chart and runner status keep their independent
refresh schedules, so the whole page is not reloaded.

`simulation_memory_latest.json` includes active monitoring alerts separately
from return observations. This prevents a drift event from being misread as a
profitable or losing backtest while still allowing the scheduler and review
agent to prioritize it.

## Safety

- `live_trading_enabled=false`;
- no account API or API key is used;
- retraining output remains a research/paper candidate;
- frozen test data remains evaluation-only;
- strategy-validation promotion gates remain authoritative.
