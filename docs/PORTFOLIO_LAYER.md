# Portfolio Construction Layer

## Purpose

The portfolio layer converts already filtered per-symbol model, strategy, risk,
liquidity, and exchange-availability outputs into cross-asset target notional
weights. It is a planning layer only. It does not create orders or enable live
trading.

## Allocation

The uncapped score is:

```text
direction * signal_strength * confidence / max(volatility, volatility_floor)
```

Allocation is proportional to absolute score and then constrained by:

- optimized long and short thresholds;
- optimized `trade_side_policy`;
- trade-probability threshold;
- symbol risk veto and risk-approved position size;
- minimum liquidity score;
- maximum single-symbol weight;
- maximum correlation-cluster exposure;
- target gross exposure;
- maximum total notional leverage;
- trailing historical Expected Shortfall/CVaR limit;
- portfolio drawdown veto.

Positive weights mean long planning exposure. Negative weights mean short
planning exposure. Zero means no approved allocation.

## Correlation Clusters

`correlation_clusters()` uses absolute trailing return correlation and
deterministic connected components. Symbols connected above the configured
threshold share one cluster cap. The input contains only current and historical
closed K-lines.

## Expected Shortfall

The portfolio layer aligns trailing closed-bar returns for every active symbol,
applies the proposed long or short weights, and measures the lower-tail mean at
the configured confidence level. If historical Expected Shortfall exceeds
`portfolio_max_cvar_loss`, all active target weights are scaled proportionally.

This is a backward-looking stress proxy, not a prediction of the next loss.
When there are no active weights or insufficient aligned history, the report
marks the estimate unavailable instead of reporting artificial zero risk.

## Reports

Each runner iteration writes:

```text
reports/portfolio_snapshot_latest.json
```

The same snapshot is embedded in `reports/runner_live_latest.json`. The report
includes inputs, constraints, clusters, target weights, reason codes, and a
hard safety block:

```json
{
  "status": "planning_only",
  "safety": {
    "live_trading_enabled": false,
    "real_orders_allowed": false,
    "api_keys_used": false
  }
}
```

## Current Limitation

Runner snapshots now read drawdown and Beijing-day return from the persistent
cross-symbol paper ledger. Backtest maximum drawdown is not substituted for
current portfolio drawdown.

The ledger currently applies close-to-close portfolio returns plus rebalance
commission and slippage. Funding, symbol-specific partial fills, and
liquidations remain in the single-symbol paper broker until both paper paths
can be reconciled without double counting.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_portfolio -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
