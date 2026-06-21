# Performance Evaluation Layer

`crypto_ai_trader/performance_report.py` is the canonical performance
calculation boundary shared by historical backtests and single-symbol paper
replays. It does not select models, change strategy decisions, or place orders.

## Canonical Inputs

The evaluator consumes a bar-level detail frame containing:

- `strategy_return`: net equity return after all modeled costs;
- `total_cost`: commission, slippage, funding, and liquidation cost as an
  equity fraction;
- `equity`: close-of-bar simulated equity;
- `open_time` or `open_datetime`: UTC source timestamp;
- optional position, exposure, turnover, and symbol fields.

Paper replay converts each completed bar into this schema. Its final forced
close is folded into the final source bar instead of creating a false extra
period.

## Metric Definitions

- `total_return`: final equity divided by initial equity minus one.
- `annualized_return`: geometric total return annualized over observed days.
- `sharpe_like`: mean bar return divided by bar-return standard deviation,
  scaled by the inferred number of bars per year. The risk-free rate is zero.
- `sortino_ratio`: mean bar return divided by downside RMS deviation, using the
  same interval-aware annualization.
- `calmar_ratio`: annualized return divided by absolute maximum drawdown.
- `fee_ratio`: total modeled cost divided by positive gross bar returns, where
  gross return is net strategy return plus modeled cost. Despite the legacy
  field name, it includes slippage, funding, and liquidation costs.
- `gross_return_before_cost`: geometric return after adding modeled cost back
  to each net bar return.
- `performance_by_year` and `performance_by_month`: bar-level net performance
  grouped after converting UTC timestamps to `Asia/Shanghai`.
- `performance_by_symbol`: the same compact statistics grouped by symbol.

Undefined positive ratios use the existing bounded sentinel `999.0`. Empty or
inactive samples remain zero and must not be described as profitable.

## Report Integration

Historical `BacktestResult` and `reports/*_paper_summary.json` expose the same
core return, drawdown, annualization, risk-adjusted, cost, calendar, and symbol
metrics. Paper summaries retain close-trade profit factor and win rate for
compatibility.

`simulation_memory.py` stores compact scalar metrics such as annualized return,
Sortino, Calmar, cost ratio, turnover, and exposure. It intentionally excludes
year/month dictionaries to prevent long-running memory files from growing
without bound.

## Limitations

- Annualization can exaggerate short samples; always inspect `duration_days`.
- Bar-level returns are simulation results, not executable account statements.
- Paper replay and backtest still use OHLCV liquidity proxies rather than
  order-book queue position.
- No metric overrides strategy-validation promotion gates.
- Live trading remains disabled.
