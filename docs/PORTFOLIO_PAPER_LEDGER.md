# Cross-Symbol Paper Portfolio Ledger

## Purpose

The cross-symbol paper ledger gives the portfolio and risk layers one persistent
equity path. It remains local simulation only and has no exchange adapter,
account credentials, or order-placement capability.

## Causal Update Order

For each new aligned closed K-line:

1. Load the previous interval-specific state.
2. Mark the previous signed target weights using current versus previous close.
3. Reset the daily reference only when the Beijing calendar day changes.
4. Calculate current total return, peak drawdown, and Beijing-day return.
5. Apply drawdown and daily-loss circuit breakers before new allocation.
6. Build new constrained portfolio target weights.
7. Deduct paper commission and slippage from weight turnover.
8. Persist state, a compact event ledger, and the latest report.

This ordering prevents current model output from earning the return that
occurred before the model decision.

## Files

```text
state/portfolio_paper_<INTERVAL>.json
reports/portfolio_paper_<INTERVAL>.jsonl
reports/portfolio_paper_latest.json
reports/portfolio_snapshot_latest.json
```

State is separated by interval. The JSONL event history is bounded by
`portfolio_paper_max_history`.

## Safety And Recovery

- Only aligned closed bars can advance the ledger.
- Duplicate or stale bars cannot repeat PnL, turnover, commission, or slippage.
- Missing prices for an active symbol block the update.
- Corrupt or unsupported state raises an error instead of resetting loss
  history.
- Unaligned market data causes an extreme-risk zero-weight portfolio decision.
- All reports explicitly keep `live_trading_enabled=false`.

## Current Cost Model

Rebalance turnover is the sum of absolute target-weight changes. Commission
uses the configured maker/taker mix. Slippage is applied to the non-maker
portion. Funding and symbol-specific partial fills remain represented in the
single-symbol paper broker and are not yet reconciled into this portfolio
ledger.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_portfolio_paper -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
