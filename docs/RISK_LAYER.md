# Risk Layer

## Purpose

The risk layer is the final authority between a strategy request and simulated
execution. It does not predict price direction and does not place live orders.

## Contracts

`StrategyDecision` requests:

- direction;
- target exposure;
- stop loss and take profit;
- holding period;
- strategy reason code.

`RiskDecision` returns:

- `allow_trade`;
- `risk_level`;
- `max_position_size`;
- risk reason code.

Strategy and risk reasons are deliberately separate. A valid alpha signal may
still be reduced or rejected by risk controls.

## Current Controls

- positive whole-number leverage validation;
- maximum strategy position fraction;
- maximum leveraged notional exposure;
- maintenance-margin and liquidation-distance validation;
- next-bar gap and intrabar liquidation detection;
- forced-flat and re-entry handling in backtest and paper replay;
- confidence-scaled exposure;
- causal closed-return EWMA volatility targeting, normalized to a 24-hour
  horizon;
- fixed-stop volatility targeting;
- row-level ATR stop-distance risk budgeting when ATR exits are enabled;
- low-liquidity exposure reduction;
- position rebalance band and maximum step;
- drawdown and loss-streak cooldown;
- cross-asset target gross exposure and total notional leverage caps;
- single-symbol and correlation-cluster exposure caps;
- portfolio liquidity, realized paper drawdown, and Beijing-day loss vetoes;
- trailing closed-bar portfolio VaR and Expected Shortfall scaling;
- directional funding-crowding veto for the side paying extreme funding;
- risk decision fields in backtest, paper, runner, and dashboard outputs.

## Current Limitations

- cooldown loss streaks count negative bars, not completed losing trades;
- paper replay does not yet reproduce every dynamic backtest sizing input;
- portfolio funding, partial-fill, and liquidation reconciliation remains
  incomplete;
- real-account daily loss control and exchange tiered maintenance margin are
  not implemented;
- live trading remains disabled.

## Verification

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli smoke
```

The default smoke result must remain unchanged after behavior-preserving risk
refactors. Risk-specific behavior is covered by `tests/test_risk.py`.
