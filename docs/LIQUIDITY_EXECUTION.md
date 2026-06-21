# Liquidity-Aware Execution

## Scope

This module improves backtest and paper realism using fields available in
closed Binance K-lines. It does not claim order-book or tick-level precision.

## Fill Capacity

For each requested order:

```text
capacity_usdt = quote_volume * max_bar_participation_rate
liquidity_fill_ratio = min(1, capacity_usdt / requested_notional_usdt)
executed_notional = normalized_order * partial_fill_ratio * liquidity_fill_ratio
```

Exchange quantity and minimum-notional rules are applied before the liquidity
cap. A continuing target retries the unfilled remainder on later bars.

## Dynamic Slippage

Taker slippage uses:

```text
participation = taker_notional / quote_volume
liquidity_stress = sqrt(trailing_quote_volume_median / current_quote_volume)
impact = coefficient * sqrt(participation) * bar_range * liquidity_stress
effective_slippage = clip(base_slippage + impact, base, configured_max)
```

The trailing median is shifted by one bar. Future volume cannot affect past
fills or slippage.

## Defaults

Project configuration:

```json
{
  "liquidity_execution_enabled": true,
  "max_bar_participation_rate": 0.01,
  "liquidity_lookback_bars": 48,
  "slippage_impact_coefficient": 1.0,
  "max_dynamic_slippage_rate": 0.02
}
```

Direct `BacktestConfig()` defaults to compatibility mode with liquidity
execution disabled. Training, optimization, validation, CLI backtests, and
paper replay receive the project configuration.

## Paper Replay

Paper replay now shares:

- exchange quantity normalization and rejection reasons;
- market participation capacity;
- deterministic and liquidity-driven partial fills;
- maker/taker commission split;
- dynamic slippage;
- exact cached funding settlements;
- liquidation trading fee, slippage, and liquidation fee.

Partial closes leave the residual position open and retry on later bars.

## Remaining Limits

- no bid/ask spread or order-book depth;
- no queue-position or limit-order fill probability;
- no exchange downtime or API-failure simulation;
- paper replay does not implement configured bar latency;
- the model is a conservative OHLCV approximation, not execution evidence.
