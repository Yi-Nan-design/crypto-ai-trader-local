# Liquidation Model

## Purpose

`liquidation.py` models isolated-margin liquidation risk for historical
backtests and local paper replay. It does not query an exchange and does not
place orders.

## Assumptions

The adverse price distance from entry is:

```text
1 / leverage - maintenance_margin_rate - liquidation_buffer
```

The default assumptions are:

- maintenance margin rate: `0.005`;
- liquidation safety buffer: `0.01`;
- liquidation fee rate: `0.005`;
- liquidation guard enabled.

Configuration is rejected when maintenance margin plus the buffer is greater
than or equal to initial margin.

## Trigger Order

For a position held across the next K-line:

1. If the next bar opens beyond the liquidation threshold, liquidation occurs.
2. Otherwise, a protective stop closer than liquidation is assumed to execute
   first.
3. If no closer protective stop exists and the next high or low crosses the
   threshold, intrabar liquidation occurs.

The next bar is used only for realized outcome simulation. It is not used as a
feature or trading signal.

## Execution Effects

After liquidation:

- the executed position is forced to zero;
- forced-close turnover is charged as taker turnover;
- normal commission and slippage apply to forced turnover;
- a separate liquidation fee is deducted;
- a continuing signal must open a new position and pay re-entry costs.

## Report Fields

Backtest details add liquidation trigger, gap trigger, price distance, price
return, fee cost, forced turnover, ending position, and reason fields.
Backtest summaries add liquidation assumptions, event counts, fee drag, and
forced turnover.

## Limitations

- Maintenance margin is a configured approximation, not Binance's symbol and
  notional-tier table.
- Insurance fund, auto-deleveraging, bankruptcy price, and mark-price basis
  are not modeled.
- Within-bar stop and liquidation ordering is inferred from OHLC data.
- Historical exchange outages and order-book gaps remain future work.
