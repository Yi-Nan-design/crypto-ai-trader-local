# Exchange Availability Guard

## Purpose

`exchange_availability.py` turns causal market-data continuity into an
execution permission. It does not alter alpha predictions or strategy
directions.

## Inputs

- `exchange_available`: optional explicit public exchange/API availability;
- `exchange_gap_before_bars`: number of missing intervals observed before the
  current closed K-line;
- `exchange_gap_recovery_bars`: configured number of observed recovery bars
  during which new orders remain blocked.

`data_validation.py` creates the gap field after sorting and deduplicating the
K-line frame. A gap is known only when the first later K-line arrives, so the
guard is causal.

## Execution Semantics

- existing positions remain open while execution is unavailable;
- new opens, closes, and rebalances are deferred;
- a later available bar can continue toward the same target;
- liquidation remains a risk override and can force the simulated position
  flat even when normal execution is blocked;
- paper replay records `EXCHANGE_UNAVAILABLE` instead of fabricating a fill.
- a realtime sync failure marks the latest cached bar unavailable; a later
  successful sync restores fetched rows to available.

Availability is execution context only. It is preserved through feature
construction for reporting and admission, but excluded from the model feature
matrix so network failures cannot become an alpha shortcut.

Backtest reports expose:

- `exchange_downtime_blocked_orders`;
- `exchange_available_rate`;
- `exchange_downtime_guard_enabled`;
- `exchange_gap_recovery_bars`.

This is a conservative OHLCV continuity guard. It is not a reconstruction of
Binance incident timelines, websocket disconnects, matching-engine status, or
private order acknowledgements.
