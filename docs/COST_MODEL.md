# Execution Cost Model

## Purpose

`cost_model.py` calculates vectorized execution costs independently from alpha,
strategy, risk, and performance reporting. It does not place orders.

## Current Cost Components

- `commission_cost`: maker turnover uses `maker_fee_rate`; taker turnover uses
  `fee_rate`;
- `slippage_cost`: taker turnover multiplied by a causal effective rate built
  from base slippage, market participation, closed-bar range, and relative
  quote-volume stress;
- `funding_cost`: cached Binance history is aligned as sparse
  `funding_payment_rate` events and charged once against the position held
  before the current bar's order;
  externally supplied continuous `funding_rate_8h` is prorated over each bar;
  otherwise the legacy absolute configured buffer is used;
- `liquidation_fee_cost`: additional fee when liquidation forces a position
  flat;
- `trade_cost`: legacy compatibility field equal to commission plus slippage;
- `total_cost`: commission plus slippage plus funding plus liquidation fee.

All values are stored as fractions of account equity for each bar.

## Execution Path

The target notional path is transformed into an executed path before PnL and
costs are calculated:

- `execution_latency_bars` delays the target by a fixed number of bars;
- `partial_fill_ratio` deterministically fills part of each requested change;
- when `liquidity_execution_enabled=true`, executed notional is also capped by
  `quote_volume * max_bar_participation_rate`;
- `min_order_notional_fraction` rejects changes smaller than the configured
  fraction of account equity;
- `exchange_quantity_step` rounds requested quantity down to the configured
  symbol step;
- `exchange_min_quantity` and `exchange_min_notional_usdt` reject orders that
  remain below configured exchange limits after rounding;
- `exchange_max_quantity` caps each simulated child order so a large target is
  reached over later bars instead of submitting an invalid oversized order;
- `exchange_price_tick_size` is cached as metadata; the current market-order
  path does not fabricate a limit price, so tick rounding waits for a
  price-bearing limit/stop order contract;
- the exchange availability guard defers normal position changes during
  explicit outages and causal K-line gap recovery;
- `maker_fill_fraction` splits executed turnover between maker and taker fees.
- liquidation-forced turnover is assigned to taker flow, resets the execution
  state to flat, and makes a continuing target pay re-entry costs.

The defaults preserve the previous behavior: zero latency, full fills, no
minimum-order rejection, and all turnover charged as taker flow.

## Reporting Compatibility

`fee_drag` remains the legacy commission-plus-slippage aggregate because model
selection and historical reports already depend on that field. New reports add:

- `commission_drag`;
- `slippage_drag`;
- `funding_drag`;
- `funding_debit`, `funding_credit`, and `funding_rate_source`;
- `total_cost_drag`;
- `average_slippage_cost`;
- `average_fill_ratio`;
- `average_liquidity_fill_ratio`, `liquidity_limited_orders`, average market
  participation, and effective slippage rates;
- `execution_events`;
- `minimum_order_rejections`;
- `exchange_filter_rejections`, `quantity_rounding_loss_usdt`, and configured
  exchange limits;
- `maximum_quantity_limited_orders`, `exchange_downtime_blocked_orders`, and
  `exchange_available_rate`;
- `maker_turnover` and `taker_turnover`;
- `execution_latency_bars`;
- `liquidation_fee_drag`, `liquidation_forced_turnover`, and liquidation event
  counts.

Older reports remain readable. The dashboard labels missing split fields as a
legacy combined cost instead of treating them as zero.

## Current Limitations

- maker participation is a configured fixed fraction rather than a queue or
  limit-order fill model;
- the liquidity model uses OHLCV proxies rather than order-book depth, spread,
  queue position, or trade-level impact;
- `partial_fill_ratio` remains a deterministic operational multiplier, while
  the additional liquidity cap is volume-driven;
- exchange filters are refreshed by `sync-market-context` and loaded from the
  per-symbol cache whenever explicit configuration remains zero;
- exchange downtime uses explicit availability and missing-K-line recovery
  guards, but does not reconstruct matching-engine incident timelines;
- liquidation uses configured maintenance-margin assumptions rather than
  exchange symbol and notional tiers;
- `sync-market-context` downloads Binance funding history and
  `load_symbol_interval` merges it causally; refresh is explicit rather than
  scheduled;
- cached Binance funding is charged at settlement events; only the
  compatibility `funding_rate_8h` input is prorated;
- cached exchange rules cover minimum notional, minimum/maximum quantity,
  quantity step, and price-tick metadata; explicit limit-order simulation
  remains a future execution enhancement;
- paper replay now shares exchange normalization, liquidity capacity, partial
  fills, maker/taker fees, dynamic slippage, funding events, and liquidation
  cost assumptions; fixed bar latency is still vectorized-backtest only.

## Verification

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli smoke
```

The cost tests verify:

```text
trade_cost = commission_cost + slippage_cost
total_cost = trade_cost + funding_cost + liquidation_fee_cost
```

They also cover default immediate fills, latency, deterministic and
liquidity-driven partial fills, dynamic slippage, minimum-order rejections,
maker/taker cost splitting, side-aware historical funding, causality,
quantity-step rounding, exchange rejection reasons, and paper-order filtering.
They also cover maximum-quantity child orders, downtime blocking, and
liquidation override behavior.
The compatibility smoke total return must remain unchanged.
