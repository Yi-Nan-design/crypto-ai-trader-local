# Binance Market Context

## Purpose

The market-context cache adds public derivatives data that does not exist in
K-line archives:

- historical USDT-M funding settlements;
- market-order minimum notional;
- market-order minimum quantity;
- market-order maximum quantity;
- market-order quantity step;
- symbol price tick.

It never reads account data, API keys, balances, positions, or private orders.

## Refresh

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli sync-market-context --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2024-01 --end 2025-12
```

The command honors the existing HTTP/HTTPS proxy handling. It writes a
Beijing-time summary to `reports/market_context_sync_*.json`.

## Storage

Each symbol uses:

```text
data/market_context/<SYMBOL>/funding_rates.csv
data/market_context/<SYMBOL>/exchange_rules.json
```

`funding_rates.csv` stores one row per Binance funding settlement:

- `funding_time`: UTC epoch milliseconds;
- `funding_rate_8h`: signed settlement rate;
- `mark_price`: public mark price reported with the event;
- `funding_datetime`: UTC display value.

`exchange_rules.json` records the source and Beijing refresh timestamp.

## Runtime Semantics

`load_symbol_interval` maps each funding settlement to the K-line containing
its timestamp and exposes sparse `funding_payment_rate` execution context.
Funding is charged once on that bar:

```text
funding_cost = position_before_execution * funding_payment_rate
```

Positive rates debit longs and credit shorts. Negative rates do the opposite.
Funding columns are excluded from the model feature allowlist.

The event is settled before the current bar's strategy order, matching the
paper replay sequence. A position opened after that event is not charged
retroactively.

Cached exchange rules fill only configuration values that remain zero. Explicit
nonzero configuration always wins. Maximum quantity limits one child order;
subsequent bars continue toward the target. Price ticks are exposed for
simulated limit-order prices and are not incorrectly applied as a market-order
price filter.

## Current Limits

- refresh is explicit, not scheduled;
- matching-engine incidents and websocket/API status are not downloaded;
- missing K-line recovery is handled by the local exchange availability guard;
- realtime REST failure marks the latest cached K-line unavailable until a
  successful refresh replaces or extends it;
- maintenance-margin tiers remain configured approximations;
- paper replay charges cached historical funding before the current decision;
- paper replay does not yet implement configured bar latency;
- old reports do not contain market-context fields.
