# Autonomous Agent

This document describes the local autonomous review commands. The current
scope is deliberately conservative: review reports, generate suggestions, and
optionally run local model optimization or training. It must not place real
orders.

## Safety boundary

- `live_trading_enabled` must remain `false`.
- The agent is for report review and local research workflows only.
- Real exchange order placement, automatic full-position trading, and live
  trading escalation are out of scope.
- If a local `config.json` or `config.default.json` sets
  `live_trading_enabled` to `true`, the helper scripts stop before invoking the
  CLI.

## Single review

Run one autonomous review pass:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_autonomous_review.ps1"
```

The script invokes:

```powershell
python -m crypto_ai_trader.cli autonomous-review
```

Common overrides:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_autonomous_review.ps1" -Symbols ETHUSDT,BNBUSDT -Interval 1h -RunnerInterval 5m
```

Pass additional CLI flags after `--%` when the core command adds new options:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\run_autonomous_review.ps1" --% --min-profit-factor 1.2 --max-drawdown-limit 0.05
```

## Timed loop

Run the autonomous review loop in the current terminal:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_autonomous_loop.ps1"
```

The script invokes:

```powershell
python -m crypto_ai_trader.cli autonomous-loop
```

Default parameters:

- Symbols: `ETHUSDT`, `BNBUSDT`
- Historical interval: `1h`
- Runner interval: `5m`
- Review cadence: `900` seconds

Override the loop cadence:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_autonomous_loop.ps1" -ReviewEverySeconds 1800
```

Allow the loop to request bounded local optimization or realtime data sync only
when you explicitly want that extra work:

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_autonomous_loop.ps1" -ExecuteOptimization
powershell -ExecutionPolicy Bypass -File ".\scripts\start_autonomous_loop.ps1" -ExecuteLiveTrain
```

## Expected outputs

The core CLI should write review artifacts under the existing local output
folders, typically:

```text
reports/
state/
```

Recommended review artifacts include a timestamped JSON report and a latest
summary file, for example:

```text
reports/autonomous_review_*.json
reports/autonomous_review_latest.json
logs/autonomous_loop.out.log
logs/autonomous_loop.err.log
```

These files are local research outputs. They are not approval to enable live
trading.
