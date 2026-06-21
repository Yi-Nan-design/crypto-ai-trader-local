from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Protocol

import numpy as np

from .progress import safe_replace_text
from .time_utils import BEIJING_TZ, beijing_now_iso, to_beijing_iso


PORTFOLIO_PAPER_SCHEMA_VERSION = 1


class PortfolioPaperConfig(Protocol):
    """Configuration required by the cross-symbol paper equity ledger."""

    portfolio_paper_initial_balance: float
    portfolio_paper_max_history: int
    maker_fill_fraction: float
    maker_fee_rate: float
    fee_rate: float
    slippage_rate: float
    max_daily_loss: float
    portfolio_max_drawdown: float


@dataclass
class PortfolioPaperState:
    """Persistent cross-symbol paper equity and target-weight state."""

    schema_version: int = PORTFOLIO_PAPER_SCHEMA_VERSION
    interval: str = ""
    initial_balance: float = 10_000.0
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    day_start_equity: float = 10_000.0
    current_day_beijing: str = ""
    last_open_time: int | None = None
    last_datetime_beijing: str = ""
    weights: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    total_turnover: float = 0.0
    commission_fees: float = 0.0
    slippage_paid: float = 0.0
    circuit_breaker_events: int = 0
    last_reason: str = "portfolio_paper_initialized"

    @property
    def total_return(self) -> float:
        return self.equity / max(self.initial_balance, 1e-12) - 1.0

    @property
    def drawdown(self) -> float:
        return self.equity / max(self.peak_equity, 1e-12) - 1.0

    @property
    def daily_return(self) -> float:
        return self.equity / max(self.day_start_equity, 1e-12) - 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "total_return": float(self.total_return),
                "drawdown": float(self.drawdown),
                "daily_return": float(self.daily_return),
            }
        )
        return payload


@dataclass(frozen=True)
class PortfolioPaperMark:
    """Latest aligned closed-bar prices extracted from runner reports."""

    open_time: int
    datetime_beijing: str
    prices: dict[str, float]


def _beijing_day(open_time_ms: int) -> str:
    return (
        datetime.fromtimestamp(open_time_ms / 1000.0, tz=timezone.utc)
        .astimezone(BEIJING_TZ)
        .date()
        .isoformat()
    )


def extract_aligned_mark(
    reports: list[dict[str, Any]],
) -> tuple[PortfolioPaperMark | None, str]:
    """Extract one common closed K-line mark without mixing symbol timestamps."""

    rows: list[tuple[str, int, float, str]] = []
    for report in reports:
        symbol = str(report.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            open_time = int(report.get("latest_open_time"))
            price = float(report.get("latest_close"))
        except (TypeError, ValueError):
            return None, f"invalid_latest_mark:{symbol}"
        if not np.isfinite(price) or price <= 0.0:
            return None, f"invalid_latest_price:{symbol}"
        rows.append(
            (
                symbol,
                open_time,
                price,
                to_beijing_iso(
                    report.get("latest_datetime")
                    or open_time / 1000.0
                ),
            )
        )
    if not rows:
        return None, "no_runner_reports"
    timestamps = {item[1] for item in rows}
    if len(timestamps) != 1:
        return None, "unaligned_latest_open_time"
    open_time = rows[0][1]
    datetime_beijing = rows[0][3] or to_beijing_iso(open_time / 1000.0)
    return (
        PortfolioPaperMark(
            open_time=open_time,
            datetime_beijing=datetime_beijing,
            prices={symbol: price for symbol, _, price, _ in rows},
        ),
        "aligned_closed_bar",
    )


def new_portfolio_paper_state(
    interval: str,
    cfg: PortfolioPaperConfig,
) -> PortfolioPaperState:
    """Create a new paper state without enabling any execution capability."""

    initial = max(float(cfg.portfolio_paper_initial_balance), 1e-9)
    return PortfolioPaperState(
        interval=interval,
        initial_balance=initial,
        equity=initial,
        peak_equity=initial,
        day_start_equity=initial,
    )


def load_portfolio_paper_state(
    path: Path,
    *,
    interval: str,
    cfg: PortfolioPaperConfig,
) -> PortfolioPaperState:
    """Load a versioned state; corrupt state raises instead of resetting risk."""

    if not path.exists():
        return new_portfolio_paper_state(interval, cfg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid portfolio paper state: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid portfolio paper state payload: {path}")
    if int(payload.get("schema_version", 0)) != PORTFOLIO_PAPER_SCHEMA_VERSION:
        raise ValueError("unsupported portfolio paper state schema")
    state = PortfolioPaperState(
        schema_version=int(payload["schema_version"]),
        interval=str(payload.get("interval") or interval),
        initial_balance=float(payload["initial_balance"]),
        equity=float(payload["equity"]),
        peak_equity=float(payload["peak_equity"]),
        day_start_equity=float(payload["day_start_equity"]),
        current_day_beijing=str(payload.get("current_day_beijing") or ""),
        last_open_time=(
            int(payload["last_open_time"])
            if payload.get("last_open_time") is not None
            else None
        ),
        last_datetime_beijing=str(
            payload.get("last_datetime_beijing") or ""
        ),
        weights={
            str(key).upper(): float(value)
            for key, value in dict(payload.get("weights") or {}).items()
        },
        prices={
            str(key).upper(): float(value)
            for key, value in dict(payload.get("prices") or {}).items()
        },
        steps=int(payload.get("steps", 0)),
        total_turnover=float(payload.get("total_turnover", 0.0)),
        commission_fees=float(payload.get("commission_fees", 0.0)),
        slippage_paid=float(payload.get("slippage_paid", 0.0)),
        circuit_breaker_events=int(
            payload.get("circuit_breaker_events", 0)
        ),
        last_reason=str(
            payload.get("last_reason") or "portfolio_paper_loaded"
        ),
    )
    if state.interval != interval:
        raise ValueError(
            f"portfolio paper interval mismatch: {state.interval} != {interval}"
        )
    core_values = np.asarray(
        [
            state.initial_balance,
            state.equity,
            state.peak_equity,
            state.day_start_equity,
        ],
        dtype=float,
    )
    if not np.isfinite(core_values).all() or float(core_values.min()) < 0.0:
        raise ValueError("portfolio paper state contains negative equity")
    if not all(np.isfinite(value) for value in state.weights.values()):
        raise ValueError("portfolio paper state contains invalid weights")
    if not all(
        np.isfinite(value) and value > 0.0
        for value in state.prices.values()
    ):
        raise ValueError("portfolio paper state contains invalid prices")
    return state


def mark_portfolio_paper_state(
    state: PortfolioPaperState,
    mark: PortfolioPaperMark | None,
    *,
    mark_reason: str,
) -> dict[str, Any]:
    """Apply one new aligned close to prior weights exactly once."""

    event: dict[str, Any] = {
        "schema_version": PORTFOLIO_PAPER_SCHEMA_VERSION,
        "created_beijing": beijing_now_iso(),
        "action": "NO_MARK",
        "reason": mark_reason,
        "advanced": False,
        "equity_before": float(state.equity),
        "equity_after_market": float(state.equity),
        "market_return": 0.0,
    }
    if mark is None:
        state.last_reason = mark_reason
        return event
    event.update(
        {
            "event_id": f"{state.interval}:{mark.open_time}",
            "open_time": mark.open_time,
            "datetime_beijing": mark.datetime_beijing,
        }
    )
    if state.last_open_time is not None and mark.open_time <= state.last_open_time:
        event["action"] = (
            "DUPLICATE_BAR"
            if mark.open_time == state.last_open_time
            else "STALE_BAR"
        )
        event["reason"] = event["action"].lower()
        state.last_reason = str(event["reason"])
        return event

    current_day = _beijing_day(mark.open_time)
    equity_before = float(state.equity)
    if state.last_open_time is None:
        state.current_day_beijing = current_day
        state.day_start_equity = equity_before
        market_return = 0.0
        event["action"] = "SEED"
    else:
        missing = [
            symbol
            for symbol, weight in state.weights.items()
            if abs(weight) > 1e-15
            and (symbol not in state.prices or symbol not in mark.prices)
        ]
        if missing:
            event["action"] = "MISSING_ACTIVE_PRICE"
            event["reason"] = f"missing_active_price:{','.join(sorted(missing))}"
            state.last_reason = str(event["reason"])
            return event
        if current_day != state.current_day_beijing:
            state.current_day_beijing = current_day
            state.day_start_equity = equity_before
        market_return = 0.0
        for symbol, weight in state.weights.items():
            if abs(weight) <= 1e-15:
                continue
            previous = max(float(state.prices[symbol]), 1e-12)
            current = float(mark.prices[symbol])
            market_return += float(weight) * (current / previous - 1.0)
        state.equity = max(equity_before * (1.0 + market_return), 0.0)
        event["action"] = "MARK"

    state.peak_equity = max(float(state.peak_equity), float(state.equity))
    state.last_open_time = int(mark.open_time)
    state.last_datetime_beijing = mark.datetime_beijing
    state.prices = dict(mark.prices)
    state.steps += 1
    state.last_reason = "portfolio_paper_marked"
    event.update(
        {
            "advanced": True,
            "reason": "portfolio_paper_marked",
            "equity_before": equity_before,
            "equity_after_market": float(state.equity),
            "market_return": float(market_return),
            "drawdown_after_market": float(state.drawdown),
            "daily_return_after_market": float(state.daily_return),
            "previous_weights": dict(state.weights),
        }
    )
    return event


def rebalance_portfolio_paper_state(
    state: PortfolioPaperState,
    target_weights: dict[str, float],
    cfg: PortfolioPaperConfig,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Charge paper turnover costs and store new planning weights once per bar."""

    if not bool(event.get("advanced")):
        event["target_weights"] = dict(state.weights)
        event["equity_after"] = float(state.equity)
        return event
    normalized = {
        str(symbol).upper(): float(weight)
        for symbol, weight in target_weights.items()
        if np.isfinite(float(weight))
    }
    symbols = set(state.weights) | set(normalized)
    turnover = float(
        sum(
            abs(
                float(normalized.get(symbol, 0.0))
                - float(state.weights.get(symbol, 0.0))
            )
            for symbol in symbols
        )
    )
    maker_fraction = float(
        np.clip(cfg.maker_fill_fraction, 0.0, 1.0)
    )
    commission_rate = (
        maker_fraction * max(float(cfg.maker_fee_rate), 0.0)
        + (1.0 - maker_fraction) * max(float(cfg.fee_rate), 0.0)
    )
    effective_slippage_rate = (
        (1.0 - maker_fraction) * max(float(cfg.slippage_rate), 0.0)
    )
    equity_before_cost = float(state.equity)
    commission = equity_before_cost * turnover * commission_rate
    slippage = equity_before_cost * turnover * effective_slippage_rate
    state.equity = max(equity_before_cost - commission - slippage, 0.0)
    state.total_turnover += turnover
    state.commission_fees += commission
    state.slippage_paid += slippage
    state.weights = normalized
    state.last_reason = (
        "portfolio_paper_rebalanced"
        if turnover > 1e-15
        else "portfolio_paper_held"
    )
    event.update(
        {
            "turnover": turnover,
            "commission_rate": commission_rate,
            "effective_slippage_rate": effective_slippage_rate,
            "commission": commission,
            "slippage": slippage,
            "target_weights": dict(normalized),
            "equity_after": float(state.equity),
            "drawdown": float(state.drawdown),
            "daily_return": float(state.daily_return),
            "total_return": float(state.total_return),
            "reason": state.last_reason,
        }
    )
    return event


def portfolio_paper_risk_state(
    state: PortfolioPaperState,
    cfg: PortfolioPaperConfig,
) -> dict[str, Any]:
    """Return current drawdown and Beijing-day circuit-breaker state."""

    drawdown_breached = state.drawdown <= -abs(
        float(cfg.portfolio_max_drawdown)
    )
    daily_loss_breached = state.daily_return <= -abs(
        float(cfg.max_daily_loss)
    )
    if drawdown_breached:
        reason = "portfolio_blocked_drawdown"
    elif daily_loss_breached:
        reason = "portfolio_blocked_daily_loss"
    else:
        reason = "portfolio_paper_risk_allowed"
    return {
        "allow_portfolio": not (drawdown_breached or daily_loss_breached),
        "reason": reason,
        "drawdown": float(state.drawdown),
        "daily_return": float(state.daily_return),
        "drawdown_limit": float(cfg.portfolio_max_drawdown),
        "daily_loss_limit": float(cfg.max_daily_loss),
        "current_day_beijing": state.current_day_beijing,
    }


def persist_portfolio_paper_state(
    state: PortfolioPaperState,
    event: dict[str, Any],
    *,
    state_path: Path,
    ledger_path: Path,
    latest_path: Path,
    max_history: int,
) -> dict[str, Any]:
    """Persist state and a compact idempotent event ledger."""

    state_payload = state.to_dict()
    state_payload["updated_beijing"] = beijing_now_iso()
    state_payload["safety"] = {
        "live_trading_enabled": False,
        "real_orders_allowed": False,
        "api_keys_used": False,
    }
    safe_replace_text(
        state_path,
        json.dumps(state_payload, indent=2, ensure_ascii=False),
    )

    existing: list[dict[str, Any]] = []
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                existing.append(item)
    if bool(event.get("advanced")):
        event_id = event.get("event_id")
        if event_id:
            existing = [
                item for item in existing if item.get("event_id") != event_id
            ]
        existing.append(event)
    keep = max(int(max_history), 1)
    compacted = existing[-keep:]
    safe_replace_text(
        ledger_path,
        "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in compacted
        )
        + "\n",
    )
    latest = {
        "schema_version": PORTFOLIO_PAPER_SCHEMA_VERSION,
        "created_beijing": beijing_now_iso(),
        "state_path": str(state_path),
        "ledger_path": str(ledger_path),
        "state": state_payload,
        "latest_event": event,
        "safety": state_payload["safety"],
    }
    safe_replace_text(
        latest_path,
        json.dumps(latest, indent=2, ensure_ascii=False),
    )
    return latest
