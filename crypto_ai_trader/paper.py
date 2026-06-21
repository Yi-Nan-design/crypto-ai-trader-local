from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, choose_position
from .contracts import RiskDecision, StrategyDecision
from .exchange_availability import (
    coerce_exchange_available,
    resolve_exchange_availability,
)
from .features import feature_only_matrix
from .liquidation import liquidation_price_distance
from .liquidity_execution import (
    dynamic_slippage_rate_scalar,
    estimate_single_order_execution,
    liquidity_execution_enabled,
)
from .models import ModelBundle
from .performance_report import evaluate_backtest_performance
from .risk import evaluate_strategy_risk, stop_distance_series
from .time_utils import beijing_now_iso


@dataclass
class PaperState:
    balance: float = 10_000.0
    equity: float = 10_000.0
    position: int = 0
    entry_price: float = 0.0
    units: float = 0.0
    realized_pnl: float = 0.0
    trades: int = 0
    liquidations: int = 0
    liquidation_fees: float = 0.0
    order_rejections: int = 0
    partial_fills: int = 0
    liquidity_limited_orders: int = 0
    commission_fees: float = 0.0
    slippage_paid: float = 0.0
    funding_net_cost: float = 0.0
    maximum_quantity_limited_orders: int = 0
    exchange_downtime_blocks: int = 0
    latency_deferred_decisions: int = 0
    pending_open_notional_usdt: float = 0.0
    stop_loss_pct: float = 0.0


class PaperBroker:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.state = PaperState(balance=cfg.initial_balance, equity=cfg.initial_balance)
        self.ledger: list[dict[str, float | int | str | bool]] = []
        self.risk_history: list[dict[str, float | bool | str]] = []
        self.quote_volume_history: list[float] = []
        self.decision_queue: deque[
            tuple[StrategyDecision, float, RiskDecision]
        ] = deque()

    def step(
        self,
        row: pd.Series,
        decision: StrategyDecision,
        prob_up: float,
        risk_decision: RiskDecision | None = None,
    ) -> None:
        price = float(row["close"])
        open_time = str(row.get("open_datetime", row.get("open_time", "")))
        quote_volume, range_proxy, trailing_quote_volume = (
            self._execution_context(row)
        )
        available_capacity = (
            quote_volume * float(self.cfg.max_bar_participation_rate)
            if liquidity_execution_enabled(self.cfg)
            else None
        )
        self._apply_funding(row, open_time)
        self._check_liquidation(
            row,
            open_time,
            quote_volume=quote_volume,
            range_proxy=range_proxy,
            trailing_quote_volume=trailing_quote_volume,
        )
        risk = risk_decision or evaluate_strategy_risk(decision, self.cfg)
        decision, prob_up, risk = self._apply_decision_latency(
            decision,
            prob_up,
            risk,
        )
        self.risk_history.append(
            {
                "time": open_time,
                "allow_trade": risk.allow_trade,
                "risk_level": risk.risk_level.value,
                "max_position_size": risk.max_position_size,
                "risk_reason": risk.reason,
                "strategy_reason_code": decision.reason_code,
            }
        )
        desired_position = decision.target_direction if risk.allow_trade else 0
        execution_available = coerce_exchange_available(
            row.get(
                "execution_available",
                row.get("exchange_available", True),
            )
        )
        order_required = (
            desired_position != self.state.position
            or (
                desired_position != 0
                and self.state.pending_open_notional_usdt > 1e-12
            )
        )
        if order_required and not execution_available:
            self.state.exchange_downtime_blocks += 1
            self.ledger.append(
                {
                    "time": open_time,
                    "action": "EXCHANGE_UNAVAILABLE",
                    "price": price,
                    "desired_position": desired_position,
                    "current_position": self.state.position,
                    "reason": str(
                        row.get(
                            "exchange_downtime_reason",
                            "exchange_unavailable",
                        )
                    ),
                    "reason_code": decision.reason_code,
                    "risk_reason": risk.reason,
                    "balance": self.state.balance,
                }
            )
            self._mark(price)
            self.quote_volume_history.append(quote_volume)
            return
        if desired_position != self.state.position:
            self.state.pending_open_notional_usdt = 0.0
            available_capacity = self._close(
                price,
                open_time,
                decision.reason_code,
                risk,
                quote_volume=quote_volume,
                range_proxy=range_proxy,
                trailing_quote_volume=trailing_quote_volume,
                available_capacity_usdt=available_capacity,
            )
            if desired_position != 0 and self.state.position == 0:
                self._open(
                    price,
                    desired_position,
                    min(decision.target_exposure, risk.max_position_size),
                    prob_up,
                    open_time,
                    decision.reason_code,
                    risk,
                    decision.stop_loss,
                    quote_volume=quote_volume,
                    range_proxy=range_proxy,
                    trailing_quote_volume=trailing_quote_volume,
                    available_capacity_usdt=available_capacity,
                )
        elif (
            desired_position != 0
            and self.state.pending_open_notional_usdt > 1e-12
        ):
            self._open(
                price,
                desired_position,
                min(decision.target_exposure, risk.max_position_size),
                prob_up,
                open_time,
                decision.reason_code,
                risk,
                decision.stop_loss,
                quote_volume=quote_volume,
                range_proxy=range_proxy,
                trailing_quote_volume=trailing_quote_volume,
                available_capacity_usdt=available_capacity,
                requested_notional_override=self.state.pending_open_notional_usdt,
            )
        self._mark(price)
        self.quote_volume_history.append(quote_volume)

    def _apply_decision_latency(
        self,
        decision: StrategyDecision,
        prob_up: float,
        risk: RiskDecision,
    ) -> tuple[StrategyDecision, float, RiskDecision]:
        """Delay paper decisions by whole closed bars using a FIFO queue."""

        latency = max(int(self.cfg.execution_latency_bars), 0)
        if latency == 0:
            return decision, prob_up, risk
        self.decision_queue.append((decision, float(prob_up), risk))
        if len(self.decision_queue) > latency:
            return self.decision_queue.popleft()
        self.state.latency_deferred_decisions += 1
        flat = StrategyDecision(
            target_direction=0,
            target_exposure=0.0,
            stop_loss=None,
            take_profit=None,
            holding_period=None,
            reason_code="execution_latency_wait",
        )
        flat_risk = RiskDecision(
            allow_trade=True,
            risk_level=risk.risk_level,
            max_position_size=0.0,
            reason="risk_no_position_requested",
        )
        return flat, 0.5, flat_risk

    def _open(
        self,
        price: float,
        side: int,
        exposure: float,
        prob_up: float,
        open_time: str,
        reason_code: str,
        risk: RiskDecision,
        stop_loss: float | None,
        *,
        quote_volume: float,
        range_proxy: float,
        trailing_quote_volume: float,
        available_capacity_usdt: float | None,
        requested_notional_override: float | None = None,
    ) -> None:
        target_notional = self.state.equity * exposure * self.cfg.leverage
        requested_notional = (
            max(float(requested_notional_override), 0.0)
            if requested_notional_override is not None
            else target_notional
        )
        execution = estimate_single_order_execution(
            side * requested_notional,
            price=price,
            quote_volume_usdt=quote_volume,
            range_proxy=range_proxy,
            trailing_quote_volume_usdt=trailing_quote_volume,
            cfg=self.cfg,
            capacity_override_usdt=available_capacity_usdt,
        )
        if not execution.accepted:
            self.state.order_rejections += 1
            self.ledger.append(
                {
                    "time": open_time,
                    "action": "ORDER_REJECTED",
                    "price": price,
                    "requested_notional_usdt": requested_notional,
                    "normalized_quantity": execution.normalized_quantity,
                    "reason": execution.reason,
                    "reason_code": reason_code,
                    "risk_reason": risk.reason,
                    "balance": self.state.balance,
                }
            )
            self.state.pending_open_notional_usdt = requested_notional
            return
        if not execution.filled:
            self.ledger.append(
                {
                    "time": open_time,
                    "action": "ORDER_UNFILLED",
                    "price": price,
                    "requested_notional_usdt": requested_notional,
                    "reason": execution.reason,
                    "reason_code": reason_code,
                    "balance": self.state.balance,
                }
            )
            self.state.pending_open_notional_usdt = requested_notional
            return
        notional = abs(execution.executed_notional_usdt)
        added_units = notional / price
        execution_cost = execution.commission_usdt + execution.slippage_usdt
        self.state.balance -= execution_cost
        if self.state.position == side and self.state.units > 0.0:
            total_units = self.state.units + added_units
            self.state.entry_price = (
                self.state.entry_price * self.state.units
                + price * added_units
            ) / total_units
            self.state.units = total_units
        else:
            self.state.position = side
            self.state.entry_price = price
            self.state.units = added_units
        self.state.pending_open_notional_usdt = max(
            requested_notional - notional,
            0.0,
        )
        self.state.stop_loss_pct = max(float(stop_loss or self.cfg.stop_loss), 0.0)
        self.state.trades += 1
        self.state.commission_fees += execution.commission_usdt
        self.state.slippage_paid += execution.slippage_usdt
        if execution.fill_ratio < 1.0 - 1e-12:
            self.state.partial_fills += 1
        if execution.liquidity_limited:
            self.state.liquidity_limited_orders += 1
        if execution.maximum_quantity_limited:
            self.state.maximum_quantity_limited_orders += 1
        self.ledger.append(
            {
                "time": open_time,
                "action": "BUY_LONG" if side == 1 else "SELL_SHORT",
                "price": price,
                "units": added_units,
                "requested_notional_usdt": requested_notional,
                "target_notional_usdt": target_notional,
                "normalized_notional_usdt": notional,
                "executed_notional_usdt": notional,
                "fill_ratio": execution.fill_ratio,
                "liquidity_fill_ratio": execution.liquidity_fill_ratio,
                "liquidity_capacity_usdt": execution.liquidity_capacity_usdt,
                "market_participation_rate": execution.market_participation_rate,
                "effective_slippage_rate": execution.effective_slippage_rate,
                "quantity_rounding_loss_usdt": execution.quantity_rounding_loss_usdt,
                "maximum_quantity_limited": execution.maximum_quantity_limited,
                "fee": execution.commission_usdt,
                "slippage": execution.slippage_usdt,
                "prob_up": prob_up,
                "reason_code": reason_code,
                "target_exposure": exposure,
                "risk_allow_trade": risk.allow_trade,
                "risk_level": risk.risk_level.value,
                "risk_reason": risk.reason,
                "balance": self.state.balance,
            }
        )

    def _close(
        self,
        price: float,
        open_time: str,
        reason: str,
        risk: RiskDecision | None = None,
        *,
        quote_volume: float,
        range_proxy: float,
        trailing_quote_volume: float,
        available_capacity_usdt: float | None,
    ) -> float | None:
        if self.state.position == 0:
            return available_capacity_usdt
        direction = self.state.position
        requested_notional = abs(price * self.state.units)
        execution = estimate_single_order_execution(
            -direction * requested_notional,
            price=price,
            quote_volume_usdt=quote_volume,
            range_proxy=range_proxy,
            trailing_quote_volume_usdt=trailing_quote_volume,
            cfg=self.cfg,
            capacity_override_usdt=available_capacity_usdt,
        )
        if not execution.accepted or not execution.filled:
            if not execution.accepted:
                self.state.order_rejections += 1
            self.ledger.append(
                {
                    "time": open_time,
                    "action": (
                        "ORDER_REJECTED"
                        if not execution.accepted
                        else "ORDER_UNFILLED"
                    ),
                    "reason": execution.reason,
                    "price": price,
                    "requested_notional_usdt": requested_notional,
                    "balance": self.state.balance,
                }
            )
            return available_capacity_usdt
        close_units = min(
            abs(execution.executed_notional_usdt) / price,
            self.state.units,
        )
        executed_notional = close_units * price
        pnl = (price - self.state.entry_price) * close_units * direction
        execution_cost = execution.commission_usdt + execution.slippage_usdt
        realized = pnl - execution_cost
        self.state.balance += realized
        self.state.realized_pnl += realized
        self.state.commission_fees += execution.commission_usdt
        self.state.slippage_paid += execution.slippage_usdt
        if execution.fill_ratio < 1.0 - 1e-12:
            self.state.partial_fills += 1
        if execution.liquidity_limited:
            self.state.liquidity_limited_orders += 1
        if execution.maximum_quantity_limited:
            self.state.maximum_quantity_limited_orders += 1
        remaining_units = max(self.state.units - close_units, 0.0)
        fully_closed = remaining_units <= 1e-12
        entry: dict[str, float | int | str | bool] = {
            "time": open_time,
            "action": "CLOSE" if fully_closed else "PARTIAL_CLOSE",
            "reason": reason,
            "price": price,
            "units": close_units,
            "requested_notional_usdt": requested_notional,
            "executed_notional_usdt": executed_notional,
            "fill_ratio": execution.fill_ratio,
            "liquidity_fill_ratio": execution.liquidity_fill_ratio,
            "liquidity_capacity_usdt": execution.liquidity_capacity_usdt,
            "market_participation_rate": execution.market_participation_rate,
            "effective_slippage_rate": execution.effective_slippage_rate,
            "maximum_quantity_limited": execution.maximum_quantity_limited,
            "fee": execution.commission_usdt,
            "slippage": execution.slippage_usdt,
            "pnl": realized,
            "balance": self.state.balance,
        }
        if risk is not None:
            entry.update(
                {
                    "risk_allow_trade": risk.allow_trade,
                    "risk_level": risk.risk_level.value,
                    "risk_reason": risk.reason,
                }
            )
        self.ledger.append(entry)
        self.state.units = remaining_units
        if fully_closed:
            self.state.position = 0
            self.state.entry_price = 0.0
            self.state.units = 0.0
            self.state.stop_loss_pct = 0.0
            self.state.pending_open_notional_usdt = 0.0
        if available_capacity_usdt is None:
            return None
        return max(
            float(available_capacity_usdt) - executed_notional,
            0.0,
        )

    def _check_liquidation(
        self,
        row: pd.Series,
        open_time: str,
        *,
        quote_volume: float,
        range_proxy: float,
        trailing_quote_volume: float,
    ) -> None:
        """Apply isolated-margin liquidation before the current close decision."""

        if (
            self.state.position == 0
            or not bool(self.cfg.liquidation_guard_enabled)
        ):
            return
        distance = liquidation_price_distance(self.cfg)
        side = self.state.position
        entry = self.state.entry_price
        bar_open = float(row.get("open", row["close"]))
        bar_high = float(row.get("high", row["close"]))
        bar_low = float(row.get("low", row["close"]))
        liquidation_price = entry * (1.0 - distance if side > 0 else 1.0 + distance)
        gap_breach = (
            bar_open <= liquidation_price if side > 0 else bar_open >= liquidation_price
        )
        intrabar_breach = (
            bar_low <= liquidation_price if side > 0 else bar_high >= liquidation_price
        )
        protected = self.state.stop_loss_pct + 1e-12 < distance
        if not gap_breach and (protected or not intrabar_breach):
            return

        pnl = (liquidation_price - entry) * self.state.units * side
        notional = abs(liquidation_price * self.state.units)
        trading_fee = notional * self.cfg.fee_rate
        slippage_rate, participation, liquidity_stress = (
            dynamic_slippage_rate_scalar(
                notional,
                quote_volume,
                range_proxy,
                trailing_quote_volume,
                self.cfg,
            )
        )
        slippage = notional * slippage_rate
        liquidation_fee = notional * self.cfg.liquidation_fee_rate
        realized = pnl - trading_fee - slippage - liquidation_fee
        self.state.balance += realized
        self.state.realized_pnl += realized
        self.state.liquidations += 1
        self.state.liquidation_fees += liquidation_fee
        self.state.commission_fees += trading_fee
        self.state.slippage_paid += slippage
        self.ledger.append(
            {
                "time": open_time,
                "action": "LIQUIDATION",
                "reason": (
                    "liquidation_gap_breach"
                    if gap_breach
                    else "liquidation_intrabar_breach"
                ),
                "price": liquidation_price,
                "units": self.state.units,
                "fee": trading_fee,
                "slippage": slippage,
                "effective_slippage_rate": slippage_rate,
                "market_participation_rate": participation,
                "liquidity_stress": liquidity_stress,
                "liquidation_fee": liquidation_fee,
                "pnl": realized,
                "balance": self.state.balance,
            }
        )
        self.state.position = 0
        self.state.entry_price = 0.0
        self.state.units = 0.0
        self.state.stop_loss_pct = 0.0
        self.state.pending_open_notional_usdt = 0.0

    def _mark(self, price: float) -> None:
        unrealized = 0.0
        if self.state.position != 0:
            unrealized = (price - self.state.entry_price) * self.state.units * self.state.position
        self.state.equity = self.state.balance + unrealized

    def _execution_context(self, row: pd.Series) -> tuple[float, float, float]:
        quote_volume = float(
            pd.to_numeric(
                pd.Series([row.get("quote_volume", 0.0)]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
        )
        close = max(float(row.get("close", 0.0)), 1e-12)
        if "high" in row and "low" in row:
            range_proxy = abs(float(row["high"]) - float(row["low"])) / close
        else:
            range_proxy = float(row.get("high_low_range", row.get("atr_14", 0.0)))
        lookback = max(int(self.cfg.liquidity_lookback_bars), 1)
        history = self.quote_volume_history[-lookback:]
        trailing = float(np.median(history)) if history else quote_volume
        return max(quote_volume, 0.0), max(range_proxy, 0.0), max(trailing, 0.0)

    def _apply_funding(self, row: pd.Series, open_time: str) -> None:
        if self.state.position == 0:
            return
        rate = pd.to_numeric(
            pd.Series([row.get("funding_payment_rate", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(rate) or abs(float(rate)) <= 1e-15:
            return
        mark_price = float(row.get("funding_mark_price", row["close"]))
        notional = abs(mark_price * self.state.units)
        funding_cost = self.state.position * notional * float(rate)
        self.state.balance -= funding_cost
        self.state.realized_pnl -= funding_cost
        self.state.funding_net_cost += funding_cost
        self.ledger.append(
            {
                "time": open_time,
                "action": "FUNDING",
                "price": mark_price,
                "funding_rate": float(rate),
                "funding_cost": funding_cost,
                "balance": self.state.balance,
            }
        )

    def close_all(
        self,
        price: float,
        open_time: str,
        row: pd.Series | None = None,
    ) -> None:
        current_row = row if row is not None else pd.Series({"close": price})
        if not coerce_exchange_available(
            current_row.get(
                "execution_available",
                current_row.get("exchange_available", True),
            )
        ):
            self.state.exchange_downtime_blocks += int(self.state.position != 0)
            if self.state.position != 0:
                self.ledger.append(
                    {
                        "time": open_time,
                        "action": "EXCHANGE_UNAVAILABLE",
                        "price": price,
                        "desired_position": 0,
                        "current_position": self.state.position,
                        "reason": str(
                            current_row.get(
                                "exchange_downtime_reason",
                                "exchange_unavailable",
                            )
                        ),
                        "balance": self.state.balance,
                    }
                )
            self._mark(price)
            return
        quote_volume, range_proxy, trailing = self._execution_context(current_row)
        capacity = (
            quote_volume * float(self.cfg.max_bar_participation_rate)
            if liquidity_execution_enabled(self.cfg)
            else None
        )
        self._close(
            price,
            open_time,
            "end_of_replay",
            quote_volume=quote_volume,
            range_proxy=range_proxy,
            trailing_quote_volume=trailing,
            available_capacity_usdt=capacity,
        )
        self._mark(price)


def _paper_total_cost_usdt(state: PaperState) -> float:
    """Return cumulative commission, slippage, funding, and liquidation cost."""

    return float(
        state.commission_fees
        + state.slippage_paid
        + state.funding_net_cost
        + state.liquidation_fees
    )


def _ledger_turnover_usdt(
    entries: list[dict[str, float | int | str | bool]],
) -> float:
    """Measure executed notional turnover from newly appended ledger events."""

    turnover = 0.0
    for entry in entries:
        action = str(entry.get("action", ""))
        if action in {"BUY_LONG", "SELL_SHORT", "CLOSE", "PARTIAL_CLOSE"}:
            turnover += abs(float(entry.get("executed_notional_usdt", 0.0)))
        elif action == "LIQUIDATION":
            turnover += abs(
                float(entry.get("price", 0.0))
                * float(entry.get("units", 0.0))
            )
    return float(turnover)


def _paper_snapshot(
    row: pd.Series,
    broker: PaperBroker,
    *,
    cumulative_turnover_usdt: float,
    symbol: str,
) -> dict[str, float | int | str]:
    """Capture one close-of-bar paper state for canonical performance analysis."""

    equity = max(float(broker.state.equity), 1e-12)
    close = float(row.get("close", 0.0))
    signed_notional = (
        float(broker.state.position) * close * float(broker.state.units)
    )
    return {
        "open_time": row.get("open_time", np.nan),
        "open_datetime": str(row.get("open_datetime", "")),
        "symbol": symbol,
        "equity": equity,
        "position": int(broker.state.position),
        "executed_notional_position": signed_notional / equity,
        "notional_exposure": abs(signed_notional) / equity,
        "cumulative_total_cost_usdt": _paper_total_cost_usdt(broker.state),
        "cumulative_turnover_usdt": float(cumulative_turnover_usdt),
    }


def _paper_performance_detail(
    snapshots: list[dict[str, float | int | str]],
    *,
    initial_balance: float,
    symbol: str,
) -> pd.DataFrame:
    """Convert paper equity snapshots to the canonical backtest detail schema."""

    detail = pd.DataFrame(snapshots)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open_datetime",
                "symbol",
                "equity",
                "strategy_return",
                "total_cost",
                "position",
                "executed_notional_position",
                "notional_exposure",
                "notional_turnover",
            ]
        )

    equity = pd.to_numeric(detail["equity"], errors="coerce").fillna(
        float(initial_balance)
    )
    prior_equity = equity.shift(1, fill_value=float(initial_balance)).clip(
        lower=1e-12
    )
    cumulative_cost = pd.to_numeric(
        detail["cumulative_total_cost_usdt"],
        errors="coerce",
    ).fillna(0.0)
    cumulative_turnover = pd.to_numeric(
        detail["cumulative_turnover_usdt"],
        errors="coerce",
    ).fillna(0.0)
    detail["strategy_return"] = equity / prior_equity - 1.0
    detail["total_cost"] = cumulative_cost.diff().fillna(cumulative_cost) / prior_equity
    detail["notional_turnover"] = (
        cumulative_turnover.diff().fillna(cumulative_turnover) / prior_equity
    )
    detail = detail.drop(
        columns=[
            "cumulative_total_cost_usdt",
            "cumulative_turnover_usdt",
        ]
    )
    detail.attrs["symbol"] = symbol
    return detail


def run_paper_replay(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    cfg: BacktestConfig,
    output_dir: str | Path,
    name: str,
) -> tuple[PaperState, Path]:
    replay_frame = frame.copy().reset_index(drop=True)
    availability = resolve_exchange_availability(replay_frame, cfg)
    replay_frame["execution_available"] = availability.available
    replay_frame["exchange_downtime_reason"] = availability.blocked_reason
    x, _ = feature_only_matrix(replay_frame, bundle.feature_columns)
    prob = bundle.predict_up_probability(x)
    broker = PaperBroker(cfg)
    stop_distances = stop_distance_series(replay_frame, cfg)
    symbol, _, interval = name.partition("_")
    snapshots: list[dict[str, float | int | str]] = []
    cumulative_turnover_usdt = 0.0

    for idx, row in replay_frame.iterrows():
        desired = choose_position(float(prob[idx]), cfg)
        decision = StrategyDecision(
            target_direction=desired,
            target_exposure=cfg.max_position_fraction if desired else 0.0,
            stop_loss=float(stop_distances[idx]) if desired else None,
            take_profit=cfg.take_profit if desired else None,
            holding_period=None,
            reason_code="alpha_threshold_long" if desired > 0 else (
                "alpha_threshold_short" if desired < 0 else "no_alpha_signal"
            ),
        )
        ledger_start = len(broker.ledger)
        broker.step(
            row,
            decision,
            float(prob[idx]),
            risk_decision=evaluate_strategy_risk(decision, cfg),
        )
        cumulative_turnover_usdt += _ledger_turnover_usdt(
            broker.ledger[ledger_start:]
        )
        snapshots.append(
            _paper_snapshot(
                row,
                broker,
                cumulative_turnover_usdt=cumulative_turnover_usdt,
                symbol=symbol,
            )
        )

    if len(replay_frame):
        last = replay_frame.iloc[-1]
        ledger_start = len(broker.ledger)
        broker.close_all(
            float(last["close"]),
            str(last.get("open_datetime", last.get("open_time", ""))),
            row=last,
        )
        cumulative_turnover_usdt += _ledger_turnover_usdt(
            broker.ledger[ledger_start:]
        )
        snapshots[-1] = _paper_snapshot(
            last,
            broker,
            cumulative_turnover_usdt=cumulative_turnover_usdt,
            symbol=symbol,
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / f"{name}_paper_ledger.csv"
    summary_path = output / f"{name}_paper_summary.json"
    pd.DataFrame(broker.ledger).to_csv(ledger_path, index=False)
    performance_detail = _paper_performance_detail(
        snapshots,
        initial_balance=cfg.initial_balance,
        symbol=symbol,
    )
    performance = evaluate_backtest_performance(
        performance_detail,
        initial_balance=cfg.initial_balance,
    )
    close_pnl = np.asarray(
        [
            float(item.get("pnl", 0.0))
            for item in broker.ledger
            if item.get("action") in {"CLOSE", "LIQUIDATION"}
        ],
        dtype=float,
    )
    gains = close_pnl[close_pnl > 0.0]
    losses = close_pnl[close_pnl < 0.0]
    gross_profit = float(gains.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0.0
        else (999.0 if gross_profit > 0.0 else 0.0)
    )
    total_cost_drag = float(
        pd.to_numeric(
            performance_detail.get(
                "total_cost",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    notional_turnover = float(
        pd.to_numeric(
            performance_detail.get(
                "notional_turnover",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    average_exposure = float(
        pd.to_numeric(
            performance_detail.get(
                "notional_exposure",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .mean()
    ) if len(performance_detail) else 0.0
    metrics = {
        "total_return": performance.total_return,
        "max_drawdown": performance.max_drawdown,
        "annualized_return": performance.annualized_return,
        "sharpe_like": performance.sharpe_like,
        "sortino_ratio": performance.sortino_ratio,
        "calmar_ratio": performance.calmar_ratio,
        "profit_factor": float(profit_factor),
        "win_rate": float((close_pnl > 0.0).mean()) if len(close_pnl) else 0.0,
        "trades": int(broker.state.trades),
        "closed_trades": int(len(close_pnl)),
        "fee_ratio": performance.fee_ratio,
        "gross_return_before_cost": performance.gross_return_before_cost,
        "total_cost_drag": total_cost_drag,
        "notional_turnover": notional_turnover,
        "average_exposure": average_exposure,
        "execution_events": int(
            (
                pd.to_numeric(
                    performance_detail.get(
                        "notional_turnover",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                )
                .fillna(0.0)
                > 1e-15
            ).sum()
        ),
        "duration_days": performance.duration_days,
        "periods_per_year": performance.periods_per_year,
        "performance_by_year": performance.performance_by_year,
        "performance_by_month": performance.performance_by_month,
        "performance_by_symbol": performance.performance_by_symbol,
        "commission_fees_usdt": float(broker.state.commission_fees),
        "slippage_paid_usdt": float(broker.state.slippage_paid),
        "funding_net_cost_usdt": float(broker.state.funding_net_cost),
    }
    risk_reason_counts = (
        pd.Series(
            [row["risk_reason"] for row in broker.risk_history]
        )
        .value_counts()
        .to_dict()
    )
    summary_path.write_text(
        json.dumps(
            {
                "created_beijing": beijing_now_iso(),
                "symbol": symbol,
                "interval": interval,
                "model_name": bundle.model_name,
                "state": asdict(broker.state),
                "metrics": metrics,
                "ledger_path": str(ledger_path),
                "risk_reason_counts": risk_reason_counts,
                "latest_risk_decision": (
                    broker.risk_history[-1] if broker.risk_history else None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return broker.state, summary_path
