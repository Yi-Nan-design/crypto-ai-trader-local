from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np


class ExchangeRuleConfig(Protocol):
    """Configuration fields required to normalize simulated exchange orders."""

    exchange_min_notional_usdt: float
    exchange_min_quantity: float
    exchange_max_quantity: float
    exchange_quantity_step: float
    exchange_price_tick_size: float


@dataclass(frozen=True)
class OrderNormalization:
    """Result of applying symbol-level quantity and notional constraints."""

    accepted: bool
    requested_notional_usdt: float
    normalized_notional_usdt: float
    normalized_quantity: float
    rounding_loss_usdt: float
    reason: str
    maximum_quantity_limited: bool = False


@dataclass(frozen=True)
class FuturesSymbolRules:
    """Public Binance Futures order filters used by local simulations."""

    symbol: str
    min_notional_usdt: float
    min_quantity: float
    quantity_step: float
    quantity_filter_type: str
    max_quantity: float = 0.0
    price_tick_size: float = 0.0
    source: str = "binance_futures_exchange_info"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "min_notional_usdt": self.min_notional_usdt,
            "min_quantity": self.min_quantity,
            "max_quantity": self.max_quantity,
            "quantity_step": self.quantity_step,
            "price_tick_size": self.price_tick_size,
            "quantity_filter_type": self.quantity_filter_type,
            "source": self.source,
        }


def parse_futures_symbol_rules(
    exchange_info: dict[str, Any],
    symbol: str,
) -> FuturesSymbolRules:
    """Parse market-order quantity and minimum-notional filters."""

    target = str(symbol).upper()
    symbols = exchange_info.get("symbols", [])
    item = next(
        (
            value
            for value in symbols
            if isinstance(value, dict) and str(value.get("symbol", "")).upper() == target
        ),
        None,
    )
    if item is None:
        raise ValueError(f"Symbol not found in Binance exchangeInfo: {target}")
    filters = {
        str(value.get("filterType", "")): value
        for value in item.get("filters", [])
        if isinstance(value, dict)
    }
    quantity_filter_type = "MARKET_LOT_SIZE"
    quantity_filter = filters.get(quantity_filter_type)
    if not quantity_filter or float(quantity_filter.get("stepSize", 0.0) or 0.0) <= 0.0:
        quantity_filter_type = "LOT_SIZE"
        quantity_filter = filters.get(quantity_filter_type, {})
    min_notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL") or {}
    min_notional = min_notional_filter.get(
        "notional",
        min_notional_filter.get("minNotional", 0.0),
    )
    price_filter = filters.get("PRICE_FILTER") or {}
    rules = FuturesSymbolRules(
        symbol=target,
        min_notional_usdt=float(min_notional or 0.0),
        min_quantity=float(quantity_filter.get("minQty", 0.0) or 0.0),
        max_quantity=float(quantity_filter.get("maxQty", 0.0) or 0.0),
        quantity_step=float(quantity_filter.get("stepSize", 0.0) or 0.0),
        price_tick_size=float(price_filter.get("tickSize", 0.0) or 0.0),
        quantity_filter_type=quantity_filter_type,
    )
    for name, value in (
        ("min_notional_usdt", rules.min_notional_usdt),
        ("min_quantity", rules.min_quantity),
        ("max_quantity", rules.max_quantity),
        ("quantity_step", rules.quantity_step),
        ("price_tick_size", rules.price_tick_size),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid Binance {name} for {target}: {value}")
    if (
        rules.max_quantity > 0.0
        and rules.min_quantity > rules.max_quantity
    ):
        raise ValueError(
            f"Invalid Binance quantity range for {target}: "
            f"{rules.min_quantity} > {rules.max_quantity}"
        )
    return rules


def exchange_rules_enabled(cfg: ExchangeRuleConfig) -> bool:
    """Return whether any configurable exchange order rule is active."""

    return any(
        float(value) > 0.0
        for value in (
            cfg.exchange_min_notional_usdt,
            cfg.exchange_min_quantity,
            getattr(cfg, "exchange_max_quantity", 0.0),
            cfg.exchange_quantity_step,
        )
    )


def normalize_order_notional_usdt(
    requested_notional_usdt: float,
    price: float,
    cfg: ExchangeRuleConfig,
) -> OrderNormalization:
    """Apply quantity-step, minimum-quantity, and minimum-notional rules."""

    requested = float(requested_notional_usdt)
    if abs(requested) <= 1e-15:
        return OrderNormalization(
            accepted=True,
            requested_notional_usdt=requested,
            normalized_notional_usdt=0.0,
            normalized_quantity=0.0,
            rounding_loss_usdt=0.0,
            reason="no_order",
        )
    if not exchange_rules_enabled(cfg):
        quantity = abs(requested) / float(price) if float(price) > 0.0 else 0.0
        return OrderNormalization(
            accepted=True,
            requested_notional_usdt=requested,
            normalized_notional_usdt=requested,
            normalized_quantity=quantity,
            rounding_loss_usdt=0.0,
            reason="exchange_rules_disabled",
        )
    if not np.isfinite(price) or float(price) <= 0.0:
        return OrderNormalization(
            accepted=False,
            requested_notional_usdt=requested,
            normalized_notional_usdt=0.0,
            normalized_quantity=0.0,
            rounding_loss_usdt=abs(requested),
            reason="invalid_order_price",
        )

    sign = 1.0 if requested > 0.0 else -1.0
    requested_quantity = abs(requested) / float(price)
    quantity_step = max(float(cfg.exchange_quantity_step), 0.0)
    if quantity_step > 0.0:
        step_count = math.floor((requested_quantity + quantity_step * 1e-12) / quantity_step)
        normalized_quantity = max(float(step_count) * quantity_step, 0.0)
    else:
        normalized_quantity = requested_quantity
    max_quantity = max(float(getattr(cfg, "exchange_max_quantity", 0.0)), 0.0)
    maximum_quantity_limited = (
        max_quantity > 0.0 and normalized_quantity > max_quantity + 1e-15
    )
    if maximum_quantity_limited:
        if quantity_step > 0.0:
            max_steps = math.floor((max_quantity + quantity_step * 1e-12) / quantity_step)
            normalized_quantity = max(float(max_steps) * quantity_step, 0.0)
        else:
            normalized_quantity = max_quantity
    normalized_abs_notional = normalized_quantity * float(price)
    rounding_loss = max(abs(requested) - normalized_abs_notional, 0.0)

    min_quantity = max(float(cfg.exchange_min_quantity), 0.0)
    if normalized_quantity + 1e-15 < min_quantity:
        return OrderNormalization(
            accepted=False,
            requested_notional_usdt=requested,
            normalized_notional_usdt=0.0,
            normalized_quantity=normalized_quantity,
            rounding_loss_usdt=abs(requested),
            reason="below_exchange_min_quantity",
            maximum_quantity_limited=maximum_quantity_limited,
        )
    min_notional = max(float(cfg.exchange_min_notional_usdt), 0.0)
    if normalized_abs_notional + 1e-12 < min_notional:
        return OrderNormalization(
            accepted=False,
            requested_notional_usdt=requested,
            normalized_notional_usdt=0.0,
            normalized_quantity=normalized_quantity,
            rounding_loss_usdt=abs(requested),
            reason="below_exchange_min_notional",
            maximum_quantity_limited=maximum_quantity_limited,
        )
    if normalized_quantity <= 0.0:
        return OrderNormalization(
            accepted=False,
            requested_notional_usdt=requested,
            normalized_notional_usdt=0.0,
            normalized_quantity=0.0,
            rounding_loss_usdt=abs(requested),
            reason="quantity_rounded_to_zero",
            maximum_quantity_limited=maximum_quantity_limited,
        )
    return OrderNormalization(
        accepted=True,
        requested_notional_usdt=requested,
        normalized_notional_usdt=sign * normalized_abs_notional,
        normalized_quantity=normalized_quantity,
        rounding_loss_usdt=rounding_loss,
        reason=(
            "quantity_capped_to_exchange_max"
            if maximum_quantity_limited
            else ("quantity_normalized" if rounding_loss > 1e-12 else "accepted")
        ),
        maximum_quantity_limited=maximum_quantity_limited,
    )
