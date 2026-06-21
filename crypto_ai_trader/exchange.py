from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BinanceFuturesConfig:
    api_key: str
    api_secret: str
    testnet: bool = True
    dry_run: bool = True
    recv_window: int = 5000

    @property
    def base_url(self) -> str:
        return "https://demo-fapi.binance.com" if self.testnet else "https://fapi.binance.com"


class BinanceFuturesClient:
    """Small urllib-based client for USD-M futures.

    Live order methods are locked unless dry_run is false and the environment
    variable BINANCE_ENABLE_TRADING is set to I_UNDERSTAND_RISK.
    """

    def __init__(self, cfg: BinanceFuturesConfig):
        self.cfg = cfg

    @classmethod
    def from_env(cls, testnet: bool = True, dry_run: bool = True) -> "BinanceFuturesClient":
        key = os.environ.get("BINANCE_API_KEY", "")
        secret = os.environ.get("BINANCE_API_SECRET", "")
        return cls(BinanceFuturesConfig(api_key=key, api_secret=secret, testnet=testnet, dry_run=dry_run))

    def _signed_params(self, params: dict[str, Any]) -> str:
        payload = dict(params)
        payload["timestamp"] = int(time.time() * 1000)
        payload["recvWindow"] = self.cfg.recv_window
        query = urlencode(payload)
        signature = hmac.new(self.cfg.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={signature}"

    def request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = params or {}
        if signed:
            if not self.cfg.api_key or not self.cfg.api_secret:
                raise RuntimeError("Missing BINANCE_API_KEY or BINANCE_API_SECRET")
            query = self._signed_params(params)
        else:
            query = urlencode(params)

        url = f"{self.cfg.base_url}{path}"
        data = None
        if method.upper() in {"GET", "DELETE"} and query:
            url = f"{url}?{query}"
        elif query:
            data = query.encode()

        request = Request(url, data=data, method=method.upper(), headers={"X-MBX-APIKEY": self.cfg.api_key})
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def exchange_info(self) -> Any:
        return self.request("GET", "/fapi/v1/exchangeInfo")

    def change_margin_type(self, symbol: str, margin_type: str) -> Any:
        return self._guarded_trade_request(
            "POST",
            "/fapi/v1/marginType",
            {"symbol": symbol.upper(), "marginType": margin_type.upper()},
        )

    def change_leverage(self, symbol: str, leverage: int) -> Any:
        return self._guarded_trade_request(
            "POST",
            "/fapi/v1/leverage",
            {"symbol": symbol.upper(), "leverage": int(leverage)},
        )

    def new_order(self, symbol: str, side: str, order_type: str, quantity: float, **extra: Any) -> Any:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        params.update(extra)
        return self._guarded_trade_request("POST", "/fapi/v1/order", params)

    def _guarded_trade_request(self, method: str, path: str, params: dict[str, Any]) -> Any:
        if self.cfg.dry_run:
            return {"dry_run": True, "method": method, "path": path, "params": params}
        if os.environ.get("BINANCE_ENABLE_TRADING") != "I_UNDERSTAND_RISK":
            raise RuntimeError("Trading locked. Set BINANCE_ENABLE_TRADING=I_UNDERSTAND_RISK to enable.")
        return self.request(method, path, params=params, signed=True)

