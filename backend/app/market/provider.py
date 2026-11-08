"""Market-data-only interface. It deliberately has NO order methods, and the Angel One adapter only ever
calls the SmartAPI endpoints for login, candles and quotes. Order placement is never wrapped or exposed.

STATUS: the Angel One adapter follows the public smartapi-python SDK (SmartConnect.generateSession,
getCandleData, ltpData) but has NOT been tested against a live account. Verify against
https://smartapi.angelone.in/docs before relying on it."""
from __future__ import annotations

import os
import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import pandas as pd

from .demo_data import demo_candles

Freshness = Literal["LIVE", "STALE", "DISCONNECTED", "DEMO", "DELAYED"]


class ProviderError(RuntimeError):
    pass


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def historical(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame: ...
    @abstractmethod
    def quote(self, symbol: str) -> dict: ...
    @abstractmethod
    def status(self) -> dict: ...


class DemoProvider(MarketDataProvider):
    name = "DEMO"

    def connect(self) -> None:
        pass

    def historical(self, symbol, interval="ONE_DAY", start=None, end=None) -> pd.DataFrame:
        return demo_candles(symbol)

    def quote(self, symbol: str) -> dict:
        df = demo_candles(symbol)
        return {"symbol": symbol, "price": float(df.close.iloc[-1]), "freshness": "DEMO", "provider": "DEMO"}

    def status(self) -> dict:
        return {"provider": "DEMO", "freshness": "DEMO", "connected": True}


class AngelOneProvider(MarketDataProvider):
    name = "ANGELONE"

    def __init__(self, tokens: dict[str, str] | None = None):
        # symbol -> instrument token. Must come from Angel One's official instrument master; never guessed.
        self.tokens = tokens or {}
        self._api = None
        self._error = ""
        self._feed_token = ""
        self._jwt_token = ""
        self._stream = None
        self._stream_thread = None

    def connect(self) -> None:
        import pyotp
        try:
            from SmartApi import SmartConnect
        except ImportError as e:
            raise ProviderError("smartapi-python is not installed") from e
        key, code, pin, secret = (os.getenv(k, "") for k in
                                  ("ANGELONE_API_KEY", "ANGELONE_CLIENT_CODE", "ANGELONE_PIN", "ANGELONE_TOTP_SECRET"))
        if not all([key, code, pin, secret]):
            self._error = "Angel One credentials missing in .env"
            raise ProviderError(self._error)
        api = SmartConnect(api_key=key)
        res = api.generateSession(code, pin, pyotp.TOTP(secret).now())
        if not res or not res.get("status"):
            self._error = f"Authentication failed: {(res or {}).get('message', 'unknown')} (check credentials, TOTP and any static-IP requirement)"
            raise ProviderError(self._error)
        self._api, self._error = api, ""
        self._feed_token = str((res.get("data") or {}).get("feedToken") or "")
        self._jwt_token = str((res.get("data") or {}).get("jwtToken") or "")
        raw_tokens = os.getenv("ANGELONE_INSTRUMENT_TOKENS", "{}")
        try:
            parsed = json.loads(raw_tokens)
            if isinstance(parsed, dict):
                self.tokens = {str(k).upper(): str(v) for k, v in parsed.items()}
        except (TypeError, ValueError):
            raise ProviderError("ANGELONE_INSTRUMENT_TOKENS must be a JSON object of symbol-to-token mappings")

    def start_stream(self, on_tick) -> None:
        """Start the official SmartAPI market-data WebSocket in a daemon thread."""
        if not self._api or not self._feed_token:
            raise ProviderError("Connect Angel One before starting its market-data stream")
        if self._stream_thread and self._stream_thread.is_alive():
            return
        if not self.tokens:
            raise ProviderError("No instrument tokens configured; set ANGELONE_INSTRUMENT_TOKENS in .env")
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        except ImportError as exc:
            raise ProviderError("smartapi-python with SmartWebSocketV2 is not installed") from exc
        api_key = os.getenv("ANGELONE_API_KEY", "")
        client_code = os.getenv("ANGELONE_CLIENT_CODE", "")
        jwt_token = str(getattr(self._api, "access_token", "") or "")
        if not jwt_token:
            # SmartConnect versions differ; retain the token from generateSession when available.
            jwt_token = self._jwt_token
        socket = SmartWebSocketV2(jwt_token, api_key, client_code, self._feed_token)
        self._stream = socket

        def opened(_ws):
            grouped = [{"exchangeType": 1, "tokens": list(self.tokens.values())}]
            socket.subscribe("verdict001", 1, grouped)  # LTP mode

        def received(_ws, message):
            try:
                if isinstance(message, dict):
                    token = str(message.get("token") or message.get("symbolToken") or "")
                    raw = message.get("last_traded_price", message.get("ltp"))
                    if token and raw is not None:
                        symbol = next((k for k, v in self.tokens.items() if v == token), token)
                        # SmartAPI market-data packets express last_traded_price in paise.
                        price = float(raw) / 100.0
                        on_tick({"symbol": symbol, "price": price, "provider": "ANGELONE", "freshness": "LIVE"})
            except Exception:
                self._error = "Could not decode an Angel One market tick"

        def failed(_ws, error):
            self._error = f"Angel One stream error: {error}"

        socket.on_open = opened
        socket.on_data = received
        socket.on_error = failed
        self._stream_thread = threading.Thread(target=socket.connect, name="angelone-market-ws", daemon=True)
        self._stream_thread.start()

    def stop_stream(self) -> None:
        if self._stream:
            try:
                self._stream.close_connection()
            except Exception:
                pass
        self._stream = None
        self._stream_thread = None

    def _token(self, symbol: str) -> str:
        if symbol not in self.tokens:
            raise ProviderError(f"No instrument token for {symbol}; load the official instrument master first")
        return self.tokens[symbol]

    def historical(self, symbol, interval="ONE_DAY", start=None, end=None) -> pd.DataFrame:
        if not self._api:
            raise ProviderError("Not connected")
        from datetime import datetime, timedelta
        end = end or datetime.now()
        start = start or (end - timedelta(days=365))
        res = self._api.getCandleData({"exchange": "NSE", "symboltoken": self._token(symbol), "interval": interval,
                                       "fromdate": start.strftime("%Y-%m-%d %H:%M"), "todate": end.strftime("%Y-%m-%d %H:%M")})
        if not res or not res.get("status"):
            raise ProviderError(f"Historical request failed: {(res or {}).get('message')}")
        df = pd.DataFrame(res["data"], columns=["time", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("time"))
        df.attrs["source"] = "ANGELONE"
        return df

    def quote(self, symbol: str) -> dict:
        if not self._api:
            raise ProviderError("Not connected")
        res = self._api.ltpData("NSE", symbol + "-EQ", self._token(symbol))
        if not res or not res.get("status"):
            raise ProviderError(f"Quote failed: {(res or {}).get('message')}")
        return {"symbol": symbol, "price": float(res["data"]["ltp"]), "freshness": "LIVE", "provider": "ANGELONE"}

    def status(self) -> dict:
        return {"provider": "ANGELONE", "connected": self._api is not None,
                "freshness": "LIVE" if self._api else "DISCONNECTED", "error": self._error}
