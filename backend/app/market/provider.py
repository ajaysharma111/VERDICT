"""Market-data-only interface. It deliberately has NO order methods, and the Angel One adapter only ever
calls the SmartAPI endpoints for login, candles and quotes. Order placement is never wrapped or exposed.

STATUS: the Angel One adapter follows the public smartapi-python SDK (SmartConnect.generateSession,
getCandleData, ltpData) but has NOT been tested against a live account. Verify against
https://smartapi.angelone.in/docs before relying on it."""
from __future__ import annotations

import os
import json
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import httpx
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


class LiveMarketProvider(MarketDataProvider):
    """Real live market data for Indian NSE equities without requiring broker credentials.
    Fetches real-time quotes and historical daily candles directly from official NSE feeds via Yahoo Finance.
    """
    name = "LIVE"

    SYMBOL_ALIASES = {
        "TATAMOTORS": "TMPV",  # Tata Motors Passenger Vehicles post-demerger
    }

    def __init__(self):
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._client = httpx.Client(
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=12.0
        )

    def _resolve(self, symbol: str) -> str:
        s = symbol.upper().replace(".NS", "").replace(".BO", "")
        return self.SYMBOL_ALIASES.get(s, s)

    def _ticker(self, symbol: str) -> str:
        return f"{self._resolve(symbol)}.NS"

    def connect(self) -> None:
        pass

    def historical(self, symbol: str, interval: str = "ONE_DAY", start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        sym = self._resolve(symbol)
        ticker = self._ticker(sym)
        now_ts = time.time()
        # 5-minute cache for daily candles
        if sym in self._cache and (now_ts - self._cache[sym][0] < 300):
            return self._cache[sym][1].copy()

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2y"
        try:
            r = self._client.get(url)
            r.raise_for_status()
            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result or not result[0]:
                err = data.get("chart", {}).get("error", {}).get("description", "No data returned")
                raise ProviderError(f"Failed to fetch historical data for {symbol}: {err}")

            res = result[0]
            timestamps = res.get("timestamp", [])
            quote_data = res.get("indicators", {}).get("quote", [{}])[0]

            df = pd.DataFrame({
                "time": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote_data.get("open", []),
                "high": quote_data.get("high", []),
                "low": quote_data.get("low", []),
                "close": quote_data.get("close", []),
                "volume": quote_data.get("volume", []),
            }).dropna()

            if len(df) < 30:
                raise ProviderError(f"Insufficient historical data for {symbol}")

            df.index = df.pop("time").dt.tz_convert("Asia/Kolkata")
            df = df.round(2)
            df.attrs["source"] = "LIVE"
            df.attrs["symbol"] = symbol.upper()

            self._cache[sym] = (now_ts, df)
            return df.copy()
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Live historical data request failed for {symbol}: {e}") from e

    def quote(self, symbol: str) -> dict:
        sym = self._resolve(symbol)
        ticker = self._ticker(sym)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        try:
            r = self._client.get(url)
            r.raise_for_status()
            data = r.json()
            res = data["chart"]["result"][0]
            meta = res["meta"]
            price = meta.get("regularMarketPrice")
            prev_close = meta.get("chartPreviousClose") or meta.get("previousClose") or price

            if price is None:
                q = res.get("indicators", {}).get("quote", [{}])[0]
                closes = [c for c in q.get("close", []) if c is not None]
                price = closes[-1] if closes else prev_close

            p_flt = float(price)
            prev_flt = float(prev_close) if prev_close else p_flt
            change = round(p_flt - prev_flt, 2)
            change_pct = round((change / prev_flt) * 100, 2) if prev_flt else 0.0

            return {
                "symbol": symbol.upper(),
                "price": round(p_flt, 2),
                "freshness": "LIVE",
                "provider": "LIVE",
                "change": change,
                "change_pct": change_pct,
                "high": round(float(meta.get("regularMarketDayHigh", p_flt)), 2),
                "low": round(float(meta.get("regularMarketDayLow", p_flt)), 2),
                "prev_close": round(prev_flt, 2),
            }
        except Exception as e:
            raise ProviderError(f"Live quote failed for {symbol}: {e}") from e

    def status(self) -> dict:
        return {"provider": "LIVE", "freshness": "LIVE", "connected": True}


def create_provider(mode: str = "demo") -> MarketDataProvider:
    if mode.lower() == "demo":
        return DemoProvider()
    if os.getenv("LIVE_PROVIDER", "").lower() == "angelone":
        return AngelOneProvider()
    angel_keys = ("ANGELONE_API_KEY", "ANGELONE_CLIENT_CODE", "ANGELONE_PIN", "ANGELONE_TOTP_SECRET")
    if all(os.getenv(k, "").strip() for k in angel_keys):
        return AngelOneProvider()
    return LiveMarketProvider()


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
