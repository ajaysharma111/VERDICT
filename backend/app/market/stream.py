"""In-process market tick fan-out for the local VERDICT dashboard.

This stream is informational only. It never submits orders or changes paper positions.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket


class MarketStreamHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self.last_tick: dict[str, Any] | None = None
        self.provider_status = "STARTING"

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        if self.last_tick:
            await ws.send_json(self.last_tick)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def publish(self, tick: dict[str, Any]) -> None:
        event = {"type": "tick", "at": datetime.now(timezone.utc).isoformat(), **tick}
        self.last_tick = event
        if event.get("provider") == "ANGELONE":
            self.provider_status = "ANGELONE_CONNECTED"
        async with self._lock:
            clients = tuple(self._clients)
        stale: list[WebSocket] = []
        for client in clients:
            try:
                await client.send_json(event)
            except Exception:
                stale.append(client)
        if stale:
            async with self._lock:
                for client in stale:
                    self._clients.discard(client)

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            count = len(self._clients)
        return {"provider": self.provider_status, "clients": count, "last_tick": self.last_tick}
