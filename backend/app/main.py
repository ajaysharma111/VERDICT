from __future__ import annotations

import asyncio
import logging
import math
import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from .agents.llm import get_agents
from .agents.pipeline import run_trial
from .backtesting.engine import run_backtest
from .config import load_settings
from .database.models import JournalEntry, RiskLog
from .market.demo_data import WATCHLIST, demo_candles
from .market.indicators import indicator_frame
from .market.freshness import assess
from .market.news import get_news
from .market.provider import AngelOneProvider, DemoProvider, ProviderError
from .market.stream import MarketStreamHub
from .risk.engine import RiskEngine, RiskLimits, is_market_open
from .trading.monitor import monitor_tick
from .trading.paper_broker import PaperBroker

settings = load_settings()  # raises if the env tries to enable live trading
log = logging.getLogger("verdict")
broker = PaperBroker(settings.database_url, settings.initial_capital, settings.slippage_bps)
risk = RiskEngine(RiskLimits(settings.risk_per_trade, settings.daily_loss_limit, settings.max_open_positions,
                             settings.max_stock_exposure))
provider = DemoProvider() if settings.market_data_mode == "demo" else AngelOneProvider()
stream_hub = MarketStreamHub()
# Latest market-data ticks are the single source used by portfolio valuation and exit monitoring.
# Values are updated by DEMO ticks or the authenticated Angel One WebSocket callback.
latest_prices: dict[str, Decimal] = {}
latest_price_at: dict[str, datetime] = {}
MAX_TICK_AGE = timedelta(seconds=15)
_tick_lock: asyncio.Lock | None = None


def _fresh_prices(now: datetime | None = None) -> dict[str, Decimal]:
    """Only expose prices backed by a recent stream tick; never invent a risk price."""
    now = now or now_utc()
    return {symbol: price for symbol, price in latest_prices.items()
            if symbol in latest_price_at and timedelta(0) <= now - latest_price_at[symbol] <= MAX_TICK_AGE}


async def _run_market_monitor():
    """Serialize stream-triggered and periodic checks to prevent duplicate exit orders."""
    global _tick_lock
    if _tick_lock is None:
        _tick_lock = asyncio.Lock()
    async with _tick_lock:
        await asyncio.to_thread(monitor_tick, broker, risk, prices(), now_utc())


async def _demo_stream_loop():
    """Publish clearly labelled synthetic ticks in DEMO mode for dashboard verification."""
    phase = 0.0
    rng = random.Random(42)
    while True:
        phase += 0.35
        for symbol in WATCHLIST:
            base = float(provider.quote(symbol)["price"])
            # A small deterministic oscillation makes the UI visibly update without implying real prices.
            price = round(base * (1 + 0.0008 * math.sin(phase + sum(map(ord, symbol)))
                                  + rng.uniform(-0.00015, 0.00015)), 2)
            tick = {"symbol": symbol, "price": price, "provider": "DEMO", "freshness": "DEMO"}
            latest_prices[symbol] = Decimal(str(price))
            latest_price_at[symbol] = now_utc()
            await _run_market_monitor()
            await stream_hub.publish(tick)
        await asyncio.sleep(1)


async def _monitor_loop():
    while True:
        try:
            await _run_market_monitor()
        except Exception:  # a failed tick must never kill the monitor
            log.exception("monitor tick failed")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_app):
    tasks = [asyncio.create_task(_monitor_loop())]
    if settings.market_data_mode == "demo":
        stream_hub.provider_status = "DEMO"
        tasks.append(asyncio.create_task(_demo_stream_loop()))
    else:
        stream_hub.provider_status = "DISCONNECTED"
    yield
    for task in tasks:
        task.cancel()
    if isinstance(provider, AngelOneProvider):
        provider.stop_stream()


app = FastAPI(title="VERDICT - paper trading only", lifespan=lifespan)


@app.middleware("http")
async def local_only(request, call_next):
    """Defends the local API against other websites: DNS-rebinding (Host check) and cross-site POSTs such as a
    malicious page calling /api/system/resume (custom header forces a CORS preflight that we never grant)."""
    host = request.headers.get("host", "")
    name = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.split(":")[0]
    if name not in ("127.0.0.1", "localhost", "::1"):
        return JSONResponse({"detail": "Forbidden host"}, 403)
    if request.method not in ("GET", "HEAD") and request.headers.get("x-verdict-client") != "1":
        return JSONResponse({"detail": "Missing X-Verdict-Client header"}, 403)
    return await call_next(request)


@app.exception_handler(ProviderError)
async def provider_error(_request, exc):
    return JSONResponse({"detail": str(exc)}, 502)  # surfaced, never replaced by demo data


def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def prices() -> dict[str, Decimal]:
    """Return only recent stream prices. Missing/stale prices are omitted so exits cannot
    trigger from stale REST quotes or silently use a historical close as current data."""
    return _fresh_prices()


def handle_live_tick(tick: dict) -> None:
    """Accept a validated provider tick, immediately run the paper exit monitor, then publish."""
    symbol = str(tick.get("symbol", "")).upper()
    raw_price = tick.get("price")
    if symbol and raw_price is not None:
        try:
            value = Decimal(str(raw_price))
            if value.is_finite() and value > 0:
                latest_prices[symbol] = value
                latest_price_at[symbol] = now_utc()
                async def process():
                    try:
                        await _run_market_monitor()
                    except Exception:
                        log.exception("stream-triggered monitor failed for %s", symbol)
                    await stream_hub.publish(tick)
                asyncio.create_task(process())
                return
        except Exception:
            log.warning("Ignoring invalid market tick for %s", symbol)
    asyncio.create_task(stream_hub.publish(tick))


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/watchlist")
def watchlist():
    return WATCHLIST


@app.get("/api/health")
def health():
    return {"ok": True, "execution_mode": "PAPER", "data_mode": settings.market_data_mode.upper(),
            "market_open": is_market_open(datetime.now(timezone.utc))}


@app.get("/api/market/stream/status")
async def market_stream_status():
    return await stream_hub.status()


@app.websocket("/ws/market")
async def market_stream(ws: WebSocket):
    host = ws.headers.get("host", "")
    hostname = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.split(":")[0]
    origin = ws.headers.get("origin", "")
    if hostname not in ("127.0.0.1", "localhost", "::1"):
        await ws.close(code=1008)
        return
    if origin:
        from urllib.parse import urlparse
        origin_host = (urlparse(origin).hostname or "").lower()
        if origin_host not in ("127.0.0.1", "localhost", "::1"):
            await ws.close(code=1008)
            return
    await stream_hub.connect(ws)
    try:
        while True:
            # Keep the connection alive; clients may send a lightweight ping.
            await ws.receive_text()
    except WebSocketDisconnect:
        await stream_hub.disconnect(ws)
    except Exception:
        await stream_hub.disconnect(ws)


@app.get("/api/auth/angelone/status")
def angel_status():
    return provider.status()


@app.post("/api/auth/angelone/connect")
def angel_connect():
    if settings.market_data_mode == "demo":
        raise HTTPException(400, "Running in DEMO mode; set MARKET_DATA_MODE=live to use Angel One")
    try:
        provider.connect()
        loop = asyncio.get_running_loop()
        provider.start_stream(lambda tick: loop.call_soon_threadsafe(handle_live_tick, tick))
        stream_hub.provider_status = "ANGELONE_CONNECTING"
    except ProviderError as e:
        raise HTTPException(502, str(e))  # never silently falls back to demo data
    return {**provider.status(), "stream": "CONNECTING"}


@app.get("/api/market/historical/{symbol}")
def historical(symbol: str):
    df = provider.historical(symbol, "ONE_DAY", None, None)
    return {"source": df.attrs.get("source"), "candles": df.tail(250).reset_index().astype({"index": str}).to_dict("records")}


@app.get("/api/market/indicators/{symbol}")
def indicators(symbol: str):
    f = indicator_frame(provider.historical(symbol, "ONE_DAY", None, None)).dropna().tail(1)
    return f.reset_index().astype({"index": str}).to_dict("records")[0]


class TrialRequest(BaseModel):
    symbol: str = "RELIANCE"
    simulation_mode: bool = False


@app.post("/api/trial/start")
def trial_start(req: TrialRequest):
    df = provider.historical(req.symbol, "ONE_DAY", None, None)
    src = df.attrs.get("source", "UNKNOWN")
    now = now_utc()
    p = prices()
    monitor_tick(broker, risk, p, now)  # stops/targets, day roll, daily-loss lock before any new entry
    status, fresh = assess(df, now)
    all_priced = all(x.symbol in p for x in broker.positions())  # unpriced positions hide losses
    p[req.symbol] = Decimal(str(df.close.iloc[-1]))
    return run_trial(req.symbol, df, src, broker, risk, p, now, data_fresh=fresh and all_priced,
                     simulation_mode=req.simulation_mode or src == "DEMO",
                     news=get_news(req.symbol, settings.market_data_mode), agents=get_agents())


@app.get("/api/trial/history")
def trial_history():
    return journal()


@app.get("/api/trial/{trial_id}")
def trial_get(trial_id: int):
    return journal_entry(trial_id)


@app.get("/api/portfolio")
def portfolio():
    p, a = prices(), broker.account()
    return {"execution_mode": "PAPER", "cash": a.cash, "equity": broker.equity(p), "realized_pnl": a.realized_pnl,
            "price_freshness": {x.symbol: ("FRESH" if x.symbol in p else "STALE_OR_UNAVAILABLE")
                                for x in broker.positions()},
            "unrealized_pnl": broker.unrealized(p), "halted": a.halted, "halt_reason": a.halt_reason,
            "positions": [{"symbol": x.symbol, "quantity": x.quantity, "avg_price": x.avg_price, "stop": x.stop,
                           "target": x.target, "i_am_wrong_if": x.thesis_invalidation} for x in broker.positions()]}


@app.get("/api/orders")
def orders():
    return [{c.name: getattr(o, c.name) for c in o.__table__.columns} for o in broker.orders()]


@app.get("/api/journal")
def journal():
    with broker.Session() as s:
        return [{"id": j.id, "at": j.at, "symbol": j.symbol, "outcome": j.outcome, "summary": j.summary,
                 "process_score": j.process_score} for j in s.scalars(select(JournalEntry).order_by(JournalEntry.id.desc()))]


@app.get("/api/journal/{entry_id}")
def journal_entry(entry_id: int):
    with broker.Session() as s:
        j = s.get(JournalEntry, entry_id)
        if not j:
            raise HTTPException(404)
        return {"summary": j.summary, "transcript": j.transcript_json, "process_score": j.process_score}


@app.post("/api/backtest/run")
def backtest(symbol: str = "RELIANCE"):
    df = provider.historical(symbol, "ONE_DAY", None, None)
    return run_backtest(df, df.attrs.get("source", "UNKNOWN"))


@app.get("/api/risk/settings")
def risk_settings():
    return {"limits": risk.limits.__dict__, "execution_mode": "PAPER"}


@app.get("/api/risk/logs")
def risk_logs():
    with broker.Session() as s:
        return [{"at": r.at, "event": r.event, "symbol": r.symbol, "detail": r.detail}
                for r in s.scalars(select(RiskLog).order_by(RiskLog.id.desc()).limit(200))]


@app.post("/api/system/stop")
def stop():
    ok = broker.stop_trading(prices(), now_utc())
    return {"halted": True, "all_positions_flattened": ok}


@app.post("/api/system/resume")
def resume():
    broker.resume(now_utc())
    return {"halted": False}


@app.post("/api/system/checkin")
def checkin():
    broker.checkin(now_utc())
    return {"checked_in": True}
