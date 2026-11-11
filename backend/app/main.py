from __future__ import annotations

import asyncio
import logging
import math
import random
import sqlite3
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
from .market.provider import (AngelOneProvider, DemoProvider, LiveMarketProvider,
                            ProviderError, create_provider)
from .market.stream import MarketStreamHub
from .risk.engine import RiskEngine, RiskLimits, is_market_open
from .trading.monitor import monitor_tick
from .trading.passport import verify_passport
from .trading.paper_broker import PaperBroker

settings = load_settings()  # raises if the env tries to enable live trading
log = logging.getLogger("verdict")

# Separate database files for DEMO and LIVE paper wallets so trades/cash do not cross over
if ":memory:" in settings.database_url:
    demo_db_url = "sqlite:///:memory:"
    live_db_url = "sqlite:///:memory:"
elif settings.database_url.endswith(".db"):
    demo_db_url = settings.database_url.replace(".db", "_demo.db")
    live_db_url = settings.database_url.replace(".db", "_live.db")
else:
    demo_db_url = f"{settings.database_url}_demo"
    live_db_url = f"{settings.database_url}_live"

broker_demo = PaperBroker(demo_db_url, settings.initial_capital, settings.slippage_bps)
broker_live = PaperBroker(live_db_url, settings.initial_capital, settings.slippage_bps)

risk = RiskEngine(RiskLimits(settings.risk_per_trade, settings.daily_loss_limit, settings.max_open_positions,
                             settings.max_stock_exposure))
current_mode = settings.market_data_mode.lower()
broker = broker_live if current_mode == "live" else broker_demo

# Historical database: read-only; never used by the active paper broker.
legacy_db_path = None
if settings.database_url.startswith("sqlite:///") and ":memory:" not in settings.database_url:
    legacy_db_path = Path(settings.database_url[len("sqlite:///"):]).resolve()


def read_legacy_journal_entry(entry_id=None):
    """Read historical journal data without creating or modifying its database."""
    if legacy_db_path is None or not legacy_db_path.is_file():
        return None if entry_id is not None else []

    uri = legacy_db_path.as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if entry_id is None:
                rows = conn.execute(
                    """SELECT id, at, symbol, outcome, summary, process_score
                       FROM trading_journal ORDER BY id DESC"""
                ).fetchall()
                return [dict(row) for row in rows]
            return conn.execute(
                """SELECT id, at, symbol, outcome, summary,
                          transcript_json, process_score
                   FROM trading_journal WHERE id = ?""",
                (entry_id,),
            ).fetchone()
    except sqlite3.Error:
        log.exception("Could not read legacy journal database")
        raise HTTPException(500, "Historical journal could not be read")
provider = create_provider(current_mode)
stream_hub = MarketStreamHub()
# Latest market-data ticks are the single source used by portfolio valuation and exit monitoring.
# Values are updated by DEMO ticks or the live data loop / authenticated Angel One WebSocket callback.
latest_prices: dict[str, Decimal] = {}
latest_price_at: dict[str, datetime] = {}
MAX_TICK_AGE = timedelta(seconds=30)
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


async def _market_stream_loop():
    """Stream market ticks continuously in both DEMO and LIVE modes."""
    phase = 0.0
    rng = random.Random(42)
    while True:
        try:
            if current_mode == "demo":
                phase += 0.35
                for symbol in WATCHLIST:
                    base = float(provider.quote(symbol)["price"])
                    price = round(base * (1 + 0.0008 * math.sin(phase + sum(map(ord, symbol)))
                                          + rng.uniform(-0.00015, 0.00015)), 2)
                    chg = round(price - base, 2)
                    chg_pct = round((chg / base) * 100, 2) if base else 0.0
                    tick = {
                        "symbol": symbol,
                        "price": price,
                        "provider": "DEMO",
                        "freshness": "DEMO",
                        "change": chg,
                        "change_pct": chg_pct,
                        "high": round(base * 1.01, 2),
                        "low": round(base * 0.99, 2),
                    }
                    latest_prices[symbol] = Decimal(str(price))
                    latest_price_at[symbol] = now_utc()
                    await _run_market_monitor()
                    await stream_hub.publish(tick)
                await asyncio.sleep(1)
            elif isinstance(provider, LiveMarketProvider):
                for symbol in WATCHLIST:
                    try:
                        q = await asyncio.to_thread(provider.quote, symbol)
                        price = Decimal(str(q["price"]))
                        latest_prices[symbol] = price
                        latest_price_at[symbol] = now_utc()
                        tick = {
                            "symbol": symbol,
                            "price": float(price),
                            "provider": "LIVE",
                            "freshness": "LIVE",
                            "change": q.get("change", 0.0),
                            "change_pct": q.get("change_pct", 0.0),
                            "high": q.get("high"),
                            "low": q.get("low"),
                            "prev_close": q.get("prev_close"),
                        }
                        await _run_market_monitor()
                        await stream_hub.publish(tick)
                    except Exception as e:
                        log.debug("Live quote fetch error for %s: %s", symbol, e)
                await asyncio.sleep(3)
            else:
                await asyncio.sleep(2)
        except Exception:
            log.exception("Stream loop error")
            await asyncio.sleep(2)


async def _monitor_loop():
    while True:
        try:
            await _run_market_monitor()
        except Exception:  # a failed tick must never kill the monitor
            log.exception("monitor tick failed")
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_app):
    tasks = [asyncio.create_task(_monitor_loop()), asyncio.create_task(_market_stream_loop())]
    if current_mode == "demo":
        stream_hub.provider_status = "DEMO"
    elif isinstance(provider, LiveMarketProvider):
        stream_hub.provider_status = "LIVE"
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
    return {"ok": True, "execution_mode": "PAPER", "data_mode": current_mode.upper(),
            "market_open": is_market_open(datetime.now(timezone.utc))}


class MarketModeRequest(BaseModel):
    mode: str


@app.get("/api/market/mode")
def get_market_mode():
    return {"mode": current_mode, "provider": provider.name}


@app.post("/api/market/mode")
def set_market_mode(req: MarketModeRequest):
    global current_mode, provider, broker
    m = req.mode.lower().strip()
    if m not in ("demo", "live"):
        raise HTTPException(400, "Mode must be 'demo' or 'live'")

    new_provider = create_provider(m)
    current_mode = m
    provider = new_provider
    broker = broker_live if m == "live" else broker_demo

    # Clear cached prices so the newly selected mode cannot reuse stale ticks.
    latest_prices.clear()
    latest_price_at.clear()

    stream_hub.provider_status = "DEMO" if m == "demo" else ("LIVE" if isinstance(provider, LiveMarketProvider) else "DISCONNECTED")
    return {"mode": current_mode, "provider": provider.name, "execution_mode": "PAPER"}


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
    global provider
    if current_mode == "demo":
        raise HTTPException(400, "Running in DEMO mode; set MARKET_DATA_MODE=live to use Angel One")
    provider = AngelOneProvider()
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
    df = provider.historical(symbol, "ONE_DAY", None, None).copy()
    df.index.name = "index"
    return {"source": df.attrs.get("source"), "candles": df.tail(250).reset_index().astype({"index": str}).to_dict("records")}


@app.get("/api/market/indicators/{symbol}")
def indicators(symbol: str):
    f = indicator_frame(provider.historical(symbol, "ONE_DAY", None, None)).dropna().tail(1).copy()
    f.index.name = "index"
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
                     news=get_news(req.symbol, current_mode), agents=get_agents())


@app.get("/api/trial/history")
def trial_history():
    return journal()


@app.get("/api/trial/{trial_id}")
def trial_get(trial_id: int):
    return journal_entry(trial_id)


@app.get("/api/portfolio")
def portfolio():
    p, a = prices(), broker.account()
    pos = broker.positions()
    return {
        "execution_mode": "PAPER",
        "cash": a.cash,
        "equity": broker.equity(p),
        "realized_pnl": a.realized_pnl,
        "price_freshness": {x.symbol: ("FRESH" if x.symbol in p else "STALE_OR_UNAVAILABLE")
                            for x in pos},
        "unrealized_pnl": broker.unrealized(p),
        "halted": a.halted,
        "halt_reason": a.halt_reason,
        "positions": [
            {
                "symbol": x.symbol,
                "quantity": x.quantity,
                "avg_price": x.avg_price,
                "current_price": p.get(x.symbol, x.avg_price),
                "unrealized_pnl": (p.get(x.symbol, x.avg_price) - x.avg_price) * x.quantity,
                "unrealized_pnl_pct": round(float((p.get(x.symbol, x.avg_price) - x.avg_price) / x.avg_price) * 100, 2) if x.avg_price > 0 else 0.0,
                "stop": x.stop,
                "target": x.target,
                "i_am_wrong_if": x.thesis_invalidation,
            }
            for x in pos
        ],
    }


@app.post("/api/wallet/reset")
def reset_wallet():
    reset_balance = broker.reset_wallet()
    latest_prices.clear()
    latest_price_at.clear()
    return {
        "success": True,
        "mode": current_mode.upper(),
        "execution_mode": "PAPER",
        "cash": reset_balance,
        "message": f"{current_mode.upper()} paper wallet reset successfully",
    }


@app.get("/api/orders")
def orders():
    return [{c.name: getattr(o, c.name) for c in o.__table__.columns} for o in broker.orders()]


@app.get("/api/journal")
def journal():
    with broker.Session() as s:
        active = [
            {
                "id": j.id, "at": j.at, "symbol": j.symbol,
                "outcome": j.outcome, "summary": j.summary,
                "process_score": j.process_score, "source": "active",
            }
            for j in s.scalars(
                select(JournalEntry).order_by(JournalEntry.id.desc())
            )
        ]

    legacy = [
        {**row, "source": "legacy"}
        for row in read_legacy_journal_entry()
    ]
    return sorted(
        active + legacy,
        key=lambda item: str(item["at"]),
        reverse=True,
    )


@app.get("/api/journal/{entry_id}")
def journal_entry(entry_id: int, source: str = "active"):
    if source == "legacy":
        j = read_legacy_journal_entry(entry_id)
        if j is None:
            raise HTTPException(404, "Historical journal entry not found")
        summary = j["summary"]
        transcript_json = j["transcript_json"]
        process_score = j["process_score"]
    elif source == "active":
        with broker.Session() as s:
            j = s.get(JournalEntry, entry_id)
            if not j:
                raise HTTPException(404, "Journal entry not found")
            summary = j.summary
            transcript_json = j.transcript_json
            process_score = j.process_score
    else:
        raise HTTPException(400, "Invalid journal source")

    try:
        transcript = __import__("json").loads(transcript_json)
    except (TypeError, ValueError):
        transcript = {}

    passport = transcript.get("trade_passport")
    integrity = "NOT_AVAILABLE"
    if isinstance(passport, dict):
        integrity = "VERIFIED" if verify_passport(passport) else "FAILED"

    return {
        "summary": summary,
        "transcript": transcript_json,
        "process_score": process_score,
        "passport_integrity": integrity,
        "passport": passport,
        "source": source,
    }


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
