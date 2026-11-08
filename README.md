# VERDICT — the trading agent on trial (AgentX PS-02, paper trading only)

Every trade stands trial before an order exists:
`Evidence -> Strategist -> Prosecutor vs Defender -> Judge -> Risk Engine (plain code) -> Paper Broker`.
Real data, virtual money (Rs 1,00,000), zero real orders.

## What is implemented
- **Risk engine** (no LLM): 1% risk sizing, mandatory stop, 3% daily loss lock, max 5 positions, 20% per-stock cap,
  cash check, market hours (IST), dead-man switch (24h check-in), stale-data block.
- **Paper broker**: market/limit orders, slippage, Indian delivery charges, stop/target monitoring driven by the latest DEMO/Angel One stream tick (with a REST quote fallback before the first tick), SQLite persistence,
  every order stamped `PAPER`. Documented fill model; fills are simulated, not exchange executions.
- **Kill switch**: persisted halt, cancels pending, flattens positions, stays halted if flatten fails.
- **Backtester**: momentum and MA-crossover vs buy-and-hold, walk-forward out-of-sample, next-bar-open execution,
  costs included; a strategy only trades if it beats buy-and-hold out of sample on Sharpe.
- **Gemini agents** (set `GEMINI_API_KEY`): Strategist may veto, Prosecutor/Defender argue, Judge rules. Output is schema- and evidence-ID-validated; bad output falls back to rule-based per role and is journaled. The LLM never sets prices or size.
- **Memory**: each trial reads the agent's own closed-trade record in that symbol from SQLite and cites it as evidence.
- **Trial pipeline**: evidence IDs, four agents with Pydantic outputs, plain-language journal, "I am wrong if...",
  process score that ignores P&L.
- **News evidence**: Google News RSS in live mode (reports UNAVAILABLE on failure, never fakes), labelled DEMO headlines in demo mode.
- **Dashboard** at `http://127.0.0.1:8000/` (no Node needed): courtroom trial, journal, positions, strategy vs buy-and-hold chart, live market-tick panel, STOP button.
- **Market WebSocket** at `/ws/market`: broadcasts labelled synthetic DEMO ticks or Angel One SmartAPI WebSocket ticks to connected local dashboards. Status is available at `/api/market/stream/status`.
- **FastAPI** endpoints for health, auth status, historical/indicators, trial, portfolio, orders, journal, backtest,
  risk logs, stop/resume/check-in.

## Not implemented yet (be honest with judges)
LangGraph orchestration (the trial runs as a plain Python pipeline), automatic instrument-master download, and NSE holiday calendar. Gemini agents are implemented and unit-tested against a mock transport only; they have not been run against the real API. The Angel One adapter and SmartAPI WebSocket are implemented as data-only integrations but remain **untested against a live account**. Live mode requires verified instrument tokens in `ANGELONE_INSTRUMENT_TOKENS`. Demo mode uses synthetic candles and labelled DEMO ticks.

## Hardening (loopholes closed)
- Stops/targets are checked every 5 s by a background monitor and before every trial (they previously only fired in tests).
- Daily-loss baseline resets each IST trading day; a daily-loss lock releases next day, a human kill switch never does.
- The paper broker itself refuses buys while halted (defence in depth behind the risk engine).
- Naive datetimes are treated as UTC, so market-hours/dead-man checks are correct on any computer's timezone.
- Stale or empty live candles are labelled STALE and block entries; positions without a price also block entries.
- Local-API guard: Host must be 127.0.0.1/localhost and every POST needs `X-Verdict-Client: 1`, so a website you visit cannot call the kill-switch/resume/check-in endpoints.
- Live-mode failures return HTTP 502 with the reason; they never fall back to demo data.
- News text is sanitised before reaching the LLM prompt.

If you ran an earlier version, delete `backend/verdict.db` once (the account table gained a column).

## Run
```
cd backend
pip install -r requirements.txt
python -m pytest -q          # includes paper-only safety and market WebSocket tests
uvicorn app.main:app --host 127.0.0.1 --port 8000   # docs at /docs
```
Windows: `.\start_verdict.ps1`. Config: copy `.env.example` to `.env`.


## v6.1 hardening update

- Stream prices are timestamped and only considered usable for valuation/risk checks for 15 seconds. Stale or missing prices are omitted; the backend no longer silently substitutes a REST quote for an open position's current risk price.
- Each accepted Angel One tick immediately schedules a serialized paper risk/exit-monitor pass. The periodic monitor remains as a safety net. DEMO ticks use the same monitor path.
- Pending LIMIT BUY bracket data (stop, target and thesis invalidation) is stored on the order record and survives backend restarts. Existing SQLite databases receive additive columns automatically.
- A pending limit order is reported as filled only when the simulated broker actually fills it.

The app remains **PAPER ONLY**. If a price is stale or unavailable, VERDICT deliberately does not invent an executable price to flatten a position; the account remains halted when a kill-switch flatten cannot be completed. Verify feed health and handle such positions manually in your paper workflow.
