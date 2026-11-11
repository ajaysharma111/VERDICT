# VERDICT — The Trading Agent on Trial

**Hackathon project · AgentX PS-02 · PAPER TRADING ONLY**

VERDICT is an evidence-first, multi-agent trading decision system for Indian equities. Before a simulated order can exist, every proposed trade goes through a structured courtroom process:

`Market Evidence → Strategist → Prosecutor vs. Defender → Judge → Deterministic Risk Engine → Paper Broker`

The agents debate the evidence; ordinary Python code—not an LLM—controls position sizing, risk limits, order eligibility, and simulated execution. VERDICT uses virtual money only and does not place real exchange orders.

---

## The problem

Many AI trading demos jump from a model-generated signal straight to an order. That makes it difficult to understand why a trade was proposed, what evidence challenged it, which risks were considered, and why the final decision was accepted or rejected.

## Our approach

VERDICT treats each proposed trade as a decision that must stand up to scrutiny:

- **Strategist** proposes a trade thesis and identifies supporting evidence.
- **Prosecutor** challenges the thesis with counter-evidence, event risk, weak signals, and uncertainty.
- **Defender** responds to the objections using the available evidence.
- **Judge** issues a structured verdict.
- **Deterministic risk engine** independently checks whether the trade is permitted and calculates the allowed size.
- **Paper broker** simulates eligible orders and records fills, costs, and risk levels.

A verdict is not a command to trade. The risk engine can reject a trade even when the Judge approves it. A `NO_TRADE` outcome is a valid result.

## Key features

### 1. Evidence-first agent trial
- Four agent roles with structured Pydantic outputs.
- Evidence IDs connect claims to the information used in a trial.
- The journal records the decision and the reasoning, including an “I am wrong if…” invalidation statement.
- A process score evaluates decision-process quality and does not use profit or loss as its score.

### 2. Independent risk engine
The risk engine is plain Python and does not delegate safety decisions to an LLM. Configured controls include:

- 1% risk-per-trade sizing.
- Mandatory stop-loss and position sizing checks.
- 3% daily-loss lock.
- Maximum 5 open positions.
- 20% per-stock exposure cap.
- Available-cash and market-hours checks.
- Stale-data protection.
- 24-hour human check-in / dead-man control.
- Persisted kill switch.

The paper broker also refuses new buys while halted. The kill switch cancels pending orders and attempts to flatten positions; if flattening cannot be completed safely, the account remains halted.

### 3. Paper broker and audit trail
- Virtual starting capital: **₹1,00,000**.
- Simulated market and limit orders, slippage, and Indian delivery charges.
- SQLite persistence for paper accounts, orders, positions, risk logs, and journal entries.
- Every order is marked `PAPER`; fills are simulated and are not exchange executions.
- Trade Passport data records the decision and execution context for successful simulated fills. Its SHA-256 digest can detect snapshot changes, but it is not an external digital signature or proof against someone rewriting both the data and digest.

### 4. Strategy evaluation
- Momentum and moving-average-crossover strategies.
- Walk-forward out-of-sample evaluation against buy-and-hold.
- Next-bar-open simulated execution with costs included.
- A strategy is eligible only when it beats buy-and-hold out of sample on the configured Sharpe comparison.

### 5. Market data modes
- **DEMO:** synthetic candles and explicitly labelled DEMO ticks.
- **LIVE market-data mode:** Angel One SmartAPI data adapter and WebSocket integration, with verified instrument tokens configured by the user.
- Google News RSS is used for live-mode news evidence; unavailable news is labelled unavailable rather than fabricated.
- The LIVE label refers to the market-data source only. **Execution remains paper-only in every mode.**

### 6. Journal and historical record recovery
The dashboard combines:
- Current active paper-wallet journal entries.
- Historical entries from the original `verdict.db`, displayed with a **LEGACY · READ ONLY** label.

Legacy records are read directly from the old SQLite database using SQLite read-only mode. They are not copied into the active wallet. Journal detail requests include the record source, so duplicate IDs across the two databases open the correct transcript.

### 7. Local dashboard and API
- Browser dashboard for trials, journal, positions, backtesting, market ticks, and risk controls.
- Market WebSocket at `/ws/market`.
- Stream status at `/api/market/stream/status`.
- FastAPI endpoints for health, market data, trials, portfolio, orders, journal, backtesting, risk logs, stop/resume, and check-in.
- API documentation at `/docs`.

---

## Architecture

```text
Market data + news
       |
       v
Evidence collection and validation
       |
       v
Strategist
       |
       +------> Prosecutor
       |             |
       +------> Defender
                     |
                     v
                   Judge
                     |
                     v
        Deterministic Python risk engine
                     |
             permitted? / rejected?
                     |
                     v
                Paper broker
                     |
          SQLite journal and records
                     |
                     v
          Dashboard + Trade Passport
```

**Technology:** Python, FastAPI, SQLAlchemy, SQLite, Pydantic, HTML/CSS/JavaScript, WebSocket, pytest, Gemini API integration (optional), and Angel One SmartAPI data integration.

---

## Run locally

Requirements: Python 3.11+ recommended. Use a virtual environment.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# Windows PowerShell: .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
cp .env.example .env             # configure only the integrations you intend to use
python -m pytest -q
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- Dashboard: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs

On Windows, the repository also includes `start_verdict.ps1`.

### Optional integrations

- Set `GEMINI_API_KEY` to configure Gemini agent calls.
- Configure Angel One credentials and verified instrument tokens as described in `.env.example` before selecting LIVE market-data mode.

Do not put API keys or account credentials in source control. Keep the app bound to localhost for the intended local demo.

---

## Suggested hackathon demo flow

1. Open the dashboard and point out the **PAPER TRADING ONLY** execution label.
2. Show the selected market-data mode and market-feed status.
3. Run a trial for a stock.
4. Walk through the evidence, Strategist thesis, Prosecutor objections, Defender response, and Judge verdict.
5. Show that the deterministic risk engine independently approves, reduces, or rejects the proposed trade.
6. If an order is simulated, inspect its paper fill, position, and Trade Passport. If the result is `NO_TRADE`, explain why rejecting a weak setup is an intentional outcome.
7. Open the journal and compare active entries with the **LEGACY · READ ONLY** historical records.
8. Show the backtest comparison and the risk controls.

Do not promise a fill during a demo: the system may correctly decide not to trade.

---

## Safety and security

- The project is **paper trading only**. No real-money order execution is implemented.
- LLM output is schema- and evidence-ID-validated. Invalid output falls back to rule-based role output and is journaled.
- The LLM does not set prices, order size, or override risk controls.
- Local API protection requires a localhost/127.0.0.1 Host and `X-Verdict-Client: 1` on POST requests.
- Live-mode data failures return an error rather than silently switching to demo data.
- News text is sanitised before being included in an LLM prompt.
- Stale or unavailable prices are not invented to force an exit. If a kill-switch flatten cannot be completed safely, the account remains halted.

---

## Verification and known limitations

The current local regression run completed with **62 tests passed**. It emitted one Starlette deprecation warning concerning its use of `httpx` through `starlette.testclient`; this was a warning, not a test failure.

Be precise about the current implementation:

- The trial orchestration is a plain Python pipeline; **LangGraph orchestration is not implemented**.
- Gemini agents have been unit-tested with a mock transport; real Gemini API calls have not been verified in this project run.
- The Angel One adapter and SmartAPI WebSocket are implemented as data integrations but have not been verified against a live account.
- Live mode requires valid instrument tokens in `ANGELONE_INSTRUMENT_TOKENS`.
- Automatic instrument-master download and an NSE holiday calendar are not implemented.
- Demo market data is synthetic and labelled as DEMO.
- Backtest results are simulations, not evidence of future performance.

These limitations are stated deliberately so judges can distinguish implemented functionality from integrations that still need external verification.

---

## Project status

VERDICT is a hackathon prototype demonstrating an auditable, multi-agent decision workflow with deterministic risk enforcement and paper-only execution. It is intended for demonstration and experimentation, not as financial advice or a production trading service.

**Repository:** https://github.com/ajaysharma111/VERDICT
