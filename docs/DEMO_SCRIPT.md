# VERDICT — 3-minute demo script (AgentX, PS-02)

Judging: problem fit 25 · agent design 25 · working demo 20 · innovation 15 · presentation 15.
Record the screen at 1080p; keep the dashboard at http://127.0.0.1:8000/ with MARKET_DATA_MODE=demo (or live if verified).

| Time | Show | Say (≈) | Scores |
|---|---|---|---|
| 0:00–0:15 | Header: PAPER ONLY badge, DEMO/LIVE badge | "Retail traders lose money to noise, skipped rules and trades nobody can explain. VERDICT makes every trade stand trial first. Real data, virtual money, zero real orders." | Problem fit |
| 0:15–0:35 | Click check-in, then **Start trial** on a stock | "The agent plans: evidence from prices, indicators, news and its own track record, then a strategy that must beat buy-and-hold out of sample." | Agent design |
| 0:35–1:20 | Courtroom panels appear; click an evidence chip | "Strategist proposes. Prosecutor argues against, Defender for, Judge rules. Every claim cites an evidence ID — click it. The LLM can't invent numbers: bad output falls back and is logged." | Agent design, innovation |
| 1:20–1:45 | Risk engine strip + positions table | "The Judge can only shrink a trade. Plain code sizes it at 1% risk, attaches the stop, and enforces limits. The paper broker fills with slippage and Indian charges." | Reliability |
| 1:45–2:05 | Journal: expand an entry | "Plain-language journal with 'I am wrong if…'. Process score is graded on decision quality, not profit." | Innovation, presentation |
| 2:05–2:30 | Backtest chart + table | "Walk-forward, out-of-sample, costs included, compared with buy-and-hold. A strategy only trades if it beats the benchmark." | Problem fit |
| 2:30–2:50 | Click **STOP TRADING**, confirm | "Human kill switch: halts, cancels orders, closes paper positions, survives restart. Dead-man switch stops entries after 24h without a check-in." | Reliability |
| 2:50–3:00 | Header again | "AI proposes. Code disposes. Real data. Virtual money. Zero real trades." | Presentation |

Before recording: run `python -m pytest -q` and show it passing once in the README/video if time allows.
Be upfront about limits: demo data is synthetic unless live Angel One is verified; no real money anywhere.
