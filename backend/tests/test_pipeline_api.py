
import json
from datetime import datetime, timezone
from decimal import Decimal as D

import pytest
from fastapi.testclient import TestClient

from app.agents import rule_based as rb
from app.agents.pipeline import run_trial
from app.agents.schemas import Argument
from app.market.demo_data import WATCHLIST, demo_candles
from app.risk.engine import RiskEngine
from app.trading.paper_broker import PaperBroker
from app.database.models import JournalEntry

NOW = datetime(2026, 11, 9, 5, 0)  # Mon 10:30 IST, naive UTC as used by the app


def test_trial_journals_every_symbol_and_enforces_risk():
    b, r = PaperBroker(), RiskEngine()
    b.checkin(NOW)
    outcomes = []

    for sym in WATCHLIST:
        df = demo_candles(sym)
        res = run_trial(
            sym, df, "DEMO", b, r,
            {sym: D(str(df.close.iloc[-1]))},
            NOW,
            simulation_mode=True,
        )
        outcomes.append(res["outcome"])
        t = res["transcript"]
        known = {e["id"] for e in t["evidence"]}

        for ids in (
            t["strategist"]["evidence_ids"],
            t["prosecutor"]["evidence_ids"],
            t["defender"]["evidence_ids"],
        ):
            assert set(ids) <= known

        assert 0 <= res["process_score"] <= 100

    assert len(b.orders()) == outcomes.count("EXECUTED")
    assert set(outcomes) <= {
        "EXECUTED",
        "NO_TRADE",
        "REJECTED_BY_JUDGE",
        "BLOCKED_BY_RISK",
    }
    assert len(b.positions()) <= 5


def test_trial_blocked_without_checkin_or_with_stale_data():
    for sym in WATCHLIST:
        df = demo_candles(sym)
        px = {sym: D(str(df.close.iloc[-1]))}
        b = PaperBroker()

        res = run_trial(
            sym, df, "DEMO", b, RiskEngine(), px, NOW,
            simulation_mode=True,
        )

        if (
            res["transcript"]["strategist"]["action"] == "BUY"
            and res["transcript"]["judge"]["decision"] != "REJECT"
        ):
            assert res["outcome"] == "BLOCKED_BY_RISK"
            assert b.orders() == []

            b.checkin(NOW)

            stale = run_trial(
                sym, df, "DEMO", b, RiskEngine(), px, NOW,
                data_fresh=False,
                simulation_mode=True,
            )

            assert stale["outcome"] == "BLOCKED_BY_RISK"
            assert b.orders() == []
            return

    pytest.skip("no demo symbol proposed a trade")


def test_judge_cannot_enlarge_and_rejects_close_calls():
    df = demo_candles("TCS")
    ev, ctx = rb.build_evidence("TCS", df, "DEMO")
    p = rb.strategist("TCS", ev, ctx)

    weak = Argument(
        role="DEFENDER",
        points=["x"],
        evidence_ids=["E1"],
        severity=0.3,
    )
    strong = Argument(
        role="PROSECUTOR",
        points=["y"],
        evidence_ids=["E1"],
        severity=0.3,
    )

    if p.action == "BUY":
        assert rb.judge(p, strong, weak).decision == "REJECT"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{tmp_path/'api.db'}",
    )
    monkeypatch.setenv("MARKET_DATA_MODE", "demo")

    import importlib
    import app.main as m

    importlib.reload(m)

    return TestClient(
        m.app,
        base_url="http://127.0.0.1:8000",
        headers={"X-Verdict-Client": "1"},
    )


def test_journal_api_reads_entry_and_returns_404_for_missing(client):
    import app.main as m

    transcript = {
        "evidence": [
            {"id": "E1", "description": "Test evidence"}
        ],
        "preliminary_assessment": {
            "decision": "REDUCE",
            "supporting_ids": ["E1"],
        },
    }

    # Insert a test journal entry into the isolated temporary database.
    with m.broker.Session.begin() as session:
        entry = JournalEntry(
            at=NOW,
            symbol="TCS",
            outcome="NO_TRADE",
            summary="Journal API test entry",
            transcript_json=json.dumps(transcript),
            process_score=95,
        )
        session.add(entry)
        session.flush()
        entry_id = entry.id

    # Test journal listing.
    listing = client.get("/api/journal")
    assert listing.status_code == 200

    rows = listing.json()
    saved = next(
        row for row in rows
        if row["id"] == entry_id
    )

    assert saved["symbol"] == "TCS"
    assert saved["outcome"] == "NO_TRADE"
    assert saved["summary"] == "Journal API test entry"
    assert saved["process_score"] == 95

    # Test journal detail and transcript retrieval.
    detail = client.get(f"/api/journal/{entry_id}")
    assert detail.status_code == 200

    body = detail.json()

    assert body["summary"] == "Journal API test entry"
    assert body["process_score"] == 95
    assert json.loads(body["transcript"]) == transcript

    # Test nonexistent journal entry.
    missing = client.get("/api/journal/999999")
    assert missing.status_code == 404


def test_api_flow_and_kill_switch(client):
    assert client.get("/api/health").json()["execution_mode"] == "PAPER"

    client.post("/api/system/checkin")

    for sym in WATCHLIST:
        client.post("/api/trial/start", json={"symbol": sym})

    pf = client.get("/api/portfolio").json()
    assert pf["execution_mode"] == "PAPER"
    assert len(client.get("/api/journal").json()) == len(WATCHLIST)

    # Seed fresh stream ticks so the kill switch has valid
    # executable paper prices to flatten at.
    import app.main as m

    for position in m.broker.positions():
        m.latest_prices[position.symbol] = position.avg_price
        m.latest_price_at[position.symbol] = m.now_utc()

    assert client.post("/api/system/stop").json()["halted"] is True

    after = client.post(
        "/api/trial/start",
        json={"symbol": "RELIANCE"},
    ).json()

    assert after["outcome"] in (
        "BLOCKED_BY_RISK",
        "NO_TRADE",
        "REJECTED_BY_JUDGE",
    )

    assert client.get("/api/portfolio").json()["positions"] == []
    assert client.post(
        "/api/backtest/run?symbol=TCS"
    ).json()["out_of_sample"] is True

    assert client.post(
        "/api/auth/angelone/connect"
    ).status_code == 400


def test_demo_market_websocket_streams_ticks(tmp_path, monkeypatch):
    """The dashboard WebSocket should receive labelled synthetic ticks."""

    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{tmp_path/'ws.db'}",
    )
    monkeypatch.setenv("MARKET_DATA_MODE", "demo")

    import importlib
    import app.main as m

    importlib.reload(m)

    with TestClient(
        m.app,
        base_url="http://127.0.0.1:8000",
    ) as c:
        with c.websocket_connect(
            "/ws/market",
            headers={
                "host": "127.0.0.1:8000",
                "origin": "http://127.0.0.1:8000",
            },
        ) as ws:
            event = ws.receive_json()

            assert event["type"] == "tick"
            assert event["provider"] == "DEMO"
            assert event["freshness"] == "DEMO"
            assert event["symbol"] in WATCHLIST
            assert event["price"] > 0


def test_pipeline_voids_trial_when_preliminary_judge_cites_unknown_evidence():
    from app.agents.schemas import Verdict

    class InvalidPreliminaryAgents:
        backends = {
            "strategist": "test",
            "prosecutor": "test",
            "defender": "test",
            "judge": "test",
        }

        preliminary_assessment = {
            "decision": "REDUCE",
            "reasoning": "Test preliminary assessment",
            "supporting_ids": ["E999"],
            "contradicting_ids": [],
            "confidence": 0.7,
        }

        def strategist(self, symbol, ev, ctx):
            return rb.strategist(symbol, ev, ctx)

        def prosecutor(self, prop, ev, ctx):
            return rb.prosecutor(prop, ev, ctx)

        def defender(self, prop, ev, ctx):
            return rb.defender(prop, ev, ctx)

        def judge(self, prop, pro, dfn, ev):
            return Verdict(
                decision="APPROVE",
                multiplier=D("1"),
                reasoning="Test final verdict",
                supporting_ids=[ev[0].id],
                contradicting_ids=[],
                i_am_wrong_if="The stop-loss is breached",
            )

    symbol = "TCS"
    df = demo_candles(symbol)
    prices = {symbol: D(str(df.close.iloc[-1]))}

    broker = PaperBroker()
    broker.checkin(NOW)

    cash_before = broker.account().cash
    positions_before = [
        (p.symbol, p.quantity, p.avg_price)
        for p in broker.positions()
    ]

    result = run_trial(
        symbol,
        df,
        "DEMO",
        broker,
        RiskEngine(),
        prices,
        NOW,
        simulation_mode=True,
        agents=InvalidPreliminaryAgents(),
    )

    assert result["outcome"] == "NO_TRADE"
    assert result["process_score"] == 0

    assert (
        result["transcript"]["preliminary_assessment"]["supporting_ids"]
        == ["E999"]
    )

    assert "voided" in " ".join(
        result["transcript"]["risk"]["reasons"]
    ).lower()

    assert broker.orders() == []
    assert broker.account().cash == cash_before

    assert [
        (p.symbol, p.quantity, p.avg_price)
        for p in broker.positions()
    ] == positions_before
