import pytest
from datetime import datetime
from decimal import Decimal as D
from fastapi.testclient import TestClient

from app.main import app
from app.market.demo_data import demo_candles
from app.agents.pipeline import run_trial
from app.trading.paper_broker import PaperBroker
from app.risk.engine import RiskEngine

NOW = datetime(2026, 11, 9, 5, 0)

def test_trading_options_pipeline():
    broker = PaperBroker()
    risk = RiskEngine()
    broker.checkin(NOW)
    df = demo_candles("RELIANCE")
    prices = {"RELIANCE": D(str(df.close.iloc[-1]))}

    for tt in ["INTRADAY", "OPTIONS", "FUTURES", "LONG_TERM"]:
        res = run_trial(
            symbol="RELIANCE",
            df=df,
            source="DEMO",
            broker=broker,
            risk=risk,
            prices=prices,
            now=NOW,
            simulation_mode=True,
            trading_type=tt,
        )
        assert res["trading_type"] == tt
        assert res["outcome"] in ("EXECUTED", "REJECTED_BY_JUDGE", "NO_TRADE", "BLOCKED_BY_RISK")
        transcript = res["transcript"]
        assert transcript["trading_type"] == tt
        assert len(transcript["evidence"]) > 0

        # Verify evidence citations exist
        known = {e["id"] for e in transcript["evidence"]}
        for ids in (
            transcript["strategist"]["evidence_ids"],
            transcript["prosecutor"]["evidence_ids"],
            transcript["defender"]["evidence_ids"],
        ):
            assert set(ids) <= known


def test_api_trial_with_trading_type():
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        client.post("/api/system/checkin", headers={"X-Verdict-Client": "1"})
        for tt in ["INTRADAY", "OPTIONS", "FUTURES", "LONG_TERM"]:
            res = client.post(
                "/api/trial/start",
                json={"symbol": "RELIANCE", "trading_type": tt},
                headers={"X-Verdict-Client": "1"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["trading_type"] == tt
            assert "transcript" in data
            assert data["transcript"]["trading_type"] == tt
