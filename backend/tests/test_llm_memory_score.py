import json
from datetime import datetime
from decimal import Decimal as D

import httpx

from app.agents import rule_based as rb
from app.agents.llm import GeminiClient, LLMAgents, RuleAgents, get_agents
from app.agents.pipeline import run_trial, score_process, symbol_memory
from app.agents.schemas import Argument, Verdict
from app.market.demo_data import WATCHLIST, demo_candles
from app.risk.engine import RiskEngine
from app.trading.paper_broker import PaperBroker

NOW = datetime(2026, 11, 9, 5, 0)


def gemini(handler):
    return GeminiClient("k", transport=httpx.MockTransport(handler))


def reply(obj):
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]})


def buy_case():
    for s in WATCHLIST:
        df = demo_candles(s)
        ev, ctx = rb.build_evidence(s, df, "DEMO", None, None)
        p = rb.strategist(s, ev, ctx)
        if p.action == "BUY":
            return s, df, ev, ctx, p
    raise AssertionError("no BUY case")


def test_llm_agents_used_when_valid_and_key_not_in_body():
    s, df, ev, ctx, p = buy_case()
    seen = {"prompts": []}
    valid_id = ev[0].id
    other_id = ev[1].id

    def h(req):
        seen["key_in_header"] = req.headers.get("x-goog-api-key") == "k"
        seen["key_in_url"] = "k" == req.url.params.get("key")
        body = json.loads(req.content)
        prompt = body["contents"][0]["parts"][0]["text"]
        seen["prompts"].append(prompt)

        if "ROLE: STRATEGIST" in prompt:
            return reply({"action": "BUY", "rationale": "ok", "evidence_ids": [valid_id]})
        if "ROLE: PROSECUTOR" in prompt:
            return reply({"points": ["late entry"], "evidence_ids": [other_id], "severity": 0.2})
        if "ROLE: DEFENDER" in prompt:
            return reply({"points": ["trend"], "evidence_ids": [valid_id], "severity": 0.9})
        if "BLIND JUDGE" in prompt:
            return reply({
                "decision": "REDUCE",
                "reasoning": "Preliminary assessment",
                "supporting_ids": [valid_id],
                "contradicting_ids": [other_id],
                "confidence": 0.7,
            })
        if "FINAL DELIBERATION" in prompt:
            return reply({
                "decision": "APPROVE",
                "reasoning": "Final assessment after reviewing both arguments",
                "supporting_ids": [valid_id],
                "contradicting_ids": [other_id],
                "i_am_wrong_if": "stop breaks",
            })
        raise AssertionError("Unexpected Gemini prompt")

    a = LLMAgents(gemini(h))
    prop = a.strategist(s, ev, ctx)
    pro, dfn = a.prosecutor(prop, ev, ctx), a.defender(prop, ev, ctx)
    v = a.judge(prop, pro, dfn, ev)

    assert set(a.backends.values()) == {"gemini"}
    assert v.decision == "APPROVE" and v.multiplier == 1
    assert seen["key_in_header"] and not seen["key_in_url"]

    blind_prompt = next(x for x in seen["prompts"] if "BLIND JUDGE" in x)
    final_prompt = next(x for x in seen["prompts"] if "FINAL DELIBERATION" in x)
    assert "PROSECUTOR:" not in blind_prompt and "DEFENDER:" not in blind_prompt
    assert "PROSECUTOR:" in final_prompt and "DEFENDER:" in final_prompt
    assert "PRELIMINARY ASSESSMENT:" in final_prompt

def test_bad_llm_output_falls_back_and_is_recorded():
    s, df, ev, ctx, p = buy_case()
    a = LLMAgents(gemini(lambda r: reply({"points": ["x"], "evidence_ids": ["E99"], "severity": 0.5})))
    pro = a.prosecutor(p, ev, ctx)
    assert a.backends["prosecutor"].startswith("fallback") and set(pro.evidence_ids) <= {e.id for e in ev}
    b = LLMAgents(gemini(lambda r: httpx.Response(500)))
    assert b.judge(p, pro, pro, ev).decision in ("APPROVE", "REDUCE", "REJECT") and b.backends["judge"].startswith("fallback")
    c = LLMAgents(gemini(lambda r: httpx.Response(200, text="not json")))
    c.defender(p, ev, ctx)
    assert c.backends["defender"].startswith("fallback")


def test_llm_cannot_set_size_and_strategist_cannot_override_gate():
    s, df, ev, ctx, p = buy_case()
    valid_id = ev[0].id

    judge_response = {
        "decision": "APPROVE",
        "reasoning": "r",
        "supporting_ids": [valid_id],
        "contradicting_ids": [],
        "i_am_wrong_if": "x",
        "confidence": 0.9,
        "multiplier": 50,
    }

    a = LLMAgents(gemini(lambda r: reply(judge_response)))
    pro = Argument(
        role="PROSECUTOR",
        points=["a"],
        evidence_ids=[valid_id],
        severity=0.1,
    )

    # Position size is calculated by Python, not Gemini.
    assert a.judge(p, pro, pro, ev).multiplier == 1

    # Gate failed -> LLM is not even consulted.
    for sym in WATCHLIST:
        df2 = demo_candles(sym)
        ev2, ctx2 = rb.build_evidence(sym, df2, "DEMO")
        if rb.strategist(sym, ev2, ctx2).action == "NO_TRADE":
            calls = []
            b = LLMAgents(gemini(lambda r: calls.append(1) or reply({})))
            assert b.strategist(sym, ev2, ctx2).action == "NO_TRADE"
            assert not calls
            return

def test_get_agents_selection(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert type(get_agents()) is RuleAgents
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert isinstance(get_agents(), LLMAgents)


def test_pipeline_journals_agent_backends_and_trial_id():
    s, df, *_ = buy_case()
    b = PaperBroker()
    b.checkin(NOW)

    class PreliminaryRuleAgents(RuleAgents):
        def __init__(self):
            super().__init__()
            self.preliminary_assessment = None

        def judge(self, p, pro, dfn, ev=None):
            self.preliminary_assessment = {
                "decision": "REDUCE",
                "reasoning": "Preliminary assessment before arguments",
                "supporting_ids": ["E2"],
                "contradicting_ids": ["E4"],
                "confidence": 0.7,
            }
            return super().judge(p, pro, dfn, ev)

    agents = PreliminaryRuleAgents()
    r = run_trial(
        s, df, "DEMO", b, RiskEngine(),
        {s: D(str(df.close.iloc[-1]))}, NOW,
        simulation_mode=True, agents=agents,
    )

    transcript = r["transcript"]
    assert r["trial_id"] == 1
    assert "agent_backends" in transcript
    assert "process_breakdown" in transcript
    assert transcript["preliminary_assessment"] == agents.preliminary_assessment


def test_memory_reads_own_track_record_and_reaches_prosecutor():
    b = PaperBroker()
    for _ in range(2):
        b.submit("TCS", "BUY", 5, D(100), NOW, stop=D(95)); b.on_prices({"TCS": D(90)}, NOW)
    m = symbol_memory(b, "TCS")
    assert m == {"closed": 2, "stops": 2, "targets": 0}
    s, df, _, _, p = buy_case()
    ev, ctx = rb.build_evidence(s, df, "DEMO", None, {"closed": 3, "stops": 3, "targets": 0})
    assert any(e.source == "MEMORY" for e in ev)
    pro = rb.prosecutor(p, ev, ctx)
    assert any("Own record" in x for x in pro.points)


def test_process_score_can_fail():
    s, df, ev, ctx, p = buy_case()
    pro, dfn = rb.prosecutor(p, ev, ctx), rb.defender(p, ev, ctx)
    v = rb.judge(p, pro, dfn)
    class O: execution_mode = "PAPER"
    good, parts = score_process(ev, p, pro, dfn, v, "EXECUTED", int(D(1000) / (p.entry - p.stop)), D(100000), D("0.01"), O, ctx)
    assert good == 100
    # oversized position -> risk management fails
    bad, parts2 = score_process(ev, p, pro, dfn, v, "EXECUTED", 100000, D(100000), D("0.01"), O, ctx)
    assert parts2["risk_management"] == 0 and bad < good
    # judge approving against a clearly stronger prosecution, with no explanation
    weak = Verdict(decision="APPROVE", multiplier=D(1), reasoning="", supporting_ids=["E1"], contradicting_ids=[], i_am_wrong_if="")
    heavy = Argument(role="PROSECUTOR", points=["x"], evidence_ids=["E1"], severity=0.9)
    sc, parts3 = score_process(ev, p, heavy, dfn, weak, "EXECUTED", int(D(1000) / (p.entry - p.stop)), D(100000), D("0.01"), O, ctx)
    assert parts3["decision_explanation"] <= 5 and parts3["strategy_consistency"] < 20 and sc < 80
