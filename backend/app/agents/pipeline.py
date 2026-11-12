"""AI TRIAL -> RISK ENGINE -> PAPER BROKER. The only path by which an entry reaches the broker."""
from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import pandas as pd

from ..database.models import JournalEntry
from ..risk.engine import AccountState, RiskEngine, TradeProposal
from ..trading.paper_broker import PaperBroker
from ..trading.passport import create_passport
from . import rule_based as rb
from .llm import RuleAgents

_STEP = {"REJECT": 0, "REDUCE": 1, "APPROVE": 2}


def symbol_memory(broker: PaperBroker, symbol: str) -> dict:
    """The agent's own closed-trade record in this symbol (cross-session memory, read from SQLite)."""
    sells = [o for o in broker.orders() if o.symbol == symbol and o.side == "SELL" and o.status == "FILLED"]
    return {"closed": len(sells), "stops": sum("STOP" in o.reason for o in sells),
            "targets": sum("TARGET" in o.reason for o in sells)}


def score_process(ev, prop, pro, dfn, verdict, outcome, qty, equity, risk_limit, order, ctx) -> tuple[int, dict]:
    """Process quality 0-100 from five 20-point checks that CAN fail. Profit and loss are not an input."""
    known = {e.id for e in ev}
    cited = list(prop.evidence_ids) + list(pro.evidence_ids) + list(dfn.evidence_ids)
    valid = bool(cited) and all(i in known for i in cited)
    cats = {e.finding for e in ev if e.id in set(cited)}
    evidence = round(20 * (1 if valid else 0) * min(1.0, len(cats) / 3))

    risk = 20
    if outcome == "EXECUTED":
        ok = prop.stop is not None and prop.stop < prop.entry and qty * (prop.entry - prop.stop) <= equity * risk_limit * Decimal("1.01")
        risk = 20 if ok else 0

    if prop.action != "BUY":
        consistency = 20
    else:
        expected = rb.judge(prop, pro, dfn).decision
        consistency = 10 + {0: 10, 1: 5}.get(abs(_STEP[verdict.decision] - _STEP[expected]), 0)
        if ctx.get("best") != prop.strategy:
            consistency -= 10

    if prop.action != "BUY":
        explanation = 20 if prop.rationale.strip() else 0
    else:
        explanation = 5 * bool(verdict.reasoning.strip()) + 5 * bool(verdict.i_am_wrong_if.strip()) + \
            5 * (len(verdict.supporting_ids) >= 2) + 5 * (len(verdict.contradicting_ids) >= 1)

    safety = 20 if (order is None or order.execution_mode == "PAPER") and (outcome != "EXECUTED" or qty > 0) else 0
    parts = {"evidence_quality": evidence, "risk_management": risk, "strategy_consistency": max(consistency, 0),
             "decision_explanation": explanation, "safety_compliance": safety}
    return sum(parts.values()), parts



def run_trial(symbol: str, df: pd.DataFrame, source: str, broker: PaperBroker, risk: RiskEngine,
              prices: dict[str, Decimal], now: datetime, data_fresh: bool = True,
              simulation_mode: bool = False, news=None, agents=None,
              trading_type: str = "INTRADAY", record_journal: bool = True) -> dict:
    agents = agents or RuleAgents()
    tt = (trading_type or "INTRADAY").upper().strip()
    ev, ctx = rb.build_evidence(symbol, df, source, news, symbol_memory(broker, symbol), trading_type=tt)

    prop = agents.strategist(symbol, ev, ctx)
    pro = agents.prosecutor(prop, ev, ctx)
    dfn = agents.defender(prop, ev, ctx)

    preliminary_assessment = None
    if prop.action == "BUY":
        try:
            verdict = agents.judge(prop, pro, dfn, ev, trading_type=tt)
        except TypeError:
            verdict = agents.judge(prop, pro, dfn, ev)
    else:
        verdict = None

    if verdict is not None:
        preliminary_assessment = getattr(agents, "preliminary_assessment", None)

    # Validate every agent's evidence citations, including both Judge stages.
    known = {e.id for e in ev}
    lists = [prop.evidence_ids, pro.evidence_ids, dfn.evidence_ids]

    if preliminary_assessment:
        lists.extend([
            preliminary_assessment.get("supporting_ids", []),
            preliminary_assessment.get("contradicting_ids", []),
        ])

    if verdict is not None:
        lists.extend([
            verdict.supporting_ids,
            verdict.contradicting_ids,
        ])

    evidence_ok = all(
        evidence_id in known
        for evidence_ids in lists
        for evidence_id in evidence_ids
    )

    outcome, risk_reasons, order, qty = "NO_TRADE", [], None, 0

    if not evidence_ok:
        risk_reasons = ["Agents cited evidence IDs that do not exist; trial voided"]

    elif prop.action == "BUY" and verdict.decision == "REJECT":
        outcome = "REJECTED_BY_JUDGE"

    elif prop.action == "BUY":
        acct = broker.account()
        state = AccountState(
            equity=broker.equity(prices),
            cash=acct.cash,
            start_of_day_equity=acct.start_of_day_equity,
            positions=broker.position_values(prices),
            halted=acct.halted,
            halt_reason=acct.halt_reason,
            last_checkin=acct.last_checkin,
            data_fresh=data_fresh,
        )

        dec = risk.check_entry(
            TradeProposal(symbol, prop.entry, prop.stop, prop.target, verdict.multiplier),
            state,
            now,
            simulation_mode,
        )
        risk_reasons = dec.reasons

        if dec.approved:
            order = broker.submit(
                symbol,
                "BUY",
                dec.quantity,
                prop.entry,
                now,
                stop=prop.stop,
                target=prop.target,
                invalidation=verdict.i_am_wrong_if,
            )
            qty = dec.quantity
            outcome = "EXECUTED" if order.status == "FILLED" else "BLOCKED_BY_RISK"

            if order.status != "FILLED":
                risk_reasons.append(order.reason)
        else:
            outcome = "BLOCKED_BY_RISK"
            broker.log("ENTRY_BLOCKED", symbol, "; ".join(dec.reasons), now)

    summary = f"[{tt}] {_plain_language(symbol, prop, pro, verdict, outcome, qty, risk_reasons, order)}"

    equity = broker.equity(prices)
    score, parts = (
        score_process(
            ev, prop, pro, dfn, verdict, outcome, qty, equity,
            risk.limits.risk_per_trade, order, ctx
        )
        if evidence_ok
        else (0, {"evidence_quality": 0})
    )

    transcript = {
        "trading_type": tt,
        "source": source,
        "agent_backends": dict(agents.backends),
        "memory": ctx.get("memory"),
        "evidence": [e.model_dump() for e in ev],
        "strategist": prop.model_dump(mode="json"),
        "prosecutor": pro.model_dump(),
        "defender": dfn.model_dump(),
        "judge": verdict.model_dump(mode="json") if verdict else None,
        "preliminary_assessment": preliminary_assessment,
        "risk": {
            "approved": outcome == "EXECUTED",
            "quantity": qty,
            "reasons": risk_reasons,
        },
        "process_breakdown": parts,
        "order_id": order.id if order else None,
    }

    # A completed analysis is a journal event even when no order is placed.
    # Trade Passports remain exclusive to successfully filled PAPER orders.
    trial_id = None
    if outcome == "EXECUTED" and order is not None and order.status == "FILLED":
        transcript["trade_passport"] = create_passport(
            order=order,
            trial={
                "symbol": symbol,
                "source": source,
                "generated_at": now,
                "evidence": transcript["evidence"],
                "strategist": transcript["strategist"],
                "prosecutor": transcript["prosecutor"],
                "defender": transcript["defender"],
                "preliminary_assessment": transcript["preliminary_assessment"],
                "judge": transcript["judge"],
                "risk": transcript["risk"],
                "process_breakdown": transcript["process_breakdown"],
            },
        )

    trial_id = None
    if outcome == "EXECUTED":
        record_journal = True

    if record_journal:
        transcript["journal"] = {
            "recorded": True,
            "record_type": "TRADE_EXECUTION" if "trade_passport" in transcript else "DECISION_ONLY",
            "outcome": outcome,
        }
        with broker.Session.begin() as s:
            j = JournalEntry(
                at=now,
                symbol=symbol,
                outcome=outcome,
                summary=summary,
                transcript_json=json.dumps(transcript),
                process_score=score,
            )
            s.add(j)
            s.flush()
            trial_id = j.id
    else:
        transcript["journal"] = {
            "recorded": False,
            "record_type": "PREVIEW_ONLY",
            "outcome": outcome,
            "reason": "Stock not selected for trading log book",
        }

    return {
        "trial_id": trial_id,
        "journal_recorded": record_journal,
        "symbol": symbol,
        "trading_type": tt,
        "outcome": outcome,
        "summary": summary,
        "process_score": score,
        "transcript": transcript,
    }


def _plain_language(symbol, prop, pro, verdict, outcome, qty, reasons, order) -> str:
    if prop.action == "NO_TRADE":
        return f"No trade in {symbol}: {prop.rationale}."
    base = (f"{symbol}: Strategist proposed a long via {prop.strategy} (entry {prop.entry}, stop {prop.stop}, target {prop.target}). "
            f"Prosecutor: {'; '.join(pro.points)}. Judge: {verdict.decision} ({verdict.reasoning}). ")
    if outcome == "EXECUTED":
        return base + f"Risk engine approved {qty} shares; simulated fill at {order.fill_price}. I am wrong if {verdict.i_am_wrong_if}."
    if outcome == "REJECTED_BY_JUDGE":
        return base + "No order was created."
    return base + f"Risk engine blocked the entry: {'; '.join(reasons)}."
