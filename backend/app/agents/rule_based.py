"""Deterministic agents (no LLM). Used for demo mode and tests, and as the interface an LLM backend must satisfy.
Each agent can only cite evidence IDs that exist; the pipeline rejects anything else."""
from __future__ import annotations

from decimal import Decimal

import pandas as pd

from ..backtesting.engine import run_backtest
from ..market.indicators import indicator_frame
from ..market.news import NewsResult, summarize
from .schemas import Argument, Evidence, Proposal, Verdict


def build_evidence(symbol: str, df: pd.DataFrame, source: str,
                   news: NewsResult | None = None, memory: dict | None = None) -> tuple[list[Evidence], dict]:
    f = indicator_frame(df)
    last = f.iloc[-1]
    ts = str(f.index[-1])
    bt = run_backtest(df, source)
    ev: list[Evidence] = []

    def add(finding, value, interp, src=source):
        ev.append(Evidence(id=f"E{len(ev) + 1}", symbol=symbol, source=src, timestamp=ts,
                           finding=finding, value=str(value), interpretation=interp))

    add("Last close", f"{last.close:.2f}", "Reference price for the proposal")
    add("EMA20 vs SMA50", f"{last.ema20:.2f} / {last.sma50:.2f}",
        "Uptrend" if last.ema20 > last.sma50 else "Downtrend or no trend")
    add("RSI14", f"{last.rsi14:.1f}", "Overbought" if last.rsi14 > 70 else "Weak" if last.rsi14 < 40 else "Neutral/healthy")
    mh = last["hist"]
    add("MACD histogram", f"{mh:.3f}", "Momentum rising" if mh > 0 else "Momentum fading")
    add("20-day momentum", f"{last.momentum20:.1%}", "Positive" if last.momentum20 > 0 else "Negative")
    add("Annualised volatility (20d)", f"{last.vol20:.1%}", "High" if last.vol20 > 0.40 else "Moderate")
    add("Volume vs 20d average", f"{last.volume_ratio:.2f}x", "Participation up" if last.volume_ratio > 1.1 else "Normal/low")
    add("ATR14", f"{last.atr14:.2f}", "Used for stop distance")
    best = max((k for k in bt["passes_gate"] if bt["passes_gate"][k]), key=lambda k: bt["metrics"][k]["sharpe"], default=None)
    m = bt["metrics"]
    add("Out-of-sample backtest vs buy-and-hold (Sharpe)",
        f"mom {m['momentum']['sharpe']:.2f}, ma {m['ma_crossover']['sharpe']:.2f}, bh {m['buy_and_hold']['sharpe']:.2f}",
        f"Strategy gate passed by {best}" if best else "No strategy beats buy-and-hold out of sample", src="BACKTEST")
    ns = None
    if news is not None:
        nsrc = "DEMO" if news.status == "DEMO" else "NEWS"
        if news.headlines:
            ns = summarize(news.headlines)
            add("News tone", f"{ns['positive']} positive / {ns['negative']} negative of {ns['count']} headlines",
                f"Overall {ns['tone']}: " + " | ".join(h.title for h in news.headlines[:3]), src=nsrc)
            if ns["event_risk"]:
                add("Event-risk headline", ns["event_risk"][0].title, "Scheduled or regulatory event can gap the price", src=nsrc)
        else:
            add("News unavailable", news.status, news.detail or "No news evidence; event risk unmeasured", src=nsrc)
    if memory is not None:
        add("Own track record on symbol", f"{memory['closed']} closed: {memory['targets']} target / {memory['stops']} stop",
            "Learned from the agent's own past paper trades in this symbol", src="MEMORY")
    return ev, {"best": best, "row": last, "backtest": bt, "news": ns, "memory": memory}


def strategist(symbol: str, ev: list[Evidence], ctx: dict) -> Proposal:
    row, best = ctx["row"], ctx["best"]
    by = {e.finding: e for e in ev}
    trend_ok = row.ema20 > row.sma50 and row.momentum20 > 0
    if not best or not trend_ok:
        why = "No strategy passes the out-of-sample gate" if not best else "Trend/momentum not aligned"
        return Proposal(action="NO_TRADE", symbol=symbol, strategy=best or "none", rationale=why,
                        evidence_ids=[by["EMA20 vs SMA50"].id, by["Out-of-sample backtest vs buy-and-hold (Sharpe)"].id])
    entry = Decimal(str(round(float(row.close), 2)))
    stop = Decimal(str(round(float(row.close - 2 * row.atr14), 2)))
    target = Decimal(str(round(float(row.close + 3 * row.atr14), 2)))
    return Proposal(action="BUY", symbol=symbol, strategy=best, entry=entry, stop=stop, target=target,
                    rationale=f"{best} passes the gate and trend/momentum agree; stop 2xATR, target 3xATR (1.5R)",
                    evidence_ids=[e.id for e in ev if e.finding in ("Last close", "EMA20 vs SMA50", "20-day momentum", "ATR14",
                                                                     "Out-of-sample backtest vs buy-and-hold (Sharpe)")])


def prosecutor(p: Proposal, ev: list[Evidence], ctx: dict) -> Argument:
    if p.action == "NO_TRADE":
        return Argument(role="PROSECUTOR", points=["No proposal to challenge; court not convened"],
                        evidence_ids=list(p.evidence_ids), severity=0.0)
    row = ctx["row"]
    pts, ids, sev = [], [], 0.0
    by = {e.finding: e.id for e in ev}
    if row.rsi14 > 70: pts.append(f"RSI {row.rsi14:.0f} is overbought; entry may be late"); ids.append(by["RSI14"]); sev += 0.3
    if row.vol20 > 0.40: pts.append(f"Volatility {row.vol20:.0%} annualised is high; stop may be noise"); ids.append(by["Annualised volatility (20d)"]); sev += 0.3
    if row["hist"] <= 0: pts.append("MACD histogram is not positive; momentum is fading"); ids.append(by["MACD histogram"]); sev += 0.25
    if row.volume_ratio < 0.9: pts.append("Volume is below average; weak confirmation"); ids.append(by["Volume vs 20d average"]); sev += 0.15
    ns = ctx.get("news")
    if ns and ns["tone"] == "negative": pts.append("Headline tone is negative"); ids.append(by["News tone"]); sev += 0.25
    if ns and ns["event_risk"]: pts.append(f"Event risk: {ns['event_risk'][0].title}"); ids.append(by["Event-risk headline"]); sev += 0.2
    mem = ctx.get("memory")
    if mem and mem["stops"] >= 2 and mem["stops"] > mem["targets"]:
        pts.append(f"Own record: {mem['stops']} stop-outs vs {mem['targets']} targets in {p.symbol}"); ids.append(by["Own track record on symbol"]); sev += 0.2
    if "News unavailable" in by: pts.append("News unavailable; event risk cannot be assessed"); ids.append(by["News unavailable"]); sev += 0.1
    if not pts: pts.append("No strong negatives found in the available evidence"); ids.append(by["RSI14"])
    return Argument(role="PROSECUTOR", points=pts, evidence_ids=ids, severity=min(sev, 1.0))


def defender(p: Proposal, ev: list[Evidence], ctx: dict) -> Argument:
    if p.action == "NO_TRADE":
        return Argument(role="DEFENDER", points=["No proposal to defend; court not convened"],
                        evidence_ids=list(p.evidence_ids), severity=0.0)
    row = ctx["row"]
    by = {e.finding: e.id for e in ev}
    pts, ids, sev = [], [], 0.0
    if row.ema20 > row.sma50: pts.append("EMA20 above SMA50: trend is up"); ids.append(by["EMA20 vs SMA50"]); sev += 0.3
    if row.momentum20 > 0: pts.append(f"20-day momentum {row.momentum20:.1%}"); ids.append(by["20-day momentum"]); sev += 0.25
    if row["hist"] > 0: pts.append("MACD histogram positive"); ids.append(by["MACD histogram"]); sev += 0.2
    mem = ctx.get("memory")
    if mem and mem["targets"] > mem["stops"]:
        pts.append(f"Own record: {mem['targets']} targets vs {mem['stops']} stop-outs in {p.symbol}"); ids.append(by["Own track record on symbol"]); sev += 0.15
    ns = ctx.get("news")
    if ns and ns["tone"] == "positive": pts.append("Headline tone is positive"); ids.append(by["News tone"]); sev += 0.2
    pts.append(f"{p.strategy} beat buy-and-hold out of sample on Sharpe"); ids.append(by["Out-of-sample backtest vs buy-and-hold (Sharpe)"]); sev += 0.25
    if ns is None: pts.append("Acknowledged: no news evidence was supplied, so event risk is unmeasured")
    elif ns["event_risk"]: pts.append("Acknowledged: a scheduled/regulatory event is pending (see Prosecutor)")
    return Argument(role="DEFENDER", points=pts, evidence_ids=ids, severity=min(sev, 1.0))


def judge(p: Proposal, pro: Argument, dfn: Argument) -> Verdict:
    gap = dfn.severity - pro.severity
    if gap < 0.15:
        d, m = "REJECT", Decimal("0")
    elif pro.severity >= 0.3:
        d, m = "REDUCE", Decimal("0.5")
    else:
        d, m = "APPROVE", Decimal("1")
    wrong = f"price closes below the stop {p.stop}, or EMA20 falls below SMA50" if p.stop else "n/a"
    return Verdict(decision=d, multiplier=m,
                   reasoning=f"Defence strength {dfn.severity:.2f} vs prosecution {pro.severity:.2f}; verdict {d}",
                   supporting_ids=dfn.evidence_ids, contradicting_ids=pro.evidence_ids, i_am_wrong_if=wrong)
