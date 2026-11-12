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
                   news: NewsResult | None = None, memory: dict | None = None,
                   trading_type: str = "INTRADAY") -> tuple[list[Evidence], dict]:
    tt = (trading_type or "INTRADAY").upper().strip()
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

    # Specialized evidence per trading option
    if tt == "INTRADAY":
        add("Intraday Trend (EMA9 vs EMA20)", f"{last.ema9:.2f} vs {last.ema20:.2f}",
            "Fast intraday momentum is bullish" if last.ema9 > last.ema20 else "Fast intraday momentum is bearish or neutral")
        add("Intraday Range & Sizing", f"ATR14 ₹{last.atr14:.2f}",
            "Tight intraday bracket: 0.8x ATR stop, 1.6x ATR target (same-day exit)")
        add("Intraday Liquidity", f"{last.volume_ratio:.2f}x volume",
            "Adequate intraday volume" if last.volume_ratio >= 0.9 else "Low intraday volume; slippage hazard")
    elif tt == "OPTIONS":
        add("Options Volatility Regime", f"{last.vol20:.1%} annual vol",
            "High volatility favors directional gamma expansion" if last.vol20 > 0.35 else "Low/moderate volatility; beware theta decay on long calls")
        add("Options Momentum Impulse", f"MACD Hist {mh:.3f}, RSI {last.rsi14:.1f}",
            "Sharp directional impulse to overcome theta decay" if mh > 0 and last.rsi14 > 50 else "Weak directional impulse; high theta risk")
        add("Options Payoff Structure", "Asymmetric 1:2.5 R:R",
            "Target 3.0x ATR vs Stop 1.2x ATR to capture explosive gamma run")
    elif tt == "FUTURES":
        add("Futures Trend Alignment", f"EMA20 {last.ema20:.2f} vs SMA50 {last.sma50:.2f}",
            "Sustained swing trend suitable for leveraged futures" if last.ema20 > last.sma50 else "Counter-trend / choppy: high margin leverage risk")
        add("Futures Margin & Whipsaw Risk", f"{last.vol20:.1%} volatility",
            "Controlled volatility; safe for multi-day swing carry" if last.vol20 <= 0.40 else "Elevated volatility; high risk of leveraged stop-out")
        add("Futures Swing Target", f"Target 4x ATR (₹{last.close + 4 * last.atr14:.2f})",
            "Multi-day swing momentum target with 2x ATR trailing stop")
    elif tt == "LONG_TERM":
        add("Long-Term Macro Trend (50 vs 200 MA)", f"SMA50 {last.sma50:.2f} vs SMA200 {last.sma200:.2f}",
            "Golden Cross: structural long-term bull market" if last.sma50 >= last.sma200 else "Death Cross: macro structural consolidation")
        add("Long-Term Structural Support", f"₹{last.close - 3.5 * last.atr14:.2f}",
            "Deep structural support buffer (3.5x ATR) to avoid premature shakeouts")
        add("Long-Term Investment Horizon", "Multi-month fundamental compounder",
            "Wealth compounding horizon targeting 7x ATR return")

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
    return ev, {"best": best, "row": last, "backtest": bt, "news": ns, "memory": memory, "trading_type": tt}


def strategist(symbol: str, ev: list[Evidence], ctx: dict) -> Proposal:
    row, best = ctx["row"], ctx["best"]
    tt = ctx.get("trading_type", "INTRADAY")
    by = {e.finding: e for e in ev}

    if tt == "INTRADAY":
        trend_ok = row.ema9 > row.ema20 and row.volume_ratio >= 0.85 and row.momentum20 > -0.05
        if not best or not trend_ok:
            why = "No strategy passes the out-of-sample gate" if not best else "Intraday momentum not aligned (EMA9 <= EMA20 or low volume)"
            return Proposal(action="NO_TRADE", symbol=symbol, strategy=best or "none", rationale=why,
                            evidence_ids=[by["Intraday Trend (EMA9 vs EMA20)"].id, by["Out-of-sample backtest vs buy-and-hold (Sharpe)"].id])
        entry = Decimal(str(round(float(row.close), 2)))
        stop = Decimal(str(round(float(row.close - 0.8 * row.atr14), 2)))
        target = Decimal(str(round(float(row.close + 1.6 * row.atr14), 2)))
        return Proposal(action="BUY", symbol=symbol, strategy=best, entry=entry, stop=stop, target=target,
                        rationale=f"[INTRADAY] Fast EMA9 > EMA20 with volume confirmation; stop 0.8xATR, target 1.6xATR (1:2 R:R same-day exit)",
                        evidence_ids=[e.id for e in ev if e.finding in ("Last close", "Intraday Trend (EMA9 vs EMA20)", "Intraday Range & Sizing",
                                                                         "Intraday Liquidity", "Out-of-sample backtest vs buy-and-hold (Sharpe)")])

    elif tt == "OPTIONS":
        trend_ok = row["hist"] > 0 and row.rsi14 >= 48 and best
        if not best or not trend_ok:
            why = "No strategy passes the out-of-sample gate" if not best else "Options momentum impulse not confirmed (MACD hist <= 0 or weak RSI)"
            return Proposal(action="NO_TRADE", symbol=symbol, strategy=best or "none", rationale=why,
                            evidence_ids=[by["Options Momentum Impulse"].id, by["Out-of-sample backtest vs buy-and-hold (Sharpe)"].id])
        entry = Decimal(str(round(float(row.close), 2)))
        stop = Decimal(str(round(float(row.close - 1.2 * row.atr14), 2)))
        target = Decimal(str(round(float(row.close + 3.0 * row.atr14), 2)))
        return Proposal(action="BUY", symbol=symbol, strategy=best, entry=entry, stop=stop, target=target,
                        rationale=f"[OPTIONS] Call breakout: Accelerating MACD histogram to overcome theta decay; asymmetric 3.0xATR target vs 1.2xATR stop",
                        evidence_ids=[e.id for e in ev if e.finding in ("Last close", "Options Volatility Regime", "Options Momentum Impulse",
                                                                         "Options Payoff Structure", "Out-of-sample backtest vs buy-and-hold (Sharpe)")])

    elif tt == "FUTURES":
        trend_ok = row.ema20 > row.sma50 and row.momentum20 > 0 and best
        if not best or not trend_ok:
            why = "No strategy passes the out-of-sample gate" if not best else "Futures swing trend not aligned (EMA20 <= SMA50 or negative momentum)"
            return Proposal(action="NO_TRADE", symbol=symbol, strategy=best or "none", rationale=why,
                            evidence_ids=[by["Futures Trend Alignment"].id, by["Out-of-sample backtest vs buy-and-hold (Sharpe)"].id])
        entry = Decimal(str(round(float(row.close), 2)))
        stop = Decimal(str(round(float(row.close - 2.0 * row.atr14), 2)))
        target = Decimal(str(round(float(row.close + 4.0 * row.atr14), 2)))
        return Proposal(action="BUY", symbol=symbol, strategy=best, entry=entry, stop=stop, target=target,
                        rationale=f"[FUTURES] Leveraged swing: Trend continuation above 50 SMA with positive momentum; trailing 2xATR stop, 4xATR swing target",
                        evidence_ids=[e.id for e in ev if e.finding in ("Last close", "Futures Trend Alignment", "Futures Margin & Whipsaw Risk",
                                                                         "Futures Swing Target", "Out-of-sample backtest vs buy-and-hold (Sharpe)")])

    else:  # LONG_TERM
        trend_ok = (row.sma50 >= row.sma200 or row.ema20 > row.sma50) and row.momentum20 > -0.05 and best
        if not best or not trend_ok:
            why = "No strategy passes the out-of-sample gate" if not best else "Long-term structural trend not aligned (Death cross or deep momentum loss)"
            return Proposal(action="NO_TRADE", symbol=symbol, strategy=best or "none", rationale=why,
                            evidence_ids=[by["Long-Term Macro Trend (50 vs 200 MA)"].id, by["Out-of-sample backtest vs buy-and-hold (Sharpe)"].id])
        entry = Decimal(str(round(float(row.close), 2)))
        stop = Decimal(str(round(float(row.close - 3.5 * row.atr14), 2)))
        target = Decimal(str(round(float(row.close + 7.0 * row.atr14), 2)))
        return Proposal(action="BUY", symbol=symbol, strategy=best, entry=entry, stop=stop, target=target,
                        rationale=f"[LONG_TERM] Structural investment: Golden cross / macro trend alignment with wide 3.5xATR support stop and 7xATR target",
                        evidence_ids=[e.id for e in ev if e.finding in ("Last close", "Long-Term Macro Trend (50 vs 200 MA)", "Long-Term Structural Support",
                                                                         "Long-Term Investment Horizon", "Out-of-sample backtest vs buy-and-hold (Sharpe)")])


def prosecutor(p: Proposal, ev: list[Evidence], ctx: dict) -> Argument:
    if p.action == "NO_TRADE":
        return Argument(role="PROSECUTOR", points=["No proposal to challenge; court not convened"],
                        evidence_ids=list(p.evidence_ids), severity=0.0)
    row = ctx["row"]
    tt = ctx.get("trading_type", "INTRADAY")
    pts, ids, sev = [], [], 0.0
    by = {e.finding: e.id for e in ev}

    # Trading option specific prosecution
    if tt == "INTRADAY":
        if row.volume_ratio < 0.95 and "Intraday Liquidity" in by:
            pts.append("Intraday volume is low; high spread and slippage hazard on day trade")
            ids.append(by["Intraday Liquidity"])
            sev += 0.3
        if row.rsi14 > 68 and "RSI14" in by:
            pts.append(f"RSI {row.rsi14:.0f} overbought; entry risks intraday session high fade")
            ids.append(by["RSI14"])
            sev += 0.25
    elif tt == "OPTIONS":
        if row.vol20 < 0.25 and "Options Volatility Regime" in by:
            pts.append("Volatility is low; call options face heavy theta decay if underlying chops")
            ids.append(by["Options Volatility Regime"])
            sev += 0.35
        if row["hist"] <= 0 and "Options Momentum Impulse" in by:
            pts.append("MACD histogram is fading; lack of explosive momentum will erode option value")
            ids.append(by["Options Momentum Impulse"])
            sev += 0.3
    elif tt == "FUTURES":
        if row.vol20 > 0.40 and "Futures Margin & Whipsaw Risk" in by:
            pts.append(f"Volatility {row.vol20:.0%} is elevated; high whipsaw risk for leveraged futures")
            ids.append(by["Futures Margin & Whipsaw Risk"])
            sev += 0.35
        if row.momentum20 <= 0 and "20-day momentum" in by:
            pts.append("20-day momentum is negative; leveraged trend continuation carries high drawdown")
            ids.append(by["20-day momentum"])
            sev += 0.25
    elif tt == "LONG_TERM":
        if row.sma50 < row.sma200 and "Long-Term Macro Trend (50 vs 200 MA)" in by:
            pts.append("Underlying is in a Death Cross (50 SMA < 200 SMA); macro structural bear market")
            ids.append(by["Long-Term Macro Trend (50 vs 200 MA)"])
            sev += 0.35

    # General prosecution points
    if row.rsi14 > 72 and "RSI14" in by and by["RSI14"] not in ids:
        pts.append(f"RSI {row.rsi14:.0f} is overbought; entry may be late")
        ids.append(by["RSI14"])
        sev += 0.25
    if row.vol20 > 0.45 and "Annualised volatility (20d)" in by and by["Annualised volatility (20d)"] not in ids:
        pts.append(f"Volatility {row.vol20:.0%} annualised is high; stop may be noise")
        ids.append(by["Annualised volatility (20d)"])
        sev += 0.25
    if row["hist"] <= 0 and "MACD histogram" in by and by["MACD histogram"] not in ids:
        pts.append("MACD histogram is not positive; momentum is fading")
        ids.append(by["MACD histogram"])
        sev += 0.2

    ns = ctx.get("news")
    if ns and ns["tone"] == "negative" and "News tone" in by:
        pts.append("Headline tone is negative")
        ids.append(by["News tone"])
        sev += 0.25
    if ns and ns["event_risk"] and "Event-risk headline" in by:
        pts.append(f"Event risk: {ns['event_risk'][0].title}")
        ids.append(by["Event-risk headline"])
        sev += 0.2

    mem = ctx.get("memory")
    if mem and mem["stops"] >= 2 and mem["stops"] > mem["targets"] and "Own track record on symbol" in by:
        pts.append(f"Own record: {mem['stops']} stop-outs vs {mem['targets']} targets in {p.symbol}")
        ids.append(by["Own track record on symbol"])
        sev += 0.2

    if "News unavailable" in by:
        pts.append("News unavailable; event risk cannot be assessed")
        ids.append(by["News unavailable"])
        sev += 0.1

    if not pts:
        fallback_id = by.get("RSI14") or by.get("Last close") or ev[0].id
        pts.append("No strong negatives found in the available evidence")
        ids.append(fallback_id)

    return Argument(role="PROSECUTOR", points=pts[:5], evidence_ids=list(dict.fromkeys(ids)), severity=min(sev, 1.0))


def defender(p: Proposal, ev: list[Evidence], ctx: dict) -> Argument:
    if p.action == "NO_TRADE":
        return Argument(role="DEFENDER", points=["No proposal to defend; court not convened"],
                        evidence_ids=list(p.evidence_ids), severity=0.0)
    row = ctx["row"]
    tt = ctx.get("trading_type", "INTRADAY")
    by = {e.finding: e.id for e in ev}
    pts, ids, sev = [], [], 0.0

    # Trading option specific defense
    if tt == "INTRADAY":
        if row.ema9 > row.ema20 and "Intraday Trend (EMA9 vs EMA20)" in by:
            pts.append("Fast EMA9 above EMA20 confirms bullish intraday momentum")
            ids.append(by["Intraday Trend (EMA9 vs EMA20)"])
            sev += 0.3
        if "Intraday Range & Sizing" in by:
            pts.append("Controlled 0.8x ATR stop limits intraday capital exposure")
            ids.append(by["Intraday Range & Sizing"])
            sev += 0.2
    elif tt == "OPTIONS":
        if row["hist"] > 0 and "Options Momentum Impulse" in by:
            pts.append("Rising MACD histogram provides directional velocity to outpace theta decay")
            ids.append(by["Options Momentum Impulse"])
            sev += 0.3
        if "Options Payoff Structure" in by:
            pts.append("Asymmetric 3x ATR target yields high risk-reward payoff for call options")
            ids.append(by["Options Payoff Structure"])
            sev += 0.25
    elif tt == "FUTURES":
        if row.ema20 > row.sma50 and "Futures Trend Alignment" in by:
            pts.append("Institutional trend aligned above 50 SMA; ideal for leveraged futures swing")
            ids.append(by["Futures Trend Alignment"])
            sev += 0.3
        if "Futures Swing Target" in by:
            pts.append("Multi-day swing target (4x ATR) captures sustained directional carry")
            ids.append(by["Futures Swing Target"])
            sev += 0.25
    elif tt == "LONG_TERM":
        if row.sma50 >= row.sma200 and "Long-Term Macro Trend (50 vs 200 MA)" in by:
            pts.append("Golden Cross confirmed (50 SMA >= 200 SMA); macro secular bull trend")
            ids.append(by["Long-Term Macro Trend (50 vs 200 MA)"])
            sev += 0.35
        if "Long-Term Structural Support" in by:
            pts.append("Wide 3.5x ATR structural buffer prevents shakeouts during market noise")
            ids.append(by["Long-Term Structural Support"])
            sev += 0.25

    # General defense points
    if row.ema20 > row.sma50 and "EMA20 vs SMA50" in by and by["EMA20 vs SMA50"] not in ids:
        pts.append("EMA20 above SMA50: trend is up")
        ids.append(by["EMA20 vs SMA50"])
        sev += 0.25
    if row.momentum20 > 0 and "20-day momentum" in by and by["20-day momentum"] not in ids:
        pts.append(f"20-day momentum is positive ({row.momentum20:.1%})")
        ids.append(by["20-day momentum"])
        sev += 0.2
    if row["hist"] > 0 and "MACD histogram" in by and by["MACD histogram"] not in ids:
        pts.append("MACD histogram positive")
        ids.append(by["MACD histogram"])
        sev += 0.15

    mem = ctx.get("memory")
    if mem and mem["targets"] > mem["stops"] and "Own track record on symbol" in by:
        pts.append(f"Own record: {mem['targets']} targets vs {mem['stops']} stop-outs in {p.symbol}")
        ids.append(by["Own track record on symbol"])
        sev += 0.15

    ns = ctx.get("news")
    if ns and ns["tone"] == "positive" and "News tone" in by:
        pts.append("Headline tone is positive")
        ids.append(by["News tone"])
        sev += 0.2

    if "Out-of-sample backtest vs buy-and-hold (Sharpe)" in by:
        pts.append(f"{p.strategy} beat buy-and-hold out of sample on Sharpe")
        ids.append(by["Out-of-sample backtest vs buy-and-hold (Sharpe)"])
        sev += 0.25

    if ns is None:
        pts.append("Acknowledged: no news evidence was supplied, so event risk is unmeasured")
    elif ns["event_risk"]:
        pts.append("Acknowledged: a scheduled/regulatory event is pending (see Prosecutor)")

    return Argument(role="DEFENDER", points=pts[:5], evidence_ids=list(dict.fromkeys(ids)), severity=min(sev, 1.0))


def judge(p: Proposal, pro: Argument, dfn: Argument, trading_type: str = "INTRADAY") -> Verdict:
    gap = dfn.severity - pro.severity
    if gap < 0.15:
        d, m = "REJECT", Decimal("0")
    elif pro.severity >= 0.3:
        d, m = "REDUCE", Decimal("0.5")
    else:
        d, m = "APPROVE", Decimal("1")

    tt = (trading_type or "INTRADAY").upper().strip()
    if tt == "INTRADAY":
        wrong = f"intraday price breaches stop {p.stop} or fast EMA9 falls below EMA20" if p.stop else "n/a"
    elif tt == "OPTIONS":
        wrong = f"price closes below stop {p.stop} or volatility collapses causing theta erosion" if p.stop else "n/a"
    elif tt == "FUTURES":
        wrong = f"leveraged swing stop {p.stop} hit or EMA20 falls below SMA50" if p.stop else "n/a"
    else:
        wrong = f"structural support at stop {p.stop} broken or death cross forms" if p.stop else "n/a"

    return Verdict(decision=d, multiplier=m,
                   reasoning=f"[{tt}] Defence strength {dfn.severity:.2f} vs prosecution {pro.severity:.2f}; verdict {d}",
                   supporting_ids=dfn.evidence_ids, contradicting_ids=pro.evidence_ids, i_am_wrong_if=wrong)
