from datetime import datetime
from decimal import Decimal as D

from app.agents import rule_based as rb
from app.agents.pipeline import run_trial
from app.market.demo_data import demo_candles
from app.market.news import Headline, NewsResult, get_news, live_news, parse_rss, summarize, tone
from app.risk.engine import RiskEngine
from app.trading.paper_broker import PaperBroker

RSS = """<rss><channel>
<item><title>TCS wins large order, shares rally</title><link>http://x/1</link><pubDate>Mon</pubDate><source>Mint</source></item>
<item><title>TCS faces SEBI probe</title><link>http://x/2</link><pubDate>Mon</pubDate><source>ET</source></item>
</channel></rss>"""


def test_rss_parse_and_tone():
    hs = parse_rss(RSS)
    assert [h.source for h in hs] == ["Mint", "ET"]
    assert tone(hs[0].title) > 0 and tone(hs[1].title) < 0
    s = summarize(hs)
    assert s["count"] == 2 and s["event_risk"]


def test_live_news_failure_is_reported_never_faked():
    def boom(url): raise OSError("blocked")
    r = live_news("TCS", boom)
    assert r.status == "UNAVAILABLE" and r.headlines == []


def test_demo_news_labelled_demo_and_live_mode_does_not_fall_back():
    assert get_news("TCS", "demo").status == "DEMO"
    assert all(h.source == "DEMO" for h in get_news("TCS", "demo").headlines)
    assert get_news("TCS", "live", fetch=lambda u: (_ for _ in ()).throw(OSError())).status == "UNAVAILABLE"


def test_news_becomes_cited_evidence_and_event_risk_reaches_prosecutor():
    df = demo_candles("TCS")
    news = NewsResult("OK", [Headline("TCS results due in two days", "ET")])
    ev, ctx = rb.build_evidence("TCS", df, "DEMO", news)
    assert any(e.finding == "Event-risk headline" for e in ev)
    p = rb.strategist("TCS", ev, ctx)
    pro = rb.prosecutor(p, ev, ctx)
    assert any("Event risk" in x for x in pro.points)
    assert set(pro.evidence_ids) <= {e.id for e in ev}


def test_unavailable_news_is_itself_evidence():
    df = demo_candles("TCS")
    ev, ctx = rb.build_evidence("TCS", df, "DEMO", NewsResult("UNAVAILABLE", [], "x"))
    assert any(e.finding == "News unavailable" for e in ev)


def test_pipeline_accepts_news():
    b = PaperBroker(); b.checkin(datetime(2026, 11, 9, 5))
    df = demo_candles("TCS")
    r = run_trial("TCS", df, "DEMO", b, RiskEngine(), {"TCS": D(str(df.close.iloc[-1]))}, datetime(2026, 11, 9, 5),
                  simulation_mode=True, news=get_news("TCS", "demo"))
    assert any(e["finding"].startswith("News") for e in r["transcript"]["evidence"])
