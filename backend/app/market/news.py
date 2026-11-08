"""News input. Live mode reads Google News RSS (no key). Demo mode uses clearly labelled DEMO headlines.
If a live fetch fails we return status UNAVAILABLE with no headlines: it never substitutes fake news."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import quote

POSITIVE = ("beat", "beats", "surge", "record", "upgrade", "upgrades", "wins", "win", "order", "profit rises",
            "growth", "strong", "buyback", "dividend", "expansion", "rallies", "gains")
NEGATIVE = ("miss", "misses", "falls", "plunge", "downgrade", "downgrades", "loss", "weak", "probe", "fraud",
            "penalty", "slump", "cuts", "decline", "lawsuit", "resigns", "raid")
EVENT_RISK = ("results", "earnings", "q2", "q3", "probe", "sebi", "raid", "fraud", "merger", "demerger", "board meeting")


@dataclass
class Headline:
    title: str
    source: str
    published: str = ""
    url: str = ""


@dataclass
class NewsResult:
    status: str                      # OK | DEMO | UNAVAILABLE
    headlines: list[Headline] = field(default_factory=list)
    detail: str = ""


def tone(title: str) -> int:
    t = title.lower()
    return sum(w in t for w in POSITIVE) - sum(w in t for w in NEGATIVE)


def summarize(headlines: list[Headline]) -> dict:
    scores = [tone(h.title) for h in headlines]
    pos, neg = sum(s > 0 for s in scores), sum(s < 0 for s in scores)
    risk = [h for h in headlines if any(w in h.title.lower() for w in EVENT_RISK)]
    label = "positive" if pos > neg else "negative" if neg > pos else "mixed/neutral"
    return {"count": len(headlines), "positive": pos, "negative": neg, "tone": label, "event_risk": risk}


def parse_rss(xml_text: str, limit: int = 10) -> list[Headline]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        src = item.find("source")
        out.append(Headline(title=(item.findtext("title") or "").strip(),
                            source=(src.text if src is not None else "Google News"),
                            published=item.findtext("pubDate") or "", url=item.findtext("link") or ""))
        if len(out) >= limit:
            break
    return out


def _default_fetch(url: str) -> str:
    import httpx
    r = httpx.get(url, timeout=8.0, follow_redirects=True, headers={"User-Agent": "verdict-paper-trader"})
    r.raise_for_status()
    return r.text


def live_news(symbol: str, fetch: Callable[[str], str] | None = None) -> NewsResult:
    url = f"https://news.google.com/rss/search?q={quote(symbol + ' NSE stock')}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        hs = parse_rss((fetch or _default_fetch)(url))
    except Exception as e:  # network, HTTP or parse error: report, never fake
        return NewsResult("UNAVAILABLE", [], f"News fetch failed: {type(e).__name__}")
    return NewsResult("OK" if hs else "UNAVAILABLE", hs, "" if hs else "No headlines returned")


_DEMO = {
    0: ["{s} Q2 results due next week; analysts expect steady growth", "{s} wins large order, shares rally"],
    1: ["{s} faces regulatory probe over disclosures", "{s} results due in two days; volatility elevated"],
    2: ["Brokerages upgrade {s} citing strong demand", "{s} announces dividend and buyback"],
    3: ["{s} shares fall as margins weak; downgrade from one broker", "{s} board meeting scheduled this week"],
}


def demo_news(symbol: str) -> NewsResult:
    k = sum(map(ord, symbol)) % 4
    hs = [Headline(t.format(s=symbol), "DEMO", url="") for t in _DEMO[k]]
    return NewsResult("DEMO", hs, "Synthetic demo headlines")


def get_news(symbol: str, mode: str, fetch=None) -> NewsResult:
    return demo_news(symbol) if mode == "demo" else live_news(symbol, fetch)
