"""
fundamental_sources.py — Multi-Domain News & Information Layer
================================================================

Prediction markets move on: news, Twitter, injuries, polls, macro events.
This module ingests information across ALL domains and matches it to our
active markets.  Output feeds into prob_engine as a fifth signal.

Domains (aligned with scanner CLUSTER_RULES):
  sports        NFL, NBA, soccer, MLB, NHL, tennis, F1, etc.
  politics      US election, UK, EU, mideast, russia_ukraine, china_taiwan
  macro         Fed, CPI, jobs, GDP, rates
  crypto        Bitcoin, Ethereum, DeFi, etc.
  entertainment Oscars, Grammys, box office

Each domain has:
  - RSS feeds (no API key)
  - Entity keywords for matching (from cluster + question)
  - Sentiment heuristics (injury = bearish for team, poll surge = bullish)

Output: FundamentalSignal per market
  p_adjustment   cents to add to p_est (+ = bullish, - = bearish)
  confidence    0–1 how much we trust this signal
  sources       list of (domain, headline_snippet) for logging
  direction     "BULLISH" | "BEARISH" | "NEUTRAL"

Env (optional):
  NEWS_API_KEY     NewsAPI.org — enables keyword search (100 req/day free)
  TWITTER_BEARER   Twitter API v2 — enables tweet volume/sentiment
"""

import re
import time
import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import aiohttp

log = logging.getLogger("fundamental")

# ── RSS feeds by domain (no API key) ─────────────────────────────────────────
# Format: (cluster_id or domain, [list of RSS URLs])
RSS_FEEDS: dict[str, list[str]] = {
    "sports": [
        "https://www.espn.com/espn/rss/news",
        "https://feeds.bleacherreport.com/articles",
        "https://www.theguardian.com/sport/football/rss",
        "https://www.skysports.com/rss/12040",  # football
    ],
    "us_politics": [
        "https://www.politico.com/rss/politics08.xml",
        "https://feeds.reuters.com/Reuters/PoliticsNews",
        "https://www.theguardian.com/us/politics/rss",
    ],
    "uk_politics": [
        "https://www.theguardian.com/politics/rss",
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
    ],
    "mideast": [
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://www.theguardian.com/world/middleeast/rss",
    ],
    "russia_ukraine": [
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://www.theguardian.com/world/russia/rss",
    ],
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://cryptonews.com/news/feed/",
    ],
    "fed_macro": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.federalreserve.gov/feeds/press_all.xml",
        "https://feeds.reuters.com/reuters/USTopNews",
    ],
    "entertainment": [
        "https://variety.com/feed/",
        "https://www.hollywoodreporter.com/feed/",
    ],
    "nfl": [
        "https://www.espn.com/espn/rss/nfl/news",
    ],
    "nba": [
        "https://www.espn.com/espn/rss/nba/news",
    ],
    "premier_league": [
        "https://www.theguardian.com/football/premierleague/rss",
    ],
    "champions_league": [
        "https://www.theguardian.com/football/championsleague/rss",
    ],
    "bundesliga": [
        "https://www.theguardian.com/football/rss",
        "https://www.espn.com/espn/rss/soccer/news",
    ],
}

# Fallback: generic news when no domain match
RSS_FEEDS["default"] = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/rss.xml",
]

# ── Sentiment keywords (title/headline) ──────────────────────────────────────
# Match these to infer BULLISH or BEARISH for the entity
BULLISH_KEYWORDS = [
    "win", "wins", "won", "victory", "beat", "beats", "defeat", "signs", "sign",
    "return", "returns", "fit", "healthy", "surge", "leads", "ahead", "rise",
    "rise", "gain", "gains", "endorse", "endorsement", "poll", "leading",
    "favored", "favourite", "favorite", "odds", "boost", "soar", "jump",
]
BEARISH_KEYWORDS = [
    "injury", "injured", "injuries", "out", "suspended", "ban", "banned",
    "lose", "loses", "lost", "loss", "defeat", "crash", "fall", "drop",
    "decline", "slump", "concern", "worries", "investigation", "scandal",
    "resign", "quit", "withdraw", "pull out", "ruled out", "doubt",
]

# ── Config ───────────────────────────────────────────────────────────────────
MAX_HEADLINES_PER_DOMAIN = 15
MAX_AGE_HOURS = 6
REQUEST_TIMEOUT = 8
CACHE_TTL_S = 300   # 5 min cache per domain


@dataclass
class NewsItem:
    title:     str
    url:       str
    ts:        float
    domain:    str
    snippet:   str = ""

    def matches_entity(self, entity: str) -> bool:
        """Entity (e.g. 'Bayern') appears in title or snippet."""
        e = entity.lower()
        return e in self.title.lower() or e in self.snippet.lower()

    def sentiment(self) -> int:
        """+1 bullish, -1 bearish, 0 neutral."""
        t = (self.title + " " + self.snippet).lower()
        bull = sum(1 for k in BULLISH_KEYWORDS if k in t)
        bear = sum(1 for k in BEARISH_KEYWORDS if k in t)
        if bull > bear:
            return 1
        if bear > bull:
            return -1
        return 0


@dataclass
class FundamentalSignal:
    p_adjustment_cents: float = 0.0   # add to p_est
    confidence:         float = 0.0   # 0–1
    direction:          str   = "NEUTRAL"  # BULLISH | BEARISH | NEUTRAL
    sources:            list[tuple[str, str]] = field(default_factory=list)
    n_headlines:        int   = 0


async def _fetch_rss(session: aiohttp.ClientSession, url: str) -> list[NewsItem]:
    """Fetch and parse RSS feed. Returns list of NewsItem."""
    items = []
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as r:
            if r.status != 200:
                return []
            text = await r.text()
    except Exception as e:
        log.debug("RSS fetch %s: %s", url[:50], e)
        return []

    try:
        root = ET.fromstring(text)
        # Handle RSS 2.0 and Atom
        for item in root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry"):
            title_el = item.find("title") or item.find("{http://www.w3.org/2005/Atom}title")
            link_el  = item.find("link") or item.find("{http://www.w3.org/2005/Atom}link")
            date_el  = item.find("pubDate") or item.find("published") or item.find("{http://www.w3.org/2005/Atom}published")
            desc_el  = item.find("description") or item.find("summary") or item.find("{http://www.w3.org/2005/Atom}summary")

            if title_el is None or title_el.text is None:
                continue

            title = title_el.text.strip()[:200]
            link  = ""
            if link_el is not None:
                # Atom: href attr; RSS 2.0: text content
                link = link_el.get("href") or (link_el.text or "")
            link = str(link or "")

            ts = time.time()
            if date_el is not None and date_el.text:
                try:
                    from datetime import datetime
                    s = date_el.text.strip().replace("Z", "+00:00")
                    if "T" in s:
                        dt = datetime.fromisoformat(s[:26])
                    else:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(date_el.text)
                    ts = dt.timestamp()
                except Exception:
                    pass

            snippet = (desc_el.text or "")[:150] if desc_el is not None else ""
            domain  = urlparse(url).netloc.replace("www.", "")[:20]

            items.append(NewsItem(title=title, url=link, ts=ts, domain=domain, snippet=snippet))
    except ET.ParseError:
        log.debug("RSS parse error: %s", url[:50])
    except Exception as e:
        log.debug("RSS parse %s: %s", url[:50], e)

    return items[:MAX_HEADLINES_PER_DOMAIN]


def _extract_entities(question: str, slug: str, cluster_id: str) -> list[str]:
    """
    Extract entity keywords for matching news.
    Uses: question words, slug tokens, cluster-specific keywords.
    """
    text = (question + " " + slug).lower()
    # Remove common stopwords
    stop = {"will", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "be", "is", "are", "this", "that"}
    words = re.findall(r"[a-z0-9]+", text)
    entities = [w for w in words if len(w) >= 3 and w not in stop]

    # Add cluster-specific expansions (e.g. "bayern" -> "bayern munich")
    cluster_entities = {
        "nfl": ["nfl", "super bowl", "chiefs", "eagles", "49ers", "ravens"],
        "nba": ["nba", "lakers", "celtics", "warriors", "bucks"],
        "premier_league": ["arsenal", "chelsea", "liverpool", "man city", "tottenham"],
        "bundesliga": ["bayern", "dortmund", "leverkusen", "leipzig", "eintracht", "bundesliga"],
        "champions_league": ["ucl", "champions league", "real madrid", "barcelona", "man city"],
        "us_politics": ["trump", "biden", "harris", "republican", "democrat", "senate"],
        "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto"],
        "fed_macro": ["fed", "rate", "inflation", "cpi", "jobs", "fomc"],
        "mideast": ["israel", "gaza", "iran", "hamas", "hezbollah"],
        "russia_ukraine": ["russia", "ukraine", "putin", "zelensky"],
    }
    extra = cluster_entities.get(cluster_id, [])
    for e in extra:
        if e not in entities:
            entities.append(e)

    return list(dict.fromkeys(entities))[:12]  # dedupe, cap


def _domain_for_cluster(cluster_id: str) -> str:
    """Map cluster_id to RSS domain key."""
    if cluster_id in RSS_FEEDS:
        return cluster_id
    for key in ["sports", "us_politics", "crypto", "fed_macro", "entertainment"]:
        if key in cluster_id or cluster_id in key:
            return key
    return "default"


async def fetch_domain_news(
    session: aiohttp.ClientSession,
    domain: str,
) -> list[NewsItem]:
    """Fetch all RSS feeds for a domain. Dedupe by title."""
    urls = RSS_FEEDS.get(domain, RSS_FEEDS["default"])
    all_items: list[NewsItem] = []
    seen_titles: set[str] = set()

    for url in urls[:4]:  # max 4 feeds per domain
        items = await _fetch_rss(session, url)
        for it in items:
            key = it.title[:80]
            if key not in seen_titles:
                seen_titles.add(key)
                cutoff = time.time() - MAX_AGE_HOURS * 3600
                if it.ts >= cutoff:
                    all_items.append(it)
        await asyncio.sleep(0.3)  # rate limit

    all_items.sort(key=lambda x: -x.ts)
    return all_items[:MAX_HEADLINES_PER_DOMAIN * 2]


def compute_fundamental_signal(
    news_items: list[NewsItem],
    entities: list[str],
) -> FundamentalSignal:
    """
    Match news to entities and compute p_adjustment.
    More relevant + stronger sentiment → larger adjustment.
    """
    if not news_items or not entities:
        return FundamentalSignal()

    matched: list[tuple[NewsItem, float]] = []
    for n in news_items:
        for e in entities:
            if n.matches_entity(e):
                # Relevance: entity match + recency
                age_hr = (time.time() - n.ts) / 3600
                recency = max(0, 1.0 - age_hr / MAX_AGE_HOURS)
                score = 0.5 + recency * 0.5
                matched.append((n, score))
                break

    if not matched:
        return FundamentalSignal()

    # Aggregate sentiment
    total_bull = 0.0
    total_bear = 0.0
    total_score = 0.0
    sources: list[tuple[str, str]] = []

    for n, score in matched[:8]:
        s = n.sentiment()
        total_score += score
        if s > 0:
            total_bull += score
        elif s < 0:
            total_bear += score
        sources.append((n.domain, n.title[:60]))

    # p_adjustment: ±3c max, scaled by net sentiment and match count
    net = total_bull - total_bear
    n_matched = len(matched)
    confidence = min(1.0, 0.2 + 0.15 * n_matched + 0.1 * total_score)
    confidence = round(confidence, 2)

    if net > 0:
        direction = "BULLISH"
        adj = min(3.0, 0.5 + net * 0.4)
    elif net < 0:
        direction = "BEARISH"
        adj = max(-3.0, -0.5 + net * 0.4)
    else:
        direction = "NEUTRAL"
        adj = 0.0

    return FundamentalSignal(
        p_adjustment_cents=round(adj, 2),
        confidence=confidence,
        direction=direction,
        sources=sources[:5],
        n_headlines=n_matched,
    )


# ── Cache: domain → (ts, list[NewsItem]) — populated by background task only ─
# Scanner NEVER fetches. It only reads. Sub-150ms latency preserved.
_news_cache: dict[str, tuple[float, list[NewsItem]]] = {}


async def refresh_news_cache_async(session: aiohttp.ClientSession) -> None:
    """
    Fetch all domains in parallel. Updates _news_cache.
    Called by background task only — never blocks the scan.
    """
    domains = list(RSS_FEEDS.keys())
    if "default" in domains:
        domains.remove("default")

    async def fetch_one(domain: str) -> None:
        try:
            items = await fetch_domain_news(session, domain)
            _news_cache[domain] = (time.time(), items)
        except Exception as e:
            log.debug("News refresh %s: %s", domain, e)

    await asyncio.gather(*[fetch_one(d) for d in domains])
    log.info("News cache refreshed: %d domains", len([d for d in domains if d in _news_cache]))


def get_news_for_markets(markets: list) -> dict[str, FundamentalSignal]:
    """
    Read from pre-populated cache only. No network. ~1ms.
    Returns dict: slug → FundamentalSignal.
    """
    result: dict[str, FundamentalSignal] = {}

    for m in markets:
        slug = getattr(m, "slug", "")
        question = getattr(m, "question", "")
        cluster = getattr(m, "cluster_id", "uncategorized")
        domain = _domain_for_cluster(cluster)

        cached = _news_cache.get(domain)
        if not cached:
            continue
        _, items = cached
        if not items:
            continue

        entities = _extract_entities(question, slug, cluster)
        sig = compute_fundamental_signal(items, entities)
        if sig.n_headlines > 0:
            result[slug] = sig

    return result


def merge_fundamental_into_prob(
    p_est: float,
    edge_cents: float,
    fundamental: Optional[FundamentalSignal],
    half_spread_cents: float,
) -> tuple[float, float]:
    """
    Merge fundamental signal into microstructure p_est.
    Returns (adjusted_p_est, adjusted_edge_cents).

    When fundamental confirms microstructure: boost.
    When fundamental contradicts: reduce (don't fight the news).
    """
    if not fundamental or fundamental.n_headlines == 0:
        return p_est, edge_cents

    adj = fundamental.p_adjustment_cents * fundamental.confidence
    # Cap adjustment to half-spread so we don't overshoot
    adj = max(-half_spread_cents * 0.5, min(half_spread_cents * 0.5, adj))

    # Convert cents to price
    p_adj = adj / 100.0
    new_p = max(0.03, min(0.97, p_est + p_adj))
    new_edge = (new_p - (p_est - edge_cents/100)) * 100  # edge = new_p - mid, mid = p_est - edge/100
    # Simpler: new_edge = edge_cents + adj
    new_edge = edge_cents + adj

    return round(new_p, 4), round(new_edge, 2)
