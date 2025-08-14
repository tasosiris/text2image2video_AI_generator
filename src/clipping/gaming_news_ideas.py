#!/usr/bin/env python3
"""
Gaming News Ideas: Aggregate gaming RSS feeds, filter to the last N days,
score articles for clickbaity YouTube potential, and print the top K ideas
with themes and short descriptions.

Defaults:
- Feeds: VG247 + GameSpot (Everything)
- Window: last 7 days
- Limit: top 20

Design goals:
- Simple, stdlib-only
- Understand common RSS fields (title, link, pubDate, category, description)
"""

import argparse
import datetime as dt
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FEEDS: Tuple[str, ...] = (
    "https://www.vg247.com/feed",
    "https://www.gamespot.com/feeds/mashup",
)


@dataclass
class FeedItem:
    title: str
    link: str
    pub_date: Optional[dt.datetime]
    categories: List[str]
    description: str


def fetch_feed_xml(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def strip_html(s: str) -> str:
    # Minimal HTML tag stripper for descriptions
    return re.sub(r"<[^>]*>", "", s)


def parse_rss(xml_text: str) -> List[FeedItem]:
    """Parse a generic RSS 2.0 feed into FeedItem entries."""
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    items: List[FeedItem] = []
    if channel is None:
        return items

    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_raw = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        categories = [
            (c.text or "").strip() for c in item.findall("category") if (c.text or "").strip()
        ]

        pub_dt: Optional[dt.datetime] = None
        if pub_date_raw:
            # Common RSS pubDate format: Mon, 11 Aug 2025 13:54:52 +0000
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"):
                try:
                    pub_dt = dt.datetime.strptime(pub_date_raw, fmt)
                    break
                except Exception:
                    pub_dt = None

        items.append(
            FeedItem(
                title=title,
                link=link,
                pub_date=pub_dt,
                categories=categories,
                description=strip_html(description),
            )
        )

    return items


def within_last_days(d: Optional[dt.datetime], days: int) -> bool:
    if d is None:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    return (now - d) <= dt.timedelta(days=days)


def score_clickbait(item: FeedItem) -> float:
    """
    Heuristic scoring for clickbait potential based on title/description/category.
    Factors (simple weights):
    - Recency already filtered; mild bonus for <48h
    - Hype keywords (beta, leaks, revealed, new, update, free, record, twitch, steam)
    - Big franchises (Battlefield, COD, GTA, Elden Ring, Genshin, Fortnite, etc.)
    - Strong words (biggest, broken, meta, overpowered, etc.)
    """
    title = item.title.lower()
    desc = item.description.lower()
    cats = [c.lower() for c in item.categories]

    score = 0.0

    hype = [
        "revealed",
        "announce",
        "leak",
        "leaked",
        "rumor",
        "beta",
        "early access",
        "update",
        "patch",
        "dlc",
        "free",
        "big",
        "massive",
        "record",
        "numbers",
        "twitch",
        "steam",
    ]
    power_franchises = [
        "battlefield",
        "call of duty",
        "cod",
        "gta",
        "grand theft auto",
        "elden ring",
        "genshin",
        "fortnite",
        "minecraft",
        "fifa",
        "fc 26",
        "pokemon",
        "halo",
    ]
    strong_words = [
        "best",
        "biggest",
        "insane",
        "wild",
        "broken",
        "overpowered",
        "op",
        "meta",
        "secret",
        "hidden",
        "lifechanging",
    ]

    text = f"{title} {desc} {' '.join(cats)}"

    for kw in hype:
        if kw in text:
            score += 1.0
    for kw in power_franchises:
        if kw in text:
            score += 1.5
    for kw in strong_words:
        if kw in text:
            score += 0.75

    # Category nudges
    for c in cats:
        if c in {"pc", "ps5", "xbox series x/s", "steam"}:
            score += 0.25

    # Time-sensitive bonus
    if any(k in text for k in ["beta", "twitch", "steam", "numbers", "record"]):
        score += 0.5

    # Very recent (< 48h) bonus
    if item.pub_date:
        now = dt.datetime.now(dt.timezone.utc)
        if (now - item.pub_date) <= dt.timedelta(hours=48):
            score += 0.5

    return score


def idea_from_item(item: FeedItem) -> Tuple[str, str, str]:
    """Generate (headline, theme, blurb) for a YouTube video idea from a feed item."""
    title = item.title.strip()
    link = item.link.strip()

    # Basic theme extraction using categories or title keywords
    theme = "General Gaming News"
    cats = [c for c in item.categories if c]
    if cats:
        theme = cats[0]
    else:
        lower = title.lower()
        if "battlefield" in lower:
            theme = "Battlefield"
        elif "call of duty" in lower or "cod" in lower:
            theme = "Call of Duty"
        elif "elden ring" in lower:
            theme = "Elden Ring"
        elif "genshin" in lower:
            theme = "Genshin Impact"

    headline = f"{title} – Why Everyone's Talking Right Now"
    blurb = (
        f"Fast breakdown of the big story: '{title}'. What changed, why it matters, and what's next. "
        f"Source: {link}"
    )
    return headline, theme, blurb


def merge_items_unique(items_list: Iterable[FeedItem]) -> List[FeedItem]:
    """Deduplicate items by link (case-insensitive), keeping the first occurrence."""
    seen: Dict[str, FeedItem] = {}
    for it in items_list:
        key = (it.link or it.title).strip().lower()
        if key and key not in seen:
            seen[key] = it
    return list(seen.values())


def load_all_feeds(urls: Sequence[str]) -> List[FeedItem]:
    all_items: List[FeedItem] = []
    for url in urls:
        try:
            xml_text = fetch_feed_xml(url)
            items = parse_rss(xml_text)
            all_items.extend(items)
        except Exception as e:
            print(f"Warning: failed to load {url}: {e}", file=sys.stderr)
    return all_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gaming RSS → Top clickbaity YouTube video ideas (last N days; defaults to top 20)"
    )
    parser.add_argument(
        "--feed",
        action="append",
        help="RSS feed URL(s). If omitted, uses defaults (VG247 + GameSpot)",
    )
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--limit", type=int, default=20, help="How many ideas to print (default: 20)")
    args = parser.parse_args()

    feed_urls = tuple(args.feed) if args.feed else DEFAULT_FEEDS

    items = load_all_feeds(feed_urls)
    if not items:
        print("No items loaded from feeds.")
        sys.exit(0)

    items = merge_items_unique(items)
    items_recent = [i for i in items if within_last_days(i.pub_date, args.days)]
    if not items_recent:
        print("No recent feed items found in the selected window.")
        sys.exit(0)

    # Score once and sort by score desc
    scored = [(score_clickbait(i), i) for i in items_recent]
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    top = ranked[: max(1, int(args.limit))]

    print("--- Top YouTube Video Ideas (Gaming feeds, last %d days) ---" % args.days)
    for idx, (score, item) in enumerate(top, 1):
        headline, theme, blurb = idea_from_item(item)
        when = item.pub_date.strftime("%Y-%m-%d %H:%M UTC") if item.pub_date else "unknown"
        print(f"\n#{idx}: {headline}")
        print(f"- Theme: {theme}")
        print(f"- Published: {when}")
        print(f"- Score: {score:.2f}")
        print(f"- Link: {item.link}")
        print(f"- About: {blurb}")

    if top:
        cutoff = top[-1][0]
        print(f"\nCutoff score (lowest in top {len(top)}): {cutoff:.2f}")


if __name__ == "__main__":
    main()


