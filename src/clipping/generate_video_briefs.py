#!/usr/bin/env python3
"""
Generate YouTube video briefs from top gaming news items.

What it does (simple):
1) Loads recent top items from gaming RSS feeds (reuses logic from `gaming_news_ideas.py`)
2) For each top item, asks the chat model to produce:
   - a clickbaity YouTube title
   - a narrated video script
   - a list of tags
3) Saves each result as an individual JSON file in `outputs/briefs/` by default
4) Narrates the script and saves audio under `outputs/narration/<timestamp>/<slug>/`
5) Transcribes the combined MP3 to a timestamped `.txt` using Deepgram (if available)
6) Aligns `background_footage` sections to narration timestamps via ChatGPT and saves
   `<audio_basename>_background_alignment.json` next to the narration files

Notes:
- Uses the same OpenAI-compatible setup as in `src/documentary_generator.py` (AIML API)
- Everything is saved under `outputs/` (no writes to `src/`)
- Keep it simple and robust with clear console output
"""

import argparse
import datetime as dt
import time
import json
import os
import sys
import urllib.request
from typing import Any, Dict, List, Tuple

# Reuse feed aggregation/scoring from the general ideas script
try:
    # When run as a module: python -m src.clipping.generate_video_briefs
    from .gaming_news_ideas import (
        DEFAULT_FEEDS,
        FeedItem,
        idea_from_item,
        load_all_feeds,
        merge_items_unique,
        score_clickbait,
        within_last_days,
    )
except Exception:
    # When run as a script: python src/clipping/generate_video_briefs.py
    import os as _os, sys as _sys
    _script_dir = _os.path.dirname(_os.path.realpath(__file__))
    _src_root = _os.path.abspath(_os.path.join(_script_dir, _os.pardir))
    if _src_root not in _sys.path:
        _sys.path.append(_src_root)
    from clipping.gaming_news_ideas import (  # type: ignore
        DEFAULT_FEEDS,
        FeedItem,
        idea_from_item,
        load_all_feeds,
        merge_items_unique,
        score_clickbait,
        within_last_days,
    )

# OpenAI-compatible client (same as in documentary_generator.py)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # Optional; it's fine if not present

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover - import-time feedback
    print("Error: 'openai' package not installed. Install with: pip install openai", file=sys.stderr)
    raise

# Optional Deepgram import for STT
try:
    from deepgram import DeepgramClient, PrerecordedOptions  # type: ignore
except Exception:
    DeepgramClient = None  # type: ignore
    PrerecordedOptions = None  # type: ignore

# Narration tool (GPU-aware with CPU fallback)
try:
    from tools import NarrationSettings, narrate_script  # type: ignore
except Exception:
    # Ensure the src/ root (parent of this folder) is on sys.path so we can import tools
    _script_dir = os.path.dirname(os.path.realpath(__file__))
    _src_root = os.path.abspath(os.path.join(_script_dir, os.pardir))
    if _src_root not in sys.path:
        sys.path.append(_src_root)
    try:
        from tools import NarrationSettings, narrate_script  # type: ignore
    except Exception:
        NarrationSettings = None  # type: ignore
        narrate_script = None  # type: ignore


def build_client() -> Tuple[Any, str]:
    """Create an OpenAI-compatible client using the same env vars as documentary_generator.py."""
    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("AIML_API_KEY")
    base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
    model_id = os.getenv("AIML_MODEL_ID", "gpt-4o")

    if not api_key:
        print("Warning: AIML_API_KEY is not set. Requests will likely fail.", file=sys.stderr)

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_id


# -------- Deepgram transcription helpers (simple) --------
def _guess_mimetype(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".wav"):
        return "audio/wav"
    if lower.endswith(".mp3"):
        return "audio/mp3"
    if lower.endswith(".m4a") or lower.endswith(".aac"):
        return "audio/mp4"
    if lower.endswith(".flac"):
        return "audio/flac"
    return "application/octet-stream"


def _build_deepgram_client() -> Any:
    if DeepgramClient is None:
        return None
    if load_dotenv:
        load_dotenv()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("[WARN] DEEPGRAM_API_KEY not set; skipping transcription.")
        return None
    try:
        return DeepgramClient(api_key)
    except Exception as exc:
        print(f"[WARN] Failed to init Deepgram client: {exc}")
        return None


def _transcribe_with_deepgram(client: Any, audio_path: str) -> Dict[str, Any]:
    if client is None or PrerecordedOptions is None:
        return {}
    try:
        mimetype = _guess_mimetype(audio_path)
        with open(audio_path, "rb") as f:
            source = {"buffer": f, "mimetype": mimetype}
            options = PrerecordedOptions(
                model="nova-2-general",
                smart_format=True,
                diarize=True,
                utterances=True,
                punctuate=True,
                paragraphs=True,
                profanity_filter=False,
            )
            resp = client.listen.rest.v("1").transcribe_file(source, options)
            try:
                return resp.to_dict()
            except Exception:
                try:
                    import json as _json
                    return _json.loads(resp.to_json())
                except Exception:
                    return {"raw": str(resp)}
    except Exception as exc:
        print(f"[WARN] Transcription failed for {audio_path}: {exc}")
        return {}


def _extract_utterances(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return (transcript.get("results", {}).get("utterances", [])) or []
    except Exception:
        return []


def _seconds_to_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    remaining = total_ms % 3_600_000
    minutes = remaining // 60_000
    remaining %= 60_000
    secs = remaining // 1000
    ms = remaining % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _txt_from_utterances(utterances: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for utt in utterances:
        start = float(utt.get("start", 0.0))
        end = float(utt.get("end", start))
        text = (utt.get("transcript") or utt.get("text") or "").strip()
        start_str = _seconds_to_srt_time(start)
        end_str = _seconds_to_srt_time(end)
        if text:
            lines.append(f"{start_str} --> {end_str}  {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _derive_txt_path(audio_path: str) -> str:
    base, _ = os.path.splitext(audio_path)
    return f"{base}.txt"


# -------- Alignment via ChatGPT (simple) --------
def _align_background_to_timestamps(
    client: Any,
    model: str,
    video_title: str,
    background_sections: Dict[str, Any],
    transcript_lines: List[str],
    output_json_path: str,
) -> None:
    try:
        system_msg = (
            "You are a video editor. Map background footage sections to narration timestamps. "
            "Return strict JSON only."
        )
        user_payload = {
            "video_title": video_title,
            "background_footage": background_sections,
            "transcript": transcript_lines[:2000],
            "instructions": (
                "Figure out which transcript timestamp windows best match each background section. "
                "Prefer contiguous windows covering the section's content. Use multiple windows if needed. "
                "Output JSON in this schema: {\n"
                "  'video_title': str,\n"
                "  'mapping': [\n"
                "    {'section_name': str, 'query': str, 'timestamps': [{'start': 'HH:MM:SS,mmm', 'end': 'HH:MM:SS,mmm'}], 'notes': str}\n"
                "  ]\n"
                "}."
            ),
        }
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        cleaned = content.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            mapping = json.loads(cleaned)
        except Exception:
            mapping = {"raw": content}
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] Alignment failed: {exc}")


def fetch_article_content(url: str) -> str:
    """Fetch the full article content from a URL. Return text or empty if failed."""
    try:
        # Basic web scraping - get the HTML and extract text roughly
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        # Very basic text extraction - remove HTML tags and get readable content
        import re
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Take first ~3000 chars to avoid overwhelming the API
        return text[:3000] if text else ""
    except Exception as e:
        print(f"[WARN] Failed to fetch content from {url}: {e}")
        return ""


def get_top_items(days: int, limit: int, feeds: List[str]) -> List[Tuple[float, FeedItem]]:
    """Load, merge, filter, score and rank feed items. Returns list of (score, item)."""
    items = load_all_feeds(tuple(feeds))
    items = merge_items_unique(items)
    items_recent = [i for i in items if within_last_days(i.pub_date, days)]
    scored = [(score_clickbait(i), i) for i in items_recent]
    ranked = sorted(scored, key=lambda t: t[0], reverse=True)
    return ranked[: max(1, int(limit))]


def _strip_code_fences(text: str) -> str:
    """Remove common markdown code fences (```json ... ```) and surrounding whitespace."""
    t = text.strip()
    if t.startswith("```"):
        # remove first fence line
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _first_nonempty_line(text: str) -> str:
    """Return the first non-empty line from text (after stripping code fences)."""
    cleaned = _strip_code_fences(text)
    for line in cleaned.splitlines():
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return ""


def call_for_title(client: Any, model: str, item: FeedItem, full_content: str) -> str:
    """Ask the model for a super clickbaity YouTube title (plain text, one line)."""
    published = item.pub_date.strftime("%Y-%m-%d %H:%M:%S %Z") if item.pub_date else "unknown"
    categories = ", ".join(item.categories)

    system_msg = (
        "You are a master YouTube title strategist. Create SUPER clickbaity titles that drive massive views. "
        "Use psychology, curiosity gaps, and emotional triggers without being misleading. "
        "NEVER use emojis, symbols, or special characters in titles."
    )
    content_preview = full_content[:500] if full_content else item.description
    user_msg = (
        "Write ONE super clickbaity YouTube video title (one line, no quotes).\n\n"
        f"Article Title: {item.title}\n"
        f"Published: {published}\n"
        f"Categories: {categories}\n"
        f"Summary: {item.description}\n"
        f"Content Preview: {content_preview}\n"
        f"Link: {item.link}\n\n"
        "Title requirements:\n"
        "- Under 90 chars, MAXIMUM clickbait appeal.\n"
        "- Use words like: SHOCKING, INSANE, UNBELIEVABLE, EXPOSED, SECRET, BANNED, etc.\n"
        "- Create massive curiosity gap and emotional hook.\n"
        "- Include franchise/platform (Battlefield, Steam, etc.) for searchability.\n"
        "- Make viewers feel they MUST click or miss out.\n"
        "- NO emojis, symbols, or special characters - plain text only.\n"
    )
    print("[DEBUG] Calling title generation…")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.8,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    preview = content[:200].replace('\n', ' ')
    print(f"[DEBUG] Title raw length: {len(content)}; preview: {preview}…")
    title = _first_nonempty_line(content)
    print(f"[DEBUG] Extracted title: {title!r}")
    return title


def call_for_script(client: Any, model: str, item: FeedItem, title: str, full_content: str) -> str:
    """Ask the model for a 10+ minute narrated script with humor and no repetition (plain text)."""
    published = item.pub_date.strftime("%Y-%m-%d %H:%M:%S %Z") if item.pub_date else "unknown"
    categories = ", ".join(item.categories)

    system_msg = (
        "You are a professional content writer who creates engaging, natural scripts. "
        "Write informative content with a confident, conversational tone. You can be casual but "
        "avoid cringy YouTuber clichés and awkward phrases. NEVER use emojis or symbols - plain text only."
    )
    content_info = full_content if full_content else item.description
    user_msg = (
        "Write a narrated script for the YouTube video below (plain text, no markdown).\n\n"
        f"Final Video Title: {title}\n"
        f"Source Article Title: {item.title}\n"
        f"Published: {published}\n"
        f"Categories: {categories}\n"
        f"Summary: {item.description}\n"
        f"Full Article Content: {content_info}\n"
        f"Link: {item.link}\n\n"
        "Script requirements:\n"
        "- At least 10 minutes of narration (target ~1,400–1,800 words).\n"
        "- Start with an engaging, direct hook - avoid cringy openings like 'Welcome back gamers'.\n"
        "- Conversational tone with occasional wit when naturally appropriate.\n"
        "- NEVER repeat the same information twice in different words.\n"
        "- Use clear sections with smooth transitions, end with natural call-to-action.\n"
        "- Be casual and accessible but avoid cringy phrases like 'absolutely insane', 'smash that like button'.\n"
        "- Include specific details from the full article content to add depth and value.\n"
        "- NO awkward YouTuber clichés, excessive enthusiasm, or forced excitement.\n"
    )
    print("[DEBUG] Calling script generation…")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    preview = content[:200].replace('\n', ' ')
    print(f"[DEBUG] Script raw length: {len(content)}; preview: {preview}…")
    script = _strip_code_fences(content)
    print(f"[DEBUG] Extracted script length: {len(script)}")

    # Retry once if clearly too short (< 800 chars)
    if len(script) < 800:
        print("[WARN] Script seems short; retrying with stricter instruction…")
        user_msg_retry = (
            "Write a narrated script for the YouTube video below (plain text, no markdown).\n\n"
            f"Final Video Title: {title}\n"
            f"Source Article Title: {item.title}\n"
            f"Published: {published}\n"
            f"Categories: {categories}\n"
            f"Summary: {item.description}\n"
            f"Full Article Content: {content_info}\n"
            f"Link: {item.link}\n\n"
            "Script requirements:\n"
            "- At least 10 minutes (target ~1,400–1,800 words).\n"
            "- Engaging hook, conversational tone, NO repetition, smooth transitions, natural CTA.\n"
            "- Casual but not cringy, informative, no awkward phrases; include specific article details.\n"
        )
        resp2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg_retry},
            ],
            temperature=0.6,
            max_tokens=2300,
        )
        content2 = resp2.choices[0].message.content if resp2 and resp2.choices else ""
        preview2 = content2[:200].replace('\n', ' ')
        print(f"[DEBUG] Retry script raw length: {len(content2)}; preview: {preview2}…")
        cleaned2 = _strip_code_fences(content2)
        if len(cleaned2) > len(script):
            script = cleaned2
        print(f"[DEBUG] Final script length after retry: {len(script)}")

    return script or ""


def call_for_background_queries(client: Any, model: str, title: str, script: str, item: FeedItem) -> Dict[str, Any]:
    """Ask the model for YouTube search queries for background gameplay footage organized by script sections."""
    system_msg = (
        "You are a video production expert who finds the best background footage for gaming videos. "
        "Analyze the script content and identify the specific games being discussed. Create sections for each major game or topic, "
        "not just generic intro/main/conclusion. Focus on which actual games make sense for background footage."
    )
    user_msg = (
        "Analyze this gaming video script and generate YouTube search queries for background gameplay footage.\n\n"
        f"Video Title: {title}\n"
        f"Full Script: {script}\n\n"
        "Instructions:\n"
        "1. Read through the ENTIRE script to identify all games mentioned\n"
        "2. Create sections for each major game or group of similar games discussed\n"
        "3. If multiple games are covered, create a section for each one (not just 3 sections)\n"
        "4. For single-game topics (like Cyberpunk mods), use general gameplay footage for that game\n"
        "5. Focus on the ACTUAL GAMES being talked about, not generic footage\n"
        "6. Duration estimates should reflect how much time is spent on each game in the script\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "primary_query": "main search query for the dominant game/topic",\n'
        '  "sections": [\n'
        '    {"section_name": "game_name_1", "query": "Game Name gameplay", "duration_estimate": "2m"},\n'
        '    {"section_name": "game_name_2", "query": "Game Name gameplay", "duration_estimate": "1m"},\n'
        '    {"section_name": "game_name_3", "query": "Game Name gameplay", "duration_estimate": "1.5m"}\n'
        '  ]\n'
        "}\n\n"
        "Requirements:\n"
        "- Each section should represent a different game or major topic from the script\n"
        "- Use the actual game names mentioned in the script\n"
        "- For compilation videos (like '26 best games'), create sections for multiple games\n"
        "- For single-game topics, use variations like 'gameplay', 'walkthrough', 'review footage'\n"
        "- Duration estimates should add up to roughly 10 minutes total\n"
        "- Section names should be clear game names or topics, not generic terms\n"
    )
    print("[DEBUG] Calling background queries generation…")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=800,  # Increased for more detailed sections
    )
    content = resp.choices[0].message.content if resp and resp.choices else "{}"
    preview = content[:200].replace('\n', ' ')
    print(f"[DEBUG] Background queries raw length: {len(content)}; preview: {preview}…")
    
    # Parse JSON response
    try:
        import json
        cleaned = _strip_code_fences(content)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            data = {"primary_query": "", "sections": []}
        return data
    except Exception as e:
        print(f"[WARN] Failed to parse background queries JSON: {e}")
        return {"primary_query": "", "sections": []}


def call_for_tags(client: Any, model: str, title: str, script: str) -> List[str]:
    """Ask the model for YouTube tags (plain text, comma-separated in one line)."""
    system_msg = (
        "You are a YouTube SEO expert. Generate relevant tags to maximize reach and searchability."
    )
    user_msg = (
        "Write 12–20 YouTube tags for this video as a single comma-separated line.\n\n"
        f"Title: {title}\n\n"
        "Context excerpt (first ~1000 chars):\n"
        f"{script[:1000]}\n\n"
        "Rules:\n"
        "- Mix short and long phrases; include franchise, platform, topical keywords.\n"
        "- No quotes or special characters; plain text only.\n"
    )
    print("[DEBUG] Calling tags generation…")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_tokens=256,
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    preview = content[:200].replace('\n', ' ')
    print(f"[DEBUG] Tags raw length: {len(content)}; preview: {preview}…")
    line = _first_nonempty_line(content)
    # Split by comma and newline to be safe
    raw_parts = [p for chunk in line.split("\n") for p in chunk.split(",")]
    seen = set()
    tags: List[str] = []
    for p in raw_parts:
        s = p.strip().strip('#').strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            tags.append(s)
    return tags[:20]


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_processed_urls(base_dir: str) -> set:
    """Load URLs that have already been processed from previous runs."""
    processed_file = os.path.join(base_dir, "processed_urls.json")
    if not os.path.exists(processed_file):
        return set()
    try:
        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_urls", []))
    except Exception as e:
        print(f"[WARN] Failed to load processed URLs: {e}")
        return set()


def save_processed_urls(base_dir: str, processed_urls: set) -> None:
    """Save the list of processed URLs to avoid duplicates."""
    processed_file = os.path.join(base_dir, "processed_urls.json")
    os.makedirs(base_dir, exist_ok=True)
    try:
        data = {
            "last_updated": dt.datetime.now(dt.timezone.utc).isoformat(),
            "processed_urls": sorted(list(processed_urls))
        }
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] Saved {len(processed_urls)} processed URLs to {processed_file}")
    except Exception as e:
        print(f"[WARN] Failed to save processed URLs: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clickbaity YouTube briefs from top gaming news (saves JSON files)"
    )
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    parser.add_argument("--limit", type=int, default=20, help="How many top items to use (default: 20)")
    parser.add_argument(
        "--feed",
        action="append",
        help="RSS feed URL(s). If omitted, uses defaults (VG247 + GameSpot)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "briefs"),
        help=(
            "Base directory to save outputs. Each run creates a timestamped subfolder"
            " (default base: outputs/briefs)"
        ),
    )
    args = parser.parse_args()

    # Resolve feeds
    feeds = list(DEFAULT_FEEDS if not args.feed else tuple(args.feed))

    # Build model client
    client, model_id = build_client()

    # Get top items
    print("--- Gathering top items ---")
    top = get_top_items(days=max(1, int(args.days)), limit=max(1, int(args.limit)), feeds=feeds)
    if not top:
        print("No items found.")
        sys.exit(0)

    # Load previously processed URLs to avoid duplicates
    processed_urls = load_processed_urls(args.out_dir)
    print(f"[DEBUG] Found {len(processed_urls)} previously processed URLs")

    # Filter out already processed items
    unprocessed_top = [(score, item) for score, item in top if item.link not in processed_urls]
    if not unprocessed_top:
        print("All top items have already been processed. No new briefs to generate.")
        sys.exit(0)
    
    print(f"[DEBUG] {len(unprocessed_top)} new items to process (out of {len(top)} total)")

    # Save a full dump of selected items for transparency
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    selected_dump = []
    for score, item in unprocessed_top:
        selected_dump.append(
            {
                "score": round(float(score), 3),
                "title": item.title,
                "link": item.link,
                "published": item.pub_date.isoformat() if item.pub_date else None,
                "categories": item.categories,
                "description": item.description,
            }
        )
    items_list_path = os.path.join(run_dir, "top_items.json")
    save_json(items_list_path, {"items": selected_dump})
    print(f"Saved items list → {items_list_path}")

    # Generate briefs (two-step: title → script, then tags)
    print("--- Generating video briefs ---")
    new_processed_urls = set()
    for idx, (score, item) in enumerate(unprocessed_top, start=1):
        idea_title, theme, _ = idea_from_item(item)
        print(f"[{idx}/{len(unprocessed_top)}] {idea_title}")

        # Fetch full article content for richer context
        print(f"[DEBUG] Fetching full content from: {item.link}")
        full_content = fetch_article_content(item.link)
        print(f"[DEBUG] Fetched content length: {len(full_content)} chars")

        # 1) Title with full content context
        gen_title = call_for_title(client, model_id, item, full_content)
        # 2) Script informed by the final title and full article content
        gen_script = call_for_script(client, model_id, item, gen_title, full_content)
        # 3) Background footage queries organized by script sections
        background_queries = call_for_background_queries(client, model_id, gen_title, gen_script, item)
        # 4) Tags informed by title and script
        gen_tags = call_for_tags(client, model_id, gen_title, gen_script)

        # 5) Narrate script and save audio into outputs/narration/<timestamp>/<slug>
        audio_result: Dict[str, Any] = {}
        safe_slug = ""  # will set below
        try:
            if NarrationSettings is not None and narrate_script is not None and gen_script:
                # Prefer env var, else default to templates/voice_template.mp3 if it exists
                voice_sample = os.getenv("NARRATION_VOICE_SAMPLE") or os.path.join("templates", "voice_template.mp3")
                if voice_sample and not os.path.exists(voice_sample):
                    voice_sample = None
                # Mirror the timestamped structure used for JSON outputs
                safe_slug_tmp = (item.title or f"item_{idx}").strip().replace(" ", "_")
                safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe_slug_tmp)[:80]
                audio_base_dir = os.path.join("outputs", "narration", timestamp, safe_slug)
                combined_mp3 = os.path.join(audio_base_dir, f"{safe_slug}_full.mp3")

                print(f"[DEBUG] Narrating script to → {combined_mp3}")
                t0 = time.time()
                # Force CUDA; fallback handled inside narration tool
                narr_settings = NarrationSettings(device="cuda:0", voice_sample_path=voice_sample)
                # Hardcode workers to 40
                fixed_workers = 40
                print(f"[DEBUG] Narration settings → device=cuda:0, workers={fixed_workers}")
                audio_info = narrate_script(
                    gen_script,
                    audio_base_dir,
                    settings=narr_settings,
                    # Higher concurrency for speed; ensure your VRAM can handle it
                    max_workers=fixed_workers,
                    max_chars_per_phrase=350,
                    combine_output_path=combined_mp3,
                    # Make delivery more expressive and varied
                    exaggeration=0.85,
                    temperature=0.80,
                )
                elapsed = time.time() - t0
                print(f"[DEBUG] Narration completed in {elapsed:.2f}s")
                audio_result = {
                    "combined_path": audio_info.get("combined_path"),
                    "segment_paths": audio_info.get("segment_paths", []),
                    "elapsed_seconds": round(float(elapsed), 2),
                }
            else:
                if gen_script:
                    print("[WARN] Narration tool not available; skipping audio generation.")
        except Exception as e: 
            print(f"[WARN] Narration failed: {e}")
            audio_result = {"error": str(e)}

        # 6) If we have a combined MP3, transcribe with Deepgram and save .txt next to it
        try:
            combined_mp3_path = audio_result.get("combined_path") if isinstance(audio_result, dict) else None
            if combined_mp3_path and os.path.exists(combined_mp3_path):
                dg_client = _build_deepgram_client()
                if dg_client is not None:
                    transcript = _transcribe_with_deepgram(dg_client, combined_mp3_path)
                    utterances = _extract_utterances(transcript)
                    txt = _txt_from_utterances(utterances)
                    txt_path = _derive_txt_path(combined_mp3_path)
                    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(txt)
                    print(f"[DEBUG] Saved transcript → {txt_path}")
                else:
                    print("[WARN] Deepgram unavailable; skipped transcription.")
        except Exception as exc:
            print(f"[WARN] Transcription step failed: {exc}")

        # 7) Align background footage sections to narration timestamps and save JSON next to audio
        try:
            combined_mp3_path = audio_result.get("combined_path") if isinstance(audio_result, dict) else None
            if combined_mp3_path and os.path.exists(combined_mp3_path) and gen_script:
                txt_path = _derive_txt_path(combined_mp3_path)
                transcript_lines: List[str] = []
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        transcript_lines = [line.rstrip("\n") for line in f]
                if transcript_lines:
                    align_out_path = os.path.splitext(combined_mp3_path)[0] + "_background_alignment.json"
                    _align_background_to_timestamps(
                        client,
                        model_id,
                        gen_title or (item.title or ""),
                        background_queries,
                        transcript_lines,
                        align_out_path,
                    )
                    print(f"[DEBUG] Saved background alignment → {align_out_path}")
                else:
                    print("[WARN] No transcript text found; skipping background alignment.")
        except Exception as exc:
            print(f"[WARN] Background alignment step failed: {exc}")

        output_obj: Dict[str, Any] = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": {
                "title": item.title,
                "link": item.link,
                "published": item.pub_date.isoformat() if item.pub_date else None,
                "categories": item.categories,
                "description": item.description,
                "full_content": full_content[:1000] + "..." if len(full_content) > 1000 else full_content,
                "score": round(float(score), 3),
            },
            "result": {
                "title": gen_title,
                "script": gen_script,
                "tags": gen_tags,
                "background_footage": background_queries,
            },
            "narration_only": {
                "script": gen_script
            },
            "narration_audio": audio_result,
        }

        slug = (item.title or f"item_{idx}").strip().replace(" ", "_")
        safe_slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in slug)[:80]
        out_path = os.path.join(run_dir, f"brief_{idx:02d}_{safe_slug}.json")
        save_json(out_path, output_obj)
        print(f"Saved → {out_path}")

        # Track this URL as processed
        new_processed_urls.add(item.link)

    # Update the master list of processed URLs under outputs only
    all_processed_urls = processed_urls.union(new_processed_urls)
    save_processed_urls(args.out_dir, all_processed_urls)
    print(f"[DEBUG] Added {len(new_processed_urls)} new URLs to processed list")


if __name__ == "__main__":
    main()


