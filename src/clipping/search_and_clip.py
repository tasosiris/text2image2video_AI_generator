#!/usr/bin/env python3
"""
Helper script: search YouTube for a phrase, pick the closest match, and feed it to the clipper.

Simple flow:
1) Use yt-dlp to search YouTube for the given query (top N results)
2) Pick the closest match to the query based on title similarity
3) Call the existing clipper (src.clipping.youtube_clipper) with the selected URL

Notes:
- No API keys needed. Relies on yt-dlp's built-in search (ytsearch:).
- Requires yt-dlp and ffmpeg (ffmpeg is used by the clipper).
"""

import argparse
import os
import subprocess
import sys
from difflib import SequenceMatcher
from typing import Dict, List, Optional

try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:
    YoutubeDL = None  # type: ignore


def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube via yt-dlp and return a list of entry dicts (title, url, etc)."""
    if YoutubeDL is None:
        print("Error: yt-dlp is not installed. Install with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    # yt-dlp supports a 'ytsearchN:query' virtual URL that returns the first N results
    search_url = f"ytsearch{max_results}:{query}"
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
        entries = info.get("entries") or []
        return entries


def select_closest_entry(query: str, entries: List[Dict]) -> Optional[Dict]:
    """Pick the entry whose title is most similar to the query (simple ratio)."""
    if not entries:
        return None
    best_entry = None
    best_score = -1.0
    q = query.strip().lower()
    for e in entries:
        title = (e.get("title") or "").strip().lower()
        if not title:
            continue
        score = SequenceMatcher(None, q, title).ratio()
        if score > best_score:
            best_score = score
            best_entry = e
    return best_entry


def run_clipper(
    project_root: str,
    url: str,
    clip_seconds: int,
    seed: Optional[int],
    no_shuffle: bool,
    output: Optional[str],
) -> int:
    """Invoke the clipper module in a subprocess to avoid argparse collisions."""
    cmd: List[str] = [
        sys.executable,
        "-m",
        "src.clipping.youtube_clipper",
        "--url",
        url,
        "--clip-seconds",
        str(max(1, int(clip_seconds))),
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if no_shuffle:
        cmd += ["--no-shuffle"]
    if output:
        cmd += ["--output", output]

    completed = subprocess.run(cmd, cwd=project_root)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search YouTube for a phrase, pick the closest match, and run the clipper."
    )
    parser.add_argument("--query", required=True, help="Search phrase for YouTube")
    parser.add_argument("--max-results", type=int, default=5, help="How many search results to consider (default: 5)")
    parser.add_argument("--clip-seconds", type=int, default=5, help="Seconds per clip (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (optional)")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle; keep original order")
    parser.add_argument("--output", default=None, help="Output file path (optional). Defaults under outputs/clips/")
    args = parser.parse_args()

    # Compute project root so we can run the clipper reliably via -m
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    print(f"--- Searching YouTube for: {args.query!r} ---")
    entries = search_youtube(args.query, max_results=max(1, int(args.max_results)))
    if not entries:
        print("No search results found.", file=sys.stderr)
        sys.exit(1)

    best = select_closest_entry(args.query, entries)
    if not best:
        print("Could not select a best match from search results.", file=sys.stderr)
        sys.exit(1)

    best_title = best.get("title") or "(untitled)"
    best_url = best.get("webpage_url") or best.get("url")
    if not best_url:
        print("Best match has no URL.", file=sys.stderr)
        sys.exit(1)

    print(f"--- Best match: {best_title} ---")
    print(f"URL: {best_url}")

    # Call the clipper with the found URL
    ret = run_clipper(
        project_root=project_root,
        url=best_url,
        clip_seconds=args.clip_seconds,
        seed=args.seed,
        no_shuffle=args.no_shuffle,
        output=args.output,
    )
    if ret != 0:
        sys.exit(ret)


if __name__ == "__main__":
    main()


