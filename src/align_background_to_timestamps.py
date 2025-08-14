#!/usr/bin/env python3
"""
Align background footage sections from briefs to narration timestamps using ChatGPT.

Workflow per brief:
- Load brief JSON from outputs/briefs/<timestamp>/brief_*.json
- Read background_footage sections (result.background_footage)
- Locate narration .txt next to the combined MP3 (same base path with .txt)
- Ask ChatGPT to map sections to timestamp windows
- Save <audio_basename>_background_alignment.json in the narration folder

Usage:
    python -m src.clipping.align_background_to_timestamps --latest-run
    python -m src.clipping.align_background_to_timestamps --run-dir outputs/briefs/20250813_203650
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    print("Error: 'openai' package not installed. Install with: pip install openai", file=sys.stderr)
    raise


BRIEFS_BASE_DIR = os.path.join("outputs", "briefs")


def _load_env() -> None:
    if load_dotenv:
        load_dotenv()


def _find_latest_run_dir(base_dir: str) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    entries = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not entries:
        return None
    entries_sorted = sorted(entries)
    return os.path.join(base_dir, entries_sorted[-1])


def _iter_brief_jsons(run_dir: str) -> Iterable[str]:
    for name in os.listdir(run_dir):
        if not name.lower().endswith(".json"):
            continue
        if not name.lower().startswith("brief_"):
            continue
        yield os.path.join(run_dir, name)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_txt_path(combined_mp3_path: str) -> str:
    base, _ext = os.path.splitext(combined_mp3_path)
    return f"{base}.txt"


def _read_transcript_lines(txt_path: str) -> List[str]:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    except Exception as exc:
        print(f"[WARN] Could not read transcript: {txt_path} ({exc})")
        return []


def _build_client() -> tuple[Any, str]:
    _load_env()
    api_key = os.getenv("AIML_API_KEY")
    base_url = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
    model_id = os.getenv("AIML_MODEL_ID", "gpt-4o")
    if not api_key:
        print("Warning: AIML_API_KEY is not set.", file=sys.stderr)
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_id


def _call_alignment_model(client: Any, model: str, title: str, background_sections: Dict[str, Any], transcript_lines: List[str]) -> Dict[str, Any]:
    # Keep prompt compact; pass only necessary fields
    system_msg = (
        "You are a video editor. Map background footage sections to narration timestamps. "
        "Return strict JSON only."
    )
    user_payload = {
        "video_title": title,
        "background_footage": background_sections,
        "transcript": transcript_lines[:2000],  # safety cap
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
    # best-effort JSON parse
    try:
        # strip possible code fences
        cleaned = content.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned)
    except Exception:
        return {"raw": content}


def _save_alignment_json(target_audio_path: str, mapping_obj: Dict[str, Any]) -> str:
    base, _ext = os.path.splitext(target_audio_path)
    out_path = f"{base}_background_alignment.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping_obj, f, ensure_ascii=False, indent=2)
    return out_path


def align_run(run_dir: str) -> None:
    client, model_id = _build_client()

    brief_files = list(_iter_brief_jsons(run_dir))
    if not brief_files:
        print(f"No briefs found in {run_dir}")
        return

    print(f"Found {len(brief_files)} brief(s) in {run_dir}")
    for brief_path in brief_files:
        data = _read_json(brief_path)
        title = ((data.get("result") or {}).get("title") or data.get("source", {}).get("title") or "").strip()
        bg = (data.get("result") or {}).get("background_footage")
        audio_info = data.get("narration_audio", {})
        combined_path = audio_info.get("combined_path") if isinstance(audio_info, dict) else None

        if not bg or not isinstance(bg, dict):
            print(f"[WARN] No background_footage in {os.path.basename(brief_path)}")
            continue
        if not combined_path or not isinstance(combined_path, str):
            print(f"[WARN] No combined narration path in {os.path.basename(brief_path)}")
            continue

        txt_path = _derive_txt_path(combined_path)
        if not os.path.exists(txt_path):
            print(f"[WARN] Transcript not found: {txt_path}. Run transcriber first.")
            continue

        transcript_lines = _read_transcript_lines(txt_path)
        if not transcript_lines:
            print(f"[WARN] Empty transcript: {txt_path}")
            continue

        print(f"Aligning → {os.path.relpath(txt_path)}")
        mapping = _call_alignment_model(client, model_id, title, bg, transcript_lines)
        out_json = _save_alignment_json(combined_path, mapping)
        print(f"  → Saved: {os.path.relpath(out_json)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align background footage sections to narration timestamps")
    parser.add_argument("--run-dir", help=f"Specific run directory under briefs (e.g., {BRIEFS_BASE_DIR}\\20250101_120000)")
    parser.add_argument("--latest-run", action="store_true", help="Select most recent run directory under outputs/briefs")
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.latest_run or not run_dir:
        run_dir = _find_latest_run_dir(BRIEFS_BASE_DIR)
    if not run_dir or not os.path.isdir(run_dir):
        print("Could not determine a valid run directory. Use --run-dir or --latest-run.")
        sys.exit(1)

    align_run(run_dir)


if __name__ == "__main__":
    main()


