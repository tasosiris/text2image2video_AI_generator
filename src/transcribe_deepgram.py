#!/usr/bin/env python3
"""
Transcribe narration audio produced by `src/clipping/generate_video_briefs.py` using Deepgram.

Features:
- Reads `DEEPGRAM_API_KEY` from the `.env`
- Finds briefs in `outputs/briefs/<timestamp>/brief_*.json`
- For each brief, locates `narration_audio.combined_path` (or per-segment paths)
- Sends audio to Deepgram prerecorded transcription with utterances and word timestamps
- Saves a plain text transcript with timestamps alongside the audio file

Usage examples:
    python -m src.clipping.transcribe_deepgram --latest-run
    python -m src.clipping.transcribe_deepgram --run-dir outputs/briefs/20250813_161601
    python -m src.clipping.transcribe_deepgram --include-segments

Outputs (next to each audio file):
    <audio_basename>.txt             # Plain text: "HH:MM:SS,mmm --> HH:MM:SS,mmm  text"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

try:
    from deepgram import DeepgramClient, PrerecordedOptions  # type: ignore
except Exception as exc:
    print(
        "Error: deepgram-sdk is not installed. Install with: pip install deepgram-sdk",
        file=sys.stderr,
    )
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
    # Sort by dirname (timestamp-like), fallback to mtime for robustness
    entries_sorted = sorted(entries)
    latest = entries_sorted[-1]
    return os.path.join(base_dir, latest)


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


def _build_txt_from_utterances(utterances: List[Dict[str, Any]]) -> str:
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


def _gather_audio_targets(brief_json_path: str, include_segments: bool) -> List[str]:
    data = _read_json(brief_json_path)
    audio_info = data.get("narration_audio", {}) if isinstance(data, dict) else {}
    targets: List[str] = []

    combined = audio_info.get("combined_path")
    if isinstance(combined, str) and combined:
        targets.append(combined)

    if include_segments:
        for seg in audio_info.get("segment_paths", []) or []:
            if isinstance(seg, str) and seg:
                targets.append(seg)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_targets: List[str] = []
    for p in targets:
        if p not in seen:
            seen.add(p)
            unique_targets.append(p)
    return unique_targets


def _transcribe_file(client: DeepgramClient, audio_path: str) -> Dict[str, Any]:
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
        response = client.listen.rest.v("1").transcribe_file(source, options)
        # response is a SDK object with to_dict/to_json
        try:
            return response.to_dict()
        except Exception:
            try:
                return json.loads(response.to_json())
            except Exception:
                # Fallback: return raw response
                return {"raw": str(response)}


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


def _derive_text_output_path(audio_path: str) -> str:
    base, _ext = os.path.splitext(audio_path)
    return f"{base}.txt"


def _extract_utterances(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Deepgram v2 JSON: transcript["results"]["utterances"]
    try:
        return (
            transcript
            .get("results", {})
            .get("utterances", [])
        ) or []
    except Exception:
        return []


def _already_done(txt_out: str) -> bool:
    return os.path.exists(txt_out) and os.path.getsize(txt_out) > 0


def transcribe_run(run_dir: str, include_segments: bool = False, force: bool = False) -> None:
    _load_env()
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Error: DEEPGRAM_API_KEY not set in environment/.env", file=sys.stderr)
        sys.exit(1)

    client = DeepgramClient(api_key)

    brief_files = list(_iter_brief_jsons(run_dir))
    if not brief_files:
        print(f"No briefs found in {run_dir}")
        return

    print(f"Found {len(brief_files)} brief(s) in {run_dir}")
    total_audio = 0
    for brief_path in brief_files:
        targets = _gather_audio_targets(brief_path, include_segments=include_segments)
        if not targets:
            continue
        for audio_path in targets:
            total_audio += 1
            txt_out = _derive_text_output_path(audio_path)
            if not force and _already_done(txt_out):
                print(f"Skip (exists): {os.path.relpath(audio_path)}")
                continue
            try:
                print(f"Transcribing: {os.path.relpath(audio_path)}")
                transcript = _transcribe_file(client, audio_path)
                utterances = _extract_utterances(transcript)
                txt = _build_txt_from_utterances(utterances)
                os.makedirs(os.path.dirname(txt_out), exist_ok=True)
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(txt)
                print(f"  → Saved: {os.path.relpath(txt_out)}")
            except Exception as exc:
                print(f"Failed: {audio_path} ({exc})")

    if total_audio == 0:
        print("No narration audio found to transcribe.")
    else:
        print(f"Done. Processed {total_audio} audio file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe narration audio from generate_video_briefs.py using Deepgram",
    )
    parser.add_argument(
        "--run-dir",
        help=f"Specific run directory under briefs (e.g., {BRIEFS_BASE_DIR}\\20250101_120000)",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Automatically select the most recent run directory under briefs",
    )
    parser.add_argument(
        "--include-segments",
        action="store_true",
        help="Also transcribe per-segment MP3s in addition to the combined track",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if outputs already exist",
    )
    args = parser.parse_args()

    target_run_dir = args.run_dir
    if args.latest_run or not target_run_dir:
        target_run_dir = _find_latest_run_dir(BRIEFS_BASE_DIR)
    if not target_run_dir or not os.path.isdir(target_run_dir):
        print("Could not determine a valid run directory. Use --run-dir or --latest-run.")
        sys.exit(1)

    transcribe_run(target_run_dir, include_segments=args.include_segments, force=args.force)


if __name__ == "__main__":
    main()


