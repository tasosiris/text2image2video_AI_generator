#!/usr/bin/env python3
"""
Create tripled-duration versions of generated scene videos.

For each input clip, the output plays forward, then in reverse, then forward again.

Defaults:
- Looks for the latest output folder under `outputs/` by highest generation number (folder names like `NN_Title`).
- Processes all `scene_*.mp4` files in its `videos/` subfolder.
- Writes results to a sibling `tripled_videos/` subfolder.

Usage examples:
  python src/triple_videos.py
  python src/triple_videos.py --output-folder "outputs/35_..." \
      --pattern "scene_*.mp4"
  python src/triple_videos.py --videos-dir "outputs/35_.../videos" \
      --out-dir "outputs/35_.../tripled_videos"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
from typing import Optional, Tuple
import re


def parse_generation_number(folder_name: str) -> Optional[int]:
    """Extract leading integer generation number from a folder name like '35_Title'."""
    m = re.match(r"^(\d+)_", folder_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def find_latest_output_folder(base_outputs_dir: str) -> Optional[str]:
    """Find the output folder with the highest generation number prefix."""
    try:
        candidates = []
        for d in os.listdir(base_outputs_dir):
            path = os.path.join(base_outputs_dir, d)
            if not os.path.isdir(path):
                continue
            gen = parse_generation_number(d)
            if gen is not None:
                candidates.append((gen, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    except Exception:
        return None


def has_audio_stream(input_path: str) -> bool:
    """Return True if the input file contains at least one audio stream."""
    try:
        # Use ffprobe to detect audio streams
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "json",
            input_path,
        ]
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        return len(streams) > 0
    except Exception:
        return False


def build_ffmpeg_triple_cmd(input_path: str, output_path: str, include_audio: bool) -> list[str]:
    """Construct an ffmpeg command for forward-reverse-forward playback."""
    if include_audio:
        filter_complex = (
            "[0:v]split=3[v0][v1][v2];"
            "[v1]reverse[v1r];"
            "[v0][v1r][v2]concat=n=3:v=1:a=0[vout];"
            "[0:a]asplit=3[a0][a1][a2];"
            "[a1]areverse[a1r];"
            "[a0][a1r][a2]concat=n=3:v=0:a=1[aout]"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    else:
        filter_complex = (
            "[0:v]split=3[v0][v1][v2];"
            "[v1]reverse[v1r];"
            "[v0][v1r][v2]concat=n=3:v=1:a=0[vout]"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    return cmd


def process_single_clip(input_path: str, output_path: str) -> Tuple[str, bool, Optional[str]]:
    """Process a single clip. Returns (basename, success, error_message)."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        include_audio = has_audio_stream(input_path)
        cmd = build_ffmpeg_triple_cmd(input_path, output_path, include_audio)
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            return (os.path.basename(input_path), False, run.stderr)
        return (os.path.basename(input_path), True, None)
    except Exception as e:
        return (os.path.basename(input_path), False, str(e))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Triple clip duration with forward-reverse-forward playback.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output-folder", type=str, default=None,
                       help="Path to a specific documentary output folder (containing videos/). If omitted, the latest under outputs/ is used.")
    group.add_argument("--videos-dir", type=str, default=None,
                       help="Direct path to a folder with input videos (overrides --output-folder).")

    parser.add_argument("--pattern", type=str, default="scene_*.mp4",
                        help="Glob pattern of clips to process within the videos dir (default: scene_*.mp4)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for tripled videos. Defaults to <output-folder>/tripled_videos or sibling of --videos-dir.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present.")

    args = parser.parse_args(argv)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    base_outputs_dir = os.path.join(project_root, "outputs")

    if args.videos_dir:
        videos_dir = args.videos_dir
        output_folder = os.path.dirname(videos_dir)
    else:
        output_folder = args.output_folder
        if not output_folder:
            output_folder = find_latest_output_folder(base_outputs_dir)
            if not output_folder:
                print("Error: No outputs found and no --output-folder provided.", file=sys.stderr)
                return 1
        videos_dir = os.path.join(output_folder, "videos")

    if not os.path.isdir(videos_dir):
        print(f"Error: videos directory not found: {videos_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or os.path.join(output_folder, "tripled_videos")
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(videos_dir, args.pattern)
    input_files = sorted(glob.glob(pattern))
    if not input_files:
        print(f"No files matched pattern '{args.pattern}' in {videos_dir}")
        return 1

    print(f"Found {len(input_files)} clips. Writing tripled clips to: {out_dir}")

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for in_path in input_files:
            base = os.path.basename(in_path)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(out_dir, f"{name}_triple{ext}")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"  -> Skipping existing: {os.path.basename(out_path)}")
                continue
            tasks.append(executor.submit(process_single_clip, in_path, out_path))

        completed = 0
        failures = 0
        for fut in concurrent.futures.as_completed(tasks):
            base, ok, err = fut.result()
            completed += 1
            if ok:
                print(f"✓ {base}")
            else:
                failures += 1
                print(f"✗ {base}: {err}", file=sys.stderr)

    print(f"Done. Successful: {completed - failures}, Failed: {failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
