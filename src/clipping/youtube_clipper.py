#!/usr/bin/env python3
"""
Simple YouTube clipper script using ffmpeg (no moviepy).

What it does:
1) Downloads a YouTube video to `outputs/clips/downloads/` using yt-dlp
2) Encodes and splits the video into N-second segments (default 5s) with ffmpeg
3) Randomly rearranges the segments
4) Concatenates segments into a single Full HD (1920x1080) video with ffmpeg
5) Saves final video to `outputs/clips/`

Usage examples:
  python -m src.clipping.youtube_clipper --url https://youtu.be/VIDEO --clip-seconds 5
  python -m src.clipping.youtube_clipper --url "https://www.youtube.com/watch?v=..." --clip-seconds 8 --seed 123

Notes:
- Requires yt-dlp and ffmpeg installed and on PATH. See requirements in project root.
- Keeps it simple with clear console output.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List

try:
    # yt_dlp is preferred over youtube_dl
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:
    YoutubeDL = None  # type: ignore


def ensure_directories(project_root: str) -> dict:
    """Ensure required directories exist and return their paths."""
    outputs_root = os.path.join(project_root, "outputs")
    clips_root = os.path.join(outputs_root, "clips")
    downloads_root = os.path.join(clips_root, "downloads")
    os.makedirs(downloads_root, exist_ok=True)
    return {
        "outputs_root": outputs_root,
        "clips_root": clips_root,
        "downloads_root": downloads_root,
    }


def download_youtube_video(url: str, downloads_root: str) -> str:
    """
    Download the YouTube video and return the path to the downloaded file.
    Uses yt-dlp with a simple mp4/mp3+mp4 best effort.
    """
    if YoutubeDL is None:
        print("Error: yt-dlp is not installed. Install with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    ydl_opts = {
        # Prefer best mp4. If not available, remux to mp4 container
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "outtmpl": os.path.join(downloads_root, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Determine output path
        if "requested_downloads" in info and info["requested_downloads"]:
            # yt-dlp returns a list of downloads with filenames
            filename = info["requested_downloads"][0].get("_filename")
            if filename and os.path.isfile(filename):
                return filename
        # Fallback: build from info data
        title = info.get("title", "video")
        ext = info.get("ext", "mp4")
        candidate = os.path.join(downloads_root, f"{title}.{ext}")
        if os.path.isfile(candidate):
            return candidate
        # As a last resort, search for most recent file in downloads_root
        candidates = [
            os.path.join(downloads_root, f)
            for f in os.listdir(downloads_root)
            if os.path.isfile(os.path.join(downloads_root, f))
        ]
        if not candidates:
            print("Error: Could not locate downloaded file.", file=sys.stderr)
            sys.exit(1)
        return max(candidates, key=os.path.getmtime)

def ensure_ffmpeg_available() -> None:
    """Check that ffmpeg is available on PATH; exit with message if not."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH. Install ffmpeg and try again.", file=sys.stderr)
        if sys.platform.startswith("win"):
            print("Tip (Windows): winget install Gyan.FFmpeg", file=sys.stderr)
        sys.exit(1)


def run_ffmpeg(args: List[str]) -> None:
    """Run an ffmpeg command and raise on non-zero exit."""
    completed = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(args)}")


def segment_with_ffmpeg(
    input_path: str,
    working_dir: str,
    seconds_per_clip: int,
) -> List[str]:
    """
    Use ffmpeg to encode to 1080p@30fps and segment into N-second clips.
    Returns a list of segment file paths (sorted by index).
    """
    os.makedirs(working_dir, exist_ok=True)

    # Output filename pattern in working dir
    segment_pattern = os.path.join(working_dir, "segment_%04d.mp4")

    # Single pass: encode + segment. Force keyframes at segment boundaries to ensure clean splitting.
    # -vf: scale to fit within 1920x1080 and pad to exactly 1920x1080
    vf_filter = (
        "scale=1920:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
        "fps=30"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-force_key_frames",
        f"expr:gte(t,n_forced*{seconds_per_clip})",
        "-f",
        "segment",
        "-segment_time",
        str(seconds_per_clip),
        "-reset_timestamps",
        "1",
        segment_pattern,
    ]

    run_ffmpeg(ffmpeg_cmd)

    # Collect segment files
    segment_files = [
        os.path.join(working_dir, f)
        for f in os.listdir(working_dir)
        if f.startswith("segment_") and f.endswith(".mp4")
    ]
    segment_files.sort()

    if not segment_files:
        raise RuntimeError("No segments were produced by ffmpeg.")

    return segment_files


def build_output_filename(clips_root: str, base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base_name)
    return os.path.join(clips_root, f"{safe_base}_shuffled_{timestamp}.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a YouTube video, split into clips, shuffle, and render to 1080p using ffmpeg.")
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--clip-seconds", type=int, default=5, help="Seconds per clip (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling (optional)")
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle; keep original order")
    parser.add_argument("--output", default=None, help="Output file path (optional). Defaults under outputs/clips/")
    args = parser.parse_args()

    # Seed randomness for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Resolve project root from this file location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # Ensure directories and ffmpeg
    paths = ensure_directories(project_root)
    ensure_ffmpeg_available()

    print("--- Downloading video ---")
    video_path = download_youtube_video(args.url, paths["downloads_root"])
    print(f"Downloaded: {video_path}")

    # Create a temporary working directory under clips root
    with tempfile.TemporaryDirectory(prefix="ytclip_", dir=paths["clips_root"]) as working_dir:
        print("--- Encoding and segmenting with ffmpeg ---")
        segments = segment_with_ffmpeg(
            input_path=video_path,
            working_dir=working_dir,
            seconds_per_clip=max(1, int(args.clip_seconds)),
        )
        print(f"Created {len(segments)} clips of ~{args.clip_seconds}s each (last may be shorter).")

        if not args.no_shuffle:
            print("--- Shuffling clips ---")
            random.shuffle(segments)
        else:
            print("--- Keeping original order ---")

        # Determine output path
        if args.output:
            output_path = args.output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = build_output_filename(paths["clips_root"], base_name)

        # Always apply smooth transitions if we have 2+ segments; else copy the single segment
        if len(segments) == 1:
            print("--- Single segment only; copying to output ---")
            concat_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                segments[0],
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                output_path,
            ]
            run_ffmpeg(concat_cmd)
            print(f"Saved: {output_path}")
        else:
            print("--- Concatenating with smooth transitions (ffmpeg xfade) ---")

            def probe_duration(path: str) -> float:
                # Use ffprobe to get media duration in seconds (float)
                cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ]
                try:
                    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    return float(out.decode("utf-8").strip())
                except Exception:
                    return 0.0

            durations = [probe_duration(s) for s in segments]
            # Fixed base transition duration
            transition_sec = 0.5
            # Guard transition duration so it is not longer than the shortest clip / 2
            min_dur = max(0.01, min(d for d in durations if d > 0) if any(durations) else 0.5)
            td = min(transition_sec, max(0.05, min_dur * 0.5))

            # Build ffmpeg command with all inputs
            cmd: List[str] = ["ffmpeg", "-y"]
            for s in segments:
                cmd += ["-i", s]

            # Build filter_complex chaining xfade/acrossfade
            filter_parts: List[str] = []
            # Initial cumulative duration is duration of first clip
            cumulative = durations[0] if durations else 0.0
            last_v = None
            last_a = None

            for idx in range(1, len(segments)):
                # Labels for inputs
                if idx == 1:
                    in1_v = f"[0:v]"
                    in1_a = f"[0:a]"
                else:
                    in1_v = f"[{last_v}]"
                    in1_a = f"[{last_a}]"
                in2_v = f"[{idx}:v]"
                in2_a = f"[{idx}:a]"

                out_v = f"v{idx:02d}"
                out_a = f"a{idx:02d}"

                # xfade offset is where fade starts on first input
                offset = max(0.0, cumulative - td)
                # Video crossfade
                filter_parts.append(
                    f"{in1_v}{in2_v} xfade=transition=fade:duration={td}:offset={offset} [{out_v}]"
                )
                # Audio crossfade
                filter_parts.append(f"{in1_a}{in2_a} acrossfade=d={td} [{out_a}]")

                # Update labels and cumulative duration for next step
                last_v = out_v
                last_a = out_a
                cumulative = cumulative + durations[idx] - td

            filter_complex = "; ".join(filter_parts)

            cmd += [
                "-filter_complex",
                filter_complex,
                "-map",
                f"[{last_v}]",
                "-map",
                f"[{last_a}]",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                output_path,
            ]
            run_ffmpeg(cmd)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


