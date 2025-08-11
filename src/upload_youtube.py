#!/usr/bin/env python3
"""
Upload the latest generated final video to YouTube.

- Discovers the most recent output folder containing `final_video_transitions.mp4`
- Generates a clickbaity SEO title, dense keyword tags, and a keyword-heavy description
  using AIMLAPI (OpenAI-compatible) based on the project's metadata
- Uses OAuth2 Installed App flow; stores token under `config/youtube_token.json`
- Sets video to public and NOT made for kids
- Optionally uploads a thumbnail (first scene image if present)

First-time setup:
- Enable YouTube Data API v3 for your Google Cloud project
- Create an OAuth 2.0 Client ID (Desktop app) and download the JSON to `config/client_secret.json`
- Install deps: `pip install -r requirements.youtube.txt`

Usage:
  python src/upload_youtube.py                # Upload latest final video
  python src/upload_youtube.py --output-dir C:\path\to\outputs\...   # Upload from a specific output folder
  python src/upload_youtube.py --auth         # Run auth only (no upload)

Programmatic use from the pipeline:
  from upload_youtube import upload_latest
  upload_latest(output_folder=...)  # or upload_latest() to auto-detect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from openai import OpenAI

# AIMLAPI (OpenAI-compatible) client for metadata generation
AIML_API_KEY = os.getenv("AIML_API_KEY")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL_ID = os.getenv("AIML_MODEL_ID", "gpt-4o")
try:
    aiml_client: Optional[OpenAI] = OpenAI(api_key=AIML_API_KEY, base_url=AIML_BASE_URL) if AIML_API_KEY else None
except Exception:
    aiml_client = None

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except Exception:  # Defer import errors to runtime messages if user hasn't installed deps
    Credentials = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    build = None  # type: ignore
    HttpError = Exception  # type: ignore
    MediaFileUpload = None  # type: ignore


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


def _project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent


def _outputs_dir() -> Path:
    return _project_root() / "outputs"


def _find_video_in_folder(folder: Path) -> Optional[Path]:
    """Finds the final video file in a folder, preferring the version with music."""
    video_with_music = folder / "final_video_with_music.mp4"
    if video_with_music.exists() and video_with_music.stat().st_size > 0:
        return video_with_music

    video_transitions = folder / "final_video_transitions.mp4"
    if video_transitions.exists() and video_transitions.stat().st_size > 0:
        return video_transitions

    return None


def load_youtube_config() -> Dict:
    """Compatibility shim: return built-in defaults (no external config file)."""
    config_dir = _project_root() / "config"
    return {
        "client_secrets_file": str(config_dir / "client_secret.json"),
        "token_file": str(config_dir / "youtube_token.json"),
        "category_id": "27",  # Education
        "upload_thumbnail": True,
        # Kept for compatibility, ignored behaviorally
        "default_privacy_status": "public",
        "tags": [],
        "playlist_id": "",
        "auto_upload": True,
    }


def find_latest_final_video() -> Optional[Tuple[Path, Path]]:
    """Find the newest output folder that contains a final video.

    Returns: (output_folder, final_video_path) or None
    """
    outputs = _outputs_dir()
    if not outputs.exists():
        return None
    candidate_folders = [p for p in outputs.iterdir() if p.is_dir()]
    if not candidate_folders:
        return None
    # Sort by modified time desc
    candidate_folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in candidate_folders:
        final_video = _find_video_in_folder(folder)
        if final_video:
            return folder, final_video
    return None


def _read_project_metadata(output_folder: Path) -> Tuple[str, Optional[str], str]:
    """Return (idea_or_folder_name, theme, narration_text)."""
    data_path = output_folder / "documentary_data.json"
    narration_path = output_folder / "narration.txt"
    idea = output_folder.name
    theme: Optional[str] = None
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            idea = data.get("idea") or idea
            theme = data.get("theme") or None
        except Exception as exc:
            print(f"Warning: Failed to parse {data_path}: {exc}", file=sys.stderr)
    narration_text = ""
    if narration_path.exists():
        try:
            with open(narration_path, "r", encoding="utf-8") as f:
                narration_text = f.read().strip()
        except Exception as exc:
            print(f"Warning: Failed to read narration: {exc}", file=sys.stderr)
    return idea, theme, narration_text


def _truncate_tags_to_limit(tags: List[str], max_total_chars: int = 470) -> List[str]:
    """YouTube tags total length limit is ~500 chars. Keep within a safe bound."""
    out: List[str] = []
    total = 0
    for t in tags:
        t = (t or "").strip()
        if not t:
            continue
        candidate = total + len(t) + (1 if out else 0)
        if candidate > max_total_chars:
            break
        out.append(t)
        total = candidate
    return out


def ai_generate_youtube_metadata(output_folder: Path) -> Tuple[str, List[str], str, Optional[Path]]:
    """Generate (title, tags, description, thumbnail_path) using AIMLAPI.

    Falls back to simple defaults if AIMLAPI is unavailable.
    """
    images_dir = output_folder / "images"
    idea, theme, narration = _read_project_metadata(output_folder)

    # Thumbnail: prefer first scene image
    thumbnail_path: Optional[Path] = None
    if images_dir.exists():
        for name in sorted(os.listdir(images_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                thumbnail_path = images_dir / name
                break

    if aiml_client is None:
        base_tags = [
            "documentary", "history", "ancient greece", "ancient rome",
            "greek history", "roman empire", "classical antiquity",
        ]
        title = idea
        desc = f"Keywords: {', '.join(base_tags)}"
        return title, _truncate_tags_to_limit(base_tags), desc, thumbnail_path

    meta_context = {
        "idea": idea,
        "theme": theme or "",
        "narration_excerpt": narration[:1200],
    }
    prompt = (
        "You are a YouTube SEO expert. Create high-CTR metadata for a public video.\n"
        "Return STRICT JSON with keys: title, tags (array), description.\n\n"
        "Requirements:\n"
        "- Title: ultra-clickbaity but accurate, <= 95 chars, maximize views, no emojis.\n"
        "- Tags: many concise, relevant keywords/phrases; include variations and long-tails; no hashtags.\n"
        "- Description: very short, skimmable, keyword-heavy bullets or lines. Minimal prose; no links.\n"
        "- Audience: Ancient Greek–Roman history.\n\n"
        f"Context JSON (for guidance):\n{json.dumps(meta_context, ensure_ascii=False)}\n"
    )

    try:
        resp = aiml_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You write viral YouTube SEO metadata. Always output valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
        )
        content = resp.model_dump()["choices"][0]["message"]["content"]
        data = json.loads(content)
        title = str(data.get("title") or idea).strip()
        tags: List[str] = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
        description = str(data.get("description") or idea).strip()
        return title, _truncate_tags_to_limit(tags), description, thumbnail_path
    except Exception as exc:
        print(f"Warning: AI metadata generation failed: {exc}", file=sys.stderr)
        fallback_tags = [
            "documentary", "history", "ancient greece", "ancient rome",
            "greek history", "roman empire", "classical antiquity",
        ]
        return idea, _truncate_tags_to_limit(fallback_tags), idea, thumbnail_path


def _ensure_google_libs_available() -> None:
    if any(x is None for x in (Credentials, InstalledAppFlow, build, MediaFileUpload)):
        print(
            "Error: Google API libraries are not installed.\n"
            "Install with: pip install -r requirements.youtube.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)


def get_authenticated_service(client_secrets_file: str, token_file: str):
    _ensure_google_libs_available()
    creds = None
    token_path = Path(token_file)
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(request=None)  # type: ignore[arg-type]
            except Exception:
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def _resumable_upload(request):
    """Perform a resumable upload and return the response dict."""
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status is not None:
            print(f"Uploading: {int(status.progress() * 100)}%")
    return response


def upload_video(
    youtube,
    video_path: Path,
    title: str,
    description: str,
    privacy_status: str,
    category_id: str,
    tags: Optional[list[str]] = None,
) -> str:
    body = {
        "snippet": {
            "title": title[:100],  # YouTube title limit 100 chars
            "description": description[:5000],  # YouTube description limit 5000 chars
            "categoryId": category_id,
            "tags": tags or [],
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = _resumable_upload(request)
    video_id = response.get("id")
    if not video_id:
        raise RuntimeError("YouTube upload failed: No video ID returned")
    return video_id


def set_thumbnail(youtube, video_id: str, thumbnail_path: Path) -> None:
    media = MediaFileUpload(str(thumbnail_path))
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()


def add_to_playlist(youtube, video_id: str, playlist_id: str) -> None:
    body = {
        "snippet": {
            "playlistId": playlist_id,
            "resourceId": {"kind": "youtube#video", "videoId": video_id},
        }
    }
    youtube.playlistItems().insert(part="snippet", body=body).execute()


def upload_latest(output_folder: Optional[str] = None) -> Optional[str]:
    cfg = load_youtube_config()
    output_dir: Optional[Path] = None
    final_video_path: Optional[Path] = None

    if output_folder:
        out_dir = Path(output_folder)
        final_video_path = _find_video_in_folder(out_dir)
        if not final_video_path:
            print(f"Error: final video not found in {out_dir}", file=sys.stderr)
            return None
        output_dir = out_dir
    else:
        result = find_latest_final_video()
        if not result:
            print("Error: No final video found in outputs.", file=sys.stderr)
            return None
        output_dir, final_video_path = result

    title, tags, description, thumbnail_path = ai_generate_youtube_metadata(output_dir)
    print(f"Uploading: {final_video_path}")
    print(f"Title: {title}")
    print("Privacy: public (not made for kids)")

    try:
        youtube = get_authenticated_service(
            client_secrets_file=cfg["client_secrets_file"],
            token_file=cfg["token_file"],
        )
    except SystemExit:
        return None
    except Exception as exc:
        print(f"Auth failed: {exc}", file=sys.stderr)
        return None

    try:
        video_id = upload_video(
            youtube=youtube,
            video_path=final_video_path,
            title=title,
            description=description,
            privacy_status="public",
            category_id=str(cfg.get("category_id", "27")),
            tags=tags,
        )
        print(f"✓ Uploaded. Video ID: {video_id}")

        # Persist upload info in the project folder
        info = {
            "video_id": video_id,
            "uploaded_at_utc": datetime.utcnow().isoformat() + "Z",
            "privacy": "public",
        }
        with open(output_dir / "youtube_upload.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        if cfg.get("upload_thumbnail", True) and thumbnail_path and thumbnail_path.exists():
            try:
                set_thumbnail(youtube, video_id, thumbnail_path)
                print("✓ Thumbnail set")
            except HttpError as e:  # type: ignore
                print(f"Warning: Failed to set thumbnail: {e}", file=sys.stderr)

        return video_id
    except HttpError as e:  # type: ignore
        print(f"YouTube API error: {e}", file=sys.stderr)
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Upload the latest final video to YouTube")
    parser.add_argument("--output-dir", type=str, default=None, help="Specific output folder to upload from")
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow only and exit")
    # no dry-run; pipeline always uploads
    args = parser.parse_args(argv)

    if args.auth:
        try:
            cfg = load_youtube_config()
            get_authenticated_service(cfg["client_secrets_file"], cfg["token_file"])
            print("Auth successful.")
            return 0
        except SystemExit:
            return 1
        except Exception as exc:
            print(f"Auth failed: {exc}", file=sys.stderr)
            return 1

    video_id = upload_latest(output_folder=args.output_dir)
    return 0 if video_id else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

