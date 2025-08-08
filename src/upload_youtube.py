#!/usr/bin/env python3
"""
Upload the latest generated final video to YouTube.

- Discovers the most recent output folder containing `final_video_transitions.mp4`
- Builds title/description from `documentary_data.json` and `narration.txt`
- Uses OAuth2 Installed App flow; stores token under `config/youtube_token.json`
- Optional playlist insertion and thumbnail upload (first scene image) if present

First-time setup:
- Enable YouTube Data API v3 for your Google Cloud project
- Create an OAuth 2.0 Client ID (Desktop app) and download the JSON to `config/client_secret.json`
- Copy `config/youtube_config.example.json` to `config/youtube_config.json` and adjust values
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
from typing import Dict, Optional, Tuple

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


def load_youtube_config() -> Dict:
    """Load config from `config/youtube_config.json` or fall back to defaults."""
    config_dir = _project_root() / "config"
    config_path = config_dir / "youtube_config.json"
    default_config = {
        "client_secrets_file": str(config_dir / "client_secret.json"),
        "token_file": str(config_dir / "youtube_token.json"),
        "default_privacy_status": "private",  # private|unlisted|public
        "category_id": "27",  # Education
        "tags": ["documentary", "history", "ancient greece", "ancient rome"],
        "playlist_id": "",
        "auto_upload": False,
        "upload_thumbnail": True,
    }
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            default_config.update(user_cfg or {})
        except Exception as exc:
            print(f"Warning: Failed to read youtube_config.json: {exc}", file=sys.stderr)
    return default_config


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
        final_video = folder / "final_video_transitions.mp4"
        if final_video.exists() and final_video.stat().st_size > 0:
            return folder, final_video
    return None


def build_title_and_description(output_folder: Path) -> Tuple[str, str, Optional[Path]]:
    """Create YouTube title and description from saved metadata.

    Returns: (title, description, thumbnail_path)
    """
    data_path = output_folder / "documentary_data.json"
    narration_path = output_folder / "narration.txt"
    images_dir = output_folder / "images"
    title = output_folder.name
    theme = None
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("idea") or title
            theme = data.get("theme") or None
        except Exception as exc:
            print(f"Warning: Failed to parse {data_path}: {exc}", file=sys.stderr)
    description_lines = [
        f"Published: {datetime.utcnow().strftime('%Y-%m-%d')} (UTC)",
    ]
    if theme:
        description_lines.append(f"Theme: {theme}")
    if narration_path.exists():
        try:
            with open(narration_path, "r", encoding="utf-8") as f:
                narration = f.read().strip()
            description_lines.append("")
            description_lines.append("Narration transcript:")
            description_lines.append(narration)
        except Exception as exc:
            print(f"Warning: Failed to read narration: {exc}", file=sys.stderr)
    description_lines.append("")
    description_lines.append("Generated automatically.")
    description = "\n".join(description_lines)

    # Prefer first scene image as thumbnail if exists
    thumbnail_path: Optional[Path] = None
    if images_dir.exists():
        for name in sorted(os.listdir(images_dir)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                thumbnail_path = images_dir / name
                break
    return title, description, thumbnail_path


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
        "status": {"privacyStatus": privacy_status},
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


def upload_latest(output_folder: Optional[str] = None, dry_run: bool = False) -> Optional[str]:
    cfg = load_youtube_config()
    if dry_run:
        print("[DRY RUN] No changes will be made.")
    if output_folder:
        out_dir = Path(output_folder)
        final_tuple = (out_dir, out_dir / "final_video_transitions.mp4")
        if not final_tuple[1].exists():
            print(f"Error: final video not found at {final_tuple[1]}", file=sys.stderr)
            return None
    else:
        result = find_latest_final_video()
        if not result:
            print("Error: No final video found in outputs.", file=sys.stderr)
            return None
        final_tuple = result

    output_dir, final_video_path = final_tuple
    title, description, thumbnail_path = build_title_and_description(output_dir)
    print(f"Uploading: {final_video_path}")
    print(f"Title: {title}")
    print(f"Privacy: {cfg['default_privacy_status']}")
    if dry_run:
        return None

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
            privacy_status=str(cfg.get("default_privacy_status", "private")),
            category_id=str(cfg.get("category_id", "27")),
            tags=list(cfg.get("tags", [])),
        )
        print(f"✓ Uploaded. Video ID: {video_id}")

        # Persist upload info in the project folder
        info = {
            "video_id": video_id,
            "uploaded_at_utc": datetime.utcnow().isoformat() + "Z",
            "privacy": cfg.get("default_privacy_status", "private"),
        }
        with open(output_dir / "youtube_upload.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        if cfg.get("upload_thumbnail", True) and thumbnail_path and thumbnail_path.exists():
            try:
                set_thumbnail(youtube, video_id, thumbnail_path)
                print("✓ Thumbnail set")
            except HttpError as e:  # type: ignore
                print(f"Warning: Failed to set thumbnail: {e}", file=sys.stderr)

        playlist_id = str(cfg.get("playlist_id") or "").strip()
        if playlist_id:
            try:
                add_to_playlist(youtube, video_id, playlist_id)
                print(f"✓ Added to playlist: {playlist_id}")
            except HttpError as e:  # type: ignore
                print(f"Warning: Failed to add to playlist: {e}", file=sys.stderr)

        return video_id
    except HttpError as e:  # type: ignore
        print(f"YouTube API error: {e}", file=sys.stderr)
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Upload the latest final video to YouTube")
    parser.add_argument("--output-dir", type=str, default=None, help="Specific output folder to upload from")
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow only and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    args = parser.parse_args(argv)

    cfg = load_youtube_config()
    if args.auth:
        try:
            get_authenticated_service(cfg["client_secrets_file"], cfg["token_file"])
            print("Auth successful.")
            return 0
        except SystemExit:
            return 1
        except Exception as exc:
            print(f"Auth failed: {exc}", file=sys.stderr)
            return 1

    video_id = upload_latest(output_folder=args.output_dir, dry_run=args.dry_run)
    return 0 if video_id else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

