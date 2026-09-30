#!/usr/bin/env python3
"""Upload generated video files to a Google Drive folder.

Authentication uses a Google Cloud service account. Set
GOOGLE_APPLICATION_CREDENTIALS to its JSON key file, or set
GDRIVE_SERVICE_ACCOUNT_JSON to the JSON content (useful with notebook secrets).
For a personal Drive, authorize a user with --oauth-client-secrets, then use
the resulting token file or GDRIVE_OAUTH_TOKEN_JSON.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload videos recursively, preserving folders and replacing same-name files."
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="File or directory containing generated videos (not needed with --auth-only)",
    )
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("GDRIVE_FOLDER_ID"),
        help="Destination Google Drive folder ID (or set GDRIVE_FOLDER_ID)",
    )
    parser.add_argument(
        "--folder-name",
        help="Destination folder path under Drive root; it is created if missing",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or Path(".secrets/gauth.json"),
        help="Credential JSON file (default: .secrets/gauth.json)",
    )
    parser.add_argument(
        "--oauth-client-secrets",
        type=Path,
        help="OAuth client JSON; opens a browser and saves .secrets/gauth-token.json",
    )
    parser.add_argument(
        "--oauth-port",
        type=int,
        default=int(os.environ.get("GDRIVE_OAUTH_PORT", "8765")),
        help="Local OAuth callback port (default: 8765; forward this port from your laptop)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List matching videos without connecting to Drive"
    )
    parser.add_argument(
        "--auth-only",
        action="store_true",
        help="Authorize and save an OAuth token without uploading videos",
    )
    return parser.parse_args()


def find_videos(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(path for path in source.rglob("*") if path.is_file())
    else:
        raise ValueError(f"Source must be a regular file or directory: {source}")

    videos = [path for path in candidates if path.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No supported video files found under: {source}")
    return videos


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find_child(service, parent_id: str, name: str, mime_type: str | None = None):
    query = (
        f"'{_escape_drive_query(parent_id)}' in parents and "
        f"name = '{_escape_drive_query(name)}' and trashed = false"
    )
    if mime_type:
        query += f" and mimeType = '{mime_type}'"
    response = (
        service.files()
        .list(
            q=query,
            pageSize=100,
            fields="files(id,name,mimeType)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        .execute()
    )
    return response.get("files", [])


def _ensure_folder(service, parent_id: str, name: str, folder_cache: dict) -> str:
    cache_key = (parent_id, name)
    if cache_key in folder_cache:
        return folder_cache[cache_key]

    matches = _find_child(service, parent_id, name, FOLDER_MIME_TYPE)
    if matches:
        folder_id = matches[0]["id"]
    else:
        created = (
            service.files()
            .create(
                body={"name": name, "mimeType": FOLDER_MIME_TYPE, "parents": [parent_id]},
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        folder_id = created["id"]
        print(f"Created Drive folder: {name}")

    folder_cache[cache_key] = folder_id
    return folder_id


def _destination_folder(service, root_id: str, relative_parent: Path, cache: dict) -> str:
    current_id = root_id
    for part in relative_parent.parts:
        current_id = _ensure_folder(service, current_id, part, cache)
    return current_id


def upload_video(service, video: Path, relative_path: Path, root_id: str, folder_cache: dict) -> None:
    from googleapiclient.http import MediaFileUpload

    parent_id = _destination_folder(service, root_id, relative_path.parent, folder_cache)
    mime_type = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(video), mimetype=mime_type, chunksize=8 * 1024 * 1024, resumable=True)
    existing = [
        item
        for item in _find_child(service, parent_id, video.name)
        if item.get("mimeType") != FOLDER_MIME_TYPE
    ]

    if existing:
        request = service.files().update(
            fileId=existing[0]["id"],
            media_body=media,
            fields="id,name,webViewLink",
            supportsAllDrives=True,
        )
        action = "Updated"
    else:
        request = service.files().create(
            body={"name": video.name, "parents": [parent_id]},
            media_body=media,
            fields="id,name,webViewLink",
            supportsAllDrives=True,
        )
        action = "Uploaded"

    response = None
    while response is None:
        progress, response = request.next_chunk()
        if progress is not None:
            print(f"  {video.name}: {progress.progress() * 100:.0f}%", end="\r", flush=True)

    link = response.get("webViewLink", "")
    print(f"{action}: {relative_path}" + (f"  {link}" if link else ""))


def build_drive_service(
    credentials_path: Path | None,
    credentials_json: str | None,
    oauth_client_secrets: Path | None,
    oauth_port: int = 8765,
):
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive libraries are missing. Install them with: "
            "pip install google-api-python-client google-auth"
        ) from exc

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as UserCredentials

    token_path = Path(credentials_path).expanduser() if credentials_path else None
    if credentials_json:
        info = json.loads(credentials_json)
        if info.get("type") == "service_account":
            credentials = Credentials.from_service_account_info(info, scopes=[DRIVE_SCOPE])
        else:
            credentials = UserCredentials.from_authorized_user_info(info, scopes=[DRIVE_SCOPE])
    elif oauth_client_secrets or (token_path and token_path.is_file()):
        client_path = Path(oauth_client_secrets).expanduser() if oauth_client_secrets else token_path
        info = json.loads(client_path.read_text(encoding="utf-8"))

        if "installed" in info or "web" in info:
            token_path = Path(".secrets/gauth-token.json")
            if not oauth_client_secrets and token_path.is_file():
                credentials = UserCredentials.from_authorized_user_file(
                    str(token_path), scopes=[DRIVE_SCOPE]
                )
            else:
                try:
                    from google_auth_oauthlib.flow import InstalledAppFlow
                except ImportError as exc:
                    raise RuntimeError(
                        "OAuth support is missing. Install it with: "
                        "pip install google-auth-oauthlib"
                    ) from exc
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(client_path), scopes=[DRIVE_SCOPE]
                )
                credentials = flow.run_local_server(port=oauth_port, open_browser=False)
        elif info.get("type") == "service_account":
            credentials = Credentials.from_service_account_file(
                str(client_path), scopes=[DRIVE_SCOPE]
            )
            token_path = None
        else:
            credentials = UserCredentials.from_authorized_user_file(
                str(client_path), scopes=[DRIVE_SCOPE]
            )
            token_path = client_path
    elif os.environ.get("GDRIVE_OAUTH_TOKEN_JSON"):
        info = json.loads(os.environ["GDRIVE_OAUTH_TOKEN_JSON"])
        credentials = UserCredentials.from_authorized_user_info(info, scopes=[DRIVE_SCOPE])
    else:
        credential_hint = f"Credential file not found: {token_path}. " if token_path else ""
        raise ValueError(
            credential_hint
            + "Set GOOGLE_APPLICATION_CREDENTIALS or GDRIVE_SERVICE_ACCOUNT_JSON, "
            "set GDRIVE_OAUTH_TOKEN_JSON, or pass --oauth-client-secrets."
        )

    if not credentials.valid:
        if isinstance(credentials, Credentials) or credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise ValueError("The Google Drive credentials are invalid or expired; authorize again.")
    if token_path and isinstance(credentials, UserCredentials):
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def main() -> int:
    args = parse_args()
    try:
        if args.auth_only:
            build_drive_service(
                args.credentials,
                os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
                or os.environ.get("GDRIVE_OAUTH_TOKEN_JSON"),
                args.oauth_client_secrets,
                args.oauth_port,
            )
            print("Google Drive authorization succeeded.")
            return 0

        if args.source is None:
            raise ValueError("Provide a video source path, or use --auth-only to authenticate.")
        source = args.source.expanduser().resolve()
        videos = find_videos(source)
        if source.is_file():
            base = source.parent
        else:
            base = source

        print(f"Found {len(videos)} video(s) under {source}")
        if args.dry_run:
            for video in videos:
                print(f"  {video.relative_to(base)} ({video.stat().st_size:,} bytes)")
            return 0

        if args.folder_id and args.folder_name:
            raise ValueError("Choose either --folder-id or --folder-name, not both.")
        if not args.folder_id and not args.folder_name:
            raise ValueError("Provide --folder-id, set GDRIVE_FOLDER_ID, or pass --folder-name.")

        service = build_drive_service(
            args.credentials,
            os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
            or os.environ.get("GDRIVE_OAUTH_TOKEN_JSON"),
            args.oauth_client_secrets,
            args.oauth_port,
        )
        folder_cache = {}
        if args.folder_name:
            root_id = _destination_folder(
                service, "root", Path(args.folder_name.strip("/")), folder_cache
            )
        else:
            root_id = args.folder_id
        for video in videos:
            upload_video(service, video, video.relative_to(base), root_id, folder_cache)
        print(f"Finished uploading {len(videos)} video(s).")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Google API exceptions vary by client version.
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
