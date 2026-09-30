#!/usr/bin/env python3
"""Download all files from a public Google Drive folder."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively download a public Google Drive folder, with resume support."
    )
    parser.add_argument(
        "folder",
        help="Public Google Drive folder URL or folder ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("videos_from_gdrive"),
        help="Download directory (default: ./videos_from_gdrive)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per file after transient failures (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retries < 0:
        print("Error: --retries must be zero or greater.", file=sys.stderr)
        return 2

    folder_url = args.folder.strip()
    if not folder_url:
        print("Error: provide a public Google Drive folder URL or ID.", file=sys.stderr)
        return 2
    if "://" not in folder_url:
        folder_url = f"https://drive.google.com/drive/folders/{folder_url}"

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("gdown") is None:
        print("Error: gdown is not installed. Install with: pip install 'gdown>=6.0'", file=sys.stderr)
        return 2
    command = [
        sys.executable,
        "-m",
        "gdown",
        folder_url,
        "--output",
        str(output),
        "--continue",
        "--retries",
        str(args.retries),
        "--no-cookies",
    ]

    try:
        result = subprocess.run(command, check=False)
    except FileNotFoundError:
        print(
            "Error: gdown is not installed. Install with: pip install 'gdown>=6.0'",
            file=sys.stderr,
        )
        return 2

    if result.returncode == 0:
        print(f"Download complete: {output}")
    else:
        print(
            "Download failed. Confirm the folder is shared as 'Anyone with the link' "
            "and rerun to resume partial files.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
