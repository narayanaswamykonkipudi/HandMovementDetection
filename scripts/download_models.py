#!/usr/bin/env python
"""Download the MediaPipe HandLandmarker model and verify its SHA-256.

Usage
-----
    python scripts/download_models.py
    python scripts/download_models.py --force

The model is not bundled with this repository so that its licence and origin
remain explicit.  MediaPipe is distributed under Apache-2.0; see
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import HAND_MODEL_PATH, MODELS_DIR

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MIN_BYTES = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = HAND_MODEL_PATH

    if target.exists() and not args.force:
        if target.stat().st_size >= MIN_BYTES:
            digest = sha256(target)
            if digest == EXPECTED_SHA256:
                print(f"Model already present and verified: {target}")
                return 0
            print(
                f"Existing model has an unexpected checksum ({digest}). "
                "Use --force to replace it."
            )
            return 2
        print(f"Existing model is too small ({target.stat().st_size} bytes); replacing.")

    temporary = target.with_suffix(".download")
    print(f"Downloading MediaPipe HandLandmarker (~7.5 MB)…\n  {MODEL_URL}")
    request = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "asl-fingerspelling-project/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        print(
            f"ERROR: download failed: {exc}\n"
            "Check your internet connection or download the file manually from the URL "
            f"above and save it as {target}.",
            file=sys.stderr,
        )
        return 2

    size = temporary.stat().st_size
    digest = sha256(temporary)
    if size < MIN_BYTES:
        temporary.unlink(missing_ok=True)
        print(f"ERROR: downloaded file is unexpectedly small ({size} bytes).", file=sys.stderr)
        return 2
    if digest != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        print(
            "ERROR: checksum mismatch. The remote file may have changed. "
            f"Expected {EXPECTED_SHA256}, got {digest}. Refusing to install it.",
            file=sys.stderr,
        )
        return 2

    os.replace(temporary, target)
    print(f"\nSaved and verified: {target}\n  size: {size} bytes\n  sha256: {digest}")
    print("\nMediaPipe model licence: see THIRD_PARTY_NOTICES.md (Apache-2.0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
