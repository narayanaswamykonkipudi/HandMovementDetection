"""Lightweight, dependency-free experiment logging.

The project does not require MLflow. Each run appends one JSON object to an
``.jsonl`` file, which is enough to compare parameters, metrics, the dataset
manifest hash, and the git revision later. MLflow can be layered on top without
changing the training code.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.dataset import sha256_file, utc_now_iso


def git_commit_hash(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return the git revision, or an explicit unavailable marker.

    The workspace used during development may not be a git checkout. Never
    substitute a fabricated hash.
    """

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return {"commit": None, "available": False, "reason": "git executable unavailable"}
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return {
            "commit": None,
            "available": False,
            "reason": "not a git repository or no commits",
        }
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    ).stdout.strip()
    return {"commit": commit, "available": True, "dirty": bool(dirty)}


def dataset_manifest_hash(processed_dir: Path | str) -> dict[str, Any]:
    manifest = Path(processed_dir) / "manifest.json"
    if not manifest.exists():
        return {"path": str(manifest), "sha256": None, "available": False}
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "available": True,
    }


def append_experiment(
    path: Path | str,
    *,
    kind: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    dataset_manifest: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Append one JSONL experiment record and return it."""

    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex[:12],
        "created_utc": utc_now_iso(),
        "kind": kind,
        "params": params,
        "metrics": metrics,
        "dataset_manifest": dataset_manifest or {},
        "git": git_commit_hash(project_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }
    if extra:
        record["extra"] = extra
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def parse_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
