"""CSV sample construction and append-only storage for data collection."""

from __future__ import annotations

import csv
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.config import (
    COLLECTION_LABELS,
    CSV_COLUMNS,
    HANDEDNESS_VALUES,
    RAW_COORD_COLUMNS,
    SOURCE_VALUES,
)


class CollectionError(ValueError):
    """Raised when a sample or CSV file is invalid."""


def _clean_identifier(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise CollectionError(f"{field_name} cannot be empty.")
    if "::" in value:
        raise CollectionError(f"{field_name} must not contain '::'.")
    if any(character in value for character in "/\\\n\r\t,"):
        raise CollectionError(f"{field_name} contains an unsupported character.")
    return value


def make_sample_row(
    landmarks: np.ndarray,
    *,
    label: str,
    subject_id: str,
    session_id: str,
    source: str,
    handedness: str,
    hand_score: float,
    timestamp: str | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Build one CSV row from 21x3 landmarks, validating every field."""

    label = str(label).strip()
    if label not in COLLECTION_LABELS:
        raise CollectionError(
            f"Label {label!r} is not supported. Use a static ASL letter "
            f"({''.join(sorted(set(COLLECTION_LABELS) - {'_unknown'}))}) "
            "or '_unknown'."
        )
    subject_id = _clean_identifier(subject_id, "subject_id")
    session_id = _clean_identifier(session_id, "session_id")
    if source not in SOURCE_VALUES:
        raise CollectionError(f"source must be one of {SOURCE_VALUES}.")
    if handedness not in HANDEDNESS_VALUES:
        raise CollectionError(f"handedness must be one of {HANDEDNESS_VALUES}.")
    hand_score = float(hand_score)
    if not 0.0 <= hand_score <= 1.0:
        raise CollectionError("hand_score must be between 0 and 1.")

    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise CollectionError(f"Expected landmarks shape (21, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise CollectionError("Landmarks contain NaN or infinite values.")

    row: dict[str, Any] = {
        "sample_id": sample_id or uuid.uuid4().hex,
        "label": label,
        "subject_id": subject_id,
        "session_id": session_id,
        "source": source,
        "handedness": handedness,
        "hand_score": round(hand_score, 6),
        "timestamp": timestamp
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    flat = points.reshape(-1)
    for column, value in zip(RAW_COORD_COLUMNS, flat, strict=True):
        row[column] = float(value)
    return row


def append_samples(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    """Append validated rows, writing the header only for a new file.

    The function checks an existing file's header and refuses to mix schemas,
    which prevents silent data corruption after a project upgrade.
    """

    path = Path(path)
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        if tuple(header) != CSV_COLUMNS:
            raise CollectionError(
                f"{path} has an incompatible header. Expected {len(CSV_COLUMNS)} "
                f"columns; found {len(header)}. Move the old file aside or start a "
                "new dataset path."
            )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            missing = [column for column in CSV_COLUMNS if column not in row]
            if missing:
                raise CollectionError(f"Row is missing columns: {missing}")
            writer.writerow(row)
        handle.flush()
    return len(rows)


def count_samples(path: Path | str) -> Counter[str]:
    path = Path(path)
    if not path.exists():
        return Counter()
    counts: Counter[str] = Counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[str(row.get("label", ""))] += 1
    return counts


def collection_progress(path: Path | str) -> dict[str, dict[str, int | set[str]]]:
    """Return counts, sessions, and subjects for every label in the CSV.

    The return value is JSON-friendly except for the internal ``set`` values,
    which callers should convert to lengths or sorted lists themselves.
    """

    path = Path(path)
    progress: dict[str, dict[str, int | set[str]]] = {}
    if not path.exists():
        return progress
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label", ""))
            entry = progress.setdefault(
                label, {"samples": 0, "sessions": set(), "subjects": set()}
            )
            entry["samples"] = int(entry["samples"]) + 1
            sessions = entry["sessions"]
            subjects = entry["subjects"]
            if isinstance(sessions, set):
                sessions.add(str(row.get("session_id", "")))
            if isinstance(subjects, set):
                subjects.add(str(row.get("subject_id", "")))
    return progress
