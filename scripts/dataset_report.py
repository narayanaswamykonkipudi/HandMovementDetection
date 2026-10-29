#!/usr/bin/env python
"""Summarise raw collection and processed split health without inventing data.

Usage
-----
    python scripts/dataset_report.py --raw data/raw/landmarks.csv \
        --processed data/processed

The report calls out missing letters, class imbalance, session overlap, invalid
quality scores, duplicate landmark rows, and whether the dataset is synthetic.
It is a diagnostic, not a recognition-accuracy report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ARTIFACTS_DIR,
    COLLECTION_LABELS,
    PROCESSED_DIR,
    RAW_COORD_COLUMNS,
    RAW_DATA_CSV,
    STATIC_ASL_LETTERS,
    UNKNOWN_LABEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_CSV)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--out", type=Path, default=ARTIFACTS_DIR / "dataset_report" / "report.md"
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _raw_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "message": "Raw dataset not found."}
    frame = pd.read_csv(path, dtype={"label": str, "subject_id": str, "session_id": str})
    counts = frame["label"].value_counts().sort_index()
    missing_known = sorted(set(STATIC_ASL_LETTERS) - set(counts.index))
    missing_collection = sorted(set(COLLECTION_LABELS) - set(counts.index))
    invalid_scores = int(
        ((frame["hand_score"] < 0) | (frame["hand_score"] > 1)).sum()
    ) if "hand_score" in frame else 0
    landmark_values = frame[RAW_COORD_COLUMNS].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(landmark_values)).sum())
    duplicates = int(
        frame[RAW_COORD_COLUMNS].round(6).duplicated().sum()
    ) if all(column in frame for column in RAW_COORD_COLUMNS) else 0
    sample_duplicates = (
        int(frame["sample_id"].duplicated().sum()) if "sample_id" in frame else 0
    )
    balance = None
    known_counts = [int(counts.get(label, 0)) for label in STATIC_ASL_LETTERS]
    if known_counts and max(known_counts):
        balance = {
            "min_letter_count": int(min(known_counts)),
            "max_letter_count": int(max(known_counts)),
            "max_to_min_ratio": (
                float(max(known_counts) / min(known_counts))
                if min(known_counts) > 0
                else None
            ),
        }
    return {
        "available": True,
        "path": str(path),
        "n_rows": int(len(frame)),
        "labels": {str(key): int(value) for key, value in counts.items()},
        "missing_static_letters": missing_known,
        "missing_collection_labels": missing_collection,
        "n_subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else None,
        "n_sessions": int(frame["session_id"].nunique()) if "session_id" in frame else None,
        "n_groups": (
            int(frame[["subject_id", "session_id"]].drop_duplicates().shape[0])
            if {"subject_id", "session_id"}.issubset(frame.columns)
            else None
        ),
        "sources": (
            {str(key): int(value) for key, value in frame["source"].value_counts().items()}
            if "source" in frame
            else {}
        ),
        "hand_score": (
            {
                "min": float(frame["hand_score"].min()),
                "p05": float(frame["hand_score"].quantile(0.05)),
                "median": float(frame["hand_score"].median()),
                "p95": float(frame["hand_score"].quantile(0.95)),
                "max": float(frame["hand_score"].max()),
                "invalid": invalid_scores,
            }
            if "hand_score" in frame and len(frame)
            else None
        ),
        "non_finite_landmark_values": non_finite,
        "duplicate_landmark_rows": duplicates,
        "duplicate_sample_ids": sample_duplicates,
        "class_balance": balance,
    }


def _processed_report(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {
            "available": False,
            "path": str(path),
            "message": "Processed manifest not found. Run preprocess_data.py first.",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: dict[str, Any] = {}
    label_sets: dict[str, set[str]] = {}
    group_sets: dict[str, set[str]] = {}
    subject_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        csv_path = path / f"{name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, dtype={"label": str, "subject_id": str, "session_id": str})
        labels = frame["label"].astype(str)
        label_sets[name] = set(labels)
        subjects = set(frame["subject_id"].astype(str))
        subject_sets[name] = subjects
        groups = set(
            frame["subject_id"].astype(str) + "::" + frame["session_id"].astype(str)
        )
        group_sets[name] = groups
        splits[name] = {
            "n_rows": int(len(frame)),
            "n_known": int((labels != UNKNOWN_LABEL).sum()),
            "n_unknown": int((labels == UNKNOWN_LABEL).sum()),
            "labels": {str(key): int(value) for key, value in labels.value_counts().sort_index().items()},
            "n_subjects": int(len(subjects)),
            "n_groups": int(len(groups)),
            "subjects": sorted(subjects),
        }
    overlap = {
        "train_val_groups": len(group_sets.get("train", set()) & group_sets.get("val", set())),
        "train_test_groups": len(group_sets.get("train", set()) & group_sets.get("test", set())),
        "val_test_groups": len(group_sets.get("val", set()) & group_sets.get("test", set())),
        "train_val_subjects": len(subject_sets.get("train", set()) & subject_sets.get("val", set())),
        "train_test_subjects": len(subject_sets.get("train", set()) & subject_sets.get("test", set())),
    }
    return {
        "available": True,
        "path": str(path),
        "dataset_kind": manifest.get("dataset", {}).get("dataset_kind", "unknown"),
        "created_utc": manifest.get("created_utc"),
        "source_sha256": manifest.get("source_sha256"),
        "split": manifest.get("split", {}),
        "splits": splits,
        "overlap": overlap,
        "test_label_coverage": sorted(label_sets.get("test", set())),
        "warnings": [
            message
            for message in [
                "Group overlap between splits is a leakage risk."
                if any(value > 0 for key, value in overlap.items() if "groups" in key)
                else None,
                "Synthetic dataset: metrics are plumbing checks, not recognition accuracy."
                if manifest.get("dataset", {}).get("dataset_kind") == "synthetic"
                else None,
            ]
            if message
        ],
    }


def main() -> int:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "raw": _raw_report(args.raw),
        "processed": _processed_report(args.processed),
        "scope": {
            "letters": list(STATIC_ASL_LETTERS),
            "excluded_dynamic": ["J", "Z"],
            "unknown_label": UNKNOWN_LABEL,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.json or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")

    raw = payload["raw"]
    processed = payload["processed"]
    lines = ["# Dataset health report", ""]
    lines.append("This report describes dataset contents and leakage checks. It is not an accuracy report.")
    lines.append("")
    lines.append("## Raw collection")
    lines.append("")
    if not raw.get("available"):
        lines.append(f"- {raw.get('message')} (`{raw.get('path')}`)")
    else:
        lines.extend(
            [
                f"- Rows: **{raw['n_rows']}**",
                f"- Subjects: {raw['n_subjects']} | sessions: {raw['n_sessions']} | subject-sessions: {raw['n_groups']}",
                f"- Hand score: median {raw['hand_score']['median']:.3f} "
                f"(p05 {raw['hand_score']['p05']:.3f}, p95 {raw['hand_score']['p95']:.3f})",
                f"- Invalid hand scores: {raw['hand_score']['invalid']} | "
                f"non-finite landmark values: {raw['non_finite_landmark_values']}",
                f"- Duplicate landmark rows: {raw['duplicate_landmark_rows']} | "
                f"duplicate sample ids: {raw['duplicate_sample_ids']}",
                f"- Missing static letters: {', '.join(raw['missing_static_letters']) or 'none'}",
                f"- Missing collection labels (including `_unknown`): "
                f"{', '.join(raw['missing_collection_labels']) or 'none'}",
            ]
        )
        if raw.get("class_balance"):
            balance = raw["class_balance"]
            lines.append(
                f"- Letter balance: min {balance['min_letter_count']}, "
                f"max {balance['max_letter_count']}, "
                f"max/min {balance['max_to_min_ratio'] if balance['max_to_min_ratio'] is not None else 'n/a'}"
            )
        lines.extend(["", "| label | samples |", "|---|---:|"])
        for label, count in raw["labels"].items():
            lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Processed splits", ""])
    if not processed.get("available"):
        lines.append(f"- {processed.get('message')} (`{processed.get('path')}`)")
    else:
        lines.append(f"- Dataset kind: `{processed['dataset_kind']}`")
        lines.extend(["", "| split | rows | known | unknown | subjects | groups |", "|---|---:|---:|---:|---:|---:|"])
        for name, split in processed["splits"].items():
            lines.append(
                f"| {name} | {split['n_rows']} | {split['n_known']} | "
                f"{split['n_unknown']} | {split['n_subjects']} | {split['n_groups']} |"
            )
        lines.extend(["", "### Leakage checks", "", "| overlap | count |", "|---|---:|"])
        for key, value in processed["overlap"].items():
            lines.append(f"| {key} | {value} |")
        if processed["warnings"]:
            lines.extend(["", "### Warnings", ""])
            for warning in processed["warnings"]:
                lines.append(f"- **{warning}**")
    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "- Fill missing letters with multiple sessions and multiple subjects.",
            "- Keep subject-session groups disjoint across splits.",
            "- Re-run `python preprocess_data.py` after changing raw data.",
            "- Never cite synthetic-data metrics as sign-recognition accuracy.",
            "",
            f"Machine-readable report: `{json_path}`",
            "",
        ]
    )
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Dataset report: {args.out}")
    print(f"JSON:           {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
