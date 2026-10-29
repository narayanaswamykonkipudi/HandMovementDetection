#!/usr/bin/env python
"""Merge imported Kaggle landmarks with your own webcam landmarks.

The importer stores landmarks only; this script combines them with samples
collected by ``collect_data.py`` (normally ``data/raw/landmarks.csv``) into a
single CSV suitable for ``preprocess_data.py``.

Usage
-----
    python scripts/merge_datasets.py

    python scripts/merge_datasets.py \
        --imported data/imported/landmarks.csv \
        --raw data/raw/landmarks.csv \
        --out data/merged/landmarks.csv

Behaviour
---------
* Both inputs must use the project's 71-column schema.
* Sample ids are checked for duplicates; the merge fails loudly rather than
  silently dropping a row.
* Missing webcam data is allowed: the imported dataset is copied through, with
  a warning that no personal samples were included.
* The output is rewritten atomically each run, so this is safe to call from
  ``run_real.ps1`` after adding new webcam recordings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import CSV_COLUMNS, STATIC_ASL_LETTERS, UNKNOWN_LABEL
from src.dataset import sha256_file, utc_now_iso, validate_dataframe


class MergeError(RuntimeError):
    """Raised when two landmark files cannot be merged safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--imported",
        type=Path,
        default=PROJECT_ROOT / "data" / "imported" / "landmarks.csv",
        help="Landmarks imported from the public image dataset.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "landmarks.csv",
        help="Your own webcam landmarks from collect_data.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "merged" / "landmarks.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON merge report path (default: beside the output).",
    )
    return parser.parse_args()


def _read(path: Path, *, required: bool) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise MergeError(f"Required landmark file is missing or empty: {path}")
        print(f"  ! skipped missing optional file: {path}")
        return None
    frame = pd.read_csv(
        path,
        dtype={"label": str, "subject_id": str, "session_id": str, "source": str},
    )
    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise MergeError(f"{path} is missing required columns: {missing}")
    extra = [column for column in frame.columns if column not in CSV_COLUMNS]
    if extra:
        print(f"  note: ignoring extra columns in {path}: {extra}")
    return frame[list(CSV_COLUMNS)].copy()


def _summarise(frame: pd.DataFrame) -> dict[str, Any]:
    known = frame[frame["label"].astype(str) != UNKNOWN_LABEL]
    return {
        "n_samples": int(len(frame)),
        "sources": {
            str(key): int(value)
            for key, value in frame["source"].astype(str).value_counts().sort_index().items()
        },
        "subjects": {
            str(key): int(value)
            for key, value in frame["subject_id"].astype(str).value_counts().sort_index().items()
        },
        "groups": int(
            (frame["subject_id"].astype(str) + "::" + frame["session_id"].astype(str)).nunique()
        ),
        "class_counts": {
            str(key): int(value)
            for key, value in frame["label"].astype(str).value_counts().sort_index().items()
        },
        "missing_static_letters": sorted(
            set(STATIC_ASL_LETTERS) - set(known["label"].astype(str))
        ),
    }


def main() -> int:
    args = parse_args()
    print("Merging landmark datasets")
    print(f"  imported: {args.imported}")
    print(f"  webcam:   {args.raw}")
    print(f"  output:   {args.out}\n")

    try:
        imported = _read(args.imported, required=True)
        assert imported is not None
        raw = _read(args.raw, required=False)
        frames = [imported] if raw is None else [imported, raw]
        merged = pd.concat(frames, ignore_index=True)
    except (MergeError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    duplicates = merged["sample_id"].astype(str).duplicated(keep=False)
    if duplicates.any():
        examples = merged.loc[duplicates, "sample_id"].astype(str).head(8).tolist()
        print(
            "ERROR: duplicate sample_id values across inputs. Refusing to merge "
            f"because this could duplicate samples across splits: {examples}",
            file=sys.stderr,
        )
        return 3

    report = validate_dataframe(merged)
    if report.errors:
        print("ERROR: merged dataset failed validation:", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
        return 3

    if raw is None:
        print(
            "  ! No webcam samples found. The model will be trained from imported "
            "images only. Collect your own samples with collect_data.py to improve "
            "webcam robustness."
        )
    else:
        print(f"  + added {len(raw)} webcam sample(s)")

    summary = _summarise(merged)
    if summary["missing_static_letters"]:
        print(
            "ERROR: merged dataset is missing static letters: "
            f"{summary['missing_static_letters']}",
            file=sys.stderr,
        )
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(args.out)

    report_path = args.report or args.out.with_name("merge_report.json")
    report_payload = {
        "created_utc": utc_now_iso(),
        "output": str(args.out.resolve()),
        "output_sha256": sha256_file(args.out),
        "inputs": {
            "imported": {
                "path": str(args.imported.resolve()),
                "sha256": sha256_file(args.imported),
                "n_samples": int(len(imported)),
            },
            **(
                {
                    "webcam": {
                        "path": str(args.raw.resolve()),
                        "sha256": sha256_file(args.raw),
                        "n_samples": int(len(raw)),
                    }
                }
                if raw is not None
                else {}
            ),
        },
        "merged": summary,
        "warnings": [
            "Groups are subject_id::session_id. Imported pseudo-sessions are only "
            "a proxy for recording sessions and are not subject-disjoint.",
            "Do not claim unseen-signer accuracy unless a separate signer was held out.",
        ],
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("\nMerged dataset summary")
    print(f"  samples:  {summary['n_samples']}")
    print(f"  subjects: {summary['subjects']}")
    print(f"  groups:   {summary['groups']}")
    print(f"  sources:  {summary['sources']}")
    print("  class counts:")
    for label, count in summary["class_counts"].items():
        print(f"    {label:>9}: {count}")
    print(f"\nWrote:  {args.out}")
    print(f"Report: {report_path}")
    print(
        "\nNext:\n"
        f"  python preprocess_data.py --input {args.out} --out data/processed_kaggle\n"
        f"  python train_model.py --processed data/processed_kaggle --model svm "
        "--out models/classifier_svm.joblib --no-tune"
    )
    print(
        "\nOr run the whole merge -> preprocess -> train -> evaluate pipeline:\n"
        "  powershell -ExecutionPolicy Bypass -File .\\run_real.ps1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
