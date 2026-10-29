#!/usr/bin/env python
"""Validate raw landmark CSV data and create leakage-free processed splits.

Usage
-----
    python preprocess_data.py --input data/raw/landmarks.csv --out data/processed

The test split is created here and must not be used for model selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, RAW_DATA_CSV, STATIC_ASL_LETTERS
from src.dataset import (
    DatasetError,
    group_ids,
    load_dataset,
    split_dataset,
    write_processed_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_CSV, help="Raw landmarks CSV.")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--test-subjects",
        nargs="+",
        default=None,
        metavar="SUBJECT",
        help="Hold out entire subjects for the test split (recommended for generalisation).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading {args.input} …")
    try:
        frame, report = load_dataset(args.input)
    except (FileNotFoundError, DatasetError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if not args.input.exists():
            print(
                "\nCollect data first:\n"
                "  python collect_data.py --label A --subject s01 --session s01a --count 150\n"
                "Or build the synthetic plumbing demo:\n"
                "  python scripts/build_demo.py",
                file=sys.stderr,
            )
        return 2

    print(f"\nDataset: {report.n_samples} samples | kind={report.dataset_kind}")
    print(f"Subjects: {', '.join(report.subjects)}")
    print(f"Sessions: {len(report.sessions)} | labels: {len(report.labels)}")
    print("Class counts:")
    for label, count in report.class_counts.items():
        print(f"  {label:>9}: {count}")
    print(f"Sources: {report.sources}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  ! {warning}")

    try:
        splits, info = split_dataset(
            frame, seed=args.seed, test_subjects=args.test_subjects
        )
        manifest_path = write_processed_splits(
            splits, info, args.input, report, args.out
        )
    except DatasetError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 3

    print(f"\nSplit method: {info['method']}")
    print(f"{'split':>6} {'samples':>9} {'groups':>7} {'subjects':>9}")
    for name in ("train", "val", "test"):
        part = splits[name]
        print(
            f"{name:>6} {len(part):>9} {group_ids(part).nunique():>7} "
            f"{part['subject_id'].nunique():>9}"
        )
    print(f"\nSplit manifest: {manifest_path}")
    print(
        "Verified: no recording session is shared between train, validation, and test."
    )
    if info.get("test_contains_only_unseen_subjects"):
        print("Verified: the test split contains only held-out subjects.")
    else:
        print(
            "Note: the test split is grouped by session, not guaranteed to be "
            "unseen-person evaluation. Use --test-subjects for that claim."
        )
    if report.dataset_kind in {"synthetic", "mixed"}:
        print(
            "\nWARNING: this dataset contains synthetic samples. Any metrics produced "
            "from it are pipeline checks, not real sign-recognition results."
        )
    if info["warnings"]:
        print("\nSplit warnings:")
        for warning in info["warnings"]:
            print(f"  ! {warning}")
    if report.dataset_kind == "synthetic":
        print(
            "\nNext (synthetic demo only): python train_model.py --processed "
            f"{args.out} --model svm --out models/demo_classifier.joblib"
        )
    else:
        print(
            "\nNext: python train_model.py --processed "
            f"{args.out} --model svm --out models/classifier.joblib"
        )
    print(
        "\nScope: American Sign Language static fingerspelling for "
        f"{len(STATIC_ASL_LETTERS)} letters; J and Z are excluded because they are dynamic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
