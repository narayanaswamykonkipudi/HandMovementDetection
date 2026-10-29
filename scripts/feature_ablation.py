#!/usr/bin/env python
"""Ablate base landmarks against added fingertip-distance and joint-angle features.

Usage
-----
    python scripts/feature_ablation.py --processed data/demo/processed --model svm

The same grouped splits, model family, augmentation, and threshold selection are
used for both feature sets.  Differences are reported, not assumed to be
statistically meaningful on a single split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, PROCESSED_DIR
from src.dataset import DatasetError, load_processed_splits
from src.evaluation import synthetic_warning, write_json
from src.experiment_log import append_experiment, dataset_manifest_hash
from src.experiments import (
    candidate_summary,
    evaluate_on_frame,
    fit_candidate,
    prepare_training_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model", choices=("svm", "rf", "mlp"), default="svm")
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "feature_ablation")
    parser.add_argument(
        "--experiment-log", type=Path, default=ARTIFACTS_DIR / "experiments.jsonl"
    )
    parser.add_argument("--no-experiment-log", action="store_true")
    return parser.parse_args()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(item: dict[str, Any]) -> str:
    if item.get("point") is None:
        return "n/a"
    if item.get("lower") is None:
        return _fmt(item.get("point"))
    return f"{_fmt(item.get('point'))} ({_fmt(item.get('lower'))}–{_fmt(item.get('upper'))})"


def main() -> int:
    args = parse_args()
    try:
        splits, manifest = load_processed_splits(args.processed, names=("train", "val", "test"))
    except FileNotFoundError:
        try:
            splits, manifest = load_processed_splits(args.processed, names=("train", "val"))
        except (FileNotFoundError, DatasetError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    except (DatasetError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    dataset_kind = str(manifest.get("dataset", {}).get("dataset_kind", "unknown"))
    warning = synthetic_warning(dataset_kind)
    if warning:
        print("\n" + "!" * 78)
        print(warning)
        print("!" * 78 + "\n")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for feature_set in ("base", "geometry"):
        print(f"Training {args.model.upper()} with {feature_set} features ...")
        prepared = prepare_training_data(
            splits,
            feature_set=feature_set,
            augment_copies=args.augment_copies,
            seed=args.seed,
        )
        candidate = fit_candidate(
            args.model,
            prepared,
            feature_set=feature_set,
            seed=args.seed,
            tune=args.tune,
        )
        validation = evaluate_on_frame(
            candidate, splits["val"], split_name="val", n_boot=args.bootstrap, seed=args.seed
        )
        test = (
            evaluate_on_frame(
                candidate, splits["test"], split_name="test", n_boot=args.bootstrap, seed=args.seed
            )
            if "test" in splits
            else None
        )
        summaries.append(candidate_summary(candidate, validation, test))
        rows.append(
            {
                "feature_set": feature_set,
                "feature_dim": int(prepared["X_train"].shape[1]),
                "validation": validation,
                "test": test,
            }
        )
        print(
            f"  {feature_set}: val macro-F1 {_fmt(validation['macro_f1'])}, "
            f"test macro-F1 {_fmt(test['macro_f1']) if test else 'n/a'}"
        )

    delta = None
    if rows[0].get("test") and rows[1].get("test"):
        delta = {
            "test_macro_f1": rows[1]["test"]["macro_f1"] - rows[0]["test"]["macro_f1"],
            "test_forced_accuracy": (
                rows[1]["test"]["forced_accuracy"] - rows[0]["test"]["forced_accuracy"]
            ),
            "validation_macro_f1": (
                rows[1]["validation"]["macro_f1"] - rows[0]["validation"]["macro_f1"]
            ),
        }

    payload = {
        "schema_version": 1,
        "dataset_kind": dataset_kind,
        "warning": warning,
        "model": args.model,
        "augment_copies": args.augment_copies,
        "seed": args.seed,
        "tuned": args.tune,
        "comparison": rows,
        "delta_geometry_minus_base": delta,
        "interpretation": [
            "Both feature sets used the same grouped splits and threshold procedure.",
            "The difference is descriptive for this dataset; no significance test is claimed.",
            "Geometry adds 25 distances/angles to the 63 normalised coordinates.",
            "Synthetic-data results are software plumbing checks, not recognition accuracy.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "ablation.json", payload)

    def table(split: str) -> str:
        lines = [
            "| feature set | dim | macro-F1 (95% CI) | forced acc. | selective / coverage | ECE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            result = row.get(split)
            if not result:
                continue
            lines.append(
                f"| {row['feature_set']} | {row['feature_dim']} | "
                f"{_ci(result['bootstrap_ci_95'].get('macro_f1', {}))} | "
                f"{_fmt(result['forced_accuracy'])} | "
                f"{_fmt(result['selective_accuracy'])} / {_fmt(result['coverage'])} | "
                f"{_fmt(result['ece'])} |"
            )
        return "\n".join(lines)

    markdown = f"""# Feature ablation — {args.model.upper()}

{f'**WARNING:** {warning}' if warning else ''}

Base features are 63 normalised landmark coordinates. Geometry adds 25
engineered features: 5 wrist-to-fingertip distances, 10 pairwise fingertip
distances, and 10 joint angles.

## Validation

{table('validation')}

## Test

{table('test') if 'test' in splits else 'No test split was available.'}

## Interpretation

- Same grouped splits, augmentation, model family, and threshold search for both rows.
- A single-split difference is not proof that one feature set is generally better.
- Per-class and per-subject details are in `ablation.json`.
- Synthetic metrics describe plumbing only and are not sign-recognition accuracy.
"""
    (args.output / "ablation.md").write_text(markdown, encoding="utf-8")

    if not args.no_experiment_log:
        record = append_experiment(
            args.experiment_log,
            kind="feature_ablation",
            params={
                "model": args.model,
                "augment_copies": args.augment_copies,
                "seed": args.seed,
                "tuned": args.tune,
            },
            metrics={
                "feature_sets": {
                    row["feature_set"]: {
                        "feature_dim": row["feature_dim"],
                        "validation_macro_f1": row["validation"]["macro_f1"],
                        "test_macro_f1": (
                            row["test"]["macro_f1"] if row.get("test") else None
                        ),
                    }
                    for row in rows
                },
                "delta_geometry_minus_base": delta,
            },
            dataset_manifest=dataset_manifest_hash(args.processed),
            extra={"output": str(args.output.resolve()), "dataset_kind": dataset_kind},
        )
        print(f"Experiment log: {args.experiment_log} ({record['run_id']})")

    print("\n" + "=" * 78)
    print(f"Ablation written to {args.output / 'ablation.md'}")
    if warning:
        print(warning)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
