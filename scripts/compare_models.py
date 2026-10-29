#!/usr/bin/env python
"""Compare SVM, RandomForest, and a small MLP on identical grouped splits.

Usage
-----
    python scripts/compare_models.py --processed data/demo/processed \
        --output artifacts/model_comparison

Models are fitted on the augmented training split, thresholds are calibrated on
validation only, and validation metrics choose the preferred candidate.  Test
metrics for every candidate are shown for transparency but must not be used to
re-select a model and then reported as an unbiased final estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, PROCESSED_DIR
from src.dataset import DatasetError, load_processed_splits
from src.evaluation import (
    plot_confusion_matrix,
    plot_reliability,
    synthetic_warning,
    write_json,
)
from src.experiment_log import append_experiment, dataset_manifest_hash
from src.experiments import (
    Candidate,
    candidate_summary,
    evaluate_on_frame,
    fit_candidate,
    prepare_training_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--models", default="svm,rf,mlp", help="Comma-separated: svm,rf,mlp"
    )
    parser.add_argument("--feature-set", choices=("base", "geometry"), default="base")
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Run grouped grid search.")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "model_comparison")
    parser.add_argument(
        "--experiment-log", type=Path, default=ARTIFACTS_DIR / "experiments.jsonl"
    )
    parser.add_argument("--no-experiment-log", action="store_true")
    return parser.parse_args()


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ci(item: dict[str, Any], digits: int = 4) -> str:
    if item.get("point") is None:
        return "n/a"
    if item.get("lower") is None:
        return _fmt(item.get("point"), digits)
    return f"{_fmt(item.get('point'), digits)} ({_fmt(item.get('lower'), digits)}–{_fmt(item.get('upper'), digits)})"


def _row(name: str, result: dict[str, Any]) -> str:
    ci = result["bootstrap_ci_95"].get("macro_f1", {})
    return (
        f"| {name} | {_ci(ci)} | {_fmt(result['forced_accuracy'])} | "
        f"{_fmt(result['selective_accuracy'])} / {_fmt(result['coverage'])} | "
        f"{_fmt(result['ece'])} |"
    )


def _table(candidates: list[dict[str, Any]], key: str) -> str:
    lines = [
        "| candidate | macro-F1 (95% CI) | forced acc. | selective acc. / coverage | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        result = candidate.get(key)
        if result:
            lines.append(_row(candidate["model"], result))
    return "\n".join(lines)


def _write_plots(
    candidate: Candidate,
    result: dict[str, Any],
    output_dir: Path,
    *,
    split: str,
    warning: str | None,
) -> list[str]:
    stem = f"{candidate.model_name}_{split}"
    confusion_path = output_dir / f"confusion_{stem}.png"
    plot_confusion_matrix(
        np.asarray(result["confusion_matrix"], dtype=int),
        candidate.labels,
        confusion_path,
        title=f"{candidate.model_name.upper()} — forced {split}",
        note=warning or f"{candidate.model_name} / {candidate.feature_set} features.",
    )
    reliability_path = output_dir / f"reliability_{stem}.png"
    plot_reliability(
        {"ece": result["ece"], "brier": result["brier"], "bins": result["bins"]},
        reliability_path,
        title=f"{candidate.model_name.upper()} reliability — {split}",
        note=warning or f"{candidate.model_name} / {candidate.feature_set} features.",
    )
    return [confusion_path.name, reliability_path.name]


def main() -> int:
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names or any(name not in {"svm", "rf", "mlp"} for name in model_names):
        print("ERROR: --models must contain svm, rf, and/or mlp.", file=sys.stderr)
        return 2

    try:
        splits, manifest = load_processed_splits(args.processed, names=("train", "val", "test"))
    except FileNotFoundError:
        try:
            splits, manifest = load_processed_splits(args.processed, names=("train", "val"))
            print("WARNING: no test split found; reporting validation only.")
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

    prepared = prepare_training_data(
        splits,
        feature_set=args.feature_set,
        augment_copies=args.augment_copies,
        seed=args.seed,
    )
    print(
        f"Training {len(model_names)} candidates on {len(prepared['X_fit'])} fitted "
        f"samples ({len(prepared['X_train'])} base + {prepared['n_augmented']} augmented)."
    )

    candidates: list[dict[str, Any]] = []
    plot_files: dict[str, list[str]] = {}
    objects: list[Candidate] = []
    results: list[tuple[Candidate, dict[str, Any], dict[str, Any] | None]] = []
    for model_name in model_names:
        print(f"  fitting {model_name.upper()} ...")
        candidate = fit_candidate(
            model_name,
            prepared,
            feature_set=args.feature_set,
            seed=args.seed,
            tune=args.tune,
            cv=args.cv,
        )
        validation = evaluate_on_frame(
            candidate,
            splits["val"],
            split_name="val",
            n_boot=args.bootstrap,
            seed=args.seed,
        )
        test = (
            evaluate_on_frame(
                candidate,
                splits["test"],
                split_name="test",
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            if "test" in splits
            else None
        )
        summary = candidate_summary(candidate, validation, test)
        candidates.append(summary)
        objects.append(candidate)
        results.append((candidate, validation, test))
        print(
            f"    val macro-F1 {_fmt(validation['macro_f1'])} | "
            f"test macro-F1 {_fmt(test['macro_f1']) if test else 'n/a'}"
        )

    selected = max(
        candidates,
        key=lambda item: (
            item["validation"]["macro_f1"] or 0.0,
            item["validation"]["forced_accuracy"] or 0.0,
        ),
    )
    selected_name = selected["model"]
    for candidate, validation, test in results:
        if candidate.model_name != selected_name:
            continue
        if test is not None:
            plot_files[candidate.model_name] = _write_plots(
                candidate, test, args.output, split="test", warning=warning
            )
        else:
            plot_files[candidate.model_name] = _write_plots(
                candidate, validation, args.output, split="val", warning=warning
            )

    payload = {
        "schema_version": 1,
        "dataset_kind": dataset_kind,
        "warning": warning,
        "processed_dir": str(Path(args.processed).resolve()),
        "feature_set": args.feature_set,
        "augment_copies": args.augment_copies,
        "seed": args.seed,
        "tuned": args.tune,
        "selection_rule": "highest validation macro-F1, then forced accuracy",
        "selected_model": selected_name,
        "candidates": candidates,
        "plots": plot_files,
        "interpretation": [
            "Model selection used validation data only.",
            "Test metrics are shown for every candidate; using them to choose a model would make the test estimate optimistic.",
            "Macro-F1 treats all 24 static letters equally; forced accuracy ignores the reject option.",
            "Selective accuracy is paired with coverage and must never be quoted alone.",
            "95% intervals use a group-level percentile bootstrap over subject-session groups.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "comparison.json", payload)

    markdown = f"""# Model comparison

{f'**WARNING:** {warning}' if warning else ''}

Dataset kind: `{dataset_kind}`  
Feature set: `{args.feature_set}`  
Selection rule: **{payload['selection_rule']}**  
Selected candidate: **{selected_name.upper()}**

## Validation (model selection)

{_table(candidates, 'validation')}

## Test (descriptive; not used for selection)

{_table(candidates, 'test') if 'test' in splits else 'No test split was available.'}

## Interpretation

- Accuracy is *forced* top-1 on known letters; the deployed app may reject samples.
- Selective accuracy is only meaningful with its paired coverage.
- ECE and Brier score assess calibration, not recognition ability.
- Intervals resample whole subject-session groups to avoid treating correlated frames as independent.
- A synthetic dataset measures software plumbing only, never sign-recognition accuracy.
- Per-class and per-subject details are in `comparison.json`.
"""
    (args.output / "comparison.md").write_text(markdown, encoding="utf-8")

    if not args.no_experiment_log:
        record = append_experiment(
            args.experiment_log,
            kind="model_comparison",
            params={
                "models": model_names,
                "feature_set": args.feature_set,
                "augment_copies": args.augment_copies,
                "tuned": args.tune,
                "seed": args.seed,
            },
            metrics={
                "candidates": {
                    item["model"]: {
                        "validation": {
                            "macro_f1": item["validation"]["macro_f1"],
                            "forced_accuracy": item["validation"]["forced_accuracy"],
                            "selective_accuracy": item["validation"]["selective_accuracy"],
                            "coverage": item["validation"]["coverage"],
                            "ece": item["validation"]["ece"],
                        },
                        "test": (
                            {
                                "macro_f1": item["test"]["macro_f1"],
                                "forced_accuracy": item["test"]["forced_accuracy"],
                                "selective_accuracy": item["test"]["selective_accuracy"],
                                "coverage": item["test"]["coverage"],
                                "ece": item["test"]["ece"],
                            }
                            if item.get("test")
                            else None
                        ),
                    }
                    for item in candidates
                },
                "selected_model": selected_name,
            },
            dataset_manifest=dataset_manifest_hash(args.processed),
            extra={"output": str(args.output.resolve()), "dataset_kind": dataset_kind},
        )
        print(f"Experiment log: {args.experiment_log} ({record['run_id']})")

    print("\n" + "=" * 78)
    print(f"Comparison written to {args.output / 'comparison.md'}")
    print(f"Selected on validation: {selected_name.upper()}")
    if warning:
        print(warning)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
