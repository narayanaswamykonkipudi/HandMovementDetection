#!/usr/bin/env python
"""Evaluate a trained model once on a processed split.

Usage
-----
    python evaluate_model.py --processed data/processed \
        --model models/classifier.joblib --split test

All numbers written by this script come from the chosen split.  For the test
split, this script should be run only after model and threshold selection are
finished.  Synthetic datasets produce a prominent warning and must not be cited
as sign-recognition accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import check_thresholds_on_split
from src.config import PROCESSED_DIR, UNKNOWN_LABEL
from src.dataset import DatasetError, group_ids, load_processed_splits
from src.evaluation import (
    classification_metrics,
    confusion_pairs,
    expected_calibration_error,
    grouped_bootstrap_ci,
    known_mask,
    per_subject_breakdown,
    plot_confusion_matrix,
    plot_reliability,
    synthetic_warning,
    write_json,
)
from src.features import build_feature_matrix, feature_table_to_array
from src.model_bundle import ModelBundle, ModelBundleError, load_model_bundle
from src.config import RAW_COORD_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Group-level bootstrap resamples for 95%% confidence intervals (0 disables).",
    )
    return parser.parse_args()


def _features(
    frame: pd.DataFrame, *, feature_dim: int, feature_set: str
) -> np.ndarray:
    if frame.empty:
        return np.empty((0, feature_dim), dtype=np.float64)
    return build_feature_matrix(
        feature_table_to_array(frame, RAW_COORD_COLUMNS), feature_set=feature_set
    )


def _format_metric(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_ci(item: dict[str, Any], digits: int = 4) -> str:
    if item.get("point") is None:
        return "n/a"
    if item.get("lower") is None:
        return _format_metric(item.get("point"), digits)
    return (
        f"{_format_metric(item.get('point'), digits)} "
        f"(95% CI {_format_metric(item.get('lower'), digits)}–"
        f"{_format_metric(item.get('upper'), digits)})"
    )


def _markdown_table(metrics: dict[str, Any], labels: list[str]) -> str:
    lines = [
        "| class | precision | recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = metrics["per_class"].get(label, {})
        lines.append(
            f"| {label} | {row.get('precision', 0):.4f} | {row.get('recall', 0):.4f} | "
            f"{row.get('f1', 0):.4f} | {row.get('support', 0)} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output or PROJECT_ROOT / "artifacts" / f"eval_{args.split}"

    try:
        bundle = load_model_bundle(args.model)
    except ModelBundleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        splits, manifest = load_processed_splits(args.processed, names=(args.split,))
    except (FileNotFoundError, DatasetError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    frame = splits[args.split]
    if frame.empty:
        print(f"ERROR: split '{args.split}' is empty.", file=sys.stderr)
        return 3

    labels = bundle.labels
    dataset_kind = str(manifest.get("dataset", {}).get("dataset_kind", "unknown"))
    warning = synthetic_warning(dataset_kind)

    X = _features(
        frame, feature_dim=bundle.feature_dim, feature_set=bundle.feature_set
    )
    probabilities = bundle.predict_proba(X)
    forced_predictions = np.asarray(labels, dtype=object)[np.argmax(probabilities, axis=1)]
    y_true = frame["label"].astype(str).to_numpy()
    is_known = known_mask(y_true)

    if not is_known.any():
        print("ERROR: split has no known-class samples.", file=sys.stderr)
        return 3

    known_metrics = classification_metrics(y_true[is_known], forced_predictions[is_known], labels)
    calibration_diagnostics = expected_calibration_error(
        y_true[is_known], probabilities[is_known], labels
    )
    rejection = check_thresholds_on_split(
        y_true, probabilities, labels, bundle.thresholds
    )

    majority_label = (
        pd.Series(y_true[is_known]).value_counts().idxmax()
        if is_known.any()
        else None
    )
    majority_baseline = (
        float((y_true[is_known] == majority_label).mean())
        if majority_label is not None
        else None
    )

    top_confusions = confusion_pairs(
        y_true[is_known], forced_predictions[is_known], limit=20
    )

    margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    accepted_mask = (
        (probabilities.max(axis=1) >= bundle.thresholds["min_confidence"])
        & (margins >= bundle.thresholds["min_margin"])
    )
    split_groups_array = group_ids(frame).to_numpy()
    if args.bootstrap > 0:
        bootstrap_ci = {
            "accuracy": grouped_bootstrap_ci(
                y_true[is_known],
                forced_predictions[is_known],
                split_groups_array[is_known],
                labels,
                metric="accuracy",
                n_boot=args.bootstrap,
            ),
            "macro_f1": grouped_bootstrap_ci(
                y_true[is_known],
                forced_predictions[is_known],
                split_groups_array[is_known],
                labels,
                metric="macro_f1",
                n_boot=args.bootstrap,
            ),
        }
    else:
        bootstrap_ci = {"note": "Disabled with --bootstrap 0."}
    subject_breakdown = per_subject_breakdown(
        y_true,
        forced_predictions,
        frame["subject_id"].astype(str).to_numpy(),
        accepted_mask,
        labels,
    )

    confusion = np.asarray(known_metrics["confusion_matrix"], dtype=int)
    confusion_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        confusion,
        labels,
        confusion_path,
        title=f"Forced predictions — {args.split} split",
        note=warning or "Confusion matrix from the selected processed split.",
    )
    reliability_path = plot_reliability(
        calibration_diagnostics,
        output_dir / "reliability.png",
        title=f"Reliability — forced predictions ({args.split})",
        note=warning or "Grouped bins; ECE is not an accuracy score.",
    )

    # Per-class error analysis: most frequent wrong destination for each class.
    per_class_errors: dict[str, list[dict[str, Any]]] = {}
    errors = pd.DataFrame(
        {
            "true": y_true[is_known],
            "predicted": forced_predictions[is_known],
        }
    )
    errors = errors[errors["true"] != errors["predicted"]]
    if not errors.empty:
        for true_label, group in errors.groupby("true", observed=True):
            counts = group["predicted"].value_counts().head(5)
            per_class_errors[str(true_label)] = [
                {"predicted": str(predicted), "count": int(count)}
                for predicted, count in counts.items()
            ]

    split_groups = sorted(group_ids(frame).unique().tolist())
    split_subjects = sorted(frame["subject_id"].astype(str).unique().tolist())
    split_info = manifest.get("split", {})
    train_subjects = set(split_info.get("splits", {}).get("train", {}).get("subjects", []))
    unseen_subjects = sorted(set(split_subjects) - train_subjects)
    subject_disjoint = bool(train_subjects) and not (set(split_subjects) & train_subjects)

    predictions_frame = frame[
        ["sample_id", "label", "subject_id", "session_id", "source", "handedness", "hand_score"]
    ].copy()
    predictions_frame["predicted"] = forced_predictions
    predictions_frame["confidence"] = probabilities.max(axis=1)
    predictions_frame["margin"] = margins
    predictions_frame["accepted_by_threshold"] = accepted_mask

    metrics_payload = {
        "schema_version": 1,
        "split": args.split,
        "dataset_kind": dataset_kind,
        "warning": warning,
        "model_file": str(args.model.resolve()),
        "model_labels": labels,
        "model_thresholds": bundle.thresholds,
        "model_metadata": bundle.public_metadata(),
        "feature_set": bundle.feature_set,
        "feature_dim": bundle.feature_dim,
        "n_samples": int(len(frame)),
        "n_known": int(is_known.sum()),
        "n_unknown": int((~is_known).sum()),
        "groups": split_groups,
        "subjects": split_subjects,
        "unseen_subjects_in_split": unseen_subjects,
        "subject_disjoint_test": subject_disjoint,
        "forced_classification": known_metrics,
        "bootstrap_ci_95": bootstrap_ci,
        "per_subject": subject_breakdown,
        "majority_baseline_accuracy": majority_baseline,
        "majority_label": majority_label,
        "calibration_diagnostics": calibration_diagnostics,
        "rejection": rejection,
        "top_confusions": top_confusions,
        "per_class_errors": per_class_errors,
        "notes": [
            "Forced accuracy ignores the reject option and is not the deployed system's accuracy.",
            "Selective accuracy is measured only on accepted samples; coverage is reported with it.",
            "Unknown-sign acceptance is measured only if '_unknown' samples exist in this split.",
            "ECE/Brier assess probability calibration; they are not accuracy scores.",
        ],
    }
    if args.split == "val":
        metrics_payload["warning"] = (
            "Validation split: thresholds/metrics may have been used for model selection. "
            "This is not an unbiased final test estimate."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics_payload)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    report_text = classification_report(
        y_true[is_known],
        forced_predictions[is_known],
        labels=labels,
        zero_division=0,
        digits=4,
    )
    rejection_summary = (
        f"Thresholds: confidence >= {bundle.thresholds['min_confidence']:.2f}, "
        f"margin >= {bundle.thresholds['min_margin']:.2f} "
        f"(source={bundle.thresholds.get('source')}, status={bundle.thresholds.get('status')})\n"
        f"Known samples accepted: {rejection['known'].get('n_known', 0) - (rejection['known'].get('rejected_known') or 0)}"
        f"/{rejection['known'].get('n_known', 0)} "
        f"(coverage={_format_metric(rejection['known'].get('coverage'))})\n"
        f"Selective accuracy (accepted known only): "
        f"{_format_metric(rejection['known'].get('selective_accuracy'))}\n"
        f"Forced accuracy (all known, ignoring rejection): "
        f"{_format_metric(rejection['known'].get('forced_accuracy'))}\n"
        f"Unknown acceptance rate: {_format_metric(rejection['unknown'].get('accept_rate'))} "
        f"({rejection['unknown'].get('n_unknown', 0)} unknown samples)\n"
    )
    (output_dir / "classification_report.txt").write_text(
        report_text + "\n" + rejection_summary, encoding="utf-8"
    )

    confusion_lines = "\n".join(
        f"- {item['true']} → {item['predicted']}: {item['count']}"
        for item in top_confusions
    ) or "- None (no misclassifications in this split)."
    subject_lines = "\n".join(
        f"| {subject} | {row['n_known']} | {_format_metric(row['forced_accuracy'])} | "
        f"{_format_metric(row['forced_macro_f1'])} | {_format_metric(row['coverage'])} | "
        f"{_format_metric(row['selective_accuracy'])} |"
        for subject, row in subject_breakdown.items()
    )
    if not subject_lines:
        subject_lines = "| n/a | 0 | n/a | n/a | n/a | n/a |"
    warning_block = f"\n> **WARNING:** {metrics_payload['warning']}\n" if metrics_payload.get("warning") else ""
    report_md = f"""# Evaluation report — {args.split} split

{warning_block}
**Dataset kind:** `{dataset_kind}`  
**Model:** `{args.model}`  
**Samples:** {len(frame)} total, {int(is_known.sum())} known-class, {int((~is_known).sum())} `_unknown`  
**Sessions/groups:** {len(split_groups)}  
**Subjects:** {', '.join(split_subjects) if split_subjects else 'n/a'}  
**Subject-disjoint from training:** {'yes' if subject_disjoint else 'no / not guaranteed'}  
**Unseen subjects in this split:** {', '.join(unseen_subjects) if unseen_subjects else 'none'}

## Forced classification (known classes only)

Accuracy ignores the reject option. Use it to diagnose the classifier, not to
describe deployed recognition performance.

- Accuracy: **{_format_metric(known_metrics['accuracy'])}**
- Accuracy 95% CI: {_format_ci(bootstrap_ci.get('accuracy', {}))}
- Macro precision: {_format_metric(known_metrics['macro_precision'])}
- Macro recall: {_format_metric(known_metrics['macro_recall'])}
- Macro F1: {_format_metric(known_metrics['macro_f1'])}
- Macro F1 95% CI: {_format_ci(bootstrap_ci.get('macro_f1', {}))}
- Weighted F1: {_format_metric(known_metrics['weighted_f1'])}
- Majority-class baseline accuracy: {_format_metric(majority_baseline)}

## Rejection behaviour

- Coverage on known samples: {_format_metric(rejection['known'].get('coverage'))}
- Selective accuracy on accepted known samples: {_format_metric(rejection['known'].get('selective_accuracy'))}
- Unknown-sign acceptance rate: {_format_metric(rejection['unknown'].get('accept_rate'))}
- Thresholds: confidence ≥ {bundle.thresholds['min_confidence']:.2f}, margin ≥ {bundle.thresholds['min_margin']:.2f}

## Calibration

- Expected calibration error (ECE): {_format_metric(calibration_diagnostics.get('ece'))}
- Brier score: {_format_metric(calibration_diagnostics.get('brier'))}

## Per-class metrics

{_markdown_table(known_metrics, labels)}

## Per-subject breakdown (known classes only)

| subject | known n | forced accuracy | forced macro-F1 | coverage | selective accuracy |
|---|---:|---:|---:|---:|---:|
{subject_lines}

## Most frequent confusions (forced predictions)

{confusion_lines}

## Interpretation limits

- This is American Sign Language **static fingerspelling**, not translation.
- J and Z are excluded because they require motion.
- A static classifier cannot recognise grammar, facial expression, or continuous signing.
- Confidence is not accuracy. Only the figures above, from this split, are measured.
- Do not tune on the test split. Do not report synthetic metrics as sign recognition.

## Files

- `metrics.json` — machine-readable results
- `classification_report.txt` — sklearn report + rejection summary
- `predictions.csv` — per-sample predictions and threshold decisions
- `confusion_matrix.png` — forced-prediction confusion matrix
- `reliability.png` — reliability diagram (confidence vs observed accuracy)
"""
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    print("\n" + "=" * 78)
    if warning:
        print(warning)
        print("=" * 78)
    print(f"Evaluation split: {args.split} | dataset kind: {dataset_kind}")
    print(
        f"Known samples: {int(is_known.sum())} | unknown: {int((~is_known).sum())} | "
        f"sessions: {len(split_groups)} | unseen subjects: {unseen_subjects or 'none'}"
    )
    print(f"Forced accuracy: {_format_metric(known_metrics['accuracy'])}")
    print(f"  accuracy CI:   {_format_ci(bootstrap_ci.get('accuracy', {}))}")
    print(f"Forced macro-F1: {_format_metric(known_metrics['macro_f1'])}")
    print(f"  macro-F1 CI:   {_format_ci(bootstrap_ci.get('macro_f1', {}))}")
    print(
        f"Selective accuracy: {_format_metric(rejection['known'].get('selective_accuracy'))} "
        f"at coverage {_format_metric(rejection['known'].get('coverage'))}"
    )
    print(
        f"Unknown acceptance rate: {_format_metric(rejection['unknown'].get('accept_rate'))} "
        f"({rejection['unknown'].get('n_unknown', 0)} samples)"
    )
    print(f"ECE: {_format_metric(calibration_diagnostics.get('ece'))}")
    print(f"Report: {output_dir / 'report.md'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
