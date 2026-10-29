#!/usr/bin/env python
"""Train a leakage-aware ASL static-fingerspelling classifier.

Usage
-----
    python train_model.py --processed data/processed --model svm \
        --out models/classifier.joblib

Only train and validation splits are read. The test split is deliberately left
untouched; use evaluate_model.py exactly once for the final estimate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.augmentation import augment_features
from src.calibration import calibrate_thresholds
from src.config import (
    ARTIFACTS_DIR,
    CLASSIFIER_PATH,
    FEATURE_DIM,
    PROCESSED_DIR,
    RAW_COORD_COLUMNS,
    STATIC_ASL_LETTERS,
    UNKNOWN_LABEL,
)
from src.dataset import (
    DatasetError,
    group_ids,
    load_processed_splits,
    utc_now_iso,
)
from src.evaluation import classification_metrics, synthetic_warning, write_json
from src.experiment_log import append_experiment, dataset_manifest_hash
from src.features import (
    FEATURE_SET_DIMS,
    build_feature_matrix,
    feature_table_to_array,
    geometry_features,
)
from src.model_bundle import (
    HYPERPARAMETER_GRIDS,
    build_estimator,
    save_model_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model", choices=("svm", "rf", "mlp"), default="svm")
    parser.add_argument(
        "--feature-set",
        choices=("base", "geometry"),
        default="base",
        help="base=63 landmarks; geometry=88 landmarks plus distances/angles.",
    )
    parser.add_argument("--out", type=Path, default=CLASSIFIER_PATH)
    parser.add_argument("--card", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Folds for hyperparameter search (grouped, never random).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip grid search and use the estimator defaults.",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=1,
        help="Extra augmented copies of each training sample (0 disables).",
    )
    parser.add_argument("--target-selective-accuracy", type=float, default=0.90)
    parser.add_argument("--max-unknown-accept-rate", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.30)
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=ARTIFACTS_DIR / "experiments.jsonl",
        help="Append-only JSONL experiment log.",
    )
    parser.add_argument("--no-experiment-log", action="store_true")
    return parser.parse_args()


def _features(frame, feature_set: str) -> np.ndarray:
    return build_feature_matrix(
        feature_table_to_array(frame, RAW_COORD_COLUMNS), feature_set=feature_set
    )


def _augment_copies(
    X: np.ndarray, copies: int, seed: int, feature_set: str
) -> tuple[np.ndarray, int]:
    if copies <= 0:
        return np.empty((0, FEATURE_SET_DIMS[feature_set]), dtype=np.float64), 0
    rng = np.random.default_rng(seed)
    augmented = np.empty(
        (len(X) * copies, FEATURE_SET_DIMS[feature_set]), dtype=np.float64
    )
    cursor = 0
    for row in X:
        for _ in range(copies):
            base = augment_features(row[:FEATURE_DIM], rng)
            if feature_set == "geometry":
                augmented[cursor] = np.concatenate([base, geometry_features(base)])
            else:
                augmented[cursor] = base
            cursor += 1
    return augmented, copies


def _environment() -> dict[str, str]:
    import joblib
    import mediapipe
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "mediapipe": mediapipe.__version__,
    }


def _find_best_params(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    cv_requested: int,
    seed: int,
) -> tuple[Any, dict[str, Any] | None, float | None, int, list[str]]:
    notes_warnings: list[str] = []
    n_groups = len(set(groups.tolist()))
    min_class = (
        int(np.unique(y, return_counts=True)[1].min()) if len(y) else 0
    )
    n_splits = min(cv_requested, n_groups, min_class)
    if n_splits < 2:
        notes_warnings.append(
            "Hyperparameter search skipped: fewer than two groups/classes available."
        )
        return estimator, None, None, 0, notes_warnings

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator,
        HYPERPARAMETER_GRIDS[model_name],
        scoring="f1_macro",
        cv=splitter,
        n_jobs=1,
        refit=True,
        error_score="raise",
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    notes_warnings.append(
        f"Grouped {n_splits}-fold search; best training-fold macro-F1={search.best_score_:.4f}."
    )
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), n_splits, notes_warnings


def main() -> int:
    args = parse_args()
    try:
        splits, manifest = load_processed_splits(
            args.processed, names=("train", "val")
        )
    except (FileNotFoundError, DatasetError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    train = splits["train"]
    val = splits["val"]
    train_known = train[train["label"].astype(str) != UNKNOWN_LABEL].copy()
    val_known = val[val["label"].astype(str) != UNKNOWN_LABEL].copy()
    val_unknown = val[val["label"].astype(str) == UNKNOWN_LABEL].copy()

    if train_known.empty or val_known.empty:
        print("ERROR: train/validation must both contain known sign samples.", file=sys.stderr)
        return 3

    missing_train = sorted(set(STATIC_ASL_LETTERS) - set(train_known["label"].astype(str)))
    if missing_train:
        print(f"ERROR: training split is missing letters: {missing_train}", file=sys.stderr)
        return 3

    dataset_kind = str(manifest.get("dataset", {}).get("dataset_kind", "unknown"))
    warning = synthetic_warning(dataset_kind)
    if warning:
        print("\n" + "!" * 78)
        print(warning)
        print("!" * 78 + "\n")

    print(f"Building features from {args.processed} …")
    X_train = _features(train_known, args.feature_set)
    X_val = _features(val_known, args.feature_set)
    X_val_unknown = (
        _features(val_unknown, args.feature_set)
        if not val_unknown.empty
        else np.empty((0, FEATURE_SET_DIMS[args.feature_set]))
    )
    y_train = train_known["label"].astype(str).to_numpy()
    y_val = val_known["label"].astype(str).to_numpy()
    groups_train = group_ids(train_known).to_numpy()

    X_augmented, copies = _augment_copies(
        X_train, args.augment_copies, args.seed, args.feature_set
    )
    if copies:
        X_fit = np.vstack([X_train, X_augmented])
        y_fit = np.concatenate([y_train, np.tile(y_train, copies)])
        groups_fit = np.concatenate([groups_train, np.tile(groups_train, copies)])
    else:
        X_fit, y_fit, groups_fit = X_train, y_train, groups_train

    if len(np.unique(y_train)) < 2:
        print("ERROR: at least two classes are required.", file=sys.stderr)
        return 3

    print(
        f"Training {args.model.upper()} on {len(X_train)} real/recorded samples "
        f"+ {len(X_augmented)} augmented copies ({len(np.unique(groups_train))} groups)."
    )

    estimator = build_estimator(args.model, seed=args.seed)
    best_params: dict[str, Any] | None = None
    cv_score: float | None = None
    cv_folds = 0
    tuning_notes: list[str] = []
    if not args.no_tune:
        try:
            estimator, best_params, cv_score, cv_folds, tuning_notes = _find_best_params(
                estimator,
                X_fit,
                y_fit,
                groups_fit,
                model_name=args.model,
                cv_requested=args.cv,
                seed=args.seed,
            )
        except ValueError as exc:
            print(f"WARNING: grouped hyperparameter search failed ({exc}); using defaults.")
            estimator = build_estimator(args.model, seed=args.seed)
            tuning_notes.append(f"Grid search failed: {exc}")
    else:
        tuning_notes.append("Hyperparameter search disabled with --no-tune.")

    if best_params is None:
        estimator.fit(X_fit, y_fit)

    labels = [str(label) for label in estimator.classes_]
    probabilities_val = estimator.predict_proba(X_val)
    predicted_val = np.asarray(labels)[np.argmax(probabilities_val, axis=1)]
    validation_metrics = classification_metrics(y_val, predicted_val, labels)

    probabilities_unknown = (
        estimator.predict_proba(X_val_unknown) if len(X_val_unknown) else None
    )
    calibration = calibrate_thresholds(
        y_val,
        probabilities_val,
        labels,
        probabilities_unknown=probabilities_unknown,
        target_selective_accuracy=args.target_selective_accuracy,
        max_unknown_accept_rate=args.max_unknown_accept_rate,
        min_coverage=args.min_coverage,
    )

    metadata = {
        "format_version": 2,
        "created_utc": utc_now_iso(),
        "feature_spec": {
            "landmark_count": 21,
            "feature_dim": FEATURE_SET_DIMS[args.feature_set],
            "feature_set": args.feature_set,
            "normalization": "wrist translation + mean palm-base scale; no orientation normalisation",
            "handedness_used": False,
            "geometry_extra_dim": 25 if args.feature_set == "geometry" else 0,
        },
        "training": {
            "dataset_kind": dataset_kind,
            "sources": manifest.get("dataset", {}).get("sources", {}),
            "n_train_known": int(len(train_known)),
            "n_train_augmented": int(len(X_augmented)),
            "n_val_known": int(len(val_known)),
            "n_val_unknown": int(len(val_unknown)),
            "n_train_groups": int(len(set(groups_train.tolist()))),
            "n_val_groups": int(val_known.groupby(["subject_id", "session_id"]).ngroups),
            "class_counts_train": {
                str(key): int(value)
                for key, value in train_known["label"].astype(str).value_counts().sort_index().items()
            },
            "model_name": args.model,
            "estimator_class": type(estimator.named_steps["classifier"]).__name__,
            "best_params": best_params,
            "cv_folds": int(cv_folds),
            "cv_macro_f1": cv_score,
            "augmentation": {
                "copies_per_sample": int(copies),
                "mirror_probability": 0.5,
                "max_rotation_degrees": 12.0,
                "scale_jitter": 0.06,
                "translation_jitter": 0.015,
                "noise_std": 0.006,
                "depth_noise_std": 0.004,
                "applied_to": "training only",
            },
            "seed": args.seed,
            "source_sha256": manifest.get("source_sha256"),
            "processed_manifest_created_utc": manifest.get("created_utc"),
            "labels": labels,
            "scope": {
                "language": "American Sign Language fingerspelling",
                "signs": "static handshapes",
                "excluded": ["J", "Z"],
                "not_supported": [
                    "dynamic signs",
                    "signed grammar",
                    "facial expression",
                    "complete translation",
                ],
            },
        },
        "split": manifest.get("split", {}),
        "validation_metrics": validation_metrics,
        "calibration": calibration,
        "environment": _environment(),
        "notes": [
            "Metrics in this bundle are validation metrics, not test accuracy.",
            "The test split was not read during training.",
            "Confidence is a model probability, not a guarantee of correctness.",
            "Rejection thresholds were selected on validation data only.",
        ],
    }

    out_path = save_model_bundle(
        args.out, estimator, thresholds=calibration, metadata=metadata
    )
    card_path = args.card or out_path.with_name(f"{out_path.stem}_card.json")
    card = {
        "model_file": out_path.name,
        "warning": warning,
        "scope": metadata["training"]["scope"],
        "dataset_kind": dataset_kind,
        "sources": metadata["training"]["sources"],
        "training": metadata["training"],
        "validation_metrics": validation_metrics,
        "calibration": calibration,
        "environment": metadata["environment"],
        "test_policy": "Test split not read; run evaluate_model.py on the held-out test split.",
        "limitations": [
            "Static single-frame classifier; J and Z are not supported.",
            "Not a translator; no grammar, facial expression, or continuous signing.",
            "Accuracy on unseen people must be measured with a subject-held-out test split.",
            "Synthetic-data metrics are plumbing checks, never recognition claims.",
        ],
    }
    write_json(card_path, card)

    if not args.no_experiment_log:
        record = append_experiment(
            args.experiment_log,
            kind="train",
            params={
                "model": args.model,
                "feature_set": args.feature_set,
                "augment_copies": args.augment_copies,
                "tune": not args.no_tune,
                "seed": args.seed,
                "best_params": best_params,
            },
            metrics={
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_selective_accuracy": calibration.get("selective_accuracy"),
                "validation_coverage": calibration.get("coverage"),
                "validation_unknown_accept_rate": calibration.get("unknown_accept_rate"),
                "min_confidence": calibration.get("min_confidence"),
                "min_margin": calibration.get("min_margin"),
            },
            dataset_manifest=dataset_manifest_hash(args.processed),
            extra={
                "model_path": str(out_path),
                "model_card_path": str(card_path),
                "dataset_kind": dataset_kind,
                "n_train_known": int(len(train_known)),
            },
        )
        print(f"  experiment log:     {args.experiment_log} ({record['run_id']})")

    print("\nTraining complete.")
    print(f"  model:              {out_path}")
    print(f"  model card:         {card_path}")
    print(f"  classes:            {len(labels)}")
    print(f"  validation accuracy {validation_metrics['accuracy']:.4f}")
    print(f"  validation macro-F1 {validation_metrics['macro_f1']:.4f}")
    print(
        f"  selective accuracy  {calibration.get('selective_accuracy')} "
        f"| coverage {calibration.get('coverage')} "
        f"| unknown accept {calibration.get('unknown_accept_rate')}"
    )
    print(
        f"  thresholds:         confidence>={calibration['min_confidence']:.2f}, "
        f"margin>={calibration['min_margin']:.2f} ({calibration.get('status')})"
    )
    for note in tuning_notes:
        print(f"  note: {note}")
    if warning:
        print("\n" + warning)
    print(
        "\nThe test split was NOT read. Final evaluation command:\n"
        f"  python evaluate_model.py --processed {args.processed} "
        f"--model {out_path} --split test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
