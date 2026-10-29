"""Reusable model-comparison and ablation experiments.

The final training script deliberately reads only train/validation splits.  This
module is for exploratory comparison: it fits several candidate models on the
same grouped training split, calibrates rejection thresholds on validation, and
then reports validation and (optionally) held-out test metrics side by side.

Selecting a model on validation is leakage-safe.  Looking at test metrics for
every candidate is still useful for a portfolio report, but the test column is
descriptive once more than one candidate has been considered; the selected
model's test result is the only one that can be described as its final estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.augmentation import augment_features
from src.calibration import calibrate_thresholds, check_thresholds_on_split
from src.config import FEATURE_DIM, RAW_COORD_COLUMNS, UNKNOWN_LABEL
from src.dataset import group_ids, load_processed_splits
from src.evaluation import (
    classification_metrics,
    expected_calibration_error,
    grouped_bootstrap_ci,
    known_mask,
    per_subject_breakdown,
)
from src.features import (
    FEATURE_SET_DIMS,
    build_feature_matrix,
    feature_table_to_array,
    geometry_features,
)
from src.model_bundle import HYPERPARAMETER_GRIDS, build_estimator


@dataclass
class Candidate:
    model_name: str
    feature_set: str
    estimator: Any
    labels: list[str]
    thresholds: dict[str, Any]
    calibration: dict[str, Any]
    training_seconds: float
    best_params: dict[str, Any] | None
    cv_score: float | None
    n_train: int


def features_for_frame(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    if feature_set not in FEATURE_SET_DIMS:
        raise ValueError(f"Unknown feature set {feature_set!r}.")
    if frame.empty:
        return np.empty((0, FEATURE_SET_DIMS[feature_set]), dtype=np.float64)
    return build_feature_matrix(
        feature_table_to_array(frame, RAW_COORD_COLUMNS), feature_set=feature_set
    )


def augment_matrix(
    X: np.ndarray,
    copies: int,
    seed: int,
    feature_set: str,
) -> np.ndarray:
    if copies <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    augmented = np.empty((len(X) * copies, X.shape[1]), dtype=np.float64)
    cursor = 0
    for row in X:
        for _ in range(copies):
            base = augment_features(row[:FEATURE_DIM], rng)
            if feature_set == "geometry":
                augmented[cursor] = np.concatenate([base, geometry_features(base)])
            else:
                augmented[cursor] = base
            cursor += 1
    return augmented


def prepare_training_data(
    splits: dict[str, pd.DataFrame],
    *,
    feature_set: str,
    augment_copies: int,
    seed: int,
) -> dict[str, Any]:
    train = splits["train"]
    val = splits["val"]
    train_known = train[train["label"].astype(str) != UNKNOWN_LABEL].copy()
    val_known = val[val["label"].astype(str) != UNKNOWN_LABEL].copy()
    val_unknown = val[val["label"].astype(str) == UNKNOWN_LABEL].copy()
    if train_known.empty or val_known.empty:
        raise ValueError("Train and validation splits must contain known samples.")

    X_train = features_for_frame(train_known, feature_set)
    y_train = train_known["label"].astype(str).to_numpy()
    groups_train = group_ids(train_known).to_numpy()
    X_augmented = augment_matrix(X_train, augment_copies, seed, feature_set)
    if len(X_augmented):
        X_fit = np.vstack([X_train, X_augmented])
        y_fit = np.concatenate([y_train, np.tile(y_train, augment_copies)])
        groups_fit = np.concatenate(
            [groups_train, np.tile(groups_train, augment_copies)]
        )
    else:
        X_fit, y_fit, groups_fit = X_train, y_train, groups_train
    return {
        "train_known": train_known,
        "val_known": val_known,
        "val_unknown": val_unknown,
        "X_train": X_train,
        "y_train": y_train,
        "X_fit": X_fit,
        "y_fit": y_fit,
        "groups_fit": groups_fit,
        "X_val": features_for_frame(val_known, feature_set),
        "y_val": val_known["label"].astype(str).to_numpy(),
        "X_val_unknown": features_for_frame(val_unknown, feature_set),
        "n_augmented": int(len(X_augmented)),
    }


def fit_candidate(
    model_name: str,
    prepared: dict[str, Any],
    *,
    feature_set: str,
    seed: int = 42,
    tune: bool = False,
    cv: int = 3,
    target_selective_accuracy: float = 0.90,
    max_unknown_accept_rate: float = 0.20,
    min_coverage: float = 0.30,
) -> Candidate:
    started = time.perf_counter()
    estimator = build_estimator(model_name, seed=seed)
    best_params: dict[str, Any] | None = None
    cv_score: float | None = None
    if tune:
        folds = min(cv, len(np.unique(prepared["groups_fit"])))
        if folds >= 2:
            search = GridSearchCV(
                estimator,
                HYPERPARAMETER_GRIDS[model_name],
                scoring="f1_macro",
                cv=StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed),
                n_jobs=1,
                refit=True,
            )
            search.fit(prepared["X_fit"], prepared["y_fit"], groups=prepared["groups_fit"])
            estimator = search.best_estimator_
            best_params = dict(search.best_params_)
            cv_score = float(search.best_score_)
        else:
            estimator.fit(prepared["X_fit"], prepared["y_fit"])
    else:
        estimator.fit(prepared["X_fit"], prepared["y_fit"])

    labels = [str(label) for label in estimator.classes_]
    probabilities_val = estimator.predict_proba(prepared["X_val"])
    probabilities_unknown = (
        estimator.predict_proba(prepared["X_val_unknown"])
        if len(prepared["X_val_unknown"])
        else None
    )
    calibration = calibrate_thresholds(
        prepared["y_val"],
        probabilities_val,
        labels,
        probabilities_unknown=probabilities_unknown,
        target_selective_accuracy=target_selective_accuracy,
        max_unknown_accept_rate=max_unknown_accept_rate,
        min_coverage=min_coverage,
    )
    return Candidate(
        model_name=model_name,
        feature_set=feature_set,
        estimator=estimator,
        labels=labels,
        thresholds={
            "min_confidence": calibration["min_confidence"],
            "min_margin": calibration["min_margin"],
            "source": calibration.get("source"),
            "status": calibration.get("status"),
        },
        calibration=calibration,
        training_seconds=float(time.perf_counter() - started),
        best_params=best_params,
        cv_score=cv_score,
        n_train=int(len(prepared["X_fit"])),
    )


def evaluate_on_frame(
    candidate: Candidate,
    frame: pd.DataFrame,
    *,
    split_name: str,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    X = features_for_frame(frame, candidate.feature_set)
    probabilities = candidate.estimator.predict_proba(X)
    forced_predictions = np.asarray(candidate.labels, dtype=object)[
        np.argmax(probabilities, axis=1)
    ]
    y_true = frame["label"].astype(str).to_numpy()
    is_known = known_mask(y_true)
    if not is_known.any():
        raise ValueError(f"Split {split_name!r} has no known-class samples.")

    known_metrics = classification_metrics(
        y_true[is_known], forced_predictions[is_known], candidate.labels
    )
    diagnostics = expected_calibration_error(
        y_true[is_known], probabilities[is_known], candidate.labels
    )
    rejection = check_thresholds_on_split(
        y_true, probabilities, candidate.labels, candidate.thresholds
    )
    margins = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    accepted = (
        (probabilities.max(axis=1) >= candidate.thresholds["min_confidence"])
        & (margins >= candidate.thresholds["min_margin"])
    )
    groups = group_ids(frame).to_numpy()
    bootstrap: dict[str, Any] = {"note": "Disabled with --bootstrap 0."}
    if n_boot > 0:
        bootstrap = {
            "accuracy": grouped_bootstrap_ci(
                y_true[is_known],
                forced_predictions[is_known],
                groups[is_known],
                candidate.labels,
                metric="accuracy",
                n_boot=n_boot,
                seed=seed,
            ),
            "macro_f1": grouped_bootstrap_ci(
                y_true[is_known],
                forced_predictions[is_known],
                groups[is_known],
                candidate.labels,
                metric="macro_f1",
                n_boot=n_boot,
                seed=seed,
            ),
        }
    known_rejection = rejection.get("known", {})
    unknown_rejection = rejection.get("unknown", {})
    return {
        "split": split_name,
        "n_samples": int(len(frame)),
        "n_known": int(is_known.sum()),
        "n_unknown": int((~is_known).sum()),
        "n_groups": int(len(np.unique(groups))),
        "subjects": sorted(frame["subject_id"].astype(str).unique().tolist()),
        "forced_accuracy": known_metrics["accuracy"],
        "macro_f1": known_metrics["macro_f1"],
        "weighted_f1": known_metrics["weighted_f1"],
        "selective_accuracy": known_rejection.get("selective_accuracy"),
        "coverage": known_rejection.get("coverage"),
        "unknown_accept_rate": unknown_rejection.get("accept_rate"),
        "ece": diagnostics.get("ece"),
        "brier": diagnostics.get("brier"),
        "bootstrap_ci_95": bootstrap,
        "per_class": known_metrics["per_class"],
        "confusion_matrix": known_metrics["confusion_matrix"],
        "per_subject": per_subject_breakdown(
            y_true,
            forced_predictions,
            frame["subject_id"].astype(str).to_numpy(),
            accepted,
            candidate.labels,
        ),
        "bins": diagnostics.get("bins", []),
        "rejection": rejection,
    }


def candidate_summary(candidate: Candidate, validation: dict[str, Any], test: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "model": candidate.model_name,
        "feature_set": candidate.feature_set,
        "labels": candidate.labels,
        "thresholds": candidate.thresholds,
        "calibration_status": candidate.calibration.get("status"),
        "training_seconds": round(candidate.training_seconds, 4),
        "n_train": candidate.n_train,
        "best_params": candidate.best_params,
        "cv_score": candidate.cv_score,
        "validation": validation,
        "test": test,
    }


def load_splits(processed_dir: Path | str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    return load_processed_splits(processed_dir, names=("train", "val", "test"))
