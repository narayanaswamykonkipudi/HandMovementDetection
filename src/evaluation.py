"""Metric computation, calibration diagnostics, and report artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import UNKNOWN_LABEL


def classification_metrics(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    labels: list[str],
) -> dict[str, Any]:
    labels = [str(label) for label in labels]
    y_true = np.asarray([str(value) for value in y_true])
    y_pred = np.asarray([str(value) for value in y_pred])
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")

    accuracy = float(accuracy_score(y_true, y_pred)) if len(y_true) else None
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_samples": int(len(y_true)),
        "labels": labels,
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(per_support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
    }


def expected_calibration_error(
    y_true: np.ndarray | list[str],
    probabilities: np.ndarray,
    labels: list[str],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute multiclass ECE and Brier score for forced top-1 predictions.

    ECE measures whether stated confidence matches observed accuracy.  It is
    reported separately from accuracy and does not validate rejection.
    """

    y_true = np.asarray([str(value) for value in y_true])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = [str(label) for label in labels]
    if len(y_true) == 0:
        return {"ece": None, "brier": None, "bins": []}
    if probabilities.shape != (len(y_true), len(labels)):
        raise ValueError("Probability matrix shape does not match labels/y_true.")

    top_indices = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(len(y_true)), top_indices]
    predictions = np.asarray(labels)[top_indices]
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for bin_index in range(n_bins):
        lower, upper = bin_edges[bin_index], bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        weight = float(mask.sum() / len(y_true))
        ece += abs(bin_accuracy - bin_confidence) * weight
        bins.append(
            {
                "lower": round(float(lower), 3),
                "upper": round(float(upper), 3),
                "n": int(mask.sum()),
                "accuracy": round(bin_accuracy, 4),
                "mean_confidence": round(bin_confidence, 4),
            }
        )

    one_hot = np.zeros_like(probabilities)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row_index, label in enumerate(y_true):
        if label in label_to_index:
            one_hot[row_index, label_to_index[label]] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    return {"ece": float(ece), "brier": brier, "bins": bins}


def _metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    metric: str,
) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )[2]
        )
    raise ValueError("metric must be 'accuracy' or 'macro_f1'.")


def grouped_bootstrap_ci(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    groups: np.ndarray | list[str],
    labels: list[str],
    *,
    metric: str = "accuracy",
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Group-aware percentile bootstrap confidence interval.

    Resampling whole sessions/subject-sessions rather than individual frames
    avoids pretending that strongly correlated frames are independent.
    """

    y_true = np.asarray([str(value) for value in y_true])
    y_pred = np.asarray([str(value) for value in y_pred])
    groups = np.asarray([str(value) for value in groups])
    labels = [str(label) for label in labels]
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred, and groups must have equal length.")
    if len(y_true) == 0:
        return {
            "metric": metric,
            "point": None,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "n_boot": 0,
            "n_groups": 0,
        }

    point = _metric_value(y_true, y_pred, labels, metric)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {
            "metric": metric,
            "point": point,
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "n_boot": 0,
            "n_groups": int(len(unique_groups)),
            "note": "At least two groups are required for a bootstrap interval.",
        }

    indices_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        estimates.append(_metric_value(y_true[indices], y_pred[indices], labels, metric))

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "metric": metric,
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_boot": int(n_boot),
        "n_groups": int(len(unique_groups)),
        "method": "group-level percentile bootstrap",
        "seed": seed,
    }


def per_subject_breakdown(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    subjects: np.ndarray | list[str],
    accepted: np.ndarray | list[bool],
    labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Forced and selective metrics for each subject (known classes only)."""

    y_true = np.asarray([str(value) for value in y_true])
    y_pred = np.asarray([str(value) for value in y_pred])
    subjects = np.asarray([str(value) for value in subjects])
    accepted = np.asarray(accepted, dtype=bool)
    labels = [str(label) for label in labels]
    if not (len(y_true) == len(y_pred) == len(subjects) == len(accepted)):
        raise ValueError("Per-subject inputs must have equal length.")

    result: dict[str, dict[str, Any]] = {}
    for subject in sorted(np.unique(subjects)):
        mask = subjects == subject
        subject_true = y_true[mask]
        subject_pred = y_pred[mask]
        subject_accepted = accepted[mask]
        known = known_mask(subject_true)
        n_known = int(known.sum())
        accepted_known = known & subject_accepted
        result[str(subject)] = {
            "n_samples": int(mask.sum()),
            "n_known": n_known,
            "forced_accuracy": (
                float((subject_true[known] == subject_pred[known]).mean())
                if n_known
                else None
            ),
            "forced_macro_f1": (
                _metric_value(subject_true[known], subject_pred[known], labels, "macro_f1")
                if n_known
                else None
            ),
            "coverage": (
                float(accepted_known.sum() / n_known) if n_known else None
            ),
            "selective_accuracy": (
                float(
                    (
                        subject_true[accepted_known]
                        == subject_pred[accepted_known]
                    ).mean()
                )
                if accepted_known.any()
                else None
            ),
        }
    return result


def plot_reliability(
    calibration_diagnostics: dict[str, Any],
    output_path: Path | str,
    *,
    title: str,
    note: str | None = None,
) -> Path:
    """Plot observed accuracy against mean confidence for non-empty bins."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = calibration_diagnostics.get("bins", [])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    if bins:
        confidence = [item["mean_confidence"] for item in bins]
        accuracy = [item["accuracy"] for item in bins]
        weights = [item["n"] for item in bins]
        axis.plot(confidence, accuracy, marker="o", color="#1f6f54", label="Observed")
        axis.scatter(confidence, accuracy, s=[max(18, weight * 1.2) for weight in weights], color="#1f6f54", alpha=0.65)
    axis.set(
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
        title=title,
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=8)
    axis.text(
        0.02,
        0.02,
        f"ECE={_format_optional(calibration_diagnostics.get('ece'))}  "
        f"Brier={_format_optional(calibration_diagnostics.get('brier'))}",
        transform=axis.transAxes,
        fontsize=8,
    )
    if note:
        figure.text(0.01, 0.01, note, fontsize=7, color="#8a3b12", wrap=True)
    figure.tight_layout(rect=(0, 0.04, 1, 1) if note else None)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def confusion_pairs(
    y_true: np.ndarray | list[str],
    y_pred: np.ndarray | list[str],
    *,
    limit: int = 15,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {"true": [str(value) for value in y_true], "predicted": [str(value) for value in y_pred]}
    )
    errors = frame[frame["true"] != frame["predicted"]]
    if errors.empty:
        return []
    pairs = (
        errors.groupby(["true", "predicted"], observed=True)
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )
    return [
        {"true": str(true), "predicted": str(predicted), "count": int(count)}
        for (true, predicted), count in pairs.items()
    ]


def plot_confusion_matrix(
    matrix: np.ndarray | list[list[int]],
    labels: list[str],
    output_path: Path | str,
    *,
    title: str,
    note: str | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(7.0, 0.34 * len(labels) + 3.0)
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )
    if note:
        figure.text(0.01, 0.01, note, fontsize=7, color="#8a3b12", wrap=True)
    figure.tight_layout(rect=(0, 0.03, 1, 1) if note else None)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def synthetic_warning(dataset_kind: str) -> str | None:
    if dataset_kind == "synthetic":
        return (
            "SYNTHETIC DATASET: these metrics describe a plumbing test on generated "
            "landmarks. They are NOT sign-language recognition accuracy and must not "
            "be reported as such."
        )
    if dataset_kind == "mixed":
        return (
            "MIXED DATASET: synthetic and real/imported samples are combined. "
            "Do not present aggregate metrics as real recognition accuracy."
        )
    return None


def known_mask(labels: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray([str(label) != UNKNOWN_LABEL for label in labels])
