"""Validation-only threshold calibration for unknown-sign rejection.

The rejection rule is intentionally simple and auditable:

    accept  <=>  top1_probability >= min_confidence
                 and (top1 - top2) >= min_margin

Thresholds are selected on the validation split.  The test split must never be
used here.  If rejection samples (``_unknown``) exist in validation, the search
also constrains how often those are wrongly accepted; otherwise the result is
explicitly marked as not validated against unknown signs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import UNKNOWN_LABEL
from src.model_bundle import (
    _prediction_from_probabilities,
    default_thresholds,
    validate_thresholds,
)


def _acceptance_stats(
    probabilities: np.ndarray,
    correct: np.ndarray,
    *,
    min_confidence: float,
    min_margin: float,
    unknown_probabilities: np.ndarray | None,
) -> dict[str, float | int | None]:
    if len(probabilities) == 0:
        return {
            "n_accepted": 0,
            "coverage": 0.0,
            "selective_accuracy": None,
            "unknown_accept_rate": None,
        }

    order = np.argsort(probabilities, axis=1)[:, ::-1]
    top1 = probabilities[np.arange(len(probabilities)), order[:, 0]]
    if probabilities.shape[1] > 1:
        top2 = probabilities[np.arange(len(probabilities)), order[:, 1]]
        margin = top1 - top2
    else:
        margin = top1
    accepted = (top1 >= min_confidence) & (margin >= min_margin)

    n_accepted = int(accepted.sum())
    selective_accuracy = float(correct[accepted].mean()) if n_accepted else None

    unknown_rate: float | None = None
    if unknown_probabilities is not None and len(unknown_probabilities):
        unknown_order = np.argsort(unknown_probabilities, axis=1)[:, ::-1]
        unknown_top1 = unknown_probabilities[np.arange(len(unknown_probabilities)), unknown_order[:, 0]]
        if unknown_probabilities.shape[1] > 1:
            unknown_top2 = unknown_probabilities[
                np.arange(len(unknown_probabilities)), unknown_order[:, 1]
            ]
            unknown_margin = unknown_top1 - unknown_top2
        else:
            unknown_margin = unknown_top1
        unknown_accepted = (unknown_top1 >= min_confidence) & (
            unknown_margin >= min_margin
        )
        unknown_rate = float(unknown_accepted.mean())

    return {
        "n_accepted": n_accepted,
        "coverage": float(n_accepted / len(probabilities)),
        "selective_accuracy": selective_accuracy,
        "unknown_accept_rate": unknown_rate,
    }


def calibrate_thresholds(
    y_known: np.ndarray | list[str],
    probabilities_known: np.ndarray,
    labels: list[str],
    *,
    probabilities_unknown: np.ndarray | None = None,
    target_selective_accuracy: float = 0.95,
    max_unknown_accept_rate: float = 0.20,
    min_coverage: float = 0.30,
) -> dict[str, Any]:
    """Choose confidence/margin thresholds using validation data.

    Returns a JSON-serialisable dictionary containing the thresholds, their
    validation statistics, the search constraints, and a status string.
    """

    labels = [str(label) for label in labels]
    y_known = np.asarray([str(value) for value in y_known], dtype=object)
    probabilities_known = np.asarray(probabilities_known, dtype=np.float64)
    if probabilities_known.ndim != 2 or probabilities_known.shape != (len(y_known), len(labels)):
        raise ValueError(
            "probabilities_known must have shape (n_known, n_labels); "
            f"got {probabilities_known.shape} for {len(y_known)} rows and {len(labels)} labels."
        )
    if len(y_known) == 0:
        raise ValueError("Cannot calibrate thresholds without validation predictions.")

    known_order = np.argsort(probabilities_known, axis=1)[:, ::-1]
    predicted = np.asarray(labels, dtype=object)[known_order[:, 0]]
    correct = predicted == y_known
    forced_accuracy = float(correct.mean())

    if probabilities_unknown is not None:
        probabilities_unknown = np.asarray(probabilities_unknown, dtype=np.float64)
        if probabilities_unknown.ndim != 2 or probabilities_unknown.shape[1] != len(labels):
            raise ValueError("probabilities_unknown has an incompatible shape.")
        if len(probabilities_unknown) == 0:
            probabilities_unknown = None

    confidence_grid = np.round(np.linspace(0.30, 0.95, 14), 4)
    margin_grid = np.round(np.linspace(0.0, 0.50, 11), 4)

    candidates: list[dict[str, Any]] = []
    for confidence in confidence_grid:
        for margin in margin_grid:
            stats = _acceptance_stats(
                probabilities_known,
                correct,
                min_confidence=float(confidence),
                min_margin=float(margin),
                unknown_probabilities=probabilities_unknown,
            )
            if int(stats["n_accepted"] or 0) == 0:
                continue
            selective = stats["selective_accuracy"]
            unknown_rate = stats["unknown_accept_rate"]
            feasible = (
                selective is not None
                and selective >= target_selective_accuracy
                and float(stats["coverage"] or 0.0) >= min_coverage
                and (
                    unknown_rate is None
                    or unknown_rate <= max_unknown_accept_rate
                )
            )
            candidates.append(
                {
                    "min_confidence": float(confidence),
                    "min_margin": float(margin),
                    "feasible": bool(feasible),
                    **stats,
                }
            )

    if not candidates:
        result = default_thresholds()
        result.update(
            {
                "source": "default_heuristic",
                "status": "calibration_failed_no_acceptances",
                "forced_accuracy": forced_accuracy,
                "n_known": int(len(y_known)),
                "n_unknown": int(len(probabilities_unknown)) if probabilities_unknown is not None else 0,
                "constraints": {
                    "target_selective_accuracy": target_selective_accuracy,
                    "max_unknown_accept_rate": max_unknown_accept_rate,
                    "min_coverage": min_coverage,
                },
                "message": "No threshold pair accepted any validation sample.",
            }
        )
        return result

    feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]
    if feasible_candidates:
        # Prefer the widest usable operating range (coverage), then higher
        # selective accuracy, then a lower false-accept rate for unknown signs.
        best = max(
            feasible_candidates,
            key=lambda item: (
                item["coverage"],
                item["selective_accuracy"] or 0.0,
                -(item["unknown_accept_rate"] or 0.0),
            ),
        )
        status = "calibrated_on_validation"
        message = (
            "Thresholds satisfy the selective-accuracy constraint on validation data."
        )
    else:
        # No pair met every constraint. Fall back to the heuristics but report
        # the closest operating point so the limitation is visible.
        best = max(
            candidates,
            key=lambda item: (
                (item["selective_accuracy"] or 0.0) - (item["unknown_accept_rate"] or 0.0),
                item["coverage"],
            ),
        )
        status = "constraints_not_met"
        message = (
            "Validation data could not satisfy the requested selective accuracy, "
            "coverage, and unknown-accept constraints. Defaults are kept; treat "
            "recognition as experimental."
        )

    result = {
        "min_confidence": float(best["min_confidence"]),
        "min_margin": float(best["min_margin"]),
        "source": "validation",
        "status": status,
        "message": message,
        "target_selective_accuracy": float(target_selective_accuracy),
        "max_unknown_accept_rate": float(max_unknown_accept_rate),
        "min_coverage": float(min_coverage),
        "coverage": float(best["coverage"]),
        "selective_accuracy": (
            float(best["selective_accuracy"]) if best["selective_accuracy"] is not None else None
        ),
        "unknown_accept_rate": (
            float(best["unknown_accept_rate"])
            if best["unknown_accept_rate"] is not None
            else None
        ),
        "n_accepted": int(best["n_accepted"]),
        "n_known": int(len(y_known)),
        "n_unknown": int(len(probabilities_unknown)) if probabilities_unknown is not None else 0,
        "forced_accuracy": forced_accuracy,
        "unknown_data_used": probabilities_unknown is not None,
        "search_space": {
            "confidence": [float(confidence_grid.min()), float(confidence_grid.max()), len(confidence_grid)],
            "margin": [float(margin_grid.min()), float(margin_grid.max()), len(margin_grid)],
        },
        "limitations": [
            "Thresholds are tuned on one validation split and may not generalise.",
            "If no '_unknown' samples were present, unknown-sign rejection is unmeasured.",
        ],
    }
    return validate_thresholds(result)


def check_thresholds_on_split(
    y_true: np.ndarray | list[str],
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Apply saved thresholds to any split and return acceptance statistics."""

    y_true = np.asarray([str(value) for value in y_true], dtype=object)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = [str(label) for label in labels]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(y_true):
        raise ValueError("Probabilities and labels have incompatible shapes.")

    known_mask = y_true != UNKNOWN_LABEL
    active = validate_thresholds(thresholds)

    known_stats: dict[str, Any] = {
        "n_known": int(known_mask.sum()),
        "coverage": None,
        "selective_accuracy": None,
        "forced_accuracy": None,
        "rejected_known": None,
    }
    if known_mask.any():
        known_probs = probabilities[known_mask]
        known_true = y_true[known_mask]
        correct = np.asarray(labels, dtype=object)[np.argmax(known_probs, axis=1)] == known_true
        stats = _acceptance_stats(
            known_probs,
            correct,
            min_confidence=active["min_confidence"],
            min_margin=active["min_margin"],
            unknown_probabilities=None,
        )
        known_stats.update(
            {
                "coverage": stats["coverage"],
                "selective_accuracy": stats["selective_accuracy"],
                "forced_accuracy": float(correct.mean()),
                "rejected_known": int(len(known_true) - (stats["n_accepted"] or 0)),
            }
        )

    unknown_stats: dict[str, Any] = {
        "n_unknown": int((~known_mask).sum()),
        "accept_rate": None,
    }
    if (~known_mask).any():
        unknown_probs = probabilities[~known_mask]
        dummy = np.ones(len(unknown_probs), dtype=bool)
        stats = _acceptance_stats(
            unknown_probs,
            dummy,
            min_confidence=active["min_confidence"],
            min_margin=active["min_margin"],
            unknown_probabilities=unknown_probs,
        )
        unknown_stats["accept_rate"] = stats["unknown_accept_rate"]

    return {
        "thresholds": active,
        "known": known_stats,
        "unknown": unknown_stats,
    }
