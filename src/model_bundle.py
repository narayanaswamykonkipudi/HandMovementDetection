"""Training pipelines, persisted model bundles, and rejection-aware prediction.

A model file is more than a pickle: inference needs the feature contract, label
order, rejection thresholds, and provenance.  We therefore persist a versioned
dictionary (``ModelBundle``) and validate every field on load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    FEATURE_DIM,
    MODEL_FORMAT_VERSION,
)
from src.dataset import utc_now_iso
from src.features import FEATURE_SET_DIMS, extract_feature_vector


class ModelBundleError(RuntimeError):
    """Raised when a model file is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class TopPrediction:
    label: str
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "probability": round(self.probability, 6)}


@dataclass(frozen=True)
class Prediction:
    """One frame-level prediction, before temporal smoothing."""

    accepted_label: str | None
    top_label: str
    confidence: float
    margin: float
    accepted: bool
    top: tuple[TopPrediction, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_label": self.accepted_label,
            "top_label": self.top_label,
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
            "accepted": self.accepted,
            "top": [item.to_dict() for item in self.top],
        }


def build_estimator(
    model_name: str = "svm",
    *,
    seed: int = 42,
    **hyperparameters: Any,
) -> Pipeline:
    """Construct an unfitted sklearn pipeline for landmark classification.

    ``svm`` is the default because RBF SVMs perform well on small, dense
    landmark datasets.  ``rf`` is offered as a lower-variance alternative and
    ``mlp`` is a small neural-network baseline for model comparison.
    """

    if model_name == "svm":
        svc_kwargs = {
            "C": 1.0,
            "gamma": "scale",
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": seed,
            "cache_size": 1024,
        }
        svc_kwargs.update(hyperparameters)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(**svc_kwargs)),
            ]
        )

    if model_name == "rf":
        rf_kwargs = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": seed,
            "n_jobs": -1,
        }
        rf_kwargs.update(hyperparameters)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", RandomForestClassifier(**rf_kwargs)),
            ]
        )

    if model_name == "mlp":
        mlp_kwargs = {
            "hidden_layer_sizes": (128,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 64,
            "learning_rate_init": 1e-3,
            "max_iter": 700,
            "random_state": seed,
        }
        mlp_kwargs.update(hyperparameters)
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", MLPClassifier(**mlp_kwargs)),
            ]
        )

    raise ValueError("model_name must be 'svm', 'rf', or 'mlp'.")


HYPERPARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "svm": {
        "classifier__C": [1.0, 10.0],
        "classifier__gamma": ["scale", 0.01],
    },
    "rf": {
        "classifier__max_depth": [None, 24],
        "classifier__min_samples_leaf": [1, 2],
    },
    "mlp": {
        "classifier__hidden_layer_sizes": [(64,), (128,)],
        "classifier__alpha": [1e-4, 1e-3],
    },
}


def validate_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    try:
        min_confidence = float(thresholds["min_confidence"])
        min_margin = float(thresholds["min_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelBundleError(
            "Model thresholds must contain numeric 'min_confidence' and 'min_margin'."
        ) from exc
    if not (0.0 <= min_confidence <= 1.0 and 0.0 <= min_margin <= 1.0):
        raise ModelBundleError("Confidence and margin thresholds must be in [0, 1].")
    cleaned = dict(thresholds)
    cleaned["min_confidence"] = min_confidence
    cleaned["min_margin"] = min_margin
    return cleaned


@dataclass
class ModelBundle:
    estimator: Pipeline
    labels: list[str]
    thresholds: dict[str, Any]
    metadata: dict[str, Any]
    feature_dim: int = FEATURE_DIM

    def __post_init__(self) -> None:
        self.labels = [str(label) for label in self.labels]
        self.thresholds = validate_thresholds(self.thresholds)
        feature_spec = self.metadata.get("feature_spec", {})
        declared_set = str(feature_spec.get("feature_set", "base"))
        if declared_set not in FEATURE_SET_DIMS:
            raise ModelBundleError(
                f"Model declares unknown feature_set {declared_set!r}."
            )
        expected_dim = FEATURE_SET_DIMS[declared_set]
        if self.feature_dim != expected_dim:
            raise ModelBundleError(
                f"Model expects {self.feature_dim} features for feature_set "
                f"{declared_set!r}; expected {expected_dim}."
            )
        try:
            estimator_labels = [str(label) for label in self.estimator.classes_]
        except AttributeError as exc:
            raise ModelBundleError("The estimator is not fitted (no classes_).") from exc
        if estimator_labels != self.labels:
            raise ModelBundleError(
                "Label order mismatch between bundle metadata and fitted estimator."
            )
        if not hasattr(self.estimator, "predict_proba"):
            raise ModelBundleError("The estimator does not support predict_proba().")

    @property
    def feature_set(self) -> str:
        return str(self.metadata.get("feature_spec", {}).get("feature_set", "base"))

    def extract_features(
        self, landmarks: np.ndarray | list[list[float]]
    ) -> np.ndarray:
        """Build the feature vector expected by this bundle from raw landmarks."""

        return extract_feature_vector(landmarks, self.feature_set)

    @property
    def dataset_kind(self) -> str:
        return str(self.metadata.get("training", {}).get("dataset_kind", "unknown"))

    @property
    def is_synthetic(self) -> bool:
        return self.dataset_kind in {"synthetic", "mixed"}

    def predict_proba(self, features: np.ndarray | list[float]) -> np.ndarray:
        """Return a validated ``(n_samples, n_classes)`` probability matrix."""

        array = np.asarray(features, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.feature_dim:
            raise ModelBundleError(
                f"Expected shape (n, {self.feature_dim}) features, got {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ModelBundleError("Feature array contains NaN or infinite values.")

        probabilities = np.asarray(self.estimator.predict_proba(array), dtype=np.float64)
        if probabilities.shape != (len(array), len(self.labels)):
            raise ModelBundleError(
                "Estimator returned probabilities with an unexpected shape: "
                f"{probabilities.shape}."
            )
        probabilities = np.clip(probabilities, 1e-9, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(
        self,
        features: np.ndarray | list[float],
        *,
        thresholds: dict[str, Any] | None = None,
    ) -> Prediction:
        """Predict one sample and apply confidence/margin rejection."""

        probabilities = self.predict_proba(features)[0]
        active = validate_thresholds(thresholds or self.thresholds)
        return _prediction_from_probabilities(
            probabilities,
            self.labels,
            min_confidence=active["min_confidence"],
            min_margin=active["min_margin"],
        )

    def public_metadata(self) -> dict[str, Any]:
        """Metadata safe to expose to the UI (no estimator internals)."""

        allowed = {
            "created_utc",
            "format_version",
            "feature_spec",
            "training",
            "validation_metrics",
            "calibration",
            "notes",
        }
        return {key: value for key, value in self.metadata.items() if key in allowed}


def _prediction_from_probabilities(
    probabilities: np.ndarray,
    labels: list[str],
    *,
    min_confidence: float,
    min_margin: float,
) -> Prediction:
    order = np.argsort(probabilities)[::-1]
    top1_index = int(order[0])
    top1 = float(probabilities[top1_index])
    if len(order) > 1:
        top2 = float(probabilities[int(order[1])])
        margin = top1 - top2
    else:
        margin = top1

    accepted = top1 >= min_confidence and margin >= min_margin
    top = tuple(
        TopPrediction(label=labels[int(index)], probability=float(probabilities[int(index)]))
        for index in order[: min(5, len(order))]
    )
    return Prediction(
        accepted_label=labels[top1_index] if accepted else None,
        top_label=labels[top1_index],
        confidence=top1,
        margin=margin,
        accepted=accepted,
        top=top,
    )


def save_model_bundle(
    path: Path | str,
    estimator: Pipeline,
    *,
    thresholds: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Persist a fitted pipeline atomically with its inference contract."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        labels = [str(label) for label in estimator.classes_]
    except AttributeError as exc:
        raise ModelBundleError("Cannot save an unfitted estimator.") from exc

    thresholds = validate_thresholds(thresholds)
    feature_dim = int(getattr(estimator, "n_features_in_", FEATURE_DIM))
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "created_utc": utc_now_iso(),
        "feature_dim": feature_dim,
        "labels": labels,
        "thresholds": thresholds,
        "metadata": metadata,
        "estimator": estimator,
    }

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return path


def load_model_bundle(path: Path | str) -> ModelBundle:
    """Load and validate a trusted joblib model bundle.

    Security note: joblib/pickle files can execute arbitrary code. Only load
    model files produced by this project or another trusted source. A model
    downloaded from the internet should never be loaded without inspection.
    """

    path = Path(path)
    if not path.exists():
        raise ModelBundleError(
            f"Model file not found: {path}\n"
            "Train a model (see README) or launch the synthetic demo with --demo."
        )
    if path.stat().st_size == 0:
        raise ModelBundleError(f"Model file is empty: {path}")

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ModelBundleError(
            f"Could not load model {path}: {exc}. The file may be corrupt or from "
            "an incompatible scikit-learn version."
        ) from exc

    if not isinstance(payload, dict):
        raise ModelBundleError("Model file format is invalid (expected a dictionary).")
    if int(payload.get("format_version", -1)) != MODEL_FORMAT_VERSION:
        raise ModelBundleError(
            f"Unsupported model format {payload.get('format_version')}; "
            f"this build expects {MODEL_FORMAT_VERSION}. Retrain the model."
        )

    required = {"estimator", "labels", "thresholds", "metadata", "feature_dim"}
    missing = required - set(payload)
    if missing:
        raise ModelBundleError(f"Model bundle is missing fields: {sorted(missing)}")

    labels = payload["labels"]
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ModelBundleError("Model bundle has an invalid label list.")
    if not isinstance(payload["metadata"], dict):
        raise ModelBundleError("Model bundle metadata must be a dictionary.")

    try:
        return ModelBundle(
            estimator=payload["estimator"],
            labels=list(labels),
            thresholds=dict(payload["thresholds"]),
            metadata=dict(payload["metadata"]),
            feature_dim=int(payload["feature_dim"]),
        )
    except ModelBundleError:
        raise
    except Exception as exc:
        raise ModelBundleError(f"Model bundle validation failed: {exc}") from exc


def default_thresholds() -> dict[str, Any]:
    """Heuristic fallback used when validation calibration is impossible."""

    return {
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "min_margin": DEFAULT_MIN_MARGIN,
        "source": "default_heuristic",
        "target_selective_accuracy": None,
        "coverage": None,
        "selective_accuracy": None,
        "unknown_accept_rate": None,
        "n_known": 0,
        "n_unknown": 0,
    }
