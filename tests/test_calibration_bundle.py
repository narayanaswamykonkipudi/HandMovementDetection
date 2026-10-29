from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.calibration import calibrate_thresholds, check_thresholds_on_split
from src.features import build_feature_matrix
from src.model_bundle import (
    ModelBundle,
    ModelBundleError,
    build_estimator,
    load_model_bundle,
    save_model_bundle,
    validate_thresholds,
)
from tests.conftest import make_landmarks


def _training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels = ["A", "B", "C"]
    X = []
    y = []
    for class_index, label in enumerate(labels):
        for offset in range(12):
            landmarks = make_landmarks(class_index * 100 + offset)
            landmarks[:, 0] += class_index * 0.25
            X.append(build_feature_matrix([landmarks])[0])
            y.append(label)
    return np.asarray(X), np.asarray(y), np.asarray(labels), labels


def test_calibrate_thresholds_prefers_feasible_region():
    labels = ["A", "B"]
    probabilities = np.asarray(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.04, 0.96],
            [0.10, 0.90],
        ]
    )
    y = np.asarray(["A", "A", "B", "B"])
    unknown = np.asarray([[0.55, 0.45], [0.34, 0.66]])
    result = calibrate_thresholds(
        y,
        probabilities,
        labels,
        probabilities_unknown=unknown,
        target_selective_accuracy=0.9,
        max_unknown_accept_rate=0.5,
        min_coverage=0.5,
    )
    assert result["source"] == "validation"
    assert result["coverage"] >= 0.5
    assert result["selective_accuracy"] >= 0.9
    assert result["unknown_accept_rate"] <= 0.5
    assert result["status"] == "calibrated_on_validation"
    assert validate_thresholds(result)["min_confidence"] == result["min_confidence"]


def test_calibrate_thresholds_handles_no_acceptable_samples():
    result = calibrate_thresholds(
        ["A", "B"],
        np.asarray([[0.4, 0.6], [0.6, 0.4]]),
        ["A", "B"],
        target_selective_accuracy=0.99,
        min_coverage=0.99,
    )
    assert result["source"] in {"default_heuristic", "validation"}
    assert result["coverage"] <= 1.0
    assert result["status"] in {
        "calibration_failed_no_acceptances",
        "constraints_not_met",
    }


def test_check_thresholds_reports_known_and_unknown():
    labels = ["A", "B"]
    probabilities = np.asarray([[0.9, 0.1], [0.45, 0.55], [0.8, 0.2]])
    y = np.asarray(["A", "A", "_unknown"])
    report = check_thresholds_on_split(
        y,
        probabilities,
        labels,
        {"min_confidence": 0.5, "min_margin": 0.2},
    )
    assert report["known"]["n_known"] == 2
    assert report["known"]["coverage"] == 0.5
    assert report["known"]["selective_accuracy"] == 1.0
    assert report["unknown"]["n_unknown"] == 1
    assert report["unknown"]["accept_rate"] == 1.0


def test_model_bundle_roundtrip_and_prediction(tmp_path):
    X, y, _, labels = _training_data()
    estimator = build_estimator("svm", seed=2)
    estimator.fit(X, y)
    thresholds = calibrate_thresholds(
        y,
        estimator.predict_proba(X),
        labels,
        target_selective_accuracy=0.8,
        min_coverage=0.2,
    )
    path = tmp_path / "model.joblib"
    save_model_bundle(
        path,
        estimator,
        thresholds=thresholds,
        metadata={
            "feature_spec": {"feature_set": "base", "feature_dim": 63},
            "training": {"dataset_kind": "synthetic"},
        },
    )
    bundle = load_model_bundle(path)
    assert bundle.feature_dim == 63
    assert bundle.feature_set == "base"
    assert bundle.labels == labels
    assert bundle.is_synthetic is True
    assert bundle.dataset_kind == "synthetic"
    prediction = bundle.predict(X[0])
    assert prediction.top_label in labels
    assert 0.0 <= prediction.confidence <= 1.0
    assert bundle.public_metadata()["training"]["dataset_kind"] == "synthetic"


def test_model_bundle_accepts_geometry_feature_dimension(tmp_path):
    X, y, _, labels = _training_data()
    geometry = np.hstack([X, np.ones((len(X), 25))])
    estimator = build_estimator("rf", seed=1)
    estimator.fit(geometry, y)
    path = tmp_path / "geometry.joblib"
    save_model_bundle(
        path,
        estimator,
        thresholds={"min_confidence": 0.1, "min_margin": 0.0},
        metadata={"feature_spec": {"feature_set": "geometry", "feature_dim": 88}},
    )
    bundle = load_model_bundle(path)
    assert bundle.feature_dim == 88
    assert bundle.extract_features(make_landmarks(1)).shape == (88,)


def test_model_bundle_rejects_missing_empty_and_bad_files(tmp_path):
    with pytest.raises(ModelBundleError, match="not found"):
        load_model_bundle(tmp_path / "missing.joblib")
    empty = tmp_path / "empty.joblib"
    empty.write_bytes(b"")
    with pytest.raises(ModelBundleError, match="empty"):
        load_model_bundle(empty)
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib")
    with pytest.raises(ModelBundleError, match="Could not load"):
        load_model_bundle(bad)
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"format_version": 999}, wrong)
    with pytest.raises(ModelBundleError, match="Unsupported model format"):
        load_model_bundle(wrong)


def test_model_bundle_rejects_label_and_dimension_mismatch(tmp_path):
    X, y, _, labels = _training_data()
    estimator = build_estimator("svm", seed=1).fit(X, y)
    path = tmp_path / "mismatch.joblib"
    save_model_bundle(
        path,
        estimator,
        thresholds={"min_confidence": 0.1, "min_margin": 0.0},
        metadata={"feature_spec": {"feature_set": "base", "feature_dim": 63}},
    )
    payload = joblib.load(path)
    payload["labels"] = ["A", "B"]
    joblib.dump(payload, path)
    with pytest.raises(ModelBundleError, match="Label order mismatch"):
        load_model_bundle(path)


def test_validate_thresholds_requires_bounds():
    with pytest.raises(ModelBundleError):
        validate_thresholds({"min_confidence": 1.5, "min_margin": 0.1})
    with pytest.raises(ModelBundleError):
        validate_thresholds({"min_confidence": 0.5, "min_margin": -0.1})
