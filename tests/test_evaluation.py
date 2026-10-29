from __future__ import annotations

import json

import numpy as np

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


def test_classification_metrics_and_confusions():
    labels = ["A", "B", "C"]
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "B", "B", "B", "C", "A"]
    metrics = classification_metrics(y_true, y_pred, labels)
    assert metrics["n_samples"] == 6
    assert metrics["accuracy"] == 0.6666666666666666
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["per_class"]["A"]["support"] == 2
    matrix = np.asarray(metrics["confusion_matrix"])
    assert matrix.shape == (3, 3)
    assert int(matrix.sum()) == 6
    pairs = confusion_pairs(y_true, y_pred, limit=5)
    assert pairs[0]["count"] == 2
    assert {"true", "predicted", "count"} <= set(pairs[0])


def test_calibration_diagnostics_bins_and_known_mask():
    labels = ["A", "B"]
    y_true = np.asarray(["A", "A", "B", "B", "_unknown"])
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.5, 0.5],
        ]
    )
    diagnostics = expected_calibration_error(y_true, probabilities, labels)
    assert diagnostics["n_samples"] == 4
    assert 0.0 <= diagnostics["ece"] <= 1.0
    assert 0.0 <= diagnostics["brier"] <= 1.0
    assert sum(item["n"] for item in diagnostics["bins"]) == 4
    assert known_mask(y_true).tolist() == [True, True, True, True, False]


def test_grouped_bootstrap_ci_is_reproducible_and_group_aware():
    labels = ["A", "B"]
    y_true = np.asarray(["A", "A", "B", "B", "A", "B"])
    y_pred = np.asarray(["A", "A", "B", "A", "A", "B"])
    groups = np.asarray(["g1", "g1", "g2", "g2", "g3", "g3"])
    first = grouped_bootstrap_ci(
        y_true, y_pred, groups, labels, n_boot=100, seed=7
    )
    second = grouped_bootstrap_ci(
        y_true, y_pred, groups, labels, n_boot=100, seed=7
    )
    assert first == second
    assert first["point"] == 0.8333333333333334
    assert first["lower"] <= first["point"] <= first["upper"]
    single = grouped_bootstrap_ci(
        y_true, y_pred, np.asarray(["g"] * 6), labels, n_boot=10
    )
    assert single["lower"] is None
    assert "two groups" in single["note"]


def test_per_subject_breakdown_uses_known_and_accepted_samples():
    y_true = np.asarray(["A", "A", "_unknown", "B", "B"])
    y_pred = np.asarray(["A", "B", "A", "B", "A"])
    subjects = np.asarray(["s1", "s1", "s1", "s2", "s2"])
    accepted = np.asarray([True, False, True, True, True])
    result = per_subject_breakdown(y_true, y_pred, subjects, accepted, ["A", "B"])
    assert result["s1"]["n_known"] == 2
    assert result["s1"]["forced_accuracy"] == 0.5
    assert result["s1"]["coverage"] == 1.0
    assert result["s2"]["forced_accuracy"] == 0.5
    assert result["s2"]["selective_accuracy"] == 0.5


def test_plots_and_json_writing(tmp_path):
    confusion = np.asarray([[2, 1], [0, 3]])
    confusion_path = plot_confusion_matrix(
        confusion, ["A", "B"], tmp_path / "confusion.png", title="Test", note="Note"
    )
    assert confusion_path.exists()
    reliability_path = plot_reliability(
        {
            "ece": 0.1,
            "brier": 0.2,
            "bins": [
                {"mean_confidence": 0.2, "accuracy": 0.1, "n": 5},
                {"mean_confidence": 0.8, "accuracy": 0.9, "n": 10},
            ],
        },
        tmp_path / "reliability.png",
        title="Reliability",
        note="Synthetic",
    )
    assert reliability_path.exists()
    payload = tmp_path / "metrics.json"
    write_json(payload, {"value": np.float64(0.5), "count": np.int64(3)})
    loaded = json.loads(payload.read_text(encoding="utf-8"))
    assert loaded["value"] == 0.5
    assert loaded["count"] == 3


def test_synthetic_warning_is_explicit_and_absent_for_real_data():
    synthetic = synthetic_warning("synthetic")
    assert synthetic is not None
    assert "NOT sign-language recognition accuracy" in synthetic
    assert synthetic_warning("recorded") is None
