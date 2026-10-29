#!/usr/bin/env python
"""Measure inference latency and precomputed-landmark throughput.

Usage
-----
    python scripts/benchmark_latency.py --model models/classifier.joblib \
        --data data/processed/test.csv

These timings are machine-specific.  ``pipeline_fps`` starts from landmark rows
and therefore excludes camera capture, JPEG encoding, UI rendering, and
MediaPipe hand tracking.  It is not a camera frame rate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR, RAW_COORD_COLUMNS
from src.experiment_log import append_experiment, dataset_manifest_hash
from src.experiments import features_for_frame
from src.features import FEATURE_SET_DIMS, build_feature_matrix, feature_table_to_array
from src.model_bundle import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "latency" / "latency.json")
    parser.add_argument(
        "--experiment-log", type=Path, default=ARTIFACTS_DIR / "experiments.jsonl"
    )
    parser.add_argument("--no-experiment-log", action="store_true")
    return parser.parse_args()


def _percentiles(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    try:
        bundle = load_model_bundle(args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    frame = pd.read_csv(args.data, nrows=args.rows)
    if frame.empty:
        print("ERROR: benchmark data is empty.", file=sys.stderr)
        return 3
    X = features_for_frame(frame, bundle.feature_set)
    rows = [X[index : index + 1] for index in range(len(X))]

    for row in rows[: args.warmup]:
        bundle.predict_proba(row)
    timings: list[float] = []
    for row in rows:
        started = time.perf_counter()
        bundle.predict_proba(row)
        timings.append((time.perf_counter() - started) * 1000.0)
    sklearn_stats = _percentiles(timings)

    raw_matrix = feature_table_to_array(frame, RAW_COORD_COLUMNS)
    for index in range(min(args.warmup, len(raw_matrix))):
        vector = build_feature_matrix(
            [raw_matrix[index]], feature_set=bundle.feature_set
        )
        bundle.predict_proba(vector)
    pipeline_started = time.perf_counter()
    for raw_row in raw_matrix:
        vector = build_feature_matrix([raw_row], feature_set=bundle.feature_set)
        bundle.predict_proba(vector)
    pipeline_seconds = time.perf_counter() - pipeline_started
    pipeline_fps = float(len(raw_matrix) / pipeline_seconds) if pipeline_seconds else None

    result = {
        "schema_version": 1,
        "model_file": str(args.model.resolve()),
        "feature_set": bundle.feature_set,
        "feature_dim": bundle.feature_dim,
        "n_rows": int(len(X)),
        "sklearn_batch1": sklearn_stats,
        "sklearn_batch1_fps": float(1000.0 / sklearn_stats["p50_ms"])
        if sklearn_stats["p50_ms"]
        else None,
        "pipeline_precomputed_landmarks": {
            "total_seconds": float(pipeline_seconds),
            "fps": pipeline_fps,
            "includes": "feature construction + single-row prediction",
            "excludes": [
                "camera capture",
                "MediaPipe tracking",
                "JPEG encoding",
                "HTTP/browser rendering",
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "notes": [
            "Timings are machine-specific and not a recognition-accuracy claim.",
            "P95 latency is a better deployment planning number than the mean.",
            "pipeline_fps is not camera FPS.",
        ],
    }

    if args.onnx:
        try:
            import onnxruntime as ort

            sidecar_path = args.onnx.with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            session = ort.InferenceSession(
                str(args.onnx), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            for row in rows[: args.warmup]:
                session.run(None, {input_name: row.astype(np.float32)})
            onnx_timings: list[float] = []
            for row in rows:
                started = time.perf_counter()
                session.run(None, {input_name: row.astype(np.float32)})
                onnx_timings.append((time.perf_counter() - started) * 1000.0)
            onnx_stats = _percentiles(onnx_timings)
            onnx_probabilities = np.asarray(
                session.run(None, {input_name: X.astype(np.float32)})[1]
            )
            sklearn_probabilities = bundle.predict_proba(X)
            result["onnx"] = {
                "model_file": str(args.onnx.resolve()),
                "labels": sidecar.get("labels"),
                **onnx_stats,
                "batch1_fps": float(1000.0 / onnx_stats["p50_ms"])
                if onnx_stats["p50_ms"]
                else None,
                "max_abs_probability_difference": float(
                    np.max(np.abs(onnx_probabilities - sklearn_probabilities))
                ),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: ONNX benchmark skipped: {exc}", file=sys.stderr)
            result["onnx"] = {"error": str(exc)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if not args.no_experiment_log:
        record = append_experiment(
            args.experiment_log,
            kind="latency_benchmark",
            params={
                "model": str(args.model),
                "rows": len(X),
                "warmup": args.warmup,
                "onnx": str(args.onnx) if args.onnx else None,
            },
            metrics={
                "sklearn_p50_ms": sklearn_stats["p50_ms"],
                "sklearn_p95_ms": sklearn_stats["p95_ms"],
                "sklearn_batch1_fps": result["sklearn_batch1_fps"],
                "pipeline_fps": pipeline_fps,
                "onnx_p50_ms": result.get("onnx", {}).get("p50_ms"),
                "onnx_p95_ms": result.get("onnx", {}).get("p95_ms"),
            },
            dataset_manifest=dataset_manifest_hash(args.data.parent),
            extra={"output": str(args.output.resolve())},
        )
        print(f"Experiment log: {args.experiment_log} ({record['run_id']})")

    print(f"Sklearn p50/p95: {sklearn_stats['p50_ms']:.3f} / {sklearn_stats['p95_ms']:.3f} ms")
    print(f"Precomputed-landmark pipeline FPS: {pipeline_fps:.1f}")
    if result.get("onnx", {}).get("p50_ms"):
        print(
            f"ONNX p50/p95: {result['onnx']['p50_ms']:.3f} / "
            f"{result['onnx']['p95_ms']:.3f} ms"
        )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
