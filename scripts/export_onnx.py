#!/usr/bin/env python
"""Export a trained model bundle to ONNX for optional low-latency inference.

Usage
-----
    python scripts/export_onnx.py --model models/classifier.joblib

The ONNX file is an optional artefact.  The FastAPI application continues to use
the joblib bundle unless an ONNX runtime path is explicitly integrated; this
script keeps the portable export and its label/threshold sidecar together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import features_for_frame
from src.model_bundle import load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify-data",
        type=Path,
        default=None,
        help="Optional processed CSV used to compare sklearn and ONNX probabilities.",
    )
    parser.add_argument("--verify-rows", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        print(f"ERROR: optional ONNX dependencies are missing: {exc}", file=sys.stderr)
        return 2

    try:
        bundle = load_model_bundle(args.model)
    except Exception as exc:  # noqa: BLE001 - CLI should report a clean error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = args.out or args.model.with_suffix(".onnx")
    initial_type = [("features", FloatTensorType([None, bundle.feature_dim]))]
    classifier = bundle.estimator.named_steps.get("classifier", bundle.estimator)
    options = {id(classifier): {"zipmap": False}}
    try:
        onnx_model = convert_sklearn(
            bundle.estimator,
            initial_types=initial_type,
            target_opset=args.opset,
            options=options,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: ONNX conversion failed: {exc}", file=sys.stderr)
        return 3

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(onnx_model.SerializeToString())
    sidecar = {
        "schema_version": 1,
        "model_file": output.name,
        "source_model": str(Path(args.model).resolve()),
        "feature_set": bundle.feature_set,
        "feature_dim": bundle.feature_dim,
        "labels": bundle.labels,
        "thresholds": bundle.thresholds,
        "opset": args.opset,
        "note": "Optional export. Validate probabilities before deployment.",
    }
    sidecar_path = output.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    verification = None
    if args.verify_data:
        try:
            import pandas as pd
        except ImportError as exc:
            print(f"ERROR: pandas is required for verification: {exc}", file=sys.stderr)
            return 2
        frame = pd.read_csv(args.verify_data, nrows=args.verify_rows)
        X = features_for_frame(frame, bundle.feature_set).astype(np.float32)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        onnx_probabilities = np.asarray(session.run(None, {input_name: X})[1])
        sklearn_probabilities = bundle.predict_proba(X.astype(np.float64))
        verification = {
            "n_rows": int(len(X)),
            "max_abs_difference": float(
                np.max(np.abs(onnx_probabilities - sklearn_probabilities))
            ),
            "mean_abs_difference": float(
                np.mean(np.abs(onnx_probabilities - sklearn_probabilities))
            ),
        }
        sidecar["verification"] = verification
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"ONNX model: {output}")
    print(f"Sidecar:    {sidecar_path}")
    if verification:
        print(
            "Probability check: max |sklearn - onnx| = "
            f"{verification['max_abs_difference']:.6g} over {verification['n_rows']} rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
