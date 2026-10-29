#!/usr/bin/env python
"""Build the full SYNTHETIC end-to-end demo in one command.

    python scripts/build_demo.py

This proves the data -> split -> train -> evaluate pipeline without a webcam or
third-party dataset.  The resulting model is trained on generated hand-like
landmarks, is saved as ``models/demo_classifier.joblib``, and is clearly marked
synthetic in both the model card and the UI.  Its metrics are NOT sign-language
recognition results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEMO_CLASSIFIER_PATH, DEMO_DATA_DIR, MODELS_DIR


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames-per-session", type=int, default=20)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    data_csv = DEMO_DATA_DIR / "landmarks.csv"
    processed = DEMO_DATA_DIR / "processed"
    model = DEMO_CLASSIFIER_PATH

    run(
        [
            python,
            "scripts/make_demo_dataset.py",
            "--output",
            str(data_csv),
            "--seed",
            str(args.seed),
            "--frames-per-session",
            str(args.frames_per_session),
            "--overwrite",
        ]
    )
    run(
        [
            python,
            "preprocess_data.py",
            "--input",
            str(data_csv),
            "--out",
            str(processed),
            "--seed",
            "42",
            "--test-subjects",
            "demo-s02",
        ]
    )
    train_command = [
        python,
        "train_model.py",
        "--processed",
        str(processed),
        "--model",
        "svm",
        "--out",
        str(model),
        "--card",
        str(MODELS_DIR / "demo_model_card.json"),
        "--seed",
        "42",
        "--augment-copies",
        str(args.augment_copies),
    ]
    if args.no_tune:
        train_command.append("--no-tune")
    run(train_command)

    if not args.skip_eval:
        run(
            [
                python,
                "evaluate_model.py",
                "--processed",
                str(processed),
                "--model",
                str(model),
                "--split",
                "test",
                "--output",
                str(PROJECT_ROOT / "artifacts" / "demo_eval"),
            ]
        )

    print(
        "\n" + "=" * 78 + "\n"
        "DEMO BUILD COMPLETE.\n"
        "Launch the UI with:\n"
        f"  python app.py --demo\n"
        "The UI will display a prominent synthetic-model warning. Predictions from this\n"
        "model are not real ASL recognition. For real use, collect data and train a model:\n"
        "  python collect_data.py --label A --subject s01 --session s01a --count 150\n"
        "  python preprocess_data.py\n"
        "  python train_model.py --processed data/processed --out models/classifier.joblib\n"
        "  python evaluate_model.py --processed data/processed --model models/classifier.joblib\n"
        "=" * 78
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: demo step failed with exit code {exc.returncode}.", file=sys.stderr)
        raise SystemExit(exc.returncode)
