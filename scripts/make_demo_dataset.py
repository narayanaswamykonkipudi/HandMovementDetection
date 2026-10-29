#!/usr/bin/env python
"""Create the explicitly SYNTHETIC dataset used for pipeline smoke tests.

    python scripts/make_demo_dataset.py --output data/demo/landmarks.csv --overwrite

These rows are generated hand-like skeletons.  They are not ASL samples and the
resulting model must never be presented as a sign-recognition result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEMO_DATA_DIR
from src.demo_data import write_demo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEMO_DATA_DIR / "landmarks.csv"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--sessions-per-subject", type=int, default=2)
    parser.add_argument("--frames-per-session", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = tuple(f"demo-s{index:02d}" for index in range(1, args.subjects + 1))
    try:
        path = write_demo_dataset(
            args.output,
            seed=args.seed,
            overwrite=args.overwrite,
            subjects=subjects,
            sessions_per_subject=args.sessions_per_subject,
            frames_per_session=args.frames_per_session,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    import pandas as pd

    frame = pd.read_csv(path, usecols=["label", "subject_id", "session_id"])
    print(
        f"Wrote {len(frame)} SYNTHETIC rows to {path}\n"
        f"  subjects: {frame['subject_id'].nunique()} | sessions: "
        f"{frame.groupby(['subject_id', 'session_id']).ngroups} | classes: "
        f"{frame['label'].nunique()}\n"
        "\nWARNING: this is not real sign-language data. Metrics from it are plumbing checks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
