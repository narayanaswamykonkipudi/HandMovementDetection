#!/usr/bin/env python
"""Record custom ASL static-fingerspelling landmark samples from a webcam.

Example
-------
    python collect_data.py --label A --subject s01 --session s01a --count 150

Only landmarks are stored, never video frames.  Record several sessions per
letter, vary hand position/angle/lighting/background, and obtain consent before
recording another person.  Use label ``_unknown`` to record non-target gestures
for rejection-threshold calibration.

Controls during capture
-----------------------
Q     abort
SPACE pause / resume
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collector import (
    CollectionError,
    append_samples,
    collection_progress,
    make_sample_row,
)
from src.config import (
    COLLECTION_LABELS,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    HAND_MODEL_PATH,
    RAW_DATA_CSV,
    UNKNOWN_LABEL,
)
from src.runtime import CameraError, open_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Letter A-Z (excluding J/Z) or _unknown.")
    parser.add_argument("--subject", required=True, help="Anonymous signer id, e.g. s01.")
    parser.add_argument(
        "--session",
        default=None,
        help="Session id (default: timestamp). Use a new session for each sitting.",
    )
    parser.add_argument("--count", type=int, default=150, help="Samples to capture.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--interval", type=float, default=0.06, help="Seconds between saved samples.")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--min-hand-score", type=float, default=0.55)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--output", type=Path, default=RAW_DATA_CSV)
    return parser.parse_args()


def _validate_ids(subject: str, session: str) -> None:
    for name, value in (("subject", subject), ("session", session)):
        if not value.strip():
            raise CollectionError(f"{name} cannot be empty.")
        if "::" in value or any(character in value for character in "/\\\n\r\t,"):
            raise CollectionError(
                f"{name} must not contain '::', slashes, commas, tabs, or newlines."
            )


def _hand_well_framed(pixels: np.ndarray, width: int, height: int) -> tuple[bool, str]:
    x_min, y_min = pixels.min(axis=0)
    x_max, y_max = pixels.max(axis=0)
    margin_x = 0.015 * width
    margin_y = 0.015 * height
    if x_min < margin_x or y_min < margin_y or x_max > width - margin_x or y_max > height - margin_y:
        return False, "Hand touches frame edge — move back"
    if (x_max - x_min) < 0.12 * width or (y_max - y_min) < 0.12 * height:
        return False, "Hand too small — move closer"
    return True, ""


def _draw_text(frame, lines: list[str], origin=(16, 32), color=(240, 240, 240)) -> None:
    import cv2

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (origin[0], origin[1] + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (10, 10, 10),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (origin[0], origin[1] + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> int:
    args = parse_args()
    label = args.label.strip()
    if label.lower() == UNKNOWN_LABEL:
        label = UNKNOWN_LABEL
    else:
        label = label.upper()
    if label not in COLLECTION_LABELS:
        print(
            f"ERROR: label must be a static ASL letter "
            f"({''.join(sorted(set(COLLECTION_LABELS) - {UNKNOWN_LABEL}))}) "
            f"or {UNKNOWN_LABEL}; J and Z are dynamic and unsupported.",
            file=sys.stderr,
        )
        return 2
    if args.count < 1 or args.count > 5000:
        print("ERROR: --count must be between 1 and 5000.", file=sys.stderr)
        return 2

    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        _validate_ids(args.subject, session)
    except CollectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Data collection reminder:\n"
        "  * Store landmarks only, not video.\n"
        "  * Obtain permission before recording another person; use an anonymous id.\n"
        "  * Record multiple sessions per letter and vary position, angle, lighting, background.\n"
        f"  * Target: {label} | subject: {args.subject} | session: {session}\n"
    )

    progress = collection_progress(args.output)
    current = progress.get(label, {"samples": 0, "sessions": set(), "subjects": set()})
    current_samples = int(current["samples"])
    current_sessions = len(current["sessions"]) if isinstance(current["sessions"], set) else 0
    print(
        f"Existing progress for {label}: {current_samples} samples across "
        f"{current_sessions} session(s). This session will add up to {args.count}."
    )

    import cv2

    if not HAND_MODEL_PATH.exists():
        print(
            f"ERROR: hand model missing at {HAND_MODEL_PATH}.\n"
            "Run: python scripts/download_models.py",
            file=sys.stderr,
        )
        return 2

    from src.hand_tracking import HandTracker, HandTrackerError

    try:
        tracker = HandTracker()
        tracker.open()
    except HandTrackerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        capture, backend = open_camera(args.camera, args.width, args.height)
    except CameraError as exc:
        tracker.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    window = "Collect Sign Samples — Q quit, SPACE pause"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"Camera opened with backend={backend}. Starting countdown…")

    saved = 0
    skipped = 0
    quality_counts: dict[str, int] = {
        "no_hand": 0,
        "low_score": 0,
        "out_of_frame": 0,
    }
    paused = False
    last_saved = 0.0
    start_countdown = time.monotonic() + max(0.0, args.countdown)
    final_message = ""

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("ERROR: camera stopped returning frames.", file=sys.stderr)
                return 3
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            timestamp_ms = int(time.monotonic() * 1000)
            try:
                detections = tracker.detect(frame, timestamp_ms=timestamp_ms)
            except HandTrackerError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 3
            primary = detections[0] if detections else None
            quality_message = ""
            framed = False
            if primary is not None:
                framed, quality_message = _hand_well_framed(primary.landmarks_px, width, height)

            elapsed = start_countdown - time.monotonic()
            counting_down = elapsed > 0

            if (
                not counting_down
                and not paused
                and saved < args.count
                and time.monotonic() - last_saved >= args.interval
            ):
                if primary is None:
                    quality_counts["no_hand"] += 1
                    skipped += 1
                elif primary.score < args.min_hand_score:
                    quality_counts["low_score"] += 1
                    skipped += 1
                elif not framed:
                    quality_counts["out_of_frame"] += 1
                    skipped += 1
                else:
                    row = make_sample_row(
                        primary.landmarks,
                        label=label,
                        subject_id=args.subject,
                        session_id=session,
                        source="webcam",
                        handedness=primary.handedness,
                        hand_score=primary.score,
                    )
                    append_samples(args.output, [row])
                    saved += 1
                    last_saved = time.monotonic()

            canvas = tracker.draw(frame, detections, label=None)
            status = "PAUSED" if paused else ("COUNTDOWN" if counting_down else "CAPTURING")
            lines = [
                f"{status} | {label} | this session {saved}/{args.count}",
                f"dataset before: {current_samples} samples / {current_sessions} sessions",
                f"subject={args.subject} session={session}",
                f"hand score={primary.score:.2f}" if primary else "No hand detected",
            ]
            if counting_down:
                lines.append(f"Starting in {elapsed:.1f}s — hold the sign")
            if quality_message:
                lines.append(quality_message)
            if primary is None:
                lines.append("Raise one hand, palm/back clearly visible")
            if saved >= args.count:
                lines.append("Target reached — press Q to finish")
            elif paused:
                lines.append("SPACE resumes")
            else:
                lines.append("Q quits | SPACE pauses")
            _draw_text(canvas, lines)
            cv2.rectangle(
                canvas,
                (0, height - 12),
                (int(width * saved / args.count), height),
                (90, 230, 130),
                -1,
            )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                final_message = "Aborted by user."
                break
            if key == ord(" "):
                paused = not paused
                if not paused:
                    start_countdown = time.monotonic()
            if saved >= args.count:
                final_message = "Capture complete."

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                final_message = "Window closed."
                break
            if saved >= args.count:
                # Show the completed state for a moment before exiting.
                for _ in range(30):
                    cv2.waitKey(20)
                break
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()

    print(f"\n{final_message} Saved {saved} sample(s) to {args.output}")
    if saved < args.count:
        print("Note: fewer than requested; weak/edge/no-hand frames were skipped.")
    print(
        "Quality skips (frames): "
        f"no hand={quality_counts['no_hand']}, "
        f"low score={quality_counts['low_score']}, "
        f"out of frame/too small={quality_counts['out_of_frame']} "
        f"(total {skipped})"
    )
    final_progress = collection_progress(args.output)
    print("\nPer-letter progress (sample count / sessions):")
    for key in sorted(COLLECTION_LABELS):
        entry = final_progress.get(key)
        samples = int(entry["samples"]) if entry else 0
        sessions = len(entry["sessions"]) if entry and isinstance(entry["sessions"], set) else 0
        marker = "*" if key == label else " "
        print(f" {marker} {key:>9}: {samples:>6} / {sessions} session(s)")
    print(
        "\nCollect more sessions for this label, then run preprocess_data.py. "
        "Do not use the same session for training and testing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
