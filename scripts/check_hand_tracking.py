#!/usr/bin/env python
"""Phase 1 check: MediaPipe hand landmark detection on an image or webcam.

Examples
--------
    python scripts/check_hand_tracking.py --camera 0
    python scripts/check_hand_tracking.py --image path/to/hand.jpg --output artifacts/hand.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CAMERA_INDEX, HAND_MODEL_PATH
from src.hand_tracking import HandTracker, HandTrackerError
from src.runtime import CameraError, open_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--output", type=Path, default=None, help="Save annotated image.")
    parser.add_argument("--num-hands", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not HAND_MODEL_PATH.exists():
        print(
            f"ERROR: model missing at {HAND_MODEL_PATH}. "
            "Run: python scripts/download_models.py",
            file=sys.stderr,
        )
        return 2

    import cv2

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"ERROR: could not read image: {args.image}", file=sys.stderr)
            return 2
        try:
            with HandTracker(running_mode="image", num_hands=args.num_hands) as tracker:
                start = time.perf_counter()
                detections = tracker.detect(frame)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                annotated = tracker.draw(frame, detections)
        except (HandTrackerError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3

        print(
            f"Image: {args.image} ({frame.shape[1]}x{frame.shape[0]})\n"
            f"Hands detected: {len(detections)} | inference: {elapsed_ms:.1f} ms"
        )
        for index, detection in enumerate(detections, start=1):
            x_min, y_min = detection.landmarks_px.min(axis=0)
            x_max, y_max = detection.landmarks_px.max(axis=0)
            print(
                f"  hand {index}: {detection.handedness} score={detection.score:.3f} "
                f"landmarks={detection.landmarks.shape} "
                f"bbox=({x_min},{y_min})-({x_max},{y_max})"
            )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), annotated)
            print(f"Annotated image: {args.output}")
        if not detections:
            print("No hand detected. This is a valid state but not a successful Phase 1 check.")
            return 4
        print("Phase 1 landmark extraction: OK")
        return 0

    try:
        capture, backend = open_camera(args.camera, 1280, 720)
    except CameraError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    window = "Phase 1 — Hand Landmark Check (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    tracker = HandTracker(num_hands=args.num_hands)
    print(f"Camera backend: {backend}. Press Q in the preview window to stop.")
    frames = 0
    with_detection = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("ERROR: camera stopped returning frames.", file=sys.stderr)
                return 3
            frame = cv2.flip(frame, 1)
            frames += 1
            try:
                detections = tracker.detect(frame, timestamp_ms=frames * 33)
            except HandTrackerError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 3
            if detections:
                with_detection += 1
            annotated = tracker.draw(frame, detections)
            cv2.putText(
                annotated,
                f"frames={frames} detected={with_detection}",
                (16, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - start, 1e-9)
    print(
        f"Frames: {frames} | frames with a hand: {with_detection} | "
        f"average FPS: {frames / elapsed:.1f}"
    )
    if frames and with_detection:
        print("Phase 1 live tracking: OK (visual check of landmarks required).")
        return 0
    print("No hand was detected. Check lighting, camera framing, and the model path.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
