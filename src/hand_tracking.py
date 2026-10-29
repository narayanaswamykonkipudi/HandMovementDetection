"""MediaPipe hand landmark detection plus OpenCV visualisation.

Uses the MediaPipe Tasks ``HandLandmarker`` API (mediapipe >= 1.0), whose legacy
``mp.solutions`` namespace no longer exists.  The model asset is downloaded by
``scripts/download_models.py``; MediaPipe never downloads anything implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import HAND_MODEL_PATH, LANDMARK_COUNT


class HandTrackerError(RuntimeError):
    """Raised when the hand landmark model cannot be initialised."""


@dataclass
class HandDetection:
    """One detected hand in one frame."""

    landmarks: np.ndarray            # (21, 3) normalised image coordinates
    landmarks_px: np.ndarray         # (21, 2) pixel coordinates for drawing
    handedness: str                  # "Left" | "Right" | "Unknown"
    handedness_score: float
    score: float                     # presence score used for filtering/sorting

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.landmarks.shape[0]), int(self.landmarks.shape[1]))


def _require_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - lazy import keeps tests fast
    except ImportError as exc:  # pragma: no cover - environment specific
        raise HandTrackerError(
            "OpenCV is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return cv2


class HandTracker:
    """Thin, testable wrapper around MediaPipe's HandLandmarker."""

    def __init__(
        self,
        model_path: Path | str = HAND_MODEL_PATH,
        *,
        running_mode: str = "video",
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.running_mode = running_mode
        self.num_hands = int(num_hands)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self._landmarker: Any | None = None
        self._connection_pairs: list[tuple[int, int]] = []
        self._closed = False

    # ------------------------------------------------------------------ setup
    def __enter__(self) -> "HandTracker":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def open(self) -> None:
        if self._landmarker is not None:
            return
        if not self.model_path.exists():
            raise HandTrackerError(
                f"Hand landmark model not found: {self.model_path}\n"
                "Run:  python scripts/download_models.py"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise HandTrackerError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        modes = {
            "image": vision.RunningMode.IMAGE,
            "video": vision.RunningMode.VIDEO,
        }
        if self.running_mode not in modes:
            raise HandTrackerError("running_mode must be 'image' or 'video'.")

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=modes[self.running_mode],
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise HandTrackerError(
                f"Could not initialise MediaPipe HandLandmarker: {exc}"
            ) from exc

        self._connection_pairs = [
            (int(connection.start), int(connection.end))
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]
        self._mp = mp

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._landmarker = None
        self._closed = True

    # ------------------------------------------------------------------ inference
    def to_mediapipe_image(self, bgr_frame: np.ndarray) -> Any:
        cs = _require_cv2()
        rgb = cs.cvtColor(bgr_frame, cs.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def detect(
        self,
        bgr_frame: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> list[HandDetection]:
        """Detect hands in a BGR frame.

        ``timestamp_ms`` must increase monotonically in video mode.
        """

        self.open()
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            raise HandTrackerError("Received an empty frame.")

        image = self.to_mediapipe_image(bgr_frame)
        try:
            if self.running_mode == "video":
                if timestamp_ms is None:
                    raise HandTrackerError("timestamp_ms is required in video mode.")
                result = self._landmarker.detect_for_video(image, int(timestamp_ms))
            else:
                result = self._landmarker.detect(image)
        except HandTrackerError:
            raise
        except Exception as exc:
            raise HandTrackerError(f"Hand landmark inference failed: {exc}") from exc

        height, width = bgr_frame.shape[:2]
        detections: list[HandDetection] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            if len(hand_landmarks) != LANDMARK_COUNT:
                continue
            points = np.array(
                [(point.x, point.y, point.z) for point in hand_landmarks],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(points)):
                continue
            pixels = np.stack(
                [points[:, 0] * width, points[:, 1] * height], axis=1
            ).astype(np.int32)

            handedness = "Unknown"
            handedness_score = 0.0
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = str(category.category_name or "Unknown")
                handedness_score = float(category.score or 0.0)

            presence_scores: list[float] = []
            for point in hand_landmarks:
                score = getattr(point, "presence", None)
                if score is not None:
                    presence_scores.append(float(score))
            score = float(np.mean(presence_scores)) if presence_scores else handedness_score

            detections.append(
                HandDetection(
                    landmarks=points,
                    landmarks_px=pixels,
                    handedness=handedness,
                    handedness_score=handedness_score,
                    score=score,
                )
            )

        detections.sort(key=lambda detection: detection.score, reverse=True)
        return detections

    # ------------------------------------------------------------------ drawing
    def draw(
        self,
        bgr_frame: np.ndarray,
        detections: list[HandDetection],
        *,
        label: str | None = None,
        highlight_color: tuple[int, int, int] = (80, 220, 120),
    ) -> np.ndarray:
        """Draw landmark skeletons and an optional label onto a copy of the frame."""

        cs = _require_cv2()
        canvas = bgr_frame.copy()
        if not detections:
            cd = _require_cv2()
            cd.putText(
                canvas,
                "No hand detected",
                (16, 34),
                cd.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 60, 220),
                2,
                cd.LINE_AA,
            )
            return canvas

        for detection in detections:
            pixels = detection.landmarks_px
            for start, end in self._connection_pairs:
                cs.line(
                    canvas,
                    tuple(int(v) for v in pixels[start]),
                    tuple(int(v) for v in pixels[end]),
                    highlight_color,
                    2,
                    cs.LINE_AA,
                )
            for point in pixels:
                cs.circle(
                    canvas,
                    (int(point[0]), int(point[1])),
                    4,
                    (255, 255, 255),
                    -1,
                    cs.LINE_AA,
                )
                cs.circle(
                    canvas,
                    (int(point[0]), int(point[1])),
                    4,
                    (30, 30, 30),
                    1,
                    cs.LINE_AA,
                )

            x_min, y_min = pixels.min(axis=0)
            x_max, y_max = pixels.max(axis=0)
            cs.rectangle(
                canvas,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                highlight_color,
                1,
                cs.LINE_AA,
            )
            caption = f"{detection.handedness} hand ({detection.score:.2f})"
            cs.putText(
                canvas,
                caption,
                (int(x_min), max(20, int(y_min) - 10)),
                cs.FONT_HERSHEY_SIMPLEX,
                0.6,
                highlight_color,
                2,
                cs.LINE_AA,
            )

        if label:
            cs.rectangle(canvas, (10, 10), (360, 76), (20, 20, 20), -1)
            cs.putText(
                canvas,
                f"Sign: {label}",
                (24, 52),
                cs.FONT_HERSHEY_SIMPLEX,
                1.1,
                (120, 255, 170),
                2,
                cs.LINE_AA,
            )
        return canvas
