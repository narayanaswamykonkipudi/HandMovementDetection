"""Application state shared by the camera thread and the web UI.

Everything that can be touched from more than one thread lives behind a single
lock.  The camera thread does the expensive work (capture, landmarks, model
inference) and only briefly holds the lock while publishing results.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEFAULT_RELEASE_FRAMES,
    DEFAULT_SMOOTHING_MIN_RATIO,
    DEFAULT_SMOOTHING_MIN_VOTES,
    DEFAULT_SMOOTHING_WINDOW,
    MAX_TEXT_LENGTH,
    UNKNOWN_LABEL,
)
from src.features import FeatureError
from src.hand_tracking import HandDetection, HandTracker, HandTrackerError
from src.model_bundle import ModelBundle, ModelBundleError, Prediction, load_model_bundle
from src.smoothing import RecognitionSession
from src.speech import SpeechService
from src.text import TextBuffer, TextError


class CameraError(RuntimeError):
    """Raised when a webcam cannot be opened or read."""


def open_camera(index: int, width: int, height: int, fps: int = 30):
    """Open a webcam, trying sensible backends and returning ``(capture, backend)``.

    On Windows, DirectShow is often more reliable for USB webcams than the
    default Media Foundation path, but the default works for many integrated
    cameras.  We try the platform default first and fall back.
    """

    import cv2

    attempts: list[tuple[str, int]] = []
    if hasattr(cv2, "CAP_DSHOW"):
        attempts.append(("dshow", cv2.CAP_DSHOW))
    attempts.append(("any", cv2.CAP_ANY))

    last_error: str = "unknown error"
    for name, backend in attempts:
        capture = cv2.VideoCapture(index, backend)
        if capture is None or not capture.isOpened():
            last_error = f"backend {name} could not open device {index}"
            if capture is not None:
                capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        capture.set(cv2.CAP_PROP_FPS, float(fps))
        ok, _ = capture.read()
        if not ok:
            last_error = f"backend {name} opened device {index} but returned no frames"
            capture.release()
            continue
        return capture, name
    raise CameraError(
        f"Could not open camera {index} ({last_error}). Close other apps that use "
        "the camera, try a different --camera index, or run with --no-camera."
    )


def _resize_letterbox(frame: np.ndarray, width: int = 960) -> np.ndarray:
    import cv2

    height, original_width = frame.shape[:2]
    if original_width <= width:
        return frame
    scale = width / original_width
    return cv2.resize(frame, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class RuntimeStatus:
    camera_running: bool = False
    camera_error: str | None = None
    camera_backend: str | None = None
    camera_index: int = 0
    camera_fps: float = 0.0
    frames_processed: int = 0
    model_error: str | None = None
    model_path: str | None = None
    model_loaded: bool = False
    model_metadata: dict[str, Any] = field(default_factory=dict)
    model_is_synthetic: bool = False
    text: str = ""
    can_undo: bool = False
    at_capacity: bool = False
    hand_present: bool = False
    handedness: str | None = None
    hand_score: float | None = None
    raw_label: str | None = None
    raw_confidence: float | None = None
    raw_accepted: bool = False
    stable_label: str | None = None
    stable_confidence: float | None = None
    repeat_blocked: bool = False
    window_filled: int = 0
    commit_progress: float = 0.0
    release_remaining: int = 0
    release_total: int = 0
    last_commit_label: str | None = None
    last_commit_utc: str | None = None
    message: str = "Idle"
    top_predictions: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    last_frame_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class SignLanguageRuntime:
    """Owns the camera thread, controller, text buffer, and speech service."""

    def __init__(
        self,
        *,
        model_path: Path | str | None = None,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        mirror: bool = True,
        speech_enabled: bool = True,
        speech_rate: int = 170,
        speech_volume: float = 1.0,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        smoothing_min_votes: int = DEFAULT_SMOOTHING_MIN_VOTES,
        smoothing_min_ratio: float = DEFAULT_SMOOTHING_MIN_RATIO,
        release_frames: int = DEFAULT_RELEASE_FRAMES,
        voice_hint: str | None = None,
        speech_engine_factory=None,
        auto_start_camera: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.mirror = bool(mirror)
        self.auto_start_camera = bool(auto_start_camera)

        self.text = TextBuffer()
        self.status = RuntimeStatus(camera_index=self.camera_index)
        self.session = RecognitionSession(
            window=smoothing_window,
            min_votes=smoothing_min_votes,
            min_ratio=smoothing_min_ratio,
            release_frames=release_frames,
        )
        self.speech = SpeechService(
            enabled=speech_enabled,
            rate=speech_rate,
            volume=speech_volume,
            voice_hint=voice_hint,
            engine_factory=speech_engine_factory,
        )

        self._model: ModelBundle | None = None
        self._tracker: HandTracker | None = None
        self._image_tracker: HandTracker | None = None
        self._capture = None
        self._camera_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame_jpeg: bytes | None = None
        self._model_path = Path(model_path) if model_path else None
        self._notices: list[str] = []
        self._last_commit_time = 0.0

    # ------------------------------------------------------------------ model
    def load_model(self, path: Path | str | None = None) -> bool:
        candidate = Path(path) if path else self._model_path
        if candidate is None:
            with self._lock:
                self.status.model_loaded = False
                self.status.model_error = (
                    "No model selected. Pass --model PATH, train one with train_model.py, "
                    "or run with --demo for the synthetic plumbing demo."
                )
            return False
        try:
            bundle = load_model_bundle(candidate)
        except ModelBundleError as exc:
            with self._lock:
                self._model = None
                self.status.model_loaded = False
                self.status.model_path = str(candidate)
                self.status.model_error = str(exc)
            return False

        with self._lock:
            self._model = bundle
            self._model_path = candidate
            self.session.reset()
            self.status.model_loaded = True
            self.status.model_error = None
            self.status.model_path = str(candidate)
            self.status.model_metadata = bundle.public_metadata()
            self.status.model_is_synthetic = bundle.is_synthetic
            self.status.thresholds = dict(bundle.thresholds)
            self.status.notices = self._compute_notices(bundle)
        return True

    def model(self) -> ModelBundle | None:
        with self._lock:
            return self._model

    def _compute_notices(self, bundle: ModelBundle) -> list[str]:
        notices: list[str] = []
        training = bundle.metadata.get("training", {})
        dataset_kind = str(training.get("dataset_kind", "unknown"))
        if dataset_kind == "synthetic":
            notices.append(
                "DEMO MODEL — trained on synthetic placeholder landmarks. Predictions are "
                "not valid sign recognition and must not be reported as accuracy results. "
                "Collect real samples with collect_data.py, then retrain."
            )
        elif dataset_kind == "mixed":
            notices.append(
                "Model mixes synthetic and recorded samples. Report metrics for recorded "
                "data only."
            )
        if UNKNOWN_LABEL not in set(bundle.labels):
            notices.append(
                "No '_unknown' class in this model: unknown-sign rejection relies on the "
                "confidence/margin thresholds alone and is unmeasured."
            )
        if bundle.thresholds.get("source") != "validation":
            notices.append(
                "Thresholds were not calibrated on a validation split "
                f"(source: {bundle.thresholds.get('source')})."
            )
        split_info = bundle.metadata.get("split", {})
        if split_info and not split_info.get("test_contains_only_unseen_subjects", False):
            notices.append(
                "The reported test split is not guaranteed to contain only unseen people. "
                "Treat generalisation claims as provisional."
            )
        return notices

    # ------------------------------------------------------------------ tracker
    def _ensure_tracker(self) -> HandTracker:
        with self._lock:
            if self._tracker is None:
                self._tracker = HandTracker()
            return self._tracker

    # ------------------------------------------------------------------ camera
    def start_camera(self, index: int | None = None) -> tuple[bool, str]:
        with self._lock:
            if index is not None:
                self.camera_index = int(index)
                self.status.camera_index = self.camera_index
            if self.status.camera_running:
                return True, "Camera already running."

        try:
            self._ensure_tracker().open()
        except HandTrackerError as exc:
            with self._lock:
                self.status.camera_error = str(exc)
            return False, str(exc)

        with self._lock:
            self.status.camera_error = None
            self.status.camera_running = True
            self.status.message = "Starting camera…"
            self._latest_frame_jpeg = None
            self._stop_event.clear()
            self._camera_thread = threading.Thread(
                target=self._camera_loop, name="camera-loop", daemon=True
            )
            self._camera_thread.start()
        return True, "Camera starting."

    def stop_camera(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        thread = self._camera_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._camera_thread = None
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:  # pragma: no cover - best effort
                    pass
                self._capture = None
            self.status.camera_running = False
            self.status.camera_error = None
            self.status.camera_backend = None
            self.status.camera_fps = 0.0
            self.status.hand_present = False
            self.status.handedness = None
            self.status.hand_score = None
            self.status.raw_label = None
            self.status.raw_confidence = None
            self.status.raw_accepted = False
            self.status.stable_label = None
            self.status.stable_confidence = None
            self.status.repeat_blocked = False
            self.status.window_filled = 0
            self.status.commit_progress = 0.0
            self.status.release_remaining = 0
            self.status.top_predictions = []
            self.status.message = "Camera stopped"
            self._latest_frame_jpeg = None
        self.session.reset()

    def camera_thread_alive(self) -> bool:
        thread = self._camera_thread
        return bool(thread and thread.is_alive())

    def _camera_loop(self) -> None:
        import cv2

        try:
            capture, backend = open_camera(self.camera_index, self.width, self.height)
        except CameraError as exc:
            with self._lock:
                self.status.camera_running = False
                self.status.camera_error = str(exc)
                self.status.message = "Camera error"
            return

        with self._lock:
            self._capture = capture
            self.status.camera_backend = backend

        frame_times: list[float] = []
        frame_index = 0
        last_error: str | None = None
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    last_error = "Camera stopped returning frames."
                    time.sleep(0.05)
                    break
                frame_index += 1
                now = time.perf_counter()
                frame_times.append(now)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                fps = (
                    (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    if len(frame_times) > 1 and frame_times[-1] > frame_times[0]
                    else 0.0
                )

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = frame_index * 33  # monotonic, independent of wall clock
                detections: list[HandDetection] = []
                tracker_error: str | None = None
                try:
                    detections = self._ensure_tracker().detect(
                        frame, timestamp_ms=timestamp_ms
                    )
                except HandTrackerError as exc:
                    tracker_error = str(exc)

                self._process_frame(frame, detections, fps, tracker_error)
        finally:
            try:
                capture.release()
            except Exception:  # pragma: no cover - best effort
                pass
            with self._lock:
                self._capture = None
                if self._stop_event.is_set():
                    self.status.camera_running = False
                else:
                    self.status.camera_running = False
                    self.status.camera_error = last_error or self.status.camera_error
                    if last_error:
                        self.status.message = "Camera stopped unexpectedly"

    def _process_frame(
        self,
        frame: np.ndarray,
        detections: list[HandDetection],
        fps: float,
        tracker_error: str | None,
    ) -> None:
        """Run recognition for one frame and publish results plus JPEG bytes."""

        import cv2

        model = self.model()
        primary = detections[0] if detections else None
        prediction: Prediction | None = None
        feature_error: str | None = None

        if primary is not None and model is not None:
            try:
                features = model.extract_features(primary.landmarks)
                prediction = model.predict(features)
            except FeatureError as exc:
                feature_error = str(exc)
            except ModelBundleError as exc:
                feature_error = str(exc)

        with self._lock:
            update = self.session.update(
                hand_present=primary is not None,
                label=prediction.accepted_label if prediction else None,
                confidence=prediction.confidence if prediction else None,
                accepted=bool(prediction and prediction.accepted),
            )
            if update.commit:
                try:
                    if not self.text.add_letter(update.commit):
                        update = replace(
                            update,
                            message=(
                                "Text is full — clear it or press Delete, then hold "
                                "the sign again."
                            ),
                        )
                except TextError:
                    pass

            snapshot = self.text.snapshot()
            self.status.camera_running = True
            self.status.camera_fps = round(float(fps), 1)
            self.status.frames_processed += 1
            self.status.hand_present = primary is not None
            self.status.handedness = primary.handedness if primary else None
            self.status.hand_score = round(primary.score, 4) if primary else None
            self.status.raw_label = prediction.top_label if prediction else None
            self.status.raw_confidence = (
                round(prediction.confidence, 4) if prediction else None
            )
            self.status.raw_accepted = bool(prediction and prediction.accepted)
            self.status.stable_label = update.stable_label
            self.status.stable_confidence = (
                round(update.stable_confidence, 4)
                if update.stable_confidence is not None
                else None
            )
            self.status.repeat_blocked = update.repeat_blocked
            self.status.window_filled = int(update.window_filled)
            self.status.commit_progress = round(float(update.commit_progress), 4)
            self.status.release_remaining = int(update.release_remaining)
            self.status.release_total = int(update.release_total)
            if update.commit:
                self.status.last_commit_label = update.commit
                self.status.last_commit_utc = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
            self.status.message = update.message
            self.status.top_predictions = (
                [item.to_dict() for item in prediction.top] if prediction else []
            )
            self.status.text = snapshot.text
            self.status.can_undo = snapshot.can_undo
            self.status.at_capacity = snapshot.at_capacity
            self.status.last_frame_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.status.notices = self._notices or self.status.notices
            if tracker_error:
                self.status.camera_error = tracker_error
            if feature_error:
                self.status.message = f"Landmark problem: {feature_error}"

        overlay_label = None
        if update.stable_label:
            overlay_label = update.stable_label
        elif prediction is not None and prediction.accepted is False:
            overlay_label = "?"
        try:
            canvas = self._ensure_tracker().draw(frame, detections, label=overlay_label)
        except HandTrackerError:
            canvas = frame

        ok, buffer = cv2.imencode(
            ".jpg", _resize_letterbox(canvas), [int(cv2.IMWRITE_JPEG_QUALITY), 78]
        )
        if ok:
            with self._frame_lock:
                self._latest_frame_jpeg = buffer.tobytes()

    def latest_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame_jpeg

    # ------------------------------------------------------------------ controls
    def add_current(self) -> tuple[bool, str]:
        with self._lock:
            label = self.session.force_current_commit()
            if label is None:
                return False, "No stable sign to add yet."
            if not self.text.add_letter(label):
                return False, "Text is full. Clear it first."
            self.status.text = self.text.text
            self.status.can_undo = self.text.snapshot().can_undo
            self.status.message = f"Added {label!r} manually."
            return True, f"Added {label}."

    def add_space(self) -> tuple[bool, str]:
        with self._lock:
            changed = self.text.add_space()
            self.status.text = self.text.text
            self.status.can_undo = self.text.snapshot().can_undo
            self.status.message = "Space added." if changed else "Nothing to separate yet."
            return (True, "Space added.") if changed else (False, "Nothing to separate yet.")

    def backspace(self) -> tuple[bool, str]:
        with self._lock:
            changed = self.text.backspace()
            self.status.text = self.text.text
            self.status.can_undo = self.text.snapshot().can_undo
            self.status.message = "Character deleted." if changed else "Text is empty."
            return (True, "Character deleted.") if changed else (False, "Text is empty.")

    def clear_text(self) -> tuple[bool, str]:
        with self._lock:
            changed = self.text.clear()
            self.status.text = ""
            self.status.can_undo = self.text.snapshot().can_undo
            self.status.message = "Text cleared." if changed else "Text is already empty."
            self.session.reset()
            return (True, "Text cleared.") if changed else (False, "Text is already empty.")

    def undo(self) -> tuple[bool, str]:
        with self._lock:
            changed = self.text.undo()
            self.status.text = self.text.text
            self.status.can_undo = self.text.snapshot().can_undo
            self.status.message = "Undo." if changed else "Nothing to undo."
            return (True, "Undo.") if changed else (False, "Nothing to undo.")

    def speak(self, text: str | None = None) -> tuple[bool, str]:
        with self._lock:
            payload = text if text is not None else self.text.text
        return self.speech.speak(payload)

    def stop_speaking(self) -> tuple[bool, str]:
        return self.speech.stop()

    def configure_speech(
        self,
        *,
        enabled: bool | None = None,
        rate: int | None = None,
        volume: float | None = None,
        voice: str | None = None,
    ) -> tuple[bool, str]:
        return self.speech.configure(
            enabled=enabled, rate=rate, volume=volume, voice=voice
        )

    # ------------------------------------------------------------------ images
    def predict_image_bytes(self, raw: bytes) -> dict[str, Any]:
        """Run landmark extraction and recognition on in-memory image bytes.

        Nothing is written to disk.  A separate MediaPipe tracker is kept in
        IMAGE mode because ``detect()`` and ``detect_for_video()`` cannot share
        one running mode safely.
        """

        import cv2

        model = self.model()
        if model is None:
            return {
                "ok": True,
                "hand_detected": False,
                "message": "No model is loaded; image recognition is disabled.",
                "model_is_synthetic": False,
                "accepted": False,
            }

        array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return {
                "ok": True,
                "hand_detected": False,
                "message": "Could not decode the uploaded image.",
                "model_is_synthetic": model.is_synthetic,
                "accepted": False,
            }

        with self._lock:
            if self._image_tracker is None:
                self._image_tracker = HandTracker(running_mode="image", num_hands=1)
            tracker = self._image_tracker
        try:
            detections = tracker.detect(frame)
        except HandTrackerError as exc:
            return {
                "ok": True,
                "hand_detected": False,
                "message": f"Hand tracking failed: {exc}",
                "model_is_synthetic": model.is_synthetic,
                "accepted": False,
            }

        if not detections:
            return {
                "ok": True,
                "hand_detected": False,
                "message": "No hand was detected in the uploaded image.",
                "model_is_synthetic": model.is_synthetic,
                "accepted": False,
            }

        try:
            features = model.extract_features(detections[0].landmarks)
            prediction = model.predict(features)
        except (FeatureError, ModelBundleError) as exc:
            return {
                "ok": True,
                "hand_detected": True,
                "message": f"Landmark problem: {exc}",
                "model_is_synthetic": model.is_synthetic,
                "accepted": False,
            }

        return {
            "ok": True,
            "hand_detected": True,
            "label": prediction.accepted_label or prediction.top_label,
            "confidence": round(prediction.confidence, 6),
            "margin": round(prediction.margin, 6),
            "accepted": prediction.accepted,
            "top": [item.to_dict() for item in prediction.top],
            "thresholds": dict(model.thresholds),
            "model_is_synthetic": model.is_synthetic,
            "message": (
                "Recognised."
                if prediction.accepted
                else "Prediction rejected by confidence/margin thresholds."
            ),
        }

    # ------------------------------------------------------------------ status
    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            payload = self.status.to_dict()
            features = {
                "smoothing_window": self.session.smoother.window,
                "smoothing_min_votes": self.session.smoother.min_votes,
                "release_frames": self.session.gate.release_frames,
                "window_filled": int(self.status.window_filled),
                "commit_progress": self.status.commit_progress,
                "release_remaining": self.status.release_remaining,
            }
        payload["scope"] = {
            "language": "American Sign Language (ASL) fingerspelling",
            "signs": "static handshapes only",
            "excluded_dynamic": ["J", "Z"],
            "note": (
                "This is a static fingerspelling recogniser, not a sign-language "
                "translator. It cannot recognise dynamic signs, grammar, or facial "
                "expression."
            ),
        }
        payload["speech"] = self.speech.status()
        payload["controller"] = features
        payload["camera_thread_alive"] = self.camera_thread_alive()
        payload["threshold_defaults"] = {
            "min_confidence": DEFAULT_MIN_CONFIDENCE,
            "min_margin": DEFAULT_MIN_MARGIN,
        }
        payload["limits"] = {"max_text_length": MAX_TEXT_LENGTH}
        return payload

    def shutdown(self) -> None:
        self.stop_camera()
        self.speech.shutdown()
        with self._lock:
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None
            if self._image_tracker is not None:
                self._image_tracker.close()
                self._image_tracker = None
