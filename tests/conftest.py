from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import create_app
from src.config import RAW_COORD_COLUMNS, STATIC_ASL_LETTERS
from src.settings import Settings
from src.text import TextBuffer


def make_landmarks(seed: int = 0, *, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 0.02, size=(21, 3))
    points[0] = 0.0
    points[5] = [0.0, 0.12 * scale, 0.0]
    points[9] = [0.0, 0.15 * scale, 0.0]
    points[13] = [0.0, 0.13 * scale, 0.0]
    points[17] = [0.0, 0.10 * scale, 0.0]
    for index in (4, 8, 12, 16, 20):
        points[index] = points[index] + [0.05, 0.10, 0.0]
    return points


def frame_from_landmarks(landmarks: np.ndarray, **metadata: Any) -> dict[str, Any]:
    row = {
        "sample_id": "sample",
        "label": "A",
        "subject_id": "s01",
        "session_id": "s01a",
        "source": "synthetic",
        "handedness": "Right",
        "hand_score": 0.99,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    row.update(metadata)
    for column, value in zip(RAW_COORD_COLUMNS, np.asarray(landmarks).reshape(-1), strict=True):
        row[column] = float(value)
    return row


def synthetic_frame(
    *,
    labels: tuple[str, ...] = STATIC_ASL_LETTERS,
    subjects: tuple[str, ...] = ("s01", "s02", "s03"),
    sessions: int = 2,
    rows_per_label: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counter = 0
    for subject in subjects:
        for session_index in range(sessions):
            for label in labels:
                for _ in range(rows_per_label):
                    counter += 1
                    rows.append(
                        frame_from_landmarks(
                            make_landmarks(counter),
                            sample_id=f"sample-{counter}",
                            label=label,
                            subject_id=subject,
                            session_id=f"{subject}-s{session_index + 1}",
                        )
                    )
    return pd.DataFrame(rows)


class FakeSpeech:
    def __init__(self) -> None:
        self.enabled = False
        self.volume = 1.0
        self.rate = 170
        self.started = False
        self.configured: list[dict[str, Any]] = []

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.started = False

    def status(self) -> dict[str, Any]:
        return {
            "backend": "pyttsx3",
            "enabled": self.enabled,
            "state": "disabled" if not self.enabled else "ready",
            "available": self.enabled,
            "speaking": False,
            "voice_count": 1,
            "voices": [{"id": "v1", "name": "Test Voice", "languages": []}],
            "selected_voice": "v1",
            "rate": self.rate,
            "volume": self.volume,
            "error": None,
        }

    def speak(self, text: str | None = None) -> tuple[bool, str]:
        return True, "Speaking."

    def stop(self) -> tuple[bool, str]:
        return True, "Stopped."

    def configure(self, **kwargs: Any) -> tuple[bool, str]:
        self.configured.append(kwargs)
        if kwargs.get("rate") is not None:
            self.rate = int(kwargs["rate"])
        if kwargs.get("volume") is not None:
            self.volume = float(kwargs["volume"])
        return True, "Speech settings updated."


class FakeRuntime:
    camera_index = 3

    def __init__(self) -> None:
        self.text = TextBuffer()
        self.speech = FakeSpeech()
        self.shutdown_called = False
        self.started_camera: int | None = None
        self.stop_camera_called = False
        self.loaded = False

    def load_model(self) -> bool:
        self.loaded = True
        return True

    def status_dict(self) -> dict[str, Any]:
        snapshot = self.text.snapshot()
        return {
            "camera_running": False,
            "camera_index": self.camera_index,
            "model_loaded": True,
            "model_is_synthetic": True,
            "model_path": "fake.joblib",
            "model_metadata": {"training": {"dataset_kind": "synthetic"}},
            "text": snapshot.text,
            "can_undo": snapshot.can_undo,
            "at_capacity": snapshot.at_capacity,
            "message": "Fake runtime",
            "speech": self.speech.status(),
            "controller": {
                "commit_progress": 0.0,
                "release_remaining": 0,
                "release_total": 8,
            },
        }

    def start_camera(self, index: int | None = None) -> tuple[bool, str]:
        self.started_camera = int(index if index is not None else self.camera_index)
        return True, "Camera starting."

    def stop_camera(self) -> None:
        self.stop_camera_called = True

    def add_current(self) -> tuple[bool, str]:
        added = self.text.add_letter("A")
        return added, "Added A" if added else "No stable sign"

    def add_space(self) -> tuple[bool, str]:
        changed = self.text.add_space()
        return changed, "Space added" if changed else "Space not added"

    def backspace(self) -> tuple[bool, str]:
        changed = self.text.backspace()
        return changed, "Deleted" if changed else "Nothing to delete"

    def clear_text(self) -> tuple[bool, str]:
        changed = self.text.clear()
        return changed, "Cleared" if changed else "Already empty"

    def undo(self) -> tuple[bool, str]:
        changed = self.text.undo()
        return changed, "Undone" if changed else "Nothing to undo"

    def speak(self, text: str | None = None) -> tuple[bool, str]:
        return self.speech.speak(text)

    def stop_speaking(self) -> tuple[bool, str]:
        return self.speech.stop()

    def configure_speech(self, **kwargs: Any) -> tuple[bool, str]:
        return self.speech.configure(**kwargs)

    def latest_frame(self) -> bytes | None:
        return None

    def predict_image_bytes(self, raw: bytes) -> dict[str, Any]:
        return {
            "ok": True,
            "hand_detected": True,
            "label": "A",
            "confidence": 0.9,
            "margin": 0.8,
            "accepted": True,
            "top": [{"label": "A", "probability": 0.9}],
            "thresholds": {"min_confidence": 0.5, "min_margin": 0.0},
            "model_is_synthetic": True,
            "message": f"Predicted from {len(raw)} bytes.",
        }

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.speech.shutdown()


@pytest.fixture()
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture()
def test_settings() -> Settings:
    return Settings(
        speech_enabled=False,
        llm_enabled=False,
        demo=True,
        reload_on_startup=False,
        camera_scan_max=2,
    )


@pytest.fixture()
def client(fake_runtime: FakeRuntime, test_settings: Settings):
    application = create_app(
        runtime=fake_runtime,
        settings=test_settings,
        load_model_on_startup=False,
    )
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def demo_model_path() -> Path:
    path = Path("models/demo_classifier.joblib")
    if not path.exists():
        pytest.skip("Demo model is not built; run python scripts/build_demo.py")
    return path
