"""Generate clearly-labelled SYNTHETIC landmark data for plumbing tests.

This is **not** a sign-language dataset.  It produces arbitrary, hand-like
skeletons from per-letter curl/spread parameters so that the complete pipeline
(split -> train -> predict -> smooth -> text) can be exercised without a webcam
or third-party images.  Models trained on this data must never be presented as
real ASL recognition, and every generated row is tagged ``source=synthetic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.collector import make_sample_row
from src.config import CSV_COLUMNS, STATIC_ASL_LETTERS, UNKNOWN_LABEL


@dataclass(frozen=True)
class FingerShape:
    curls: tuple[float, float, float, float, float]  # thumb, index, middle, ring, pinky
    spread: float
    thumb_angle: float
    wrist_rotation: float = 0.0
    thumb_tuck: float = 0.0


# These are arbitrary synthetic shape parameters, loosely inspired by finger
# extension patterns.  They are not authoritative ASL descriptions.
LETTER_SHAPES: dict[str, FingerShape] = {
    "A": FingerShape((0.55, 1.00, 1.00, 1.00, 1.00), -0.02, -0.35, 0.0, 0.2),
    "B": FingerShape((0.10, 0.00, 0.00, 0.00, 0.00), 0.00, -0.15, 0.0, 0.0),
    "C": FingerShape((0.55, 0.55, 0.55, 0.55, 0.55), 0.04, -0.30, 0.05, 0.0),
    "D": FingerShape((0.45, 0.00, 1.00, 1.00, 1.00), 0.02, -0.20, 0.0, 0.0),
    "E": FingerShape((0.85, 0.92, 0.92, 0.92, 0.92), -0.05, -0.45, 0.10, 0.0),
    "F": FingerShape((0.65, 0.80, 0.00, 0.00, 0.00), 0.03, -0.55, 0.0, 0.0),
    "G": FingerShape((0.10, 0.18, 1.00, 1.00, 1.00), 0.00, 0.20, 0.60, 0.0),
    "H": FingerShape((0.60, 0.00, 0.00, 1.00, 1.00), 0.00, -0.30, 0.20, 0.0),
    "I": FingerShape((0.80, 1.00, 1.00, 1.00, 0.00), 0.02, -0.40, 0.0, 0.0),
    "K": FingerShape((0.05, 0.00, 0.00, 1.00, 1.00), 0.12, 0.05, -0.05, 0.0),
    "L": FingerShape((0.00, 0.00, 1.00, 1.00, 1.00), 0.03, 0.10, 0.0, 0.0),
    "M": FingerShape((1.00, 0.92, 0.92, 0.92, 1.00), -0.02, -0.40, 0.05, 0.7),
    "N": FingerShape((0.95, 0.92, 0.92, 1.00, 1.00), -0.02, -0.40, 0.03, 0.4),
    "O": FingerShape((0.42, 0.62, 0.62, 0.62, 0.62), 0.01, -0.30, 0.05, 0.0),
    "P": FingerShape((0.18, 0.28, 0.28, 1.00, 1.00), 0.00, 0.05, 0.75, 0.0),
    "Q": FingerShape((0.22, 0.35, 1.00, 1.00, 1.00), -0.04, 0.15, 0.85, 0.0),
    "R": FingerShape((0.50, 0.05, 0.05, 1.00, 1.00), -0.12, -0.20, 0.0, 0.0),
    "S": FingerShape((1.00, 1.00, 1.00, 1.00, 1.00), -0.02, -0.10, 0.0, 0.5),
    "T": FingerShape((0.78, 1.00, 1.00, 1.00, 1.00), -0.04, -0.25, 0.0, 0.8),
    "U": FingerShape((0.70, 0.00, 0.00, 1.00, 1.00), -0.03, -0.25, 0.0, 0.0),
    "V": FingerShape((0.62, 0.00, 0.00, 1.00, 1.00), 0.16, -0.25, 0.0, 0.0),
    "W": FingerShape((0.52, 0.00, 0.00, 0.00, 1.00), 0.04, -0.30, 0.0, 0.0),
    "X": FingerShape((0.72, 0.62, 1.00, 1.00, 1.00), 0.03, -0.40, 0.0, 0.0),
    "Y": FingerShape((0.05, 1.00, 1.00, 1.00, 0.00), 0.02, 0.20, 0.0, 0.0),
}

FINGER_STARTS = {
    "thumb": (0.20, -0.08),
    "index": (-0.20, -0.32),
    "middle": (-0.02, -0.36),
    "ring": (0.16, -0.34),
    "pinky": (0.32, -0.29),
}
FINGER_LENGTHS = {
    "thumb": (0.18, 0.12, 0.10),
    "index": (0.22, 0.13, 0.10),
    "middle": (0.24, 0.15, 0.11),
    "ring": (0.22, 0.14, 0.10),
    "pinky": (0.18, 0.11, 0.08),
}
# Base angles measured from +x (up is -pi/2). Index/pinky fan out.
FINGER_ANGLES = {
    "thumb": -0.30,
    "index": -1.68,
    "middle": -1.57,
    "ring": -1.45,
    "pinky": -1.30,
}
LANDMARK_RANGES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _finger_points(
    start: tuple[float, float],
    angle: float,
    lengths: tuple[float, float, float],
    curl: float,
    *,
    curl_per_segment: float = 1.25,
) -> list[tuple[float, float, float]]:
    x, y = start
    points: list[tuple[float, float, float]] = [(x, y, 0.0)]
    current_angle = angle
    for segment_index, length in enumerate(lengths, start=1):
        current_angle += curl * curl_per_segment
        x += length * np.cos(current_angle)
        y += length * np.sin(current_angle)
        z = -0.035 * curl * segment_index
        points.append((x, y, z))
    return points


def synthesize_hand(
    shape: FingerShape,
    rng: np.random.Generator,
    *,
    subject_shift: float = 0.0,
    session_shift: float = 0.0,
    noise_std: float = 0.005,
) -> np.ndarray:
    """Create one 21x3 hand skeleton in image-normalised coordinates."""

    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = (0.0, 0.0, 0.0)

    for finger_index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        curl = float(np.clip(shape.curls[finger_index] + rng.normal(0, 0.035), 0.0, 1.0))
        spread = shape.spread * (1.0 if finger in {"index", "middle", "pinky"} else 0.0)
        if finger == "thumb":
            spread += shape.thumb_angle * 0.25
        angle = FINGER_ANGLES[finger] + spread
        start = FINGER_STARTS[finger]
        generated = _finger_points(start, angle, FINGER_LENGTHS[finger], curl)
        for landmark_index, point in zip(LANDMARK_RANGES[finger], generated, strict=True):
            points[landmark_index] = point

    # Synthetic "thumb tucked across palm" variation for M/N/T/S-like shapes.
    if shape.thumb_tuck:
        for landmark_index in (2, 3, 4):
            points[landmark_index, 0] = float(
                points[landmark_index, 0] * (1.0 - 0.35 * shape.thumb_tuck)
                - 0.14 * shape.thumb_tuck
            )
            points[landmark_index, 1] += 0.14 * shape.thumb_tuck

    # Session/subject variation: roll the hand, scale, and translate it.
    rotation = shape.wrist_rotation + subject_shift + session_shift
    rotation += rng.normal(0, 0.055)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rotated = points.copy()
    rotated[:, 0] = points[:, 0] * cos_r - points[:, 1] * sin_r
    rotated[:, 1] = points[:, 0] * sin_r + points[:, 1] * cos_r
    rotated *= rng.uniform(0.82, 1.18)
    rotated[:, 0] += 0.52 + rng.normal(0, 0.035)
    rotated[:, 1] += 0.55 + rng.normal(0, 0.035)
    rotated[:, 2] += rng.normal(0, 0.01)
    rotated += rng.normal(0, noise_std, size=rotated.shape)
    rotated[:, 2] += rng.normal(0, noise_std * 0.6, size=21)
    return rotated


def generate_demo_rows(
    *,
    seed: int = 7,
    subjects: tuple[str, ...] = ("demo-s01", "demo-s02", "demo-s03"),
    sessions_per_subject: int = 2,
    frames_per_session: int = 20,
    unknown_per_session: int = 14,
) -> pd.DataFrame:
    """Return a synthetic DataFrame with trainable static-letter classes."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    letters = list(STATIC_ASL_LETTERS)

    for subject_index, subject in enumerate(subjects):
        for session_index in range(sessions_per_subject):
            session = f"{subject}-s{session_index + 1}"
            subject_shift = 0.035 * (subject_index - 1)
            session_shift = 0.02 * (session_index - 0.5)
            for letter in letters:
                shape = LETTER_SHAPES[letter]
                for _ in range(frames_per_session):
                    landmarks = synthesize_hand(
                        shape,
                        rng,
                        subject_shift=subject_shift,
                        session_shift=session_shift,
                    )
                    rows.append(
                        make_sample_row(
                            landmarks,
                            label=letter,
                            subject_id=subject,
                            session_id=session,
                            source="synthetic",
                            handedness="Right" if rng.random() < 0.5 else "Left",
                            hand_score=float(rng.uniform(0.90, 1.0)),
                        )
                    )
            for _ in range(unknown_per_session):
                shape = FingerShape(
                    curls=tuple(rng.uniform(0.05, 0.95, size=5).tolist()),
                    spread=float(rng.uniform(-0.2, 0.2)),
                    thumb_angle=float(rng.uniform(-0.5, 0.4)),
                    wrist_rotation=float(rng.uniform(-0.8, 0.8)),
                    thumb_tuck=float(rng.uniform(0, 0.8)),
                )
                landmarks = synthesize_hand(
                    shape,
                    rng,
                    subject_shift=subject_shift,
                    session_shift=session_shift,
                )
                rows.append(
                    make_sample_row(
                        landmarks,
                        label=UNKNOWN_LABEL,
                        subject_id=subject,
                        session_id=session,
                        source="synthetic",
                        handedness="Right",
                        hand_score=float(rng.uniform(0.90, 1.0)),
                    )
                )

    frame = pd.DataFrame(rows)
    return frame[list(CSV_COLUMNS)]


def write_demo_dataset(
    output_path: Path | str,
    *,
    seed: int = 7,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to regenerate it."
        )
    frame = generate_demo_rows(seed=seed, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
