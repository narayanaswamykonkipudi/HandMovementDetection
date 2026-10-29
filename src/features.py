"""Landmark normalisation and feature-vector construction.

Design decision
---------------
The model consumes 21 MediaPipe hand landmarks (63 numbers).  Raw image
coordinates are not comparable across frames because the hand moves and changes
distance from the camera.  We therefore:

1. translate every landmark so the wrist is at the origin; and
2. divide by the mean wrist-to-palm-base distance (landmarks 5, 9, 13, 17).

The second step makes the features scale-invariant while *preserving
orientation*.  Orientation is linguistically relevant in signed languages, so
we deliberately do not rotate hands into a canonical pose.  Small rotations are
added only as training-time augmentation.

Handedness is kept as metadata for analysis but is not a model feature.  Static
fingerspelled handshapes are not defined by which hand is used, and including it
would let the classifier learn signer-specific shortcuts.

An optional ``geometry`` feature set appends 25 engineered descriptors: 5
wrist-to-fingertip distances, 10 pairwise fingertip distances, and 10 joint
angles (two per finger). These are evaluated as an ablation, not silently mixed
into the default model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.config import FEATURE_DIM, LANDMARK_COUNT, PALM_BASE_INDICES, WRIST_INDEX

FEATURE_SETS = ("base", "geometry")
GEOMETRY_EXTRA_DIM = 25
FEATURE_DIM_GEOMETRY = FEATURE_DIM + GEOMETRY_EXTRA_DIM
FEATURE_SET_DIMS = {"base": FEATURE_DIM, "geometry": FEATURE_DIM_GEOMETRY}
FINGERTIP_INDICES = (4, 8, 12, 16, 20)
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class FeatureError(ValueError):
    """Raised when landmarks cannot be converted into a valid feature vector."""


def normalize_landmarks(landmarks: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a 63-element, translation- and scale-normalised feature vector.

    Parameters
    ----------
    landmarks:
        Array-like with shape ``(21, 3)`` containing MediaPipe hand landmarks
        in image-normalised coordinates (x, y, z).

    Raises
    ------
    FeatureError
        If the shape, numeric type, or values are invalid.
    """

    try:
        points = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise FeatureError(f"Landmarks are not numeric: {exc}") from exc

    if points.shape != (LANDMARK_COUNT, 3):
        raise FeatureError(
            f"Expected landmarks with shape ({LANDMARK_COUNT}, 3), got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise FeatureError("Landmarks contain NaN or infinite values.")

    wrist = points[WRIST_INDEX]
    relative = points - wrist

    palm_widths = np.linalg.norm(relative[list(PALM_BASE_INDICES)], axis=1)
    scale = float(np.mean(palm_widths))
    if not np.isfinite(scale) or scale < 1e-6:
        raise FeatureError(
            "Degenerate hand geometry: palm scale is zero. The hand may be "
            "partially outside the frame or landmarks are unreliable."
        )

    normalized = relative / scale
    if not np.all(np.isfinite(normalized)):
        raise FeatureError("Normalisation produced non-finite features.")

    return normalized.reshape(FEATURE_DIM)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at ``b`` in radians divided by pi (range 0..1)."""

    first = a - b
    second = c - b
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-12:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi)


def geometry_features(normalized: np.ndarray) -> np.ndarray:
    """Append 25 scale-normalised distances and joint angles.

    The input is assumed to be the output of :func:`normalize_landmarks`, so
    distances are already in palm-scale units.
    """

    points = np.asarray(normalized, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
    wrist = points[WRIST_INDEX]
    tips = points[list(FINGERTIP_INDICES)]
    distances = [float(np.linalg.norm(tip - wrist)) for tip in tips]
    for first in range(len(FINGERTIP_INDICES)):
        for second in range(first + 1, len(FINGERTIP_INDICES)):
            distances.append(float(np.linalg.norm(tips[first] - tips[second])))
    angles: list[float] = []
    for joints in FINGER_JOINTS.values():
        mcp, pip, dip, tip = joints
        angles.append(_joint_angle(points[mcp], points[pip], points[dip]))
        angles.append(_joint_angle(points[pip], points[dip], points[tip]))
    extra = np.asarray(distances + angles, dtype=np.float64)
    if extra.shape != (GEOMETRY_EXTRA_DIM,) or not np.all(np.isfinite(extra)):
        raise FeatureError("Geometry feature construction produced invalid values.")
    return extra


def extract_feature_vector(
    landmarks: Sequence[Sequence[float]] | np.ndarray,
    feature_set: str = "base",
) -> np.ndarray:
    """Return the configured feature vector for one hand."""

    if feature_set not in FEATURE_SET_DIMS:
        raise FeatureError(
            f"Unknown feature set {feature_set!r}; expected one of {FEATURE_SETS}."
        )
    base = normalize_landmarks(landmarks)
    if feature_set == "base":
        return base
    return np.concatenate([base, geometry_features(base)])


def build_feature_matrix(
    landmark_rows: Iterable[Sequence[Sequence[float]] | np.ndarray],
    *,
    feature_set: str = "base",
) -> np.ndarray:
    """Normalise a sequence of hands into a ``(n_samples, feature_dim)`` matrix."""

    rows = [extract_feature_vector(row, feature_set) for row in landmark_rows]
    if not rows:
        return np.empty((0, FEATURE_SET_DIMS[feature_set]), dtype=np.float64)
    return np.vstack(rows)


def landmarks_from_feature_vector(features: Sequence[float] | np.ndarray) -> np.ndarray:
    """Reshape a flat feature vector back to ``(21, 3)`` for tests/debugging."""

    vector = np.asarray(features, dtype=np.float64)
    if vector.shape != (FEATURE_DIM,):
        raise FeatureError(f"Expected {FEATURE_DIM} features, got {vector.shape}.")
    return vector.reshape(LANDMARK_COUNT, 3)


def feature_table_to_array(frame, columns: Sequence[str]) -> np.ndarray:
    """Convert a pandas DataFrame of raw coordinate columns to a 3-D array."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureError(f"Missing coordinate columns: {missing}")
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    if values.shape[1] != FEATURE_DIM:
        raise FeatureError(
            f"Expected {FEATURE_DIM} coordinate columns, got {values.shape[1]}."
        )
    return values.reshape(len(frame), LANDMARK_COUNT, 3)
