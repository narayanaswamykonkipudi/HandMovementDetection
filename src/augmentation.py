"""Conservative landmark augmentations used only on the training split.

Augmentation is applied *after* wrist/scale normalisation, so the model sees
plausible variation in hand roll, scale, and landmark noise without changing the
class label.  Mirroring is valid for static fingerspelling because the same
handshape produced with the other hand is still that letter; we do not add
large 3-D rotations, which could turn one ASL handshape into another.

Never call this on validation or test data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config import LANDMARK_COUNT
from src.features import normalize_landmarks


def _rotation_matrix_z(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def augment_features(
    features: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
    *,
    mirror_probability: float = 0.5,
    max_rotation_degrees: float = 12.0,
    scale_jitter: float = 0.06,
    translation_jitter: float = 0.015,
    noise_std: float = 0.006,
    depth_noise_std: float = 0.004,
) -> np.ndarray:
    """Return one augmented 63-dimensional feature vector.

    The parameters are deliberately small because landmark coordinates are
    already normalised.  Values are documented in ``README.md`` so the
    augmentation is reproducible and auditable.
    """

    points = np.asarray(features, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot augment non-finite features.")

    if rng.random() < mirror_probability:
        points[:, 0] *= -1.0

    if max_rotation_degrees > 0:
        angle = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
        points = points @ _rotation_matrix_z(angle).T

    if scale_jitter > 0:
        points *= rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)

    if translation_jitter > 0:
        jitter = rng.uniform(
            -translation_jitter, translation_jitter, size=2
        )
        points[:, 0] += jitter[0]
        points[:, 1] += jitter[1]

    if noise_std > 0:
        points += rng.normal(0.0, noise_std, size=points.shape)

    if depth_noise_std > 0:
        points[:, 2] += rng.normal(0.0, depth_noise_std, size=LANDMARK_COUNT)

    # Re-normalise after geometric edits so every training row has exactly the
    # same wrist-centred, palm-scaled representation used at inference time.
    return normalize_landmarks(points)


def augment_dataset(
    features: np.ndarray,
    copies_per_sample: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """Create ``copies_per_sample`` augmented versions of every row.

    The output contains only synthetic copies; the caller must concatenate the
    originals.  This function is intentionally not used by validation/testing.
    """

    if copies_per_sample <= 0:
        return np.empty((0, features.shape[1]), dtype=np.float64)

    output = np.empty(
        (features.shape[0] * copies_per_sample, features.shape[1]),
        dtype=np.float64,
    )
    cursor = 0
    for row in features:
        for _ in range(copies_per_sample):
            output[cursor] = augment_features(row, rng, **kwargs)
            cursor += 1
    return output
