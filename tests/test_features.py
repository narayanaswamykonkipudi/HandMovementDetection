from __future__ import annotations

import numpy as np
import pytest

from src.features import (
    FEATURE_DIM,
    FEATURE_DIM_GEOMETRY,
    FEATURE_SET_DIMS,
    FeatureError,
    build_feature_matrix,
    extract_feature_vector,
    geometry_features,
    normalize_landmarks,
)
from tests.conftest import make_landmarks


def test_normalize_landmarks_centres_wrist_and_scales():
    landmarks = make_landmarks(3, scale=2.0)
    normalized = normalize_landmarks(landmarks)
    points = normalized.reshape(21, 3)
    assert normalized.shape == (FEATURE_DIM,)
    assert np.allclose(points[0], 0.0)
    palm_scale = np.mean(np.linalg.norm(points[[5, 9, 13, 17]], axis=1))
    assert palm_scale == pytest.approx(1.0)


def test_normalize_rejects_wrong_shape_nonfinite_and_degenerate():
    with pytest.raises(FeatureError, match="shape"):
        normalize_landmarks(np.zeros((20, 3)))
    bad = make_landmarks(1)
    bad[2, 1] = np.nan
    with pytest.raises(FeatureError, match="finite"):
        normalize_landmarks(bad)
    with pytest.raises(FeatureError, match="Degenerate"):
        normalize_landmarks(np.zeros((21, 3)))


def test_geometry_feature_set_adds_distances_and_angles():
    base = extract_feature_vector(make_landmarks(4), "base")
    geometry = extract_feature_vector(make_landmarks(4), "geometry")
    assert base.shape == (FEATURE_DIM,)
    assert geometry.shape == (FEATURE_DIM_GEOMETRY,)
    assert np.allclose(base, geometry[:FEATURE_DIM])
    extra = geometry[FEATURE_DIM:]
    assert extra.shape == (25,)
    assert np.all(extra >= 0.0)
    assert np.all(extra <= 2.0)
    assert geometry_features(base).shape == (25,)


def test_feature_set_dimensions_and_empty_matrix():
    assert FEATURE_SET_DIMS == {"base": 63, "geometry": 88}
    empty = build_feature_matrix([], feature_set="geometry")
    assert empty.shape == (0, 88)
    matrix = build_feature_matrix([make_landmarks(1), make_landmarks(2)])
    assert matrix.shape == (2, 63)
    with pytest.raises(FeatureError, match="Unknown feature set"):
        extract_feature_vector(make_landmarks(1), "unknown")
