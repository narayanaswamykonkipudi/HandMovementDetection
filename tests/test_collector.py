from __future__ import annotations

import numpy as np
import pytest

from src.collector import (
    CollectionError,
    append_samples,
    collection_progress,
    count_samples,
    make_sample_row,
)
from src.config import CSV_COLUMNS
from tests.conftest import make_landmarks


def test_make_sample_row_validates_and_flattens():
    row = make_sample_row(
        make_landmarks(1),
        label="A",
        subject_id="s01",
        session_id="sess",
        source="webcam",
        handedness="Right",
        hand_score=0.9,
        sample_id="abc",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert row["label"] == "A"
    assert row["sample_id"] == "abc"
    assert len(row) == len(CSV_COLUMNS)
    assert all(np.isfinite(row[column]) for column in CSV_COLUMNS[8:])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"label": "J"}, "not supported"),
        ({"subject_id": ""}, "subject_id"),
        ({"session_id": "bad/name"}, "unsupported"),
        ({"source": "phone"}, "source"),
        ({"handedness": "Both"}, "handedness"),
        ({"hand_score": 1.2}, "between 0 and 1"),
    ],
)
def test_make_sample_row_rejects_bad_metadata(kwargs, message):
    base = {
        "label": "A",
        "subject_id": "s01",
        "session_id": "sess",
        "source": "webcam",
        "handedness": "Right",
        "hand_score": 0.9,
    }
    base.update(kwargs)
    with pytest.raises(CollectionError, match=message):
        make_sample_row(make_landmarks(1), **base)


def test_make_sample_row_rejects_bad_landmarks():
    base = {
        "label": "A",
        "subject_id": "s01",
        "session_id": "sess",
        "source": "webcam",
        "handedness": "Right",
        "hand_score": 0.9,
    }
    with pytest.raises(CollectionError, match="shape"):
        make_sample_row(np.zeros((20, 3)), **base)
    bad = make_landmarks(1)
    bad[0, 0] = np.inf
    with pytest.raises(CollectionError, match="finite"):
        make_sample_row(bad, **base)


def test_append_count_and_progress(tmp_path):
    path = tmp_path / "landmarks.csv"
    first = make_sample_row(
        make_landmarks(1),
        label="A",
        subject_id="s01",
        session_id="s1",
        source="webcam",
        handedness="Right",
        hand_score=0.9,
    )
    second = make_sample_row(
        make_landmarks(2),
        label="A",
        subject_id="s02",
        session_id="s2",
        source="webcam",
        handedness="Left",
        hand_score=0.95,
    )
    third = make_sample_row(
        make_landmarks(3),
        label="B",
        subject_id="s01",
        session_id="s1",
        source="webcam",
        handedness="Right",
        hand_score=0.8,
    )
    assert append_samples(path, [first]) == 1
    assert append_samples(path, [second, third]) == 2
    assert count_samples(path)["A"] == 2
    progress = collection_progress(path)
    assert progress["A"]["samples"] == 2
    assert len(progress["A"]["sessions"]) == 2
    assert len(progress["A"]["subjects"]) == 2
    assert progress["B"]["samples"] == 1


def test_append_rejects_incompatible_header(tmp_path):
    path = tmp_path / "landmarks.csv"
    path.write_text("wrong,header\n1,2\n", encoding="utf-8")
    with pytest.raises(CollectionError, match="incompatible header"):
        append_samples(
            path,
            [
                make_sample_row(
                    make_landmarks(1),
                    label="A",
                    subject_id="s01",
                    session_id="s1",
                    source="webcam",
                    handedness="Right",
                    hand_score=0.9,
                )
            ],
        )


def test_append_empty_and_count_missing(tmp_path):
    path = tmp_path / "landmarks.csv"
    assert append_samples(path, []) == 0
    assert count_samples(path) == {}
    assert collection_progress(path) == {}
