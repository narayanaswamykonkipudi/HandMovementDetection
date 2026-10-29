from __future__ import annotations

import pytest

from src.smoothing import (
    CommitGate,
    PredictionSmoother,
    RecognitionSession,
)
from src.text import TextBuffer, TextError


def test_text_buffer_add_undo_clear_and_capacity():
    buffer = TextBuffer(max_length=4)
    assert buffer.snapshot().text == ""
    assert buffer.add_letter("a") is True
    assert buffer.text == "A"
    assert buffer.add_space() is True
    assert buffer.add_space() is False
    assert buffer.add_letter("B") is True
    assert buffer.add_letter("C") is True
    assert buffer.add_letter("D") is False
    assert buffer.snapshot().at_capacity is True
    assert buffer.undo() is True
    assert buffer.text == "A B"
    assert buffer.backspace() is True
    assert buffer.text == "A "
    assert buffer.backspace() is True
    assert buffer.text == "A"
    assert buffer.clear() is True
    assert buffer.text == ""
    assert buffer.clear() is False
    assert buffer.undo() is True
    assert buffer.text == "A"
    while buffer.undo():
        pass
    assert buffer.snapshot().can_undo is False


def test_text_buffer_rejects_invalid_and_restores():
    buffer = TextBuffer(max_length=5)
    with pytest.raises(TextError):
        buffer.add_letter("J")
    with pytest.raises(TextError):
        buffer.add_letter("AB")
    with pytest.raises(ValueError):
        TextBuffer(max_length=0)
    buffer.set_text("ABC")
    assert buffer.text == "ABC"
    with pytest.raises(TextError):
        buffer.set_text("TOOLONG")


def test_smoother_requires_votes_and_reports_stability():
    smoother = PredictionSmoother(window=4, min_votes=3, min_ratio=0.6)
    updates = [
        smoother.update(hand_present=True, label="A", confidence=0.9, accepted=True)
        for _ in range(2)
    ]
    assert updates[-1].stable_label is None
    stable = smoother.update(
        hand_present=True, label="A", confidence=0.8, accepted=True
    )
    assert stable.stable_label == "A"
    assert stable.became_stable is True
    assert stable.stable_confidence == pytest.approx(0.8666667)
    assert smoother.current_stable == "A"
    smoother.reset()
    assert smoother.current_stable is None


def test_smoother_validates_configuration_and_ignores_rejected_frames():
    with pytest.raises(ValueError):
        PredictionSmoother(window=0)
    with pytest.raises(ValueError):
        PredictionSmoother(window=2, min_votes=3)
    with pytest.raises(ValueError):
        PredictionSmoother(window=2, min_votes=1, min_ratio=0)
    smoother = PredictionSmoother(window=2, min_votes=1, min_ratio=0.5)
    update = smoother.update(
        hand_present=True, label="A", confidence=0.9, accepted=False
    )
    assert update.stable_label is None


def test_commit_gate_repeat_and_release():
    gate = CommitGate(release_frames=2)
    assert gate.update("A", hand_present=True) == "A"
    assert gate.update("A", hand_present=True) is None
    assert gate.is_blocked("A") is True
    assert gate.update(None, hand_present=False) is None
    assert gate.armed is False
    assert gate.update(None, hand_present=False) is None
    assert gate.armed is True
    assert gate.update("A", hand_present=True) == "A"
    gate.note_manual_commit("A")
    assert gate.is_blocked("A") is True
    assert gate.update("B", hand_present=True) == "B"
    gate.reset()
    assert gate.armed is True
    with pytest.raises(ValueError):
        CommitGate(release_frames=0)


def test_recognition_session_progress_release_and_manual_commit():
    session = RecognitionSession(window=4, min_votes=3, min_ratio=0.6, release_frames=2)
    first = session.update(hand_present=True, label="A", confidence=0.9, accepted=True)
    assert first.message.startswith("Stabilising")
    assert 0.0 < first.commit_progress < 1.0
    second = session.update(hand_present=True, label="A", confidence=0.9, accepted=True)
    assert second.commit is None
    third = session.update(hand_present=True, label="A", confidence=0.9, accepted=True)
    assert third.commit == "A"
    assert third.message == "Added 'A'"
    assert session.force_current_commit() == "A"
    released = session.update(hand_present=False)
    assert released.release_remaining == 1
    assert released.message == "No hand detected"
    session.update(hand_present=False)
    for _ in range(2):
        session.update(hand_present=True, label="A", confidence=0.9, accepted=True)
    unblocked = session.update(hand_present=True, label="A", confidence=0.9, accepted=True)
    assert unblocked.commit == "A"
    session.reset()
    assert session.smoother.current_stable is None
    assert session.force_current_commit() is None
