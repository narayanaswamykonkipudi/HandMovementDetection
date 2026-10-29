"""Temporal smoothing and hold-to-repeat suppression.

Frame-level classifiers flicker.  Two independent mechanisms handle this:

1. ``PredictionSmoother`` requires a majority of the recent accepted frames to
   agree before a sign is considered stable.
2. ``CommitGate`` ensures a stable sign is appended to the text only once.
   Repeating the same letter requires briefly removing the hand (a "release"),
   or using the explicit *Add current* button.  Changing to a different sign
   commits immediately after smoothing.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from src.config import (
    DEFAULT_RELEASE_FRAMES,
    DEFAULT_SMOOTHING_MIN_RATIO,
    DEFAULT_SMOOTHING_MIN_VOTES,
    DEFAULT_SMOOTHING_WINDOW,
)


@dataclass(frozen=True)
class SmootherUpdate:
    stable_label: str | None
    stable_confidence: float | None
    became_stable: bool
    window_filled: int
    counts: dict[str, int] = field(default_factory=dict)


class PredictionSmoother:
    def __init__(
        self,
        window: int = DEFAULT_SMOOTHING_WINDOW,
        min_votes: int = DEFAULT_SMOOTHING_MIN_VOTES,
        min_ratio: float = DEFAULT_SMOOTHING_MIN_RATIO,
    ) -> None:
        if window < 1:
            raise ValueError("window must be at least 1.")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be between 1 and window.")
        if not 0 < min_ratio <= 1:
            raise ValueError("min_ratio must be in (0, 1].")
        self.window = window
        self.min_votes = min_votes
        self.min_ratio = min_ratio
        self._buffer: deque[tuple[str, float] | None] = deque(maxlen=window)
        self._previous_stable: str | None = None

    def update(
        self,
        *,
        hand_present: bool,
        label: str | None,
        confidence: float | None,
        accepted: bool,
    ) -> SmootherUpdate:
        entry: tuple[str, float] | None = None
        if hand_present and accepted and label:
            entry = (label, float(confidence or 0.0))
        self._buffer.append(entry)

        counts = Counter(entry[0] for entry in self._buffer if entry is not None)
        filled = len(self._buffer)
        candidate: str | None = None
        stable_confidence: float | None = None

        if counts:
            top_label, top_count = counts.most_common(1)[0]
            if top_count >= self.min_votes and top_count / filled >= self.min_ratio:
                candidate = top_label
                confidences = [
                    entry[1] for entry in self._buffer if entry and entry[0] == top_label
                ]
                stable_confidence = sum(confidences) / len(confidences)

        became_stable = candidate is not None and candidate != self._previous_stable
        self._previous_stable = candidate
        return SmootherUpdate(
            stable_label=candidate,
            stable_confidence=stable_confidence,
            became_stable=became_stable,
            window_filled=filled,
            counts=dict(counts),
        )

    @property
    def current_stable(self) -> str | None:
        return self._previous_stable

    def reset(self) -> None:
        self._buffer.clear()
        self._previous_stable = None


class CommitGate:
    """Emit a sign once per activation, preventing held-gesture repeats."""

    def __init__(self, release_frames: int = DEFAULT_RELEASE_FRAMES) -> None:
        if release_frames < 1:
            raise ValueError("release_frames must be at least 1.")
        self.release_frames = release_frames
        self.last_committed: str | None = None
        self.armed = True
        self.missing_frames = 0

    def update(self, stable_label: str | None, *, hand_present: bool) -> str | None:
        if not hand_present:
            self.missing_frames += 1
            if self.missing_frames >= self.release_frames:
                self.armed = True
            return None

        if stable_label is None:
            # No *accepted* stable sign yet. During the first frames of a new
            # gesture the smoother still reports the previous, decaying label,
            # so the gate must not re-arm from that stale value.
            return None

        self.missing_frames = 0

        if stable_label != self.last_committed:
            self.last_committed = stable_label
            self.armed = False
            return stable_label

        if self.armed:
            self.armed = False
            return stable_label
        return None

    def note_manual_commit(self, label: str) -> None:
        """Synchronise the gate after the user clicks *Add current*."""

        self.last_committed = label
        self.armed = False

    def is_blocked(self, label: str | None) -> bool:
        """True when ``label`` is currently held after already being added."""

        return label is not None and label == self.last_committed and not self.armed

    def reset(self) -> None:
        self.last_committed = None
        self.armed = True
        self.missing_frames = 0


@dataclass(frozen=True)
class SessionUpdate:
    hand_present: bool
    raw_accepted: bool
    stable_label: str | None
    stable_confidence: float | None
    commit: str | None
    repeat_blocked: bool
    window_filled: int
    message: str
    commit_progress: float = 0.0
    release_remaining: int = 0
    release_total: int = 0


class RecognitionSession:
    """Combine smoothing, gating, and human-readable status for one stream."""

    def __init__(
        self,
        *,
        window: int = DEFAULT_SMOOTHING_WINDOW,
        min_votes: int = DEFAULT_SMOOTHING_MIN_VOTES,
        min_ratio: float = DEFAULT_SMOOTHING_MIN_RATIO,
        release_frames: int = DEFAULT_RELEASE_FRAMES,
    ) -> None:
        self.smoother = PredictionSmoother(window, min_votes, min_ratio)
        self.gate = CommitGate(release_frames)

    def update(
        self,
        *,
        hand_present: bool,
        label: str | None = None,
        confidence: float | None = None,
        accepted: bool = False,
    ) -> SessionUpdate:
        smoothed = self.smoother.update(
            hand_present=hand_present,
            label=label,
            confidence=confidence,
            accepted=accepted,
        )
        commit = self.gate.update(
            smoothed.stable_label, hand_present=hand_present
        )

        commit_progress = 0.0
        if hand_present and accepted and smoothed.counts:
            top_count = max(smoothed.counts.values())
            vote_progress = top_count / max(1, self.smoother.min_votes)
            ratio_progress = (
                (top_count / max(1, smoothed.window_filled)) / self.smoother.min_ratio
            )
            commit_progress = max(0.0, min(1.0, min(vote_progress, ratio_progress)))
        if smoothed.stable_label is not None:
            commit_progress = 1.0

        release_total = self.gate.release_frames
        release_remaining = 0
        if not hand_present and self.gate.last_committed is not None and not self.gate.armed:
            release_remaining = max(0, release_total - self.gate.missing_frames)

        if not hand_present:
            message = "No hand detected"
        elif not accepted:
            message = "Uncertain prediction — hold a clear sign"
        elif smoothed.stable_label is None:
            message = "Stabilising…"
        elif commit:
            message = f"Added {smoothed.stable_label!r}"
        elif self.gate.is_blocked(smoothed.stable_label):
            message = (
                f"{smoothed.stable_label!r} already added — lower hand to repeat"
            )
        else:
            message = f"Stable sign {smoothed.stable_label!r}"

        return SessionUpdate(
            hand_present=hand_present,
            raw_accepted=accepted,
            stable_label=smoothed.stable_label,
            stable_confidence=smoothed.stable_confidence,
            commit=commit,
            repeat_blocked=self.gate.is_blocked(smoothed.stable_label),
            window_filled=smoothed.window_filled,
            message=message,
            commit_progress=commit_progress,
            release_remaining=release_remaining,
            release_total=release_total,
        )

    def force_current_commit(self) -> str | None:
        label = self.smoother.current_stable
        if label is None:
            return None
        self.gate.note_manual_commit(label)
        return label

    def reset(self) -> None:
        self.smoother.reset()
        self.gate.reset()
