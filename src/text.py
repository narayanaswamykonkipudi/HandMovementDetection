"""Committed-text construction with undo support.

The recogniser emits static letters.  This module deliberately does not add
automatic spaces or guess words: spacing is an explicit user decision, and
inserting guessed spaces would make recognition errors harder to correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MAX_TEXT_LENGTH, STATIC_ASL_LETTERS


class TextError(ValueError):
    """Raised for invalid direct text operations."""


@dataclass(frozen=True)
class TextSnapshot:
    text: str
    length: int
    can_undo: bool
    at_capacity: bool


class TextBuffer:
    def __init__(self, max_length: int = MAX_TEXT_LENGTH) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        self.max_length = max_length
        self._text = ""
        self._history: list[str] = []
        self._max_history = 100

    @property
    def text(self) -> str:
        return self._text

    def _remember(self) -> None:
        self._history.append(self._text)
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def add_letter(self, label: str) -> bool:
        label = str(label).strip().upper()
        if len(label) != 1 or label not in STATIC_ASL_LETTERS:
            raise TextError(f"Not a supported static ASL letter: {label!r}")
        if len(self._text) >= self.max_length:
            return False
        self._remember()
        self._text += label
        return True

    def add_space(self) -> bool:
        if not self._text or len(self._text) >= self.max_length:
            return False
        if self._text.endswith(" "):
            return False
        self._remember()
        self._text += " "
        return True

    def backspace(self) -> bool:
        if not self._text:
            return False
        self._remember()
        self._text = self._text[:-1]
        return True

    def clear(self) -> bool:
        if not self._text:
            return False
        self._remember()
        self._text = ""
        return True

    def undo(self) -> bool:
        if not self._history:
            return False
        self._text = self._history.pop()
        return True

    def set_text(self, value: str) -> None:
        """Replace text (used only by tests/administrative restore)."""

        value = str(value)
        if len(value) > self.max_length:
            raise TextError("Text exceeds the configured maximum length.")
        self._remember()
        self._text = value

    def snapshot(self) -> TextSnapshot:
        return TextSnapshot(
            text=self._text,
            length=len(self._text),
            can_undo=bool(self._history),
            at_capacity=len(self._text) >= self.max_length,
        )
