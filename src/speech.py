"""Threaded text-to-speech using pyttsx3, with graceful degradation.

pyttsx3 and the Windows SAPI5 driver are not thread-safe in every situation, so
the engine is created and used inside one worker thread.  Flask request threads
only enqueue commands or request a stop.  If initialization fails (no audio
device, no eSpeak on Linux, broken driver), the service reports ``unavailable``
and the web UI can fall back to the browser's SpeechSynthesis API.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any


def _default_engine_factory() -> Any:
    import pyttsx3  # imported lazily so the rest of the app works without audio

    return pyttsx3.init()


class SpeechService:
    def __init__(
        self,
        *,
        enabled: bool = True,
        rate: int = 170,
        volume: float = 1.0,
        voice_hint: str | None = None,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.voice_hint = voice_hint
        self._engine_factory = engine_factory or _default_engine_factory

        self._commands: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._engine: Any | None = None
        self._state = "disabled" if not enabled else "stopped"
        self._error: str | None = None
        self._speaking = False
        self._last_text = ""
        self._voice_count = 0
        self._voices: list[dict[str, Any]] = []
        self._selected_voice: str | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if not self.enabled:
            with self._lock:
                self._state = "disabled"
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._state = "initializing"
            self._error = None
            self._thread = threading.Thread(
                target=self._worker, name="speech-worker", daemon=True
            )
            self._thread.start()

    def _worker(self) -> None:
        try:
            engine = self._engine_factory()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
            self._select_voice(engine)
            with self._lock:
                self._engine = engine
                self._state = "ready"
        except Exception as exc:  # broad: audio backends raise platform errors
            with self._lock:
                self._state = "unavailable"
                self._error = (
                    f"Text-to-speech engine could not start: {exc}. "
                    "Check your audio device or use the browser voice fallback."
                )
            return

        while True:
            try:
                command, payload = self._commands.get()
            except Exception:  # pragma: no cover - queue cannot realistically fail
                break
            if command == "shutdown":
                break
            if command == "stop":
                self._safe_stop()
                continue
            if command == "configure":
                self._apply_settings(engine, payload or {})
                continue
            if command == "speak":
                text = (payload or "").strip()
                if not text:
                    continue
                with self._lock:
                    self._speaking = True
                    self._last_text = text
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    with self._lock:
                        self._error = f"Speech failed: {exc}"
                finally:
                    with self._lock:
                        self._speaking = False

        self._safe_stop()
        with self._lock:
            self._engine = None
            if self._state != "unavailable":
                self._state = "stopped"

    def _select_voice(self, engine: Any) -> None:
        try:
            voices = engine.getProperty("voices") or []
            normalized: list[dict[str, Any]] = []
            for voice in voices:
                languages: list[str] = []
                for language in getattr(voice, "languages", []) or []:
                    if isinstance(language, bytes):
                        languages.append(language.decode("utf-8", errors="replace"))
                    else:
                        languages.append(str(language))
                normalized.append(
                    {
                        "id": str(getattr(voice, "id", "")),
                        "name": str(getattr(voice, "name", "Unknown voice")),
                        "languages": languages,
                    }
                )
            with self._lock:
                self._voice_count = len(voices)
                self._voices = normalized
            if not self.voice_hint:
                if normalized:
                    self._selected_voice = normalized[0]["id"]
                return
            hint = self.voice_hint.lower()
            for voice in normalized:
                haystack = f"{voice['id']} {voice['name']}".lower()
                if hint in haystack:
                    engine.setProperty("voice", voice["id"])
                    with self._lock:
                        self._selected_voice = voice["id"]
                    return
        except Exception:
            # Voice selection is optional; never fail startup because of it.
            return

    def _apply_settings(self, engine: Any, payload: dict[str, Any]) -> None:
        try:
            if "rate" in payload and payload["rate"] is not None:
                engine.setProperty("rate", int(payload["rate"]))
            if "volume" in payload and payload["volume"] is not None:
                engine.setProperty("volume", max(0.0, min(1.0, float(payload["volume"]))))
            voice = payload.get("voice")
            if voice:
                self.voice_hint = str(voice)
                self._select_voice(engine)
        except Exception as exc:  # pragma: no cover - backend dependent
            with self._lock:
                self._error = f"Could not apply speech settings: {exc}"

    def _safe_stop(self) -> None:
        engine = self._engine
        if engine is None:
            return
        try:
            engine.stop()
        except Exception as exc:  # pragma: no cover - backend dependent
            with self._lock:
                self._error = f"Could not stop speech cleanly: {exc}"

    # ------------------------------------------------------------------ commands
    def speak(self, text: str) -> tuple[bool, str]:
        normalized = " ".join(str(text).split())
        if not normalized:
            return False, "There is no text to speak."
        with self._lock:
            if not self.enabled:
                return False, "Server-side speech is disabled."
            if self._state == "initializing":
                return False, "Speech engine is still starting. Try again in a moment."
            if self._state != "ready":
                return False, self._error or "Speech engine is unavailable."
            if self._speaking and normalized == self._last_text:
                return False, "This text is already being spoken."

        self._discard_pending_speech()
        self._commands.put(("speak", normalized))
        return True, "Speaking."

    def _discard_pending_speech(self) -> None:
        """Drop queued speak commands while preserving stop/shutdown commands."""

        retained: list[tuple[str, str | None]] = []
        while True:
            try:
                item = self._commands.get_nowait()
            except queue.Empty:
                break
            if item[0] != "speak":
                retained.append(item)
        for item in retained:
            self._commands.put(item)

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self.enabled:
                return False, "Server-side speech is disabled."
            engine_available = self._engine is not None
            self._commands.put(("stop", None))
        if engine_available:
            self._safe_stop()
            return True, "Stopped."
        return False, "Speech engine is not running."

    def configure(
        self,
        *,
        enabled: bool | None = None,
        rate: int | None = None,
        volume: float | None = None,
        voice: str | None = None,
    ) -> tuple[bool, str]:
        """Apply voice settings without touching the pyttsx3 engine off-thread."""

        with self._lock:
            if enabled is not None:
                if enabled and not self.enabled:
                    self.enabled = True
                    if self._thread and self._thread.is_alive():
                        self._state = "ready" if self._engine is not None else "stopped"
                    else:
                        self._state = "stopped"
                elif not enabled and self.enabled:
                    self.enabled = False
                    self._state = "disabled"
                    self._speaking = False
                    self._commands.put(("stop", None))
            if rate is not None:
                self.rate = int(rate)
            if volume is not None:
                self.volume = max(0.0, min(1.0, float(volume)))
            if voice is not None:
                self.voice_hint = voice or None

            payload = {
                "rate": self.rate,
                "volume": self.volume,
                "voice": self.voice_hint,
            }
            if self.enabled:
                self._commands.put(("configure", payload))
            if self.enabled and not (self._thread and self._thread.is_alive()):
                self.start()

        return True, "Speech settings updated."

    def shutdown(self) -> None:
        self._commands.put(("shutdown", None))
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backend": "pyttsx3",
                "enabled": self.enabled,
                "state": self._state,
                "available": self._state == "ready",
                "speaking": self._speaking,
                "voice_count": self._voice_count,
                "voices": list(self._voices),
                "selected_voice": self._selected_voice,
                "rate": self.rate,
                "volume": self.volume,
                "error": self._error,
            }
