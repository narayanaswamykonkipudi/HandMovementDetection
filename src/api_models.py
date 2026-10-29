"""Pydantic v2 request/response contracts for the FastAPI application.

Every HTTP endpoint in ``app.py`` uses one of these models.  Keeping the
contracts here (instead of returning untyped dictionaries) gives the OpenAPI
schema at ``/docs`` real meaning and makes request validation errors
predictable for the browser client.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import MAX_TEXT_LENGTH


class ApiMessage(BaseModel):
    """Base success response used by simple actions."""

    ok: bool = True
    message: str = ""


class ErrorResponse(BaseModel):
    """Consistent error body for validation and HTTP errors."""

    ok: Literal[False] = False
    error: str
    message: str
    details: Any | None = None


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str
    status: str
    model_loaded: bool
    model_is_synthetic: bool
    camera_running: bool
    timestamp: str
    uptime_seconds: float


class VersionResponse(BaseModel):
    ok: bool = True
    name: str
    version: str
    api_version: str
    python_version: str
    platform: str
    model_format_version: int
    scope: str


class CameraStartRequest(BaseModel):
    index: int | None = Field(
        default=None, ge=0, le=10, description="OpenCV device index; omit to reuse current."
    )


class CameraDevice(BaseModel):
    index: int
    label: str
    current: bool = False


class CameraListResponse(ApiMessage):
    cameras: list[CameraDevice]
    current_index: int


class CameraActionResponse(ApiMessage):
    index: int
    camera_running: bool


class TextActionResponse(ApiMessage):
    text: str
    can_undo: bool = False
    at_capacity: bool = False


class SpeakRequest(BaseModel):
    text: str | None = Field(
        default=None,
        max_length=MAX_TEXT_LENGTH,
        description="Text to speak. Omit to speak the committed text buffer.",
    )


class VoiceSettingsRequest(BaseModel):
    rate: int | None = Field(default=None, ge=60, le=400)
    volume: float | None = Field(default=None, ge=0.0, le=1.0)
    voice: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None


class VoiceInfo(BaseModel):
    id: str
    name: str
    languages: list[str] = Field(default_factory=list)


class SpeechState(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = "pyttsx3"
    enabled: bool = True
    state: str = "unknown"
    available: bool = False
    speaking: bool = False
    voice_count: int = 0
    voices: list[VoiceInfo] = Field(default_factory=list)
    selected_voice: str | None = None
    rate: int = 170
    volume: float = 1.0
    error: str | None = None


class SpeechActionResponse(ApiMessage):
    speech: SpeechState


class TopPredictionModel(BaseModel):
    label: str
    probability: float


class ThresholdModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    min_confidence: float
    min_margin: float
    source: str | None = None
    status: str | None = None


class ScopeInfo(BaseModel):
    language: str
    signs: str
    excluded_dynamic: list[str]
    note: str


class ControllerInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    smoothing_window: int = 8
    smoothing_min_votes: int = 5
    release_frames: int = 8
    window_filled: int = 0
    commit_progress: float = 0.0
    release_remaining: int = 0


class LimitsInfo(BaseModel):
    max_text_length: int = MAX_TEXT_LENGTH


class ThresholdDefaults(BaseModel):
    min_confidence: float
    min_margin: float


class StateResponse(BaseModel):
    """Complete live state. Extra fields are preserved for forward compatibility."""

    model_config = ConfigDict(extra="allow")

    camera_running: bool = False
    camera_error: str | None = None
    camera_backend: str | None = None
    camera_index: int = 0
    camera_fps: float = 0.0
    frames_processed: int = 0
    model_error: str | None = None
    model_path: str | None = None
    model_loaded: bool = False
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    model_is_synthetic: bool = False
    text: str = ""
    can_undo: bool = False
    at_capacity: bool = False
    hand_present: bool = False
    handedness: str | None = None
    hand_score: float | None = None
    raw_label: str | None = None
    raw_confidence: float | None = None
    raw_accepted: bool = False
    stable_label: str | None = None
    stable_confidence: float | None = None
    repeat_blocked: bool = False
    message: str = "Idle"
    top_predictions: list[TopPredictionModel] = Field(default_factory=list)
    thresholds: ThresholdModel | dict[str, Any] | None = None
    notices: list[str] = Field(default_factory=list)
    last_frame_utc: str | None = None
    scope: ScopeInfo | None = None
    speech: SpeechState | None = None
    controller: ControllerInfo | None = None
    camera_thread_alive: bool = False
    threshold_defaults: ThresholdDefaults | None = None
    limits: LimitsInfo | None = None


class ModelReloadResponse(ApiMessage):
    model_loaded: bool
    model_metadata: dict[str, Any] = Field(default_factory=dict)


class ImagePredictionResponse(BaseModel):
    ok: bool = True
    hand_detected: bool
    label: str | None = None
    confidence: float | None = None
    margin: float | None = None
    accepted: bool = False
    top: list[TopPredictionModel] = Field(default_factory=list)
    thresholds: ThresholdModel | dict[str, Any] | None = None
    model_is_synthetic: bool = False
    message: str


class SuggestionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class SuggestionResponse(BaseModel):
    ok: bool = True
    configured: bool
    suggestions: list[str] = Field(default_factory=list)
    message: str
    disclaimer: str = (
        "Post-processing only. Suggestions are language-model guesses from committed "
        "letters, not ASL translation."
    )
