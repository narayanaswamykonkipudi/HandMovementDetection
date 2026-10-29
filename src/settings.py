"""Typed application settings loaded from environment variables and ``.env``.

All settings use the ``ASL_`` prefix.  For example::

    ASL_PORT=9000
    ASL_MODEL_PATH=models/classifier.joblib
    ASL_SPEECH_ENABLED=false

Command-line flags in ``app.py`` override these values, but the FastAPI app
factory remains usable on its own::

    uvicorn app:app --host 127.0.0.1 --port 8501

The settings object deliberately contains no web-framework objects so it can be
imported by CLIs and tests without starting a server.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import (
    CLASSIFIER_PATH,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEMO_CLASSIFIER_PATH,
    PROJECT_ROOT,
)


class Settings(BaseSettings):
    """Runtime configuration with validation and environment support."""

    model_config = SettingsConfigDict(
        env_prefix="ASL_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server ---------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=8501, ge=1, le=65535)
    log_level: str = "INFO"
    cors_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    # Camera ---------------------------------------------------------------
    camera_index: int = Field(default=DEFAULT_CAMERA_INDEX, ge=0, le=10)
    frame_width: int = Field(default=DEFAULT_FRAME_WIDTH, ge=320, le=3840)
    frame_height: int = Field(default=DEFAULT_FRAME_HEIGHT, ge=240, le=2160)
    mirror: bool = True
    autostart_camera: bool = False
    camera_scan_max: int = Field(default=5, ge=1, le=10)

    # Model ----------------------------------------------------------------
    model_path: Path | None = None
    demo: bool = False
    reload_on_startup: bool = True

    # Speech ---------------------------------------------------------------
    speech_enabled: bool = True
    speech_rate: int = Field(default=170, ge=60, le=400)
    speech_volume: float = Field(default=1.0, ge=0.0, le=1.0)
    speech_voice: str | None = None

    # Optional LLM post-processing (never receives video or landmarks) ------
    llm_enabled: bool = False
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_seconds: float = Field(default=12.0, ge=1.0, le=60.0)

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        normalized = str(value).upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    def resolved_model_path(self) -> Path | None:
        """Return the explicit model, the demo model, or the default model."""

        if self.model_path:
            candidate = Path(self.model_path).expanduser()
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if self.demo:
            return DEMO_CLASSIFIER_PATH
        return CLASSIFIER_PATH

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_enabled and self.llm_api_key)


def get_settings(**overrides: object) -> Settings:
    """Build settings, applying explicit command-line overrides."""

    return Settings(**overrides)
