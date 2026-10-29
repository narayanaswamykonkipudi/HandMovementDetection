#!/usr/bin/env python
"""FastAPI application for real-time ASL static-fingerspelling recognition.

Run with uvicorn::

    uvicorn app:app --host 127.0.0.1 --port 8501

Or use the compatibility wrapper, which keeps the original flags::

    python app.py --demo
    python app.py --model models/classifier.joblib --autostart

The server binds to localhost by default. Webcam frames are processed locally
and are never uploaded; only local JPEG preview frames and numeric state are
served to the browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import __version__
from src.api_models import (
    ApiMessage,
    CameraActionResponse,
    CameraDevice,
    CameraListResponse,
    CameraStartRequest,
    ControllerInfo,
    ErrorResponse,
    HealthResponse,
    ImagePredictionResponse,
    ModelReloadResponse,
    ScopeInfo,
    SpeakRequest,
    SpeechActionResponse,
    SpeechState,
    StateResponse,
    SuggestionRequest,
    SuggestionResponse,
    TextActionResponse,
    ThresholdModel,
    VersionResponse,
    VoiceSettingsRequest,
)
from src.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_MARGIN,
    DEMO_CLASSIFIER_PATH,
    MAX_TEXT_LENGTH,
    MODEL_FORMAT_VERSION,
    STATIC_ASL_LETTERS,
)
from src.llm import LLMError, suggest_text
from src.runtime import SignLanguageRuntime
from src.settings import Settings, get_settings

LOGGER = logging.getLogger("asl.app")
API_VERSION = "2.0"
PLACEHOLDER_SIZE = (960, 540)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_logging(level: str = "INFO") -> None:
    """Configure concise, key-value structured logs for the service."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _placeholder(message: str) -> bytes:
    """Create a JPEG placeholder so the preview frame is never broken."""

    import cv2
    import numpy as np

    width, height = PLACEHOLDER_SIZE
    canvas = np.full((height, width, 3), (28, 24, 22), dtype=np.uint8)
    cv2.circle(canvas, (width // 2, 205), 58, (60, 54, 50), -1, cv2.LINE_AA)
    cv2.circle(canvas, (width // 2, 205), 58, (90, 82, 76), 2, cv2.LINE_AA)
    cv2.line(canvas, (width // 2 - 34, 205), (width // 2 + 34, 205), (120, 112, 104), 3)
    cv2.putText(
        canvas,
        "Camera preview",
        (width // 2 - 100, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 195, 188),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message[:80],
        (max(20, width // 2 - min(360, len(message) * 7)), 365),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (140, 135, 128),
        1,
        cv2.LINE_AA,
    )
    ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buffer.tobytes() if ok else b""


_PLACEHOLDER_CACHE: dict[str, bytes] = {}


def placeholder_bytes(message: str) -> bytes:
    if message not in _PLACEHOLDER_CACHE:
        _PLACEHOLDER_CACHE[message] = _placeholder(message)
    return _PLACEHOLDER_CACHE[message]


def build_runtime(settings: Settings) -> SignLanguageRuntime:
    model_path = settings.resolved_model_path()
    return SignLanguageRuntime(
        model_path=model_path,
        camera_index=settings.camera_index,
        width=settings.frame_width,
        height=settings.frame_height,
        mirror=settings.mirror,
        speech_enabled=settings.speech_enabled,
        speech_rate=settings.speech_rate,
        speech_volume=settings.speech_volume,
        voice_hint=settings.speech_voice,
    )


def _error_detail(error: str, message: str, details: Any | None = None) -> dict[str, Any]:
    return {"error": error, "message": message, "details": details}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(
    runtime: SignLanguageRuntime | None = None,
    *,
    settings: Settings | None = None,
    load_model_on_startup: bool = True,
) -> FastAPI:
    """Build the FastAPI app.

    ``runtime`` can be replaced by a fake object in tests.  The lifespan handler
    owns speech startup, optional model load, camera autostart, and clean
    shutdown, so no worker threads leak between tests.
    """

    settings = settings or get_settings()
    runtime = runtime or build_runtime(settings)
    started_at = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        LOGGER.info(
            "event=startup model=%s demo=%s camera_autostart=%s speech=%s",
            settings.resolved_model_path(),
            settings.demo,
            settings.autostart_camera,
            settings.speech_enabled,
        )
        runtime.speech.start()
        if load_model_on_startup and settings.reload_on_startup:
            runtime.load_model()
        if settings.autostart_camera:
            started, message = runtime.start_camera(settings.camera_index)
            if not started:
                LOGGER.warning("event=camera_autostart_failed message=%r", message)
        try:
            yield
        finally:
            LOGGER.info("event=shutdown")
            runtime.shutdown()

    app = FastAPI(
        title="Sign to Speech — ASL Static Fingerspelling API",
        description=(
            "Local FastAPI service for 24 static ASL fingerspelling handshapes. "
            "J and Z require motion and are out of scope. This is not a sign-language "
            "translator. Synthetic demo models are labelled in every response."
        ),
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.runtime = runtime
    app.state.settings = settings
    app.state.started_at = started_at

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        if request.url.path != "/video_feed":
            LOGGER.info(
                "event=http_request method=%s path=%s status=%s duration_ms=%.1f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        return response

    templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))
    app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

    def current_runtime(request: Request) -> SignLanguageRuntime:
        return request.app.state.runtime

    def state_payload(request: Request) -> dict[str, Any]:
        return current_runtime(request).status_dict()

    # ------------------------------------------------------------------ errors
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        from fastapi.encoders import jsonable_encoder

        LOGGER.info("event=validation_error path=%s", request.url.path)
        payload = ErrorResponse(
            error="validation_error",
            message="Request validation failed.",
            details=jsonable_encoder(exc.errors()),
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            error = str(detail.get("error", "http_error"))
            message = str(detail.get("message", detail))
            details = detail.get("details")
        else:
            error = "http_error" if exc.status_code != 404 else "not_found"
            message = str(detail)
            details = None
        payload = ErrorResponse(error=error, message=message, details=details)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    # -------------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "static_letters": "".join(STATIC_ASL_LETTERS),
                "max_text_length": MAX_TEXT_LENGTH,
                "api_version": API_VERSION,
                "app_version": __version__,
            },
        )

    # ------------------------------------------------------------------- state
    @app.get("/api/state", response_model=StateResponse, tags=["state"])
    async def api_state(request: Request) -> StateResponse:
        return StateResponse.model_validate(state_payload(request))

    @app.get("/api/events", tags=["state"])
    async def api_events(request: Request) -> StreamingResponse:
        """Server-Sent Events stream of state snapshots (SSE, no polling)."""

        async def event_stream() -> AsyncIterator[str]:
            yield "retry: 2000\n\n"
            last_payload: str | None = None
            last_keepalive = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    state = StateResponse.model_validate(
                        current_runtime(request).status_dict()
                    ).model_dump(mode="json")
                    payload = json.dumps(state, separators=(",", ":"), default=str)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("event=sse_serialization_error error=%s", exc)
                    payload = json.dumps(
                        {"ok": False, "message": "State serialization failed."}
                    )
                if payload != last_payload:
                    yield f"event: state\ndata: {payload}\n\n"
                    last_payload = payload
                    last_keepalive = time.monotonic()
                elif time.monotonic() - last_keepalive > 15:
                    yield ": keepalive\n\n"
                    last_keepalive = time.monotonic()
                await asyncio.sleep(0.2)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    async def api_health(request: Request) -> HealthResponse:
        status = state_payload(request)
        return HealthResponse(
            ok=True,
            service="asl-fingerspelling",
            version=__version__,
            status="ok" if status.get("model_loaded") else "degraded",
            model_loaded=bool(status.get("model_loaded")),
            model_is_synthetic=bool(status.get("model_is_synthetic")),
            camera_running=bool(status.get("camera_running")),
            timestamp=utc_now_iso(),
            uptime_seconds=round(time.monotonic() - started_at, 2),
        )

    @app.get("/api/version", response_model=VersionResponse, tags=["health"])
    async def api_version() -> VersionResponse:
        return VersionResponse(
            name="Sign to Speech — ASL Static Fingerspelling",
            version=__version__,
            api_version=API_VERSION,
            python_version=platform.python_version(),
            platform=platform.platform(),
            model_format_version=MODEL_FORMAT_VERSION,
            scope="ASL static fingerspelling; 24 letters; J and Z excluded; not a translator.",
        )

    # ------------------------------------------------------------------ camera
    @app.get("/api/cameras", response_model=CameraListResponse, tags=["camera"])
    async def api_cameras(request: Request) -> CameraListResponse:
        active = current_runtime(request)
        return CameraListResponse(
            message=(
                "Server-side device indices. Browser labels are shown where the browser "
                "exposes them after camera permission is granted."
            ),
            cameras=[
                CameraDevice(
                    index=index,
                    label=f"Camera {index}",
                    current=index == active.camera_index,
                )
                for index in range(settings.camera_scan_max + 1)
            ],
            current_index=active.camera_index,
        )

    @app.post("/api/camera/start", response_model=CameraActionResponse, tags=["camera"])
    async def api_camera_start(
        request: Request, body: CameraStartRequest | None = None
    ) -> CameraActionResponse:
        active = current_runtime(request)
        index = active.camera_index if body is None or body.index is None else body.index
        started, message = active.start_camera(index)
        if not started:
            raise HTTPException(409, _error_detail("camera_unavailable", message))
        return CameraActionResponse(
            message=message, index=index, camera_running=True
        )

    @app.post("/api/camera/stop", response_model=ApiMessage, tags=["camera"])
    async def api_camera_stop(request: Request) -> ApiMessage:
        current_runtime(request).stop_camera()
        return ApiMessage(message="Camera stopped.")

    # ------------------------------------------------------------------- model
    @app.post("/api/model/reload", response_model=ModelReloadResponse, tags=["model"])
    async def api_model_reload(request: Request) -> ModelReloadResponse:
        active = current_runtime(request)
        if not active.load_model():
            status = active.status_dict()
            raise HTTPException(
                409,
                _error_detail(
                    "model_reload_failed",
                    status.get("model_error") or "Could not reload model.",
                ),
            )
        status = active.status_dict()
        return ModelReloadResponse(
            message="Model reloaded.",
            model_loaded=bool(status.get("model_loaded")),
            model_metadata=status.get("model_metadata", {}),
        )

    # -------------------------------------------------------------------- text
    def _text_response(active: SignLanguageRuntime, ok: bool, message: str) -> TextActionResponse:
        snapshot = active.text.snapshot()
        return TextActionResponse(
            ok=ok,
            message=message,
            text=snapshot.text,
            can_undo=snapshot.can_undo,
            at_capacity=snapshot.at_capacity,
        )

    @app.post("/api/text/add", response_model=TextActionResponse, tags=["text"])
    async def api_text_add(request: Request) -> TextActionResponse:
        active = current_runtime(request)
        added, message = active.add_current()
        if not added:
            raise HTTPException(409, _error_detail("no_stable_sign", message))
        return _text_response(active, True, message)

    @app.post("/api/text/space", response_model=TextActionResponse, tags=["text"])
    async def api_text_space(request: Request) -> TextActionResponse:
        active = current_runtime(request)
        changed, message = active.add_space()
        return _text_response(active, changed, message)

    @app.post("/api/text/delete", response_model=TextActionResponse, tags=["text"])
    async def api_text_delete(request: Request) -> TextActionResponse:
        active = current_runtime(request)
        changed, message = active.backspace()
        return _text_response(active, changed, message)

    @app.post("/api/text/clear", response_model=TextActionResponse, tags=["text"])
    async def api_text_clear(request: Request) -> TextActionResponse:
        active = current_runtime(request)
        changed, message = active.clear_text()
        return _text_response(active, changed, message)

    @app.post("/api/text/undo", response_model=TextActionResponse, tags=["text"])
    async def api_text_undo(request: Request) -> TextActionResponse:
        active = current_runtime(request)
        changed, message = active.undo()
        return _text_response(active, changed, message)

    @app.post("/api/text/suggest", response_model=SuggestionResponse, tags=["text"])
    async def api_text_suggest(
        request: Request, body: SuggestionRequest
    ) -> SuggestionResponse:
        active_settings = request.app.state.settings
        try:
            result = await asyncio.to_thread(suggest_text, body.text, active_settings)
        except LLMError as exc:
            LOGGER.warning("llm_postprocess_failed error=%s", type(exc.__cause__).__name__)
            raise HTTPException(
                502, _error_detail("llm_request_failed", "The optional LLM request failed.")
            ) from exc
        return SuggestionResponse(
            configured=result.configured,
            suggestions=result.suggestions,
            message=result.message,
        )

    # ------------------------------------------------------------------ speech
    @app.post("/api/speech/speak", response_model=SpeechActionResponse, tags=["speech"])
    async def api_speech_speak(
        request: Request, body: SpeakRequest | None = None
    ) -> SpeechActionResponse:
        active = current_runtime(request)
        text = None if body is None else body.text
        spoken, message = active.speak(text)
        if not spoken:
            raise HTTPException(503, _error_detail("speech_unavailable", message))
        return SpeechActionResponse(
            message=message, speech=SpeechState.model_validate(active.speech.status())
        )

    @app.post("/api/speech/stop", response_model=SpeechActionResponse, tags=["speech"])
    async def api_speech_stop(request: Request) -> SpeechActionResponse:
        active = current_runtime(request)
        _, message = active.stop_speaking()
        return SpeechActionResponse(
            message=message, speech=SpeechState.model_validate(active.speech.status())
        )

    @app.post(
        "/api/speech/settings", response_model=SpeechActionResponse, tags=["speech"]
    )
    async def api_speech_settings(
        request: Request, body: VoiceSettingsRequest
    ) -> SpeechActionResponse:
        active = current_runtime(request)
        ok, message = active.configure_speech(
            enabled=body.enabled,
            rate=body.rate,
            volume=body.volume,
            voice=body.voice,
        )
        if not ok:
            raise HTTPException(409, _error_detail("speech_settings_failed", message))
        return SpeechActionResponse(
            message=message, speech=SpeechState.model_validate(active.speech.status())
        )

    # ------------------------------------------------------------------ stream
    @app.get("/video_feed", include_in_schema=False)
    async def video_feed(request: Request) -> StreamingResponse:
        def generate() -> Any:
            while True:
                jpeg = current_runtime(request).latest_frame()
                if jpeg is None:
                    status = current_runtime(request).status_dict()
                    message = (
                        "Click Start camera to begin."
                        if not status.get("camera_error")
                        else "Camera error — see the message below."
                    )
                    jpeg = placeholder_bytes(message)
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                time.sleep(0.033)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # --------------------------------------------------------------- image API
    @app.post(
        "/api/predict/image",
        response_model=ImagePredictionResponse,
        tags=["prediction"],
    )
    async def api_predict_image(request: Request) -> ImagePredictionResponse:
        """Predict from one uploaded image (deployment/test fallback, no webcam).

        The image is processed in memory and is never written to disk.
        """

        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(
                422,
                _error_detail("missing_file", "Upload a JPEG/PNG image in form field 'file'."),
            )
        raw = await upload.read()  # type: ignore[union-attr]
        if not raw:
            raise HTTPException(422, _error_detail("empty_file", "Uploaded file is empty."))
        if len(raw) > 8 * 1024 * 1024:
            raise HTTPException(413, _error_detail("file_too_large", "Image exceeds 8 MB."))
        return current_runtime(request).predict_image_bytes(raw)

    return app


# The module-level ASGI object is what ``uvicorn app:app`` serves.
app = create_app()


# ---------------------------------------------------------------------------
# Compatibility CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="Trained model bundle.")
    parser.add_argument(
        "--demo", action="store_true", help="Use the synthetic demo model (clearly marked)."
    )
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--autostart", action="store_true", help="Open the camera on launch.")
    parser.add_argument("--no-camera", action="store_true", help="Never auto-open the camera.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> Settings:
    if args.model and args.demo:
        raise SystemExit("ERROR: use --model or --demo, not both.")
    overrides: dict[str, Any] = {"log_level": "DEBUG" if args.debug else "INFO"}
    if args.demo:
        overrides["demo"] = True
        overrides["model_path"] = None
    elif args.model:
        overrides["model_path"] = args.model
        overrides["demo"] = False
    if args.camera is not None:
        overrides["camera_index"] = args.camera
    if args.width is not None:
        overrides["frame_width"] = args.width
    if args.height is not None:
        overrides["frame_height"] = args.height
    if args.no_mirror:
        overrides["mirror"] = False
    if args.no_speech:
        overrides["speech_enabled"] = False
    if args.autostart:
        overrides["autostart_camera"] = True
    if args.no_camera:
        overrides["autostart_camera"] = False
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    return Settings(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = settings_from_args(args)
    except SystemExit as exc:
        LOGGER.error("event=argument_error message=%s", exc)
        return 2

    configure_logging(settings.log_level)
    runtime = build_runtime(settings)
    application = create_app(runtime, settings=settings)

    model_path = settings.resolved_model_path()
    LOGGER.info("event=banner")
    LOGGER.info("ASL static fingerspelling recogniser (FastAPI)")
    LOGGER.info("Scope: 24 static ASL letters; J and Z are dynamic and NOT supported.")
    LOGGER.info("This is a fingerspelling demo, not a sign-language translator.")
    LOGGER.info("Model: %s", model_path)
    if settings.demo:
        LOGGER.warning(
            "DEMO model trained on synthetic landmarks. Predictions are not valid ASL "
            "recognition. Build real data before drawing accuracy conclusions."
        )
    if model_path == DEMO_CLASSIFIER_PATH and not model_path.exists():
        LOGGER.warning("Demo model missing. Build it with: python scripts/build_demo.py")
    LOGGER.info("Open http://%s:%s  (API docs: /docs)", settings.host, settings.port)

    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
