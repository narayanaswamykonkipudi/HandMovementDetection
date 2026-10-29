from __future__ import annotations

from fastapi.testclient import TestClient

from app import create_app
from src.settings import Settings
from tests.conftest import FakeRuntime


def test_pages_and_static_assets(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert "Sign to Speech" in page.text
    assert "Synthetic" in page.text or "demo" in page.text.lower()
    assert "https://cdn" not in page.text
    favicon = client.get("/static/favicon.svg")
    assert favicon.status_code == 200
    assert "svg" in favicon.headers["content-type"]


def test_health_version_and_state(client: TestClient):
    health = client.get("/api/health")
    assert health.status_code == 200
    body = health.json()
    assert body["ok"] is True
    assert body["model_loaded"] is True
    assert body["model_is_synthetic"] is True
    assert body["version"]
    version = client.get("/api/version").json()
    assert version["api_version"]
    assert "static fingerspelling" in version["scope"].lower()
    state = client.get("/api/state").json()
    assert state["model_loaded"] is True
    assert state["speech"]["enabled"] is False
    assert state["controller"]["release_total"] == 8


def test_text_actions_work_through_api(client: TestClient):
    assert client.post("/api/text/add").json()["text"] == "A"
    assert client.post("/api/text/space").json()["text"] == "A "
    assert client.post("/api/text/delete").json()["text"] == "A"
    undo = client.post("/api/text/undo").json()
    assert undo["text"] == "A "
    assert undo["can_undo"] is True
    cleared = client.post("/api/text/clear").json()
    assert cleared["text"] == ""
    assert cleared["ok"] is True
    assert client.post("/api/text/clear").json()["ok"] is False


def test_camera_endpoints_and_validation(client: TestClient, fake_runtime):
    cameras = client.get("/api/cameras")
    assert cameras.status_code == 200
    assert cameras.json()["ok"] is True
    started = client.post("/api/camera/start", json={"index": 2})
    assert started.status_code == 200
    assert fake_runtime.started_camera == 2
    stopped = client.post("/api/camera/stop")
    assert stopped.status_code == 200
    assert fake_runtime.stop_camera_called is True
    invalid = client.post("/api/camera/start", json={"index": -1})
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"


def test_model_reload_uses_runtime(client: TestClient, fake_runtime):
    response = client.post("/api/model/reload")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert fake_runtime.loaded is True


def test_speech_routes_use_runtime(client: TestClient):
    spoken = client.post("/api/speech/speak", json={"text": "HELLO"})
    assert spoken.status_code == 200
    assert spoken.json()["ok"] is True
    stopped = client.post("/api/speech/stop")
    assert stopped.json()["ok"] is True
    settings = client.post("/api/speech/settings", json={"rate": 180, "volume": 0.4})
    assert settings.status_code == 200
    assert settings.json()["speech"]["rate"] == 180
    invalid = client.post("/api/speech/settings", json={"rate": 999})
    assert invalid.status_code == 422


def test_image_upload_validation_and_success(client: TestClient):
    missing = client.post("/api/predict/image")
    assert missing.status_code == 422
    empty = client.post(
        "/api/predict/image", files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert empty.status_code == 422
    assert empty.json()["error"] == "empty_file"
    large = client.post(
        "/api/predict/image",
        files={"file": ("large.jpg", b"x" * (8 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert large.status_code == 413
    assert large.json()["error"] == "file_too_large"
    success = client.post(
        "/api/predict/image", files={"file": ("hand.jpg", b"fake bytes", "image/jpeg")}
    )
    assert success.status_code == 200
    payload = success.json()
    assert payload["hand_detected"] is True
    assert payload["label"] == "A"
    assert payload["model_is_synthetic"] is True


def test_llm_suggestion_is_gracefully_disabled(client: TestClient):
    response = client.post("/api/text/suggest", json={"text": "HE LLO"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["configured"] is False
    assert payload["suggestions"] == []
    assert "disabled" in payload["message"].lower()
    assert "not asl translation" in payload["disclaimer"].lower()
    invalid = client.post("/api/text/suggest", json={"text": ""})
    assert invalid.status_code == 422


def test_docs_and_custom_404(client: TestClient):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


def test_localhost_cors_preflight(client: TestClient):
    response = client.options(
        "/api/state",
        headers={
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"


def test_configured_llm_suggestions_reach_response(monkeypatch):
    runtime_settings = Settings(
        speech_enabled=False,
        llm_enabled=False,
        demo=False,
        reload_on_startup=False,
    )
    application = create_app(
        runtime=FakeRuntime(),
        settings=runtime_settings,
        load_model_on_startup=False,
    )
    # Endpoint-level configuration is covered here without network access.
    import app as app_module

    class Result:
        configured = True
        suggestions = ["HELLO"]
        message = "Received 1 post-processing suggestion."

    monkeypatch.setattr(app_module, "suggest_text", lambda *args, **kwargs: Result())
    with TestClient(application) as test_client:
        response = test_client.post("/api/text/suggest", json={"text": "HE LLO"})
    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert response.json()["suggestions"] == ["HELLO"]
