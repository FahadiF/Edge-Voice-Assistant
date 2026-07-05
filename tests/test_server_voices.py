"""Voice API tests (M4, ADR-022) — fake engine (server_fakes), no real
models or audio hardware. Requires a running engine (voices are populated
from the loaded TTS engine's capability discovery during preload).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app
from tests.server_fakes import build_fake_assistant


@pytest.fixture
def client(app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("eva.engine.build_assistant", build_fake_assistant)
    return TestClient(create_app(app_paths))


@pytest.fixture
def running_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
    client.post("/api/v1/engine/start")
    yield client
    client.post("/api/v1/engine/stop")


class TestRequiresEngine:
    def test_list_requires_engine(self, client: TestClient) -> None:
        assert client.get("/api/v1/voices").status_code == 409

    def test_preview_requires_engine(self, client: TestClient) -> None:
        r = client.post("/api/v1/voices/test-voice/preview", json={})
        assert r.status_code == 409


class TestListVoices:
    def test_list_returns_engine_voices(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/voices")
        assert r.status_code == 200
        ids = {v["id"] for v in r.json()}
        assert "test-voice" in ids

    def test_list_reflects_active_engine(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/voices")
        for voice in r.json():
            assert voice["engine"] == "kokoro"  # default settings.tts.engine


class TestPreview:
    def test_preview_returns_pcm_bytes(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/voices/test-voice/preview", json={})
        assert r.status_code == 200
        assert len(r.content) > 0
        assert r.headers["content-type"] == "application/octet-stream"

    def test_preview_accepts_custom_phrase(self, running_client: TestClient) -> None:
        r = running_client.post(
            "/api/v1/voices/test-voice/preview", json={"phrase": "Testing 123"}
        )
        assert r.status_code == 200
