"""Model manager API tests — catalog operations only, no real downloads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


class TestListAndGet:
    def test_list_all_models(self, client: TestClient) -> None:
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert len(body) > 5
        assert all("installed" in m and "compatible" in m for m in body)

    def test_filter_by_kind(self, client: TestClient) -> None:
        r = client.get("/api/v1/models", params={"kind": "llm"})
        assert r.status_code == 200
        assert all(m["kind"] == "llm" for m in r.json())

    def test_get_single_model_card(self, client: TestClient) -> None:
        r = client.get("/api/v1/models/qwen3.5-4b-instruct-q4_k_m")
        assert r.status_code == 200
        body = r.json()
        assert body["provider"] == "Alibaba (Qwen)"
        assert body["installed"] is False

    def test_get_unknown_model_is_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/models/does-not-exist")
        assert r.status_code == 404
        assert r.json()["error_type"] == "RegistryError"

    def test_model_id_with_slash_resolves(self, client: TestClient) -> None:
        r = client.get("/api/v1/models/faster-whisper/small")
        assert r.status_code == 200
        assert r.json()["id"] == "faster-whisper/small"


class TestDownloadAndRemove:
    def test_download_starts_background_task(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[str] = []
        monkeypatch.setattr(
            "eva.server.state.ServerState.start_download", lambda self, mid: started.append(mid)
        )
        r = client.post("/api/v1/models/kokoro-82m-v1.0/download")
        assert r.status_code == 200
        assert r.json() == {"model_id": "kokoro-82m-v1.0", "status": "started"}
        assert started == ["kokoro-82m-v1.0"]

    def test_download_engine_managed_model_is_a_noop(self, client: TestClient) -> None:
        r = client.post("/api/v1/models/faster-whisper/small/download")
        assert r.status_code == 200
        assert r.json()["status"] == "not_applicable"

    def test_remove_uninstalled_model_is_safe(self, client: TestClient) -> None:
        r = client.delete("/api/v1/models/kokoro-82m-v1.0")
        assert r.status_code == 200

    def test_remove_bundled_model_rejected(self, client: TestClient) -> None:
        r = client.delete("/api/v1/models/silero-vad-v5")
        assert r.status_code == 502  # ModelError


class TestActivate:
    def test_activate_sets_model_and_switches_to_custom(self, client: TestClient) -> None:
        r = client.post("/api/v1/models/qwen3-1.7b-instruct-q4_k_m/activate")
        assert r.status_code == 200
        assert r.json()["active"] is True
        settings = client.get("/api/v1/settings").json()
        assert settings["llm"]["model"] == "qwen3-1.7b-instruct-q4_k_m"
        assert settings["profile"] == "custom"

    def test_activate_persists_across_requests(self, client: TestClient) -> None:
        client.post("/api/v1/models/faster-whisper/base/activate")
        settings = client.get("/api/v1/settings").json()
        assert settings["asr"]["model"] == "faster-whisper/base"
