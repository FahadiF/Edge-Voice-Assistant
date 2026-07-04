"""Engine lifecycle + conversation API tests, using fake engines (server_fakes)
so the running-engine path is covered without real models or audio hardware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.server_fakes import build_fake_assistant

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("eva.engine.build_assistant", build_fake_assistant)
    return TestClient(create_app(app_paths))


class TestReadinessAndStartStop:
    def test_readiness_false_without_models(self, client: TestClient) -> None:
        r = client.get("/api/v1/engine/readiness")
        assert r.status_code == 200
        assert r.json()["ready"] is False

    def test_start_blocked_when_not_ready(self, client: TestClient) -> None:
        r = client.post("/api/v1/engine/start")
        assert r.status_code == 409
        assert "problems" in r.json()["detail"]

    def test_start_stop_with_fake_engine(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "eva.onboarding.check_readiness",
            lambda settings, paths: [],
        )
        r = client.post("/api/v1/engine/start")
        assert r.status_code == 200
        assert r.json()["running"] is True

        status = client.get("/api/v1/engine/status").json()
        assert status["running"] is True

        r = client.post("/api/v1/engine/stop")
        assert r.status_code == 200
        assert r.json()["running"] is False
        assert client.get("/api/v1/engine/status").json()["running"] is False

    def test_start_twice_is_idempotent(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        client.post("/api/v1/engine/start")
        r = client.post("/api/v1/engine/start")
        assert r.status_code == 200
        assert r.json()["running"] is True
        client.post("/api/v1/engine/stop")


class TestConversationWithoutEngine:
    def test_history_requires_engine(self, client: TestClient) -> None:
        r = client.get("/api/v1/conversation/history")
        assert r.status_code == 409

    def test_interrupt_requires_engine(self, client: TestClient) -> None:
        r = client.post("/api/v1/conversation/interrupt")
        assert r.status_code == 409

    def test_export_requires_engine(self, client: TestClient) -> None:
        r = client.get("/api/v1/conversation/export")
        assert r.status_code == 409


@pytest.fixture
def running_client(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
    client.post("/api/v1/engine/start")
    yield client
    client.post("/api/v1/engine/stop")  # avoid leaking the orchestrator task past the test


class TestConversationWithEngine:
    def test_history_starts_empty(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/conversation/history")
        assert r.status_code == 200
        assert r.json() == []

    def test_interrupt_with_nothing_running_is_false(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/conversation/interrupt")
        assert r.status_code == 200
        assert r.json() == {"interrupted": False}

    def test_cancel_is_an_alias_of_interrupt(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/conversation/cancel")
        assert r.status_code == 200
        assert "interrupted" in r.json()

    def test_clear_history(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/conversation/clear")
        assert r.status_code == 200

    def test_export_shape(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/conversation/export")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == 1
        assert body["turns"] == []
        assert "exported_at" in body

    def test_import_then_export_round_trips(self, running_client: TestClient) -> None:
        payload = {"turns": [{"user": "hi", "assistant": "hello"}]}
        r = running_client.post("/api/v1/conversation/import", json=payload)
        assert r.status_code == 200
        exported = running_client.get("/api/v1/conversation/export").json()
        assert exported["turns"] == [{"user": "hi", "assistant": "hello"}]

    def test_current_turn_reports_state(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/conversation/current")
        assert r.status_code == 200
        assert r.json()["state"] in ("idle", "listening", "thinking", "speaking")

    def test_diagnostics_reflect_running_engine(self, running_client: TestClient) -> None:
        body = running_client.get("/api/v1/diagnostics").json()
        assert body["devices"] == {"llm": "cpu", "asr": "cpu", "tts": "cpu", "vad": "cpu"}
