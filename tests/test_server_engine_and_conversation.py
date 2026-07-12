"""Engine lifecycle + conversation API tests, using fake engines (server_fakes)
so the running-engine path is covered without real models or audio hardware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app
from tests.server_fakes import build_fake_assistant


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

    def test_say_requires_engine(self, client: TestClient) -> None:
        r = client.post("/api/v1/conversation/say", json={"text": "hello"})
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

    def test_say_accepts_typed_text(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M5.3 composer endpoint: text enters the orchestrator's event
        queue like an utterance. Needs the context-manager TestClient so all
        requests share one event loop — the orchestrator's run() task must
        still be alive when /say arrives (in production uvicorn has a single
        long-lived loop, so this is purely a test-harness concern)."""
        from tests.server_fakes import build_fake_assistant

        monkeypatch.setattr("eva.engine.build_assistant", build_fake_assistant)
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        with TestClient(create_app(app_paths)) as shared_loop_client:
            assert shared_loop_client.post("/api/v1/engine/start").status_code == 200
            r = shared_loop_client.post("/api/v1/conversation/say", json={"text": "hello there"})
            assert r.status_code == 200
            assert r.json() == {"status": "accepted"}
            shared_loop_client.post("/api/v1/engine/stop")

    def test_say_rejects_empty_text(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/conversation/say", json={"text": ""})
        assert r.status_code == 422  # SayRequest min_length=1

    def test_websocket_disconnect_leaves_engine_running(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M5.5 §10: a browser closing its tab must never take the engine
        down — the WebSocket is an observer, not a lifeline."""
        monkeypatch.setattr("eva.engine.build_assistant", build_fake_assistant)
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        with TestClient(create_app(app_paths)) as shared:
            assert shared.post("/api/v1/engine/start").status_code == 200
            with shared.websocket_connect("/api/v1/ws") as ws:
                snapshot = ws.receive_json()
                assert snapshot["type"] == "snapshot"
            # WebSocket closed (browser gone) — the engine must be unaffected.
            status = shared.get("/api/v1/engine/status").json()
            assert status["running"] is True
            shared.post("/api/v1/engine/stop")

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


class TestSystemShutdown:
    """POST /system/shutdown (M5.6): the graceful remote-stop path used by
    `eva stop` — the engine stops first, then uvicorn's exit hook fires."""

    def test_shutdown_without_hook_is_503(self, client: TestClient) -> None:
        r = client.post("/api/v1/system/shutdown")
        assert r.status_code == 503
        assert "shutdown hook" in r.json()["detail"]

    def test_shutdown_stops_engine_then_fires_hook(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        client.post("/api/v1/engine/start")
        assert client.get("/api/v1/engine/status").json()["running"] is True

        fired: list[bool] = []
        client.app.state.eva.shutdown_callback = lambda: fired.append(True)
        r = client.post("/api/v1/system/shutdown")
        assert r.status_code == 200
        assert r.json() == {"status": "shutting down"}
        assert fired == [True]
        # The engine was stopped before the process-exit hook fired.
        assert client.get("/api/v1/engine/status").json()["running"] is False


class TestResumeConversation:
    """POST /conversation/resume (M5.6): reopen a stored conversation where
    it ended — id, turns, summary linkage, and title all preserved."""

    def _start(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        assert client.post("/api/v1/engine/start").status_code == 200

    def test_resume_restores_history_and_continues_same_id(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._start(client, monkeypatch)
        # Populate the first conversation, then leave it.
        client.post(
            "/api/v1/conversation/import",
            json={"turns": [{"user": "remember the plan", "assistant": "noted"}]},
        )
        first_id = client.get("/api/v1/memory/export").json()["conversations"][0]["conversation"][
            "id"
        ]
        client.patch(f"/api/v1/memory/conversations/{first_id}", json={"title": "The plan"})
        client.post("/api/v1/conversation/clear")
        assert client.get("/api/v1/conversation/history").json() == []

        r = client.post("/api/v1/conversation/resume", json={"conversation_id": first_id})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "resumed"
        assert body["conversation_id"] == first_id
        assert body["title"] == "The plan"
        assert body["turns"] == 1

        history = client.get("/api/v1/conversation/history").json()
        assert [t["user"] for t in history] == ["remember the plan"]

        # New turns land in the RESUMED conversation, not a fresh one.
        client.post(
            "/api/v1/conversation/import",
            json={"turns": [{"user": "and the follow-up", "assistant": "done"}]},
        )
        export = client.get(f"/api/v1/memory/export?conversation_id={first_id}").json()
        texts = [t["text"] for t in export["conversations"][0]["turns"]]
        assert "and the follow-up" in texts

    def test_resume_unknown_conversation_is_404(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._start(client, monkeypatch)
        r = client.post("/api/v1/conversation/resume", json={"conversation_id": "does-not-exist"})
        assert r.status_code == 404
        assert r.json()["error_type"] == "MemoryNotFoundError"


class TestMicrophoneMute:
    """POST /conversation/microphone (M5.7)."""

    def _start(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.onboarding.check_readiness", lambda settings, paths: [])
        assert client.post("/api/v1/engine/start").status_code == 200

    def test_mute_and_unmute_roundtrip(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._start(client, monkeypatch)
        assert client.post("/api/v1/conversation/microphone", json={"muted": True}).json() == {
            "muted": True
        }
        assert client.get("/api/v1/diagnostics").json()["microphone_muted"] is True
        assert client.post("/api/v1/conversation/microphone", json={"muted": False}).json() == {
            "muted": False
        }
        assert client.get("/api/v1/diagnostics").json()["microphone_muted"] is False

    def test_microphone_requires_running_engine(self, client: TestClient) -> None:
        assert (
            client.post("/api/v1/conversation/microphone", json={"muted": True}).status_code == 409
        )

    def test_snapshot_reports_microphone_availability(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._start(client, monkeypatch)
        snap = client.get("/api/v1/diagnostics").json()
        # Default settings grant microphone permission.
        assert snap["microphone_available"] is True
        assert snap["microphone_muted"] is False
