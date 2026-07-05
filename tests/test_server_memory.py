"""Memory management API tests (M4, Part 12) — fake engine (server_fakes),
no real models or audio hardware.
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


def _import_pair(client: TestClient, user: str = "hi", assistant: str = "hello") -> None:
    client.post(
        "/api/v1/conversation/import",
        json={"turns": [{"user": user, "assistant": assistant}]},
    )


class TestRequiresEngine:
    def test_search_requires_engine(self, client: TestClient) -> None:
        r = client.post("/api/v1/memory/search", json={"query": "hi"})
        assert r.status_code == 409

    def test_stats_requires_engine(self, client: TestClient) -> None:
        assert client.get("/api/v1/memory/stats").status_code == 409


class TestSearchAndStats:
    def test_search_finds_stored_turn(self, running_client: TestClient) -> None:
        _import_pair(running_client, "what is the weather", "sunny")
        r = running_client.post("/api/v1/memory/search", json={"query": "weather"})
        assert r.status_code == 200
        results = r.json()
        assert any("weather" in res["turn"]["text"] for res in results)

    def test_stats_reflects_content(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        r = running_client.get("/api/v1/memory/stats")
        assert r.status_code == 200
        stats = r.json()
        assert stats["turn_count"] == 2


class TestManagementVerbs:
    def _first_turn_id(self, client: TestClient) -> int:
        results = client.post("/api/v1/memory/search", json={"query": "target"}).json()
        return int(results[0]["turn"]["id"])

    def test_forget_turn(self, running_client: TestClient) -> None:
        _import_pair(running_client, "target text", "reply")
        turn_id = self._first_turn_id(running_client)
        r = running_client.delete(f"/api/v1/memory/turns/{turn_id}")
        assert r.status_code == 200
        after = running_client.post("/api/v1/memory/search", json={"query": "target"})
        assert after.json() == []

    def test_forget_unknown_turn_returns_404(self, running_client: TestClient) -> None:
        r = running_client.delete("/api/v1/memory/turns/999999")
        assert r.status_code == 404

    def test_pin_and_unpin_turn(self, running_client: TestClient) -> None:
        _import_pair(running_client, "target text", "reply")
        turn_id = self._first_turn_id(running_client)
        r = running_client.post(f"/api/v1/memory/turns/{turn_id}/pin")
        assert r.status_code == 200
        assert r.json()["status"] == "pinned"
        r = running_client.post(f"/api/v1/memory/turns/{turn_id}/pin?pinned=false")
        assert r.json()["status"] == "unpinned"

    def test_favorite_turn(self, running_client: TestClient) -> None:
        _import_pair(running_client, "target text", "reply")
        turn_id = self._first_turn_id(running_client)
        r = running_client.post(f"/api/v1/memory/turns/{turn_id}/favorite")
        assert r.json()["status"] == "favorited"


def _active_conversation_id(client: TestClient) -> str:
    exported = client.get("/api/v1/memory/export").json()
    conv_id: str = exported["conversations"][0]["conversation"]["id"]
    return conv_id


class TestConversationOperations:
    def test_archive_and_restore_conversation(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        conv_id = _active_conversation_id(running_client)

        r = running_client.post(f"/api/v1/memory/conversations/{conv_id}/archive")
        assert r.status_code == 200
        assert r.json()["status"] == "archived"

        r = running_client.post(f"/api/v1/memory/conversations/{conv_id}/archive?archived=false")
        assert r.json()["status"] == "restored"

    def test_delete_conversation(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        conv_id = _active_conversation_id(running_client)
        r = running_client.delete(f"/api/v1/memory/conversations/{conv_id}")
        assert r.status_code == 200

    def test_delete_all_memory(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        r = running_client.delete("/api/v1/memory")
        assert r.status_code == 200
        stats = running_client.get("/api/v1/memory/stats").json()
        assert stats["turn_count"] == 0


class TestContextPreview:
    def test_context_preview_returns_messages_and_trace(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/memory/context-preview", params={"text": "hello there"})
        assert r.status_code == 200
        body = r.json()
        assert body["messages"][-1]["content"] == "hello there"
        assert "trace" in body
        assert body["trace"]["recent_turn_count"] >= 0

    def test_context_preview_reflects_recent_turns(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        r = running_client.get("/api/v1/memory/context-preview", params={"text": "what next"})
        body = r.json()
        assert body["trace"]["recent_turn_count"] == 2


class TestExportImport:
    def test_export_returns_conversations(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        r = running_client.get("/api/v1/memory/export")
        assert r.status_code == 200
        assert r.json()["version"] == 1

    def test_import_round_trips(self, running_client: TestClient) -> None:
        _import_pair(running_client)
        exported = running_client.get("/api/v1/memory/export").json()
        r = running_client.post("/api/v1/memory/import", json=exported)
        assert r.status_code == 200


class TestSummarize:
    def test_summarize_conversation(self, running_client: TestClient) -> None:
        _import_pair(running_client, "hi", "hello there.")
        conv_id = _active_conversation_id(running_client)
        r = running_client.post(f"/api/v1/memory/conversations/{conv_id}/summarize")
        assert r.status_code == 200
        assert r.json()["status"] == "summarized"

    def test_summarize_empty_conversation(self, running_client: TestClient) -> None:
        # A conversation with zero turns doesn't exist as an exportable row in
        # this fake store's simplified model; use a non-existent id and expect
        # a graceful "no_turns" response, not a crash.
        r = running_client.post("/api/v1/memory/conversations/does-not-exist/summarize")
        assert r.status_code == 200
        assert r.json()["status"] == "no_turns"
