"""Persona API tests (M4, ADR-022) — works without a running engine (personas
are configuration, not conversation data).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


class TestListAndGet:
    def test_list_includes_builtins(self, client: TestClient) -> None:
        r = client.get("/api/v1/personas")
        assert r.status_code == 200
        ids = {p["id"] for p in r.json()}
        assert {"default", "professional", "friendly", "technical", "minimal", "creative"} <= ids

    def test_get_builtin_persona(self, client: TestClient) -> None:
        r = client.get("/api/v1/personas/technical")
        assert r.status_code == 200
        assert r.json()["verbosity"] == "detailed"

    def test_get_unknown_persona_returns_404(self, client: TestClient) -> None:
        r = client.get("/api/v1/personas/does-not-exist")
        assert r.status_code == 404


class TestCreateAndDelete:
    def test_create_custom_persona(self, client: TestClient) -> None:
        # A unique id, not shared with any other test file — persona_registry
        # is a process-wide singleton (ADR-010), so ids must be unique across
        # the whole test suite, not just within one file.
        payload = {
            "id": "server-api-buccaneer",
            "display_name": "Buccaneer",
            "system_prompt": "Speak like a pirate.",
            "tone": "boisterous",
        }
        r = client.post("/api/v1/personas", json=payload)
        assert r.status_code == 200
        assert r.json()["display_name"] == "Buccaneer"

        listed = client.get("/api/v1/personas").json()
        assert any(p["id"] == "server-api-buccaneer" for p in listed)

    def test_create_persists_across_requests(self, client: TestClient) -> None:
        client.post(
            "/api/v1/personas",
            json={"id": "custom-1", "display_name": "C1", "system_prompt": "Be C1."},
        )
        r = client.get("/api/v1/personas/custom-1")
        assert r.status_code == 200

    def test_cannot_shadow_a_builtin_id(self, client: TestClient) -> None:
        r = client.post(
            "/api/v1/personas",
            json={"id": "default", "display_name": "Fake", "system_prompt": "x"},
        )
        assert r.status_code == 400

    def test_delete_custom_persona(self, client: TestClient) -> None:
        client.post(
            "/api/v1/personas",
            json={"id": "temp", "display_name": "Temp", "system_prompt": "x"},
        )
        r = client.delete("/api/v1/personas/temp")
        assert r.status_code == 200
        assert client.get("/api/v1/personas/temp").status_code == 404

    def test_cannot_delete_a_builtin(self, client: TestClient) -> None:
        r = client.delete("/api/v1/personas/default")
        assert r.status_code == 400
        assert client.get("/api/v1/personas/default").status_code == 200

    def test_editing_a_custom_persona_updates_it(self, client: TestClient) -> None:
        client.post(
            "/api/v1/personas",
            json={"id": "editable", "display_name": "V1", "system_prompt": "x"},
        )
        client.post(
            "/api/v1/personas",
            json={"id": "editable", "display_name": "V2", "system_prompt": "y"},
        )
        r = client.get("/api/v1/personas/editable")
        assert r.json()["display_name"] == "V2"
