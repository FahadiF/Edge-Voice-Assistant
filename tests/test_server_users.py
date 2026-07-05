"""User profile API tests (M4, ADR-022) — fake engine (server_fakes), no
real models or audio hardware. Requires a running engine (profiles live in
the memory database).
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
        assert client.get("/api/v1/users").status_code == 409

    def test_create_requires_engine(self, client: TestClient) -> None:
        r = client.post("/api/v1/users", json={"nickname": "Fahad"})
        assert r.status_code == 409


class TestCreateAndGet:
    def test_create_generates_id_when_omitted(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/users", json={"nickname": "Fahad"})
        assert r.status_code == 200
        body = r.json()
        assert body["id"]
        assert body["nickname"] == "Fahad"

    def test_create_with_explicit_id(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/users", json={"id": "u1", "nickname": "Fahad"})
        assert r.status_code == 200
        assert r.json()["id"] == "u1"

    def test_get_returns_created_profile(self, running_client: TestClient) -> None:
        running_client.post("/api/v1/users", json={"id": "u1", "nickname": "Fahad"})
        r = running_client.get("/api/v1/users/u1")
        assert r.status_code == 200
        assert r.json()["nickname"] == "Fahad"

    def test_get_unknown_returns_404(self, running_client: TestClient) -> None:
        r = running_client.get("/api/v1/users/does-not-exist")
        assert r.status_code == 404

    def test_list_returns_all_profiles(self, running_client: TestClient) -> None:
        running_client.post("/api/v1/users", json={"id": "u1"})
        running_client.post("/api/v1/users", json={"id": "u2"})
        r = running_client.get("/api/v1/users")
        assert {p["id"] for p in r.json()} == {"u1", "u2"}

    def test_defaults_applied(self, running_client: TestClient) -> None:
        r = running_client.post("/api/v1/users", json={"id": "u1"})
        body = r.json()
        assert body["units"] == "metric"
        assert body["timezone"] == "UTC"
        assert body["nickname"] == ""


class TestUpdate:
    def test_partial_update_only_changes_given_fields(self, running_client: TestClient) -> None:
        running_client.post(
            "/api/v1/users", json={"id": "u1", "nickname": "Fahad", "units": "imperial"}
        )
        r = running_client.patch("/api/v1/users/u1", json={"nickname": "Fahian"})
        assert r.status_code == 200
        body = r.json()
        assert body["nickname"] == "Fahian"
        assert body["units"] == "imperial"  # untouched

    def test_update_unknown_returns_404(self, running_client: TestClient) -> None:
        r = running_client.patch("/api/v1/users/does-not-exist", json={"nickname": "x"})
        assert r.status_code == 404


class TestActivateAndDelete:
    def test_activate_sets_active_profile(self, running_client: TestClient) -> None:
        running_client.post("/api/v1/users", json={"id": "u1"})
        running_client.post("/api/v1/users", json={"id": "u2"})
        r = running_client.post("/api/v1/users/u2/activate")
        assert r.status_code == 200

    def test_delete_removes_profile(self, running_client: TestClient) -> None:
        running_client.post("/api/v1/users", json={"id": "u1"})
        r = running_client.delete("/api/v1/users/u1")
        assert r.status_code == 200
        assert running_client.get("/api/v1/users/u1").status_code == 404

    def test_delete_unknown_returns_404(self, running_client: TestClient) -> None:
        r = running_client.delete("/api/v1/users/does-not-exist")
        assert r.status_code == 404
