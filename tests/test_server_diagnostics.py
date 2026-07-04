"""Diagnostics API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


def test_diagnostics_without_engine_is_idle(client: TestClient) -> None:
    r = client.get("/api/v1/diagnostics")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["devices"] == {
        "llm": "unloaded",
        "asr": "unloaded",
        "tts": "unloaded",
        "vad": "unloaded",
    }
    assert body["resources"]["ram_total_mb"] > 0
    assert body["turns_completed"] == 0


def test_diagnostics_reflects_persisted_settings(client: TestClient) -> None:
    client.patch("/api/v1/settings", json={"conversation": {"language": "fi"}})
    body = client.get("/api/v1/diagnostics").json()
    assert body["language"] == "fi"


def test_system_hardware_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/system/hardware")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] in ("cpu-only", "gpu-6gb", "gpu-12gb")
    assert body["ram_mb"] > 0


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
