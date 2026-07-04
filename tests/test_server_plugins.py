"""Plugin API tests — no real plugins installed; discovery legitimately empty."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


def test_list_plugins_empty_by_default(client: TestClient) -> None:
    r = client.get("/api/v1/plugins")
    assert r.status_code == 200
    assert r.json() == []


def test_get_unknown_plugin_is_404(client: TestClient) -> None:
    r = client.get("/api/v1/plugins/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error_type"] == "PluginError"


def test_enable_unknown_plugin_is_404(client: TestClient) -> None:
    r = client.post("/api/v1/plugins/does-not-exist/enable")
    assert r.status_code == 404
