"""Settings API tests — isolated home, no real models needed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


class TestGetAndSchema:
    def test_get_settings_returns_current_document(self, client: TestClient) -> None:
        r = client.get("/api/v1/settings")
        assert r.status_code == 200
        body = r.json()
        assert body["llm"]["model"]
        from eva.config.settings import SETTINGS_SCHEMA_VERSION

        assert body["schema_version"] == SETTINGS_SCHEMA_VERSION

    def test_schema_is_json_schema_with_bounds(self, client: TestClient) -> None:
        r = client.get("/api/v1/settings/schema")
        assert r.status_code == 200
        schema = r.json()
        vad = schema["$defs"]["VADSettings"]["properties"]["threshold"]
        assert vad["minimum"] == 0.0
        assert vad["maximum"] == 1.0
        assert "description" in vad


class TestPutPatchReset:
    def test_put_replaces_whole_document(self, client: TestClient) -> None:
        current = client.get("/api/v1/settings").json()
        current["tts"]["voice"] = "af_bella"
        r = client.put("/api/v1/settings", json=current)
        assert r.status_code == 200
        assert r.json()["tts"]["voice"] == "af_bella"
        # Persisted: a fresh GET reflects it.
        assert client.get("/api/v1/settings").json()["tts"]["voice"] == "af_bella"

    def test_put_rejects_invalid_document(self, client: TestClient) -> None:
        r = client.put("/api/v1/settings", json={"vad": {"threshold": 5.0}})
        assert r.status_code == 422

    def test_patch_merges_nested_fields_only(self, client: TestClient) -> None:
        before = client.get("/api/v1/settings").json()
        r = client.patch("/api/v1/settings", json={"conversation": {"temperature": 0.9}})
        assert r.status_code == 200
        body = r.json()
        assert body["conversation"]["temperature"] == 0.9
        assert body["conversation"]["max_tokens"] == before["conversation"]["max_tokens"]
        assert body["llm"] == before["llm"]  # untouched section preserved

    def test_patch_out_of_bounds_rejected_with_details(self, client: TestClient) -> None:
        r = client.patch("/api/v1/settings", json={"vad": {"threshold": 5.0}})
        assert r.status_code == 422
        assert r.json()["error_type"] == "ValidationError"
        assert r.json()["errors"]

    def test_patch_unknown_field_rejected(self, client: TestClient) -> None:
        r = client.patch("/api/v1/settings", json={"llm": {"not_a_field": 1}})
        assert r.status_code == 422

    def test_reset_restores_defaults(self, client: TestClient) -> None:
        client.patch("/api/v1/settings", json={"conversation": {"temperature": 1.5}})
        r = client.post("/api/v1/settings/reset")
        assert r.status_code == 200
        assert r.json()["conversation"]["temperature"] == 0.4


class TestValidateEndpoint:
    def test_valid_payload(self, client: TestClient) -> None:
        payload = client.get("/api/v1/settings").json()
        r = client.post("/api/v1/settings/validate", json=payload)
        assert r.json() == {"valid": True, "errors": []}

    def test_invalid_payload_reports_errors_not_422(self, client: TestClient) -> None:
        r = client.post("/api/v1/settings/validate", json={"llm": {"model": 123}})
        assert r.status_code == 200  # validation result, not a request error
        body = r.json()
        assert body["valid"] is False
        assert body["errors"][0]["loc"] == ["llm", "model"]
