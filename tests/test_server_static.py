"""Web UI static hosting tests (ADR-023).

The API must stay byte-for-byte API-only when no built frontend exists, and
serve the SPA (with index.html fallback for client-side routes) when one
does. `EVA_UI_DIST` is authoritative when set — conftest points it at a
bogus path by default so a real `web/dist` build in the checkout never
leaks into unrelated tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.server import create_app
from eva.server.static import ui_dist_dir


def _make_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>EVA UI</body></html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('eva')", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return dist


class TestNoUiBuild:
    def test_root_is_404_when_no_dist_exists(self, app_paths: AppPaths) -> None:
        client = TestClient(create_app(app_paths))
        assert client.get("/").status_code == 404

    def test_api_works_without_dist(self, app_paths: AppPaths) -> None:
        client = TestClient(create_app(app_paths))
        assert client.get("/api/v1/health").status_code == 200

    def test_env_override_pointing_nowhere_disables_ui(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVA_UI_DIST", str(tmp_path / "does-not-exist"))
        assert ui_dist_dir() is None

    def test_empty_dir_without_index_html_does_not_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("EVA_UI_DIST", str(empty))
        assert ui_dist_dir() is None


class TestUiMounted:
    @pytest.fixture
    def ui_client(
        self, app_paths: AppPaths, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> TestClient:
        dist = _make_dist(tmp_path)
        monkeypatch.setenv("EVA_UI_DIST", str(dist))
        return TestClient(create_app(app_paths))

    def test_root_serves_index_html(self, ui_client: TestClient) -> None:
        r = ui_client.get("/")
        assert r.status_code == 200
        assert "EVA UI" in r.text

    def test_spa_fallback_for_client_side_routes(self, ui_client: TestClient) -> None:
        r = ui_client.get("/settings/audio")
        assert r.status_code == 200
        assert "EVA UI" in r.text  # index.html, not a 404

    def test_real_files_served_directly(self, ui_client: TestClient) -> None:
        assert ui_client.get("/favicon.svg").text == "<svg/>"
        assert "eva" in ui_client.get("/assets/app.js").text

    def test_api_routes_win_over_spa_fallback(self, ui_client: TestClient) -> None:
        r = ui_client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_openapi_docs_win_over_spa_fallback(self, ui_client: TestClient) -> None:
        assert ui_client.get("/openapi.json").status_code == 200
        assert "EVA UI" not in ui_client.get("/openapi.json").text

    def test_path_escape_falls_back_to_index(self, ui_client: TestClient) -> None:
        # A traversal-shaped path must never serve a file outside dist.
        r = ui_client.get("/..%2f..%2fpyproject.toml")
        assert r.status_code == 200
        assert "EVA UI" in r.text
