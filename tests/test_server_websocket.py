"""WebSocket event stream tests."""

from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

from eva.config.paths import AppPaths
from eva.core.events import TurnStarted
from eva.server import create_app


@pytest.fixture
def client(app_paths: AppPaths) -> TestClient:
    return TestClient(create_app(app_paths))


def test_connect_receives_idle_snapshot_first(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["data"]["state"] == "idle"


def test_published_event_is_forwarded_live(client: TestClient) -> None:
    with client.websocket_connect("/api/v1/ws") as ws:
        ws.receive_json()  # snapshot
        bus = client.app.state.eva.bus
        bus.publish(TurnStarted(epoch=7))
        msg = ws.receive_json()
        assert msg["type"] == "TurnStarted"
        assert msg["data"]["epoch"] == 7


def test_multiple_clients_each_get_events(client: TestClient) -> None:
    with (
        client.websocket_connect("/api/v1/ws") as ws1,
        client.websocket_connect("/api/v1/ws") as ws2,
    ):
        ws1.receive_json()
        ws2.receive_json()
        client.app.state.eva.bus.publish(TurnStarted(epoch=1))
        assert ws1.receive_json()["type"] == "TurnStarted"
        assert ws2.receive_json()["type"] == "TurnStarted"


def test_disconnect_unsubscribes(client: TestClient) -> None:
    bus = client.app.state.eva.bus
    before = len(bus._subscribers)  # whitebox check for a subscriber leak
    with client.websocket_connect("/api/v1/ws") as ws:
        ws.receive_json()
        assert len(bus._subscribers) == before + 1
    assert len(bus._subscribers) == before


class TestOriginPolicy:
    """CORS middleware does not cover WebSocket handshakes (M5.6): the /ws
    endpoint enforces the localhost-only browser policy itself. Without it,
    any website the user visits could read live transcripts."""

    @pytest.mark.parametrize(
        "origin",
        [
            "http://localhost:5173",
            "http://127.0.0.1:8765",
            "https://localhost",
        ],
    )
    def test_localhost_browser_origins_accepted(self, client: TestClient, origin: str) -> None:
        with client.websocket_connect("/api/v1/ws", headers={"origin": origin}) as ws:
            assert ws.receive_json()["type"] == "snapshot"

    @pytest.mark.parametrize(
        "origin",
        [
            "https://evil.example",
            "http://localhost.evil.example",  # prefix-spoofed hostname
            "http://127.0.0.1.evil.example",
            "null",  # sandboxed iframe / file:// pages
        ],
    )
    def test_foreign_origins_rejected(self, client: TestClient, origin: str) -> None:
        from starlette.websockets import WebSocketDisconnect

        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/api/v1/ws", headers={"origin": origin}) as ws,
        ):
            ws.receive_json()

    def test_no_origin_header_accepted(self, client: TestClient) -> None:
        """No Origin = non-browser client (CLI, desktop shell, curl) — the
        header is a browser mechanism; its absence must not lock those out."""
        with client.websocket_connect("/api/v1/ws") as ws:
            assert ws.receive_json()["type"] == "snapshot"


def test_bus_close_unsubscribes_open_stream(client: TestClient) -> None:
    """M5.7: closing the bus (server shutdown) wakes the WS handler so it
    unsubscribes and returns — no task left blocked in queue.get()."""
    bus = client.app.state.eva.bus
    before = len(bus._subscribers)
    with client.websocket_connect("/api/v1/ws") as ws:
        ws.receive_json()  # snapshot
        assert len(bus._subscribers) == before + 1
        bus.close()
        # The handler wakes on the sentinel, closes, and unsubscribes.
        with contextlib.suppress(Exception):
            ws.receive_json()
    assert len(bus._subscribers) == before


def test_origin_allowed_unit() -> None:
    from eva.server.security import origin_allowed

    assert origin_allowed(None)
    assert origin_allowed("http://localhost:3000")
    assert origin_allowed("http://127.0.0.1")
    assert not origin_allowed("https://example.com")
    assert not origin_allowed("http://localhost.attacker.example")
    assert not origin_allowed("file://")
    assert not origin_allowed("null")
