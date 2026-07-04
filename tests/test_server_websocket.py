"""WebSocket event stream tests."""

from __future__ import annotations

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
