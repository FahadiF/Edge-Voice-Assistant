"""Desktop shell tests (ADR-023) — `pywebview` is an optional extra not
installed in the test environment, so `main()`'s window/event-loop call is
exercised against a fake module injected into `sys.modules` rather than the
real package. `find_free_port`/`wait_for_health` need no mocking."""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Iterator

import pytest

from eva.desktop import find_free_port, main, wait_for_health


def test_find_free_port_returns_a_bindable_port() -> None:
    import socket

    port = find_free_port()
    assert 0 < port < 65536
    # Immediately bindable — proves it was actually free, not just a guess.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_wait_for_health_returns_true_once_server_responds() -> None:
    import http.server

    port = find_free_port()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_a: object) -> None:  # silence default logging
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert wait_for_health(f"http://127.0.0.1:{port}/", timeout_s=5) is True
    finally:
        server.shutdown()


def test_wait_for_health_times_out_when_nothing_listens() -> None:
    port = find_free_port()  # guaranteed free -> nothing answers
    assert wait_for_health(f"http://127.0.0.1:{port}/", timeout_s=0.3) is False


@pytest.fixture
def fake_webview() -> Iterator[types.SimpleNamespace]:
    calls: dict[str, object] = {}

    def create_window(title: str, url: str, **kwargs: object) -> None:
        calls["title"] = title
        calls["url"] = url
        calls["kwargs"] = kwargs

    def start() -> None:
        calls["started"] = True

    fake = types.SimpleNamespace(create_window=create_window, start=start, calls=calls)
    sys.modules["webview"] = fake  # type: ignore[assignment]
    yield fake
    del sys.modules["webview"]


def test_main_opens_a_window_at_the_health_checked_url(
    fake_webview: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("eva.desktop._run_server", lambda host, port: None)
    monkeypatch.setattr("eva.desktop.wait_for_health", lambda url, timeout_s=30.0: True)
    assert main() == 0
    assert fake_webview.calls["title"] == "Edge Voice Assistant"
    assert fake_webview.calls["url"].startswith("http://127.0.0.1:")
    assert fake_webview.calls.get("started") is True


def test_main_returns_error_code_when_backend_never_becomes_healthy(
    fake_webview: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("eva.desktop._run_server", lambda host, port: None)
    monkeypatch.setattr("eva.desktop.wait_for_health", lambda url, timeout_s=30.0: False)
    assert main() == 1
    assert "started" not in fake_webview.calls
