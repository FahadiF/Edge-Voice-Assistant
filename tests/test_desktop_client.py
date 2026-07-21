"""DesktopClient tests (M6.1, ADR-027): maps shell actions to /api/v1 calls
with an injected opener — no server needed. The desktop app is another client
(ADR-007); this is the tested boundary."""

from __future__ import annotations

from typing import Any

from eva.desktop.client import DesktopClient


class _Resp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_start_engine_posts_to_engine_start() -> None:
    seen: dict[str, Any] = {}

    def opener(request: Any, timeout: float = 0) -> _Resp:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        return _Resp(200)

    client = DesktopClient("127.0.0.1", 8765, opener=opener)
    assert client.start_engine() is True
    assert seen["url"] == "http://127.0.0.1:8765/api/v1/engine/start"
    assert seen["method"] == "POST"


def test_start_engine_is_best_effort_on_failure() -> None:
    def opener(request: Any, timeout: float = 0) -> _Resp:
        raise ConnectionError("server not up yet")

    client = DesktopClient("127.0.0.1", 8765, opener=opener)
    assert client.start_engine() is False  # logged, never raised


def test_wildcard_host_normalized() -> None:
    seen: dict[str, Any] = {}

    def opener(request: Any, timeout: float = 0) -> _Resp:
        seen["url"] = request.full_url
        return _Resp(200)

    DesktopClient("0.0.0.0", 9000, opener=opener).start_engine()
    assert seen["url"].startswith("http://127.0.0.1:9000/")
