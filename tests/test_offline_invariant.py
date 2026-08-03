"""The offline invariant (H9) and the egress boundary that makes it testable (H2).

"EVA runs fully offline" is the product's central claim, and before Batch 6 it
was enforced by convention: `ARCHITECTURE.md §10` said so, and nothing checked.
These tests turn it into a property.

The whole difficulty is that a naive "no outbound sockets" test breaks three
things that legitimately speak HTTP — to **localhost**. The desktop shell drives
the engine over `/api/v1`, `eva start/stop` supervises a detached server the
same way, and `eva status` probes its health endpoint. None of that leaves the
machine, so none of it is egress. `eva.core.net.is_loopback_host()` is the one
definition both this fixture and production code use, so the two cannot drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from pathlib import Path

import numpy as np
import pytest

from eva.audio.segmenter import UtteranceEnd
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core import net
from eva.core.events import EventBus, TurnFinished
from tests.server_fakes import FakeASR, FakeAudioSystem, FakeLLM, FakeMemoryStore, FakeTTS

#: Modules allowed to import `urllib.request`, and why. Anything else that
#: appears here is either new egress that must go through `eva.core.net`, or
#: new loopback IPC that must be added deliberately — never by accident.
_URLLIB_ALLOWLIST: dict[str, str] = {
    "eva/core/net.py": "the single audited egress point (H2)",
    "eva/desktop/client.py": "loopback IPC — desktop shell → local /api/v1",
    "eva/service.py": "loopback IPC — server supervisor health/shutdown probes",
    "eva/cli.py": "loopback IPC — `eva status` health/engine probe",
}

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: A reserved documentation address (RFC 5737 TEST-NET-3). Used instead of a
#: real hostname on purpose: an IP literal needs no DNS, so `connect()` is
#: always the first network call and the guard below is always what stops it.
#: With a hostname, a machine without DNS would fail in `getaddrinfo` before
#: any connection was attempted, and a test asserting "the guard fired" would
#: fail for the wrong reason.
_REMOTE_URL = "http://203.0.113.1/model.gguf"


class EgressBlockedError(OSError):
    """Raised in place of a real outbound connection during these tests.

    An `OSError` subclass on purpose: that is what production code already
    catches for network failure, so a blocked connection is indistinguishable
    from an unreachable network — which is exactly the condition being
    simulated.
    """


async def _drive_one_turn(orch: Orchestrator, bus: EventBus) -> bool:
    """Run a single utterance through the orchestrator and shut it down.

    Shared so both conversation tests use one correct lifecycle: the
    orchestrator task is always awaited after `request_shutdown()`, never left
    dangling for the event loop to garbage-collect mid-turn.
    """
    queue = bus.subscribe()
    run_task = asyncio.create_task(orch.run())
    await asyncio.sleep(0)  # let run() bind the loop
    orch.feed_audio_event(UtteranceEnd(np.ones(16_000, dtype=np.int16), 1000, 800, False))

    finished = False
    for _ in range(400):  # up to 8 s; fakes normally finish in well under 1 s
        while not queue.empty():
            if isinstance(queue.get_nowait(), TurnFinished):
                finished = True
        if finished:
            break
        await asyncio.sleep(0.02)

    orch.request_shutdown()
    await asyncio.wait_for(run_task, 10)
    return finished


def _address_host(address: object) -> str | None:
    """The host portion of whatever `socket.connect` was handed.

    AF_INET is `(host, port)`, AF_INET6 is `(host, port, flow, scope)`, and
    AF_UNIX is a filesystem path — a path never leaves the machine, so it is
    reported as loopback.
    """
    if isinstance(address, str):
        return "127.0.0.1"  # AF_UNIX socket path: local by definition
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


@pytest.fixture
def block_egress(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Deny every non-loopback socket connection; permit loopback.

    Patches `socket.socket.connect`/`connect_ex` rather than any HTTP library,
    so it catches egress from *any* stack — urllib, huggingface_hub, httpx, a
    future provider SDK — instead of only the one we thought to mock.
    `socket.create_connection` needs no separate patch: it calls `connect`.

    Returns the list of blocked addresses so a test can assert on what was
    attempted, not merely that nothing raised.
    """
    blocked: list[object] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guard(address: object) -> None:
        if not net.is_loopback_host(_address_host(address)):
            blocked.append(address)
            raise EgressBlockedError(f"outbound connection to {address!r} blocked (offline test)")

    def guarded_connect(self: socket.socket, address: object) -> None:
        guard(address)
        real_connect(self, address)  # type: ignore[arg-type]

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        guard(address)
        return int(real_connect_ex(self, address))  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    return blocked


# ──────────────────────── the loopback/egress boundary ────────────────────────


class TestLoopbackClassification:
    """Getting this wrong in either direction is the whole risk H2 names: too
    strict breaks the desktop shell, too loose lets real egress through."""

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.0.0.53", "::1", "[::1]", "localhost", "LOCALHOST", "0.0.0.0"],
    )
    def test_local_hosts_are_loopback(self, host: str) -> None:
        assert net.is_loopback_host(host) is True

    @pytest.mark.parametrize(
        "host",
        [
            "huggingface.co",
            "8.8.8.8",
            "example.com",
            "127.0.0.1.evil.com",  # loopback-looking prefix, remote host
            "2606:4700::1111",
            "",
            None,
        ],
    )
    def test_remote_or_unknown_hosts_are_egress(self, host: str | None) -> None:
        """An unparseable or empty host is treated as egress: the safe default
        for "I cannot tell" is to deny, not to permit."""
        assert net.is_loopback_host(host) is False

    def test_url_classification_matches_host_classification(self) -> None:
        assert net.is_loopback_url("http://127.0.0.1:8000/api/v1/health") is True
        assert net.is_loopback_url("http://localhost:8000/api/v1/engine/status") is True
        assert net.is_loopback_url("https://huggingface.co/model.gguf") is False

    def test_the_real_catalog_urls_are_egress_not_loopback(self) -> None:
        """Guards the classifier against a change that would quietly reclassify
        model downloads as local and exempt them from the invariant."""
        from eva.models.catalog import BUILTIN_CATALOG

        urls = [f.url for m in BUILTIN_CATALOG for f in m.files]
        assert urls, "expected the built-in catalog to declare download URLs"
        assert all(not net.is_loopback_url(url) for url in urls)


# ──────────────────────── import-direction enforcement ────────────────────────


class TestEgressImportDirection:
    """H2's structural guarantee: `urllib.request` appears in exactly four
    files, each for a stated reason."""

    @staticmethod
    def _urllib_importers() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if "urllib" in line and line.strip().startswith(("import ", "from "))
            ]
            if lines:
                found[path.relative_to(_SRC_ROOT).as_posix()] = lines
        return found

    def test_only_the_allowlisted_modules_import_urllib(self) -> None:
        importers = self._urllib_importers()
        unexpected = set(importers) - set(_URLLIB_ALLOWLIST)
        assert not unexpected, (
            f"new urllib.request users: {sorted(unexpected)}. Outbound traffic must go "
            "through eva.core.net (H2); loopback IPC must be added to the allowlist "
            "in this test with a stated reason."
        )

    def test_every_allowlisted_module_still_uses_urllib(self) -> None:
        """The allowlist is an inventory, not a permission slip — a stale entry
        means the boundary moved and this test is no longer describing reality."""
        importers = self._urllib_importers()
        stale = set(_URLLIB_ALLOWLIST) - set(importers)
        assert not stale, f"allowlist entries no longer import urllib: {sorted(stale)}"

    def test_the_model_manager_no_longer_imports_urllib_directly(self) -> None:
        """The point of the batch: the one egress caller routes through the
        boundary instead of opening sockets itself.

        Checked against imports, not mentions — `manager.py` still *refers* to
        `urllib.error.URLError` in a comment explaining why catching `OSError`
        is sufficient, and forbidding the word would punish the explanation."""
        assert "eva/models/manager.py" not in self._urllib_importers()
        source = (_SRC_ROOT / "eva/models/manager.py").read_text(encoding="utf-8")
        assert "from eva.core import net" in source
        assert "net.open_url(" in source

    def test_no_other_http_client_library_appears_in_production_code(self) -> None:
        """`requests`/`httpx`/`aiohttp` would each be a second egress stack,
        bypassing the boundary entirely. `httpx` is a test-only dependency
        (FastAPI's TestClient) and must stay out of `src/`."""
        offenders: dict[str, str] = {}
        for path in sorted(_SRC_ROOT.rglob("*.py")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(
                    lib in stripped for lib in ("requests", "httpx", "aiohttp")
                ):
                    offenders[path.relative_to(_SRC_ROOT).as_posix()] = stripped
        assert not offenders, f"second HTTP stack introduced: {offenders}"


# ──────────────────────── the offline invariant itself ────────────────────────


class TestOfflineConversation:
    """H9: a full conversation must complete with every outbound socket denied."""

    @staticmethod
    def _orchestrator() -> tuple[Orchestrator, EventBus, FakeAudioSystem]:
        settings = Settings()
        settings.conversation.system_prompt = "test"
        bus = EventBus()
        audio = FakeAudioSystem()
        orch = Orchestrator(
            settings, bus, audio, FakeASR(), FakeLLM(), FakeTTS(), FakeMemoryStore()
        )
        return orch, bus, audio

    def test_a_full_turn_completes_with_all_egress_blocked(
        self, block_egress: list[object]
    ) -> None:
        """ASR → LLM → sentence chunking → TTS → playback, end to end, with the
        socket layer refusing anything that is not loopback."""

        async def scenario() -> None:
            orch, bus, audio = self._orchestrator()
            finished = await _drive_one_turn(orch, bus)
            assert finished, "the turn never completed"
            assert audio.spoken, "nothing was synthesized — the turn did not really run"
            assert orch.metrics.turns, "no turn metrics recorded"

        asyncio.run(scenario())
        assert block_egress == [], f"the conversation attempted egress: {block_egress}"

    def test_the_fixture_actually_blocks_egress(self, block_egress: list[object]) -> None:
        """A guard that silently permits everything would make the test above
        pass forever. Prove the teeth are real.

        `assert block_egress` is the load-bearing assertion, not `raises`:
        a *successful* request to a real host can also raise `OSError`
        (`HTTPError` on a 401/404 subclasses it), so "it raised" alone would
        pass even if the connection had gone out."""
        with pytest.raises(OSError):
            net.open_url(_REMOTE_URL, timeout=1)
        assert block_egress, "the fixture did not record the blocked attempt"

    def test_egress_attempted_during_a_turn_is_detected(self, block_egress: list[object]) -> None:
        """The meta-test that keeps the invariant honest.

        `test_a_full_turn_completes_with_all_egress_blocked` asserts a clean
        conversation makes no outbound connection — which is worth nothing
        unless a *dirty* one would be caught. This injects an engine that
        phones home mid-turn (the exact shape a future remote provider would
        take) and proves the guard catches it through the orchestrator's
        producer thread, not just on the main thread.
        """
        from collections.abc import Callable, Generator

        from eva.core.tools import ToolDefinition
        from eva.llm.base import ChatMessage, GenerationOutcome, GenerationParams, LLMEngine

        class _LeakyLLM(LLMEngine):
            """Shaped like a remote provider adapter, and matching the real
            `stream()` signature rather than the looser one the older test fakes
            still use (BACKLOG E8) — so this test adds nothing to that drift."""

            device = "cpu"

            def load(self) -> None: ...
            def unload(self) -> None: ...

            def stream(
                self,
                messages: list[ChatMessage],
                params: GenerationParams,
                should_abort: Callable[[], bool],
                *,
                tools: tuple[ToolDefinition, ...] = (),
            ) -> Generator[str, None, GenerationOutcome]:
                with contextlib.suppress(OSError):
                    net.open_url(_REMOTE_URL, timeout=1)
                yield "leaked"
                return GenerationOutcome(reason="stop")

        async def scenario() -> None:
            settings = Settings()
            settings.conversation.system_prompt = "test"
            bus = EventBus()
            orch = Orchestrator(
                settings,
                bus,
                FakeAudioSystem(),
                FakeASR(),
                _LeakyLLM(),
                FakeTTS(),
                FakeMemoryStore(),
            )
            await _drive_one_turn(orch, bus)

        asyncio.run(scenario())
        assert block_egress, "egress from inside a turn went undetected"

    def test_a_model_download_is_blocked_while_offline(
        self, block_egress: list[object], tmp_path: Path
    ) -> None:
        """The one production egress path must not succeed while offline.

        `block_egress` is required here even though it is not asserted on: it
        installs the guard. Without it this test attempts a real multi-hundred-
        megabyte download on any machine that has internet.

        Asserts the *outcome* rather than which layer stopped it: on a machine
        with no DNS the catalog hostname fails in `getaddrinfo`, and on one with
        DNS it fails at the guarded `connect()`. Both are the invariant holding;
        pinning it to one would make the test environment-dependent.
        """
        from eva.config.paths import AppPaths
        from eva.core.errors import ModelError
        from eva.models.manager import ModelManager

        paths = AppPaths(
            config_dir=tmp_path / "c",
            data_dir=tmp_path / "d",
            models_dir=tmp_path / "m",
            logs_dir=tmp_path / "l",
            conversations_dir=tmp_path / "v",
        )
        paths.ensure_exists()
        with pytest.raises(ModelError, match="Download failed"):
            ModelManager(paths).download("kokoro-82m-v1.0")
        assert not ModelManager(paths).is_installed("kokoro-82m-v1.0")


# ──────────────────────── loopback clients keep working ────────────────────────


class TestLoopbackClientsUnaffected:
    """The explicit regression check Batch 6's acceptance criteria demand: a
    naive version of the invariant above breaks all three of these."""

    def test_service_health_probe_is_permitted_to_reach_loopback(
        self, block_egress: list[object]
    ) -> None:
        """No server is listening, so the probe returns False — the point is
        that it fails by *refused connection*, never by the egress guard."""
        from eva import service

        assert service.probe_health(service.health_url("127.0.0.1", 8765), timeout_s=1) is False
        assert block_egress == [], "the loopback health probe was misclassified as egress"

    def test_service_probe_against_a_remote_host_is_blocked(
        self, block_egress: list[object]
    ) -> None:
        """The mirror image: the same code path aimed off-machine is denied, so
        the exemption is about the destination, not about the module."""
        from eva import service

        assert service.probe_health(_REMOTE_URL, timeout_s=1) is False
        assert block_egress, "a remote probe should have been blocked"

    def test_graceful_shutdown_request_is_permitted_to_reach_loopback(
        self, block_egress: list[object]
    ) -> None:
        from eva import service

        assert service.request_graceful_shutdown("127.0.0.1", 8765, timeout_s=1) is False
        assert block_egress == []

    def test_desktop_client_is_permitted_to_reach_loopback(
        self, block_egress: list[object]
    ) -> None:
        from eva.desktop.client import DesktopClient

        assert DesktopClient("127.0.0.1", 8765).start_engine() is False
        assert block_egress == [], "the desktop client was misclassified as egress"

    def test_desktop_client_resolves_the_wildcard_bind_to_loopback(
        self, block_egress: list[object]
    ) -> None:
        """`0.0.0.0` means "listen everywhere" and is not connectable;
        `display_host` rewrites it to 127.0.0.1. If that ever regressed, the
        desktop shell would look like egress."""
        from eva.desktop.client import DesktopClient

        assert DesktopClient("0.0.0.0", 8765).start_engine() is False
        assert block_egress == []

    def test_cli_status_with_no_server_exits_before_probing(
        self, block_egress: list[object], capsys: pytest.CaptureFixture[str]
    ) -> None:
        import argparse

        from eva.cli import _cmd_status

        assert _cmd_status(argparse.Namespace()) == 1
        assert "not running" in capsys.readouterr().out
        assert block_egress == []

    def test_cli_status_probe_actually_reaches_loopback(
        self, block_egress: list[object], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`eva status` short-circuits on a missing PID file, so without one it
        never touches the network and would pass vacuously. Recording a live
        PID (this test process — `read_server_pid` requires an existing python
        process) drives it into the real `probe_health` loopback call, which is
        the path the acceptance criterion is about."""
        import argparse
        import os

        from eva import service
        from eva.cli import _cmd_status
        from eva.config.paths import get_app_paths

        paths = get_app_paths()
        paths.ensure_exists()
        service.pid_file(paths).write_text(str(os.getpid()), encoding="utf-8")

        assert _cmd_status(argparse.Namespace()) == 1  # API not responding
        output = capsys.readouterr().out
        assert "running (PID" in output  # it got past the PID check...
        assert "not responding" in output  # ...and really attempted the probe
        assert block_egress == [], "the CLI status probe was misclassified as egress"
