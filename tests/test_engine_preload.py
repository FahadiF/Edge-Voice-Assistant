"""Engine preload tests (M5.5, ADR-026): parallel loading, progress events,
GPU ordering, lazy TTS."""

from __future__ import annotations

import asyncio
import threading

import pytest

from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.events import ComponentLoadFinished, ComponentLoadStarted, Event
from tests.server_fakes import build_fake_assistant


async def _collect_preload_events(assistant) -> list[Event]:
    bus = assistant.bus
    bus.bind_loop(asyncio.get_running_loop())
    queue = bus.subscribe()
    await asyncio.to_thread(assistant.preload)
    await asyncio.sleep(0.05)  # let threadsafe publishes flush
    events: list[Event] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


class TestPreloadProgress:
    def test_every_component_reports_start_and_finish(self, app_paths: AppPaths) -> None:
        async def scenario() -> None:
            assistant = build_fake_assistant(Settings(), app_paths)
            events = await _collect_preload_events(assistant)
            started = {e.component for e in events if isinstance(e, ComponentLoadStarted)}
            finished = {e.component for e in events if isinstance(e, ComponentLoadFinished)}
            assert {"llm", "asr", "tts"} <= started
            assert started == finished
            assert all(not e.error for e in events if isinstance(e, ComponentLoadFinished))

        asyncio.run(scenario())

    def test_gpu_order_llm_strictly_before_asr(self, app_paths: AppPaths) -> None:
        """ADR-015 §5 must survive parallelization: the LLM claims the GPU
        before ASR loads; only CPU components may overlap them."""

        async def scenario() -> None:
            assistant = build_fake_assistant(Settings(), app_paths)
            order: list[str] = []
            lock = threading.Lock()
            real_llm_load = assistant.llm.load
            real_asr_load = assistant.asr.load

            def llm_load() -> None:
                with lock:
                    order.append("llm")
                real_llm_load()

            def asr_load() -> None:
                with lock:
                    order.append("asr")
                real_asr_load()

            assistant.llm.load = llm_load  # type: ignore[method-assign]
            assistant.asr.load = asr_load  # type: ignore[method-assign]
            await asyncio.to_thread(assistant.preload)
            assert order.index("llm") < order.index("asr")

        asyncio.run(scenario())

    def test_lazy_tts_skips_tts_at_preload(self, app_paths: AppPaths) -> None:
        async def scenario() -> None:
            assistant = build_fake_assistant(Settings(), app_paths)
            assistant.settings.tts.lazy_load = True
            loaded = {"tts": False}
            real_load = assistant.tts.load

            def tts_load() -> None:
                loaded["tts"] = True
                real_load()

            assistant.tts.load = tts_load  # type: ignore[method-assign]
            events = await _collect_preload_events(assistant)
            assert loaded["tts"] is False
            started = {e.component for e in events if isinstance(e, ComponentLoadStarted)}
            assert "tts" not in started

        asyncio.run(scenario())

    def test_component_failure_is_reported_then_raised(self, app_paths: AppPaths) -> None:
        async def scenario() -> None:
            assistant = build_fake_assistant(Settings(), app_paths)

            def broken_load() -> None:
                raise RuntimeError("VRAM exhausted")

            assistant.llm.load = broken_load  # type: ignore[method-assign]
            bus = assistant.bus
            bus.bind_loop(asyncio.get_running_loop())
            queue = bus.subscribe()
            with pytest.raises(RuntimeError, match="VRAM exhausted"):
                await asyncio.to_thread(assistant.preload)
            await asyncio.sleep(0.05)
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
            failures = [e for e in events if isinstance(e, ComponentLoadFinished) and e.error]
            assert any(f.component == "llm" and "VRAM" in f.error for f in failures)

        asyncio.run(scenario())
