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


class TestMicrophonePermissionAudio:
    """M5.6: mic-off must still start PLAYBACK (typed conversations speak,
    and the playback queue drains so turns can finish) — before this,
    mic-off skipped audio entirely and every typed turn wedged in the
    'speaking' state waiting on a queue nothing ever drained."""

    def test_mic_on_starts_duplex(self, app_paths: AppPaths) -> None:
        settings = Settings()
        assistant = build_fake_assistant(settings, app_paths)
        assistant.start_audio()
        assert assistant.audio.started_mode == "duplex"
        assert assistant._audio_started is True

    def test_mic_off_starts_playback_only(self, app_paths: AppPaths) -> None:
        settings = Settings()
        settings.permissions.devices.microphone = False
        assistant = build_fake_assistant(settings, app_paths)
        assistant.start_audio()
        assert assistant.audio.started_mode == "playback-only"
        # stop() must stop the playback-only stream too.
        assert assistant._audio_started is True


class TestStopReleasesModels:
    """Engine restart in one process (M7.3).

    `Assistant.stop()` used to leave every model resident: the adapters sit in
    reference cycles, so dropping the assistant frees nothing until a full
    `gc.collect()`. Measured on the reference platform, an engine holds 4541 MB
    of 6144 MB VRAM and stop() released *zero*. The next engine then loaded on
    top — 9 GB requested on a 6 GB card — and WDDM silently paged GPU memory to
    host RAM, slowing every GPU stage ~30x with no error and no device
    fallback. ASR went from ~300 ms to ~13 s per utterance.
    """

    def test_stop_unloads_every_model_engine(self, app_paths: AppPaths) -> None:
        assistant = build_fake_assistant(Settings(), app_paths)
        assistant.preload()
        assert assistant.llm.loaded and assistant.asr.loaded and assistant.tts.loaded

        assistant.stop()

        assert not assistant.llm.loaded, "LLM weights still resident after stop()"
        assert not assistant.asr.loaded, "ASR weights still resident after stop()"
        assert not assistant.tts.loaded, "TTS weights still resident after stop()"

    def test_one_failing_unload_does_not_abort_teardown(self, app_paths: AppPaths) -> None:
        """Teardown is exception-proof (ADR-026): a component that cannot
        unload must not strand the other two in VRAM."""
        assistant = build_fake_assistant(Settings(), app_paths)
        assistant.preload()

        def boom() -> None:
            raise RuntimeError("driver wedged")

        assistant.llm.unload = boom  # type: ignore[method-assign]

        assistant.stop()

        assert not assistant.asr.loaded
        assert not assistant.tts.loaded
