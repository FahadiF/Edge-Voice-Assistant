"""Turn orchestration tests with fake engines — no models, no audio hardware.

These cover the M2/M3-critical control flow: streaming order, cancellation on
barge-in, superseding utterances, stale-artifact suppression, and failure
containment.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from typing import overload

import numpy as np
import pytest

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame
from eva.audio.segmenter import BargeIn, UtteranceEnd, UtteranceProgress
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core.events import (
    BargeInLatencyMeasured,
    Event,
    EventBus,
    FinalTranscript,
    LlmFinished,
    LlmStarted,
    PartialTranscript,
    SpeechFinished,
    TurnCancelled,
    TurnFinished,
)
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.tts.base import TTSEngine
from tests.server_fakes import FakeMemoryStore

AUDIO = np.ones(16_000, dtype=np.int16)


class FakeASR(ASREngine):
    def __init__(self, text: str = "hello assistant", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.prompts: list[str | None] = []  # records the bias prompt per call

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(
        self, audio: Frame, language: str | None = None, *, prompt: str | None = None
    ) -> TranscriptionResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("asr broken")
        return TranscriptionResult(text=self.text)


class FakeLLM(LLMEngine):
    def __init__(self, tokens: list[str] | None = None, delay_s: float = 0.0) -> None:
        self.tokens = tokens if tokens is not None else ["Hello ", "there. ", "All ", "good."]
        self.delay_s = delay_s
        self.aborted = False

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        # Honor max_tokens like the real engine does — the auto-title
        # generation (M5.4) relies on its 16-token cap being respected.
        for token in self.tokens[: params.max_tokens]:
            if should_abort():
                self.aborted = True
                return
            if self.delay_s:
                time.sleep(self.delay_s)
            yield token


class FakeTTS(TTSEngine):
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def synthesize(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Frame:
        self.synthesized.append(text)
        return np.ones(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["test-voice"]


class SlowCleanupTTS(TTSEngine):
    """Shaped like KokoroTTS's timing profile (not its API): multi-chunk
    streaming synthesis with a real per-chunk delay, and a slow `finally`
    block on close — simulating real, non-trivial phonemizer/session
    teardown cost on the stream's dedicated owner thread. Used to check
    whether a NEW turn's `synthesize_stream()` call can ever start while an
    OLD turn's cleanup is still in flight — real KokoroTTS's shared,
    non-thread-safe phonemizer state corrupts under exactly that overlap
    (measured separately, outside this suite, against the real engine)."""

    def __init__(self, chunk_delay_s: float = 0.03, close_delay_s: float = 0.1) -> None:
        self.chunk_delay_s = chunk_delay_s
        self.close_delay_s = close_delay_s
        self.close_in_progress = False
        self.overlap_detected = False
        self.calls = 0

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def synthesize(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Frame:
        return np.ones(1600, dtype=np.int16)

    def synthesize_stream(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Iterator[Frame]:
        self.calls += 1
        if self.close_in_progress:
            self.overlap_detected = True
        try:
            for _ in range(4):
                time.sleep(self.chunk_delay_s)
                yield np.ones(160, dtype=np.int16)
        finally:
            self.close_in_progress = True
            time.sleep(self.close_delay_s)
            self.close_in_progress = False

    def voices(self) -> list[str]:
        return ["test-voice"]


class FakeAudioOut:
    def __init__(self) -> None:
        self.spoken: list[Frame] = []
        self.stops = 0

    def say(self, pcm: Frame) -> None:
        self.spoken.append(pcm)

    def finish_utterance(self) -> None:
        pass

    def stop_speaking(self) -> None:
        self.stops += 1

    @property
    def is_speaking(self) -> bool:
        return False  # playback drains instantly in tests


@overload
def make_orchestrator(
    asr: FakeASR | None = None, llm: FakeLLM | None = None, tts: None = None
) -> tuple[Orchestrator, EventBus, FakeAudioOut, FakeTTS]: ...
@overload
def make_orchestrator(
    asr: FakeASR | None = None, llm: FakeLLM | None = None, *, tts: TTSEngine
) -> tuple[Orchestrator, EventBus, FakeAudioOut, TTSEngine]: ...
def make_orchestrator(
    asr: FakeASR | None = None,
    llm: FakeLLM | None = None,
    tts: TTSEngine | None = None,
) -> tuple[Orchestrator, EventBus, FakeAudioOut, TTSEngine]:
    settings = Settings()
    settings.conversation.system_prompt = "test"
    bus = EventBus()
    audio = FakeAudioOut()
    tts_engine = tts or FakeTTS()
    orch = Orchestrator(
        settings, bus, audio, asr or FakeASR(), llm or FakeLLM(), tts_engine, FakeMemoryStore()
    )
    return orch, bus, audio, tts_engine


async def drive(
    orch: Orchestrator,
    bus: EventBus,
    script: Callable[[], asyncio.Future[None] | object] | None = None,
    *,
    timeout: float = 20.0,
) -> list[Event]:
    """Run the orchestrator until shutdown; collect all published events."""
    queue = bus.subscribe()
    events: list[Event] = []

    async def collector() -> None:
        while True:
            events.append(await queue.get())

    collect_task = asyncio.create_task(collector())
    run_task = asyncio.create_task(orch.run())
    await asyncio.sleep(0)  # let run() bind the loop
    if script is not None:
        result = script()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result  # type: ignore[misc]
    orch.request_shutdown()
    await asyncio.wait_for(run_task, timeout)
    await asyncio.sleep(0.05)  # flush remaining events
    collect_task.cancel()
    return events


def names(events: list[Event]) -> list[str]:
    return [e.name for e in events]


async def wait_for_event(bus_events: list[Event], kind: type[Event], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(isinstance(e, kind) for e in bus_events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{kind.__name__} not observed within {timeout}s")


class TestNormalTurn:
    def test_complete_turn_event_order(self) -> None:
        async def scenario() -> None:
            orch, bus, audio, tts = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                # wait for the turn to complete
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            order = names(events)
            for expected in (
                "SpeechFinished",
                "TurnStarted",
                "FinalTranscript",
                "LlmStarted",
                "LlmToken",
                "LlmSentence",
                "TtsStarted",
                "TtsAudioReady",
                "LlmFinished",
                "TtsFinished",
                "TurnFinished",
            ):
                assert expected in order, f"missing {expected} in {order}"
            assert order.index("SpeechFinished") < order.index("TurnStarted")
            assert order.index("FinalTranscript") < order.index("LlmStarted")
            assert order.index("LlmStarted") < order.index("TtsStarted")
            # Audio actually reached the output.
            assert audio.spoken
            assert tts.synthesized == ["Hello there.", "All good."]
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.text == "Hello there. All good."
            speech_finished = next(e for e in events if isinstance(e, SpeechFinished))
            assert speech_finished.duration_ms == 800  # speech_ms from UtteranceEnd

        asyncio.run(scenario())

    def test_markdown_stripped_for_tts_but_canonical_in_storage_and_events(self) -> None:
        """ADR-024: the TTS engine must never receive Markdown formatting,
        while memory and events keep the raw Markdown untouched."""

        async def scenario() -> None:
            llm = FakeLLM(tokens=["My name is ", "**Edge Voice Assistant**. ", "Use `eva run`."])
            orch, bus, _audio, tts = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)

            # Spoken text: formatting characters gone, content intact.
            assert tts.synthesized == ["My name is Edge Voice Assistant.", "Use eva run."]

            # Event stream: raw Markdown (the web UI renders it).
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.text == "My name is **Edge Voice Assistant**. Use `eva run`."

            # Storage: raw Markdown is canonical (ADR-024).
            stored = orch.memory.recent_turns(orch.conversation_id, 10)
            assistant_turns = [t for t in stored if t.speaker == "assistant"]
            assert assistant_turns
            assert assistant_turns[-1].text == (
                "My name is **Edge Voice Assistant**. Use `eva run`."
            )

        asyncio.run(scenario())

    def test_text_turn_runs_full_pipeline_without_asr(self) -> None:
        """M5.3 composer: submit_text() must produce a complete turn — LLM
        reply, TTS audio, memory write — with the typed text as the final
        transcript and no ASR involvement."""

        async def scenario() -> None:
            asr = FakeASR()
            orch, bus, audio, _tts = make_orchestrator(asr=asr)

            async def script() -> None:
                assert orch.submit_text("Hello from the composer") is True
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            final = next(e for e in events if isinstance(e, FinalTranscript))
            assert final.text == "Hello from the composer"
            assert final.asr_ms == 0  # ASR stage skipped
            assert asr.calls == 0  # and never invoked
            assert audio.spoken  # reply was synthesized and played
            stored = orch.memory.recent_turns(orch.conversation_id, 10)
            assert any(t.text == "Hello from the composer" for t in stored)

        asyncio.run(scenario())

    def test_stated_name_persists_across_the_session(self) -> None:
        """The identity fix: once the user states their name, it stays in the
        prompt for every later turn — so the assistant can't know it for one
        question and forget it for the next (the reported contradiction)."""

        async def scenario() -> None:
            orch, bus, _audio, _tts = make_orchestrator()
            built_prompts: list[str] = []
            original_build = orch._context_builder.build

            def spy_build(conv_id: str, text: str, **kwargs: object) -> object:
                result = original_build(conv_id, text, **kwargs)
                built_prompts.append(result.messages[0].content)
                return result

            orch._context_builder.build = spy_build  # type: ignore[method-assign]

            async def wait_idle() -> None:
                for _ in range(300):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            async def script() -> None:
                orch.submit_text("Hi, my name is Fahad")
                await wait_idle()
                # A later, unrelated turn — the kind that used to lose the name.
                orch.submit_text("What is the weather like?")
                await wait_idle()

            await drive(orch, bus, script)
            assert orch._session_name == "Fahad"
            # The second (unrelated) turn's system prompt still carries the name.
            assert built_prompts, "no prompts captured"
            assert "The user's name is Fahad." in built_prompts[-1]

        asyncio.run(scenario())

    def test_asr_prompt_biases_with_the_stated_name(self) -> None:
        """The ASR fix: once the user's name is known this session, later
        transcriptions are given it as decoding context (measured to fix
        proper-noun mis-spelling like Fahad→Fahed) — via a spoken turn so the
        ASR path (not just typed) is exercised."""

        async def scenario() -> None:
            asr = FakeASR(text="my name is Fahad")
            orch, bus, _audio, _tts = make_orchestrator(asr=asr)

            async def wait_calls(n: int) -> None:
                # Wait until the ASR has been invoked n times AND that turn has
                # settled (a bare _turn_task.done() check races: it still points
                # at the previous, already-done turn right after feeding a new
                # utterance).
                for _ in range(500):
                    if asr.calls >= n and orch._turn_task is not None and orch._turn_task.done():
                        return
                    await asyncio.sleep(0.01)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_calls(1)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_calls(2)

            await drive(orch, bus, script)
            assert orch._session_name == "Fahad"
            assert asr.calls >= 2
            # The domain hint is always present; the name appears once captured
            # (turn 1 transcribed without it, turn 2 with it).
            assert all("Edge Voice Assistant" in (p or "") for p in asr.prompts)
            assert asr.prompts[0] is not None and "Fahad" not in asr.prompts[0]
            assert asr.prompts[-1] is not None and "Fahad" in asr.prompts[-1]

        asyncio.run(scenario())

    def test_turns_are_embedded_at_write_time(self) -> None:
        """M5.4 §1 root cause: nothing embedded new turns, so semantic
        retrieval had nothing to find in later conversations. Both sides of
        an exchange must now be embedded when a provider is wired."""
        import numpy as np

        from eva.embedding.base import EmbeddingProvider

        class FakeEmbedding(EmbeddingProvider):
            def load(self) -> None: ...
            def unload(self) -> None: ...

            def embed(self, text: str) -> np.ndarray:
                return np.full(4, float(len(text)), dtype=np.float32)

        async def scenario() -> None:
            settings = Settings()
            settings.conversation.system_prompt = "test"
            bus = EventBus()
            memory = FakeMemoryStore()
            orch = Orchestrator(
                settings,
                bus,
                FakeAudioOut(),
                FakeASR(),
                FakeLLM(),
                FakeTTS(),
                memory,
                embedding=FakeEmbedding(),
            )

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            await drive(orch, bus, script)
            rows = memory.embeddings_for(None)
            assert len(rows) == 2  # user turn + assistant turn
            for _turn_id, vector, dim in rows:
                assert dim == 4
                assert len(vector) == 4 * 4  # float32 bytes

        asyncio.run(scenario())

    def test_first_exchange_auto_titles_the_conversation(self) -> None:
        """M5.4 §2: the conversation gets an LLM-generated topic title after
        the first completed exchange — exactly one generation, even across
        later turns."""

        async def scenario() -> None:
            llm = FakeLLM()
            orch, bus, _audio, _tts = make_orchestrator(llm=llm)

            async def script() -> None:
                for _ in range(2):  # two turns; title generated once
                    orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                    for _ in range(200):
                        if orch._turn_task is not None and orch._turn_task.done():
                            break
                        await asyncio.sleep(0.01)

            await drive(orch, bus, script)
            conversations = orch.memory.all_conversations()
            active = next(c for c in conversations if c.id == orch.conversation_id)
            # FakeLLM answers "Hello there. All good." to everything —
            # including the title prompt; what matters is that a non-empty,
            # markdown-free title was stored.
            assert active.title
            assert "*" not in active.title

        asyncio.run(scenario())

    def test_submit_text_rejects_blank_and_before_run(self) -> None:
        orch, _bus, _audio, _tts = make_orchestrator()
        assert orch.submit_text("   ") is False  # blank
        assert orch.submit_text("hi") is False  # loop not bound yet (run() not started)

    def test_empty_transcript_skips_llm(self) -> None:
        async def scenario() -> None:
            orch, bus, audio, _ = make_orchestrator(asr=FakeASR(text=""))

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.2)

            events = await drive(orch, bus, script)
            assert "FinalTranscript" in names(events)
            assert "LlmStarted" not in names(events)
            assert "TurnFinished" in names(events)
            assert not audio.spoken

        asyncio.run(scenario())

    def test_history_carries_across_turns(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            assert names(events).count("TurnFinished") == 2
            assert len(orch.conversation_turns) == 2

        asyncio.run(scenario())


class TestMicrophoneMute:
    """M5.7: muting drops captured-speech events at the door; typed input
    still works; the state is broadcast so clients stay in sync."""

    def test_muted_microphone_ignores_captured_speech(self) -> None:
        async def scenario() -> None:
            orch, bus, _audio, _tts = make_orchestrator()

            async def script() -> None:
                assert orch.set_microphone_muted(True) is True
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.2)

            events = await drive(orch, bus, script)
            # A muted mic produces no turn at all.
            assert "TurnStarted" not in names(events)
            assert "FinalTranscript" not in names(events)
            # But the mute change was announced for clients.
            assert "MicrophoneMuted" in names(events)

        asyncio.run(scenario())

    def test_typed_input_works_while_muted(self) -> None:
        async def scenario() -> None:
            orch, bus, _audio, _tts = make_orchestrator()

            async def script() -> None:
                orch.set_microphone_muted(True)
                assert orch.submit_text("hello despite mute") is True
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            assert "TurnFinished" in names(events)
            assert len(orch.conversation_turns) == 1

        asyncio.run(scenario())

    def test_unmute_restores_listening(self) -> None:
        async def scenario() -> None:
            orch, bus, _audio, _tts = make_orchestrator()

            async def script() -> None:
                orch.set_microphone_muted(True)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.1)
                orch.set_microphone_muted(False)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            # Exactly one turn — the one fed after unmuting.
            assert names(events).count("TurnFinished") == 1

        asyncio.run(scenario())

    def test_set_same_state_is_noop_no_event(self) -> None:
        orch, bus, _audio, _tts = make_orchestrator()
        queue = bus.subscribe()
        assert orch.set_microphone_muted(False) is False  # already unmuted
        assert queue.empty()  # no redundant MicrophoneMuted published
        assert orch.microphone_muted is False


class TestCancellation:
    def test_barge_in_cancels_generation(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)  # ~2 s generation
            orch, bus, audio, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                await asyncio.sleep(0.1)  # some tokens flow
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await wait_for_event(collected, TurnCancelled)

            await drive(orch, bus, script)
            pump_task.cancel()
            assert llm.aborted
            assert audio.stops >= 1
            assert any(isinstance(e, TurnCancelled) and e.reason == "barge-in" for e in collected)

        asyncio.run(scenario())

    def test_new_utterance_supersedes_running_turn(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.15)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(4.0)  # let the second turn finish (slow CI margin)

            events = await drive(orch, bus, script, timeout=10)
            assert any(isinstance(e, TurnCancelled) and e.reason == "superseded" for e in events)
            finished = [e for e in events if isinstance(e, TurnFinished)]
            assert len(finished) >= 1

        asyncio.run(scenario())

    def test_no_speech_after_cancellation(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["Sentence one. "] + ["tok "] * 80, delay_s=0.02)
            orch, bus, audio, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.15)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.3)
                spoken_at_cancel = len(audio.spoken)
                await asyncio.sleep(0.3)
                assert len(audio.spoken) == spoken_at_cancel  # nothing spoken after

            await drive(orch, bus, script)

        asyncio.run(scenario())

    def test_rapid_double_barge_in_stays_clean(self) -> None:
        """Two barge-ins fired back-to-back (e.g. a false start immediately
        followed by the real interruption) must not crash, must not leave a
        stale turn task running, and must settle to a clean listening state.
        `_dispatch` serializes events one at a time, so the second BargeIn is
        handled only once the first cancellation has fully settled — this
        confirms that sequence is safe even with nothing left to cancel the
        second time."""

        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, audio, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())
            epoch_before = orch.current_epoch

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.2)

            await drive(orch, bus, script)
            pump_task.cancel()

            assert orch.current_epoch >= epoch_before + 2  # both barge-ins advanced the epoch
            assert audio.stops >= 2
            assert orch.state in ("listening", "idle")
            assert orch._turn_task is None or orch._turn_task.done()
            cancelled = [e for e in collected if isinstance(e, TurnCancelled)]
            assert any(e.reason == "barge-in" for e in cancelled)

        asyncio.run(scenario())

    def test_barge_in_publishes_latency_measurement(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, _, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await wait_for_event(collected, BargeInLatencyMeasured)

            await drive(orch, bus, script)
            pump_task.cancel()
            measured = next(e for e in collected if isinstance(e, BargeInLatencyMeasured))
            assert measured.detected_to_silent_ms >= 0
            assert orch.barge_in_count == 1
            assert orch.last_barge_in_latency_ms == measured.detected_to_silent_ms

        asyncio.run(scenario())

    def test_repeated_barge_ins_stay_clean(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 50, delay_s=0.01)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                for _ in range(5):
                    orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                    await asyncio.sleep(0.08)
                    orch.feed_audio_event(BargeIn(speech_ms=200))
                    await asyncio.sleep(0.05)
                # Final turn must still complete normally.
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(1.5)

            events = await drive(orch, bus, script, timeout=15)
            cancelled = [e for e in events if isinstance(e, TurnCancelled)]
            assert len(cancelled) >= 4
            finished = [e for e in events if isinstance(e, TurnFinished) and e.error is None]
            assert finished

        asyncio.run(scenario())


class TestFailures:
    def test_asr_failure_finishes_turn_with_error(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(asr=FakeASR(fail=True))

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            finished = next(e for e in events if isinstance(e, TurnFinished))
            assert finished.error is not None

        asyncio.run(scenario())

    def test_llm_failure_does_not_hang(self) -> None:
        class BrokenLLM(FakeLLM):
            def stream(
                self,
                messages: list[ChatMessage],
                params: GenerationParams,
                should_abort: Callable[[], bool],
            ) -> Iterator[str]:
                raise RuntimeError("llm broken")
                yield ""  # pragma: no cover

        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(llm=BrokenLLM())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.5)

            events = await drive(orch, bus, script)
            assert "TurnFinished" in names(events)

        asyncio.run(scenario())


class TestPartials:
    def test_partial_transcript_published(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(asr=FakeASR(text="partial words"))

            async def script() -> None:
                orch.feed_audio_event(UtteranceProgress(AUDIO, 1200))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            partials = [e for e in events if isinstance(e, PartialTranscript)]
            assert partials and partials[0].text == "partial words"

        asyncio.run(scenario())

    def test_partials_disabled_by_setting(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()
            orch._settings.asr.partial_transcripts = False

            async def script() -> None:
                orch.feed_audio_event(UtteranceProgress(AUDIO, 1200))
                await asyncio.sleep(0.2)

            events = await drive(orch, bus, script)
            assert not any(isinstance(e, PartialTranscript) for e in events)

        asyncio.run(scenario())


class TestQueueBackpressure:
    def test_bounded_token_queue_delivers_every_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shrink the token queue to force real wraparound; a slow-draining
        consumer (slow FakeTTS-driven speak_worker) must apply backpressure
        to the producer thread rather than lose or crash on tokens."""
        import eva.conversation.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "_TOKEN_QUEUE_MAXSIZE", 2)
        monkeypatch.setattr(orch_mod, "_SENTENCE_QUEUE_MAXSIZE", 1)

        async def scenario() -> None:
            tokens = [f"word{i} " for i in range(40)]
            llm = FakeLLM(tokens=tokens)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(300):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.tokens == len(tokens)
            assert finished.text == "".join(tokens).strip()

        asyncio.run(scenario())

    def test_tight_bounds_and_short_timeout_never_crash_the_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: with a 1-slot queue and a near-zero backpressure
        timeout, the pipeline must still complete cleanly (no QueueFull
        crash, no hang) even in the tightest configuration."""
        import eva.conversation.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "_TOKEN_QUEUE_MAXSIZE", 1)
        monkeypatch.setattr(orch_mod, "_QUEUE_BACKPRESSURE_TIMEOUT_S", 0.05)

        async def scenario() -> None:
            llm = FakeLLM(tokens=["a "] * 10)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(300):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            # Must not raise/hang even though some tokens may be dropped.
            events = await drive(orch, bus, script)
            assert "TurnFinished" in names(events)

        asyncio.run(scenario())


class TestMetrics:
    def test_metrics_recorded_for_completed_turn(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.5)

            await drive(orch, bus, script)
            turns = orch.metrics.turns
            assert len(turns) == 1
            assert turns[0].tokens == 4
            assert turns[0].ttfa_ms > 0
            assert not turns[0].cancelled

        asyncio.run(scenario())


class TestTtsCleanupSerialization:
    """A TTS engine that holds shared, non-thread-safe state (real KokoroTTS:
    one phonemizer/session instance) must never have two synthesize_stream()
    calls in flight at once — measured separately to corrupt/crash real
    Kokoro. `FakeTTS` completes instantly, so it can't expose whether a NEW
    turn's TTS call could start while an OLD turn's cleanup (a slow `finally`
    on `_drive_stream`'s dedicated owner thread) is still running;
    `SlowCleanupTTS` gives that cleanup real wall-clock duration so an
    overlap — if the orchestrator's cancellation sequencing had a hole in
    it — would actually be observable."""

    def test_no_overlap_across_rapid_double_barge_in_and_concurrent_interrupt(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["word "] * 60, delay_s=0.005)
            tts = SlowCleanupTTS(chunk_delay_s=0.03, close_delay_s=0.15)
            orch, bus, _, _ = make_orchestrator(llm=llm, tts=tts)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                deadline = time.monotonic() + 5.0
                while tts.calls == 0 and time.monotonic() < deadline:
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.01)  # ensure we're mid-chunk, not already done

                # Rapid double barge-in, back-to-back with zero gap — the
                # exact adversarial shape `test_rapid_double_barge_in_stays_
                # clean` covers for event ordering; here it's covered for TTS
                # cleanup overlap instead.
                orch.feed_audio_event(BargeIn(speech_ms=200))
                orch.feed_audio_event(BargeIn(speech_ms=200))

                # The API's interrupt() is a second, independent entry point
                # into _cancel_turn (same event loop, different call path) —
                # schedule it as its own task so it is genuinely interleaved
                # by the loop with the barge-ins' in-flight cancellation,
                # not just sequentially awaited after them.
                interrupt_task = asyncio.create_task(orch.interrupt())
                await asyncio.sleep(0.2)
                await interrupt_task

                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.5)

            await drive(orch, bus, script, timeout=15)

            assert tts.calls >= 2  # the second UtteranceEnd's turn did reach TTS
            assert not tts.overlap_detected, (
                "a new synthesize_stream() call started while a previous "
                "call's cleanup was still running — this would corrupt a "
                "real, non-thread-safe TTS engine like KokoroTTS"
            )

        asyncio.run(scenario())
