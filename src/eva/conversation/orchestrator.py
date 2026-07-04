"""Turn orchestrator: segmenter events in, spoken responses out.

The asyncio composition of the whole pipeline (ADR-006):

  capture thread ──feed_audio_event()──► event queue ──► run() loop
      UtteranceEnd  → start a turn task (ASR → LLM tokens → sentences → TTS)
      BargeIn       → advance the epoch, stop playback, cancel the turn task
      UtteranceProgress → opportunistic partial transcript

Inside a turn, three concurrent units pipeline the response:
  producer thread   llm.stream() pushes tokens into an asyncio queue,
                    aborting via epoch staleness (per-token check);
  token consumer    publishes LlmToken events and feeds the sentence chunker;
  speak worker      synthesizes sentences sequentially and queues playback —
                    sentence N plays while N+1 synthesizes while tokens arrive.

Every hand-off point checks the turn epoch; a stale turn stops at the next
boundary and its artifacts are never spoken. The orchestrator holds no model
code: engines are ports, audio out is a three-method protocol, so the entire
control flow is unit-testable with fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Protocol

from eva.asr.base import ASREngine
from eva.audio.frames import Frame
from eva.audio.segmenter import (
    BargeIn,
    SegmenterEvent,
    SpeechStart,
    UtteranceDiscarded,
    UtteranceEnd,
    UtteranceProgress,
)
from eva.config.settings import Settings
from eva.conversation.chunker import SentenceChunker
from eva.conversation.history import ConversationHistory
from eva.conversation.language import (
    effective_asr_language,
    effective_system_prompt,
    effective_voice,
    resolve_language,
)
from eva.core.events import (
    BargeInDetected,
    EventBus,
    FinalTranscript,
    LlmFinished,
    LlmSentence,
    LlmStarted,
    LlmToken,
    PartialTranscript,
    SpeechStarted,
    StateChanged,
    TtsAudioReady,
    TtsFinished,
    TtsStarted,
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)
from eva.core.turn import TurnController
from eva.llm.base import GenerationParams, LLMEngine
from eva.metrics.turn import MetricsCollector, TurnMetrics
from eva.tts.base import TTSEngine

logger = logging.getLogger(__name__)

_PLAYBACK_POLL_S = 0.05


class AudioOutput(Protocol):
    """The only audio surface the orchestrator needs."""

    def say(self, pcm: Frame) -> None: ...

    def stop_speaking(self) -> None: ...

    @property
    def is_speaking(self) -> bool: ...


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        audio_out: AudioOutput,
        asr: ASREngine,
        llm: LLMEngine,
        tts: TTSEngine,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._audio = audio_out
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._metrics = metrics or MetricsCollector()
        self._controller = TurnController()
        language = resolve_language(settings)
        self._asr_language = effective_asr_language(settings, language)
        self._voice = effective_voice(settings, language)
        self._history = ConversationHistory(
            effective_system_prompt(settings, language),
            max_turns=settings.conversation.max_history_turns,
        )
        self._params = GenerationParams(
            temperature=settings.conversation.temperature,
            top_p=settings.conversation.top_p,
            max_tokens=settings.conversation.max_tokens,
            stop=tuple(settings.conversation.stop_sequences),
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state: str = "idle"
        self._events: asyncio.Queue[SegmenterEvent | None] = asyncio.Queue()
        self._turn_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._partial_busy = False

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def state(self) -> str:
        """Current pipeline state: idle / listening / thinking / speaking."""
        return self._state

    @property
    def pending_audio_events(self) -> int:
        """Depth of the capture→orchestrator event queue (diagnostics)."""
        return self._events.qsize()

    @property
    def current_epoch(self) -> int:
        return self._controller.epoch

    def _set_state(self, state: str) -> None:
        self._state = state
        self._bus.publish(StateChanged(state=state))

    # ── input bridge (called from the capture thread) ──

    def feed_audio_event(self, event: SegmenterEvent) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def request_shutdown(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._events.put_nowait, None)

    # ── main loop ──

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._bus.bind_loop(self._loop)
        self._set_state("listening")
        try:
            while True:
                event = await self._events.get()
                if event is None:
                    break
                await self._dispatch(event)
        finally:
            await self._cancel_turn("shutdown")
            self._set_state("idle")

    async def _dispatch(self, event: SegmenterEvent) -> None:
        match event:
            case SpeechStart():
                self._bus.publish(SpeechStarted(epoch=self._controller.epoch))
            case BargeIn():
                self._bus.publish(BargeInDetected(epoch=self._controller.epoch))
                await self._cancel_turn("barge-in")
            case UtteranceEnd(audio=audio):
                # A new utterance always supersedes whatever is in flight
                # (covers half-duplex mode, where no BargeIn event exists).
                if self._turn_task is not None and not self._turn_task.done():
                    await self._cancel_turn("superseded")
                self._turn_task = asyncio.create_task(self._run_turn(audio))
            case UtteranceProgress(audio=audio):
                if self._settings.asr.partial_transcripts and not self._partial_busy:
                    self._partial_task = asyncio.create_task(self._partial_transcribe(audio))
            case UtteranceDiscarded():
                logger.debug("Utterance discarded by noise gate")

    async def _cancel_turn(self, reason: str) -> None:
        stale_epoch = self._controller.epoch
        self._controller.advance()  # all in-flight work is now stale
        self._audio.stop_speaking()
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._bus.publish(TurnCancelled(epoch=stale_epoch, reason=reason))
            self._set_state("listening")

    # ── partial transcription (best-effort, never blocks the pipeline) ──

    async def _partial_transcribe(self, audio: Frame) -> None:
        self._partial_busy = True
        try:
            epoch = self._controller.epoch
            result = await asyncio.to_thread(self._asr.transcribe, audio, self._asr_language)
            if result.text and self._controller.is_current(epoch):
                self._bus.publish(PartialTranscript(epoch=epoch, text=result.text))
        except Exception:
            logger.exception("Partial transcription failed")
        finally:
            self._partial_busy = False

    # ── the turn pipeline ──

    async def _run_turn(self, audio: Frame) -> None:
        epoch = self._controller.advance()
        t0 = time.perf_counter()
        self._bus.publish(TurnStarted(epoch=epoch))
        self._set_state("thinking")
        error: str | None = None
        metrics = TurnMetrics(epoch=epoch)
        try:
            metrics = await self._pipeline(epoch, audio, t0)
        except asyncio.CancelledError:
            self._metrics.record(metrics.model_copy(update={"cancelled": True}))
            raise
        except Exception as exc:
            logger.exception("Turn %d failed", epoch)
            error = str(exc)
        self._metrics.record(metrics)
        self._bus.publish(TurnFinished(epoch=epoch, error=error))
        if self._controller.is_current(epoch):
            self._set_state("listening")

    async def _pipeline(self, epoch: int, audio: Frame, t0: float) -> TurnMetrics:
        def elapsed_ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        # ── ASR ──
        result = await asyncio.to_thread(self._asr.transcribe, audio, self._asr_language)
        asr_ms = elapsed_ms()
        user_text = result.text.strip()
        if self._controller.is_stale(epoch):
            return TurnMetrics(epoch=epoch, asr_ms=asr_ms, cancelled=True)
        self._bus.publish(FinalTranscript(epoch=epoch, text=user_text, asr_ms=asr_ms))
        if not user_text:
            return TurnMetrics(epoch=epoch, asr_ms=asr_ms)

        # ── LLM producer thread → token queue ──
        self._bus.publish(LlmStarted(epoch=epoch))
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue[str | None] = asyncio.Queue()
        messages = self._history.messages(user_text)

        def produce() -> None:
            def push(item: str | None) -> None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(tokens.put_nowait, item)

            try:
                for token in self._llm.stream(
                    messages, self._params, should_abort=lambda: self._controller.is_stale(epoch)
                ):
                    push(token)
            except Exception:
                logger.exception("LLM generation failed")
            finally:
                push(None)

        producer = asyncio.create_task(asyncio.to_thread(produce))

        # ── speak worker: sentences → TTS → playback ──
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        first_audio_ms = 0
        tts_first_ms = 0

        async def speak_worker() -> None:
            nonlocal first_audio_ms, tts_first_ms
            started = False
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                if self._controller.is_stale(epoch):
                    continue  # drain without speaking
                if not started:
                    started = True
                    self._bus.publish(TtsStarted(epoch=epoch))
                    self._set_state("speaking")
                synth_start = time.perf_counter()
                pcm = await asyncio.to_thread(
                    self._tts.synthesize,
                    sentence,
                    voice=self._voice,
                    speed=self._settings.tts.speed,
                )
                if self._controller.is_stale(epoch):
                    continue
                if first_audio_ms == 0:
                    tts_first_ms = int((time.perf_counter() - synth_start) * 1000)
                    first_audio_ms = elapsed_ms()
                    self._bus.publish(TtsAudioReady(epoch=epoch, ttfa_ms=first_audio_ms))
                self._audio.say(pcm)

        speaker = asyncio.create_task(speak_worker())

        # ── token consumer ──
        chunker = SentenceChunker(
            min_chars=self._settings.conversation.sentence_min_chars,
            max_chars=self._settings.conversation.sentence_max_chars,
        )
        reply_parts: list[str] = []
        token_count = 0
        ttft_ms = 0
        llm_start = time.perf_counter()
        try:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                if self._controller.is_stale(epoch):
                    continue  # keep draining so the producer can finish
                if ttft_ms == 0:
                    ttft_ms = elapsed_ms()
                token_count += 1
                reply_parts.append(token)
                self._bus.publish(LlmToken(epoch=epoch, token=token))
                for sentence in chunker.feed(token):
                    self._bus.publish(LlmSentence(epoch=epoch, text=sentence))
                    sentences.put_nowait(sentence)
            tail = chunker.flush()
            if tail is not None and self._controller.is_current(epoch):
                self._bus.publish(LlmSentence(epoch=epoch, text=tail))
                sentences.put_nowait(tail)
        finally:
            sentences.put_nowait(None)
            await asyncio.gather(producer, speaker)

        llm_ms = int((time.perf_counter() - llm_start) * 1000)
        reply = "".join(reply_parts).strip()
        if self._controller.is_stale(epoch):
            return TurnMetrics(
                epoch=epoch, asr_ms=asr_ms, ttft_ms=ttft_ms, tokens=token_count, cancelled=True
            )
        self._bus.publish(
            LlmFinished(
                epoch=epoch, text=reply, tokens=token_count, ttft_ms=ttft_ms, duration_ms=llm_ms
            )
        )
        if reply:
            self._history.add_turn(user_text, reply)

        # ── wait for playback to drain (stays interruptible) ──
        while self._audio.is_speaking and self._controller.is_current(epoch):
            await asyncio.sleep(_PLAYBACK_POLL_S)
        self._bus.publish(TtsFinished(epoch=epoch))

        return TurnMetrics(
            epoch=epoch,
            asr_ms=asr_ms,
            ttft_ms=ttft_ms,
            llm_ms=llm_ms,
            tokens=token_count,
            tts_first_ms=tts_first_ms,
            ttfa_ms=first_audio_ms,
            total_ms=elapsed_ms(),
        )
