"""Typed engine events and the event bus.

Events are the engine's public narration: every stage of a turn publishes what
it is doing, and consumers (CLI today; WebSocket clients, UI panels, and plugins
later) subscribe instead of polling. Events are immutable pydantic models so
they serialize to JSON for the API without a translation layer.

The bus is asyncio-native: each subscriber owns a bounded queue drained by its
own task. `publish()` never blocks — a slow subscriber loses oldest events
rather than stalling the pipeline (the same fresh-data-wins policy as the audio
rings). `publish_threadsafe()` lets worker/audio threads publish safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

_SUBSCRIBER_QUEUE_SIZE = 256


FinishReason = Literal["stop", "length", "abort", "error"]
"""Why generation ended.

- `stop`   — the model emitted its end-of-turn token: the reply is complete.
- `length` — the `max_tokens` ceiling was hit: the reply is CUT OFF mid-thought.
- `abort`  — `should_abort()` went True (barge-in, supersede): deliberately stopped.
- `error`  — the adapter raised.

`length` is the one that must never be mistaken for `stop`. Before M7.3 the
adapter discarded llama.cpp's reason entirely, so a reply truncated at 512
tokens was stored, replayed into the next turn's history, and described by the
model as complete — which is exactly what a user then argues with.

Lives in `core` rather than `eva.llm` so `core.events` can name it without
importing a subsystem: dependencies point inward (ADR-010).
"""


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    @property
    def name(self) -> str:
        return type(self).__name__


# ── Turn lifecycle ──


class TurnStarted(Event):
    epoch: int


class TurnFinished(Event):
    epoch: int
    error: str | None = None


class TurnCancelled(Event):
    epoch: int
    reason: Literal["barge-in", "superseded", "shutdown", "manual"]


# ── Capture / ASR ──


class SpeechStarted(Event):
    epoch: int


class SpeechFinished(Event):
    epoch: int
    duration_ms: int


class BargeInDetected(Event):
    epoch: int  # epoch of the *cancelled* turn


class BargeInLatencyMeasured(Event):
    """Audible-stop latency for a barge-in (M3 validation target: < 150 ms).

    `detected_to_silent_ms` is measured from the moment the turn epoch was
    advanced (cancellation started) to the moment playback reports silent —
    best-effort: if a new utterance starts speaking again before the old
    audio drains, this can read as capped/elevated rather than accurate,
    since playback silence is observed, not epoch-tagged.
    """

    epoch: int  # epoch of the *cancelled* turn
    detected_to_silent_ms: int


class PartialTranscript(Event):
    epoch: int
    text: str


class FinalTranscript(Event):
    epoch: int
    text: str
    asr_ms: int


# ── LLM ──


class LlmStarted(Event):
    epoch: int


class LlmToken(Event):
    epoch: int
    token: str


class LlmSentence(Event):
    epoch: int
    text: str


class LlmFinished(Event):
    epoch: int
    text: str
    tokens: int
    ttft_ms: int
    duration_ms: int
    finish_reason: FinishReason = "stop"
    """Why generation ended. `length` means `text` is cut off mid-thought —
    consumers must not present it as a finished answer (M7.3)."""
    speakable_end: int = -1
    """Character offset in `text` after which nothing will ever be spoken
    (a trailing code fence, table, or other display-only content). -1 when
    not computed. Lets a speech-paced view reveal that tail as soon as the
    cursor reaches it, instead of holding it back until the turn ends —
    it cannot run ahead of speech, because it is never spoken (ADR-028)."""


# ── TTS / playback ──


class TtsStarted(Event):
    epoch: int


class TtsAudioReady(Event):
    epoch: int
    ttfa_ms: int  # time from utterance end to first audio queued


class TtsSentenceStarted(Event):
    """One sentence of the reply just started coming out of the speaker
    (M7.1, ADR-028) — published from the playback clock, not when the audio
    was synthesized or queued.

    This is the synchronization point display surfaces use to reveal text at
    speaking pace instead of at generation pace. `text` is the raw segment as
    published by `LlmSentence` (Markdown intact — the speech filter's output
    is what reaches the TTS engine, not what is displayed), and `index` is its
    1-based position in the reply, so a client can order/deduplicate without
    string matching.
    """

    epoch: int
    index: int
    text: str


class TtsFinished(Event):
    epoch: int


class StateChanged(Event):
    state: Literal["idle", "listening", "thinking", "speaking"]


class MicrophoneMuted(Event):
    """The user muted/unmuted the microphone (M5.7). Muted = the assistant
    stops acting on captured speech (typed chat still works); the audio
    device stays open so echo cancellation and playback are unaffected."""

    muted: bool


# ── Model management (platform API) ──


class ModelDownloadProgress(Event):
    model_id: str
    filename: str
    bytes_done: int
    bytes_total: int


class ModelDownloadCompleted(Event):
    model_id: str


class ModelDownloadFailed(Event):
    model_id: str
    error: str


# ── Engine lifecycle (platform API) ──


class ComponentLoadStarted(Event):
    """A model/component began loading during engine startup (M5.5,
    ADR-026) — drives the per-component startup progress UI."""

    component: str  # "llm" | "asr" | "tts" | "embedding" | "audio"
    label: str  # human-readable, e.g. "Loading language model…"


class ComponentLoadFinished(Event):
    component: str
    ms: int
    error: str = ""  # non-empty when the component failed to load


class EngineStarted(Event):
    pass


class EngineStopped(Event):
    pass


class ErrorOccurred(Event):
    """Surfaced to clients for errors that happen outside a specific turn
    (e.g. a background model download failure)."""

    message: str
    context: str = ""


class _StreamClosed(Event):
    """Internal wake-up pushed to every subscriber on `EventBus.close()` so
    long-lived consumers (WebSocket streams) return promptly at shutdown
    instead of being cancelled mid-`get()` (M5.7). Never serialized to a
    client — consumers check identity against `STREAM_CLOSED` and stop."""


STREAM_CLOSED: Event = _StreamClosed()


class EventBus:
    """Fan-out pub/sub bound to one asyncio loop."""

    def __init__(self, history_size: int = 100) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: deque[Event] = deque(maxlen=history_size)
        self._closed = False

    def recent_events(self) -> list[Event]:
        """The most recent published events (diagnostics; newest last)."""
        return list(self._history)

    @property
    def closed(self) -> bool:
        return self._closed

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the loop used by `publish_threadsafe` (call once at startup)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.append(queue)
        # A subscriber that connects after close() (e.g. a request racing
        # shutdown) is told immediately, so it never blocks forever.
        if self._closed:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(STREAM_CLOSED)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    def close(self) -> None:
        """Wake every subscriber with `STREAM_CLOSED` so streaming consumers
        exit their `get()` loop promptly at shutdown (M5.7). Without this, an
        open WebSocket task blocks in `queue.get()` until uvicorn's graceful
        timeout cancels it — which logs "Cancel N running task(s)" and adds a
        multi-second wait. Idempotent."""
        self._closed = True
        for queue in self._subscribers:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # make room; the sentinel must land
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(STREAM_CLOSED)

    def publish(self, event: Event) -> None:
        """Publish from the event-loop thread. Never blocks."""
        self._history.append(event)
        for queue in self._subscribers:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # drop oldest
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread; no-op (with a log) before the loop is bound."""
        if self._loop is None or self._loop.is_closed():
            logger.debug("Event %s dropped: bus not bound to a loop", event.name)
            return
        self._loop.call_soon_threadsafe(self.publish, event)
