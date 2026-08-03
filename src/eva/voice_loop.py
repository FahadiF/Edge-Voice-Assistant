"""Interactive voice loop for the CLI (`eva run`).

Subscribes to the event bus and renders a live console view of the
conversation. All logic lives in the orchestrator; this is presentation only.
"""

from __future__ import annotations

import asyncio
import contextlib

from eva.core.events import (
    BargeInDetected,
    Event,
    FinalTranscript,
    LlmFinished,
    LlmToken,
    PartialTranscript,
    StateChanged,
    TtsAudioReady,
    TtsSentenceStarted,
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)
from eva.engine import Assistant
from eva.llm.base import engine_device


def _render(event: Event) -> str | None:
    match event:
        case StateChanged(state=state):
            return f"[{state}]"
        case PartialTranscript(text=text):
            return f"  … {text}"
        case FinalTranscript(text=text):
            return f"You: {text}" if text else "You: (nothing recognized)"
        case BargeInDetected():
            return "— interrupted —"
        case TurnCancelled(reason=reason):
            return f"(turn cancelled: {reason})"
        case TtsAudioReady(ttfa_ms=ttfa):
            return f"  [first audio after {ttfa} ms]"
        case TurnFinished(error=error) if error:
            return f"(turn failed: {error})"
    return None


def _metrics_line(event: LlmFinished) -> str:
    speed = event.tokens / (event.duration_ms / 1000) if event.duration_ms else 0.0
    return f"  [{event.tokens} tokens, ttft {event.ttft_ms} ms, {speed:.1f} tok/s]"


class ConsoleRenderer:
    """Console view of the conversation — presentation only, no pipeline state.

    With `ui.sync_text_to_speech` on (default, M7.1) the reply is written out
    sentence by sentence as each one *starts being spoken* (`TtsSentenceStarted`,
    ADR-028), so the console never runs ahead of the voice, and the per-turn
    metrics line is held back until the turn ends rather than landing mid-reply.
    With it off, tokens stream to the console as the model produces them (the
    pre-M7.1 behavior).
    """

    def __init__(self, *, sync_to_speech: bool = True) -> None:
        self._sync = sync_to_speech
        self._inline_open = False
        self._reply: str | None = None  # authoritative text, for the fallback
        self._metrics: str | None = None
        self._spoken_any = False

    def handle(self, event: Event) -> None:
        match event:
            case TurnStarted():
                self._reset()
            case LlmToken(token=token):
                if not self._sync:
                    self._write_inline(token)
            case TtsSentenceStarted(text=text):
                if not self._sync:
                    return
                self._write_inline(("Assistant: " if not self._spoken_any else " ") + text)
                self._spoken_any = True
            case LlmFinished() as finished:
                self._reply = finished.text
                self._metrics = _metrics_line(finished)
                if not self._sync:
                    # Generation is the pace: print the whole reply and its
                    # metrics right away, exactly as before M7.1.
                    self._close_inline()
                    self._print(f"Assistant: {finished.text}")
                    self._flush_metrics()
            case TurnFinished():
                self._close_inline()
                if self._sync and not self._spoken_any and self._reply:
                    # Nothing was ever spoken (e.g. TTS unavailable) — the reply
                    # must still reach the user.
                    self._print(f"Assistant: {self._reply}")
                self._flush_metrics()
                line = _render(event)
                if line is not None:
                    self._print(line)
                self._reset()
            case _:
                line = _render(event)
                if line is not None:
                    self._close_inline()
                    self._print(line)

    def _reset(self) -> None:
        self._reply = None
        self._metrics = None
        self._spoken_any = False

    def _flush_metrics(self) -> None:
        if self._metrics is not None:
            self._print(self._metrics)
            self._metrics = None

    def _write_inline(self, text: str) -> None:
        print(text, end="", flush=True)
        self._inline_open = True

    def _close_inline(self) -> None:
        if self._inline_open:
            print()
            self._inline_open = False

    def _print(self, line: str) -> None:
        self._close_inline()
        print(line, flush=True)


async def run_voice_loop(assistant: Assistant) -> None:
    queue = assistant.bus.subscribe()

    async def render_events() -> None:
        renderer = ConsoleRenderer(sync_to_speech=assistant.settings.ui.sync_text_to_speech)
        while True:
            renderer.handle(await queue.get())

    renderer = asyncio.create_task(render_events())
    try:
        await assistant.orchestrator.run()
    finally:
        renderer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renderer


def _startup_banner(assistant: Assistant) -> None:
    """Show exactly which profile, models, and devices are active (ADR-015),
    plus which persona/user profile/voice/memory state is active (M4) —
    without this, there was no on-screen confirmation the M4 subsystems were
    doing anything."""
    from eva.conversation.language import effective_voice, resolve_language
    from eva.conversation.personas import resolve_persona
    from eva.hardware import detect_hardware, recommend_profile
    from eva.models.manager import ModelManager

    settings = assistant.settings
    tier = recommend_profile(detect_hardware())

    def display_name(model_id: str) -> str:
        from eva.config.paths import get_app_paths

        try:
            return ModelManager(get_app_paths()).info(model_id).display_name
        except Exception:
            return model_id

    print(f"\nProfile: {settings.profile} (hardware tier: {tier.display_name})")
    print(f"  LLM: {display_name(settings.llm.model)}  [{engine_device(assistant.llm)}]")
    print(f"  ASR: {display_name(settings.asr.model)}  [{assistant.asr.device}]")
    print(f"  TTS: {display_name(settings.tts.model)}  [{assistant.tts.device}]")
    print(f"  VAD: {settings.vad.engine}  [cpu]")
    print(f"  Language: {settings.conversation.language}")

    persona = resolve_persona(settings)
    active_profile = assistant.profiles.active()
    user_line = f"{active_profile.nickname or active_profile.id}" if active_profile else "none"
    memory_stats = assistant.memory.stats()
    print(f"  Persona: {persona.display_name} ({persona.id})")
    print(f"  User profile: {user_line}")
    print(f"  Voice: {effective_voice(settings, resolve_language(settings))}")
    print(
        f"  Memory: {memory_stats.conversation_count} conversation(s), "
        f"{memory_stats.turn_count} turn(s) stored"
    )


def main_run(assistant: Assistant) -> int:
    """Load models, start the voice loop, and always exit cleanly on Ctrl+C —
    whether the interrupt lands during model loading, audio startup, or an
    active conversation (M3: no stage should ever surface a raw traceback)."""
    try:
        print("Loading models — this can take a minute on first run...")
        assistant.preload()
        _startup_banner(assistant)
        assistant.start_audio()
        print("\nReady. Speak into the microphone; interrupt any time by talking over it.")
        print("Ctrl+C to exit.\n")
        asyncio.run(run_voice_loop(assistant))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        assistant.stop()
        if assistant.orchestrator.metrics.turns:
            print("\n" + assistant.orchestrator.metrics.summary())
    return 0
