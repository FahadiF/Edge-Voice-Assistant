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
    TurnCancelled,
    TurnFinished,
)
from eva.engine import Assistant


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
        case LlmFinished(text=text, tokens=tokens, ttft_ms=ttft, duration_ms=dur):
            speed = tokens / (dur / 1000) if dur else 0.0
            return f"Assistant: {text}\n  [{tokens} tokens, ttft {ttft} ms, {speed:.1f} tok/s]"
        case TurnFinished(error=error) if error:
            return f"(turn failed: {error})"
    return None


async def run_voice_loop(assistant: Assistant) -> None:
    queue = assistant.bus.subscribe()

    async def render_events() -> None:
        token_line_open = False
        while True:
            event = await queue.get()
            if isinstance(event, LlmToken):
                # Stream tokens inline as they arrive.
                print(event.token, end="", flush=True)
                token_line_open = True
                continue
            line = _render(event)
            if line is not None:
                if token_line_open:
                    print()
                    token_line_open = False
                print(line, flush=True)

    renderer = asyncio.create_task(render_events())
    try:
        await assistant.orchestrator.run()
    finally:
        renderer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renderer


def main_run(assistant: Assistant) -> int:
    print("Loading models — this can take a minute on first run...")
    assistant.preload()
    assistant.start_audio()
    print("\nReady. Speak into the microphone; interrupt any time by talking over it.")
    print("Ctrl+C to exit.\n")
    try:
        asyncio.run(run_voice_loop(assistant))
    except KeyboardInterrupt:
        pass
    finally:
        assistant.stop()
        print("\n" + assistant.orchestrator.metrics.summary())
    return 0
