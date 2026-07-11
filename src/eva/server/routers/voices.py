"""Voice API (M4, ADR-022): list voices for the active TTS engine, preview
one. A running engine is required — `voice_registry` is normally populated
during `Assistant.preload()`; with `tts.lazy_load` (M5.5) the first request
here triggers the load + registration instead.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Response

from eva.server.deps import StateDep
from eva.server.schemas import VoicePreviewRequest
from eva.tts.voices import VoiceInfo, preview_text, register_voices_for_engine, voices_for_engine

router = APIRouter(prefix="/voices", tags=["voices"])


@router.get("", response_model=list[VoiceInfo])
async def list_voices(state: StateDep) -> list[VoiceInfo]:
    assistant = state.require_assistant()
    engine_id = assistant.settings.tts.engine
    voices = voices_for_engine(engine_id)
    if not voices:
        # tts.lazy_load skipped registration at preload; the adapter
        # self-loads inside voices() — run off the event loop (model load).
        voices = await asyncio.to_thread(register_voices_for_engine, engine_id, assistant.tts)
    return voices


@router.post("/{voice_id}/preview")
def preview_voice(voice_id: str, state: StateDep, payload: VoicePreviewRequest) -> Response:
    """Synthesize a short phrase in `voice_id` and return raw 16 kHz mono
    int16 PCM — the same pipeline audio format used everywhere else in this
    codebase, not a WAV/container format (no UI consumes this yet, per the
    milestone brief; a container format is a UI-driven decision for M5)."""
    assistant = state.require_assistant()
    pcm = preview_text(assistant.tts, voice_id, phrase=payload.phrase)
    return Response(content=pcm.tobytes(), media_type="application/octet-stream")
