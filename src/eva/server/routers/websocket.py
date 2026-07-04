"""Live event stream (Part 2). Every internal event — transcripts, LLM
tokens, TTS/playback progress, state transitions, model downloads, engine
lifecycle, errors — is forwarded here. Clients subscribe once and never poll.

On connect, a `snapshot` message carries the current diagnostics state so a
client has something to render immediately, before the first live event.
"""

from __future__ import annotations

import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from eva.metrics.diagnostics import DiagnosticsProvider, snapshot_idle

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def event_stream(websocket: WebSocket) -> None:
    from eva.server.state import ServerState  # avoid import cycle at module load

    state: ServerState = websocket.app.state.eva
    await websocket.accept()

    snapshot = (
        DiagnosticsProvider(state.assistant).snapshot()
        if state.assistant is not None
        else snapshot_idle(state.settings)
    )
    await websocket.send_json({"type": "snapshot", "data": snapshot.model_dump(mode="json")})

    queue = state.bus.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json({"type": event.name, "data": event.model_dump(mode="json")})
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    finally:
        state.bus.unsubscribe(queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()
