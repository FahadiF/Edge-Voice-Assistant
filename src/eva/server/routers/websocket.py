"""Live event stream (Part 2). Every internal event — transcripts, LLM
tokens, TTS/playback progress, state transitions, model downloads, engine
lifecycle, errors — is forwarded here. Clients subscribe once and never poll.

On connect, a `snapshot` message carries the current diagnostics state so a
client has something to render immediately, before the first live event.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from eva.core.events import STREAM_CLOSED
from eva.metrics.diagnostics import DiagnosticsProvider, snapshot_idle
from eva.server.security import origin_allowed

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws")
async def event_stream(websocket: WebSocket) -> None:
    from eva.server.state import ServerState  # avoid import cycle at module load

    # CORS middleware does not cover WebSocket handshakes — enforce the same
    # localhost-only browser policy here (M5.6): without this, any website
    # the user visits could open ws://127.0.0.1 and read live transcripts.
    origin = websocket.headers.get("origin")
    if not origin_allowed(origin):
        logger.warning("WebSocket connection rejected: disallowed origin %r", origin)
        await websocket.close(code=1008)  # policy violation
        return

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
            if event is STREAM_CLOSED:
                # Server shutting down (M5.7): return promptly so this task
                # is gone before uvicorn's graceful-shutdown pass runs — no
                # "Cancel N running task(s)" message, no timeout wait.
                break
            await websocket.send_json({"type": event.name, "data": event.model_dump(mode="json")})
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except (ConnectionError, OSError) as exc:
        # An abrupt client drop (browser tab closed, network gone) surfaces
        # as ConnectionResetError / WinError 10054 the next time we send —
        # that is a normal disconnect, not a server fault, so it stays at
        # debug and never becomes a stack trace (M5.7).
        logger.debug("WebSocket connection closed by peer: %s", exc)
    except asyncio.CancelledError:
        # Belt-and-suspenders: if a connection is still open when uvicorn's
        # graceful timeout expires it gets cancelled — an orderly end of the
        # stream, swallowed so Ctrl+C never prints a traceback (M5.6). With
        # the STREAM_CLOSED wake-up above this path is now rarely reached.
        logger.debug("WebSocket closed by server shutdown")
    finally:
        state.bus.unsubscribe(queue)
        with contextlib.suppress(RuntimeError, ConnectionError, OSError):
            await websocket.close()
