"""FastAPI dependency accessors — thin wrappers over `request.app.state.eva`.

Routers depend on `StateDep` (an `Annotated` alias) rather than writing
`Depends(get_state)` inline at every endpoint — FastAPI's recommended style
since 0.95, and it keeps every handler signature identical regardless of how
many endpoints a router has.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from eva.server.state import ServerState


def get_state(request: Request) -> ServerState:
    state: ServerState = request.app.state.eva
    return state


StateDep = Annotated[ServerState, Depends(get_state)]
