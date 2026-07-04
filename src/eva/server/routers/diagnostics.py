"""Developer diagnostics API (Part 5) — the backend for the future Developer page."""

from __future__ import annotations

from fastapi import APIRouter

from eva.metrics.diagnostics import DiagnosticsProvider, RuntimeSnapshot, snapshot_idle
from eva.server.deps import StateDep

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("", response_model=RuntimeSnapshot)
def get_diagnostics(state: StateDep) -> RuntimeSnapshot:
    if state.assistant is not None:
        return DiagnosticsProvider(state.assistant).snapshot()
    return snapshot_idle(state.settings)
