"""Engine lifecycle API.

Not explicitly named in a "Part", but required infrastructure: opening audio
devices and loading models is an explicit, visible action — never an implicit
side effect of the HTTP server starting — so conversation/diagnostics
endpoints have something to operate on. `eva run`'s onboarding preflight
(`eva.onboarding.check_readiness`) is reused here, not reimplemented.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from eva.server.deps import StateDep
from eva.server.schemas import EngineStatusResponse, ReadinessResponse
from eva.server.state import ServerState

router = APIRouter(prefix="/engine", tags=["engine"])


def _readiness_problems(state: ServerState) -> list[str]:
    # Imported lazily (not at module load) so tests can monkeypatch
    # `eva.onboarding.check_readiness` and have it take effect here.
    from eva.onboarding import check_readiness, readiness_problems

    state.reload_settings()
    return readiness_problems(check_readiness(state.settings, state.paths))


@router.get("/status", response_model=EngineStatusResponse)
def engine_status(state: StateDep) -> EngineStatusResponse:
    running = state.engine_running
    pipeline_state = state.assistant.orchestrator.state if running and state.assistant else "idle"
    return EngineStatusResponse(running=running, state=pipeline_state)


@router.get("/readiness", response_model=ReadinessResponse)
def engine_readiness(state: StateDep) -> ReadinessResponse:
    problems = _readiness_problems(state)
    return ReadinessResponse(ready=not problems, problems=problems)


@router.post("/start", response_model=EngineStatusResponse)
async def start_engine(state: StateDep) -> EngineStatusResponse:
    problems = _readiness_problems(state)
    if problems:
        raise HTTPException(
            status_code=409, detail={"message": "setup incomplete", "problems": problems}
        )
    assistant = await state.start_engine()
    return EngineStatusResponse(running=True, state=assistant.orchestrator.state)


@router.post("/stop", response_model=EngineStatusResponse)
async def stop_engine(state: StateDep) -> EngineStatusResponse:
    await state.stop_engine()
    return EngineStatusResponse(running=False, state="idle")
