"""Health, hardware, and process-shutdown endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import eva
from eva.hardware import detect_hardware, recommend_profile
from eva.server.deps import StateDep
from eva.server.schemas import HardwareSummary, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=eva.__version__)


@router.post("/system/shutdown")
async def shutdown_server(state: StateDep) -> JSONResponse:
    """Gracefully stop the whole server process (M5.6).

    This is how `eva stop` ends a background server cleanly: the engine
    stops first (audio released, memory flushed), then uvicorn exits via
    the callback `eva serve` registered. On Windows there is no portable
    graceful signal for a detached process — TerminateProcess is a hard
    kill that skips all cleanup — so a localhost API call IS the graceful
    path (the API is localhost-only, ADR-017; browser pages are kept out
    by the CORS/Origin policy, eva.server.security).
    """
    if state.shutdown_callback is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "this server was not started with a shutdown hook (eva serve)"},
        )
    await state.stop_engine()
    state.shutdown_callback()
    return JSONResponse(content={"status": "shutting down"})


@router.get("/system/hardware", response_model=HardwareSummary)
def hardware_summary(_state: StateDep) -> HardwareSummary:
    report = detect_hardware()
    tier = recommend_profile(report)
    gpu = report.best_gpu
    return HardwareSummary(
        tier=tier.id,
        tier_name=tier.display_name,
        cpu=report.cpu.name,
        gpu=gpu.name if gpu else None,
        vram_mb=gpu.vram_total_mb if gpu else 0,
        ram_mb=report.memory.total_mb,
    )
