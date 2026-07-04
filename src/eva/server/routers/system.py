"""Health and hardware endpoints."""

from __future__ import annotations

from fastapi import APIRouter

import eva
from eva.hardware import detect_hardware, recommend_profile
from eva.server.deps import StateDep
from eva.server.schemas import HardwareSummary, HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=eva.__version__)


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
