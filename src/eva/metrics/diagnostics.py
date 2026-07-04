"""Developer diagnostics: one snapshot of everything the runtime is doing.

The backend the desktop/web UI's Performance and Diagnostics pages will consume
(via the M5 API). Snapshots are plain pydantic models — JSON-serializable and
cheap enough to sample at UI refresh rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psutil
from pydantic import BaseModel, ConfigDict

from eva.hardware.detect import run_probe
from eva.metrics.turn import TurnMetrics

if TYPE_CHECKING:
    from eva.engine import Assistant


class ResourceUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    cpu_percent: float
    ram_used_mb: int
    ram_total_mb: int
    gpu_percent: float | None = None
    vram_used_mb: int | None = None
    vram_total_mb: int | None = None


def sample_resources() -> ResourceUsage:
    """Current system utilization. GPU numbers are None without nvidia-smi."""
    vm = psutil.virtual_memory()
    gpu_percent: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None
    output = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if output:
        first = output.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 3:
            try:
                gpu_percent = float(parts[0])
                vram_used = int(float(parts[1]))
                vram_total = int(float(parts[2]))
            except ValueError:
                pass
    return ResourceUsage(
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_used_mb=int((vm.total - vm.available) / 1_048_576),
        ram_total_mb=int(vm.total / 1_048_576),
        gpu_percent=gpu_percent,
        vram_used_mb=vram_used,
        vram_total_mb=vram_total,
    )


class RuntimeSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Configuration
    profile: str
    language: str
    models: dict[str, str]  # kind → model id
    devices: dict[str, str]  # kind → device actually in use
    # Pipeline
    state: str  # idle / listening / thinking / speaking
    epoch: int
    playback_active: bool
    input_level_dbfs: float
    pending_audio_events: int
    capture_ring_depth: int
    capture_frames_dropped: int
    # Performance
    resources: ResourceUsage
    last_turn: TurnMetrics | None
    turns_completed: int
    metrics_summary: str
    # Events
    recent_events: list[str]  # newest last


class DiagnosticsProvider:
    """Aggregates runtime information from a live Assistant."""

    def __init__(self, assistant: Assistant) -> None:
        self._assistant = assistant

    def snapshot(self) -> RuntimeSnapshot:
        a = self._assistant
        turns = a.orchestrator.metrics.turns
        return RuntimeSnapshot(
            profile=a.settings.profile,
            language=a.settings.conversation.language,
            models=a.active_models(),
            devices={
                "llm": a.llm.device,
                "asr": a.asr.device,
                "tts": a.tts.device,
                "vad": "cpu",
            },
            state=a.orchestrator.state,
            epoch=a.orchestrator.current_epoch,
            playback_active=a.audio.is_speaking,
            input_level_dbfs=a.audio.pipeline.level_dbfs,
            pending_audio_events=a.orchestrator.pending_audio_events,
            capture_ring_depth=len(a.audio.capture_ring),
            capture_frames_dropped=a.audio.capture_ring.dropped,
            resources=sample_resources(),
            last_turn=turns[-1] if turns else None,
            turns_completed=sum(1 for t in turns if not t.cancelled),
            metrics_summary=a.orchestrator.metrics.summary(),
            recent_events=[e.name for e in a.bus.recent_events()],
        )
