"""Developer diagnostics: one snapshot of everything the runtime is doing.

The backend the desktop/web UI's Performance and Diagnostics pages will consume
(via the M5 API). Snapshots are plain pydantic models — JSON-serializable and
cheap enough to sample at UI refresh rates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psutil
from pydantic import BaseModel, ConfigDict

from eva.config.settings import Settings
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
    microphone_available: bool  # mic permission on AND capturing (M5.7)
    microphone_muted: bool  # user muted capture; typed chat still works (M5.7)
    playback_active: bool
    input_level_dbfs: float
    pending_audio_events: int
    capture_ring_depth: int
    capture_frames_dropped: int
    token_queue_depth: int
    sentence_queue_depth: int
    playback_queued_seconds: float
    # Performance
    resources: ResourceUsage
    last_turn: TurnMetrics | None  # per-turn stage timing: asr_ms/ttft_ms/tts_first_ms/ttfa_ms
    turns_completed: int
    metrics_summary: str
    barge_in_count: int
    last_barge_in_latency_ms: int | None
    # Memory (M4, ADR-019/ADR-020)
    memory_enabled: bool
    memory_turn_count: int
    memory_db_size_bytes: int
    memory_embedding_count: int
    last_retrieval_ms: int | None
    last_retrieval_score_top1: float | None
    # Personalization (M4, ADR-022)
    active_persona_id: str
    active_profile_id: str | None
    active_voice: str
    # Events
    recent_events: list[str]  # newest last


def snapshot_idle(settings: Settings) -> RuntimeSnapshot:
    """Snapshot for when no assistant is built/running yet (e.g. server up,
    engine not started). Configuration and system resources are still real;
    pipeline/device fields report their at-rest values."""
    from eva.conversation.personas import resolve_persona

    return RuntimeSnapshot(
        profile=settings.profile,
        language=settings.conversation.language,
        models={
            "llm": settings.llm.model,
            "asr": settings.asr.model,
            "tts": settings.tts.model,
            "vad": settings.vad.engine,
        },
        devices={"llm": "unloaded", "asr": "unloaded", "tts": "unloaded", "vad": "unloaded"},
        state="idle",
        epoch=0,
        microphone_available=False,  # nothing capturing until the engine starts
        microphone_muted=False,
        playback_active=False,
        input_level_dbfs=-120.0,
        pending_audio_events=0,
        capture_ring_depth=0,
        capture_frames_dropped=0,
        token_queue_depth=0,
        sentence_queue_depth=0,
        playback_queued_seconds=0.0,
        resources=sample_resources(),
        last_turn=None,
        turns_completed=0,
        metrics_summary="No completed turns.",
        barge_in_count=0,
        last_barge_in_latency_ms=None,
        memory_enabled=settings.memory.embedding_enabled,
        memory_turn_count=0,
        memory_db_size_bytes=0,
        memory_embedding_count=0,
        last_retrieval_ms=None,
        last_retrieval_score_top1=None,
        active_persona_id=resolve_persona(settings).id,
        active_profile_id=settings.conversation.active_profile_id,
        active_voice=settings.tts.voice,
        recent_events=[],
    )


class DiagnosticsProvider:
    """Aggregates runtime information from a live Assistant."""

    def __init__(self, assistant: Assistant) -> None:
        self._assistant = assistant

    def snapshot(self) -> RuntimeSnapshot:
        from eva.conversation.personas import resolve_persona

        a = self._assistant
        turns = a.orchestrator.metrics.turns
        memory_stats = a.memory.stats()
        active_profile = a.profiles.active()
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
            microphone_available=a.settings.permissions.devices.microphone,
            microphone_muted=a.orchestrator.microphone_muted,
            playback_active=a.audio.is_speaking,
            input_level_dbfs=a.audio.pipeline.level_dbfs,
            pending_audio_events=a.orchestrator.pending_audio_events,
            capture_ring_depth=len(a.audio.capture_ring),
            capture_frames_dropped=a.audio.capture_ring.dropped,
            token_queue_depth=a.orchestrator.token_queue_depth,
            sentence_queue_depth=a.orchestrator.sentence_queue_depth,
            playback_queued_seconds=a.audio.playback.queued_seconds(),
            resources=sample_resources(),
            last_turn=turns[-1] if turns else None,
            # Lifetime count via a counter, not len(window) — the samples
            # list is bounded (see MetricsCollector), so summing it would
            # undercount once a long session exceeds the window.
            turns_completed=a.orchestrator.metrics.non_cancelled_count,
            metrics_summary=a.orchestrator.metrics.summary(),
            barge_in_count=a.orchestrator.barge_in_count,
            last_barge_in_latency_ms=a.orchestrator.last_barge_in_latency_ms,
            memory_enabled=a.embedding is not None,
            memory_turn_count=memory_stats.turn_count,
            memory_db_size_bytes=memory_stats.db_size_bytes,
            memory_embedding_count=memory_stats.embedded_turn_count,
            last_retrieval_ms=a.orchestrator.last_retrieval_ms,
            last_retrieval_score_top1=a.orchestrator.last_retrieval_score_top1,
            active_persona_id=resolve_persona(a.settings).id,
            active_profile_id=active_profile.id if active_profile is not None else None,
            active_voice=a.settings.tts.voice,
            recent_events=[e.name for e in a.bus.recent_events()],
        )
