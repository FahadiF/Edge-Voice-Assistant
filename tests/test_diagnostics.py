"""Diagnostics API tests — stubbed assistant, no models or audio hardware."""

from __future__ import annotations

from types import SimpleNamespace

from eva.config.settings import Settings
from eva.core.events import EventBus, TurnStarted
from eva.memory.models import MemoryStats
from eva.metrics.diagnostics import DiagnosticsProvider, sample_resources, snapshot_idle
from eva.metrics.turn import MetricsCollector, TurnMetrics


class TestResources:
    def test_sample_resources_returns_sane_values(self) -> None:
        usage = sample_resources()
        assert 0 <= usage.cpu_percent <= 100
        assert 0 < usage.ram_used_mb <= usage.ram_total_mb
        # GPU fields are optional (None on machines without nvidia-smi).
        if usage.vram_total_mb is not None:
            assert usage.vram_total_mb > 0

    def test_resource_usage_serializes(self) -> None:
        assert "ram_total_mb" in sample_resources().model_dump_json()


def _stub_assistant() -> SimpleNamespace:
    settings = Settings()
    bus = EventBus()
    bus.publish(TurnStarted(epoch=1))
    metrics = MetricsCollector()
    metrics.record(TurnMetrics(epoch=1, asr_ms=100, ttft_ms=300, ttfa_ms=900, total_ms=2000))
    orchestrator = SimpleNamespace(
        state="listening",
        current_epoch=1,
        pending_audio_events=0,
        token_queue_depth=0,
        sentence_queue_depth=0,
        barge_in_count=2,
        last_barge_in_latency_ms=85,
        last_retrieval_ms=12,
        last_retrieval_score_top1=0.83,
        microphone_muted=False,
        metrics=metrics,
    )
    audio = SimpleNamespace(
        is_speaking=False,
        pipeline=SimpleNamespace(level_dbfs=-42.0),
        capture_ring=_SizedRing(),
        playback=SimpleNamespace(queued_seconds=lambda: 0.0),
    )
    memory_stats = MemoryStats(
        conversation_count=1,
        turn_count=4,
        embedded_turn_count=4,
        summary_count=0,
        db_size_bytes=65536,
        fts_enabled=True,
    )
    return SimpleNamespace(
        settings=settings,
        bus=bus,
        orchestrator=orchestrator,
        audio=audio,
        llm=SimpleNamespace(device="cuda"),
        asr=SimpleNamespace(device="cuda"),
        tts=SimpleNamespace(device="cpu"),
        memory=SimpleNamespace(stats=lambda: memory_stats),
        profiles=SimpleNamespace(active=lambda: None),
        embedding=SimpleNamespace(),  # non-None: embeddings enabled
        active_models=lambda: {
            "llm": settings.llm.model,
            "asr": settings.asr.model,
            "tts": settings.tts.model,
            "vad": settings.vad.engine,
        },
    )


class _SizedRing:
    dropped = 2

    def __len__(self) -> int:
        return 3


class TestSnapshot:
    def test_snapshot_aggregates_runtime_state(self) -> None:
        provider = DiagnosticsProvider(_stub_assistant())  # type: ignore[arg-type]
        snap = provider.snapshot()
        assert snap.state == "listening"
        assert snap.devices == {"llm": "cuda", "asr": "cuda", "tts": "cpu", "vad": "cpu"}
        assert snap.models["llm"]
        assert snap.input_level_dbfs == -42.0
        assert snap.capture_ring_depth == 3
        assert snap.capture_frames_dropped == 2
        assert snap.last_turn is not None
        assert snap.last_turn.ttfa_ms == 900
        assert snap.recent_events == ["TurnStarted"]
        assert snap.token_queue_depth == 0
        assert snap.sentence_queue_depth == 0
        assert snap.playback_queued_seconds == 0.0
        assert snap.barge_in_count == 2
        assert snap.last_barge_in_latency_ms == 85
        assert snap.memory_enabled is True
        assert snap.memory_turn_count == 4
        assert snap.memory_db_size_bytes == 65536
        assert snap.memory_embedding_count == 4
        assert snap.last_retrieval_ms == 12
        assert snap.last_retrieval_score_top1 == 0.83
        assert snap.active_persona_id == "default"
        assert snap.active_profile_id is None
        assert snap.active_voice == Settings().tts.voice

    def test_snapshot_serializes_to_json(self) -> None:
        provider = DiagnosticsProvider(_stub_assistant())  # type: ignore[arg-type]
        payload = provider.snapshot().model_dump_json()
        assert "metrics_summary" in payload


class TestSnapshotIdle:
    def test_idle_snapshot_reports_personalization_defaults(self) -> None:
        settings = Settings()
        snap = snapshot_idle(settings)
        assert snap.state == "idle"
        assert snap.active_persona_id == "default"
        assert snap.active_profile_id is None
        assert snap.active_voice == settings.tts.voice
