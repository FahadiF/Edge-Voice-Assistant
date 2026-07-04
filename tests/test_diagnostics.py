"""Diagnostics API tests — stubbed assistant, no models or audio hardware."""

from __future__ import annotations

from types import SimpleNamespace

from eva.config.settings import Settings
from eva.core.events import EventBus, TurnStarted
from eva.metrics.diagnostics import DiagnosticsProvider, sample_resources
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
        metrics=metrics,
    )
    ring = SimpleNamespace(dropped=0)
    audio = SimpleNamespace(
        is_speaking=False,
        pipeline=SimpleNamespace(level_dbfs=-42.0),
        capture_ring=_SizedRing(),
    )
    return SimpleNamespace(
        settings=settings,
        bus=bus,
        orchestrator=orchestrator,
        audio=audio,
        llm=SimpleNamespace(device="cuda"),
        asr=SimpleNamespace(device="cuda"),
        tts=SimpleNamespace(device="cpu"),
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

    def test_snapshot_serializes_to_json(self) -> None:
        provider = DiagnosticsProvider(_stub_assistant())  # type: ignore[arg-type]
        payload = provider.snapshot().model_dump_json()
        assert "metrics_summary" in payload
