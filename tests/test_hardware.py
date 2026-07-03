from __future__ import annotations

from eva.hardware.detect import (
    CpuInfo,
    GpuInfo,
    HardwareReport,
    MemoryInfo,
    _parse_nvidia_smi,
    detect_hardware,
)
from eva.hardware.profiles import recommend_profile


def _report(gpus: list[GpuInfo]) -> HardwareReport:
    return HardwareReport(
        os_name="TestOS",
        os_version="1.0",
        python_version="3.12.0",
        cpu=CpuInfo(name="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_mb=16_000, available_mb=8_000),
        gpus=gpus,
    )


def test_parse_nvidia_smi_single_gpu() -> None:
    out = "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 5800, 595.79\n"
    gpus = _parse_nvidia_smi(out)
    assert len(gpus) == 1
    assert gpus[0].vram_total_mb == 6144
    assert gpus[0].vram_free_mb == 5800
    assert gpus[0].backend == "cuda"
    assert gpus[0].driver_version == "595.79"


def test_parse_nvidia_smi_garbage_lines_skipped() -> None:
    assert _parse_nvidia_smi("garbage\nno commas here\n") == []
    assert _parse_nvidia_smi("name, not-a-number, x, y\n") == []


def test_best_gpu_picks_largest() -> None:
    small = GpuInfo(name="A", backend="cuda", vram_total_mb=4096)
    big = GpuInfo(name="B", backend="cuda", vram_total_mb=12_288)
    assert _report([small, big]).best_gpu == big


def test_best_gpu_none_when_no_gpus() -> None:
    assert _report([]).best_gpu is None


def test_recommend_profile_boundaries() -> None:
    def profile_for(vram_mb: int) -> str:
        gpus = [GpuInfo(name="G", backend="cuda", vram_total_mb=vram_mb)] if vram_mb else []
        return recommend_profile(_report(gpus)).id

    assert profile_for(0) == "cpu-only"
    assert profile_for(4_096) == "cpu-only"  # below 6 GB floor → don't overpromise
    assert profile_for(6_144) == "gpu-6gb"  # RTX 3060 Laptop
    assert profile_for(8_192) == "gpu-6gb"
    assert profile_for(12_288) == "gpu-12gb"
    assert profile_for(24_576) == "gpu-12gb"


def test_rocm_gpu_with_unknown_vram_stays_cpu_tier() -> None:
    rocm = GpuInfo(name="AMD GPU", backend="rocm", vram_total_mb=0)
    assert recommend_profile(_report([rocm])).id == "cpu-only"


def test_detect_hardware_runs_on_real_machine() -> None:
    report = detect_hardware()
    assert report.cpu.logical_cores >= 1
    assert report.memory.total_mb > 0
