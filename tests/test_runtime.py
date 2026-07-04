from __future__ import annotations

import sys

import pytest

from eva.hardware.detect import CpuInfo, GpuInfo, HardwareReport, MemoryInfo
from eva.runtime import (
    LLAMA_WHEEL_INDEX,
    build_llama_install_command,
    choose_variant,
    missing_runtimes,
    probe_runtimes,
)


def _report(gpus: list[GpuInfo]) -> HardwareReport:
    return HardwareReport(
        os_name="TestOS",
        os_version="1",
        python_version="3.12.0",
        cpu=CpuInfo(name="cpu", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_mb=16000, available_mb=8000),
        gpus=gpus,
    )


class TestProbe:
    def test_probe_lists_all_runtimes(self) -> None:
        modules = {s.module for s in probe_runtimes()}
        assert {"sounddevice", "faster_whisper", "kokoro_onnx", "llama_cpp"} <= modules

    def test_installed_base_dep_reports_ok(self) -> None:
        numpy_status = next(s for s in probe_runtimes() if s.module == "numpy")
        assert numpy_status.installed
        assert numpy_status.remedy == ""

    def test_missing_runtime_carries_remedy(self) -> None:
        # 'llama_cpp' may or may not be installed depending on the env; the
        # contract under test is that any missing runtime carries a remedy.
        for status in missing_runtimes():
            assert status.remedy


class TestVariantSelection:
    def test_cuda_when_nvidia_present(self) -> None:
        gpu = GpuInfo(name="RTX 3060", backend="cuda", vram_total_mb=6144)
        assert choose_variant(_report([gpu])) == "cuda"

    def test_cpu_when_no_gpu(self) -> None:
        assert choose_variant(_report([])) == "cpu"

    def test_cpu_when_rocm_only(self) -> None:
        gpu = GpuInfo(name="AMD", backend="rocm", vram_total_mb=8192)
        assert choose_variant(_report([gpu])) == "cpu"

    def test_override_wins(self) -> None:
        assert choose_variant(_report([]), override="cuda") == "cuda"

    def test_invalid_override_rejected(self) -> None:
        with pytest.raises(ValueError):
            choose_variant(_report([]), override="rocm")


class TestInstallCommand:
    def test_cpu_command(self) -> None:
        cmd = build_llama_install_command("cpu", python="py")
        assert cmd[:3] == ["py", "-m", "pip"]
        assert "llama-cpp-python>=0.3" in cmd
        assert LLAMA_WHEEL_INDEX["cpu"] in cmd
        assert "--prefer-binary" in cmd
        assert "nvidia-cuda-runtime-cu12" not in cmd

    def test_cuda_command_includes_nvidia_wheels(self) -> None:
        cmd = build_llama_install_command("cuda", python="py")
        assert LLAMA_WHEEL_INDEX["cuda"] in cmd
        assert "nvidia-cuda-runtime-cu12" in cmd
        assert "nvidia-cublas-cu12" in cmd

    def test_defaults_to_current_interpreter(self) -> None:
        assert build_llama_install_command("cpu")[0] == sys.executable

    def test_invalid_variant_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_llama_install_command("rocm")
