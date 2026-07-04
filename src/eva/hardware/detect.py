"""Hardware probing.

Uses lightweight OS-level probes (psutil, ``nvidia-smi``, ``rocm-smi``) instead of
importing GPU frameworks — detection must be fast and must work before any ML
dependency is installed or loaded. Probes never raise: a failed probe yields an
empty/absent value, and detection degrades to a CPU-only report.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import psutil
from pydantic import BaseModel, ConfigDict

GpuBackend = Literal["cuda", "rocm", "none"]

_PROBE_TIMEOUT_S = 10


class CpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    physical_cores: int
    logical_cores: int


class MemoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_mb: int
    available_mb: int


class GpuInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    backend: GpuBackend
    vram_total_mb: int
    vram_free_mb: int | None = None
    driver_version: str | None = None


class HardwareReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    os_name: str
    os_version: str
    python_version: str
    cpu: CpuInfo
    memory: MemoryInfo
    gpus: list[GpuInfo]

    @property
    def best_gpu(self) -> GpuInfo | None:
        """The GPU with the most VRAM, or None when running CPU-only."""
        usable = [g for g in self.gpus if g.backend != "none"]
        return max(usable, key=lambda g: g.vram_total_mb, default=None)


def run_probe(cmd: list[str]) -> str | None:
    """Run an external probe command; None on any failure."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output."""
    gpus: list[GpuInfo] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, total, free, driver = parts[0], parts[1], parts[2], parts[3]
        try:
            gpus.append(
                GpuInfo(
                    name=name,
                    backend="cuda",
                    vram_total_mb=int(float(total)),
                    vram_free_mb=int(float(free)),
                    driver_version=driver or None,
                )
            )
        except ValueError:
            continue
    return gpus


def _detect_nvidia() -> list[GpuInfo]:
    output = run_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    return _parse_nvidia_smi(output)


def _detect_rocm() -> list[GpuInfo]:
    """Minimal ROCm presence probe. Reported VRAM parsing varies across rocm-smi
    versions, so we only assert presence; profile logic treats unknown VRAM as CPU-tier.
    """
    output = run_probe(["rocm-smi", "--showproductname"])
    if output is None:
        return []
    names = [
        line.split(":", maxsplit=2)[-1].strip()
        for line in output.splitlines()
        if "Card" in line and ":" in line
    ]
    return [GpuInfo(name=n or "AMD GPU", backend="rocm", vram_total_mb=0) for n in names]


def _cpu_name() -> str:
    """Human-readable CPU model name.

    ``platform.processor()`` on Windows returns the family/stepping identifier;
    the marketing name lives in the registry. Linux exposes it in /proc/cpuinfo.
    """
    # sys.platform (not platform.system()) so type checkers analyze the right
    # branch per platform — winreg only exists in Windows stubs.
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
                if name:
                    return name
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            for line in cpuinfo.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", maxsplit=1)[1].strip()
        except (OSError, IndexError):
            pass
    return platform.processor() or platform.machine()


def detect_hardware() -> HardwareReport:
    """Probe the current machine. Never raises; degrades to CPU-only."""
    vm = psutil.virtual_memory()
    cpu = CpuInfo(
        name=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
    )
    memory = MemoryInfo(
        total_mb=int(vm.total / 1_048_576),
        available_mb=int(vm.available / 1_048_576),
    )
    gpus = _detect_nvidia() + _detect_rocm()
    return HardwareReport(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        cpu=cpu,
        memory=memory,
        gpus=gpus,
    )
