"""Run provenance: everything needed to reproduce a measurement months later.

Relocated here in Batch 11 from `eva.audio.capture_probe`, where it was first
written for the M7.2 capture probe. Two unrelated callers now need the same
facts — the capture probe and the M8 benchmark report generator — and the
alternative was a second copy of the git/version/GPU probing in
`eva.benchmark`. Living in `eva.core` also puts it in the right layer: nothing
about "which commit and which GPU produced this number" is audio-specific,
and a diagnostic module was never the right owner (the same class of
misplacement as review finding M3).

The field set is unchanged from the capture probe's original, so the JSON that
`eva capture-test` writes keeps exactly the shape it had.
"""

from __future__ import annotations

import contextlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass(frozen=True)
class Environment:
    """Everything needed to reproduce this measurement months later."""

    timestamp_utc: str
    eva_version: str
    git_commit: str | None
    git_dirty: bool | None
    backend: str
    gpu_name: str | None
    gpu_vram_mb: int | None
    cuda_device_count: int
    faster_whisper_version: str | None
    ctranslate2_version: str | None
    python_version: str
    platform: str


def git_state() -> tuple[str | None, bool | None]:
    """(short commit, dirty). Both None outside a git checkout."""
    from eva.core.proc import no_window_kwargs

    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                **no_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return (None, None)
    status = git("status", "--porcelain")
    return (commit, bool(status) if status is not None else None)


def package_version(name: str) -> str | None:
    """Installed version of `name`, or None when it isn't installed."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def capture_environment(backend: str) -> Environment:
    """Snapshot the current machine/checkout. Every probe is failure-tolerant:
    provenance must never be the reason a measurement doesn't get recorded."""
    from eva import __version__

    gpu_name: str | None = None
    gpu_vram: int | None = None
    with contextlib.suppress(Exception):
        from eva.hardware import detect_hardware

        gpus = detect_hardware().gpus
        if gpus:
            gpu_name, gpu_vram = gpus[0].name, gpus[0].vram_total_mb

    cuda_devices = 0
    with contextlib.suppress(Exception):
        import ctranslate2

        cuda_devices = int(ctranslate2.get_cuda_device_count())

    commit, dirty = git_state()
    return Environment(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        eva_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        backend=backend,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        cuda_device_count=cuda_devices,
        faster_whisper_version=package_version("faster-whisper"),
        ctranslate2_version=package_version("ctranslate2"),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({sys.platform})",
    )
