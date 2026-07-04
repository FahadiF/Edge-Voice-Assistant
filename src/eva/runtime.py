"""Runtime dependency probing and guided installation.

Two responsibilities, kept pure and testable:

- **Probe** which optional runtimes are importable and, for each missing one,
  the exact remedy — so the app reports a clear diagnostic instead of letting a
  bare ``ModuleNotFoundError`` escape.
- **Build the install command** for the llama.cpp runtime. llama-cpp-python has
  no PyPI wheels (sdist only), so it is installed from the llama.cpp wheel index
  with the variant matched to the detected hardware (ADR-013). The command is
  constructed as data (a list) so it can be shown, dry-run, and unit-tested; a
  thin executor runs it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass

from eva.hardware.detect import HardwareReport

# llama.cpp prebuilt-wheel index (https://github.com/abetlen/llama-cpp-python).
# The index auto-resolves the correct wheel for the platform and Python version,
# which is why this is preferred over pinning per-platform wheel URLs.
LLAMA_WHEEL_INDEX = {
    "cpu": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
    "cuda": "https://abetlen.github.io/llama-cpp-python/whl/cu124",
}

_REINSTALL = 'reinstall the package: pip install -e "."'
_RUN_SETUP = "run: eva setup"

# (import name, human purpose, remedy when missing)
_RUNTIME_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numerics", _REINSTALL),
    ("sounddevice", "audio input/output", _REINSTALL),
    ("livekit", "echo cancellation / noise suppression", _REINSTALL),
    ("pysilero_vad", "voice activity detection", _REINSTALL),
    ("faster_whisper", "speech recognition (ASR)", _REINSTALL),
    ("kokoro_onnx", "speech synthesis (TTS)", _REINSTALL),
    ("llama_cpp", "language model (LLM)", _RUN_SETUP),
)


@dataclass(frozen=True)
class RuntimeStatus:
    module: str
    purpose: str
    installed: bool
    remedy: str = ""


def probe_runtimes() -> list[RuntimeStatus]:
    """Report import availability of every runtime dependency.

    Uses ``find_spec`` (presence check) rather than importing — fast, side-effect
    free, and it does not trigger native GPU initialization logging.
    """
    statuses: list[RuntimeStatus] = []
    for module, purpose, remedy in _RUNTIME_CHECKS:
        installed = _is_importable(module)
        statuses.append(RuntimeStatus(module, purpose, installed, "" if installed else remedy))
    return statuses


def _is_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing_runtimes() -> list[RuntimeStatus]:
    return [s for s in probe_runtimes() if not s.installed]


def llm_runtime_available() -> bool:
    return _is_importable("llama_cpp")


def choose_variant(report: HardwareReport, override: str | None = None) -> str:
    """Pick 'cuda' when an NVIDIA GPU is present, otherwise 'cpu'."""
    if override is not None:
        if override not in LLAMA_WHEEL_INDEX:
            raise ValueError(f"unknown variant '{override}'")
        return override
    gpu = report.best_gpu
    return "cuda" if gpu is not None and gpu.backend == "cuda" else "cpu"


def build_llama_install_command(variant: str, *, python: str | None = None) -> list[str]:
    """Construct the pip command that installs the llama.cpp runtime.

    CUDA also pulls the NVIDIA cudart/cuBLAS redistributable wheels (from PyPI),
    which the CUDA llama.cpp wheel links against at load time.
    """
    if variant not in LLAMA_WHEEL_INDEX:
        raise ValueError(f"unknown variant '{variant}'")
    cmd = [
        python or sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--prefer-binary",  # take the index wheel, never build the PyPI sdist
        "llama-cpp-python>=0.3",
        "--extra-index-url",
        LLAMA_WHEEL_INDEX[variant],
    ]
    if variant == "cuda":
        cmd += ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"]
    return cmd


def install_llama_runtime(variant: str, *, dry_run: bool = False) -> int:
    """Run the llama.cpp install command. Returns the pip exit code (0 = ok)."""
    cmd = build_llama_install_command(variant)
    print(f"Installing the llama.cpp LLM runtime ({variant} build):")
    print("  " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode
