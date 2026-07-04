"""Guided first-run onboarding.

A first-time user runs ``eva run`` and is walked through the remaining setup
without reading documentation. This module owns that experience but contains no
installation logic of its own — it orchestrates the existing building blocks
(hardware detection, ``eva.runtime`` install, ``ModelManager`` downloads,
runtime probing), so there is a single implementation of each concern.

Structure:

- :func:`check_readiness` — one authoritative view of what is installed, shared
  by ``eva doctor``, the ``run``/``bench`` preflight, and the wizard.
- :func:`build_plan` — turns readiness + hardware into a `SetupPlan` with the
  work to do and honest size/time estimates.
- :func:`run_onboarding` — the interactive wizard: explain, confirm, execute
  each step with progress, verify, and report friendly errors (never a
  traceback). Designed so future steps (model updates, calibration, a demo
  conversation, config migration) slot in as additional `_Step`s.
- :class:`SetupState` — a persisted marker of completion, for messaging and
  future migration; the authoritative readiness gate is always the real
  installed artifacts, not this flag.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from eva import __version__
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.errors import EvaError
from eva.hardware import HardwareProfile, HardwareReport, detect_hardware, recommend_profile
from eva.models.catalog import ModelInfo
from eva.models.manager import ModelManager
from eva.runtime import (
    choose_variant,
    install_llama_runtime,
    llm_runtime_available,
    probe_runtimes,
)

# Rough download size of the llama.cpp runtime wheels, for the setup estimate.
_RUNTIME_DOWNLOAD_MB = {"cpu": 40, "cuda": 500}
# Estimate transfer time assuming a conservatively slow to a healthy connection.
_SLOW_MB_PER_S = 3.0
_FAST_MB_PER_S = 20.0


# ──────────────────────────── readiness ────────────────────────────


@dataclass(frozen=True)
class ReadinessItem:
    name: str
    category: str  # "runtime" | "model"
    ok: bool
    detail: str
    remedy: str = ""


def check_readiness(settings: Settings, paths: AppPaths) -> list[ReadinessItem]:
    """Single source of truth for what is installed. No side effects."""
    from eva.engine import required_models

    items = [
        ReadinessItem(s.module, "runtime", s.installed, s.purpose, s.remedy)
        for s in probe_runtimes()
    ]
    manager = ModelManager(paths)
    for model_id in required_models(settings):
        installed = manager.is_installed(model_id)
        items.append(
            ReadinessItem(
                model_id,
                "model",
                installed,
                manager.info(model_id).display_name,
                "" if installed else f"eva models download {model_id}",
            )
        )
    return items


def readiness_problems(items: list[ReadinessItem]) -> list[str]:
    """Human-readable one-liners for everything not yet ready."""
    return [f"missing {i.category} '{i.name}' ({i.detail}) — {i.remedy}" for i in items if not i.ok]


def is_ready(settings: Settings, paths: AppPaths) -> bool:
    return all(i.ok for i in check_readiness(settings, paths))


# ──────────────────────────── plan ────────────────────────────


@dataclass(frozen=True)
class ModelRequirement:
    info: ModelInfo
    installed: bool

    @property
    def download_mb(self) -> int:
        return 0 if self.installed else self.info.download_mb


@dataclass(frozen=True)
class SetupPlan:
    report: HardwareReport
    profile: HardwareProfile
    variant: str
    runtime_installed: bool
    models: tuple[ModelRequirement, ...]

    @property
    def runtime_needed(self) -> bool:
        return not self.runtime_installed

    @property
    def missing_models(self) -> list[ModelRequirement]:
        return [m for m in self.models if not m.installed]

    @property
    def is_complete(self) -> bool:
        return self.runtime_installed and not self.missing_models

    @property
    def total_download_mb(self) -> int:
        runtime = 0 if self.runtime_installed else _RUNTIME_DOWNLOAD_MB.get(self.variant, 100)
        return runtime + sum(m.download_mb for m in self.missing_models)

    @property
    def estimated_minutes(self) -> tuple[int, int]:
        mb = self.total_download_mb
        fast = max(1, round(mb / (_FAST_MB_PER_S * 60)))
        slow = max(fast + 1, round(mb / (_SLOW_MB_PER_S * 60)))
        return fast, slow


def build_plan(settings: Settings, paths: AppPaths) -> SetupPlan:
    from eva.engine import required_models

    report = detect_hardware()
    profile = recommend_profile(report)
    variant = choose_variant(report)
    manager = ModelManager(paths)
    models = tuple(
        ModelRequirement(manager.info(mid), manager.is_installed(mid))
        for mid in required_models(settings)
    )
    return SetupPlan(
        report=report,
        profile=profile,
        variant=variant,
        runtime_installed=llm_runtime_available(),
        models=models,
    )


# ──────────────────────────── wizard ────────────────────────────


@dataclass
class OnboardingResult:
    ready: bool
    declined: bool = False
    error: str | None = None


@dataclass
class _Step:
    label: str
    action: Callable[[], None]


def _fmt_gb(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def _print_welcome(plan: SetupPlan, first_time: bool) -> None:
    line = "─" * 56
    print(f"\n{line}")
    print("  Welcome to Edge Voice Assistant")
    print(line)
    if first_time:
        print("\n  It looks like this is your first time running EVA.")
    print("  Let's get your system ready.\n")

    gpu = plan.report.best_gpu
    print("  Detected hardware")
    if gpu is not None:
        print(f"    - {gpu.name} ({gpu.backend.upper()}, {gpu.vram_total_mb} MB VRAM)")
    else:
        print("    - CPU only (no compatible GPU detected)")
    print(f"    - {plan.report.cpu.name}")

    print("\n  Recommended runtime")
    installed = "already installed" if plan.runtime_installed else "will be installed"
    print(f"    - llama.cpp ({plan.variant.upper()} build) — {installed}")

    print("\n  Required AI models")
    for m in plan.models:
        state = "installed" if m.installed else f"download {_fmt_gb(m.info.download_mb)}"
        print(f"    - {m.info.display_name} ({state})")

    if not plan.is_complete:
        fast, slow = plan.estimated_minutes
        print(f"\n  Estimated download:  ~{_fmt_gb(plan.total_download_mb)}")
        print(f"  Estimated setup time: ~{fast}-{slow} minutes (depends on your connection)")
    print(f"{line}\n")


def _confirm(assume_yes: bool, interactive: bool) -> bool:
    if assume_yes:
        return True
    if not interactive:
        return False
    try:
        answer = input("  Would you like to continue? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("", "y", "yes")


def _build_steps(plan: SetupPlan, settings: Settings, paths: AppPaths) -> list[_Step]:
    steps: list[_Step] = []
    if plan.runtime_needed:
        steps.append(
            _Step(
                f"Installing the llama.cpp runtime ({plan.variant} build)",
                partial(_install_runtime, plan.variant),
            )
        )
    manager = ModelManager(paths)
    for requirement in plan.missing_models:
        model_id = requirement.info.id
        if requirement.info.managed_by == "manager":
            action: Callable[[], None] = partial(_download_model, manager, model_id)
        else:  # engine-managed weights download when the engine first loads
            action = partial(_warm_up_asr, settings, paths)
        steps.append(_Step(f"Downloading {requirement.info.display_name}", action))
    steps.append(_Step("Verifying installation", partial(_verify, settings, paths)))
    return steps


def _install_runtime(variant: str) -> None:
    code = install_llama_runtime(variant)
    if code != 0:
        raise EvaError(
            "the language-model runtime failed to install. "
            "Check your internet connection and try again, or run `eva setup` manually."
        )


def _download_model(manager: ModelManager, model_id: str) -> None:
    last = -1

    def progress(filename: str, done: int, total: int) -> None:
        nonlocal last
        pct = int(done * 100 / total) if total else 0
        if pct != last:
            last = pct
            print(f"\r    {filename}: {done // 1_048_576} MB ({pct}%)", end="", flush=True)

    manager.download(model_id, progress)
    print()


def _warm_up_asr(settings: Settings, paths: AppPaths) -> None:
    """Fetch and verify the engine-managed ASR weights, then release them."""
    from eva.asr.registry import create_asr

    asr = create_asr(settings, paths)
    asr.load()
    asr.unload()


def _verify(settings: Settings, paths: AppPaths) -> None:
    problems = readiness_problems(check_readiness(settings, paths))
    if problems:
        raise EvaError("setup finished but some components are still missing: " + problems[0])


def run_onboarding(
    settings: Settings,
    paths: AppPaths,
    *,
    assume_yes: bool = False,
    force: bool = False,
    interactive: bool | None = None,
) -> OnboardingResult:
    """Ensure the system is ready, guiding the user through any missing setup.

    Returns a result the caller uses to decide whether to start the assistant.
    Never raises for expected setup failures — they are reported and returned.
    """
    if interactive is None:
        interactive = sys.stdin.isatty()

    state = SetupState.load(paths)
    plan = build_plan(settings, paths)

    if plan.is_complete and not force:
        return OnboardingResult(ready=True)

    _print_welcome(plan, first_time=not state.completed)

    if plan.is_complete:
        print("  Everything is already installed and ready.\n")
        return OnboardingResult(ready=True)

    if not interactive and not assume_yes:
        # Scripted / non-terminal context: don't block on a prompt. This is a
        # "cannot proceed" outcome (exit non-zero), not a user decline.
        print("  Setup is required. Run `eva first-run` in a terminal, or:")
        for problem in readiness_problems(check_readiness(settings, paths)):
            print(f"    - {problem}")
        return OnboardingResult(ready=False, declined=False)

    if not _confirm(assume_yes, interactive):
        print("\n  Setup cancelled. Run `eva first-run` when you're ready.\n")
        return OnboardingResult(ready=False, declined=True)

    steps = _build_steps(plan, settings, paths)
    print()
    for index, step in enumerate(steps, start=1):
        print(f"  [{index}/{len(steps)}] {step.label}...")
        try:
            step.action()
        except EvaError as exc:
            return _fail(step.label, str(exc))
        except Exception as exc:  # convert anything to friendly text, never a traceback
            return _fail(step.label, f"an unexpected problem occurred ({exc}).")

    SetupState(completed=True, app_version=__version__, runtime_variant=plan.variant).save(paths)
    print("\n  Setup complete. Starting the assistant...\n")
    return OnboardingResult(ready=True)


def _fail(step_label: str, reason: str) -> OnboardingResult:
    print("\n  Setup could not finish.")
    print(f"    Step:   {step_label}")
    print(f"    Reason: {reason}")
    print("    Fix:    resolve the issue above and run `eva first-run` again.")
    print("            `eva doctor` shows exactly what is still missing.\n")
    return OnboardingResult(ready=False, error=reason)


# ──────────────────────────── persisted state ────────────────────────────


@dataclass
class SetupState:
    completed: bool = False
    app_version: str = ""
    runtime_variant: str = ""

    _FILENAME = "setup_state.json"

    @classmethod
    def load(cls, paths: AppPaths) -> SetupState:
        import json

        path = paths.config_dir / cls._FILENAME
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                completed=bool(data.get("completed", False)),
                app_version=str(data.get("app_version", "")),
                runtime_variant=str(data.get("runtime_variant", "")),
            )
        except (OSError, ValueError):
            return cls()  # a corrupt marker is not fatal; treat as first run

    def save(self, paths: AppPaths) -> None:
        import json

        paths.config_dir.mkdir(parents=True, exist_ok=True)
        path = paths.config_dir / self._FILENAME
        payload = {
            "completed": self.completed,
            "app_version": self.app_version,
            "runtime_variant": self.runtime_variant,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
