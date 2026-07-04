"""Onboarding wizard tests — no models, no network, no audio hardware.

The install/download steps are dependency-injected via monkeypatch so the whole
flow (plan, estimates, confirmation, step execution, failure handling, state
persistence) is exercised deterministically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eva.config.paths import AppPaths, get_app_paths
from eva.config.settings import Settings
from eva.core.errors import ModelError
from eva.hardware.detect import CpuInfo, GpuInfo, HardwareReport, MemoryInfo
from eva.hardware.profiles import recommend_profile
from eva.models.catalog import model_catalog, register_builtin_models
from eva.onboarding import (
    ModelRequirement,
    SetupPlan,
    SetupState,
    build_plan,
    check_readiness,
    is_ready,
    readiness_problems,
    run_onboarding,
)


def _fake_report() -> HardwareReport:
    return HardwareReport(
        os_name="TestOS",
        os_version="1",
        python_version="3.12.0",
        cpu=CpuInfo(name="Test CPU", physical_cores=8, logical_cores=16),
        memory=MemoryInfo(total_mb=16000, available_mb=8000),
        gpus=[GpuInfo(name="Test GPU", backend="cuda", vram_total_mb=6144)],
    )


def _plan(*, runtime_installed: bool, model_id: str = "kokoro-82m-v1.0") -> SetupPlan:
    """A deterministic incomplete plan that touches no network or host state."""
    register_builtin_models()
    report = _fake_report()
    return SetupPlan(
        report=report,
        profile=recommend_profile(report),
        variant="cpu",
        runtime_installed=runtime_installed,
        models=(ModelRequirement(model_catalog.get(model_id), installed=False),),
    )


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def paths(app_paths: AppPaths) -> AppPaths:
    return app_paths


class TestReadiness:
    def test_check_readiness_covers_runtimes_and_models(
        self, settings: Settings, paths: AppPaths
    ) -> None:
        items = check_readiness(settings, paths)
        categories = {i.category for i in items}
        assert categories == {"runtime", "model"}
        # Models are absent in the isolated test home.
        assert any(i.category == "model" and not i.ok for i in items)

    def test_readiness_problems_are_actionable(self, settings: Settings, paths: AppPaths) -> None:
        problems = readiness_problems(check_readiness(settings, paths))
        assert problems
        assert all("eva models download" in p or "eva setup" in p for p in problems)

    def test_is_ready_false_without_models(self, settings: Settings, paths: AppPaths) -> None:
        assert not is_ready(settings, paths)


class TestPlan:
    def test_plan_lists_missing_models(self, settings: Settings, paths: AppPaths) -> None:
        plan = build_plan(settings, paths)
        assert not plan.is_complete
        assert plan.missing_models
        assert plan.total_download_mb > 0

    def test_estimate_is_a_sane_range(self, settings: Settings, paths: AppPaths) -> None:
        plan = build_plan(settings, paths)
        fast, slow = plan.estimated_minutes
        assert 0 < fast <= slow


class TestWizard:
    def test_already_complete_starts_immediately(
        self, settings: Settings, paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eva.onboarding.is_ready", lambda s, p: True)
        # Force plan.is_complete via a fully-installed stub.
        import eva.onboarding as ob

        real_build = ob.build_plan

        def complete_plan(s: Settings, p: AppPaths) -> ob.SetupPlan:
            plan = real_build(s, p)
            return ob.SetupPlan(
                report=plan.report,
                profile=plan.profile,
                variant=plan.variant,
                runtime_installed=True,
                models=tuple(ob.ModelRequirement(m.info, installed=True) for m in plan.models),
            )

        monkeypatch.setattr(ob, "build_plan", complete_plan)
        result = run_onboarding(settings, paths, interactive=False)
        assert result.ready

    def test_non_interactive_incomplete_reports_and_blocks(
        self, settings: Settings, paths: AppPaths, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_onboarding(settings, paths, interactive=False, assume_yes=False)
        assert not result.ready
        assert not result.declined  # non-interactive block, not a user decline
        assert "Setup is required" in capsys.readouterr().out

    def test_declined_by_user(
        self, settings: Settings, paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        result = run_onboarding(settings, paths, interactive=True)
        assert not result.ready
        assert result.declined

    def test_full_run_executes_steps_and_persists_state(
        self, settings: Settings, paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        # Deterministic plan + patched step executors: no network, no host state.
        monkeypatch.setattr(
            "eva.onboarding.build_plan", lambda s, p: _plan(runtime_installed=False)
        )
        monkeypatch.setattr(
            "eva.onboarding._install_runtime", lambda variant: calls.append(f"runtime:{variant}")
        )
        monkeypatch.setattr(
            "eva.onboarding._download_model", lambda mgr, mid: calls.append(f"model:{mid}")
        )
        monkeypatch.setattr("eva.onboarding._verify", lambda s, p: None)

        result = run_onboarding(settings, paths, assume_yes=True, interactive=False)
        assert result.ready
        assert any(c.startswith("runtime:") for c in calls)
        assert any(c.startswith("model:") for c in calls)
        state = SetupState.load(paths)
        assert state.completed
        assert state.app_version

    def test_step_failure_is_friendly(
        self,
        settings: Settings,
        paths: AppPaths,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "eva.onboarding.build_plan", lambda s, p: _plan(runtime_installed=False)
        )

        def boom(variant: str) -> None:
            raise ModelError("network unreachable")

        monkeypatch.setattr("eva.onboarding._install_runtime", boom)
        result = run_onboarding(settings, paths, assume_yes=True, interactive=False)
        assert not result.ready
        assert result.error
        out = capsys.readouterr().out
        assert "Setup could not finish" in out
        assert "network unreachable" in out
        assert "Traceback" not in out  # never leak a traceback


class TestSetupState:
    def test_missing_state_is_first_run(self, paths: AppPaths) -> None:
        assert not SetupState.load(paths).completed

    def test_round_trip(self, paths: AppPaths) -> None:
        SetupState(completed=True, app_version="1.2.3", runtime_variant="cuda").save(paths)
        loaded = SetupState.load(paths)
        assert loaded.completed
        assert loaded.app_version == "1.2.3"
        assert loaded.runtime_variant == "cuda"

    def test_corrupt_state_treated_as_first_run(self, paths: AppPaths) -> None:
        (paths.config_dir / "setup_state.json").write_text("{not json", encoding="utf-8")
        assert not SetupState.load(paths).completed


def test_get_app_paths_smoke(isolated_home: Path) -> None:
    # Guard: onboarding must not touch real user directories in tests.
    assert str(get_app_paths().config_dir).startswith(str(isolated_home))
