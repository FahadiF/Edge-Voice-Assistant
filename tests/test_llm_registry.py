"""LLM registry factory tests (Batch 8 / C1: transport-neutral port split).

Constructing `LlamaCppLLM` does not import the native `llama_cpp` runtime (that
is deferred to `.load()`), so the factory wiring is exercised headless. These
guard that engine settings — and crucially `developer.debug` → llama.cpp's
`verbose` load report — reach the adapter, so GPU-offload diagnostics can be
turned on without touching the native path.

`create_llm`'s signature is now `(Settings, AppPaths)`, matching
`eva.asr.registry` — each factory resolves its own path/config internally
rather than the caller resolving a model path and handing it in. `ModelManager
.files_for` is monkeypatched rather than installing a real model: this file
tests the *factory wiring*, not model installation (that is
`test_model_manager.py`'s job).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.errors import ModelError
from eva.llm.base import LocalWeights, engine_device, is_local
from eva.llm.llamacpp import LlamaCppLLM
from eva.llm.openai_compat import OpenAICompatibleLLM
from eva.llm.registry import LOCAL_ENGINE_IDS, create_llm, llm_registry, register_builtins
from eva.models.manager import ModelManager


@pytest.fixture(autouse=True)
def _fake_installed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real model is installed in the test environment; the llamacpp
    factory only needs *a* path, never opens it (loading is deferred)."""
    monkeypatch.setattr(
        ModelManager, "files_for", lambda self, model_id: {"model": Path("model.gguf")}
    )


def _build_llamacpp(settings: Settings, app_paths: AppPaths) -> LlamaCppLLM:
    llm = create_llm(settings, app_paths)
    assert isinstance(llm, LlamaCppLLM)
    return llm


class TestLlamaCppFactory:
    def test_factory_threads_engine_settings(self, app_paths: AppPaths) -> None:
        settings = Settings()
        settings.llm.providers.local.context_length = 4096
        settings.llm.providers.local.gpu_layers = 20
        settings.llm.providers.local.threads = 6
        settings.llm.providers.local.batch_size = 256

        llm = _build_llamacpp(settings, app_paths)

        assert llm._context_length == 4096
        assert llm._gpu_layers == 20
        assert llm._threads == 6
        assert llm._batch_size == 256

    def test_verbose_follows_developer_debug(self, app_paths: AppPaths) -> None:
        quiet = _build_llamacpp(Settings(), app_paths)
        assert quiet._verbose is False  # quiet by default (M5.7 clean-output behavior)

        debug_settings = Settings()
        debug_settings.developer.debug = True
        assert _build_llamacpp(debug_settings, app_paths)._verbose is True

    def test_llamacpp_is_local_weights(self, app_paths: AppPaths) -> None:
        """The port split (C1) is real: llama.cpp implements `LocalWeights`
        structurally, and `is_local()`/`engine_device()` agree with it."""
        llm = _build_llamacpp(Settings(), app_paths)
        assert isinstance(llm, LocalWeights)
        assert is_local(llm)
        assert engine_device(llm) == "unloaded"  # before load() ever runs
        assert "llamacpp" in LOCAL_ENGINE_IDS


class TestOpenAICompatibleFactory:
    def test_engine_id_resolves_to_the_adapter(self, app_paths: AppPaths) -> None:
        settings = Settings()
        settings.llm.engine = "openai-compatible"
        settings.llm.providers.openai_compatible.base_url = "http://127.0.0.1:11434/v1"
        settings.llm.providers.openai_compatible.model = "llama3"

        llm = create_llm(settings, app_paths)

        assert isinstance(llm, OpenAICompatibleLLM)
        assert "openai-compatible" not in LOCAL_ENGINE_IDS

    def test_a_remote_endpoint_is_rejected_at_construction(self, app_paths: AppPaths) -> None:
        """Decision 8.3: this milestone ships local providers only. A
        non-loopback base_url must fail loudly, not silently attempt egress."""
        settings = Settings()
        settings.llm.engine = "openai-compatible"
        settings.llm.providers.openai_compatible.base_url = "https://api.example.com/v1"

        with pytest.raises(ModelError, match="local address"):
            create_llm(settings, app_paths)


class _FakeRemoteLLM:
    """A minimal stand-in for a real remote-provider adapter: implements
    only the transport-neutral `LLMEngine` shape (no `load`/`unload`/
    `device`), proving the port split actually separates the two concerns
    rather than every adapter still needing the local lifecycle by
    convention. Does not inherit `LLMEngine` — `is_local()`/`engine_device()`
    are structural (`LocalWeights` is a `Protocol`), so this has to hold for
    ANY object shaped like an engine, not only ones that subclass the ABC.
    """

    def stream(self, messages: object, params: object, should_abort: object, **kw: object) -> None:
        raise NotImplementedError


class TestRemoteProviderIsNotLocalWeights:
    """Proves the split is real: a remote provider is simply not
    `LocalWeights` — no flag, no opt-out, structural absence."""

    def test_not_isinstance_of_local_weights(self) -> None:
        assert not isinstance(_FakeRemoteLLM(), LocalWeights)

    def test_is_local_returns_false(self) -> None:
        assert is_local(_FakeRemoteLLM()) is False  # type: ignore[arg-type]

    def test_engine_device_reports_remote_not_an_attribute_error(self) -> None:
        assert engine_device(_FakeRemoteLLM()) == "remote"  # type: ignore[arg-type]


class TestLocalEngineIdsMatchRegistry:
    """`LOCAL_ENGINE_IDS` (`eva.llm.registry`) is a hand-maintained frozenset
    that `required_models()` (`eva.engine`) trusts to decide whether the LLM
    model belongs in readiness/preflight checks. Nothing at the registration
    site enforces it stays in sync with which registered factories actually
    produce a `LocalWeights`-satisfying engine — this builds every registered
    engine and checks structurally, so a future provider that forgets to
    update the set (in either direction) fails a test instead of silently
    dropping out of preflight. No engine ids are hardcoded here: the id list
    comes from the registry itself.
    """

    def test_every_registered_engine_matches_local_engine_ids(self, app_paths: AppPaths) -> None:
        register_builtins()
        engine_ids = llm_registry.ids()
        assert engine_ids  # guard against an empty registry masking this test

        for engine_id in engine_ids:
            settings = Settings()
            engine = llm_registry.get(engine_id)(settings, app_paths)
            assert is_local(engine) == (engine_id in LOCAL_ENGINE_IDS), (
                f"'{engine_id}' is_local()={is_local(engine)} but "
                f"LOCAL_ENGINE_IDS membership={engine_id in LOCAL_ENGINE_IDS} — "
                "the registered engine's actual LocalWeights-conformance and "
                "the hand-maintained set have drifted apart."
            )

        assert set(engine_ids) >= LOCAL_ENGINE_IDS, (
            "LOCAL_ENGINE_IDS names an id that isn't even registered anymore"
        )
