"""Plugin manager unit tests, with a fake entry point standing in for an
installed plugin package (no real third-party plugin exists yet)."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import EntryPoint

import pytest

from eva.config.settings import Settings
from eva.conversation.personas import PersonaProfile, persona_registry
from eva.core.errors import PluginError, RegistryError
from eva.plugins.context import PluginContext
from eva.plugins.manager import ENTRY_POINT_GROUP, PluginManager
from eva.plugins.manifest import PluginManifest


@pytest.fixture(autouse=True)
def _clean_persona_registry() -> Iterator[None]:
    before = set(persona_registry.ids())
    yield
    for pid in set(persona_registry.ids()) - before:
        persona_registry.unregister(pid)


def _good_manifest() -> PluginManifest:
    return PluginManifest(id="demo", name="Demo Plugin", version="1.0.0", contributes=("tool",))


def _fake_entry_points(*eps: EntryPoint) -> object:
    class _Result:
        def __iter__(self) -> object:
            return iter(eps)

    def selectable(group: str) -> object:
        return list(eps) if group == ENTRY_POINT_GROUP else []

    return selectable


def _make_ep(name: str, value: str, loader: object) -> EntryPoint:
    ep = EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)
    # EntryPoint.load() normally imports `value`; monkeypatch load directly.
    object.__setattr__(ep, "load", loader)
    return ep


class TestDiscovery:
    def test_no_plugins_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [])
        manager = PluginManager()
        assert manager.discover(Settings()) == []

    def test_discovers_healthy_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("demo", "demo.pkg:manifest", lambda: _good_manifest)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover(Settings())
        assert state.manifest.id == "demo"
        # A newly discovered plugin defaults to disabled (ADR-011) — a
        # plugin package must never gain a live capability merely by being
        # installed. Deliberate behavior change from auto-enable (H4).
        assert state.enabled is False
        assert state.healthy is True

    def test_broken_plugin_is_unhealthy_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> None:
            raise RuntimeError("import exploded")

        ep = _make_ep("broken", "broken.pkg:manifest", boom)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover(Settings())
        assert state.healthy is False
        assert "import exploded" in (state.error or "")

    def test_manifest_wrong_type_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("bad", "bad.pkg:manifest", lambda: "not a manifest")
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover(Settings())
        assert state.healthy is False

    def test_discover_is_cached_until_forced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def counting_entry_points(group: str | None = None) -> list[EntryPoint]:
            calls.append(1)
            return []

        monkeypatch.setattr("eva.plugins.manager.entry_points", counting_entry_points)
        manager = PluginManager()
        settings = Settings()
        manager.discover(settings)
        manager.discover(settings)
        assert len(calls) == 1
        manager.discover(settings, force=True)
        assert len(calls) == 2


class TestLifecycle:
    def _manager_with_demo(self, monkeypatch: pytest.MonkeyPatch) -> PluginManager:
        ep = _make_ep("demo", "demo.pkg:manifest", lambda: _good_manifest)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        manager.discover(Settings())
        return manager

    def test_enable_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = self._manager_with_demo(monkeypatch)
        settings = Settings()
        manager.disable("demo", settings)
        assert manager.get("demo").enabled is False
        manager.enable("demo", settings)
        assert manager.get("demo").enabled is True
        assert "demo" in settings.plugins.enabled

    def test_disable_removes_the_persisted_enabled_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = self._manager_with_demo(monkeypatch)
        settings = Settings()
        manager.enable("demo", settings)
        assert "demo" in settings.plugins.enabled
        manager.disable("demo", settings)
        assert "demo" not in settings.plugins.enabled

    def test_enable_unhealthy_plugin_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("broken", "x:y", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        settings = Settings()
        manager.discover(settings)
        with pytest.raises(PluginError):
            manager.enable("broken", settings)

    def test_unknown_plugin_raises_with_known_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = self._manager_with_demo(monkeypatch)
        with pytest.raises(PluginError, match="demo"):
            manager.get("nope")

    def test_reload_reruns_discovery_for_one_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = self._manager_with_demo(monkeypatch)
        reloaded = manager.reload("demo")
        assert reloaded.manifest.id == "demo"

    def test_reload_uninstalled_plugin_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = self._manager_with_demo(monkeypatch)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [])
        with pytest.raises(PluginError, match="no longer installed"):
            manager.reload("demo")


class _PersonaPlugin:
    """A fake capability-bearing plugin: exposes `.manifest` plus
    `setup(ctx)`/`teardown(ctx)`, proving the dual entry-point contract
    (bare `PluginManifest` vs. an object exposing one) end-to-end."""

    def __init__(self, plugin_id: str, local_id: str, prompt: str) -> None:
        self.manifest = PluginManifest(
            id=plugin_id, name=plugin_id, version="1.0.0", contributes=("persona",)
        )
        self._local_id = local_id
        self._prompt = prompt
        self.teardown_calls = 0

    def setup(self, ctx: PluginContext) -> None:
        ctx.personas.register(
            self._local_id,
            PersonaProfile(id="unset", display_name=self.manifest.name, system_prompt=self._prompt),
        )

    def teardown(self, ctx: PluginContext) -> None:
        self.teardown_calls += 1


class _FailingSetupPlugin:
    """A plugin whose `setup()` always raises, for testing that a failed
    activation leaves the plugin disabled and cleans up after itself."""

    def __init__(self, plugin_id: str) -> None:
        self.manifest = PluginManifest(id=plugin_id, name=plugin_id, version="1.0.0")

    def setup(self, ctx: PluginContext) -> None:
        # Register one persona successfully, then fail — proves a partial
        # registration is rolled back, not left orphaned.
        ctx.personas.register(
            "partial", PersonaProfile(id="unset", display_name="x", system_prompt="x")
        )
        raise RuntimeError("setup exploded")


def _eps_for(*plugins: _PersonaPlugin | _FailingSetupPlugin) -> list[EntryPoint]:
    # `ep.load()` must return the zero-arg factory (mirroring a real entry
    # point), which `_load_one` then calls to obtain the loaded object.
    return [
        _make_ep(p.manifest.id, f"{p.manifest.id}.pkg:factory", lambda p=p: lambda: p)
        for p in plugins
    ]


class TestCapabilityWiring:
    def test_enabling_registers_a_persona_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        plugin = _PersonaPlugin("demo", "cheerful", "Be upbeat.")
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: _eps_for(plugin))
        manager = PluginManager()
        settings = Settings()
        manager.discover(settings)

        manager.enable("demo", settings)
        persona = persona_registry.get("demo:cheerful")
        assert persona.system_prompt == "Be upbeat."

        manager.disable("demo", settings)
        assert plugin.teardown_calls == 1
        with pytest.raises(RegistryError):
            persona_registry.get("demo:cheerful")

    def test_a_failed_setup_leaves_the_plugin_disabled_and_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = _FailingSetupPlugin("broken-setup")
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: _eps_for(plugin))
        manager = PluginManager()
        settings = Settings()
        manager.discover(settings)

        with pytest.raises(RuntimeError, match="setup exploded"):
            manager.enable("broken-setup", settings)

        assert manager.get("broken-setup").enabled is False
        assert "broken-setup" not in settings.plugins.enabled
        # The persona registered before the failure must not be orphaned.
        with pytest.raises(RegistryError):
            persona_registry.get("broken-setup:partial")

    def test_activation_survives_a_fresh_manager_from_the_same_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates a process restart: a fresh `PluginManager` reads the
        same (as-if reloaded-from-disk) `Settings` and reactivates an
        already-enabled plugin from `discover()` alone."""
        plugin = _PersonaPlugin("demo", "cheerful", "Be upbeat.")
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: _eps_for(plugin))
        settings = Settings()
        settings.plugins.enabled = ["demo"]  # as if persisted from a previous session

        manager = PluginManager()  # a "fresh process" — no explicit enable() call
        manager.discover(settings)

        assert manager.get("demo").enabled is True
        assert persona_registry.get("demo:cheerful").system_prompt == "Be upbeat."

        manager.disable("demo", settings)  # test cleanup

    def test_two_plugins_may_use_the_same_local_persona_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Namespacing must let two plugins coexist under the same local id,
        and per-plugin cleanup must remove only the disabled plugin's
        registration — not its still-enabled sibling's."""
        alpha = _PersonaPlugin("alpha", "assistant", "Alpha voice.")
        beta = _PersonaPlugin("beta", "assistant", "Beta voice.")
        monkeypatch.setattr(
            "eva.plugins.manager.entry_points", lambda group=None: _eps_for(alpha, beta)
        )
        manager = PluginManager()
        settings = Settings()
        manager.discover(settings)

        manager.enable("alpha", settings)
        manager.enable("beta", settings)

        alpha_persona = persona_registry.get("alpha:assistant")
        beta_persona = persona_registry.get("beta:assistant")
        assert alpha_persona.system_prompt == "Alpha voice."
        assert beta_persona.system_prompt == "Beta voice."
        assert alpha_persona is not beta_persona

        manager.disable("alpha", settings)
        with pytest.raises(RegistryError):
            persona_registry.get("alpha:assistant")
        # beta must be entirely untouched by alpha's cleanup
        assert persona_registry.get("beta:assistant").system_prompt == "Beta voice."

        manager.enable("alpha", settings)
        assert persona_registry.get("alpha:assistant").system_prompt == "Alpha voice."

        manager.disable("alpha", settings)
        manager.disable("beta", settings)
