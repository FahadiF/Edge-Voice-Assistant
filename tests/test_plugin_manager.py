"""Plugin manager unit tests, with a fake entry point standing in for an
installed plugin package (no real third-party plugin exists yet)."""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from eva.core.errors import PluginError
from eva.plugins.manager import ENTRY_POINT_GROUP, PluginManager
from eva.plugins.manifest import PluginManifest


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
        assert manager.discover() == []

    def test_discovers_healthy_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("demo", "demo.pkg:manifest", lambda: _good_manifest)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover()
        assert state.manifest.id == "demo"
        assert state.enabled is True
        assert state.healthy is True

    def test_broken_plugin_is_unhealthy_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> None:
            raise RuntimeError("import exploded")

        ep = _make_ep("broken", "broken.pkg:manifest", boom)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover()
        assert state.healthy is False
        assert "import exploded" in (state.error or "")

    def test_manifest_wrong_type_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("bad", "bad.pkg:manifest", lambda: "not a manifest")
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        [state] = manager.discover()
        assert state.healthy is False

    def test_discover_is_cached_until_forced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def counting_entry_points(group: str | None = None) -> list[EntryPoint]:
            calls.append(1)
            return []

        monkeypatch.setattr("eva.plugins.manager.entry_points", counting_entry_points)
        manager = PluginManager()
        manager.discover()
        manager.discover()
        assert len(calls) == 1
        manager.discover(force=True)
        assert len(calls) == 2


class TestLifecycle:
    def _manager_with_demo(self, monkeypatch: pytest.MonkeyPatch) -> PluginManager:
        ep = _make_ep("demo", "demo.pkg:manifest", lambda: _good_manifest)
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        manager.discover()
        return manager

    def test_enable_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = self._manager_with_demo(monkeypatch)
        manager.disable("demo")
        assert manager.get("demo").enabled is False
        manager.enable("demo")
        assert manager.get("demo").enabled is True

    def test_enable_unhealthy_plugin_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = _make_ep("broken", "x:y", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
        monkeypatch.setattr("eva.plugins.manager.entry_points", lambda group=None: [ep])
        manager = PluginManager()
        manager.discover()
        with pytest.raises(PluginError):
            manager.enable("broken")

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
