"""Unit tests for `PluginContext`/`PersonaHandle` in isolation (Batch 2).

`persona_registry` is process-global, so every test here cleans up anything
it registers via an autouse fixture — a leaked registration would otherwise
pollute unrelated tests run later in the same process.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from eva.conversation.personas import PersonaProfile, persona_registry
from eva.core.errors import RegistryError
from eva.plugins.context import PersonaHandle, PluginContext


def _profile(prompt: str = "x") -> PersonaProfile:
    return PersonaProfile(id="unset", display_name="Test", system_prompt=prompt)


@pytest.fixture(autouse=True)
def _clean_persona_registry() -> Iterator[None]:
    before = set(persona_registry.ids())
    yield
    for pid in set(persona_registry.ids()) - before:
        persona_registry.unregister(pid)


class TestPersonaHandle:
    def test_register_namespaces_the_id(self) -> None:
        handle = PersonaHandle("demo")
        namespaced = handle.register("cheerful", _profile())
        assert namespaced == "demo:cheerful"
        assert persona_registry.get("demo:cheerful") is not None

    def test_registered_persona_id_field_matches_the_namespaced_key(self) -> None:
        """Mirrors how built-in and custom personas both work: the registry
        key always equals `PersonaProfile.id`."""
        handle = PersonaHandle("demo")
        namespaced = handle.register("cheerful", _profile())
        assert persona_registry.get(namespaced).id == namespaced

    def test_two_handles_may_use_the_same_local_id(self) -> None:
        alpha = PersonaHandle("alpha")
        beta = PersonaHandle("beta")
        alpha.register("assistant", _profile("Alpha voice."))
        beta.register("assistant", _profile("Beta voice."))
        assert persona_registry.get("alpha:assistant").system_prompt == "Alpha voice."
        assert persona_registry.get("beta:assistant").system_prompt == "Beta voice."

    def test_registering_the_same_local_id_twice_raises(self) -> None:
        """No silent replace: a handle never passes `replace=True`."""
        handle = PersonaHandle("demo")
        handle.register("cheerful", _profile())
        with pytest.raises(RegistryError):
            handle.register("cheerful", _profile())

    def test_release_unregisters_everything_this_handle_registered(self) -> None:
        handle = PersonaHandle("demo")
        handle.register("cheerful", _profile())
        handle.register("serious", _profile())
        handle.release()
        with pytest.raises(RegistryError):
            persona_registry.get("demo:cheerful")
        with pytest.raises(RegistryError):
            persona_registry.get("demo:serious")

    def test_release_is_idempotent(self) -> None:
        handle = PersonaHandle("demo")
        handle.register("cheerful", _profile())
        handle.release()
        handle.release()  # must not raise

    def test_release_does_not_touch_another_plugins_registrations(self) -> None:
        alpha = PersonaHandle("alpha")
        beta = PersonaHandle("beta")
        alpha.register("assistant", _profile("Alpha voice."))
        beta.register("assistant", _profile("Beta voice."))
        alpha.release()
        with pytest.raises(RegistryError):
            persona_registry.get("alpha:assistant")
        assert persona_registry.get("beta:assistant").system_prompt == "Beta voice."
        beta.release()


class TestPluginContext:
    def test_context_exposes_a_persona_handle_and_a_scoped_logger(self) -> None:
        ctx = PluginContext("demo")
        assert isinstance(ctx.personas, PersonaHandle)
        assert ctx.logger.name == "eva.plugins.demo"

    def test_release_delegates_to_the_persona_handle(self) -> None:
        ctx = PluginContext("demo")
        ctx.personas.register("cheerful", _profile())
        ctx.release()
        with pytest.raises(RegistryError):
            persona_registry.get("demo:cheerful")
