"""Secret storage tests (Batch 8 / H6).

`EnvSecretStore` is the base-install store; `KeyringSecretStore` needs the
optional `keyring` extra and is tested with it monkeypatched into
`sys.modules`, since the test environment does not install it (matching the
base-install posture decision 8.4 chose).

The redaction tests are the concrete, testable form of M7.4's stated exit
criterion ("no credential material appears in any export or diagnostic") —
proven structurally: settings only ever hold `api_key_ref` (a reference
string), never a resolved secret value, so there is nothing for a dump to
leak in the first place.
"""

from __future__ import annotations

import sys
import types

import pytest

from eva.config.settings import Settings
from eva.core.errors import SecretError
from eva.core.secrets import EnvSecretStore, KeyringSecretStore, SecretStore, resolve_secret
from eva.metrics.diagnostics import snapshot_idle


class TestEnvSecretStore:
    def test_resolves_a_set_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVA_SECRET_EVA_OPENAI", "sk-test-12345")
        assert EnvSecretStore().get("eva/openai") == "sk-test-12345"

    def test_missing_variable_raises_secret_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EVA_SECRET_EVA_OPENAI", raising=False)
        with pytest.raises(SecretError, match="EVA_SECRET_EVA_OPENAI"):
            EnvSecretStore().get("eva/openai")

    def test_empty_variable_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVA_SECRET_EVA_OPENAI", "")
        with pytest.raises(SecretError):
            EnvSecretStore().get("eva/openai")

    def test_ref_is_sanitized_into_a_valid_env_var_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ref may contain characters that aren't valid in an env var name
        (`/`, `-`, ...); the mapping must still be deterministic."""
        monkeypatch.setenv("EVA_SECRET_MY_PROVIDER_KEY", "value")
        assert EnvSecretStore().get("my-provider/key") == "value"


class TestKeyringSecretStore:
    def test_requires_the_secrets_extra_when_keyring_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "keyring", None)  # force ImportError
        with pytest.raises(SecretError, match=r'pip install -e ".\[secrets\]"'):
            KeyringSecretStore().get("eva/openai")

    def test_resolves_via_the_keyring_module_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_keyring = types.SimpleNamespace(get_password=lambda service, ref: "sk-from-keyring")
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
        assert KeyringSecretStore().get("eva/openai") == "sk-from-keyring"

    def test_missing_keychain_entry_raises_secret_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_keyring = types.SimpleNamespace(get_password=lambda service, ref: None)
        monkeypatch.setitem(sys.modules, "keyring", fake_keyring)
        with pytest.raises(SecretError, match="OS keychain"):
            KeyringSecretStore().get("eva/openai")


class TestResolveSecret:
    def test_defaults_to_env_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVA_SECRET_EVA_OPENAI", "sk-default-path")
        assert resolve_secret("eva/openai") == "sk-default-path"

    def test_accepts_an_explicit_store(self) -> None:
        class _FixedStore(SecretStore):
            def get(self, ref: str) -> str:
                return f"fixed-{ref}"

        assert resolve_secret("x", store=_FixedStore()) == "fixed-x"


class TestNoCredentialLeak:
    """The concrete, testable form of M7.4's exit criterion."""

    _FAKE_SECRET = "sk-should-never-be-serialized-anywhere-zzq7"

    def test_settings_export_never_contains_the_resolved_secret_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVA_SECRET_EVA_OPENAI", self._FAKE_SECRET)
        settings = Settings()
        settings.llm.providers.openai_compatible.api_key_ref = "eva/openai"

        dumped = settings.model_dump_json()

        assert "eva/openai" in dumped  # the reference is fine to persist
        assert self._FAKE_SECRET not in dumped  # the resolved value must never appear

    def test_diagnostics_snapshot_never_contains_the_resolved_secret_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVA_SECRET_EVA_OPENAI", self._FAKE_SECRET)
        settings = Settings()
        settings.llm.providers.openai_compatible.api_key_ref = "eva/openai"

        snapshot = snapshot_idle(settings)

        assert self._FAKE_SECRET not in snapshot.model_dump_json()
