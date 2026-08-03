"""Secret storage (Batch 8 / H6): settings hold references, never raw values.

`settings.json` is plain JSON that flows verbatim into `eva diagnose` and any
exported settings document (ADR-015). An API key stored directly on
`LLMSettings` would land there by default, with no dedicated redaction path.
Instead, a provider config field is named `api_key_ref` — an opaque string a
`SecretStore` resolves only at the moment a credential is actually needed
(inside an adapter's request path). The reference travels through settings;
the secret value never does, so there is nothing for a diagnostics dump or
settings export to redact — it was never there to begin with.

Decision 8.4 (Final Execution Roadmap, frozen): `EnvSecretStore` ships in the
base install (reads `EVA_SECRET_<REF>` environment variables). `keyring`-
backed storage is an optional extra (`pip install -e ".[secrets]"`), imported
lazily inside `KeyringSecretStore.get()` so a bare install never imports it.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from eva.core.errors import SecretError

_ENV_PREFIX = "EVA_SECRET_"
_NON_ENV_CHARS = re.compile(r"[^A-Z0-9_]")


def _env_var_name(ref: str) -> str:
    """ "eva/openai" → "EVA_SECRET_EVA_OPENAI" — uppercased, non-alphanumerics
    replaced with underscores, since env var names can't hold "/"."""
    return _ENV_PREFIX + _NON_ENV_CHARS.sub("_", ref.upper())


class SecretStore(ABC):
    """Resolves an opaque reference to its secret value.

    Called only from inside an adapter's request path, immediately before an
    authenticated call — never at settings load, never cached on an object
    that might be serialized (diagnostics, export).
    """

    @abstractmethod
    def get(self, ref: str) -> str:
        """Return the secret value for `ref`. Raises `SecretError` if absent."""


class EnvSecretStore(SecretStore):
    """Base-install secret store: reads `EVA_SECRET_<REF>` environment
    variables. No keychain dependency; identical behavior on every platform,
    and the natural fit for a headless or containerized deployment."""

    def get(self, ref: str) -> str:
        var = _env_var_name(ref)
        value = os.environ.get(var)
        if not value:
            raise SecretError(f"Secret '{ref}' is not set — export {var}")
        return value


class KeyringSecretStore(SecretStore):
    """OS-keychain-backed store (Windows Credential Manager / macOS Keychain
    / Secret Service on Linux). Requires the optional `keyring` package
    (`pip install -e ".[secrets]"`) — imported inside `get()`, not at module
    import time, so a base install never pulls it in."""

    _SERVICE = "eva"

    def get(self, ref: str) -> str:
        try:
            import keyring
        except ImportError as exc:
            raise SecretError(
                "KeyringSecretStore requires the 'secrets' extra: pip install -e \".[secrets]\""
            ) from exc
        value = keyring.get_password(self._SERVICE, ref)
        if not value:
            raise SecretError(f"Secret '{ref}' is not set in the OS keychain")
        return str(value)


def resolve_secret(ref: str, *, store: SecretStore | None = None) -> str:
    """Resolve `ref` via `store` (defaults to `EnvSecretStore`).

    The one call site adapter code should use — never construct a
    `SecretStore` ad hoc in an adapter, so swapping the default backend later
    touches this single function.
    """
    return (store or EnvSecretStore()).get(ref)
