"""Application error hierarchy.

Every EVA-originated exception derives from :class:`EvaError` so callers can
distinguish expected application failures from programming errors. Adapters must
wrap third-party exceptions into one of these before they cross a port boundary.
"""

from __future__ import annotations


class EvaError(Exception):
    """Base class for all Edge Voice Assistant errors."""


class ConfigError(EvaError):
    """Invalid, unreadable, or unwritable configuration."""


class HardwareError(EvaError):
    """Hardware probing or capability failure."""


class AudioError(EvaError):
    """Audio device or stream failure."""


class ModelError(EvaError):
    """Model download, load, or inference failure."""


class ModelNotInstalledError(ModelError):
    """A referenced model is not present in the local registry."""


class PluginError(EvaError):
    """Plugin discovery, load, or execution failure."""


class RegistryError(EvaError):
    """Unknown id, duplicate registration, or invalid registry operation."""


class MemoryStoreError(EvaError):
    """Memory persistence, search, or management failure.

    Named ``MemoryStoreError`` (not ``MemoryError``) because ``MemoryError``
    is a Python builtin (``MemoryError``, raised on allocation failure) —
    reusing it would shadow a standard-library exception.
    """


class MemoryNotFoundError(MemoryStoreError):
    """A referenced conversation, turn, or user profile does not exist."""
