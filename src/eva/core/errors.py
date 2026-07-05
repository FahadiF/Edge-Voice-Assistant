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


class InvalidChatSequenceError(ModelError):
    """The composed chat message list violates the chat-template contract
    every template-based chat format requires: exactly one system message,
    first, followed by strictly alternating user/assistant turns. Caught
    here, before the messages reach the engine, where the violation would
    otherwise surface as an opaque template-engine error (e.g. llama.cpp's
    Qwen template: "System message must be at the beginning.")."""


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
