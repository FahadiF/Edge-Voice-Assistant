"""Package integrity guard.

Imports every module under ``eva``. This catches two failure classes that
slip past targeted unit tests:

- a package missing from the checkout (e.g. a .gitignore pattern accidentally
  matching a source directory — the cause of a real incident where
  ``src/eva/models/`` was never committed), and
- import-time side effects or import errors introduced into rarely-imported
  modules (e.g. a heavy dependency imported at module level).
"""

from __future__ import annotations

import importlib
import pkgutil

import eva

# Modules whose import requires optional native pieces are still imported —
# base dependencies must make every module importable (lazy engine imports are
# an explicit design rule; see docs/DEVELOPMENT.md).


def _walk_modules() -> list[str]:
    names: list[str] = []
    for module in pkgutil.walk_packages(eva.__path__, prefix="eva."):
        names.append(module.name)
    return names


def test_every_module_imports() -> None:
    names = _walk_modules()
    assert len(names) > 30, f"suspiciously few modules found: {names}"
    failures: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # collect all, report together
            failures.append(f"{name}: {exc}")
    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def test_critical_subpackages_present() -> None:
    """The subsystem packages the engine is assembled from must all exist."""
    for package in (
        "eva.audio",
        "eva.vad",
        "eva.asr",
        "eva.llm",
        "eva.tts",
        "eva.models",
        "eva.conversation",
        "eva.core",
        "eva.config",
        "eva.hardware",
        "eva.metrics",
        "eva.benchmark",
    ):
        importlib.import_module(package)
