"""`web/src/api/types.generated.ts` must not drift from the backend it mirrors.

The settings portion is now generated (Batch 3, `npm run generate:types`), but
generation only guarantees sync at the moment someone runs it — a later
backend change with no regeneration, or a stale committed file, drifts exactly
as silently as the old hand-written mirror did (`tts.lazy_load` was missing
for two milestones; `recommendation` would have been missed the same way).
These tests remain the actual CI enforcement.

`ModelCard` still has no backend schema to generate from (`ModelManager.
describe()` returns a plain dict) and lives in
`web/src/api/manual/dict-response-types.ts`, checked separately below.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from eva.config.settings import Settings

TYPES_TS = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "types.generated.ts"

# Settings section → the TypeScript interface mirroring it.
SETTINGS_SECTIONS = {
    "audio": "AudioSettings",
    "vad": "VADSettings",
    "asr": "ASRSettings",
    "llm": "LLMSettings",
    "tts": "TTSSettings",
    "conversation": "ConversationSettings",
    "memory": "MemorySettings",
    "server": "ServerSettings",
    "ui": "UISettings",
    "developer": "DeveloperSettings",
    "desktop": "DesktopSettings",
    "plugins": "PluginsSettings",
}


def _interface_fields(source: str, name: str) -> set[str]:
    """Top-level field names of an interface, wherever it's declared.

    Hand-maintained files declare `export interface <name> { … }` directly.
    `types.generated.ts` instead exports `type <name> = components['schemas']
    ['<name>']`, an alias into the nested `<name>: { … }` block under
    `components.schemas` — so that indented form is tried as a fallback.

    Brace-depth aware so inline object literals (PermissionsSettings) do not
    leak their members into the parent's field set.
    """
    match = re.search(rf"export interface {re.escape(name)}\s*\{{", source)
    if match is None:
        match = re.search(rf"(?:\r?\n)\s*{re.escape(name)}:\s*\{{", source)
    if match is None:
        raise AssertionError(f"types.ts has no `export interface {name}`")

    fields: set[str] = set()
    depth = 0
    for line in source[match.end() :].splitlines():
        stripped = line.strip()
        if depth == 0 and stripped.startswith("}"):
            break
        if depth == 0:
            field = re.match(r"([A-Za-z_][\w]*)\??\s*:", stripped)
            if field:
                fields.add(field.group(1))
        depth += line.count("{") - line.count("}")
    return fields


@pytest.fixture(scope="module")
def types_source() -> str:
    if not TYPES_TS.exists():  # pragma: no cover - source checkouts always have it
        pytest.skip("web/ sources not present in this checkout")
    api_dir = TYPES_TS.parent
    contents = []
    for p in api_dir.rglob("*.ts"):
        if not p.name.endswith(".test.ts"):
            contents.append(p.read_text(encoding="utf-8"))
    return "\n".join(contents)



@pytest.mark.parametrize(("section", "interface"), sorted(SETTINGS_SECTIONS.items()))
def test_settings_section_matches_typescript(
    types_source: str, section: str, interface: str
) -> None:
    """Every settings field the API serves must exist in the mirror.

    The settings UI is schema-driven, so a missing field still *renders* — but
    anything reading it in typed code (defaults, conditionals) silently sees
    `undefined`.
    """
    model = type(getattr(Settings(), section))
    backend = set(model.model_fields)
    mirrored = _interface_fields(types_source, interface)
    missing = backend - mirrored
    extra = mirrored - backend
    assert not missing, f"{interface} is missing field(s) the backend sends: {sorted(missing)}"
    assert not extra, f"{interface} declares field(s) the backend does not send: {sorted(extra)}"


def test_settings_root_sections_match_typescript(types_source: str) -> None:
    """A whole new settings *section* must appear in the root interface."""
    backend = set(Settings.model_fields)
    mirrored = _interface_fields(types_source, "Settings")
    assert backend == mirrored, (
        f"Settings root drifted — missing {sorted(backend - mirrored)}, "
        f"extra {sorted(mirrored - backend)}"
    )


def test_model_card_matches_describe_output(types_source: str, tmp_path: Path) -> None:
    """`ModelCard` mirrors `ModelManager.describe()`, which returns a plain dict
    and therefore has no schema of its own to check against — the real response
    is the only source of truth."""
    from eva.config.paths import AppPaths
    from eva.models.manager import ModelManager

    paths = AppPaths(
        config_dir=tmp_path,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        conversations_dir=tmp_path / "conversations",
        logs_dir=tmp_path / "logs",
    )
    card = ModelManager(paths).describe("qwen3.5-4b-instruct-q4_k_m", Settings())
    mirrored = _interface_fields(types_source, "ModelCard")
    missing = set(card) - mirrored
    extra = mirrored - set(card)
    assert not missing, f"ModelCard is missing field(s) describe() returns: {sorted(missing)}"
    assert not extra, f"ModelCard declares field(s) describe() does not return: {sorted(extra)}"


def test_the_guard_actually_detects_drift() -> None:
    """The guard is only worth having if it fails when it should — pin that,
    so a future refactor of the parser cannot quietly turn it into a no-op."""
    source = "export interface Example {\n  kept: string;\n  nested: { inner: number };\n}\n"
    assert _interface_fields(source, "Example") == {"kept", "nested"}
    with pytest.raises(AssertionError, match="no `export interface Missing`"):
        _interface_fields(source, "Missing")
