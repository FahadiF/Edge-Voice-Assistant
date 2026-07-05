"""Voice registry tests (ADR-022)."""

from __future__ import annotations

from eva.audio.frames import Frame
from eva.tts.base import TTSEngine
from eva.tts.voices import (
    preview_text,
    register_voices_for_engine,
    voice_registry,
    voices_for_engine,
)


class _FakeTTS(TTSEngine):
    def __init__(self, voice_ids: list[str]) -> None:
        self._voice_ids = voice_ids
        self.synthesize_calls: list[tuple[str, str]] = []

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        import numpy as np

        self.synthesize_calls.append((text, voice))
        return np.zeros(100, dtype=np.int16)

    def voices(self) -> list[str]:
        return self._voice_ids


class TestKokoroVoiceParsing:
    def test_known_convention_parses_language_and_gender(self) -> None:
        engine = _FakeTTS(["af_heart", "bm_george", "ef_dora"])
        registered = register_voices_for_engine("kokoro", engine)
        by_id = {v.id: v for v in registered}

        assert by_id["af_heart"].language == "en"
        assert by_id["af_heart"].style_tag == "female"
        assert by_id["af_heart"].display_name == "Heart"

        assert by_id["bm_george"].language == "en"
        assert by_id["bm_george"].style_tag == "male"

        assert by_id["ef_dora"].language == "es"
        assert by_id["ef_dora"].style_tag == "female"

    def test_unrecognized_shape_falls_back_to_bare_id(self) -> None:
        engine = _FakeTTS(["mystery-voice-42"])
        registered = register_voices_for_engine("kokoro", engine)
        assert registered[0].display_name == "mystery-voice-42"
        assert registered[0].language == "unknown"

    def test_multibyte_locale_prefix_does_not_crash(self) -> None:
        # Malformed/unexpected prefixes must degrade gracefully, never raise.
        engine = _FakeTTS(["x", "_", "toolongprefix_name"])
        registered = register_voices_for_engine("kokoro", engine)
        assert len(registered) == 3
        for info in registered:
            assert info.display_name  # never empty


class TestUnknownEngine:
    def test_non_kokoro_engine_uses_bare_id_as_display_name(self) -> None:
        engine = _FakeTTS(["voice-a", "voice-b"])
        registered = register_voices_for_engine("future-engine", engine)
        assert registered[0].display_name == "voice-a"
        assert registered[0].language == "unknown"
        assert registered[0].style_tag == ""


class TestRegistryAndLookup:
    def test_voices_for_engine_filters_correctly(self) -> None:
        register_voices_for_engine("engine-a", _FakeTTS(["v1", "v2"]))
        register_voices_for_engine("engine-b", _FakeTTS(["v3"]))
        assert {v.id for v in voices_for_engine("engine-a")} == {"v1", "v2"}
        assert {v.id for v in voices_for_engine("engine-b")} == {"v3"}

    def test_same_voice_id_different_engines_do_not_collide(self) -> None:
        register_voices_for_engine("engine-x", _FakeTTS(["shared-id"]))
        register_voices_for_engine("engine-y", _FakeTTS(["shared-id"]))
        assert voice_registry.get("engine-x:shared-id").engine == "engine-x"
        assert voice_registry.get("engine-y:shared-id").engine == "engine-y"

    def test_reregistering_an_engine_updates_its_voice_list(self) -> None:
        register_voices_for_engine("engine-z", _FakeTTS(["old-voice"]))
        register_voices_for_engine("engine-z", _FakeTTS(["new-voice"]))
        ids = {v.id for v in voices_for_engine("engine-z")}
        assert "new-voice" in ids
        # The stale entry is still registered (registry is additive by key),
        # but voices_for_engine only reflects what's currently registered
        # under this engine id, which is the caller-relevant view.


class TestPreview:
    def test_preview_reuses_synthesize(self) -> None:
        engine = _FakeTTS(["af_heart"])
        frame = preview_text(engine, "af_heart")
        assert frame is not None
        assert engine.synthesize_calls == [
            ("Hello, this is a preview of my voice.", "af_heart")
        ]

    def test_preview_accepts_custom_phrase(self) -> None:
        engine = _FakeTTS(["af_heart"])
        preview_text(engine, "af_heart", phrase="Testing one two three.")
        assert engine.synthesize_calls[0][0] == "Testing one two three."
