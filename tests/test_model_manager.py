from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eva.config.paths import AppPaths
from eva.core.errors import ModelError, ModelNotInstalledError
from eva.models.catalog import BUILTIN_CATALOG, ModelFile, ModelInfo, model_catalog
from eva.models.manager import ModelManager


@pytest.fixture
def manager(app_paths: AppPaths) -> ModelManager:
    return ModelManager(app_paths)


def _register_test_model(suffix: str, *, payload: bytes, verified: bool) -> str:
    """Register (once) a synthetic single-file catalog entry for download
    tests. `verified=True` stamps the payload's real size and SHA-256 into
    the catalog entry, exercising the M5.6 integrity path; `verified=False`
    leaves both unset (the pre-M5.6 trust model)."""
    model_id = f"test-download-{suffix}"
    if model_id not in model_catalog:
        model_catalog.register(
            model_id,
            ModelInfo(
                id=model_id,
                kind="tts",
                display_name=f"Test model ({suffix})",
                engine="test",
                license="none",
                files=(
                    ModelFile(
                        key="model",
                        url=f"https://example.invalid/{model_id}.bin",
                        filename=f"{model_id}.bin",
                        size_mb=1,
                        size_bytes=len(payload) if verified else 0,
                        sha256=hashlib.sha256(payload).hexdigest() if verified else "",
                    ),
                ),
            ),
        )
    return model_id


class _FakeResponse:
    """Minimal urlopen-response stand-in supporting truncation and ranges."""

    def __init__(self, payload: bytes, *, claimed_total: int, status: int = 200) -> None:
        self._payload = payload
        self._pos = 0
        self.status = status
        self.headers = {"Content-Length": str(claimed_total)}

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestCatalog:
    def test_ids_unique(self) -> None:
        ids = [m.id for m in BUILTIN_CATALOG]
        assert len(ids) == len(set(ids))

    def test_urls_are_https(self) -> None:
        for model in BUILTIN_CATALOG:
            for file in model.files:
                assert file.url.startswith("https://"), f"{model.id}: {file.url}"

    def test_settings_defaults_exist_in_catalog(self) -> None:
        from eva.config.settings import Settings

        ids = {m.id for m in BUILTIN_CATALOG}
        s = Settings()
        assert s.llm.model in ids
        assert s.asr.model in ids

    def test_every_tier_resolves_to_catalogued_models(self) -> None:
        """Hardware profiles no longer carry their own model copies — a tier's
        models come from the presets. This replaces the old
        `test_profile_models_exist_in_catalog`, whose coverage is a strict
        subset of `test_presets_and_persistence`'s all-presets check; kept here
        because it pins the *tier ids* the two modules must agree on."""
        from eva.hardware.presets import preset_registry, register_builtin_presets
        from eva.hardware.profiles import PROFILES

        register_builtin_presets()
        ids = {m.id for m in BUILTIN_CATALOG}
        balanced = preset_registry.get("balanced")
        for tier_id in PROFILES:
            models = balanced.for_tier(tier_id)
            assert models.llm_model in ids, tier_id
            assert models.asr_model in ids, tier_id


class TestManager:
    def test_not_installed_initially(self, manager: ModelManager) -> None:
        assert not manager.is_installed("qwen3.5-4b-instruct-q4_k_m")
        assert manager.installed("llm") == []

    def test_bundled_models_always_installed(self, manager: ModelManager) -> None:
        assert manager.is_installed("silero-vad-v5")

    def test_files_for_missing_model_raises(self, manager: ModelManager) -> None:
        with pytest.raises(ModelNotInstalledError, match="eva models download"):
            manager.files_for("kokoro-82m-v1.0")

    def test_download_and_resolve(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(file: object, target: Path, progress: object) -> None:
            target.write_bytes(b"weights")

        monkeypatch.setattr(manager, "_download_file", fake_download)
        manager.download("kokoro-82m-v1.0")
        assert manager.is_installed("kokoro-82m-v1.0")
        files = manager.files_for("kokoro-82m-v1.0")
        assert set(files) == {"model", "voices"}
        assert files["model"].exists()

    def test_remove(self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            manager,
            "_download_file",
            lambda file, target, progress: target.write_bytes(b"x"),
        )
        manager.download("kokoro-82m-v1.0")
        manager.remove("kokoro-82m-v1.0")
        assert not manager.is_installed("kokoro-82m-v1.0")

    def test_remove_bundled_rejected(self, manager: ModelManager) -> None:
        with pytest.raises(ModelError):
            manager.remove("silero-vad-v5")

    def test_failed_download_is_not_installed(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.request

        def broken_urlopen(request: object) -> object:
            raise OSError("network down")

        monkeypatch.setattr(urllib.request, "urlopen", broken_urlopen)
        with pytest.raises(ModelError, match="Download failed"):
            manager.download("kokoro-82m-v1.0")
        assert not manager.is_installed("kokoro-82m-v1.0")

    def test_truncated_download_raises_incomplete(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dropped connection (short read, no exception) must not install."""
        import urllib.request

        monkeypatch.setattr(
            urllib.request, "urlopen", lambda request: _FakeResponse(b"half", claimed_total=100)
        )
        with pytest.raises(ModelError, match="incomplete"):
            manager.download("kokoro-82m-v1.0")
        assert not manager.is_installed("kokoro-82m-v1.0")
        # Partial data is retained for resume.
        assert list(manager.model_dir("kokoro-82m-v1.0").glob("*.part"))

    def test_resume_completes_partial_download(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.request

        payload = b"0123456789"
        model_id = _register_test_model("resume", payload=payload, verified=False)
        info = manager.info(model_id)
        target_dir = manager.model_dir(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        seen_ranges: list[str | None] = []

        # Pre-seed a partial file.
        first = info.files[0]
        (target_dir / (first.filename + ".part")).write_bytes(payload[:4])

        def fake_urlopen(request: urllib.request.Request) -> _FakeResponse:
            range_header = request.headers.get("Range")
            seen_ranges.append(range_header)
            if range_header:
                start = int(range_header.split("=")[1].rstrip("-"))
                return _FakeResponse(
                    payload[start:], claimed_total=len(payload) - start, status=206
                )
            return _FakeResponse(payload, claimed_total=len(payload))

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        manager.download(model_id)
        assert manager.is_installed(model_id)
        assert (target_dir / first.filename).read_bytes() == payload
        assert "bytes=4-" in seen_ranges

    def test_checksum_verified_download_installs(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.request

        payload = b"verified model weights"
        model_id = _register_test_model("sha-ok", payload=payload, verified=True)
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _FakeResponse(payload, claimed_total=len(payload)),
        )
        manager.download(model_id)
        assert manager.is_installed(model_id)

    def test_checksum_mismatch_discards_file(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wrong bytes of the right length must be rejected AND deleted —
        a poisoned .part must never be 'resumed' into an install."""
        import urllib.request

        good = b"the authentic payload!"
        model_id = _register_test_model("sha-bad", payload=good, verified=True)
        tampered = b"x" * len(good)  # right size, wrong content
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _FakeResponse(tampered, claimed_total=len(tampered)),
        )
        with pytest.raises(ModelError, match="Checksum mismatch"):
            manager.download(model_id)
        assert not manager.is_installed(model_id)
        assert not list(manager.model_dir(model_id).glob("*.part"))

    def test_oversized_download_discards_file(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """More bytes than the published size is wrong content (proxy error
        page, changed upstream file) — discard, don't 'resume'."""
        import urllib.request

        good = b"tiny"
        model_id = _register_test_model("oversize", payload=good, verified=True)
        bloated = b"<html>proxy error page pretending to be a model</html>"
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _FakeResponse(bloated, claimed_total=len(bloated)),
        )
        with pytest.raises(ModelError, match="published size"):
            manager.download(model_id)
        assert not manager.is_installed(model_id)
        assert not list(manager.model_dir(model_id).glob("*.part"))

    def test_missing_content_length_with_known_size_detects_truncation(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-M5.6 hole: no Content-Length used to mean 'no check at
        all' — with the catalog's size_bytes, truncation is still caught."""
        import urllib.request

        payload = b"complete payload bytes"
        model_id = _register_test_model("no-cl", payload=payload, verified=True)
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _FakeResponse(payload[: len(payload) // 2], claimed_total=0),
        )
        with pytest.raises(ModelError, match="incomplete"):
            manager.download(model_id)
        assert not manager.is_installed(model_id)

    def test_builtin_catalog_carries_integrity_metadata(self) -> None:
        """Every manager-downloaded builtin file must publish its exact size;
        hashes are required wherever the publisher exposes them (HF LFS)."""
        for model in BUILTIN_CATALOG:
            for file in model.files:
                assert file.size_bytes > 0, f"{model.id}/{file.filename}: no size_bytes"
                if "huggingface.co" in file.url:
                    assert len(file.sha256) == 64, f"{model.id}/{file.filename}: no sha256"

    def test_describe_full_model_card(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from eva.config.settings import Settings

        card = manager.describe("qwen3.5-4b-instruct-q4_k_m", Settings())
        for key in (
            "name",
            "version",
            "provider",
            "license",
            "languages",
            "context_length",
            "quantization",
            "vram_mb",
            "ram_mb",
            "disk_usage_mb",
            "installed",
            "update_available",
            "active",
            "compatible",
        ):
            assert key in card, key
        assert card["provider"] == "Alibaba (Qwen)"
        assert card["active"] is True  # it is the settings default
        assert card["installed"] is False  # isolated test home

    def test_describe_detects_the_active_model_for_every_kind(self, manager: ModelManager) -> None:
        """`POST /models/{id}/activate` can set all five kinds, so `describe()`
        must be able to report all five as active. Two do not live at
        `settings.<kind>.model`: embeddings are configured under `memory`, and
        `vad` stores an *engine* id rather than a model id."""
        from eva.config.settings import Settings

        settings = Settings()
        for model_id in (
            settings.llm.model,
            settings.asr.model,
            settings.tts.model,
            settings.memory.embedding_model,
            "silero-vad-v5",  # matched on engine id, not model id
        ):
            assert manager.describe(model_id, settings)["active"] is True, model_id

    def test_describe_does_not_mark_unrelated_models_active(self, manager: ModelManager) -> None:
        from eva.config.settings import Settings

        card = manager.describe("qwen3.5-9b-instruct-q4_k_m", Settings())
        assert card["active"] is False

    def test_turbo_is_catalogued_as_a_multilingual_option(self, manager: ModelManager) -> None:
        """large-v3-turbo exists and is selectable. It is deliberately NOT a
        tier default yet — that waits on the benchmark (docs/BACKLOG.md)."""
        turbo = manager.info("faster-whisper/large-v3-turbo")
        assert turbo.kind == "asr"
        assert turbo.engine == "faster-whisper"
        assert turbo.managed_by == "engine"
        assert turbo.languages == "multilingual"
        assert turbo.vram_mb == 1200  # measured: 1106-1121 MiB resident
        assert turbo.download_mb == 1600

    def test_no_asr_model_claims_a_gpu_tier_it_was_never_measured_on(
        self, manager: ModelManager
    ) -> None:
        """`distil-large-v3` used to advertise "for 12 GB+ GPUs" on an estimate
        no one had checked, while the *larger* turbo measured 1121 MiB on a 6 GB
        card. Guidance text must not imply a hardware requirement."""
        for model in manager.available("asr"):
            blurb = f"{model.notes} {model.recommendation}"
            assert "12 GB" not in blurb, model.id
            assert "GB+" not in blurb, model.id

    def test_engine_managed_models_record_their_upstream_repo(self, manager: ModelManager) -> None:
        """Provenance for engine-managed weights. Informational until M1b makes
        downloads repository-aware, but recorded now because the engine's alias
        map is an implementation detail — `large-v3-turbo` resolves to a
        third-party repo, unlike the first-party Systran entries."""
        for model in manager.available():
            if model.managed_by != "engine":
                continue
            assert model.hf_repo, f"{model.id} has no hf_repo"
            assert "/" in model.hf_repo, f"{model.id} hf_repo is not owner/name"
        turbo = manager.info("faster-whisper/large-v3-turbo")
        assert turbo.hf_repo == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
        # Only revisions verified against a real download are pinned; a guessed
        # one would look authoritative while being fiction.
        assert len(turbo.hf_revision) == 40, "turbo revision must be a full SHA"

    def test_describe_carries_catalog_recommendation(self, manager: ModelManager) -> None:
        """Guidance shown in pickers is catalog data, not a UI conditional, so
        every client renders the same text and a new model ships its own."""
        card = manager.describe("qwen3.5-4b-instruct-q4_k_m")
        assert card["recommendation"] == "Recommended for most users"
        # Models without guidance return "" rather than omitting the key, so
        # clients can render unconditionally.
        assert manager.describe("silero-vad-v5")["recommendation"] == ""

    def test_every_downloadable_model_states_a_size(self, manager: ModelManager) -> None:
        """The Models page shows the download size on the button; a zero here
        silently degrades it back to a bare "Download"."""
        for model in manager.available():
            if model.managed_by == "bundled":
                continue
            assert model.download_mb > 0, f"{model.id} has no download size"

    def test_describe_flags_incompatible_models(
        self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from eva.hardware.detect import CpuInfo, HardwareReport, MemoryInfo

        cpu_only = HardwareReport(
            os_name="t",
            os_version="1",
            python_version="3.12",
            cpu=CpuInfo(name="c", physical_cores=4, logical_cores=8),
            memory=MemoryInfo(total_mb=8000, available_mb=4000),
            gpus=[],
        )
        monkeypatch.setattr("eva.hardware.detect.detect_hardware", lambda: cpu_only)
        monkeypatch.setattr("eva.hardware.detect_hardware", lambda: cpu_only)
        card = manager.describe("qwen3.5-9b-instruct-q4_k_m")
        assert card["compatible"] is False
        assert "VRAM" in str(card["compatibility_notes"])

    def test_disk_usage(self, manager: ModelManager, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            manager,
            "_download_file",
            lambda file, target, progress: target.write_bytes(b"0" * 2_097_152),
        )
        manager.download("kokoro-82m-v1.0")
        assert manager.disk_usage_mb("kokoro-82m-v1.0") == 4  # 2 files x 2 MB
