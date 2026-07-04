from __future__ import annotations

from pathlib import Path

import pytest

from eva.config.paths import AppPaths
from eva.core.errors import ModelError, ModelNotInstalledError
from eva.models.catalog import BUILTIN_CATALOG
from eva.models.manager import ModelManager


@pytest.fixture
def manager(app_paths: AppPaths) -> ModelManager:
    return ModelManager(app_paths)


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

    def test_profile_models_exist_in_catalog(self) -> None:
        from eva.hardware.profiles import PROFILES

        ids = {m.id for m in BUILTIN_CATALOG}
        for profile in PROFILES.values():
            assert profile.llm_model in ids, profile.id
            assert profile.asr_model in ids, profile.id


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
        def fake_download(url: str, target: Path, filename: str, progress: object) -> None:
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
            lambda url, target, filename, progress: target.write_bytes(b"x"),
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

        info = manager.info("kokoro-82m-v1.0")
        target_dir = manager.model_dir("kokoro-82m-v1.0")
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = b"0123456789"
        seen_ranges: list[str | None] = []

        # Pre-seed a partial file for the first catalog file only.
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
        manager.download("kokoro-82m-v1.0")
        assert manager.is_installed("kokoro-82m-v1.0")
        assert (target_dir / first.filename).read_bytes() == payload
        assert "bytes=4-" in seen_ranges

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
            lambda url, target, filename, progress: target.write_bytes(b"0" * 2_097_152),
        )
        manager.download("kokoro-82m-v1.0")
        assert manager.disk_usage_mb("kokoro-82m-v1.0") == 4  # 2 files x 2 MB
