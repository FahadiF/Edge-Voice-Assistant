"""Model manager: install, resolve, and remove models from the catalog.

The only component in the product that touches the network (ADR-008). Files
download to `<models_dir>/<kind>/<model_id>/` via a temporary `.part` file and
atomic rename, so an interrupted download never leaves a model half-"installed".
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from eva.config.paths import AppPaths
from eva.core.errors import ModelError, ModelNotInstalledError
from eva.models.catalog import ModelInfo, model_catalog, register_builtin_models

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 256
_DOWNLOAD_ATTEMPTS = 3

ProgressCallback = Callable[[str, int, int], None]
"""(filename, bytes_done, bytes_total) — bytes_total may be 0 if unknown."""


class ModelManager:
    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths
        register_builtin_models()

    # ── catalog queries ──

    def available(self, kind: str | None = None) -> list[ModelInfo]:
        models = list(model_catalog.snapshot().values())
        if kind is not None:
            models = [m for m in models if m.kind == kind]
        return sorted(models, key=lambda m: (m.kind, m.id))

    def info(self, model_id: str) -> ModelInfo:
        return model_catalog.get(model_id)

    # ── installation state ──

    def model_dir(self, model_id: str) -> Path:
        info = self.info(model_id)
        return self._paths.models_dir / info.kind / model_id.replace("/", "_")

    def is_installed(self, model_id: str) -> bool:
        info = self.info(model_id)
        if info.managed_by == "bundled":
            return True
        if info.managed_by == "engine":
            # Engine-managed weights land under the manager's directory tree;
            # treat presence of any content as installed, absence as "first use
            # will download".
            return any(self.model_dir(model_id).glob("**/*"))
        return all((self.model_dir(model_id) / f.filename).exists() for f in info.files)

    def installed(self, kind: str | None = None) -> list[ModelInfo]:
        return [m for m in self.available(kind) if self.is_installed(m.id)]

    def files_for(self, model_id: str) -> dict[str, Path]:
        """Resolve the installed file paths keyed by their engine role."""
        info = self.info(model_id)
        if not self.is_installed(model_id):
            raise ModelNotInstalledError(
                f"Model '{model_id}' is not installed — run: eva models download {model_id}"
            )
        return {f.key: self.model_dir(model_id) / f.filename for f in info.files}

    def describe(self, model_id: str, settings: object = None) -> dict[str, object]:
        """Complete model card for UIs: metadata + install state + compatibility.

        `settings` (a Settings instance) marks which models are active; omitted
        in contexts that only need catalog + install state.
        """
        from eva.hardware import detect_hardware

        info = self.info(model_id)
        installed = self.is_installed(model_id)
        report = detect_hardware()
        gpu = report.best_gpu
        vram_available = gpu.vram_total_mb if gpu else 0
        fits_gpu = info.vram_mb == 0 or info.vram_mb <= vram_available
        fits_ram = info.ram_mb <= report.memory.total_mb

        active = False
        if settings is not None:
            active_ids = {
                getattr(getattr(settings, kind, None), "model", None)
                for kind in ("llm", "asr", "tts")
            }
            active = model_id in active_ids

        return {
            "id": info.id,
            "name": info.display_name,
            "kind": info.kind,
            "version": info.version,
            "provider": info.provider,
            "license": info.license,
            "languages": info.languages,
            "context_length": info.context_length,
            "quantization": info.quantization,
            "vram_mb": info.vram_mb,
            "ram_mb": info.ram_mb,
            "download_mb": info.download_mb,
            "disk_usage_mb": self.disk_usage_mb(model_id) if installed else 0,
            "engine": info.engine,
            "managed_by": info.managed_by,
            "installed": installed,
            "installed_version": info.version if installed else None,
            "update_available": False,  # populated when remote catalogs land
            "active": active,
            "compatible": fits_gpu and fits_ram,
            "compatibility_notes": (
                "" if fits_gpu else f"needs {info.vram_mb} MB VRAM, {vram_available} MB detected"
            ),
            "notes": info.notes,
        }

    def disk_usage_mb(self, model_id: str) -> int:
        directory = self.model_dir(model_id)
        if not directory.exists():
            return 0
        return sum(f.stat().st_size for f in directory.glob("**/*") if f.is_file()) // 1_048_576

    # ── install / remove ──

    def download(self, model_id: str, progress: ProgressCallback | None = None) -> None:
        info = self.info(model_id)
        if info.managed_by != "manager":
            logger.info("Model '%s' is %s-managed; nothing to download", model_id, info.managed_by)
            return
        target_dir = self.model_dir(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in info.files:
            target = target_dir / file.filename
            if target.exists():
                logger.info("%s already present — skipping", file.filename)
                continue
            self._download_file(file.url, target, file.filename, progress)
        logger.info("Model '%s' installed", model_id)

    def remove(self, model_id: str) -> None:
        info = self.info(model_id)
        if info.managed_by == "bundled":
            raise ModelError(f"Model '{model_id}' is bundled and cannot be removed")
        directory = self.model_dir(model_id)
        if directory.exists():
            shutil.rmtree(directory)
            logger.info("Model '%s' removed", model_id)

    def _download_file(
        self, url: str, target: Path, filename: str, progress: ProgressCallback | None
    ) -> None:
        """Download with resume (HTTP Range) and size verification.

        A dropped connection surfaces as a short read, not an exception, so the
        received byte count MUST be checked against Content-Length — a silently
        truncated model file loads as "corrupted" much later and is far harder
        to diagnose. Partial data stays in the `.part` file and is resumed on
        retry (here and on any later download attempt).
        """
        part = target.with_suffix(target.suffix + ".part")
        logger.info("Downloading %s", url)
        total = 0
        try:
            for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
                done = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={done}-"} if done else {}
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request) as response:
                    resumed = response.status == 206
                    if done and not resumed:
                        done = 0  # server ignored the range request; restart
                    total = done + int(response.headers.get("Content-Length", 0))
                    with part.open("ab" if done else "wb") as out:
                        while True:
                            chunk = response.read(_CHUNK_SIZE)
                            if not chunk:
                                break
                            out.write(chunk)
                            done += len(chunk)
                            if progress is not None:
                                progress(filename, done, total)
                if not total or done >= total:
                    break
                logger.warning(
                    "Short read for %s (%d of %d bytes), retry %d/%d",
                    filename,
                    done,
                    total,
                    attempt,
                    _DOWNLOAD_ATTEMPTS,
                )
        except (urllib.error.URLError, OSError) as exc:
            raise ModelError(f"Download failed for {filename}: {exc}") from exc

        received = part.stat().st_size if part.exists() else 0
        if total and received != total:
            raise ModelError(
                f"Download of {filename} is incomplete ({received} of {total} bytes); "
                "re-run the download to resume"
            )
        part.replace(target)
