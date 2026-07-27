"""Model manager: install, resolve, and remove models from the catalog.

The only component in the product that touches the network (ADR-008).

Two ownership models live here. Manager-managed files download to
`<models_dir>/<kind>/<model_id>/` via a temporary `.part` file and atomic
rename, so an interrupted download never leaves a model half-"installed".
Engine-managed weights (ASR) belong to the engine's own Hugging Face cache
under `<models_dir>/<kind>/models--<org>--<repo>/`; the manager reads and
prefetches that layout rather than duplicating it, so a model installed here
and one the engine fetched lazily are the same bytes on disk.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from eva.config.paths import AppPaths
from eva.core.errors import ModelError, ModelNotInstalledError
from eva.models.catalog import ModelFile, ModelInfo, model_catalog, register_builtin_models

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 256
_DOWNLOAD_ATTEMPTS = 3

ProgressCallback = Callable[[str, int, int], None]
"""(filename, bytes_done, bytes_total) — bytes_total may be 0 if unknown."""


def _reporting_tqdm(label: str, progress: ProgressCallback) -> type:
    """A `tqdm` subclass for `snapshot_download(tqdm_class=…)` that forwards
    byte counts to `progress` instead of drawing a terminal bar.

    `snapshot_download` builds two bars from this class: a files-completed one
    and a shared byte counter that every worker thread updates. Only the byte
    counter is useful here — the file counter would jump 0 → 20 → 100% on a
    model whose weights live in a single 1.5 GB file — so the unit is checked
    before reporting. The byte bar is also created with `disable=` derived from
    the logging level, which switches it off under EVA's default configuration;
    it is forced on, and its output sent nowhere, because this process reports
    through the event bus rather than stdout.
    """
    from huggingface_hub.utils.tqdm import tqdm as hf_tqdm

    class _Sink:
        def write(self, _: str) -> int:
            return 0

        def flush(self) -> None:
            return None

    class _ReportingTqdm(hf_tqdm):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["disable"] = False
            kwargs["file"] = _Sink()
            super().__init__(*args, **kwargs)  # type: ignore[no-untyped-call]

        def update(self, n: float | None = 1) -> bool | None:
            updated = super().update(n)
            if self.unit == "B":
                # The total is only an estimate until every file's metadata has
                # arrived, and the sum of actual sizes can overshoot it.
                total = int(self.total or 0)
                progress(label, min(int(self.n), total) if total else int(self.n), total)
            return updated  # type: ignore[no-any-return]

    return _ReportingTqdm


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
        if info.managed_by == "engine":
            return self._hf_cache_dir(info)
        return self._paths.models_dir / info.kind / model_id.replace("/", "_")

    def _kind_cache_dir(self, info: ModelInfo) -> Path:
        """The Hugging Face cache root an engine-managed model downloads into.

        Engines are handed `<models_dir>/<kind>` as their download root (see
        `eva.asr.registry`), and `huggingface_hub` lays out
        `<root>/models--<org>--<repo>/snapshots/<revision>/…` beneath it. The
        manager mirrors that layout rather than owning one, so prefetching here
        and lazy-loading in the engine populate the *same* files.
        """
        return self._paths.models_dir / info.kind

    def _hf_cache_dir(self, info: ModelInfo) -> Path:
        repo = info.hf_repo or info.id
        return self._kind_cache_dir(info) / f"models--{repo.replace('/', '--')}"

    def _hf_snapshots(self, info: ModelInfo) -> list[Path]:
        """Snapshot directories holding real weights, newest-mtime first.

        A snapshot is only counted when it contains a `config.json` *and* at
        least one weights file: an interrupted download leaves the directory
        and its symlinks in place, so mere existence proves nothing.
        """
        root = self._hf_cache_dir(info) / "snapshots"
        if not root.is_dir():
            return []
        found = [
            snapshot
            for snapshot in root.iterdir()
            if snapshot.is_dir()
            and (snapshot / "config.json").exists()
            and any(snapshot.glob("*.bin"))
        ]
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def is_installed(self, model_id: str) -> bool:
        info = self.info(model_id)
        if info.managed_by == "bundled":
            return True
        if info.managed_by == "engine":
            # Any complete snapshot counts, including one the engine fetched
            # lazily from `main`. Gating on `hf_revision` would report a
            # working model as missing whenever upstream moves the branch.
            return bool(self._hf_snapshots(info))
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
            # Every kind `POST /models/{id}/activate` can set must be detectable
            # here, or the UI shows no active model for that kind (M7 UX fix).
            # Two of them are not `settings.<kind>.model`:
            #   embedding → settings.memory.embedding_model
            #   vad       → settings.vad.engine holds an ENGINE id, not a model
            #               id, so it is compared against info.engine instead.
            active_ids = {
                getattr(getattr(settings, kind, None), "model", None)
                for kind in ("llm", "asr", "tts")
            }
            memory = getattr(settings, "memory", None)
            active_ids.add(getattr(memory, "embedding_model", None))
            if info.kind == "vad":
                vad = getattr(settings, "vad", None)
                active = info.engine == getattr(vad, "engine", None)
            else:
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
            "recommendation": info.recommendation,
            "notes": info.notes,
        }

    def disk_usage_mb(self, model_id: str) -> int:
        directory = self.model_dir(model_id)
        if self.info(model_id).managed_by == "engine":
            # Hugging Face stores one copy per file in `blobs/` and links it
            # into each snapshot. Walking the whole tree counts every byte
            # twice (links included), so measure the blobs alone.
            directory = directory / "blobs"
        if not directory.exists():
            return 0
        return sum(f.stat().st_size for f in directory.glob("**/*") if f.is_file()) // 1_048_576

    # ── install / remove ──

    def download(self, model_id: str, progress: ProgressCallback | None = None) -> None:
        info = self.info(model_id)
        if info.managed_by == "bundled":
            logger.info("Model '%s' ships with EVA; nothing to download", model_id)
            return
        if info.managed_by == "engine":
            self._download_from_hub(info, progress)
            return
        target_dir = self.model_dir(model_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in info.files:
            target = target_dir / file.filename
            if target.exists():
                logger.info("%s already present — skipping", file.filename)
                continue
            self._download_file(file, target, progress)
        logger.info("Model '%s' installed", model_id)

    def remove(self, model_id: str) -> None:
        info = self.info(model_id)
        if info.managed_by == "bundled":
            raise ModelError(f"Model '{model_id}' is bundled and cannot be removed")
        directory = self.model_dir(model_id)
        if directory.exists():
            shutil.rmtree(directory)
            logger.info("Model '%s' removed", model_id)

    def _download_from_hub(self, info: ModelInfo, progress: ProgressCallback | None) -> None:
        """Prefetch an engine-managed model into the engine's own cache.

        The engine would fetch these weights itself on first use; doing it here
        makes the download explicit and observable, and is what lets the UI
        offer a Download button instead of a silent multi-minute stall inside
        the first turn. Because the destination is the engine's cache root, a
        prefetched model and a lazily-fetched one are the same files — this
        adds no second copy.
        """
        from huggingface_hub import snapshot_download

        cache_dir = self._kind_cache_dir(info)
        cache_dir.mkdir(parents=True, exist_ok=True)
        repo = info.hf_repo or info.id
        logger.info("Downloading %s from the Hugging Face Hub", repo)
        try:
            snapshot_download(
                repo_id=repo,
                revision=info.hf_revision or None,
                cache_dir=str(cache_dir),
                tqdm_class=_reporting_tqdm(repo, progress) if progress else None,
            )
        except Exception as exc:  # network, auth, missing repo — all fatal here
            raise ModelError(f"Could not download '{info.id}' from {repo}: {exc}") from exc
        logger.info("Model '%s' installed", info.id)

    def _download_file(
        self, file: ModelFile, target: Path, progress: ProgressCallback | None
    ) -> None:
        """Download with resume (HTTP Range) and integrity verification.

        A dropped connection surfaces as a short read, not an exception, so the
        received byte count MUST be checked against Content-Length — a silently
        truncated model file loads as "corrupted" much later and is far harder
        to diagnose. Partial data stays in the `.part` file and is resumed on
        retry (here and on any later download attempt).

        Verification ladder (M5.6): the catalog's `size_bytes` (exact upstream
        size) is checked in addition to Content-Length — it also covers the
        case where the server sends no Content-Length at all, which the old
        byte-count check silently waved through. When the catalog carries a
        `sha256`, the completed file is hashed and MUST match; a mismatched
        file is deleted (never resumed — the bytes themselves are wrong).
        A file with neither hash nor known size is accepted with a logged
        warning, never silently.
        """
        filename = file.filename
        part = target.with_suffix(target.suffix + ".part")
        logger.info("Downloading %s", file.url)
        total = 0
        try:
            for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
                done = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={done}-"} if done else {}
                request = urllib.request.Request(file.url, headers=headers)
                with urllib.request.urlopen(request) as response:
                    resumed = response.status == 206
                    if done and not resumed:
                        done = 0  # server ignored the range request; restart
                    total = done + int(response.headers.get("Content-Length", 0))
                    if not total and file.size_bytes:
                        total = file.size_bytes  # no Content-Length; catalog knows the size
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
        expected = file.size_bytes or total
        if expected and received < expected:
            raise ModelError(
                f"Download of {filename} is incomplete ({received} of {expected} bytes); "
                "re-run the download to resume"
            )
        if file.size_bytes and received != file.size_bytes:
            # More bytes than the published file has: not a resumable gap but
            # wrong content (changed upstream file, proxy error page, ...).
            part.unlink(missing_ok=True)
            raise ModelError(
                f"Download of {filename} does not match its published size "
                f"({received} bytes received, {file.size_bytes} expected) — "
                "the partial file was discarded; re-run the download"
            )
        if file.sha256:
            digest = self._sha256_of(part)
            if digest != file.sha256:
                part.unlink(missing_ok=True)
                raise ModelError(
                    f"Checksum mismatch for {filename}: the downloaded file does not "
                    "match the published SHA-256 — the file was discarded. Re-run the "
                    "download; if this repeats, the upstream file may have changed."
                )
            logger.info("Checksum verified for %s", filename)
        elif not file.size_bytes and not total:
            logger.warning(
                "%s could not be verified (no Content-Length, no published size or "
                "checksum in the catalog) — accepting as-is",
                filename,
            )
        part.replace(target)

    @staticmethod
    def _sha256_of(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()
