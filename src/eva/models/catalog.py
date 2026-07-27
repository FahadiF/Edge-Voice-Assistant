"""Built-in model catalog.

The catalog is **data**: each entry describes what a model is (kind, license,
resource needs) and how to obtain it (download URLs, filenames). The settings
UI, model manager, hardware presets, and benchmark suite all read this one
source. Third-party catalogs can extend it through the registry at runtime.

`managed_by="engine"` entries (faster-whisper) are downloaded by their engine
into the manager's directory rather than by the manager itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from eva.core.registry import Registry

ModelKind = Literal["llm", "asr", "tts", "vad", "embedding"]


class ModelFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str  # role of the file for the engine ("model", "voices", …)
    url: str
    filename: str
    size_mb: int
    # Integrity metadata (M5.6). `size_bytes` is the exact upstream file size;
    # `sha256` the upstream content hash — both from the publisher's own
    # metadata (HF LFS API / GitHub release assets), never computed from a
    # downloaded copy. Empty/zero means "publisher exposes no such metadata";
    # the manager then verifies what it can and logs that the file is only
    # size-checked. A mismatch on either is a hard failure, not a warning.
    size_bytes: int = 0
    sha256: str = ""


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: ModelKind
    display_name: str
    engine: str  # engine registry id this model runs on
    provider: str = ""  # who publishes the weights (Alibaba, Systran, …)
    version: str = "1.0"  # catalog entry version; update checks compare this
    license: str
    files: tuple[ModelFile, ...] = ()
    managed_by: Literal["manager", "engine", "bundled"] = "manager"
    vram_mb: int = 0  # 0 = CPU-resident
    ram_mb: int = 0
    context_length: int | None = None  # LLM only
    quantization: str | None = None
    languages: str = "multilingual"
    hf_repo: str = ""
    """Upstream Hugging Face repository for engine-managed weights.

    **Informational only today.** The faster-whisper adapter still passes the
    model *alias* (e.g. `large-v3-turbo`), which the engine resolves through its
    own internal map — so this field does not yet control what is downloaded.
    It is recorded now because that map is an engine implementation detail that
    can change between versions, and `large-v3-turbo` in particular resolves to
    a third-party repository rather than the first-party `Systran` ones. M1b
    makes downloads repository-aware, at which point this becomes authoritative.
    """
    hf_revision: str = ""
    """Pinned upstream commit for `hf_repo`; empty means "whatever is current".

    Same status as `hf_repo`: informational until M1b. Only populated where the
    revision has been verified against a real local download — a guessed
    revision is worse than none, because it would look authoritative.
    """
    recommendation: str = ""
    """Short guidance shown next to the model in pickers ("Recommended for most
    users", "Fastest", "English only", …). Catalog data rather than a UI
    conditional so a new model ships its own guidance and every client — web
    UI, CLI, future pickers — shows the same thing."""
    notes: str = ""
    # Download-size hint for engine-managed models (whose files are fetched by
    # the engine, not the manager, so `files` is empty). 0 = derive from `files`.
    download_mb_hint: int = 0

    @property
    def download_mb(self) -> int:
        """Approximate download size in MB (0 if nothing to download)."""
        return self.download_mb_hint or sum(f.size_mb for f in self.files)


def _hf(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


BUILTIN_CATALOG: tuple[ModelInfo, ...] = (
    # ── LLM (GGUF for llama.cpp) ──
    ModelInfo(
        id="qwen3.5-4b-instruct-q4_k_m",
        kind="llm",
        display_name="Qwen3.5 4B Instruct (Q4_K_M)",
        engine="llamacpp",
        provider="Alibaba (Qwen)",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url=_hf("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
                filename="Qwen3.5-4B-Q4_K_M.gguf",
                size_mb=2700,
                size_bytes=2_740_937_888,
                sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
            ),
        ),
        vram_mb=3400,
        ram_mb=1000,
        context_length=32768,
        quantization="Q4_K_M",
        recommendation="Recommended for most users",
        notes="Default assistant model for the gpu-6gb tier.",
    ),
    ModelInfo(
        id="qwen3-4b-instruct-q4_k_m",
        kind="llm",
        display_name="Qwen3 4B Instruct 2507 (Q4_K_M)",
        engine="llamacpp",
        provider="Alibaba (Qwen)",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url=_hf(
                    "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
                ),
                filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                size_mb=2500,
                size_bytes=2_497_281_120,
                sha256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
            ),
        ),
        vram_mb=3200,
        ram_mb=1000,
        context_length=32768,
        quantization="Q4_K_M",
        recommendation="Compatibility fallback",
        notes="Fallback default if the runtime predates Qwen3.5 support.",
    ),
    ModelInfo(
        id="qwen3.5-9b-instruct-q4_k_m",
        kind="llm",
        display_name="Qwen3.5 9B Instruct (Q4_K_M)",
        engine="llamacpp",
        provider="Alibaba (Qwen)",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url=_hf("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf"),
                filename="Qwen3.5-9B-Q4_K_M.gguf",
                size_mb=5800,
                size_bytes=5_680_522_464,
                sha256="03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
            ),
        ),
        vram_mb=7200,
        ram_mb=1500,
        context_length=32768,
        quantization="Q4_K_M",
        recommendation="Best quality · needs 12 GB GPU",
        notes="Default for the gpu-12gb tier.",
    ),
    ModelInfo(
        id="qwen3-1.7b-instruct-q4_k_m",
        kind="llm",
        display_name="Qwen3 1.7B (Q4_K_M)",
        engine="llamacpp",
        provider="Alibaba (Qwen)",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url=_hf("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
                filename="Qwen3-1.7B-Q4_K_M.gguf",
                size_mb=1100,
                size_bytes=1_107_409_472,
                sha256="b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897",
            ),
        ),
        vram_mb=1600,
        ram_mb=800,
        context_length=32768,
        quantization="Q4_K_M",
        recommendation="Low memory · fastest",
        notes="CPU-only and low-memory tiers.",
    ),
    # ── ASR (faster-whisper sizes; weights fetched by the engine) ──
    ModelInfo(
        id="faster-whisper/small",
        kind="asr",
        display_name="Whisper small (CTranslate2 int8)",
        engine="faster-whisper",
        provider="OpenAI / Systran",
        license="MIT",
        managed_by="engine",
        vram_mb=600,
        ram_mb=900,
        download_mb_hint=460,
        hf_repo="Systran/faster-whisper-small",
        hf_revision="536b0662742c02347bc0e980a01041f333bce120",
        recommendation="Recommended for most users",
        notes="Default ASR for GPU tiers; ~460 MB download on first use.",
    ),
    ModelInfo(
        id="faster-whisper/base",
        kind="asr",
        display_name="Whisper base (CTranslate2 int8)",
        engine="faster-whisper",
        provider="OpenAI / Systran",
        license="MIT",
        managed_by="engine",
        vram_mb=300,
        ram_mb=500,
        download_mb_hint=140,
        hf_repo="Systran/faster-whisper-base",
        recommendation="Low memory · fastest",
        notes="CPU-tier ASR; ~140 MB download on first use.",
    ),
    ModelInfo(
        id="faster-whisper/distil-large-v3",
        kind="asr",
        display_name="Distil-Whisper large-v3 (CTranslate2)",
        engine="faster-whisper",
        provider="Hugging Face / Systran",
        license="MIT",
        managed_by="engine",
        # vram_mb/ram_mb here are the original catalog estimates and have never
        # been measured. Deliberately left as-is rather than replaced with a
        # fresh guess; measuring them is a benchmark task (docs/BACKLOG.md).
        # For scale: large-v3-turbo is the larger model (~809M vs ~756M
        # parameters) and measured 1121 MiB resident on a 6 GB card, so the
        # "12 GB+ GPUs" guidance this entry used to carry was unsupported.
        vram_mb=1600,
        ram_mb=2000,
        languages="en",
        download_mb_hint=1500,
        hf_repo="Systran/faster-distil-whisper-large-v3",
        recommendation="English only",
        notes="English-only alternative; ~1.5 GB download on first use.",
    ),
    ModelInfo(
        id="faster-whisper/large-v3-turbo",
        kind="asr",
        display_name="Whisper large-v3-turbo (CTranslate2)",
        engine="faster-whisper",
        # Third-party CTranslate2 conversion — every other ASR entry is
        # first-party Systran. Named honestly so the difference is visible.
        provider="OpenAI / mobiuslabsgmbh",
        license="MIT",
        managed_by="engine",
        # Measured on an RTX 3060 Laptop (6 GB): 1106-1121 MiB resident
        # alongside the 4B LLM, 1.6 GB on disk. ram_mb follows the existing
        # catalog convention rather than a measurement.
        vram_mb=1200,
        ram_mb=1600,
        download_mb_hint=1600,
        hf_repo="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        hf_revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        recommendation="Best accuracy · multilingual",
        notes=(
            "Multilingual high-accuracy ASR; fits a 6 GB GPU alongside a 4B LLM. "
            "~1.6 GB download on first use. Not a tier default yet — pending the "
            "benchmark recorded in docs/BACKLOG.md."
        ),
    ),
    # ── TTS ──
    ModelInfo(
        id="kokoro-82m-v1.0",
        kind="tts",
        display_name="Kokoro 82M v1.0 (ONNX)",
        engine="kokoro",
        provider="Hexgrad",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
                filename="kokoro-v1.0.onnx",
                size_mb=310,
                # GitHub release assets expose exact sizes but no content
                # hash — size-verified only (logged by the manager).
                size_bytes=325_532_387,
            ),
            ModelFile(
                key="voices",
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
                filename="voices-v1.0.bin",
                size_mb=27,
                size_bytes=28_214_398,
            ),
        ),
        ram_mb=700,
        languages="en, es, fr, hi, it, ja, pt, zh",
        notes="Default TTS; runs faster than real time on CPU.",
    ),
    # ── VAD (bundled inside pysilero-vad) ──
    ModelInfo(
        id="silero-vad-v5",
        kind="vad",
        display_name="Silero VAD v5 (ONNX)",
        engine="silero",
        provider="Silero Team",
        license="MIT",
        managed_by="bundled",
        ram_mb=50,
    ),
    # ── Embedding (M4, ADR-020: semantic memory search) ──
    ModelInfo(
        id="all-minilm-l6-v2-onnx",
        kind="embedding",
        display_name="all-MiniLM-L6-v2 (ONNX)",
        engine="onnx-embedding",
        provider="sentence-transformers / Xenova (ONNX export)",
        license="Apache-2.0",
        files=(
            ModelFile(
                key="model",
                url=_hf("Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx"),
                filename="all-minilm-l6-v2.onnx",
                size_mb=23,
                size_bytes=22_972_370,
                sha256="afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
            ),
            ModelFile(
                key="tokenizer",
                url=_hf("Xenova/all-MiniLM-L6-v2", "tokenizer.json"),
                filename="all-minilm-l6-v2-tokenizer.json",
                size_mb=1,
                size_bytes=711_661,
                sha256="da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
            ),
        ),
        ram_mb=200,
        languages="en",
        notes=(
            "384-dim sentence embeddings for memory semantic search (ADR-020). "
            "Optional: memory search still works via keyword/FTS without it."
        ),
    ),
)

model_catalog: Registry[ModelInfo] = Registry("model")


def register_builtin_models() -> None:
    for info in BUILTIN_CATALOG:
        if info.id not in model_catalog:
            model_catalog.register(info.id, info)
