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
        vram_mb=1600,
        ram_mb=2000,
        languages="en",
        download_mb_hint=1500,
        notes="High-accuracy English ASR for 12 GB+ GPUs; ~1.5 GB download on first use.",
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
