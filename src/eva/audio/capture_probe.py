"""Capture probe: raw vs processed microphone audio, decoded side by side.

A **temporary developer diagnostic** for the M7.2 recognition-quality
investigation, not part of the conversation engine and not used by it. It
answers one question: *where is speech information lost?*

One pass of the real capture chain produces two aligned recordings —

    microphone ──► mic gain ──► [raw tap]  ──────────────────────► raw.wav
                                    │
                                    └► WebRTC APM (AEC/NS/AGC) ──► processed.wav

— which are then transcribed with the ordinary `ASREngine` adapter. Reading
the result:

- raw already wrong  → the loss is outside EVA (microphone, driver, room,
  the device's own enhancements, or the 16 kHz capture rate)
- raw right, processed wrong → EVA's front end (`AudioSettings`) is the cause
- both right, live conversation still wrong → segmentation or runtime
  behavior; the segmentation block of the report is the next place to look

The probe deliberately changes nothing: it reads the active `Settings` and
uses the same processor, VAD, segmenter, and ASR engine the assistant runs
with, so a measurement here describes the shipping configuration.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import platform
import subprocess
import sys
import time
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from eva.asr.base import ASREngine
from eva.audio.chunker import FrameChunker
from eva.audio.duplex import DuplexAudioStream
from eva.audio.frames import SAMPLE_RATE, Frame, int16_to_float
from eva.audio.playback import PlaybackQueue
from eva.audio.processor import create_processor
from eva.audio.ring import FrameRing
from eva.audio.segmenter import SegmenterEvent, SpeechSegmenter, UtteranceDiscarded, UtteranceEnd
from eva.config.settings import Settings

_IDLE_SLEEP_S = 0.002
_LOW_LEVEL_DBFS = -50.0  # below this, a recording is too quiet to conclude anything
_CLIP_THRESHOLD = 32_700  # int16 samples at/above this are treated as clipped
_FRICATIVE_HZ = 3_500.0  # /f/ /s/ /v/ energy lives above this


# ──────────────────────────── measurement ────────────────────────────


@dataclass(frozen=True)
class SignalMetrics:
    """Objective description of one recording."""

    duration_s: float
    sample_rate: int
    peak_dbfs: float
    rms_dbfs: float
    clipping_percent: float
    high_freq_energy_percent: float
    """Share of total energy above 3.5 kHz. Speech normally carries a few
    percent here; a near-zero value means the fricatives are gone, which is
    what turns "fox" into "box"."""


def signal_metrics(pcm: Frame, sample_rate: int = SAMPLE_RATE) -> SignalMetrics:
    """Level/quality metrics for one recording. Empty input reads as silence."""
    if pcm.size == 0:
        return SignalMetrics(0.0, sample_rate, -120.0, -120.0, 0.0, 0.0)
    samples = int16_to_float(pcm)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(pcm.size, 1.0 / sample_rate)
    total = float(np.sum(spectrum))
    high = float(np.sum(spectrum[freqs > _FRICATIVE_HZ])) if total > 0 else 0.0
    clipped = float(np.count_nonzero(np.abs(pcm) >= _CLIP_THRESHOLD))
    return SignalMetrics(
        duration_s=round(pcm.size / sample_rate, 3),
        sample_rate=sample_rate,
        peak_dbfs=round(20.0 * float(np.log10(max(peak, 1e-6))), 1),
        rms_dbfs=round(20.0 * float(np.log10(max(rms, 1e-6))), 1),
        clipping_percent=round(100.0 * clipped / pcm.size, 3),
        high_freq_energy_percent=round(100.0 * high / total, 3) if total > 0 else 0.0,
    )


def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation — the usual WER normalization."""
    kept = [c if c.isalnum() or c.isspace() else " " for c in text.lower().replace("'", "")]
    return "".join(kept).split()


def word_error_rate(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """(WER %, edit distance, reference word count). 0 reference words → 0 %."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return (0.0, 0, 0)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (0 if r == h else 1))
            )
        previous = current
    distance = previous[-1]
    return (round(100.0 * distance / len(ref), 1), distance, len(ref))


# ──────────────────────── provenance ────────────────────────


@dataclass(frozen=True)
class AsrConfig:
    """The decode configuration that was **actually used**.

    Captured by observing the real call into faster-whisper rather than
    restating the adapter's arguments here: a second copy of those defaults
    would drift the first time anyone edits `FasterWhisperASR`, and a report
    that lies about its own settings is worse than no report. Parameters the
    adapter leaves alone are filled in from the library's own signature
    defaults, so every field reflects the effective value.
    """

    engine: str
    model_id: str  # EVA catalogue id, e.g. "faster-whisper/small"
    model_name: str | None  # what was handed to WhisperModel, e.g. "small"
    device: str | None  # resolved backend: "cuda" | "cpu"
    compute_type_configured: str  # from settings ("auto" resolves at load)
    compute_type_resolved: str | None  # what CTranslate2 actually runs
    language: str | None
    beam_size: int | None
    best_of: int | None
    temperature: Any
    initial_prompt: str | None
    vad_filter: bool | None
    condition_on_previous_text: bool | None
    compression_ratio_threshold: float | None
    log_prob_threshold: float | None
    no_speech_threshold: float | None


@dataclass(frozen=True)
class Environment:
    """Everything needed to reproduce this measurement months later."""

    timestamp_utc: str
    eva_version: str
    git_commit: str | None
    git_dirty: bool | None
    backend: str
    gpu_name: str | None
    gpu_vram_mb: int | None
    cuda_device_count: int
    faster_whisper_version: str | None
    ctranslate2_version: str | None
    python_version: str
    platform: str


_TRACKED_ASR_ARGS = (
    "language",
    "beam_size",
    "best_of",
    "temperature",
    "initial_prompt",
    "vad_filter",
    "condition_on_previous_text",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
)


@contextlib.contextmanager
def _spy_on_transcribe(sink: dict[str, Any]) -> Iterator[None]:
    """Record the effective arguments of every `WhisperModel.transcribe` call.

    Wraps the library method for the duration of the probe and always restores
    it. Binding against the real signature and applying defaults means
    arguments the adapter never mentions (temperature, best_of, the
    thresholds) are still reported at their true values. A non-faster-whisper
    ASR engine simply leaves the sink empty.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:  # pragma: no cover - faster-whisper is a base dependency
        yield
        return

    original = WhisperModel.transcribe
    signature = inspect.signature(original)

    @functools.wraps(original)
    def spy(self: Any, audio: Any, *args: Any, **kwargs: Any) -> Any:
        with contextlib.suppress(TypeError):
            bound = signature.bind(self, audio, *args, **kwargs)
            bound.apply_defaults()
            sink.update({k: v for k, v in bound.arguments.items() if k not in ("self", "audio")})
        # The resolved compute type/device live on the CTranslate2 model, which
        # is the only place the "auto" settings value has been turned into a
        # concrete choice.
        model = getattr(self, "model", None)
        sink["_compute_type"] = getattr(model, "compute_type", None)
        sink["_device"] = getattr(model, "device", None)
        return original(self, audio, *args, **kwargs)

    WhisperModel.transcribe = spy
    try:
        yield
    finally:
        WhisperModel.transcribe = original


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value
    return repr(value)


def _asr_config(settings: Settings, asr: ASREngine, observed: dict[str, Any]) -> AsrConfig:
    seen = {key: _jsonable(observed.get(key)) for key in _TRACKED_ASR_ARGS}
    return AsrConfig(
        engine=settings.asr.engine,
        model_id=settings.asr.model,
        model_name=getattr(asr, "_model_name", None),
        device=observed.get("_device") or getattr(asr, "device", None),
        compute_type_configured=settings.asr.compute_type,
        compute_type_resolved=observed.get("_compute_type"),
        **seen,
    )


def _git_state() -> tuple[str | None, bool | None]:
    """(short commit, dirty). Both None outside a git checkout."""
    from eva.core.proc import no_window_kwargs

    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                **no_window_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "--short", "HEAD")
    if commit is None:
        return (None, None)
    status = git("status", "--porcelain")
    return (commit, bool(status) if status is not None else None)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment(backend: str) -> Environment:
    from eva import __version__

    gpu_name: str | None = None
    gpu_vram: int | None = None
    with contextlib.suppress(Exception):
        from eva.hardware import detect_hardware

        gpus = detect_hardware().gpus
        if gpus:
            gpu_name, gpu_vram = gpus[0].name, gpus[0].vram_total_mb

    cuda_devices = 0
    with contextlib.suppress(Exception):
        import ctranslate2

        cuda_devices = int(ctranslate2.get_cuda_device_count())

    commit, dirty = _git_state()
    return Environment(
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        eva_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        backend=backend,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        cuda_device_count=cuda_devices,
        faster_whisper_version=_package_version("faster-whisper"),
        ctranslate2_version=_package_version("ctranslate2"),
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({sys.platform})",
    )


# ──────────────────────────── report ────────────────────────────


@dataclass
class VariantReport:
    """One of the two recordings: what it looks like and what it decodes to."""

    name: str
    path: str
    metrics: SignalMetrics
    transcript: str
    decode_ms: int
    wer_percent: float | None = None
    wer_edits: int | None = None
    wer_ref_words: int | None = None


@dataclass
class SegmentationReport:
    """What the engine's endpointing made of the same audio (ADR-006)."""

    utterance_detected: bool
    discarded_as_noise: bool
    duration_ms: int
    speech_ms: int


@dataclass
class CaptureReport:
    recorded_at: str
    reference: str | None
    processor: str
    echo_cancellation: bool
    noise_suppression: bool
    auto_gain_control: bool
    mic_gain: float
    input_device: str | None
    device_native_sample_rate: float | None
    capture_ms: int
    frames_dropped: int
    segmentation: SegmentationReport
    asr: AsrConfig | None = None
    environment: Environment | None = None
    variants: list[VariantReport] = field(default_factory=list)

    @property
    def apm_enabled(self) -> bool:
        return self.processor != "passthrough"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "",
            "Capture probe report",
            "====================",
            f"Recorded at:       {self.recorded_at}",
            f"Reference:         {self.reference or '(none supplied)'}",
            "",
            "Front end",
            "---------",
            f"Processor:         {self.processor}  (APM {'ON' if self.apm_enabled else 'OFF'})",
            f"  echo cancel:     {self.echo_cancellation}",
            f"  noise suppress:  {self.noise_suppression}",
            f"  auto gain:       {self.auto_gain_control}",
            f"Mic gain:          {self.mic_gain}",
            f"Input device:      {self.input_device or 'system default'}",
            f"Device native SR:  {self.device_native_sample_rate or 'unknown'} Hz"
            f"  (captured at {SAMPLE_RATE} Hz)",
            f"Capture time:      {self.capture_ms} ms",
            f"Frames dropped:    {self.frames_dropped}",
            *self._asr_lines(),
            *self._environment_lines(),
            "",
            "Segmentation (what the engine would have done)",
            "----------------------------------------------",
            f"Utterance found:   {self.segmentation.utterance_detected}",
            f"Discarded as noise:{self.segmentation.discarded_as_noise}",
            f"Duration:          {self.segmentation.duration_ms} ms"
            f"  (speech {self.segmentation.speech_ms} ms)",
            "",
            "Recordings",
            "----------",
        ]
        for v in self.variants:
            m = v.metrics
            lines += [
                f"[{v.name}]  {v.path}",
                f"  duration        {m.duration_s:.2f} s @ {m.sample_rate} Hz",
                f"  peak / rms      {m.peak_dbfs:.1f} / {m.rms_dbfs:.1f} dBFS",
                f"  clipping        {m.clipping_percent:.3f} %",
                f"  energy >3.5 kHz {m.high_freq_energy_percent:.3f} %",
                f"  decode          {v.decode_ms} ms",
                f"  transcript      {v.transcript!r}",
            ]
            if v.wer_percent is not None:
                lines.append(
                    f"  WER             {v.wer_percent:.1f} %"
                    f"  ({v.wer_edits}/{v.wer_ref_words} words)"
                )
            lines.append("")
        lines += ["Interpretation", "--------------", *self._verdict()]
        return "\n".join(lines)

    def _asr_lines(self) -> list[str]:
        a = self.asr
        if a is None:
            return []
        compute = a.compute_type_resolved or "?"
        if a.compute_type_configured != compute:
            compute = f"{compute}  (settings: {a.compute_type_configured})"
        return [
            "",
            "ASR configuration (as actually called)",
            "--------------------------------------",
            f"Engine / model:    {a.engine} / {a.model_id}"
            f"{'' if a.model_name is None else f'  [{a.model_name}]'}",
            f"Device:            {a.device or 'unknown'}",
            f"Compute type:      {compute}",
            f"Language:          {a.language}",
            f"beam_size:         {a.beam_size}",
            f"best_of:           {a.best_of}",
            f"temperature:       {a.temperature}",
            f"initial_prompt:    {a.initial_prompt!r}",
            f"vad_filter:        {a.vad_filter}",
            f"condition_on_prev: {a.condition_on_previous_text}",
            f"compression_ratio_threshold: {a.compression_ratio_threshold}",
            f"log_prob_threshold:          {a.log_prob_threshold}",
            f"no_speech_threshold:         {a.no_speech_threshold}",
        ]

    def _environment_lines(self) -> list[str]:
        e = self.environment
        if e is None:
            return []
        commit = e.git_commit or "not a git checkout"
        if e.git_dirty:
            commit += " (dirty — uncommitted changes present)"
        return [
            "",
            "Environment",
            "-----------",
            f"Timestamp (UTC):   {e.timestamp_utc}",
            f"EVA version:       {e.eva_version}",
            f"Git commit:        {commit}",
            f"Backend:           {e.backend}",
            f"GPU:               {e.gpu_name or 'none detected'}"
            f"{'' if e.gpu_vram_mb is None else f' ({e.gpu_vram_mb} MB)'}"
            f", cuda devices: {e.cuda_device_count}",
            f"faster-whisper:    {e.faster_whisper_version}   ctranslate2: {e.ctranslate2_version}",
            f"Python / OS:       {e.python_version} on {e.platform}",
        ]

    def _verdict(self) -> list[str]:
        by_name = {v.name: v for v in self.variants}
        raw, processed = by_name.get("raw"), by_name.get("processed")
        if raw is None or processed is None:
            return ["  (incomplete run)"]
        if raw.metrics.rms_dbfs < _LOW_LEVEL_DBFS:
            return [
                f"  Raw level is only {raw.metrics.rms_dbfs:.1f} dBFS — too quiet to conclude",
                "  anything. Check the microphone level before trusting this run.",
            ]
        if self.reference is None:
            return ["  No reference text supplied — compare the two transcripts by eye."]
        raw_wer, proc_wer = raw.wer_percent or 0.0, processed.wer_percent or 0.0
        if raw_wer > 0 and proc_wer > 0:
            return [
                "  BOTH recordings decode incorrectly → the loss is OUTSIDE EVA's front end",
                "  (microphone, driver enhancements, room, or the capture sample rate).",
            ]
        if raw_wer == 0 and proc_wer > 0:
            return [
                "  Raw is correct, processed is wrong → EVA's FRONT END is responsible",
                "  (WebRTC AEC/NS/AGC in AudioSettings).",
            ]
        if raw_wer > 0 and proc_wer == 0:
            return ["  Processed is correct, raw is wrong → the front end is HELPING here."]
        return [
            "  Both recordings decode correctly. If live conversation still fails, the",
            "  cause is segmentation or runtime behavior, not the captured signal.",
        ]


# ──────────────────────────── capture ────────────────────────────


def _write_wav(path: Path, pcm: Frame, sample_rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _device_native_rate(input_device: str | int | None) -> float | None:
    try:
        import sounddevice as sd

        info = sd.query_devices(input_device, kind="input")
        return float(info["default_samplerate"])
    except Exception:  # diagnostics must never fail on a probe
        return None


def _segment(settings: Settings, processed: Frame) -> SegmentationReport:
    """Replay the engine's endpointing over the processed recording."""
    from eva.vad.registry import create_vad

    vad = create_vad(settings.vad.engine)
    segmenter = SpeechSegmenter(settings.vad)
    chunker = FrameChunker(vad.chunk_samples)
    events: list[SegmenterEvent] = []
    for start in range(0, processed.size, vad.chunk_samples):
        block = processed[start : start + vad.chunk_samples]
        for chunk in chunker.push(block):
            events.extend(segmenter.feed(chunk, vad.process(chunk), False))
    end = next((e for e in events if isinstance(e, UtteranceEnd)), None)
    discarded = next((e for e in events if isinstance(e, UtteranceDiscarded)), None)
    return SegmentationReport(
        utterance_detected=end is not None,
        discarded_as_noise=discarded is not None,
        duration_ms=end.duration_ms if end else 0,
        speech_ms=end.speech_ms if end else (discarded.speech_ms if discarded else 0),
    )


def _record(settings: Settings, seconds: float) -> tuple[Frame, Frame, int, int]:
    """One pass of the real capture chain → (raw, processed, ms, dropped).

    Both rings are sized for the whole recording so nothing is dropped, and
    both are drained in step, which keeps the two streams sample-aligned:
    `DuplexAudioStream` pushes the raw and the processed copy of every frame
    in the same callback tick.
    """
    capacity = int(seconds * 100) + 200  # 10 ms frames, generous headroom
    processor = create_processor(settings.audio)
    clean_ring, raw_ring = FrameRing(capacity), FrameRing(capacity)
    stream = DuplexAudioStream(
        processor,
        PlaybackQueue(),  # stays empty: the probe never plays anything
        clean_ring,
        input_device=settings.audio.input_device,
        output_device=settings.audio.output_device,
        mic_gain=settings.audio.mic_gain,
        raw_tap=raw_ring,
    )
    raw_frames: list[Frame] = []
    clean_frames: list[Frame] = []
    started = time.perf_counter()
    stream.start()
    try:
        while time.perf_counter() - started < seconds:
            raw, clean = raw_ring.pop(), clean_ring.pop()
            if raw is not None:
                raw_frames.append(raw)
            if clean is not None:
                clean_frames.append(clean)
            if raw is None and clean is None:
                time.sleep(_IDLE_SLEEP_S)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
    # Drain whatever the callback pushed after the loop exited.
    while (frame := raw_ring.pop()) is not None:
        raw_frames.append(frame)
    while (frame := clean_ring.pop()) is not None:
        clean_frames.append(frame)
    capture_ms = int((time.perf_counter() - started) * 1000)
    count = min(len(raw_frames), len(clean_frames))  # trim to keep them aligned
    empty = np.zeros(0, dtype=np.int16)
    raw_pcm: Frame = np.concatenate(raw_frames[:count]) if count else empty
    clean_pcm: Frame = np.concatenate(clean_frames[:count]) if count else empty
    return raw_pcm, clean_pcm, capture_ms, raw_ring.dropped + clean_ring.dropped


def _decode(asr: ASREngine, pcm: Frame, language: str, prompt: str | None) -> tuple[str, int]:
    started = time.perf_counter()
    result = asr.transcribe(pcm, language, prompt=prompt)
    return result.text.strip(), int((time.perf_counter() - started) * 1000)


def run_capture_test(
    settings: Settings,
    paths: Any,
    *,
    reference: str | None = None,
    seconds: float = 6.0,
    out_dir: Path | None = None,
    prompt: str | None = None,
) -> int:
    """Record once, save raw + processed WAVs, decode both, print the report.

    `prompt` defaults to None rather than the assistant's own `initial_prompt`:
    this probe isolates the audio path, so the decode is kept free of the
    context bias (measured neutral on real speech in the M7.2 experiments).
    """
    from eva.asr.registry import create_asr
    from eva.conversation.language import effective_asr_language, resolve_language
    from eva.llm.llamacpp import _register_cuda_dll_paths

    # CTranslate2 needs the pip-installed CUDA runtime DLLs on PATH. In the
    # assistant that happens as a side effect of building the LLM, which always
    # precedes the ASR load; this probe never touches the LLM, so it has to do
    # it itself. Without this, `load()` still reports "cuda" and the failure
    # only surfaces later inside encode() as a missing cublas64_12.dll.
    _register_cuda_dll_paths()

    out_dir = out_dir or (Path(paths.logs_dir) / "capture-probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    print(f"Recording {seconds:.0f} s — say the phrase now (Ctrl+C to stop early).")
    raw_pcm, clean_pcm, capture_ms, dropped = _record(settings, seconds)
    if raw_pcm.size == 0:
        print("No audio captured — is the microphone available and unmuted?")
        return 1
    print(f"Captured {raw_pcm.size / SAMPLE_RATE:.2f} s. Loading ASR and decoding...")

    raw_path = out_dir / f"{stamp}_raw.wav"
    clean_path = out_dir / f"{stamp}_processed.wav"
    _write_wav(raw_path, raw_pcm)
    _write_wav(clean_path, clean_pcm)

    asr = create_asr(settings, paths)
    asr.load()
    language = effective_asr_language(settings, resolve_language(settings))

    variants: list[VariantReport] = []
    observed: dict[str, Any] = {}
    with _spy_on_transcribe(observed):
        for name, pcm, path in (("raw", raw_pcm, raw_path), ("processed", clean_pcm, clean_path)):
            text, decode_ms = _decode(asr, pcm, language, prompt)
            variant = VariantReport(
                name=name,
                path=str(path),
                metrics=signal_metrics(pcm),
                transcript=text,
                decode_ms=decode_ms,
            )
            if reference:
                variant.wer_percent, variant.wer_edits, variant.wer_ref_words = word_error_rate(
                    reference, text
                )
            variants.append(variant)
    asr_config = _asr_config(settings, asr, observed)

    report = CaptureReport(
        recorded_at=stamp,
        reference=reference,
        processor=create_processor(settings.audio).name,
        echo_cancellation=settings.audio.echo_cancellation,
        noise_suppression=settings.audio.noise_suppression,
        auto_gain_control=settings.audio.auto_gain_control,
        mic_gain=settings.audio.mic_gain,
        input_device=settings.audio.input_device,
        device_native_sample_rate=_device_native_rate(settings.audio.input_device),
        capture_ms=capture_ms,
        frames_dropped=dropped,
        segmentation=_segment(settings, clean_pcm),
        asr=asr_config,
        environment=_environment(asr_config.device or "unknown"),
        variants=variants,
    )
    json_path = out_dir / f"{stamp}_report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print(f"Saved: {raw_path.name}, {clean_path.name}, {json_path.name}")
    print(f"       in {out_dir}")
    return 0
