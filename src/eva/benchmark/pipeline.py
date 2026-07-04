"""End-to-end pipeline benchmark — no microphone required.

Uses TTS to synthesize a spoken question, runs it through ASR → LLM → TTS
exactly like a live turn, and reports per-stage timings. Because the input
audio is generated, results are reproducible and comparable across models,
quantizations, and hardware — the foundation the M7 benchmark reports build on.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel, ConfigDict

from eva.asr.base import ASREngine
from eva.conversation.chunker import SentenceChunker
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class StageTiming(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    duration_ms: int
    detail: str = ""


class PipelineReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    transcript: str
    reply: str
    stages: tuple[StageTiming, ...]
    ttft_ms: int
    ttfa_ms: int  # simulated: ASR + LLM-to-first-sentence + first-sentence TTS
    tokens: int
    tokens_per_s: float
    tts_rtf: float  # synthesis time / audio duration (lower is better)

    def render(self) -> str:
        lines = [
            f"Question:   {self.question}",
            f"Transcript: {self.transcript}",
            f"Reply:      {self.reply[:120]}{'…' if len(self.reply) > 120 else ''}",
            "",
            "Stage timings",
            "-------------",
        ]
        lines.extend(f"{s.name:<28} {s.duration_ms:>6} ms  {s.detail}" for s in self.stages)
        lines += [
            "",
            f"{'Time to first token':<28} {self.ttft_ms:>6} ms",
            f"{'Time to first audio (est.)':<28} {self.ttfa_ms:>6} ms",
            f"{'LLM speed':<28} {self.tokens_per_s:>6.1f} tok/s ({self.tokens} tokens)",
            f"{'TTS real-time factor':<28} {self.tts_rtf:>6.2f}",
        ]
        return "\n".join(lines)


class PipelineBenchmark:
    def __init__(
        self,
        asr: ASREngine,
        llm: LLMEngine,
        tts: TTSEngine,
        *,
        voice: str,
        system_prompt: str,
        params: GenerationParams | None = None,
    ) -> None:
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._voice = voice
        self._system_prompt = system_prompt
        self._params = params or GenerationParams()

    def run(self, question: str) -> PipelineReport:
        stages: list[StageTiming] = []

        def timed(name: str, detail: str = "") -> _StageTimer:
            return _StageTimer(name, detail, stages)

        # 1. Synthesize the spoken question (input generation, not measured as a
        #    pipeline stage — but reported for reference).
        with timed("TTS (question synthesis)"):
            question_audio = self._tts.synthesize(question, voice=self._voice)
        question_secs = question_audio.shape[0] / 16_000

        # 2. ASR
        with timed("ASR (transcription)", f"{question_secs:.1f}s of audio"):
            transcript = self._asr.transcribe(question_audio).text.strip()

        # 3. LLM streaming with sentence chunking (mirrors the live pipeline)
        chunker = SentenceChunker()
        messages = [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=transcript or question),
        ]
        tokens = 0
        ttft_ms = 0
        first_sentence: str | None = None
        first_sentence_ms = 0
        reply_parts: list[str] = []
        llm_start = time.perf_counter()
        for token in self._llm.stream(messages, self._params, should_abort=lambda: False):
            now_ms = int((time.perf_counter() - llm_start) * 1000)
            if ttft_ms == 0:
                ttft_ms = now_ms
            tokens += 1
            reply_parts.append(token)
            if first_sentence is None:
                for sentence in chunker.feed(token):
                    first_sentence = sentence
                    first_sentence_ms = now_ms
                    break
        llm_ms = int((time.perf_counter() - llm_start) * 1000)
        reply = "".join(reply_parts).strip()
        if first_sentence is None:
            first_sentence = chunker.flush() or reply
            first_sentence_ms = llm_ms
        stages.append(StageTiming(name="LLM (full generation)", duration_ms=llm_ms))
        stages.append(StageTiming(name="LLM (first sentence ready)", duration_ms=first_sentence_ms))

        # 4. TTS of the first reply sentence (the TTFA-critical synthesis)
        with timed("TTS (first reply sentence)", f'"{first_sentence[:40]}…"'):
            first_pcm = self._tts.synthesize(first_sentence, voice=self._voice)
        tts_first_ms = stages[-1].duration_ms
        first_audio_secs = max(first_pcm.shape[0] / 16_000, 1e-6)

        asr_ms = stages[1].duration_ms
        return PipelineReport(
            question=question,
            transcript=transcript,
            reply=reply,
            stages=tuple(stages),
            ttft_ms=asr_ms + ttft_ms,
            ttfa_ms=asr_ms + first_sentence_ms + tts_first_ms,
            tokens=tokens,
            tokens_per_s=tokens / (llm_ms / 1000) if llm_ms else 0.0,
            tts_rtf=(tts_first_ms / 1000) / first_audio_secs,
        )


class _StageTimer:
    def __init__(self, name: str, detail: str, sink: list[StageTiming]) -> None:
        self._name = name
        self._detail = detail
        self._sink = sink
        self._start = 0.0

    def __enter__(self) -> _StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._sink.append(
            StageTiming(
                name=self._name,
                duration_ms=int((time.perf_counter() - self._start) * 1000),
                detail=self._detail,
            )
        )
