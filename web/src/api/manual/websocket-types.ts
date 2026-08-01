/**
 * WebSocket event payloads (eva/core/events.py) — hand-maintained.
 *
 * Not generatable from OpenAPI: these travel over the `/ws` connection as
 * `{"type": event.name, "data": event.model_dump()}`, not as an HTTP
 * response, so FastAPI's `/openapi.json` has no knowledge of their shape at
 * all (verified — none of these names appear in `components.schemas`).
 * Kept separate from `manual/dict-response-types.ts`: this file's entries
 * are real, fully-schemed pydantic models on the backend, they are simply
 * invisible to OpenAPI because of the transport, not because they lack a
 * schema. A future non-OpenAPI generator reading `eva.core.events` directly
 * could replace this file; none exists yet.
 */

export interface WsEnvelope {
  type: string;
  data: Record<string, unknown>;
}

/** Not a named schema in OpenAPI — generation inlines this literal union
 * wherever it appears (`EngineStatusResponse.state`, `RuntimeSnapshot.state`).
 * Named here because several frontend files import it directly. */
export type PipelineState = "idle" | "listening" | "thinking" | "speaking";

export interface TurnStartedEvent { epoch: number }
export interface TurnFinishedEvent { epoch: number; error: string | null }
export interface TurnCancelledEvent {
  epoch: number;
  reason: "barge-in" | "superseded" | "shutdown" | "manual";
}
export interface SpeechStartedEvent { epoch: number }
export interface SpeechFinishedEvent { epoch: number; duration_ms: number }
export interface BargeInDetectedEvent { epoch: number }
export interface BargeInLatencyMeasuredEvent { epoch: number; detected_to_silent_ms: number }
export interface PartialTranscriptEvent { epoch: number; text: string }
export interface FinalTranscriptEvent { epoch: number; text: string; asr_ms: number }
export interface LlmStartedEvent { epoch: number }
export interface LlmTokenEvent { epoch: number; token: string }
export interface LlmSentenceEvent { epoch: number; text: string }
export interface LlmFinishedEvent {
  epoch: number;
  text: string;
  tokens: number;
  ttft_ms: number;
  duration_ms: number;
  /** "stop" | "length" | "abort" | "error" — `length` means `text` is cut off. */
  finish_reason: string;
  /** Offset after which nothing in `text` is ever spoken; -1 if not computed. */
  speakable_end: number;
}
export interface TtsStartedEvent { epoch: number }
export interface TtsAudioReadyEvent { epoch: number; ttfa_ms: number }
export interface TtsFinishedEvent { epoch: number }
export interface StateChangedEvent { state: PipelineState }
export interface ModelDownloadProgressEvent {
  model_id: string;
  filename: string;
  bytes_done: number;
  bytes_total: number;
}
export interface ModelDownloadCompletedEvent { model_id: string }
export interface ModelDownloadFailedEvent { model_id: string; error: string }
export interface ErrorOccurredEvent { message: string; context: string }
export interface ComponentLoadStartedEvent { component: string; label: string }
export interface ComponentLoadFinishedEvent { component: string; ms: number; error: string }
