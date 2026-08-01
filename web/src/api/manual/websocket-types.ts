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

/**
 * Fields every event payload carries, from `eva.core.events.Event`.
 *
 * `seq` is the bus's monotonic publication counter, stamped in
 * `EventBus.publish()`. Subscriber queues are bounded and silently drop their
 * oldest entry when full, so a gap in `seq` is the only signal a client gets
 * that it missed something — `ws/store.ts` watches for one and forces a
 * reconnect, which replays a fresh snapshot.
 */
export interface EventBase {
  seq: number;
}

export interface TurnStartedEvent extends EventBase { epoch: number }
export interface TurnFinishedEvent extends EventBase { epoch: number; error: string | null }
export interface TurnCancelledEvent extends EventBase {
  epoch: number;
  reason: "barge-in" | "superseded" | "shutdown" | "manual";
}
export interface SpeechStartedEvent extends EventBase { epoch: number }
export interface SpeechFinishedEvent extends EventBase { epoch: number; duration_ms: number }
export interface BargeInDetectedEvent extends EventBase { epoch: number }
export interface BargeInLatencyMeasuredEvent extends EventBase {
  epoch: number;
  detected_to_silent_ms: number;
}
export interface PartialTranscriptEvent extends EventBase { epoch: number; text: string }
export interface FinalTranscriptEvent extends EventBase {
  epoch: number;
  text: string;
  asr_ms: number;
}
export interface LlmStartedEvent extends EventBase { epoch: number }
export interface LlmTokenEvent extends EventBase { epoch: number; token: string }
export interface LlmSentenceEvent extends EventBase { epoch: number; text: string }
export interface LlmFinishedEvent extends EventBase {
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
export interface TtsStartedEvent extends EventBase { epoch: number }
export interface TtsAudioReadyEvent extends EventBase { epoch: number; ttfa_ms: number }
export interface TtsFinishedEvent extends EventBase { epoch: number }
export interface StateChangedEvent extends EventBase { state: PipelineState }
export interface ModelDownloadProgressEvent extends EventBase {
  model_id: string;
  filename: string;
  bytes_done: number;
  bytes_total: number;
}
export interface ModelDownloadCompletedEvent extends EventBase { model_id: string }
export interface ModelDownloadFailedEvent extends EventBase { model_id: string; error: string }
export interface ErrorOccurredEvent extends EventBase { message: string; context: string }
export interface ComponentLoadStartedEvent extends EventBase { component: string; label: string }
export interface ComponentLoadFinishedEvent extends EventBase {
  component: string;
  ms: number;
  error: string;
}
