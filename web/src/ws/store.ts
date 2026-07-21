/**
 * Live-event store (ADR-023): one zustand store fed by the WebSocket.
 *
 * The backend pushes a `snapshot` on connect and every engine event after
 * (ADR-017 — "clients never poll"). Reducers here enforce epoch discipline
 * (ADR-006): transcript/token events from a stale epoch are dropped, and a
 * new TurnStarted resets the live-turn state.
 */

import { create } from "zustand";
import type {
  ConversationTurn,
  PipelineState,
  RuntimeSnapshot,
  WsEnvelope,
} from "../api/types";

export interface LiveTurn {
  epoch: number;
  partialTranscript: string;
  finalTranscript: string | null;
  assistantText: string;
  cancelled: boolean;
  cancelReason: string | null;
  startedAt: number;
}

export interface TranscriptEntry {
  epoch: number;
  user: string;
  assistant: string;
  cancelled: boolean;
  cancelReason: string | null;
  at: number; // client arrival time (ms since epoch)
}

export interface DownloadProgress {
  modelId: string;
  filename: string;
  bytesDone: number;
  bytesTotal: number;
  status: "downloading" | "completed" | "failed";
  error?: string;
}

export interface EventLogEntry {
  at: number;
  type: string;
  summary: string;
}

export interface ComponentLoadState {
  component: string;
  label: string;
  done: boolean;
  ms: number | null;
  error: string | null;
}

const EVENT_LOG_LIMIT = 200;

/** Called when server-side state may have changed out from under the REST
 * cache: engine start/stop and WebSocket reconnect. App.tsx registers a
 * TanStack Query invalidator here — the store stays UI-library-agnostic. */
let onServerStateChanged: (() => void) | null = null;

export function registerServerStateListener(listener: (() => void) | null): void {
  onServerStateChanged = listener;
}

export interface WsState {
  connected: boolean;
  snapshot: RuntimeSnapshot | null;
  pipelineState: PipelineState;
  microphoneAvailable: boolean;
  microphoneMuted: boolean;
  liveTurn: LiveTurn | null;
  transcript: TranscriptEntry[];
  downloads: Record<string, DownloadProgress>;
  componentLoading: Record<string, ComponentLoadState>;
  eventLog: EventLogEntry[];
  lastError: string | null;

  setConnected: (connected: boolean) => void;
  handleMessage: (envelope: WsEnvelope) => void;
  seedTranscript: (turns: ConversationTurn[]) => void;
  clearTranscript: () => void;
  clearDownload: (modelId: string) => void;
  clearEventLog: () => void;
}

function summarize(type: string, data: Record<string, unknown>): string {
  switch (type) {
    case "PartialTranscript":
    case "FinalTranscript":
    case "LlmSentence":
      return String(data.text ?? "").slice(0, 60);
    case "StateChanged":
      return String(data.state ?? "");
    case "TtsAudioReady":
      return `ttfa ${data.ttfa_ms} ms`;
    case "TurnCancelled":
      return String(data.reason ?? "");
    case "ModelDownloadProgress": {
      const done = Number(data.bytes_done ?? 0);
      const total = Number(data.bytes_total ?? 0);
      return total ? `${Math.round((done / total) * 100)}%` : "";
    }
    case "ErrorOccurred":
      return String(data.message ?? "");
    default:
      return "";
  }
}

export const useWsStore = create<WsState>((set, get) => ({
  connected: false,
  snapshot: null,
  pipelineState: "idle",
  microphoneAvailable: false,
  microphoneMuted: false,
  liveTurn: null,
  transcript: [],
  downloads: {},
  componentLoading: {},
  eventLog: [],
  lastError: null,

  setConnected: (connected) => set({ connected }),

  seedTranscript: (turns) =>
    set({
      transcript: turns.map((t, i) => ({
        epoch: -1 - i, // history entries carry no epoch; keep keys unique
        user: t.user,
        assistant: t.assistant,
        cancelled: false,
        cancelReason: null,
        at: 0,
      })),
    }),

  clearTranscript: () => set({ transcript: [], liveTurn: null }),

  clearDownload: (modelId) =>
    set((s) => {
      const downloads = { ...s.downloads };
      delete downloads[modelId];
      return { downloads };
    }),

  clearEventLog: () => set({ eventLog: [] }),

  handleMessage: (envelope) => {
    const { type, data } = envelope;
    const now = Date.now();

    // Bounded event log for the diagnostics page — every event, LlmToken
    // excluded (hundreds per turn would drown everything else).
    if (type !== "LlmToken" && type !== "snapshot") {
      set((s) => ({
        eventLog: [...s.eventLog.slice(-(EVENT_LOG_LIMIT - 1)), { at: now, type, summary: summarize(type, data) }],
      }));
    }

    switch (type) {
      case "snapshot": {
        const snapshot = data as unknown as RuntimeSnapshot;
        const isReconnect = get().snapshot !== null;
        set({
          snapshot,
          pipelineState: snapshot.state,
          microphoneAvailable: snapshot.microphone_available,
          microphoneMuted: snapshot.microphone_muted,
        });
        // A snapshot after the first one means the socket reconnected —
        // anything cached from before the gap may be stale.
        if (isReconnect) onServerStateChanged?.();
        return;
      }
      case "StateChanged":
        set({ pipelineState: data.state as PipelineState });
        return;

      case "MicrophoneMuted":
        set({ microphoneMuted: data.muted as boolean });
        return;

      case "TurnStarted": {
        const epoch = data.epoch as number;
        set({
          liveTurn: {
            epoch,
            partialTranscript: "",
            finalTranscript: null,
            assistantText: "",
            cancelled: false,
            cancelReason: null,
            startedAt: now,
          },
        });
        return;
      }
      case "PartialTranscript": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return; // stale epoch: drop
        set({ liveTurn: { ...turn, partialTranscript: data.text as string } });
        return;
      }
      case "FinalTranscript": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return;
        set({ liveTurn: { ...turn, finalTranscript: data.text as string } });
        return;
      }
      case "LlmToken": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return;
        set({ liveTurn: { ...turn, assistantText: turn.assistantText + (data.token as string) } });
        return;
      }
      case "LlmFinished": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return;
        // Authoritative full text (tokens can be missed across reconnects).
        set({ liveTurn: { ...turn, assistantText: data.text as string } });
        return;
      }
      case "TurnCancelled": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return;
        set({ liveTurn: { ...turn, cancelled: true, cancelReason: data.reason as string } });
        return;
      }
      case "TurnFinished": {
        const turn = get().liveTurn;
        if (!turn || turn.epoch !== data.epoch) return;
        if (turn.finalTranscript || turn.assistantText) {
          set((s) => ({
            transcript: [
              ...s.transcript,
              {
                epoch: turn.epoch,
                user: turn.finalTranscript ?? turn.partialTranscript,
                assistant: turn.assistantText,
                cancelled: turn.cancelled,
                cancelReason: turn.cancelReason,
                at: now,
              },
            ],
            liveTurn: null,
          }));
        } else {
          set({ liveTurn: null });
        }
        return;
      }

      case "ModelDownloadProgress":
        set((s) => ({
          downloads: {
            ...s.downloads,
            [data.model_id as string]: {
              modelId: data.model_id as string,
              filename: data.filename as string,
              bytesDone: data.bytes_done as number,
              bytesTotal: data.bytes_total as number,
              status: "downloading",
            },
          },
        }));
        return;
      case "ModelDownloadCompleted":
        set((s) => {
          const existing = s.downloads[data.model_id as string];
          return {
            downloads: {
              ...s.downloads,
              [data.model_id as string]: {
                ...(existing ?? {
                  modelId: data.model_id as string,
                  filename: "",
                  bytesDone: 0,
                  bytesTotal: 0,
                }),
                status: "completed" as const,
              },
            },
          };
        });
        return;
      case "ModelDownloadFailed":
        set((s) => {
          const existing = s.downloads[data.model_id as string];
          return {
            downloads: {
              ...s.downloads,
              [data.model_id as string]: {
                ...(existing ?? {
                  modelId: data.model_id as string,
                  filename: "",
                  bytesDone: 0,
                  bytesTotal: 0,
                }),
                status: "failed" as const,
                error: data.error as string,
              },
            },
          };
        });
        return;

      case "ErrorOccurred":
        set({ lastError: data.message as string });
        return;

      case "ComponentLoadStarted":
        set((s) => ({
          componentLoading: {
            ...s.componentLoading,
            [data.component as string]: {
              component: data.component as string,
              label: data.label as string,
              done: false,
              ms: null,
              error: null,
            },
          },
        }));
        return;
      case "ComponentLoadFinished":
        set((s) => {
          const existing = s.componentLoading[data.component as string];
          return {
            componentLoading: {
              ...s.componentLoading,
              [data.component as string]: {
                component: data.component as string,
                label: existing?.label ?? (data.component as string),
                done: true,
                ms: data.ms as number,
                error: (data.error as string) || null,
              },
            },
          };
        });
        return;
      case "EngineStarted":
        // Fresh engine starts unmuted; clear the startup progress list.
        set({ componentLoading: {}, microphoneMuted: false });
        onServerStateChanged?.();
        return;
      case "EngineStopped":
        // Engine lifecycle changes invalidate REST caches (engine status,
        // models, voices, memory stats, ...) — pushed, not polled.
        set({ microphoneMuted: false, microphoneAvailable: false });
        onServerStateChanged?.();
        return;
    }
  },
}));
