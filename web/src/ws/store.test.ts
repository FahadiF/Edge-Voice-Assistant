import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerServerStateListener, useWsStore } from "./store";

function reset() {
  useWsStore.setState({
    connected: false,
    snapshot: null,
    pipelineState: "idle",
    liveTurn: null,
    transcript: [],
    downloads: {},
    eventLog: [],
    lastError: null,
  });
}

describe("WebSocket store reducers", () => {
  beforeEach(reset);

  it("applies the snapshot and pipeline state on connect", () => {
    // A partial snapshot: only the fields the reducer reads. The cast is
    // through Record (honest about being partial), not `as never`.
    const snapshot: Record<string, unknown> = { state: "listening", epoch: 3 };
    useWsStore.getState().handleMessage({ type: "snapshot", data: snapshot });
    expect(useWsStore.getState().snapshot).toEqual(snapshot);
    expect(useWsStore.getState().pipelineState).toBe("listening");
  });

  it("updates pipeline state on StateChanged", () => {
    useWsStore.getState().handleMessage({ type: "StateChanged", data: { state: "thinking" } });
    expect(useWsStore.getState().pipelineState).toBe("thinking");
  });

  it("starts a live turn on TurnStarted and streams partial/final transcript", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 5 } });
    s.handleMessage({ type: "PartialTranscript", data: { epoch: 5, text: "hel" } });
    expect(useWsStore.getState().liveTurn?.partialTranscript).toBe("hel");
    s.handleMessage({ type: "FinalTranscript", data: { epoch: 5, text: "hello" } });
    expect(useWsStore.getState().liveTurn?.finalTranscript).toBe("hello");
  });

  it("drops PartialTranscript events from a stale epoch (ADR-006)", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 5 } });
    s.handleMessage({ type: "PartialTranscript", data: { epoch: 4, text: "stale" } });
    expect(useWsStore.getState().liveTurn?.partialTranscript).toBe("");
  });

  it("drops events entirely when no turn is active", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "PartialTranscript", data: { epoch: 1, text: "x" } });
    expect(useWsStore.getState().liveTurn).toBeNull();
  });

  it("accumulates LlmToken text and LlmFinished is authoritative", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
    s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "Hel" } });
    s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "lo" } });
    expect(useWsStore.getState().liveTurn?.assistantText).toBe("Hello");
    s.handleMessage({
      type: "LlmFinished",
      data: { epoch: 1, text: "Hello there", tokens: 2, ttft_ms: 10, duration_ms: 20 },
    });
    expect(useWsStore.getState().liveTurn?.assistantText).toBe("Hello there");
  });

  it("moves a finished turn into the transcript and clears liveTurn", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 2 } });
    s.handleMessage({ type: "FinalTranscript", data: { epoch: 2, text: "hi" } });
    s.handleMessage({ type: "LlmToken", data: { epoch: 2, token: "hey" } });
    s.handleMessage({ type: "TurnFinished", data: { epoch: 2, error: null } });
    const state = useWsStore.getState();
    expect(state.liveTurn).toBeNull();
    expect(state.transcript).toHaveLength(1);
    expect(state.transcript[0]).toMatchObject({ user: "hi", assistant: "hey", cancelled: false });
  });

  it("marks a turn cancelled and records the reason", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 3 } });
    s.handleMessage({ type: "FinalTranscript", data: { epoch: 3, text: "hi" } });
    s.handleMessage({ type: "TurnCancelled", data: { epoch: 3, reason: "barge-in" } });
    expect(useWsStore.getState().liveTurn?.cancelled).toBe(true);
    expect(useWsStore.getState().liveTurn?.cancelReason).toBe("barge-in");
  });

  it("does not add an empty turn to the transcript on TurnFinished", () => {
    const s = useWsStore.getState();
    s.handleMessage({ type: "TurnStarted", data: { epoch: 9 } });
    s.handleMessage({ type: "TurnFinished", data: { epoch: 9, error: "no speech" } });
    expect(useWsStore.getState().transcript).toHaveLength(0);
  });

  it("tracks model download progress, completion, and failure", () => {
    const s = useWsStore.getState();
    s.handleMessage({
      type: "ModelDownloadProgress",
      data: { model_id: "m1", filename: "model.gguf", bytes_done: 50, bytes_total: 100 },
    });
    expect(useWsStore.getState().downloads.m1).toMatchObject({
      bytesDone: 50,
      bytesTotal: 100,
      status: "downloading",
    });
    s.handleMessage({ type: "ModelDownloadCompleted", data: { model_id: "m1" } });
    expect(useWsStore.getState().downloads.m1.status).toBe("completed");

    s.handleMessage({
      type: "ModelDownloadProgress",
      data: { model_id: "m2", filename: "x", bytes_done: 1, bytes_total: 10 },
    });
    s.handleMessage({ type: "ModelDownloadFailed", data: { model_id: "m2", error: "disk full" } });
    expect(useWsStore.getState().downloads.m2).toMatchObject({ status: "failed", error: "disk full" });
  });

  it("seeds the transcript from server history with unique keys", () => {
    useWsStore.getState().seedTranscript([
      { user: "hi", assistant: "hello" },
      { user: "bye", assistant: "goodbye" },
    ]);
    const transcript = useWsStore.getState().transcript;
    expect(transcript).toHaveLength(2);
    expect(new Set(transcript.map((t) => t.epoch)).size).toBe(2);
  });

  describe("server-state invalidation listener", () => {
    afterEach(() => registerServerStateListener(null));

    it("fires on EngineStarted and EngineStopped", () => {
      const listener = vi.fn();
      registerServerStateListener(listener);
      useWsStore.getState().handleMessage({ type: "EngineStarted", data: {} });
      useWsStore.getState().handleMessage({ type: "EngineStopped", data: {} });
      expect(listener).toHaveBeenCalledTimes(2);
    });

    it("fires on a re-connect snapshot but not the first snapshot", () => {
      const listener = vi.fn();
      registerServerStateListener(listener);
      useWsStore.getState().handleMessage({ type: "snapshot", data: { state: "idle", epoch: 0 } });
      expect(listener).not.toHaveBeenCalled(); // first connect: caches are fresh
      useWsStore.getState().handleMessage({ type: "snapshot", data: { state: "idle", epoch: 0 } });
      expect(listener).toHaveBeenCalledTimes(1); // reconnect: invalidate
    });
  });

  it("bounds the event log and excludes LlmToken spam", () => {
    const s = useWsStore.getState();
    for (let i = 0; i < 5; i++) {
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "x" } });
    }
    expect(useWsStore.getState().eventLog).toHaveLength(0);
    s.handleMessage({ type: "BargeInDetected", data: { epoch: 1 } });
    expect(useWsStore.getState().eventLog).toHaveLength(1);
  });
});
