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

  describe("trailing display-only reveal (M7.3)", () => {
    // Measured from a real session: a spoken preamble followed by a fenced
    // code block left the code invisible for 13 seconds, because the cursor
    // only advances on TtsSentenceStarted and no marker ever fires for a
    // segment the speech filter drops. Revealing it cannot put prose ahead
    // of playback — it is never spoken.
    const PROSE = "Here is the file. ";
    const FENCE = "```";
    const CODE = [FENCE + "html", "<p>hi</p>", FENCE].join("\n");
    const FULL = PROSE + CODE;

    it("reveals a trailing code block once the last spoken sentence starts", () => {
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: FULL } });
      s.handleMessage({
        type: "LlmFinished",
        data: {
          epoch: 1,
          text: FULL,
          tokens: 9,
          ttft_ms: 5,
          duration_ms: 9,
          finish_reason: "stop",
          speakable_end: "Here is the file.".length,
        },
      });
      // Generation finished first; the prose has not been spoken yet.
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(0);
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 1, text: "Here is the file." },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(FULL.length);
    });

    it("reveals a code-only reply immediately, since no marker will ever fire", () => {
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({
        type: "LlmFinished",
        data: {
          epoch: 1,
          text: CODE,
          tokens: 9,
          ttft_ms: 5,
          duration_ms: 9,
          finish_reason: "stop",
          speakable_end: 0,
        },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(CODE.length);
    });

    it("still holds back unspoken prose (ADR-028 is not weakened)", () => {
      const s = useWsStore.getState();
      const text = "Hi there. All good.";
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: text } });
      s.handleMessage({
        type: "LlmFinished",
        data: {
          epoch: 1,
          text,
          tokens: 4,
          ttft_ms: 5,
          duration_ms: 9,
          finish_reason: "stop",
          speakable_end: text.length,
        },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(0);
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 1, text: "Hi there." },
      });
      // Only the first sentence — the second is generated but unheard.
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe("Hi there.".length);
    });

    it("records a length-truncated generation", () => {
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({
        type: "LlmFinished",
        data: {
          epoch: 1,
          text: "cut off",
          tokens: 2048,
          ttft_ms: 5,
          duration_ms: 9,
          finish_reason: "length",
          speakable_end: 7,
        },
      });
      expect(useWsStore.getState().liveTurn?.finishReason).toBe("length");
    });
  });

  describe("speech-synchronized reveal (M7.1, ADR-028)", () => {
    /** Streams two sentences, then reports them spoken one at a time. */
    function generateTwoSentences() {
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "Hi there. " } });
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "All good." } });
      return s;
    }

    it("reveals nothing until a sentence is actually spoken", () => {
      generateTwoSentences();
      const turn = useWsStore.getState().liveTurn!;
      expect(turn.assistantText).toBe("Hi there. All good."); // generated…
      expect(turn.spokenChars).toBe(0); // …but not yet heard
    });

    it("advances the cursor to the end of each spoken sentence", () => {
      const s = generateTwoSentences();
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 1, text: "Hi there." },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe("Hi there.".length);
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 2, text: "All good." },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe("Hi there. All good.".length);
    });

    it("keeps the cursor put when generation finishes ahead of speech", () => {
      const s = generateTwoSentences();
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 1, text: "Hi there." },
      });
      s.handleMessage({
        type: "LlmFinished",
        data: { epoch: 1, text: "Hi there. All good.", tokens: 2, ttft_ms: 5, duration_ms: 9 },
      });
      // LlmFinished must not reveal the rest — that is the defect being fixed.
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe("Hi there.".length);
    });

    it("reveals a skipped segment along with the next spoken one", () => {
      // A fenced code block is never spoken (the speech filter drops it), so
      // the cursor jumps past it when the sentence after it starts.
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({
        type: "LlmToken",
        data: { epoch: 1, token: "Run this.\n```\nputs 'hi'\n```\nDone." },
      });
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 3, text: "Done." },
      });
      const turn = useWsStore.getState().liveTurn!;
      expect(turn.assistantText.slice(0, turn.spokenChars)).toContain("```"); // code shown
      expect(turn.spokenChars).toBe(turn.assistantText.length);
    });

    it("drops TtsSentenceStarted from a stale epoch (ADR-006)", () => {
      const s = generateTwoSentences();
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 0, index: 1, text: "Hi there." },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(0);
    });

    it("still advances when the spoken text cannot be located", () => {
      // Tokens can be lost across a reconnect, so the sentence may not appear
      // in `assistantText` — the reveal must keep moving, bounded by what is
      // actually there.
      const s = useWsStore.getState();
      s.handleMessage({ type: "TurnStarted", data: { epoch: 1 } });
      s.handleMessage({ type: "LlmToken", data: { epoch: 1, token: "xy" } });
      s.handleMessage({
        type: "TtsSentenceStarted",
        data: { epoch: 1, index: 1, text: "not in the stream" },
      });
      expect(useWsStore.getState().liveTurn?.spokenChars).toBe(2);
    });
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
