/** Diagnostics event-log tooling tests (M6 polish): the log was previously
 * un-copyable / un-exportable — these lock the format function and the
 * Copy all / Export / Clear actions added to fix that. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Diagnostics, formatEventLog } from "./Diagnostics";
import { ToastHost } from "../components/common";
import { useWsStore } from "../ws/store";
import type { RuntimeSnapshot } from "../api/types";

const FAKE_SNAPSHOT_RESPONSE = {
  state: "idle",
  epoch: 0,
  playback_active: false,
  playback_queued_seconds: 0,
  input_level_dbfs: -60,
  pending_audio_events: 0,
  capture_ring_depth: 0,
  capture_frames_dropped: 0,
  token_queue_depth: 0,
  sentence_queue_depth: 0,
  resources: {
    cpu_percent: 10,
    ram_used_mb: 100,
    ram_total_mb: 1000,
    gpu_percent: null,
    vram_used_mb: null,
    vram_total_mb: null,
  },
  last_turn: null,
  barge_in_count: 0,
  last_barge_in_latency_ms: null,
};

vi.mock("../api/endpoints", () => ({
  diagnostics: { snapshot: vi.fn(() => Promise.resolve(FAKE_SNAPSHOT_RESPONSE)) },
  system: { hardware: vi.fn(() => Promise.resolve(null)) },
}));

describe("formatEventLog", () => {
  it("formats one line per entry with an ISO timestamp", () => {
    const text = formatEventLog([
      { at: Date.parse("2026-07-21T10:00:00Z"), type: "TurnStarted", summary: "" },
      { at: Date.parse("2026-07-21T10:00:01Z"), type: "FinalTranscript", summary: "hello" },
    ]);
    expect(text).toBe(
      "2026-07-21T10:00:00.000Z  TurnStarted\n" + "2026-07-21T10:00:01.000Z  FinalTranscript  hello",
    );
  });

  it("returns an empty string for an empty log", () => {
    expect(formatEventLog([])).toBe("");
  });
});

function seedSnapshot(): void {
  useWsStore.setState({
    snapshot: FAKE_SNAPSHOT_RESPONSE as unknown as RuntimeSnapshot,
    eventLog: [
      { at: Date.parse("2026-07-21T10:00:00Z"), type: "TurnStarted", summary: "" },
      { at: Date.parse("2026-07-21T10:00:01Z"), type: "FinalTranscript", summary: "hello there" },
    ],
  });
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <Diagnostics />
      <ToastHost />
    </QueryClientProvider>,
  );
}

describe("Diagnostics event log toolbar", () => {
  beforeEach(seedSnapshot);
  afterEach(() => {
    useWsStore.setState({ snapshot: null, eventLog: [] });
    vi.restoreAllMocks();
  });

  it("copies the formatted log to the clipboard", async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.assign(navigator, { clipboard: { writeText } });
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: "Copy all" }));
    await Promise.resolve();

    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("FinalTranscript  hello there"),
    );
  });

  it("exports the log as a downloaded file", () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const createUrl = vi.fn(() => "blob:fake");
    vi.stubGlobal("URL", { createObjectURL: createUrl, revokeObjectURL: vi.fn() });
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: "Export .txt" }));

    expect(clickSpy).toHaveBeenCalled();
    expect(createUrl).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("clears the event log", () => {
    renderPage();
    expect(screen.getByText("FinalTranscript")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear log" }));

    expect(useWsStore.getState().eventLog).toHaveLength(0);
    expect(screen.getByText("No events yet this session.")).toBeInTheDocument();
  });

  it("disables the toolbar actions when the log is empty", () => {
    useWsStore.setState({ eventLog: [] });
    renderPage();
    expect(screen.getByRole("button", { name: "Copy all" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Export .txt" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear log" })).toBeDisabled();
  });
});
