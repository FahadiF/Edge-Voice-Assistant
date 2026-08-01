import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { reconnectNow, startWebSocket, stopWebSocket } from "./socket";

/**
 * A minimal WebSocket stand-in that records every instance and lets the test
 * drive readyState + fire events, so we can assert the restore-from-tray
 * reconnect (visibilitychange → visible reconnects a dropped socket instantly).
 */
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = FakeWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null;
  closed = false;

  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }
  drop() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
  close() {
    this.closed = true;
    this.readyState = FakeWebSocket.CLOSED;
  }
}

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { value: state, configurable: true });
  document.dispatchEvent(new Event("visibilitychange"));
}

describe("reconnecting WebSocket (restore-from-tray)", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    setVisibility("visible");
  });

  afterEach(() => {
    stopWebSocket();
    vi.unstubAllGlobals();
  });

  it("opens exactly one socket on start", () => {
    startWebSocket();
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("reconnects immediately when the window becomes visible with a dropped socket", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();
    FakeWebSocket.instances[0].drop(); // socket dies while hidden (renderer froze)

    setVisibility("hidden");
    setVisibility("visible"); // window restored from tray

    // A fresh socket was created at once — no waiting on the backoff timer.
    expect(FakeWebSocket.instances.length).toBeGreaterThanOrEqual(2);
    expect(FakeWebSocket.instances.at(-1)!.closed).toBe(false);
  });

  it("does not open a second socket when already connected on visibility", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();
    setVisibility("visible");
    expect(FakeWebSocket.instances).toHaveLength(1); // still the one open socket
  });

  it("ignores visibility changes after stop", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();
    stopWebSocket();
    const count = FakeWebSocket.instances.length;
    setVisibility("visible");
    expect(FakeWebSocket.instances).toHaveLength(count); // listener removed
  });
});

describe("gap-triggered reconnect", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
    setVisibility("visible");
  });

  afterEach(() => {
    stopWebSocket();
    vi.unstubAllGlobals();
  });

  it("reconnects an OPEN socket — the connection is fine, the stream is not", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();

    reconnectNow();

    // Unlike visibility restore, a healthy socket is deliberately replaced:
    // only a new connection replays the snapshot that closes the gap.
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(FakeWebSocket.instances[0].closed).toBe(true);
  });

  it("collapses a burst of gap reports into one reconnect", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();

    // Three gaps detected in the same tick, before any callback can run on
    // the socket the first reconnect creates — it is still CONNECTING.
    reconnectNow();
    reconnectNow();
    reconnectNow();

    expect(FakeWebSocket.instances).toHaveLength(2); // one replacement, not three
    expect(FakeWebSocket.instances.at(-1)!.readyState).toBe(FakeWebSocket.CONNECTING);
  });

  it("allows a later reconnect once the replacement socket is established", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();
    reconnectNow();
    FakeWebSocket.instances[1].open(); // no longer CONNECTING

    reconnectNow(); // a genuinely new gap, later on

    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("does nothing after stop", () => {
    startWebSocket();
    FakeWebSocket.instances[0].open();
    stopWebSocket();
    const count = FakeWebSocket.instances.length;

    reconnectNow();

    expect(FakeWebSocket.instances).toHaveLength(count);
  });
});
