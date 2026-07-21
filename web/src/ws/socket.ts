/**
 * Reconnecting WebSocket client for /api/v1/ws (ADR-017).
 *
 * One connection per app lifetime; exponential backoff on drop (the local
 * server may restart during model switches). Every message is the
 * `{type, data}` envelope, dispatched into the zustand store.
 */

import { useWsStore } from "./store";

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 10_000;

let socket: WebSocket | null = null;
let backoff = INITIAL_BACKOFF_MS;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let started = false;

function wsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/api/v1/ws`;
}

function connect(): void {
  socket = new WebSocket(wsUrl());

  socket.onopen = () => {
    backoff = INITIAL_BACKOFF_MS;
    useWsStore.getState().setConnected(true);
  };

  socket.onmessage = (event) => {
    try {
      const envelope = JSON.parse(event.data as string);
      if (envelope && typeof envelope.type === "string") {
        useWsStore.getState().handleMessage(envelope);
      }
    } catch {
      // Malformed frame — ignore; the next frame resyncs us.
    }
  };

  socket.onclose = () => {
    useWsStore.getState().setConnected(false);
    if (!started) return; // stopped deliberately — no reconnect
    reconnectTimer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
  };

  socket.onerror = () => {
    socket?.close();
  };
}

/**
 * When the page becomes visible again (window restored from the tray / tab
 * refocused), reconnect immediately if the socket isn't open. A renderer that
 * was frozen while hidden can have a dropped or half-open socket with a pending
 * backoff timer; on restore we want the live stream back at once, not after a
 * multi-second backoff. No-op when the socket is already open.
 */
function onVisibilityChange(): void {
  if (!started || document.visibilityState !== "visible") return;
  if (socket && socket.readyState === WebSocket.OPEN) return;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  backoff = INITIAL_BACKOFF_MS; // restore should feel instant
  if (socket) {
    socket.onclose = null; // don't let the old socket schedule a competing reconnect
    socket.onerror = null;
    socket.onmessage = null;
    try {
      socket.close();
    } catch {
      // already closing/closed
    }
    socket = null;
  }
  connect();
}

/** Idempotent: called once from App mount. */
export function startWebSocket(): void {
  if (started) return;
  started = true;
  document.addEventListener("visibilitychange", onVisibilityChange);
  connect();
}

/** Tear down fully (unmount/tests): detach handlers BEFORE closing so the
 * close event can't schedule a zombie reconnect. */
export function stopWebSocket(): void {
  started = false;
  backoff = INITIAL_BACKOFF_MS;
  document.removeEventListener("visibilitychange", onVisibilityChange);
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (socket) {
    socket.onclose = null;
    socket.onerror = null;
    socket.onmessage = null;
    socket.close();
    socket = null;
  }
}
