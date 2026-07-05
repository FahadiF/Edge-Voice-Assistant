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
    reconnectTimer = setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
  };

  socket.onerror = () => {
    socket?.close();
  };
}

/** Idempotent: called once from App mount. */
export function startWebSocket(): void {
  if (started) return;
  started = true;
  connect();
}

/** For tests. */
export function stopWebSocket(): void {
  started = false;
  if (reconnectTimer) clearTimeout(reconnectTimer);
  socket?.close();
  socket = null;
}
