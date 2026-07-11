/** App shell: sidebar navigation + header with live engine state. */

import { NavLink, Outlet } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { engine } from "../api/endpoints";
import { ApiError } from "../api/client";
import { useWsStore } from "../ws/store";
import { StatusPill, ToastHost, toast } from "./common";
import "./layout.css";
import "./composer.css"; // .mode-selector lives with the composer styles

const NAV = [
  { to: "/", label: "Dashboard" },
  { to: "/conversation", label: "Conversation" },
  { to: "/memory", label: "Memory" },
  { to: "/personas", label: "Personas" },
  { to: "/users", label: "User Profiles" },
  { to: "/models", label: "Models" },
  { to: "/voices", label: "Voices" },
  { to: "/settings", label: "Settings" },
  { to: "/diagnostics", label: "Diagnostics" },
  { to: "/plugins", label: "Plugins" },
];

function EngineControls() {
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: ["engine-status"],
    queryFn: engine.status,
    refetchInterval: 5000,
  });

  const start = useMutation({
    mutationFn: engine.start,
    onSuccess: () => {
      toast("success", "Engine started");
      queryClient.invalidateQueries({ queryKey: ["engine-status"] });
    },
    onError: (err) => {
      if (err instanceof ApiError && err.status === 409) {
        const detail = err.detail as { problems?: string[] };
        toast("error", `Setup incomplete: ${(detail.problems ?? []).join("; ")}`);
      } else {
        toast("error", `Engine start failed: ${err.message}`);
      }
    },
  });

  const stop = useMutation({
    mutationFn: engine.stop,
    onSuccess: () => {
      toast("info", "Engine stopped");
      queryClient.invalidateQueries({ queryKey: ["engine-status"] });
    },
    onError: (err) => toast("error", `Engine stop failed: ${err.message}`),
  });

  const running = status.data?.running ?? false;
  const componentLoading = useWsStore((s) => s.componentLoading);
  const loadingEntries = Object.values(componentLoading);
  const activeLoad = loadingEntries.find((c) => !c.done);
  const startLabel = start.isPending
    ? (activeLoad?.label ?? "Starting…")
    : "Start engine";
  return (
    <div className="engine-controls">
      <label className="mode-selector" title="Online providers are a future capability">
        Mode
        <select
          value="offline"
          aria-label="Conversation mode"
          onChange={() => {
            /* only Offline exists in this build */
          }}
        >
          <option value="offline">Offline (local)</option>
          <option value="online" disabled>
            Online (coming soon)
          </option>
        </select>
      </label>
      {running ? (
        <button onClick={() => stop.mutate()} disabled={stop.isPending}>
          {stop.isPending ? "Stopping…" : "Stop engine"}
        </button>
      ) : (
        <button
          className="primary"
          onClick={() => start.mutate()}
          disabled={start.isPending}
          aria-live="polite"
        >
          {startLabel}
        </button>
      )}
    </div>
  );
}

export function Layout() {
  const pipelineState = useWsStore((s) => s.pipelineState);
  const connected = useWsStore((s) => s.connected);

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ●
          </span>
          Edge Voice Assistant
        </div>
        <nav aria-label="Main">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="main">
        <header className="header">
          <StatusPill state={pipelineState} />
          <div className="header-right">
            <span
              className={`ws-indicator ${connected ? "ws-on" : "ws-off"}`}
              title={connected ? "Live connection active" : "Disconnected — reconnecting"}
            >
              {connected ? "● live" : "○ offline"}
            </span>
            <EngineControls />
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
      <ToastHost />
    </div>
  );
}
