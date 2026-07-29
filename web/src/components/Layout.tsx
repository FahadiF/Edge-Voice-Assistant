/** App shell: sidebar navigation + header with live engine state. */

import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { engine } from "../api/endpoints";
import { ApiError } from "../api/client";
import { useWsStore } from "../ws/store";
import { StatusPill, ToastHost, toast } from "./common";
import { AboutModal } from "./AboutModal";
import "./layout.css";
import "./composer.css"; // .mode-selector lives with the composer styles

import {
  Home,
  MessageSquare,
  Brain,
  Bot,
  User,
  Box,
  Volume2,
  Activity,
  Puzzle,
  Settings,
  Info
} from "lucide-react";

// Icons are decorative (aria-hidden): the label is the accessible name. Order
// puts the things you use while talking first and Settings last, where
// configuration conventionally lives.
const NAV = [
  { to: "/", label: "Dashboard", icon: Home },
  { to: "/conversation", label: "Conversation", icon: MessageSquare },
  { to: "/memory", label: "Memory", icon: Brain },
  { to: "/personas", label: "Personas", icon: Bot },
  { to: "/users", label: "User Profiles", icon: User },
  { to: "/models", label: "Models", icon: Box },
  { to: "/voices", label: "Voices", icon: Volume2 },
  { to: "/diagnostics", label: "Diagnostics", icon: Activity },
  { to: "/plugins", label: "Plugins", icon: Puzzle },
  { to: "/settings", label: "Settings", icon: Settings },
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

export function EvaLogo({ size = 24 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 6h16" />
      <path d="M4 12h8" />
      <path d="M4 18h12" />
    </svg>
  );
}

export function Layout() {
  const pipelineState = useWsStore((s) => s.pipelineState);
  const connected = useWsStore((s) => s.connected);
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <EvaLogo size={20} />
          </span>
          <span className="brand-text">Edge Voice Assistant</span>
        </div>
        <nav aria-label="Main">
          {NAV.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
              >
                <span className="nav-icon" aria-hidden="true">
                  <Icon size={18} />
                </span>
                <span className="nav-label">{item.label}</span>
              </NavLink>
            );
          })}
          <button
            className="nav-link"
            style={{
              background: "none",
              border: "none",
              width: "100%",
              textAlign: "left",
              padding: "8px 10px",
              cursor: "pointer",
            }}
            onClick={() => setAboutOpen(true)}
          >
            <span className="nav-icon" aria-hidden="true">
              <Info size={18} />
            </span>
            <span className="nav-label">About</span>
          </button>
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
      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
