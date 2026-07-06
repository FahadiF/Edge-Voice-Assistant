/** Small shared building blocks used across pages. */

import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { PipelineState } from "../api/types";
import "./common.css";

export function Card({
  title,
  children,
  actions,
}: {
  title?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card-header">
          {title && <h2>{title}</h2>}
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

const STATE_LABEL: Record<PipelineState, string> = {
  idle: "Idle",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function StatusPill({ state }: { state: PipelineState }) {
  return (
    <span className={`status-pill status-${state}`} role="status" aria-live="polite">
      <span className="status-dot" aria-hidden="true" />
      {STATE_LABEL[state] ?? state}
    </span>
  );
}

export function ProgressBar({ value, max, label }: { value: number; max: number; label?: string }) {
  const pct = max > 0 ? Math.min(100, Math.round((value / max) * 100)) : 0;
  return (
    <div
      className="progress"
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div className="progress-fill" style={{ width: `${pct}%` }} />
      <span className="progress-label">{pct}%</span>
    </div>
  );
}

export function Meter({
  label,
  value,
  max,
  unit,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="meter-value">
          {Math.round(value)} / {Math.round(max)} {unit}
        </span>
      </div>
      <div className="meter-track" aria-hidden="true">
        <div
          className="meter-fill"
          style={{ width: `${pct}%`, background: pct > 90 ? "var(--danger)" : "var(--accent)" }}
        />
      </div>
    </div>
  );
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  danger = false,
  requireText,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  title: string;
  body: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  /** e.g. "DELETE" — user must type it to enable the confirm button. */
  requireText?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const [typed, setTyped] = useState("");

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      setTyped("");
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  return (
    <dialog ref={ref} onCancel={onCancel} aria-labelledby="confirm-title">
      <h2 id="confirm-title">{title}</h2>
      <div className="dialog-body">{body}</div>
      {requireText && (
        <label className="dialog-require">
          Type <code>{requireText}</code> to confirm:
          <input value={typed} onChange={(e) => setTyped(e.target.value)} autoFocus />
        </label>
      )}
      <div className="dialog-buttons">
        <button onClick={onCancel}>Cancel</button>
        <button
          className={danger ? "danger" : "primary"}
          disabled={requireText !== undefined && typed !== requireText}
          onClick={onConfirm}
        >
          {confirmLabel}
        </button>
      </div>
    </dialog>
  );
}

interface ToastItem {
  id: number;
  kind: "info" | "success" | "error";
  text: string;
}

let toastId = 0;
let pushToastFn: ((item: Omit<ToastItem, "id">) => void) | null = null;

export function toast(kind: ToastItem["kind"], text: string): void {
  pushToastFn?.({ kind, text });
}

export function ToastHost() {
  const [items, setItems] = useState<ToastItem[]>([]);

  useEffect(() => {
    pushToastFn = (item) => {
      const id = ++toastId;
      setItems((prev) => [...prev, { ...item, id }]);
      setTimeout(() => setItems((prev) => prev.filter((t) => t.id !== id)), 5000);
    };
    return () => {
      pushToastFn = null;
    };
  }, []);

  return (
    <div className="toast-host" aria-live="polite">
      {items.map((item) => (
        <div key={item.id} className={`toast toast-${item.kind}`}>
          {item.text}
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ children }: { children: ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

/** Download any JSON-serializable value as a file (shared by the export
 * buttons on Conversation/Memory/Users — one implementation, not three). */
export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
