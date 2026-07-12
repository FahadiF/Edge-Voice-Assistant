/**
 * ChatGPT-style message composer (M5.3).
 *
 * Typed messages go through POST /conversation/say — the same turn pipeline
 * as speech, minus ASR. The attachment surface (+ menu, drag-and-drop,
 * paste) is architecture-first: files become visible placeholder chips, but
 * uploads are not available in this build (image/document understanding is
 * a planned platform capability, ADR-025) — the chips say so instead of
 * silently dropping the file.
 */

import { useEffect, useRef, useState } from "react";
import type { DragEvent, ClipboardEvent, KeyboardEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { conversation, engine, settings } from "../api/endpoints";
import { useWsStore } from "../ws/store";
import { toast } from "./common";
import "./composer.css";

interface AttachmentChip {
  id: number;
  name: string;
  kind: "image" | "document";
}

let chipId = 0;

function chipFromFile(file: File): AttachmentChip {
  return {
    id: ++chipId,
    name: file.name || "pasted file",
    kind: file.type.startsWith("image/") ? "image" : "document",
  };
}

export function Composer({ engineRunning }: { engineRunning: boolean }) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<AttachmentChip[]>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const pipelineState = useWsStore((s) => s.pipelineState);
  const micMuted = useWsStore((s) => s.microphoneMuted);
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: settings.get });
  // The mic can be toggled only when the engine is running AND the user
  // granted microphone permission (otherwise EVA runs typed-chat-only,
  // ADR-025 — there is nothing to mute).
  const micPermission = settingsQuery.data?.permissions?.devices?.microphone ?? false;
  const micAvailable = engineRunning && micPermission;

  // The + menu dismisses like a real menu (M5.6): click anywhere outside
  // or press Escape — before this it only toggled via its own button.
  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onEscape = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onEscape);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onEscape);
    };
  }, [menuOpen]);

  const send = useMutation({
    mutationFn: (message: string) => conversation.say(message),
    onSuccess: (r) => {
      if (r.status !== "accepted") {
        toast("error", "The engine did not accept the message — is it still starting?");
        return;
      }
      setText("");
      if (attachments.length > 0) {
        toast("info", "Attachments are waiting for Vision support — sent the text only.");
        setAttachments([]);
      }
      textareaRef.current?.focus();
    },
    onError: (e) => toast("error", `Send failed: ${e.message}`),
  });

  const canSend = engineRunning && text.trim().length > 0 && !send.isPending;

  const submit = () => {
    if (canSend) send.mutate(text.trim());
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const addFiles = (files: FileList | File[]) => {
    const chips = Array.from(files).map(chipFromFile);
    if (chips.length > 0) setAttachments((prev) => [...prev, ...chips]);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  };

  const onPaste = (e: ClipboardEvent) => {
    const files = Array.from(e.clipboardData.files);
    if (files.length > 0) {
      e.preventDefault();
      addFiles(files);
    }
  };

  const placeholderAction = (label: string) => {
    setMenuOpen(false);
    toast("info", `${label} — waiting for Vision support (coming soon).`);
  };

  const queryClient = useQueryClient();
  const startEngine = useMutation({
    mutationFn: engine.start,
    onSuccess: () => {
      toast("success", "Engine started — you can talk now");
      queryClient.invalidateQueries({ queryKey: ["engine-status"] });
    },
    onError: (e) => toast("error", `Engine start failed: ${e.message}`),
  });
  const interrupt = useMutation({
    mutationFn: conversation.interrupt,
    onError: (e) => toast("error", e.message),
  });
  const toggleMic = useMutation({
    mutationFn: () => conversation.setMicrophone(!micMuted),
    onError: (e) => toast("error", e.message),
  });

  const onMicClick = () => {
    if (!engineRunning) {
      startEngine.mutate(); // stopped → tap to start the engine
    } else if (micAvailable) {
      toggleMic.mutate(); // running → tap to mute/unmute listening
    }
    // running without mic permission: button is disabled (nothing to mute).
    // Interrupting a reply lives on the dedicated ⏹ Stop button.
  };

  const micLabel = !engineRunning
    ? "Start the engine"
    : !micPermission
      ? "Microphone permission is off — enable it in Settings (typed chat only)"
      : micMuted
        ? "Microphone muted — tap to unmute"
        : "Microphone on — tap to mute";

  return (
    <div
      className={`composer ${dragOver ? "composer-dragover" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      {attachments.length > 0 && (
        <div className="composer-chips">
          {attachments.map((chip) => (
            <span key={chip.id} className="chip" title="Waiting for Vision support">
              {chip.kind === "image" ? "🖼" : "📄"} {chip.name}
              <button
                className="tag-remove"
                aria-label={`Remove ${chip.name}`}
                onClick={() => setAttachments((prev) => prev.filter((c) => c.id !== chip.id))}
              >
                ×
              </button>
            </span>
          ))}
          <span className="field-help">waiting for Vision support (coming soon)</span>
        </div>
      )}
      <div className="composer-row">
        <div className="composer-plus" ref={menuRef}>
          <button
            aria-label="Add attachment (attachments await Vision support)"
            aria-expanded={menuOpen}
            aria-haspopup="menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            +
          </button>
          {menuOpen && (
            <div className="composer-menu" role="menu">
              {/* Honest placeholders (M5.6): each entry says "coming soon"
                  up front instead of only after being clicked. */}
              <button role="menuitem" onClick={() => placeholderAction("Image upload")}>
                🖼 Attach image <span className="field-help">(coming soon)</span>
              </button>
              <button role="menuitem" onClick={() => placeholderAction("Document upload")}>
                📄 Attach document <span className="field-help">(coming soon)</span>
              </button>
              <button role="menuitem" onClick={() => placeholderAction("Screenshot capture")}>
                📷 Capture screenshot <span className="field-help">(coming soon)</span>
              </button>
            </div>
          )}
        </div>
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          placeholder={
            engineRunning
              ? "Type a message — or just speak. Enter to send, Shift+Enter for a new line."
              : "Start the engine to send messages."
          }
          aria-label="Message"
          disabled={!engineRunning}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
        />
        <button
          className={`composer-mic mic-${pipelineState}${micMuted ? " mic-muted" : ""}`}
          aria-label={micLabel}
          aria-pressed={micAvailable ? micMuted : undefined}
          title={micLabel}
          disabled={startEngine.isPending || toggleMic.isPending || (engineRunning && !micAvailable)}
          onClick={onMicClick}
        >
          {micMuted ? "🔇" : "🎙"}
        </button>
        {engineRunning && (pipelineState === "speaking" || pipelineState === "thinking") && (
          <button
            className="composer-interrupt"
            aria-label="Stop the current reply"
            title="Stop the current reply"
            onClick={() => interrupt.mutate()}
          >
            ⏹ Stop
          </button>
        )}
        <button
          className="primary composer-send"
          disabled={!canSend}
          aria-label="Send message"
          onClick={submit}
        >
          {send.isPending ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
