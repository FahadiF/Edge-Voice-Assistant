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

import { useRef, useState } from "react";
import type { DragEvent, ClipboardEvent, KeyboardEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { conversation, engine } from "../api/endpoints";
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
  const pipelineState = useWsStore((s) => s.pipelineState);

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

  const onMicClick = () => {
    if (!engineRunning) {
      startEngine.mutate();
    } else if (pipelineState === "speaking" || pipelineState === "thinking") {
      interrupt.mutate(); // tap the mic to cut the assistant off
    }
    // listening/idle while running: nothing to do — it's always listening
  };

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
        <div className="composer-plus">
          <button
            aria-label="Add attachment"
            aria-expanded={menuOpen}
            aria-haspopup="menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            +
          </button>
          {menuOpen && (
            <div className="composer-menu" role="menu">
              <button role="menuitem" onClick={() => placeholderAction("Image upload")}>
                🖼 Attach image
              </button>
              <button role="menuitem" onClick={() => placeholderAction("Document upload")}>
                📄 Attach document
              </button>
              <button role="menuitem" onClick={() => placeholderAction("Screenshot capture")}>
                📷 Capture screenshot
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
          className={`composer-mic mic-${pipelineState}`}
          aria-label={
            engineRunning
              ? pipelineState === "speaking" || pipelineState === "thinking"
                ? "Interrupt the assistant"
                : `Microphone active — assistant is ${pipelineState}`
              : "Start the engine"
          }
          title={
            engineRunning
              ? pipelineState === "speaking" || pipelineState === "thinking"
                ? "Tap to interrupt the assistant"
                : `Always listening (${pipelineState}) — just talk`
              : "Tap to start the engine and enable the microphone"
          }
          disabled={startEngine.isPending}
          onClick={onMicClick}
        >
          🎙
        </button>
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
