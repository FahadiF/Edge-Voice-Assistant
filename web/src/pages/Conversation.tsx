/** Conversation (Part 4): streaming transcript + history management. */

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { conversation } from "../api/endpoints";
import { useWsStore } from "../ws/store";
import type { TranscriptEntry } from "../ws/store";
import { Card, EmptyState, toast } from "../components/common";
import "./conversation.css";

function timestamp(at: number): string {
  if (!at) return "";
  return new Date(at).toLocaleTimeString();
}

function TurnBlock({ entry }: { entry: TranscriptEntry }) {
  return (
    <div className="turn">
      {entry.user && (
        <div className="bubble bubble-user">
          <div className="bubble-meta">
            You {entry.at ? `· ${timestamp(entry.at)}` : ""}
          </div>
          {entry.user}
        </div>
      )}
      {entry.assistant && (
        <div className="bubble bubble-assistant">
          <div className="bubble-meta">
            Assistant {entry.at ? `· ${timestamp(entry.at)}` : ""}
          </div>
          {entry.assistant}
          {entry.cancelled && (
            <div className="interrupted-marker">— interrupted ({entry.cancelReason}) —</div>
          )}
        </div>
      )}
    </div>
  );
}

export function Conversation() {
  const liveTurn = useWsStore((s) => s.liveTurn);
  const transcript = useWsStore((s) => s.transcript);
  const seedTranscript = useWsStore((s) => s.seedTranscript);
  const clearTranscriptLocal = useWsStore((s) => s.clearTranscript);
  const [search, setSearch] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const history = useQuery({ queryKey: ["conversation-history"], queryFn: conversation.history });

  // Seed the local transcript from server history once (live turns append after).
  const seeded = useRef(false);
  useEffect(() => {
    if (!seeded.current && history.data && transcript.length === 0) {
      seeded.current = true;
      seedTranscript(history.data);
    }
  }, [history.data, transcript.length, seedTranscript]);

  // Keep the newest turn in view.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [transcript.length, liveTurn?.assistantText, liveTurn?.partialTranscript]);

  const interrupt = useMutation({
    mutationFn: conversation.interrupt,
    onSuccess: (r) => toast(r.interrupted ? "success" : "info", r.interrupted ? "Turn interrupted" : "Nothing to interrupt"),
    onError: (e) => toast("error", e.message),
  });

  const clear = useMutation({
    mutationFn: conversation.clear,
    onSuccess: () => {
      clearTranscriptLocal();
      toast("success", "Conversation cleared");
    },
    onError: (e) => toast("error", e.message),
  });

  const doExport = async () => {
    try {
      const data = await conversation.exportJson();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `eva-conversation-${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      toast("error", `Export failed: ${(e as Error).message}`);
    }
  };

  const doImport = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text());
      const turns = Array.isArray(parsed) ? parsed : parsed.turns;
      if (!Array.isArray(turns)) throw new Error("File has no 'turns' array");
      await conversation.importJson(turns);
      seeded.current = false;
      clearTranscriptLocal();
      await history.refetch();
      toast("success", `Imported ${turns.length} turn(s)`);
    } catch (e) {
      toast("error", `Import failed: ${(e as Error).message}`);
    }
  };

  const query = search.trim().toLowerCase();
  const visible = query
    ? transcript.filter(
        (t) => t.user.toLowerCase().includes(query) || t.assistant.toLowerCase().includes(query),
      )
    : transcript;

  return (
    <div className="conversation-page">
      <h1>Conversation</h1>
      <Card
        actions={
          <>
            <input
              type="search"
              placeholder="Search this session…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label="Search conversation"
            />
            <button onClick={() => interrupt.mutate()}>Interrupt</button>
            <button onClick={doExport}>Export</button>
            <button onClick={() => fileRef.current?.click()}>Import</button>
            <button className="danger" onClick={() => clear.mutate()}>
              Clear
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="application/json"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void doImport(f);
                e.target.value = "";
              }}
            />
          </>
        }
      >
        <div className="transcript" ref={scrollRef}>
          {visible.length === 0 && !liveTurn && (
            <EmptyState>
              No conversation yet. Start the engine and speak — the live transcript appears
              here. Past sessions live in <Link to="/memory">Memory</Link>.
            </EmptyState>
          )}
          {visible.map((entry) => (
            <TurnBlock key={`${entry.epoch}-${entry.at}`} entry={entry} />
          ))}
          {liveTurn && !query && (
            <div className="turn turn-live">
              {(liveTurn.finalTranscript ?? liveTurn.partialTranscript) && (
                <div className="bubble bubble-user">
                  <div className="bubble-meta">You · now</div>
                  {liveTurn.finalTranscript ?? (
                    <em className="partial">{liveTurn.partialTranscript}</em>
                  )}
                </div>
              )}
              {liveTurn.assistantText && (
                <div className="bubble bubble-assistant">
                  <div className="bubble-meta">Assistant · now</div>
                  {liveTurn.assistantText}
                  <span className="cursor" aria-hidden="true">
                    ▌
                  </span>
                  {liveTurn.cancelled && (
                    <div className="interrupted-marker">
                      — interrupted ({liveTurn.cancelReason}) —
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
