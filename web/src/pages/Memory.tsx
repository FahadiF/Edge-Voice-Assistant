/**
 * Memory manager (Part 5): search, browse, curate, and inspect the
 * assistant's persistent memory. Requires a running engine (the memory
 * store lives on the assistant — ADR-019).
 */

import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { conversation as conversationApi, engine, memory } from "../api/endpoints";
import type { MemoryTurn } from "../api/types";
import { useWsStore } from "../ws/store";
import {
  Card,
  ConfirmDialog,
  EmptyState,
  downloadJson,
  formatBytes,
  toast,
} from "../components/common";
import "./memory.css";

function TurnActions({ turn, onChanged }: { turn: MemoryTurn; onChanged: () => void }) {
  const act = (fn: () => Promise<unknown>, label: string) => async () => {
    try {
      await fn();
      toast("success", label);
      onChanged();
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };
  if (turn.id === null) return null;
  const id = turn.id;
  return (
    <span className="turn-actions">
      <button
        title={turn.pinned ? "Unpin" : "Pin (boosts retrieval, survives cleanup)"}
        onClick={act(() => memory.pin(id, !turn.pinned), turn.pinned ? "Unpinned" : "Pinned")}
      >
        {turn.pinned ? "★ pinned" : "☆ pin"}
      </button>
      <button
        title={turn.favorite ? "Unfavorite" : "Favorite (boosts retrieval)"}
        onClick={act(
          () => memory.favorite(id, !turn.favorite),
          turn.favorite ? "Unfavorited" : "Favorited",
        )}
      >
        {turn.favorite ? "♥" : "♡"}
      </button>
      <button
        className="danger"
        title="Forget permanently"
        onClick={act(() => memory.forget(id), "Turn forgotten")}
      >
        forget
      </button>
    </span>
  );
}

function ContextPreview() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<Awaited<ReturnType<typeof memory.contextPreview>> | null>(
    null,
  );
  const run = async () => {
    if (!text.trim()) return;
    try {
      setResult(await memory.contextPreview(text));
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };
  return (
    <Card title="Context inspector">
      <p className="field-help">
        The developer's view into the assistant's "mind": type anything the user might
        say and see the exact prompt the language model would receive for it — which
        memories were retrieved (with relevance scores), whether a summary was
        included, the active persona and profile, and the final composed message
        list — all without spending a generation. If a remembered fact isn't being
        recalled in conversation, this is where you find out why.
      </p>
      <div className="context-input">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. what's my favorite color?"
          aria-label="Context preview input"
          onKeyDown={(e) => e.key === "Enter" && void run()}
        />
        <button className="primary" onClick={() => void run()}>
          Preview
        </button>
      </div>
      {result && (
        <div className="context-result">
          <h3>Trace</h3>
          <ul>
            <li>Persona: {result.trace.persona_id}</li>
            <li>Profile: {result.trace.profile_id ?? "none"}</li>
            <li>Language: {result.trace.language_code}</li>
            <li>Recent turns included: {result.trace.recent_turn_count}</li>
            <li>Summary included: {result.trace.summary_included ? "yes" : "no"}</li>
            {result.trace.trimmed_sections.length > 0 && (
              <li>Trimmed for budget: {result.trace.trimmed_sections.join(", ")}</li>
            )}
          </ul>
          {result.trace.retrieved_memories.length > 0 && (
            <>
              <h3>Retrieved memories</h3>
              <table>
                <thead>
                  <tr>
                    <th>Turn</th>
                    <th>Score</th>
                    <th>Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trace.retrieved_memories.map((m) => (
                    <tr key={m.turn_id}>
                      <td>#{m.turn_id}</td>
                      <td>{m.score.toFixed(3)}</td>
                      <td>{m.text_preview}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          <h3>Exact prompt messages</h3>
          <div className="context-messages">
            {result.messages.map((msg, i) => (
              <div key={i} className={`context-msg role-${msg.role}`}>
                <span className="chip">{msg.role}</span>
                <pre>{msg.content}</pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

export function Memory() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const clearTranscriptLocal = useWsStore((s) => s.clearTranscript);
  const status = useQuery({ queryKey: ["engine-status"], queryFn: engine.status });
  const running = status.data?.running ?? false;

  // Continue a stored conversation exactly where it ended (M5.6): switch the
  // engine's active conversation, drop the locally shown transcript (it
  // belongs to the previous conversation), and let the Conversation page
  // reseed itself from the freshly invalidated history query.
  const continueConversation = async (conversationId: string, title: string) => {
    try {
      const r = await conversationApi.resume(conversationId);
      clearTranscriptLocal();
      await queryClient.invalidateQueries({ queryKey: ["conversation-history"] });
      toast("success", `Continuing "${title || "untitled conversation"}" (${r.turns} turns)`);
      navigate("/conversation");
    } catch (e) {
      toast("error", `Continue failed: ${(e as Error).message}`);
    }
  };

  const stats = useQuery({ queryKey: ["memory-stats"], queryFn: memory.stats, enabled: running });
  const exportAll = useQuery({
    queryKey: ["memory-export"],
    queryFn: () => memory.exportJson(),
    enabled: running,
  });

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Awaited<ReturnType<typeof memory.search>> | null>(null);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const importFileRef = useRef<HTMLInputElement>(null);
  const [mergeSource, setMergeSource] = useState("");
  const [mergeTarget, setMergeTarget] = useState("");

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["memory-stats"] });
    queryClient.invalidateQueries({ queryKey: ["memory-export"] });
  };

  const runSearch = async () => {
    if (!query.trim()) return;
    try {
      setResults(await memory.search(query, 50));
    } catch (e) {
      toast("error", (e as Error).message);
    }
  };

  const commitRename = async (conversationId: string) => {
    const title = renameDraft.trim();
    setRenaming(null);
    if (!title) return;
    try {
      await memory.rename(conversationId, title);
      refresh();
      toast("success", "Conversation renamed");
    } catch (e) {
      toast("error", `Rename failed: ${(e as Error).message}`);
    }
  };

  const doImportConversation = async (file: File) => {
    try {
      const payload = JSON.parse(await file.text());
      const r = await memory.importJson(payload);
      refresh();
      toast("success", `Imported ${r.turns} turn(s)`);
    } catch (e) {
      toast("error", `Import failed: ${(e as Error).message}`);
    }
  };

  const deleteAll = useMutation({
    mutationFn: memory.deleteAll,
    onSuccess: () => {
      setConfirmDeleteAll(false);
      setResults(null);
      refresh();
      toast("success", "All memory deleted");
    },
    onError: (e) => toast("error", e.message),
  });

  if (!running) {
    return (
      <div>
        <h1>Memory</h1>
        <EmptyState>
          Memory lives on the running assistant — start the engine (header button) to browse
          and manage it.
        </EmptyState>
      </div>
    );
  }

  const conversations = exportAll.data?.conversations ?? [];

  return (
    <div>
      <h1>Memory</h1>
      <div className="grid-2">
        <Card title="Statistics">
          {stats.data ? (
            <table>
              <tbody>
                <tr><th scope="row">Conversations</th><td>{stats.data.conversation_count}</td></tr>
                <tr><th scope="row">Turns</th><td>{stats.data.turn_count}</td></tr>
                <tr><th scope="row">Embedded (searchable)</th><td>{stats.data.embedded_turn_count}</td></tr>
                <tr><th scope="row">Summaries</th><td>{stats.data.summary_count}</td></tr>
                <tr><th scope="row">Database size</th><td>{formatBytes(stats.data.db_size_bytes)}</td></tr>
                <tr><th scope="row">Full-text search</th><td>{stats.data.fts_enabled ? "FTS5" : "LIKE fallback"}</td></tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>Loading…</EmptyState>
          )}
          <div className="model-actions" style={{ marginTop: 10 }}>
            <button
              disabled={(stats.data?.turn_count ?? 0) === 0}
              title={
                (stats.data?.turn_count ?? 0) === 0
                  ? "Nothing to export yet — have a conversation first"
                  : undefined
              }
              onClick={async () => downloadJson(await memory.exportJson(), "eva-memory-export.json")}
            >
              Export all
            </button>
            <button
              className="danger"
              disabled={(stats.data?.turn_count ?? 0) === 0}
              onClick={() => setConfirmDeleteAll(true)}
            >
              Delete all memory
            </button>
          </div>
        </Card>

        <Card title="Search memories">
          <div className="context-input">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search everything I remember..."
              aria-label="Search memories"
              onKeyDown={(e) => e.key === "Enter" && void runSearch()}
            />
            <button className="primary" onClick={() => void runSearch()}>
              Search
            </button>
          </div>
          {results && results.length === 0 && <EmptyState>No matches.</EmptyState>}
          {results && results.length > 0 && (
            <ul className="search-results">
              {results.map((r) => (
                <li key={r.turn.id}>
                  <div className="search-result-head">
                    <span className="chip">{r.match_reason}</span>
                    <span className="chip">score {r.score.toFixed(2)}</span>
                    <span className="chip">{r.turn.speaker}</span>
                    <TurnActions turn={r.turn} onChanged={() => void runSearch()} />
                  </div>
                  <p>{r.turn.text}</p>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <ContextPreview />

      <Card
        title="Conversations"
        actions={
          <>
            <button onClick={() => importFileRef.current?.click()}>Import conversation</button>
            <input
              ref={importFileRef}
              type="file"
              accept="application/json"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void doImportConversation(f);
                e.target.value = "";
              }}
            />
          </>
        }
      >
        {conversations.length === 0 ? (
          <EmptyState>
            No stored conversations yet. Every conversation is saved here
            automatically — have one on the Conversation page (voice or typed) and it
            appears in this list with an auto-generated title, searchable and
            manageable. Each restart starts a fresh conversation; press Continue on
            any stored one to reopen it exactly where it ended.
          </EmptyState>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Title</th>
                <th>Started</th>
                <th>ID</th>
                <th>Turns</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {conversations.map((entry) => (
                <tr key={entry.conversation.id}>
                  <td>
                    {renaming === entry.conversation.id ? (
                      <input
                        autoFocus
                        value={renameDraft}
                        aria-label="Conversation title"
                        onChange={(e) => setRenameDraft(e.target.value)}
                        onBlur={() => void commitRename(entry.conversation.id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") void commitRename(entry.conversation.id);
                          if (e.key === "Escape") setRenaming(null);
                        }}
                      />
                    ) : (
                      <>
                        {entry.conversation.title || <em>Untitled</em>}{" "}
                        <button
                          className="link-button"
                          aria-label={`Rename ${entry.conversation.title || "untitled conversation"}`}
                          title="Rename"
                          onClick={() => {
                            setRenaming(entry.conversation.id);
                            setRenameDraft(entry.conversation.title);
                          }}
                        >
                          ✎
                        </button>
                      </>
                    )}
                  </td>
                  <td>{new Date(entry.conversation.started_at).toLocaleString()}</td>
                  <td>
                    <code>{entry.conversation.id.slice(0, 8)}…</code>
                    {entry.conversation.archived && <span className="chip">archived</span>}
                  </td>
                  <td>{entry.turns.length}</td>
                  <td className="turn-actions">
                    <button
                      className="primary"
                      title="Reopen this conversation where it ended — the next message continues it"
                      onClick={() =>
                        void continueConversation(
                          entry.conversation.id,
                          entry.conversation.title,
                        )
                      }
                    >
                      Continue
                    </button>
                    <button
                      title="Download this conversation as JSON"
                      onClick={async () =>
                        downloadJson(
                          await memory.exportJson(entry.conversation.id),
                          `eva-conversation-${(entry.conversation.title || entry.conversation.id.slice(0, 8)).replaceAll(/[^\w-]+/g, "-")}.json`,
                        )
                      }
                    >
                      Export
                    </button>
                    <button
                      onClick={async () => {
                        try {
                          const r = await memory.summarize(entry.conversation.id);
                          toast(
                            r.summary ? "success" : "info",
                            r.summary ?? "Nothing to summarize",
                          );
                          refresh();
                        } catch (e) {
                          toast("error", (e as Error).message);
                        }
                      }}
                    >
                      Summarize
                    </button>
                    <button
                      onClick={async () => {
                        await memory.archive(entry.conversation.id, !entry.conversation.archived);
                        refresh();
                      }}
                    >
                      {entry.conversation.archived ? "Restore" : "Archive"}
                    </button>
                    <button
                      className="danger"
                      onClick={async () => {
                        await memory.deleteConversation(entry.conversation.id);
                        toast("success", "Conversation deleted");
                        refresh();
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {conversations.length >= 2 && (
          <div className="merge-row">
            <span>Merge:</span>
            <select value={mergeSource} onChange={(e) => setMergeSource(e.target.value)} aria-label="Merge source">
              <option value="">source…</option>
              {conversations.map((c) => (
                <option key={c.conversation.id} value={c.conversation.id}>
                  {c.conversation.title || c.conversation.id.slice(0, 8)} ({c.turns.length} turns)
                </option>
              ))}
            </select>
            <span>into</span>
            <select value={mergeTarget} onChange={(e) => setMergeTarget(e.target.value)} aria-label="Merge target">
              <option value="">target…</option>
              {conversations
                .filter((c) => c.conversation.id !== mergeSource)
                .map((c) => (
                  <option key={c.conversation.id} value={c.conversation.id}>
                    {c.conversation.title || c.conversation.id.slice(0, 8)} ({c.turns.length} turns)
                  </option>
                ))}
            </select>
            <button
              disabled={!mergeSource || !mergeTarget}
              onClick={async () => {
                try {
                  await memory.merge(mergeSource, mergeTarget);
                  setMergeSource("");
                  setMergeTarget("");
                  refresh();
                  toast("success", "Conversations merged");
                } catch (e) {
                  toast("error", (e as Error).message);
                }
              }}
            >
              Merge
            </button>
          </div>
        )}
      </Card>

      <ConfirmDialog
        open={confirmDeleteAll}
        title="Delete ALL memory?"
        body="Every conversation, turn, embedding, and summary is permanently erased. User profiles are kept. This cannot be undone."
        confirmLabel="Delete everything"
        danger
        requireText="DELETE"
        onConfirm={() => deleteAll.mutate()}
        onCancel={() => setConfirmDeleteAll(false)}
      />
    </div>
  );
}
