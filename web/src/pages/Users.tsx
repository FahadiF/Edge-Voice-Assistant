/**
 * User Profiles (Part 7): create/switch/edit/delete, plus client-side
 * import/export — there's no dedicated import/export endpoint (per the
 * API inventory), so export downloads GET /users as JSON and import
 * POSTs each entry; this is a UI convenience over existing CRUD, not new
 * backend surface.
 */

import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { engine, models, users, voices } from "../api/endpoints";
import type { UserProfile } from "../api/types";
import { Card, ConfirmDialog, EmptyState, toast } from "../components/common";
import "./personas.css";

interface UserFormState {
  id?: string;
  nickname: string;
  preferred_language: string;
  preferred_voice: string;
  preferred_llm_model: string;
  conversation_style: string;
  units: "metric" | "imperial";
  timezone: string;
}

const EMPTY_FORM: UserFormState = {
  nickname: "",
  preferred_language: "",
  preferred_voice: "",
  preferred_llm_model: "",
  conversation_style: "",
  units: "metric",
  timezone: "UTC",
};

function UserForm({
  initial,
  onCancel,
  onSaved,
}: {
  initial: UserFormState;
  onCancel: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState(initial);
  const status = useQuery({ queryKey: ["engine-status"], queryFn: engine.status });
  const running = status.data?.running ?? false;
  const voiceList = useQuery({
    queryKey: ["voices"],
    queryFn: voices.list,
    enabled: running,
    retry: false,
  });
  const llmList = useQuery({ queryKey: ["models", "llm"], queryFn: () => models.list("llm") });

  const clean = (form: UserFormState) => ({
    nickname: form.nickname,
    preferred_language: form.preferred_language || null,
    preferred_voice: form.preferred_voice || null,
    preferred_llm_model: form.preferred_llm_model || null,
    conversation_style: form.conversation_style,
    units: form.units,
    timezone: form.timezone,
  });

  const save = useMutation({
    mutationFn: () =>
      form.id ? users.update(form.id, clean(form)) : users.create(clean(form)),
    onSuccess: () => {
      toast("success", form.id ? "Profile updated" : "Profile created");
      onSaved();
    },
    onError: (e) => toast("error", e.message),
  });

  return (
    <div className="user-form">
      <label>
        Nickname
        <input value={form.nickname} onChange={(e) => setForm({ ...form, nickname: e.target.value })} />
      </label>
      <label>
        Preferred language (code)
        <input
          value={form.preferred_language}
          placeholder="e.g. en, fi"
          onChange={(e) => setForm({ ...form, preferred_language: e.target.value })}
        />
      </label>
      <label>
        Preferred voice
        <select
          value={form.preferred_voice}
          onChange={(e) => setForm({ ...form, preferred_voice: e.target.value })}
        >
          <option value="">(none)</option>
          {(voiceList.data ?? []).map((v) => (
            <option key={v.id} value={v.id}>
              {v.display_name}
            </option>
          ))}
        </select>
        {!running && <span className="field-help">Start the engine to pick from real voices</span>}
      </label>
      <label>
        Preferred LLM
        <select
          value={form.preferred_llm_model}
          onChange={(e) => setForm({ ...form, preferred_llm_model: e.target.value })}
        >
          <option value="">(none)</option>
          {(llmList.data ?? []).map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Conversation style
        <input
          value={form.conversation_style}
          placeholder="e.g. casual, formal"
          onChange={(e) => setForm({ ...form, conversation_style: e.target.value })}
        />
      </label>
      <label>
        Units
        <select
          value={form.units}
          onChange={(e) => setForm({ ...form, units: e.target.value as UserFormState["units"] })}
        >
          <option value="metric">metric</option>
          <option value="imperial">imperial</option>
        </select>
      </label>
      <label>
        Timezone
        <input value={form.timezone} onChange={(e) => setForm({ ...form, timezone: e.target.value })} />
      </label>
      <div className="model-actions">
        <button className="primary" disabled={save.isPending} onClick={() => save.mutate()}>
          Save
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

export function Users() {
  const queryClient = useQueryClient();
  const status = useQuery({ queryKey: ["engine-status"], queryFn: engine.status });
  const running = status.data?.running ?? false;
  const list = useQuery({ queryKey: ["users"], queryFn: users.list, enabled: running });

  const [editing, setEditing] = useState<UserFormState | null>(null);
  const [creating, setCreating] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["users"] });
    setEditing(null);
    setCreating(false);
  };

  const activate = useMutation({
    mutationFn: (id: string) => users.activate(id),
    onSuccess: () => {
      toast("success", "Profile activated");
      queryClient.invalidateQueries({ queryKey: ["users"] });
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e) => toast("error", e.message),
  });

  const remove = useMutation({
    mutationFn: (id: string) => users.remove(id),
    onSuccess: () => {
      setConfirmDelete(null);
      toast("success", "Profile deleted");
      invalidate();
    },
    onError: (e) => toast("error", e.message),
  });

  const doExport = async () => {
    const data = await users.list();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "eva-user-profiles.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const doImport = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text()) as UserProfile[];
      for (const profile of parsed) {
        await users.create({
          nickname: profile.nickname,
          preferred_language: profile.preferred_language,
          preferred_voice: profile.preferred_voice,
          preferred_llm_model: profile.preferred_llm_model,
          conversation_style: profile.conversation_style,
          units: profile.units,
          timezone: profile.timezone,
        });
      }
      invalidate();
      toast("success", `Imported ${parsed.length} profile(s)`);
    } catch (e) {
      toast("error", `Import failed: ${(e as Error).message}`);
    }
  };

  if (!running) {
    return (
      <div>
        <h1>User Profiles</h1>
        <EmptyState>
          User profiles live on the running assistant — start the engine (header button)
          to manage them.
        </EmptyState>
      </div>
    );
  }

  return (
    <div>
      <h1>User Profiles</h1>
      <Card
        actions={
          <>
            <button onClick={doExport}>Export</button>
            <button onClick={() => fileRef.current?.click()}>Import</button>
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
        {(list.data ?? []).length === 0 ? (
          <EmptyState>No user profiles yet.</EmptyState>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Nickname</th>
                <th>Language</th>
                <th>Voice</th>
                <th>Units</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).map((profile) => (
                <tr key={profile.id}>
                  <td>
                    {profile.nickname || <em>unnamed</em>}
                    {profile.active && <span className="chip chip-accent" style={{ marginLeft: 6 }}>active</span>}
                  </td>
                  <td>{profile.preferred_language ?? "—"}</td>
                  <td>{profile.preferred_voice ?? "—"}</td>
                  <td>{profile.units}</td>
                  <td className="turn-actions">
                    {!profile.active && (
                      <button onClick={() => activate.mutate(profile.id)}>Activate</button>
                    )}
                    <button
                      onClick={() =>
                        setEditing({
                          id: profile.id,
                          nickname: profile.nickname,
                          preferred_language: profile.preferred_language ?? "",
                          preferred_voice: profile.preferred_voice ?? "",
                          preferred_llm_model: profile.preferred_llm_model ?? "",
                          conversation_style: profile.conversation_style,
                          units: profile.units,
                          timezone: profile.timezone,
                        })
                      }
                    >
                      Edit
                    </button>
                    <button className="danger" onClick={() => setConfirmDelete(profile.id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {!creating && !editing && (
        <button className="primary" style={{ marginTop: 16 }} onClick={() => setCreating(true)}>
          + New profile
        </button>
      )}

      {(creating || editing) && (
        <Card title={editing ? "Edit profile" : "New profile"}>
          <UserForm
            initial={editing ?? EMPTY_FORM}
            onCancel={() => {
              setCreating(false);
              setEditing(null);
            }}
            onSaved={invalidate}
          />
        </Card>
      )}

      <ConfirmDialog
        open={confirmDelete !== null}
        title="Delete this profile?"
        body="This cannot be undone."
        confirmLabel="Delete"
        danger
        onConfirm={() => confirmDelete && remove.mutate(confirmDelete)}
        onCancel={() => setConfirmDelete(null)}
      />
    </div>
  );
}
