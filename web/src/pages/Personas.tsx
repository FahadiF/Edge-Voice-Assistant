/**
 * Personas (Part 6): list/activate built-ins + custom, create/edit/
 * duplicate/delete custom ones, with a prompt preview.
 *
 * A persona counts as "custom" (deletable) when its id appears in
 * settings.conversation.custom_personas — built-ins are code (ADR-022) and
 * the API rejects deleting or overriding them.
 */

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { personas, settings as settingsApi } from "../api/endpoints";
import type { PersonaProfile, PersonaSettingsEntry } from "../api/types";
import { Card, ConfirmDialog, EmptyState, toast } from "../components/common";
import "./personas.css";

const EMPTY_FORM: PersonaSettingsEntry = {
  id: "",
  display_name: "",
  system_prompt: "",
  verbosity: "normal",
  tone: "neutral",
  reasoning_style: "direct",
  temperature_override: null,
};

function PersonaForm({
  initial,
  onCancel,
  onSaved,
}: {
  initial: PersonaSettingsEntry;
  onCancel: () => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState(initial);

  const save = useMutation({
    mutationFn: () => personas.create(form),
    onSuccess: () => {
      toast("success", `Persona '${form.id}' saved`);
      onSaved();
    },
    onError: (e) => toast("error", e.message),
  });

  return (
    <div className="persona-form">
      <label>
        Id (unique, lowercase-with-dashes)
        <input
          value={form.id}
          disabled={initial.id !== ""}
          onChange={(e) => setForm({ ...form, id: e.target.value })}
        />
      </label>
      <label>
        Display name
        <input
          value={form.display_name}
          onChange={(e) => setForm({ ...form, display_name: e.target.value })}
        />
      </label>
      <label>
        System prompt
        <textarea
          rows={4}
          value={form.system_prompt}
          onChange={(e) => setForm({ ...form, system_prompt: e.target.value })}
        />
      </label>
      <div className="persona-form-row">
        <label>
          Verbosity
          <select
            value={form.verbosity}
            onChange={(e) =>
              setForm({ ...form, verbosity: e.target.value as PersonaSettingsEntry["verbosity"] })
            }
          >
            {["minimal", "concise", "normal", "detailed"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label>
          Tone
          <input value={form.tone} onChange={(e) => setForm({ ...form, tone: e.target.value })} />
        </label>
        <label>
          Reasoning style
          <input
            value={form.reasoning_style}
            onChange={(e) => setForm({ ...form, reasoning_style: e.target.value })}
          />
        </label>
        <label>
          Temperature override
          <input
            type="number"
            min={0}
            max={2}
            step={0.1}
            value={form.temperature_override ?? ""}
            placeholder="inherit"
            onChange={(e) =>
              setForm({
                ...form,
                temperature_override: e.target.value === "" ? null : Number(e.target.value),
              })
            }
          />
        </label>
      </div>
      <div className="model-actions">
        <button
          className="primary"
          disabled={!form.id || !form.display_name || !form.system_prompt || save.isPending}
          onClick={() => save.mutate()}
        >
          Save
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

export function Personas() {
  const queryClient = useQueryClient();
  const list = useQuery({ queryKey: ["personas"], queryFn: personas.list });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: settingsApi.get });
  const activeId = settingsQuery.data?.conversation.persona;
  const customIds = useMemo(
    () => new Set((settingsQuery.data?.conversation.custom_personas ?? []).map((p) => p.id)),
    [settingsQuery.data],
  );

  const [editing, setEditing] = useState<PersonaSettingsEntry | null>(null);
  const [creating, setCreating] = useState(false);
  const [preview, setPreview] = useState<PersonaProfile | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: ["personas"] });
    queryClient.invalidateQueries({ queryKey: ["settings"] });
    setEditing(null);
    setCreating(false);
  };

  const activate = useMutation({
    mutationFn: (id: string) => settingsApi.patch({ conversation: { persona: id } }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["settings"], updated);
      toast("success", `Persona '${updated.conversation.persona}' activated`);
    },
    onError: (e) => toast("error", e.message),
  });

  const remove = useMutation({
    mutationFn: (id: string) => personas.remove(id),
    onSuccess: () => {
      setConfirmDelete(null);
      toast("success", "Persona deleted");
      invalidateAll();
    },
    onError: (e) => toast("error", e.message),
  });

  if (list.isLoading) return <p>Loading personas…</p>;

  return (
    <div>
      <h1>Personas</h1>
      <div className="grid-3">
        {(list.data ?? []).map((persona) => {
          const isCustom = customIds.has(persona.id);
          const isActive = persona.id === activeId;
          return (
            <Card key={persona.id} title={persona.display_name}>
              <p className="field-help">
                {isCustom ? "custom" : "built-in"} · {persona.verbosity} · {persona.tone}
                {isActive && <span className="chip chip-accent" style={{ marginLeft: 6 }}>active</span>}
              </p>
              <p className="persona-prompt-snippet">{persona.system_prompt}</p>
              <div className="model-actions">
                {!isActive && (
                  <button className="primary" onClick={() => activate.mutate(persona.id)}>
                    Activate
                  </button>
                )}
                <button onClick={() => setPreview(persona)}>Preview</button>
                {isCustom && (
                  <button
                    onClick={() =>
                      setEditing({
                        id: persona.id,
                        display_name: persona.display_name,
                        system_prompt: persona.system_prompt,
                        verbosity: persona.verbosity as PersonaSettingsEntry["verbosity"],
                        tone: persona.tone,
                        reasoning_style: persona.reasoning_style,
                        temperature_override: persona.temperature_override,
                      })
                    }
                  >
                    Edit
                  </button>
                )}
                <button
                  onClick={() =>
                    setEditing({
                      id: "",
                      display_name: `${persona.display_name} copy`,
                      system_prompt: persona.system_prompt,
                      verbosity: persona.verbosity as PersonaSettingsEntry["verbosity"],
                      tone: persona.tone,
                      reasoning_style: persona.reasoning_style,
                      temperature_override: persona.temperature_override,
                    })
                  }
                >
                  Duplicate
                </button>
                {isCustom && (
                  <button className="danger" onClick={() => setConfirmDelete(persona.id)}>
                    Delete
                  </button>
                )}
              </div>
            </Card>
          );
        })}
      </div>

      {!creating && !editing && (
        <button className="primary" style={{ marginTop: 16 }} onClick={() => setCreating(true)}>
          + New persona
        </button>
      )}

      {(creating || editing) && (
        <Card title={editing?.id ? `Edit '${editing.id}'` : "New persona"}>
          <PersonaForm
            initial={editing ?? EMPTY_FORM}
            onCancel={() => {
              setCreating(false);
              setEditing(null);
            }}
            onSaved={invalidateAll}
          />
        </Card>
      )}

      {preview && (
        <Card title={`Prompt preview — ${preview.display_name}`}>
          <p className="field-help">
            The identity and technical-facts blocks are always prepended at runtime
            (ADR-021) — this shows only this persona's own contribution. Use the Memory
            page's context inspector to see the full composed prompt.
          </p>
          <pre className="persona-prompt-full">{preview.system_prompt}</pre>
          <button onClick={() => setPreview(null)}>Close</button>
        </Card>
      )}

      {list.data?.length === 0 && <EmptyState>No personas registered.</EmptyState>}

      <ConfirmDialog
        open={confirmDelete !== null}
        title={`Delete persona '${confirmDelete}'?`}
        body="This cannot be undone. If it's the active persona, the default persona takes over."
        confirmLabel="Delete"
        danger
        onConfirm={() => confirmDelete && remove.mutate(confirmDelete)}
        onCancel={() => setConfirmDelete(null)}
      />
    </div>
  );
}
