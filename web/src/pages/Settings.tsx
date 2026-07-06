/**
 * Settings (Part 10): fully schema-driven per ADR-009 — sections and fields
 * come from GET /settings/schema; values from GET /settings; saves are
 * validated server-side (POST /settings/validate) before PATCH.
 */

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { settings as settingsApi } from "../api/endpoints";
import type { Settings, ValidationErrorDetail } from "../api/types";
import type { JsonSchema } from "../components/SchemaForm";
import { SchemaSection } from "../components/SchemaForm";
import { Card, ConfirmDialog, toast } from "../components/common";
import { applyUiSettings } from "../theme/ThemeProvider";

const SECTION_LABELS: Record<string, string> = {
  audio: "Audio",
  vad: "Voice Detection",
  asr: "Speech Recognition",
  llm: "Language Model",
  tts: "Speech Synthesis",
  conversation: "Conversation",
  memory: "Memory & Privacy",
  permissions: "Permissions",
  server: "Server",
  ui: "Appearance",
  developer: "Developer",
};

export function SettingsPage() {
  const queryClient = useQueryClient();
  const schemaQuery = useQuery({ queryKey: ["settings-schema"], queryFn: settingsApi.schema });
  const valuesQuery = useQuery({ queryKey: ["settings"], queryFn: settingsApi.get });

  const [section, setSection] = useState("audio");
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [confirmReset, setConfirmReset] = useState(false);
  const [pendingSection, setPendingSection] = useState<string | null>(null);

  const schema = schemaQuery.data as
    | { properties?: Record<string, JsonSchema>; $defs?: Record<string, JsonSchema> }
    | undefined;

  const sectionNames = useMemo(() => {
    if (!schema?.properties) return [];
    return Object.entries(schema.properties)
      .filter(([, prop]) => prop.$ref)
      .map(([name]) => name);
  }, [schema]);

  const sectionSchema = useMemo((): JsonSchema | null => {
    if (!schema?.properties || !schema.$defs) return null;
    const ref = schema.properties[section]?.$ref;
    if (!ref) return null;
    const defName = ref.split("/").pop()!;
    return schema.$defs[defName] ?? null;
  }, [schema, section]);

  const saved = valuesQuery.data;
  const sectionValues =
    draft ?? ((saved?.[section as keyof Settings] ?? {}) as Record<string, unknown>);

  const save = useMutation({
    mutationFn: async (values: Record<string, unknown>) => {
      const patch = { [section]: values };
      const validation = await settingsApi.validate({
        ...(saved as unknown as Record<string, unknown>),
        ...patch,
      });
      if (!validation.valid) {
        const errors: Record<string, string> = {};
        for (const err of validation.errors as ValidationErrorDetail[]) {
          // loc is like ["conversation", "temperature"]
          const field = String(err.loc[err.loc.length - 1]);
          errors[field] = err.message;
        }
        setFieldErrors(errors);
        throw new Error("Validation failed — fix the highlighted fields");
      }
      return settingsApi.patch(patch);
    },
    onSuccess: (updated) => {
      setDraft(null);
      setFieldErrors({});
      queryClient.setQueryData(["settings"], updated);
      applyUiSettings(updated.ui);
      toast("success", "Settings saved");
    },
    onError: (e) => toast("error", e.message),
  });

  const reset = useMutation({
    mutationFn: settingsApi.reset,
    onSuccess: (updated) => {
      setDraft(null);
      setFieldErrors({});
      setConfirmReset(false);
      queryClient.setQueryData(["settings"], updated);
      applyUiSettings(updated.ui);
      toast("success", "Settings reset to defaults");
    },
    onError: (e) => toast("error", e.message),
  });

  if (schemaQuery.isLoading || valuesQuery.isLoading) {
    return <p>Loading settings…</p>;
  }
  if (!schema || !saved) {
    return <p role="alert">Could not load settings from the server.</p>;
  }

  const dirty = draft !== null;

  return (
    <div>
      <h1>Settings</h1>
      <Card>
        <div className="settings-layout">
          <nav className="settings-nav" aria-label="Settings sections">
            {sectionNames.map((name) => (
              <button
                key={name}
                className={name === section ? "active" : ""}
                onClick={() => {
                  if (dirty) {
                    setPendingSection(name);
                    return;
                  }
                  setDraft(null);
                  setFieldErrors({});
                  setSection(name);
                }}
              >
                {SECTION_LABELS[name] ?? name}
              </button>
            ))}
          </nav>
          <div>
            <h2>{SECTION_LABELS[section] ?? section}</h2>
            {sectionSchema && (
              <SchemaSection
                sectionSchema={sectionSchema}
                values={sectionValues}
                errors={fieldErrors}
                onFieldChange={(field, value) =>
                  setDraft({ ...(sectionValues as Record<string, unknown>), [field]: value })
                }
              />
            )}
            <div className="settings-toolbar">
              <button
                className="primary"
                disabled={!dirty || save.isPending}
                onClick={() => draft && save.mutate(draft)}
              >
                {save.isPending ? "Saving…" : "Save changes"}
              </button>
              <button disabled={!dirty} onClick={() => { setDraft(null); setFieldErrors({}); }}>
                Discard
              </button>
              <button className="danger" onClick={() => setConfirmReset(true)}>
                Reset all to defaults
              </button>
            </div>
            <p className="field-help" style={{ marginTop: 10 }}>
              Model/engine changes take effect the next time the engine starts.
            </p>
          </div>
        </div>
      </Card>
      <ConfirmDialog
        open={confirmReset}
        title="Reset all settings?"
        body="Every section returns to its default value. Installed models and memory are not affected."
        confirmLabel="Reset everything"
        danger
        onConfirm={() => reset.mutate()}
        onCancel={() => setConfirmReset(false)}
      />
      <ConfirmDialog
        open={pendingSection !== null}
        title="Discard unsaved changes?"
        body={`You have unsaved changes in ${SECTION_LABELS[section] ?? section}.`}
        confirmLabel="Discard changes"
        danger
        onConfirm={() => {
          setDraft(null);
          setFieldErrors({});
          if (pendingSection) setSection(pendingSection);
          setPendingSection(null);
        }}
        onCancel={() => setPendingSection(null)}
      />
    </div>
  );
}
