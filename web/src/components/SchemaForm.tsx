/**
 * JSON-Schema-driven settings form (ADR-009: the settings UI is generated
 * from `GET /settings/schema`, never hand-coded field lists).
 *
 * Handles the shapes pydantic v2 actually emits for the Settings model:
 * scalars (string/number/integer/boolean), enums, `anyOf: [T, null]`
 * (nullable), arrays of strings (tag editor), and numeric bounds
 * (minimum/maximum → slider + input). Arrays of objects (custom_personas)
 * are managed by their dedicated page and skipped here.
 */

import { useId, useState } from "react";
import "./schemaform.css";

export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  items?: JsonSchema & { $ref?: string };
  anyOf?: (JsonSchema & { $ref?: string })[];
  properties?: Record<string, JsonSchema>;
  $ref?: string;
  additionalProperties?: boolean;
}

/** Unwrap `anyOf: [T, {type: null}]` into (T, nullable=true). */
function unwrapNullable(schema: JsonSchema): { schema: JsonSchema; nullable: boolean } {
  if (!schema.anyOf) return { schema, nullable: false };
  const nonNull = schema.anyOf.filter((s) => s.type !== "null");
  if (nonNull.length === 1) {
    return { schema: { ...schema, ...nonNull[0], anyOf: undefined }, nullable: true };
  }
  return { schema, nullable: false };
}

function TagEditor({
  value,
  onChange,
  label,
}: {
  value: string[];
  onChange: (v: string[]) => void;
  label: string;
}) {
  const [draft, setDraft] = useState("");
  const add = () => {
    const t = draft.trim();
    if (t && !value.includes(t)) onChange([...value, t]);
    setDraft("");
  };
  return (
    <div className="tag-editor">
      <div className="tags">
        {value.map((tag) => (
          <span key={tag} className="chip">
            {tag}
            <button
              className="tag-remove"
              aria-label={`Remove ${tag}`}
              onClick={() => onChange(value.filter((t) => t !== tag))}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div className="tag-input">
        <input
          value={draft}
          placeholder="Add…"
          aria-label={`Add to ${label}`}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              add();
            }
          }}
        />
        <button onClick={add}>Add</button>
      </div>
    </div>
  );
}

export function SchemaField({
  name,
  schema,
  value,
  onChange,
  error,
}: {
  name: string;
  schema: JsonSchema;
  value: unknown;
  onChange: (v: unknown) => void;
  error?: string;
}) {
  const id = useId();
  const { schema: inner, nullable } = unwrapNullable(schema);
  const label = inner.title ?? name;

  // Arrays of objects are page-managed (personas) — not rendered generically.
  if (inner.type === "array" && inner.items?.$ref) return null;

  let control: React.ReactNode;

  if (inner.enum) {
    control = (
      <select
        id={id}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" && nullable ? null : e.target.value)}
      >
        {nullable && <option value="">(default)</option>}
        {inner.enum.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    );
  } else if (inner.type === "boolean") {
    control = (
      <input
        id={id}
        type="checkbox"
        role="switch"
        checked={Boolean(value)}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  } else if (inner.type === "integer" || inner.type === "number") {
    const hasBounds = inner.minimum !== undefined && inner.maximum !== undefined;
    const parse = (raw: string) => {
      if (raw === "") return nullable ? null : undefined;
      const n = inner.type === "integer" ? parseInt(raw, 10) : parseFloat(raw);
      return Number.isNaN(n) ? undefined : n;
    };
    control = (
      <div className="number-control">
        {hasBounds && (
          <input
            type="range"
            aria-hidden="true"
            tabIndex={-1}
            min={inner.minimum}
            max={inner.maximum}
            step={inner.type === "integer" ? 1 : (inner.maximum! - inner.minimum!) / 100}
            value={typeof value === "number" ? value : Number(inner.default ?? inner.minimum)}
            onChange={(e) => onChange(parse(e.target.value))}
          />
        )}
        <input
          id={id}
          type="number"
          min={inner.minimum}
          max={inner.maximum}
          step={inner.type === "integer" ? 1 : "any"}
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => {
            const parsed = parse(e.target.value);
            if (parsed !== undefined) onChange(parsed);
          }}
        />
      </div>
    );
  } else if (inner.type === "array") {
    control = (
      <TagEditor
        label={label}
        value={Array.isArray(value) ? (value as string[]) : []}
        onChange={onChange}
      />
    );
  } else {
    // string (and anything unrecognized degrades to a text input)
    control = (
      <input
        id={id}
        type="text"
        value={value === null || value === undefined ? "" : String(value)}
        placeholder={nullable ? "(default)" : undefined}
        onChange={(e) => onChange(e.target.value === "" && nullable ? null : e.target.value)}
      />
    );
  }

  return (
    <div className={`schema-field ${error ? "has-error" : ""}`}>
      <label htmlFor={id} className="field-label">
        {label}
      </label>
      {control}
      {inner.description && <p className="field-help">{inner.description}</p>}
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

export function SchemaSection({
  sectionSchema,
  values,
  onFieldChange,
  errors,
}: {
  sectionSchema: JsonSchema;
  values: Record<string, unknown>;
  onFieldChange: (field: string, value: unknown) => void;
  errors: Record<string, string>;
}) {
  const properties = sectionSchema.properties ?? {};
  return (
    <div className="schema-section">
      {Object.entries(properties).map(([field, fieldSchema]) => (
        <SchemaField
          key={field}
          name={field}
          schema={fieldSchema}
          value={values[field]}
          onChange={(v) => onFieldChange(field, v)}
          error={errors[field]}
        />
      ))}
    </div>
  );
}
