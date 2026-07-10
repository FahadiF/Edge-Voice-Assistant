/**
 * SchemaForm tests against a fixture captured from the real backend
 * (`GET /settings/schema`, UISettings/VADSettings $defs) — pins the shape
 * the settings page depends on (ADR-023: a pydantic schema change must
 * break this test, not a user).
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SchemaSection } from "./SchemaForm";
import type { JsonSchema } from "./SchemaForm";

const UI_SETTINGS_SCHEMA: JsonSchema = {
  additionalProperties: false,
  properties: {
    theme: {
      default: "system",
      description: "Color theme",
      enum: ["dark", "light", "system"],
      title: "Theme",
      type: "string",
    },
    scale: {
      default: 1.0,
      description: "UI scale factor",
      maximum: 2.0,
      minimum: 0.75,
      title: "Scale",
      type: "number",
    },
    reduced_motion: {
      default: false,
      description: "Disable non-essential animations",
      title: "Reduced Motion",
      type: "boolean",
    },
  },
  title: "UISettings",
  type: "object",
};

const AUDIO_INPUT_DEVICE_SCHEMA: JsonSchema = {
  anyOf: [{ type: "string" }, { type: "null" }],
  default: null,
  description: "Input device name; None = system default",
  title: "Input Device",
};

const STOP_SEQUENCES_SCHEMA: JsonSchema = {
  description: "Extra sequences that end generation",
  items: { type: "string" },
  title: "Stop Sequences",
  type: "array",
};

describe("SchemaForm renders every field type the real settings schema uses", () => {
  it("renders an enum as a select with the current value", () => {
    render(
      <SchemaSection
        sectionSchema={UI_SETTINGS_SCHEMA}
        values={{ theme: "dark", scale: 1, reduced_motion: false }}
        onFieldChange={vi.fn()}
        errors={{}}
      />,
    );
    expect(screen.getByLabelText("Theme")).toHaveValue("dark");
  });

  it("renders a bounded number as a range + numeric input", () => {
    render(
      <SchemaSection
        sectionSchema={UI_SETTINGS_SCHEMA}
        values={{ theme: "dark", scale: 1.25, reduced_motion: false }}
        onFieldChange={vi.fn()}
        errors={{}}
      />,
    );
    expect(screen.getByLabelText("Scale")).toHaveValue(1.25);
  });

  it("renders a boolean as a checkbox switch", () => {
    render(
      <SchemaSection
        sectionSchema={UI_SETTINGS_SCHEMA}
        values={{ theme: "dark", scale: 1, reduced_motion: true }}
        onFieldChange={vi.fn()}
        errors={{}}
      />,
    );
    expect(screen.getByLabelText("Reduced Motion")).toBeChecked();
  });

  it("calls onFieldChange with the new value when a select changes", () => {
    const onFieldChange = vi.fn();
    render(
      <SchemaSection
        sectionSchema={UI_SETTINGS_SCHEMA}
        values={{ theme: "dark", scale: 1, reduced_motion: false }}
        onFieldChange={onFieldChange}
        errors={{}}
      />,
    );
    fireEvent.change(screen.getByLabelText("Theme"), { target: { value: "light" } });
    expect(onFieldChange).toHaveBeenCalledWith("theme", "light");
  });

  it("shows a field-level validation error", () => {
    render(
      <SchemaSection
        sectionSchema={UI_SETTINGS_SCHEMA}
        values={{ theme: "dark", scale: 1, reduced_motion: false }}
        onFieldChange={vi.fn()}
        errors={{ scale: "must be <= 2.0" }}
      />,
    );
    expect(screen.getByText("must be <= 2.0")).toBeInTheDocument();
  });

  it("renders a nullable string (anyOf [string, null]) as a text input", () => {
    render(
      <SchemaSection
        sectionSchema={{ properties: { input_device: AUDIO_INPUT_DEVICE_SCHEMA } }}
        values={{ input_device: null }}
        onFieldChange={vi.fn()}
        errors={{}}
      />,
    );
    const input = screen.getByLabelText("Input Device") as HTMLInputElement;
    expect(input.value).toBe("");
    expect(input.placeholder).toBe("(default)");
  });

  it("clears a nullable field back to null when emptied", () => {
    const onFieldChange = vi.fn();
    render(
      <SchemaSection
        sectionSchema={{ properties: { input_device: AUDIO_INPUT_DEVICE_SCHEMA } }}
        values={{ input_device: "Mic 1" }}
        onFieldChange={onFieldChange}
        errors={{}}
      />,
    );
    fireEvent.change(screen.getByLabelText("Input Device"), { target: { value: "" } });
    expect(onFieldChange).toHaveBeenCalledWith("input_device", null);
  });

  it("renders an array of strings as a tag editor with existing tags", () => {
    render(
      <SchemaSection
        sectionSchema={{ properties: { stop_sequences: STOP_SEQUENCES_SCHEMA } }}
        values={{ stop_sequences: ["<|end|>", "STOP"] }}
        onFieldChange={vi.fn()}
        errors={{}}
      />,
    );
    expect(screen.getByText("<|end|>")).toBeInTheDocument();
    expect(screen.getByText("STOP")).toBeInTheDocument();
  });

  it("renders nested $ref groups as fieldsets and bubbles nested changes", () => {
    // Shape captured from the real /settings/schema after the ADR-025
    // regroup: the permissions section's properties are $refs to sub-groups.
    const defs = {
      GeneralPermissions: {
        title: "GeneralPermissions",
        type: "object",
        properties: {
          internet: { type: "boolean", title: "Internet", default: false },
          date_time: { type: "boolean", title: "Date Time", default: true },
        },
      },
    };
    const permissionsSchema: JsonSchema = {
      type: "object",
      properties: { general: { $ref: "#/$defs/GeneralPermissions" } },
    };
    const onFieldChange = vi.fn();
    render(
      <SchemaSection
        sectionSchema={permissionsSchema}
        values={{ general: { internet: false, date_time: true } }}
        onFieldChange={onFieldChange}
        errors={{}}
        defs={defs}
      />,
    );
    expect(screen.getByRole("group", { name: "General" })).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("Internet"));
    // The change bubbles up as the whole updated sub-object.
    expect(onFieldChange).toHaveBeenCalledWith("general", {
      internet: true,
      date_time: true,
    });
  });

  it("adds a tag on Enter", () => {
    const onFieldChange = vi.fn();
    render(
      <SchemaSection
        sectionSchema={{ properties: { stop_sequences: STOP_SEQUENCES_SCHEMA } }}
        values={{ stop_sequences: ["STOP"] }}
        onFieldChange={onFieldChange}
        errors={{}}
      />,
    );
    const input = screen.getByLabelText("Add to Stop Sequences");
    fireEvent.change(input, { target: { value: "END" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onFieldChange).toHaveBeenCalledWith("stop_sequences", ["STOP", "END"]);
  });
});
