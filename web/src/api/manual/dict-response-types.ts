/**
 * Schema-less dict responses — hand-maintained.
 *
 * Not generatable from OpenAPI: each backing endpoint returns a plain
 * `dict`/`dict[str, Any]` with no `response_model` declared (verified —
 * `ModelManager.describe()` and `routers/memory.py`'s `export_memory` both
 * do this), so FastAPI's schema for these routes carries no field
 * information to generate from. Unlike `manual/websocket-types.ts`, the
 * backend genuinely has no schema here yet, not merely an OpenAPI-invisible
 * transport — each entry becomes a generation candidate the moment its
 * endpoint gains a real `response_model` (see BACKLOG).
 */

import type { MemoryTurn } from "../types.generated";

export type ModelKind = "llm" | "asr" | "tts" | "vad" | "embedding";

export interface ModelCard {
  id: string;
  name: string;
  kind: ModelKind;
  version: string;
  provider: string;
  license: string;
  languages: string;
  context_length: number | null;
  quantization: string | null;
  vram_mb: number;
  ram_mb: number;
  download_mb: number;
  disk_usage_mb: number;
  engine: string;
  managed_by: "manager" | "engine" | "bundled";
  installed: boolean;
  installed_version: string | null;
  update_available: boolean;
  active: boolean;
  compatible: boolean;
  compatibility_notes: string;
  recommendation: string;
  notes: string;
}

export interface MemoryExport {
  version?: number;
  conversations?: {
    conversation: { id: string; started_at: string; title: string; language: string; archived: boolean };
    turns: MemoryTurn[];
  }[];
  [key: string]: unknown;
}
