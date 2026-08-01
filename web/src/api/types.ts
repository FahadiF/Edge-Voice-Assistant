/**
 * TypeScript mirror of the backend's API contract (ADR-023, Batch 3).
 *
 * Re-exports two categories:
 * - `types.generated.ts` — generated from `/openapi.json` via
 *   `npm run generate:types`. Do not hand-edit that file; regenerate it.
 * - `manual/*.ts` — hand-maintained, for the parts of the contract OpenAPI
 *   cannot describe (see each file's own header for why).
 *
 * A handful of names differ between the generated schema and this app's
 * established naming; aliased below so no import site elsewhere had to
 * change when this file switched from fully hand-written to generated.
 */

export * from "./types.generated";
export * from "./manual/websocket-types";
export * from "./manual/dict-response-types";

export type { PluginStatusResponse as PluginStatus } from "./types.generated";
export type { ContextTraceResponse as ContextTrace } from "./types.generated";
export type {
  RetrievedMemoryTraceResponse as RetrievedMemoryTrace,
} from "./types.generated";
