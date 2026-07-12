/** Typed wrappers for every REST endpoint the UI consumes (ADR-017). */

import { api } from "./client";
import type {
  ContextPreviewResponse,
  ConversationExport,
  ConversationTurn,
  DownloadStartedResponse,
  EngineStatusResponse,
  HardwareSummary,
  HealthResponse,
  MemoryExport,
  MemorySearchResult,
  MemoryStats,
  ModelCard,
  ModelKind,
  PersonaProfile,
  PersonaSettingsEntry,
  PluginStatus,
  ReadinessResponse,
  RuntimeSnapshot,
  Settings,
  SettingsValidationResult,
  UserProfile,
  VoiceInfo,
} from "./types";

export const system = {
  health: () => api.get<HealthResponse>("/health"),
  hardware: () => api.get<HardwareSummary>("/system/hardware"),
};

export const settings = {
  get: () => api.get<Settings>("/settings"),
  schema: () => api.get<Record<string, unknown>>("/settings/schema"),
  put: (doc: unknown) => api.put<Settings>("/settings", doc),
  patch: (partial: unknown) => api.patch<Settings>("/settings", partial),
  validate: (doc: unknown) => api.post<SettingsValidationResult>("/settings/validate", doc),
  reset: () => api.post<Settings>("/settings/reset"),
};

export const models = {
  list: (kind?: ModelKind) =>
    api.get<ModelCard[]>(kind ? `/models?kind=${kind}` : "/models"),
  get: (id: string) => api.get<ModelCard>(`/models/${id}`),
  download: (id: string) => api.post<DownloadStartedResponse>(`/models/${id}/download`),
  remove: (id: string) => api.delete<{ model_id: string; status: string }>(`/models/${id}`),
  activate: (id: string) => api.post<ModelCard>(`/models/${id}/activate`),
};

export const engine = {
  status: () => api.get<EngineStatusResponse>("/engine/status"),
  readiness: () => api.get<ReadinessResponse>("/engine/readiness"),
  start: () => api.post<EngineStatusResponse>("/engine/start"),
  stop: () => api.post<EngineStatusResponse>("/engine/stop"),
};

export const conversation = {
  history: () => api.get<ConversationTurn[]>("/conversation/history"),
  say: (text: string) => api.post<{ status: string }>("/conversation/say", { text }),
  setMicrophone: (muted: boolean) =>
    api.post<{ muted: boolean }>("/conversation/microphone", { muted }),
  interrupt: () => api.post<{ interrupted: boolean }>("/conversation/interrupt"),
  clear: () => api.post<{ status: string }>("/conversation/clear"),
  resume: (conversationId: string) =>
    api.post<{ status: string; conversation_id: string; title: string; turns: number }>(
      "/conversation/resume",
      { conversation_id: conversationId },
    ),
  exportJson: () => api.get<ConversationExport>("/conversation/export"),
  importJson: (turns: ConversationTurn[]) =>
    api.post<{ status: string; turns: string }>("/conversation/import", { turns }),
};

export const memory = {
  search: (query: string, limit = 20, conversationId?: string) =>
    api.post<MemorySearchResult[]>("/memory/search", {
      query,
      limit,
      conversation_id: conversationId ?? null,
    }),
  stats: () => api.get<MemoryStats>("/memory/stats"),
  contextPreview: (text: string, conversationId?: string) => {
    const params = new URLSearchParams({ text });
    if (conversationId) params.set("conversation_id", conversationId);
    return api.get<ContextPreviewResponse>(`/memory/context-preview?${params}`);
  },
  forget: (turnId: number) => api.delete<{ status: string }>(`/memory/turns/${turnId}`),
  pin: (turnId: number, pinned = true) =>
    api.post<{ status: string }>(`/memory/turns/${turnId}/pin?pinned=${pinned}`),
  favorite: (turnId: number, favorite = true) =>
    api.post<{ status: string }>(`/memory/turns/${turnId}/favorite?favorite=${favorite}`),
  rename: (conversationId: string, title: string) =>
    api.patch<{ status: string; title: string }>(`/memory/conversations/${conversationId}`, {
      title,
    }),
  archive: (conversationId: string, archived = true) =>
    api.post<{ status: string }>(
      `/memory/conversations/${conversationId}/archive?archived=${archived}`,
    ),
  deleteConversation: (conversationId: string) =>
    api.delete<{ status: string }>(`/memory/conversations/${conversationId}`),
  merge: (sourceId: string, targetId: string) =>
    api.post<{ status: string }>("/memory/conversations/merge", {
      source_id: sourceId,
      target_id: targetId,
    }),
  summarize: (conversationId: string) =>
    api.post<{ status: string; summary: string | null }>(
      `/memory/conversations/${conversationId}/summarize`,
    ),
  exportJson: (conversationId?: string) =>
    api.get<MemoryExport>(
      conversationId ? `/memory/export?conversation_id=${conversationId}` : "/memory/export",
    ),
  importJson: (payload: unknown) =>
    api.post<{ status: string; turns: string }>("/memory/import", payload),
  deleteAll: () => api.delete<{ status: string }>("/memory"),
};

export const personas = {
  list: () => api.get<PersonaProfile[]>("/personas"),
  get: (id: string) => api.get<PersonaProfile>(`/personas/${id}`),
  create: (entry: PersonaSettingsEntry) => api.post<PersonaProfile>("/personas", entry),
  remove: (id: string) => api.delete<{ status: string }>(`/personas/${id}`),
};

export const users = {
  list: () => api.get<UserProfile[]>("/users"),
  get: (id: string) => api.get<UserProfile>(`/users/${id}`),
  create: (profile: Partial<UserProfile>) => api.post<UserProfile>("/users", profile),
  update: (id: string, patch: Partial<UserProfile>) =>
    api.patch<UserProfile>(`/users/${id}`, patch),
  activate: (id: string) => api.post<{ status: string }>(`/users/${id}/activate`),
  remove: (id: string) => api.delete<{ status: string }>(`/users/${id}`),
};

export const voices = {
  list: () => api.get<VoiceInfo[]>("/voices"),
  preview: (id: string, phrase?: string) =>
    api.postBinary(`/voices/${id}/preview`, { phrase: phrase ?? "Hello, this is a preview of my voice." }),
};

export const plugins = {
  list: () => api.get<PluginStatus[]>("/plugins"),
  enable: (id: string) => api.post<PluginStatus>(`/plugins/${id}/enable`),
  disable: (id: string) => api.post<PluginStatus>(`/plugins/${id}/disable`),
  reload: (id: string) => api.post<PluginStatus>(`/plugins/${id}/reload`),
};

export const diagnostics = {
  snapshot: () => api.get<RuntimeSnapshot>("/diagnostics"),
};
