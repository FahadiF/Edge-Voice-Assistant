/**
 * TypeScript mirrors of the backend's pydantic schemas (ADR-023).
 *
 * Hand-maintained 1:1 with `src/eva/server/schemas.py`, `eva/config/
 * settings.py`, `eva/metrics/diagnostics.py`, `eva/memory/models.py`,
 * `eva/core/events.py`. When a pydantic model changes, change the mirror —
 * the SchemaForm fixture test pins the settings shape so drift fails CI.
 */

// ── Settings (eva/config/settings.py) ──────────────────────────────────────

export interface AudioSettings {
  input_device: string | null;
  output_device: string | null;
  sample_rate: number;
  mic_gain: number;
  speaker_volume: number;
  echo_cancellation: boolean;
  noise_suppression: boolean;
  auto_gain_control: boolean;
  fade_out_ms: number;
  duplex_mode: string;
}

export interface VADSettings {
  engine: string;
  threshold: number;
  silence_timeout_ms: number;
  min_speech_ms: number;
  max_utterance_s: number;
  barge_in_enabled: boolean;
  barge_in_confirm_ms: number;
}

export interface ASRSettings {
  engine: string;
  model: string;
  language: string;
  device: string;
  compute_type: string;
  partial_transcripts: boolean;
  partial_interval_ms: number;
}

export interface LLMSettings {
  engine: string;
  model: string;
  context_length: number;
  gpu_layers: number;
  threads: number;
  batch_size: number;
}

export interface TTSSettings {
  engine: string;
  model: string;
  voice: string;
  speed: number;
  pitch: number;
  streaming: boolean;
}

export interface PersonaSettingsEntry {
  id: string;
  display_name: string;
  system_prompt: string;
  verbosity: "minimal" | "concise" | "normal" | "detailed";
  tone: string;
  reasoning_style: string;
  temperature_override: number | null;
}

export interface ConversationSettings {
  system_prompt: string;
  persona: string;
  language: string;
  memory_enabled: boolean;
  max_history_turns: number;
  temperature: number;
  top_p: number;
  max_tokens: number;
  stop_sequences: string[];
  sentence_min_chars: number;
  sentence_max_chars: number;
  first_sentence_min_chars: number;
  active_profile_id: string | null;
  custom_personas: PersonaSettingsEntry[];
}

export interface MemorySettings {
  engine: string;
  embedding_enabled: boolean;
  embedding_engine: string;
  embedding_model: string;
  retention_days: number;
  max_turns_per_conversation: number;
  auto_cleanup_enabled: boolean;
  encrypt_at_rest: boolean;
  retrieval_top_k: number;
  retrieval_scan_limit: number;
  max_memory_chars: number;
  max_summary_chars: number;
  recency_half_life_days: number;
  pinned_boost: number;
  favorite_boost: number;
  summarize_after_turns: number;
}

export interface PermissionsSettings {
  date_time: boolean;
  timezone: boolean;
  locale: boolean;
  cpu: boolean;
  gpu: boolean;
  ram: boolean;
  os: boolean;
  internet: boolean;
  local_files: boolean;
  camera: boolean;
  clipboard: boolean;
  browser: boolean;
  shell: boolean;
  python: boolean;
  plugins: boolean;
}

export interface ServerSettings {
  host: string;
  port: number;
}

export interface UISettings {
  theme: "dark" | "light" | "system";
  scale: number;
  reduced_motion: boolean;
}

export interface DeveloperSettings {
  debug: boolean;
  log_level: string;
  log_json: boolean;
  metrics_enabled: boolean;
}

export interface Settings {
  schema_version: number;
  profile: string;
  audio: AudioSettings;
  vad: VADSettings;
  asr: ASRSettings;
  llm: LLMSettings;
  tts: TTSSettings;
  conversation: ConversationSettings;
  memory: MemorySettings;
  permissions: PermissionsSettings;
  server: ServerSettings;
  ui: UISettings;
  developer: DeveloperSettings;
}

export interface ValidationErrorDetail {
  loc: (string | number)[];
  message: string;
  type: string;
}

export interface SettingsValidationResult {
  valid: boolean;
  errors: ValidationErrorDetail[];
}

// ── Models (eva/models/manager.py describe()) ──────────────────────────────

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
  notes: string;
}

export interface DownloadStartedResponse {
  model_id: string;
  status: "started" | "already_running" | "not_applicable";
}

// ── Engine / system ────────────────────────────────────────────────────────

export type PipelineState = "idle" | "listening" | "thinking" | "speaking";

export interface EngineStatusResponse {
  running: boolean;
  state: string;
}

export interface ReadinessResponse {
  ready: boolean;
  problems: string[];
}

export interface HealthResponse {
  status: "ok";
  version: string;
}

export interface HardwareSummary {
  tier: string;
  tier_name: string;
  cpu: string;
  gpu: string | null;
  vram_mb: number;
  ram_mb: number;
}

// ── Conversation ───────────────────────────────────────────────────────────

export interface ConversationTurn {
  user: string;
  assistant: string;
}

export interface ConversationExport {
  version: number;
  exported_at: string;
  profile: string;
  language: string;
  turns: ConversationTurn[];
}

// ── Memory (eva/memory/models.py) ──────────────────────────────────────────

export type Speaker = "user" | "assistant";

export interface MemoryTurn {
  id: number | null;
  conversation_id: string;
  created_at: string;
  speaker: Speaker;
  text: string;
  language: string;
  metadata: Record<string, unknown>;
  pinned: boolean;
  favorite: boolean;
  deleted: boolean;
}

export interface MemorySearchResult {
  turn: MemoryTurn;
  score: number;
  match_reason: "semantic" | "keyword" | "pinned";
}

export interface MemoryStats {
  conversation_count: number;
  turn_count: number;
  embedded_turn_count: number;
  summary_count: number;
  db_size_bytes: number;
  fts_enabled: boolean;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

export interface RetrievedMemoryTrace {
  turn_id: number;
  score: number;
  text_preview: string;
}

export interface ContextTrace {
  persona_id: string;
  profile_id: string | null;
  language_code: string;
  retrieved_memories: RetrievedMemoryTrace[];
  summary_included: boolean;
  summary_text_preview: string | null;
  recent_turn_count: number;
  trimmed_sections: string[];
}

export interface ContextPreviewResponse {
  messages: ChatMessage[];
  trace: ContextTrace;
}

export interface MemoryExport {
  version?: number;
  conversations?: {
    conversation: { id: string; started_at: string; title: string; language: string; archived: boolean };
    turns: MemoryTurn[];
  }[];
  [key: string]: unknown;
}

// ── Personas / users / voices ──────────────────────────────────────────────

export interface PersonaProfile {
  id: string;
  display_name: string;
  system_prompt: string;
  verbosity: string;
  tone: string;
  reasoning_style: string;
  temperature_override: number | null;
}

export interface UserProfile {
  id: string;
  nickname: string;
  preferred_language: string | null;
  preferred_voice: string | null;
  preferred_llm_model: string | null;
  conversation_style: string;
  units: "metric" | "imperial";
  timezone: string;
  extra: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  active: boolean;
}

export interface VoiceInfo {
  id: string;
  engine: string;
  display_name: string;
  language: string;
  style_tag: string;
}

// ── Plugins ────────────────────────────────────────────────────────────────

export interface PluginStatus {
  id: string;
  name: string;
  version: string;
  description: string;
  enabled: boolean;
  healthy: boolean;
  error: string | null;
  contributes: string[];
  permissions: string[];
}

// ── Diagnostics (eva/metrics/diagnostics.py RuntimeSnapshot) ───────────────

export interface ResourceUsage {
  cpu_percent: number;
  ram_used_mb: number;
  ram_total_mb: number;
  gpu_percent: number | null;
  vram_used_mb: number | null;
  vram_total_mb: number | null;
}

export interface TurnMetrics {
  epoch: number;
  asr_ms: number;
  ttft_ms: number;
  llm_ms: number;
  tokens: number;
  tts_first_ms: number;
  ttfa_ms: number;
  total_ms: number;
  cancelled: boolean;
}

export interface RuntimeSnapshot {
  profile: string;
  language: string;
  models: Record<string, string>;
  devices: Record<string, string>;
  state: PipelineState;
  epoch: number;
  playback_active: boolean;
  input_level_dbfs: number;
  pending_audio_events: number;
  capture_ring_depth: number;
  capture_frames_dropped: number;
  token_queue_depth: number;
  sentence_queue_depth: number;
  playback_queued_seconds: number;
  resources: ResourceUsage;
  last_turn: TurnMetrics | null;
  turns_completed: number;
  metrics_summary: string;
  barge_in_count: number;
  last_barge_in_latency_ms: number | null;
  memory_enabled: boolean;
  memory_turn_count: number;
  memory_db_size_bytes: number;
  memory_embedding_count: number;
  last_retrieval_ms: number | null;
  last_retrieval_score_top1: number | null;
  active_persona_id: string;
  active_profile_id: string | null;
  active_voice: string;
  recent_events: string[];
}

// ── WebSocket events (eva/core/events.py) ──────────────────────────────────

export interface WsEnvelope {
  type: string;
  data: Record<string, unknown>;
}

export interface TurnStartedEvent { epoch: number }
export interface TurnFinishedEvent { epoch: number; error: string | null }
export interface TurnCancelledEvent {
  epoch: number;
  reason: "barge-in" | "superseded" | "shutdown" | "manual";
}
export interface SpeechStartedEvent { epoch: number }
export interface SpeechFinishedEvent { epoch: number; duration_ms: number }
export interface BargeInDetectedEvent { epoch: number }
export interface BargeInLatencyMeasuredEvent { epoch: number; detected_to_silent_ms: number }
export interface PartialTranscriptEvent { epoch: number; text: string }
export interface FinalTranscriptEvent { epoch: number; text: string; asr_ms: number }
export interface LlmStartedEvent { epoch: number }
export interface LlmTokenEvent { epoch: number; token: string }
export interface LlmSentenceEvent { epoch: number; text: string }
export interface LlmFinishedEvent {
  epoch: number;
  text: string;
  tokens: number;
  ttft_ms: number;
  duration_ms: number;
}
export interface TtsStartedEvent { epoch: number }
export interface TtsAudioReadyEvent { epoch: number; ttfa_ms: number }
export interface TtsFinishedEvent { epoch: number }
export interface StateChangedEvent { state: PipelineState }
export interface ModelDownloadProgressEvent {
  model_id: string;
  filename: string;
  bytes_done: number;
  bytes_total: number;
}
export interface ModelDownloadCompletedEvent { model_id: string }
export interface ModelDownloadFailedEvent { model_id: string; error: string }
export interface ErrorOccurredEvent { message: string; context: string }
