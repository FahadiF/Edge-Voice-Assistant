# ADR-016: Multilingual architecture — language profiles as registry data

Status: Accepted · Date: 2026-07-04

## Context
The platform must eventually converse in dozens of languages. Whisper (ASR)
and Qwen (LLM) are already broadly multilingual; TTS voices are the narrow
end. Language concerns touch four places — ASR hints, LLM prompting, TTS voice
selection, and (future) automatic detection — and must not be scattered as
per-language conditionals through the pipeline.

## Decision
1. **One `LanguageProfile` per language** (`eva/conversation/language.py`),
   registered in a language registry (ADR-010): BCP-47 code, display name,
   ASR language hint, a system-prompt note instructing the LLM to respond in
   that language, and preferred TTS voices per engine. Adding a language is a
   data entry, not a pipeline change.
2. **Single resolution point.** The orchestrator resolves the active language
   once at construction and derives: effective system prompt (base + language
   note), effective ASR hint (explicit `asr.language` override wins), and the
   effective TTS voice.
3. **Graceful capability degradation.** When the active TTS engine has no
   voice for the language, the configured default voice is used with a logged
   warning — the assistant still understands and answers in the requested
   language with a non-native voice, rather than failing. Capability honesty
   is surfaced, not hidden.
4. Registered at introduction: English, Finnish, Swedish, Bengali (the tested
   set), plus German and Spanish. Automatic language detection (Whisper
   already returns a detected language) is a future consumer of the same
   registry; per-language model overrides slot into presets when needed.

## Consequences
- `conversation.language` is a persisted setting like any other; the future
  UI renders the choices from the registry.
- The TTS gap for fi/sv/bn is explicit: closing it means adding a voice/model
  to the catalog and one line to the language profile.
