# ADR-022: Personas, user profiles, and voices

Status: Accepted · Date: 2026-07-05

## Context

`ConversationSettings.persona` has existed since M2 (default `"default"`)
with nothing behind it. M4 asks for personality profiles, user profiles
(nickname, preferred language/voice/models, units, timezone, style), and
multi-voice support, all persisted and all designed to "reuse without
redesign" for future plugins and multiple users.

## Decision

### Personas (Part 5)

`eva.conversation.personas` — a registry exactly mirroring the existing
`eva.conversation.language` pattern: `PersonaProfile` (`id`, `display_name`,
`system_prompt`, `verbosity`, `tone`, `reasoning_style`, optional
`temperature_override`), `persona_registry: Registry[PersonaProfile]`,
`register_builtin_personas()` (Default, Professional, Friendly, Technical,
Minimal, Creative), `resolve_persona(settings)`. Built-ins are code, exactly
like built-in languages. **Custom personas are settings data, not memory
data**: a new `custom_personas: list[PersonaProfile]` field on
`ConversationSettings`, registered into `persona_registry` at startup after
the built-ins. Rationale: a persona is configuration (what should the
assistant sound like), not conversation history — it belongs with every
other piece of deterministic, `settings.json`-persisted configuration
(ADR-015), the same category `ConversationSettings.persona` (the *selector*)
has always been in.

### User profiles (Part 6)

Different category from personas: a user profile is *data about a person*
(nickname, preferred language/voice/model, conversation style, units,
timezone), not application configuration, and the brief explicitly wants
this to extend to multiple users without redesign — something a single
`settings.json` document (one process-wide document, not row-based) isn't
shaped for. **User profiles live in the memory database** (`user_profiles`
table, ADR-019 §4) behind a small `UserProfileStore` port, sharing
`eva/memory/db.py`'s connection rather than opening a second database file.
`ConversationSettings.active_profile_id` (new field) is the only
profile-related thing that lives in `settings.json` — *which* profile is
active is a small piece of deterministic configuration; the profile's
*content* is data.

### Voices (Part 7)

**Not a new top-level subsystem.** A "voice" is metadata about a capability
an existing `TTSEngine` already exposes (`voices()` — capability discovery,
present since M2), not an independently pluggable capability the way
ASR/LLM/TTS/VAD engines themselves are — there is nothing to "swap" about a
voice; you select one from whatever the active TTS engine offers. New
`eva/tts/voices.py`: `VoiceInfo` (`id`, `engine`, `display_name`,
`language`, `style_tag`) and `voice_registry: Registry[VoiceInfo]`,
populated by combining each engine's `voices()` output with a small curated
metadata table for known engines (Kokoro's ~50 voice ids get real display
names/languages; anything unrecognized — a future engine, a plugin-provided
voice — falls back to the bare id, same graceful-degradation shape as
ADR-016's TTS-voice-fallback). Preview reuses the already-loaded
`TTSEngine.synthesize()` on a short fixed phrase — no new synthesis path.
Selection already persists via the existing `settings.tts.voice` field; no
new settings needed here, only the registry that makes the existing field's
valid values discoverable and describable.

### Naming: "profile" already means something else here

`Settings.profile` and the `eva profiles` CLI command already mean the
hardware/model preset (Balanced/Fast/High Accuracy/…, ADR-015). To avoid
two unrelated concepts sharing one word, the new per-person concept is
called **"user profile"** everywhere — code (`UserProfile`,
`UserProfileStore`), settings (`active_profile_id` lives under
`conversation`, not a bare `profile_id` that could be misread), and API
paths (`/api/v1/users`, deliberately not `/api/v1/profiles`).

## Rationale

The three features sit in three different places because they're three
different *kinds* of thing — configuration (personas), per-person data (user
profiles), and capability metadata (voices) — and each already has an
established home in this codebase for its kind. Forcing all three into one
new "personalization" package would be organizing by feature-of-the-month
instead of by what the thing actually is, the opposite of ADR-010's
package-by-subsystem principle.

## Consequences

- `eva.conversation` gains one more registry-backed concept (personas),
  consistent with how it already owns language profiles.
- `eva.memory` gains a second port (`UserProfileStore`) alongside
  `MemoryStore`, sharing one connection — no second SQLite file, no
  duplicated connection-management code.
- `eva.tts` gains a registry without gaining a new subsystem package —
  voices are TTS metadata, so they live where TTS already lives.
