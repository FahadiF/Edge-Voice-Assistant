# ADR-025: Permissions and local system information

Status: Accepted · Date: 2026-07-06

## Context

M5.3 manual testing surfaced two related gaps. First, a *local* assistant
could not answer "what time is it?" or "how much RAM do I have?" — facts
the machine trivially knows — because nothing put them in the prompt.
Second, the platform roadmap adds capabilities that touch the user's system
and data (internet, files, camera, shell, …), and there was no user-facing
control surface for what the assistant may know or do. Bolting consent onto
each future capability separately would scatter the policy.

## Decision

### 1. One `PermissionsSettings` section — the consent surface for everything

`settings.permissions` (ADR-009: schema-driven, so the Settings UI renders
it automatically as a "Permissions" section) holds one toggle per
capability: `date_time`, `timezone`, `locale`, `cpu`, `gpu`, `ram`, `os`,
`internet`, `local_files`, `camera`, `clipboard`, `browser`, `shell`,
`python`, `plugins`.

Defaults follow one rule: **read-only local facts are on; anything that
acts or exfiltrates is off.** The action permissions (`internet`,
`local_files`, `camera`, `clipboard`, `browser`, `shell`, `python`) are
also *unimplemented* in this build — the toggle exists now because it is
the contract every future capability provider must consult before doing
anything, not because flipping it does something today. Their descriptions
say so ("not available in this build").

### 2. System Information provider (`eva.conversation.system_info`)

`system_facts_block(permissions)` renders the permitted facts into the
system prompt each turn (ADR-021 Amendment 3 hierarchy: after memories,
before the technical-facts section): current date/time (fresh every turn —
this is what makes "what time is it?" answerable), timezone, locale, OS,
and CPU/GPU/RAM from the existing `eva.hardware` detection, **cached once
per process** (the probes shell out to `nvidia-smi`/WMI; hardware doesn't
change mid-session, the clock does).

The Context Builder's capability guidance closes the loop: when a fact is
absent because its permission is off, the assistant says the user hasn't
granted that permission — never "I can't know that" (the same
honest-scoping rule ADR-021 Amendment 3 set for unbuilt capabilities).

### 3. Typed input: `POST /conversation/say` and the composer

The web UI gains a ChatGPT-style composer. Typed messages call
`POST /conversation/say` → `Orchestrator.submit_text()`, which enters the
**same event queue** as audio events (`_TextInput` beside the segmenter
events) and runs the same turn pipeline minus the ASR stage — one
sequencing path, no parallel text pipeline, same barge-in/supersede
semantics, reply spoken and streamed like any turn. The composer's
attachment surface (+ menu, drag-and-drop, paste) is architecture-first:
files become visible placeholder chips labeled "not available in this
build" — the UI shape ships now, the upload capability arrives with the
image/document permissions above.

## Consequences

- Future capability providers (a shell tool, a file reader, an online LLM
  provider) MUST check `settings.permissions.<toggle>` — the checkbox is
  the user's contract, and it already exists in every settings surface
  (UI/API/CLI) with no further wiring.
- The "Online" conversation-mode selector in the UI is a disabled
  placeholder for future providers; only local/offline exists (the
  product's core promise is unchanged).
- Prompt size grows by a few lines when permissions allow; each fact is one
  short line and the block is omitted entirely when everything is off.
- `submit_text` is the first non-audio input path; anything else that
  wants to start turns (plugins, scheduled prompts) should follow the same
  event-queue pattern rather than calling the pipeline directly.

## Amendment (M5.4, 2026-07-06): grouped permissions, real enforcement

The flat 15-toggle list regrouped into five sections (`general`, `files`,
`devices`, `tools`, `privacy`) with coarser, clearer toggles —
`general.date_time` covers date/time/timezone; `general.system_information`
covers OS/CPU/GPU/RAM/locale. New toggles: `files.write_files`,
`devices.microphone`, `privacy.remember_conversations`,
`privacy.learn_preferences` (reserved). `clipboard` was dropped (never
implemented, not in the regrouped design).

Three toggles are now *enforced*, not declaratory:
- `general.*` — gates the system-facts prompt section (as before);
- `devices.microphone` — `Assistant.start_audio()` skips audio capture
  entirely when off: a typed-chat-only assistant (composer + TTS still
  work);
- `privacy.remember_conversations` — the orchestrator stores nothing when
  off. This replaces `conversation.memory_enabled`, which was **dead
  code** (defined, displayed, enforced nowhere) since M4.

Settings documents migrate in memory on load (`SETTINGS_SCHEMA_VERSION` 1→2,
dict-level `_migrate_raw`, mirroring the memory DB's numbered-migration
pattern); v1 flat keys map into the groups and `memory_enabled` carries
into `privacy.remember_conversations`. The SchemaForm renders the nested
groups automatically (one-level `$ref` resolution — ADR-009 still holds:
no hand-coded settings UI).
