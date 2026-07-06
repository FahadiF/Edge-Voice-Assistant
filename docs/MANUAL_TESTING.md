# Manual Testing Guide — M4 (Memory, Personalization & Intelligence) & M5 (Web UI)

This guide lets someone with no source-code access validate every M4/M5
capability end-to-end: memory persistence, semantic retrieval, summaries,
personas, user profiles, voices, the REST API, the WebSocket stream, raw
SQLite inspection, diagnostics, and — as of M5 — the React web UI and the
minimal desktop shell. Sections 1–12 use the `eva` CLI or plain
`curl`/`sqlite3`; [section 14](#14-web-ui-m5) walks the same capabilities
through the browser. Nothing here requires reading the codebase.

Assumes `eva doctor` reports all checks passing (models installed, runtime
available). If not, run `eva first-run` first.

All data lives under your `EVA_HOME` (defaults to a per-OS app-data
directory; `eva diagnose` prints the exact `Conversations:` path — the
memory database is `memory.db` inside it).

---

## 1. Assistant identity (does it call itself "Edge Voice Assistant"?)

```bash
eva run
```

Once you see `Ready. Speak into the microphone...`, ask:

- **"Who are you?"** — expect an answer naming itself **Edge Voice
  Assistant**, never the underlying LLM.
- **"Which LLM powers you?"** or **"What model are you running on?"** —
  expect it to now answer honestly with the real backend (e.g.
  `qwen3.5-4b-instruct-q4_k_m`). This is the one case it's allowed to name
  the model.

Exit with Ctrl+C.

---

## 2. Memory persistence across restart

```bash
eva run
```
Say something memorable, e.g. "My favorite color is teal." Exit (Ctrl+C).

```bash
eva memory list                 # your conversation id is printed
eva memory show <conversation_id>
```
You should see both turns (yours and the assistant's reply) with
timestamps. Now **fully restart**:

```bash
eva run
```
Ask: "What's my favorite color?" — if the embedding model is installed
(`eva models list` shows `all-minilm-l6-v2-onnx` installed), the assistant
should recall it via semantic retrieval even though it's a new conversation
(memory search spans *all* conversations, not just the active one).

---

## 3. Semantic retrieval + context preview (no generation needed)

Start the server and engine:
```bash
eva serve &
curl -X POST http://127.0.0.1:8765/api/v1/engine/start
```
Preview exactly what the LLM would receive for a given input, without
spending a generation on it:
```bash
curl "http://127.0.0.1:8765/api/v1/memory/context-preview?text=what%20is%20my%20favorite%20color"
```
The response's `trace.retrieved_memories` should list the earlier "teal"
turn with a similarity score, and `messages` shows the exact prompt sent to
the LLM (identity block, technical-facts block, retrieved memory, recent
turns, current utterance — in that fixed order).

---

## 4. Conversation summaries

```bash
eva memory summarize <conversation_id>
```
This loads the LLM once (no ASR/TTS/audio) and prints a 2-4 sentence
summary. Confirm the originals are untouched:
```bash
eva memory show <conversation_id>   # all original turns still present
```

---

## 5. Persona switching (measurable effect on responses)

```bash
eva personas list                       # see all built-ins + custom, * = active
eva personas use technical
eva run
```
Ask a question, e.g. "Explain how a neural network works." Exit, then:
```bash
eva personas use creative
eva run
```
Ask the *same* question. The `technical` persona should answer precisely
and step-by-step; `creative` should be more imaginative/exploratory — a
directly observable difference from the same prompt.

Create a custom persona:
```bash
eva personas create --id pirate --name Pirate \
  --prompt "Speak like a pirate at all times." --tone boisterous
eva personas use pirate
eva run   # ask anything — replies should be pirate-flavored
eva personas use default   # switch back
eva personas delete pirate
```

---

## 6. User profiles (CRUD + influence on conversation)

```bash
eva users create --nickname Alice --units imperial --style casual
eva users list                          # note the generated id
eva users activate <user_id>
eva profile show                        # quick view of the active profile
eva run
```
Ask "What's my name?" or a temperature/distance question — the assistant
should address you as Alice and use imperial units (profile preferences are
injected into the system prompt).

```bash
eva users edit <user_id> --nickname Al
eva users delete <user_id>
```

---

## 7. Voice selection (persists across restart)

```bash
eva voices list                         # requires the TTS model installed
eva voices preview af_heart             # writes a WAV file, prints its path
eva voices use af_heart
```
Restart and confirm the choice stuck:
```bash
eva voices list                         # '*' should now mark af_heart
eva run                                 # spoken replies use the new voice
```

---

## 8. Memory management verbs

```bash
eva memory search "teal"
eva memory pin <turn_id>
eva memory favorite <turn_id>
eva memory archive <conversation_id>
eva memory archive <conversation_id> --unset     # restore
eva memory export --out snapshot.json
eva memory import snapshot.json
eva memory forget <turn_id>
eva memory delete-conversation <conversation_id>
eva memory delete-all --yes             # destructive — wipes everything
eva memory stats
```

---

## 9. REST API walkthrough

```bash
eva serve
```
In another terminal:
```bash
curl http://127.0.0.1:8765/api/v1/health
curl -X POST http://127.0.0.1:8765/api/v1/engine/start

curl http://127.0.0.1:8765/api/v1/personas
curl http://127.0.0.1:8765/api/v1/users
curl -X POST http://127.0.0.1:8765/api/v1/users \
  -H "Content-Type: application/json" \
  -d '{"nickname": "Bea", "units": "metric"}'
curl -X POST http://127.0.0.1:8765/api/v1/users/<id>/activate

curl -X POST http://127.0.0.1:8765/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "teal", "limit": 5}'
curl http://127.0.0.1:8765/api/v1/memory/stats

curl http://127.0.0.1:8765/api/v1/voices
curl -X POST http://127.0.0.1:8765/api/v1/voices/af_heart/preview \
  -H "Content-Type: application/json" -d '{"phrase": "Testing"}' \
  --output preview.pcm   # raw 16 kHz mono int16 PCM, no container
```
Every response should be JSON (or raw PCM for the voice preview) — full
endpoint list in [API.md](API.md). Errors follow
`{"detail": ..., "error_type": ...}`.

---

## 10. WebSocket event stream

```bash
python -c "
import asyncio, websockets, json
async def main():
    async with websockets.connect('ws://127.0.0.1:8765/api/v1/ws') as ws:
        async for msg in ws:
            print(json.loads(msg))
asyncio.run(main())
"
```
You should immediately receive `{"type": "snapshot", "data": {...}}`, then
live events (`PartialTranscript`, `LlmToken`, `TtsAudioReady`, ...) as you
talk to `eva run` or trigger conversation activity through the API.

---

## 11. Raw SQLite inspection

```bash
sqlite3 "$(eva diagnose | grep Conversations | awk '{print $2}')/memory.db"
```
```sql
.tables
-- conversations, turns, embeddings, summaries, user_profiles, schema_migrations
SELECT id, speaker, text FROM turns ORDER BY id DESC LIMIT 5;
SELECT id, nickname, active FROM user_profiles;
SELECT COUNT(*) FROM embeddings;
```

---

## 12. Diagnostics

```bash
eva diagnose                            # static config/hardware report
curl http://127.0.0.1:8765/api/v1/diagnostics | python -m json.tool
```
Confirm the JSON includes `active_persona_id`, `active_profile_id`,
`active_voice`, and the `memory_*` fields reflecting current state.

---

## 13. Restart persistence checklist

After steps 5-7 above, fully restart (`eva run` or `eva serve` +
`engine/start` again) and re-check without changing anything:

- [ ] `eva personas list` still shows the same `*` active persona
- [ ] `eva profile show` still shows the same active user
- [ ] `eva voices list` still shows the same `*` active voice
- [ ] `eva memory list` still shows every prior conversation
- [ ] The startup banner (`eva run`) prints Persona / User profile / Voice /
      Memory counts matching the above

---

## 14. Web UI (M5)

The React web UI is a pure client of the same REST/WebSocket API sections
1–12 already exercise — nothing here is a separate code path. Build it once,
then either serve it from the backend or run it against a dev server.

### 14.0 Build and serve

```bash
cd web
npm ci
npm run build            # produces web/dist/
cd ..
eva serve --open         # opens http://127.0.0.1:8765/ once the build is found
```
Or, for live-reload frontend development against a running backend:
```bash
eva serve                # terminal 1
cd web && npm run dev    # terminal 2 — http://localhost:5173, proxies /api
```

### 14.1 Dashboard

Start the engine from the header button. Confirm live updates with **no
page refresh**: assistant state pill changes (idle → listening → thinking →
speaking) as you talk to `eva run`-equivalent activity (or trigger via
`POST /conversation` activity through another client), microphone level bar
moves, active models/persona/profile/voice match `eva diagnose`, memory
stats match `eva memory stats`, and latency numbers appear after one turn.

### 14.2 Conversation

Have a conversation (via the engine — any client that drives it, since audio
stays server-side per ADR-007). Confirm: partial transcript appears in
italics and is replaced by the final one, the assistant's reply streams
token-by-token, an interrupted turn shows an "— interrupted —" marker,
**Export** downloads a JSON file matching `eva conversation export`-shape
data, **Import** re-loads it, **Clear** empties the view, and the search box
filters the visible transcript.

**Auto-scroll behavior:** while a reply streams, the view stays pinned to
the bottom. Scroll up mid-stream to re-read — the view must *stop*
following (it should not yank you back down on every token). Scroll back to
the bottom and following resumes.

### 14.2a Markdown rendering & speech (ADR-024)

Ask the assistant something that elicits formatting (e.g. "show me a Python
hello-world and a two-row table of two models"). Verify **rendering** in the
assistant bubble:

- [ ] `**bold**` shows as **bold**, not literal asterisks
- [ ] `*italic*` / `_italic_` shows as italic
- [ ] `# Heading` shows as a heading, not a literal `#`
- [ ] `` `inline code` `` shows in a monospace pill
- [ ] a fenced ` ``` ` block shows as a code box **with a working Copy button**
- [ ] `> quote` shows as an indented blockquote
- [ ] numbered and bulleted lists show as real lists
- [ ] a GFM table shows as a bordered table (scrolls horizontally if wide)
- [ ] links are clickable and open in a new tab
- [ ] raw HTML in a reply (e.g. an `<img onerror=…>`) is shown/escaped, never
      executed — no injected element appears in the DOM

Verify **speech**: listen to the same reply (or check what reaches TTS). The
assistant must say "Edge Voice Assistant", **not** "asterisk asterisk Edge
Voice Assistant"; it must not read backticks, table pipes, heading hashes,
or link URLs; and it must **skip** fenced code block contents entirely
(the code is on screen, not spoken).

Verify **canonical storage** (Markdown preserved everywhere but the two
presentation layers): after such a reply, `eva memory search` / the Memory
page / `GET /api/v1/conversation/export` all still contain the raw Markdown
(`**`, fences, etc.). Only the rendered bubble and the audio are transformed.

### 14.3 Memory

With the engine running: search returns results with score/match-reason
chips; pin/favorite/forget update immediately; a conversation can be
archived, restored, merged into another, summarized, and deleted; the
**Context inspector** for a text like "what's my favorite color?" shows the
exact single system message (identity + technical facts + retrieved
memories + summary, all one block — ADR-021 Amendment 2) followed by
strictly alternating user/assistant messages, matching what
`eva memory show`/`GET /memory/context-preview` report.

### 14.4 Personas

Activate a different persona from the grid and confirm the Dashboard's
"Personalization" card updates. Create a custom persona, duplicate an
existing one, edit it, then delete it — confirm attempting to delete a
built-in is not offered as an option. The prompt-preview panel matches
`eva personas show <id>`.

### 14.5 User profiles

Create a profile, activate it, edit its nickname, export the list (JSON
download), delete it, then import it back from the downloaded file.

### 14.6 Models

Confirm every installed model shows correct provider/license/languages/
VRAM/RAM/size matching `eva models list`/`info`. Trigger a download (if any
model is missing) and watch the live progress bar driven by WebSocket
`ModelDownloadProgress` events — no polling, no page refresh needed.
Activate a different model and confirm the "takes effect on restart" notice.

### 14.7 Voices

With the engine running, search/filter by language and style, click
**Preview** and confirm audible playback (decoded from raw PCM via Web
Audio — no container format round-trip), then **Use** a voice and confirm
`eva voices list` shows it as active after the next restart.

### 14.8 Settings

Open a few different sections (Audio, Conversation, Memory, Appearance) and
confirm every field, description, and bound (min/max, default) matches
`eva config schema` for that section — nothing is hardcoded in the UI.
Toggle the theme and confirm it applies instantly; save an invalid value
(e.g. a temperature above the schema maximum) and confirm the field-level
error appears before anything is saved; **Reset all to defaults** restores
`eva config show`'s baseline.

### 14.9 Diagnostics

Confirm CPU/RAM/(GPU/VRAM if present) meters and sparklines move over a few
seconds, queue depths and dropped-frame counters match
`GET /api/v1/diagnostics`, and the event log fills as you interact with the
assistant (excluding the high-frequency `LlmToken` stream, which is
intentionally not logged here).

### 14.10 Plugins

If no plugins are installed, confirm the empty-state explanation appears
rather than a blank page. If any are installed, confirm enable/disable/
reload work and match `curl .../api/v1/plugins`.

### 14.11 Desktop shell

```bash
pip install -e ".[desktop]"
eva-desktop
```
Confirm a native window opens showing the same UI, and that closing the
window also stops the backend process (no orphaned `eva serve` left
running — check with your OS's process list).

### 14.12 Responsive & accessibility spot-check

Resize the browser to a narrow (mobile-width) viewport and confirm the
sidebar collapses to a horizontal scrollable bar rather than overlapping
content. Tab through a page using only the keyboard and confirm every
interactive element shows a visible focus ring. Toggle your OS's
"prefers-reduced-motion" setting (or the Settings page's "Reduced Motion")
and confirm the state-pill pulse animation stops.

---

## 15. Conversational evaluation (M5.2)

These checks judge *behavior*, not plumbing — run them against the real
model (`eva run`, or the web UI with the engine started). Small local models
vary run to run; judge the pattern, not one sample.

### 15.1 Conversational continuity

1. Say: *"Create a markdown table of two planets."* → expect a table.
2. Then say only: *"with rows and columns."*
   - [ ] The assistant extends/reproduces the **table** — it must not treat
         the fragment as a brand-new request.
3. Then: *"add a third one."*
   - [ ] A third planet appears; the topic carries.

### 15.2 Pronouns and interrupted thoughts

1. *"Tell me about the Eiffel Tower."* → answer.
2. *"How tall is it?"*
   - [ ] "it" resolves to the Eiffel Tower (≈330 m / 1,083 ft).
3. Start a sentence, pause mid-way so the VAD cuts you off, then continue
   with the rest in the next turn.
   - [ ] The reply treats both parts as one thought.

### 15.3 Helpfulness over literalness

- *"Act as a spreadsheet: sum 2, 4 and 6."*
  - [ ] The answer is **12** — never "I am not a spreadsheet."
- *"Pretend you're a travel agent and plan my Saturday."*
  - [ ] It plans the Saturday; no identity disclaimer first.

### 15.4 Capability honesty (image messaging)

- *"Can you look at this image for me?"*
  - [ ] The answer is **build-scoped** — "not enabled in this current
        build" / "I can't view images in this build" — and ideally offers an
        alternative.
  - [ ] It never claims image understanding is permanently impossible.

### 15.5 Natural identity

- *"What's a good breakfast?"* (any ordinary question)
  - [ ] The reply does **not** mention "Edge Voice Assistant".
- *"Who are you?"*
  - [ ] Now it names itself — once, naturally.
- *"Which LLM powers you?"*
  - [ ] Honest technical answer (the M4 contract still holds).

### 15.6 Personas sound different

Ask the SAME question (e.g. *"Explain how photosynthesis works"*) under
each persona (`eva personas use <id>` or the Personas page + engine
restart):

- [ ] **default** — warm, natural, a few conversational sentences
- [ ] **minimal** — one to two terse sentences, no preamble
- [ ] **teacher** — an everyday analogy + step-by-step build-up
- [ ] **technical** — precise terminology, no analogies, trade-offs
- [ ] **professional** — direct answer first, then ordered points
- [ ] **friendly** — upbeat, encouraging
- [ ] **creative** — vivid, offers an unexpected angle

The differences must be obvious without knowing which persona is active.

### 15.7 Memory recall feels natural

1. In one session: *"My favorite color is teal."* Restart.
2. New session: *"What color should I paint my office?"*
   - [ ] Teal comes up naturally — not "according to my records" or
         "based on earlier context, your favorite color is teal."

### 15.8 Long conversations

Hold a 15+ turn conversation, then reference something from the first few
turns.

- [ ] The assistant still has it (20-turn window) or recalls it via memory.
- [ ] Replies don't degrade into repetition or forget the persona's voice.

### 15.9 System information & permissions (M5.3)

With default permissions (Settings → Permissions):

- *"What time is it?"* → [ ] the actual current local time.
- *"What day is it today?"* → [ ] correct date.
- *"How much RAM does this machine have?"* / *"What GPU do I have?"* →
  [ ] real values matching `eva diagnose`.
- Turn OFF **Date/Time** in Settings → Permissions, save, restart the
  engine, ask the time again → [ ] the assistant says the *permission is
  not granted* — not that it can never know the time.

### 15.10 Typed conversation (composer, M5.3)

On the Conversation page with the engine running:

- [ ] Type a message, press **Enter** → it appears as your turn, the reply
      streams in AND is spoken aloud (same pipeline as voice).
- [ ] **Shift+Enter** inserts a newline instead of sending.
- [ ] With the engine stopped, the composer is disabled with an explanatory
      placeholder.
- [ ] The **+** menu offers attach image/document/screenshot; each explains
      it is not available in this build.
- [ ] Drag a file onto the composer (or paste an image) → a placeholder
      chip appears, removable, labeled as not processed in this build;
      sending still delivers the text.
- [ ] The **Mode** selector next to the engine controls shows *Offline
      (local)* selected; *Online* is visibly disabled.

### 15.11 Markdown-to-speech hardening (M5.3)

Elicit a reply with heavy formatting (bold mid-sentence, a code block, a
table). Listen:

- [ ] No "asterisk", "underscore", "backtick", or "pipe" is ever spoken —
      including when a bold phrase spans a sentence boundary
      (previously "**Generate**" could leak as "asterisk asterisk Generate").
- [ ] HTML entities (`&amp;` etc.) are spoken as their characters ("and"
      context), not as "ampersand a-m-p semicolon".

### 15.12 Ambiguity handling

- *"Make it shorter."* (with nothing plausible to shorten in view)
  - [ ] It either makes the most helpful assumption or asks ONE short
        clarifying question — not a refusal, not an essay about ambiguity.

---

## Naming note

`eva profiles` (plural) is the **hardware/model preset** command (Balanced,
Fast, ...) from earlier milestones — unrelated to `eva profile` (singular)
and `eva users`, which are the M4 **user profile** (personal
nickname/preferences) commands. Don't confuse the two.
