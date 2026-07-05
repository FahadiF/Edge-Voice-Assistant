# Manual Testing Guide — M4 (Memory, Personalization & Intelligence)

This guide lets someone with no source-code access validate every M4
capability end-to-end: memory persistence, semantic retrieval, summaries,
personas, user profiles, voices, the REST API, the WebSocket stream, raw
SQLite inspection, and diagnostics. Every step uses either the `eva` CLI or
plain `curl`/`sqlite3` — nothing here requires reading the codebase.

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

## Naming note

`eva profiles` (plural) is the **hardware/model preset** command (Balanced,
Fast, ...) from earlier milestones — unrelated to `eva profile` (singular)
and `eva users`, which are the M4 **user profile** (personal
nickname/preferences) commands. Don't confuse the two.
