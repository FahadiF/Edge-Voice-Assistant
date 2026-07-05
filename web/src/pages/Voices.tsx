/**
 * Voices (Part 9): browse/search/preview/select TTS voices.
 *
 * Preview decodes the backend's raw 16 kHz mono int16 PCM with the Web
 * Audio API (ADR-023 — no container format, no backend change). Requires a
 * running engine (voices are discovered from the loaded TTS engine).
 */

import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { engine, settings as settingsApi, voices } from "../api/endpoints";
import { ApiError } from "../api/client";
import { Card, EmptyState, toast } from "../components/common";
import "./models.css";

const SAMPLE_RATE = 16_000;

async function playPcm(buffer: ArrayBuffer, ctxRef: { current: AudioContext | null }): Promise<void> {
  const samples = new Int16Array(buffer);
  if (samples.length === 0) throw new Error("Empty audio");
  ctxRef.current ??= new AudioContext();
  const ctx = ctxRef.current;
  const audioBuffer = ctx.createBuffer(1, samples.length, SAMPLE_RATE);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) {
    channel[i] = samples[i] / 32768;
  }
  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.start();
}

export function Voices() {
  const queryClient = useQueryClient();
  const status = useQuery({ queryKey: ["engine-status"], queryFn: engine.status });
  const running = status.data?.running ?? false;

  const voiceList = useQuery({
    queryKey: ["voices"],
    queryFn: voices.list,
    enabled: running,
    retry: false,
  });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: settingsApi.get });
  const activeVoice = settingsQuery.data?.tts.voice;

  const [search, setSearch] = useState("");
  const [language, setLanguage] = useState("");
  const [style, setStyle] = useState("");
  const [previewing, setPreviewing] = useState<string | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);

  const filtered = useMemo(() => {
    const list = voiceList.data ?? [];
    const q = search.trim().toLowerCase();
    return list.filter(
      (v) =>
        (!q || v.id.toLowerCase().includes(q) || v.display_name.toLowerCase().includes(q)) &&
        (!language || v.language === language) &&
        (!style || v.style_tag === style),
    );
  }, [voiceList.data, search, language, style]);

  const languages = useMemo(
    () => [...new Set((voiceList.data ?? []).map((v) => v.language))].sort(),
    [voiceList.data],
  );
  const styles = useMemo(
    () => [...new Set((voiceList.data ?? []).map((v) => v.style_tag).filter(Boolean))].sort(),
    [voiceList.data],
  );

  const preview = async (voiceId: string) => {
    setPreviewing(voiceId);
    try {
      const pcm = await voices.preview(voiceId);
      await playPcm(pcm, audioCtx);
    } catch (e) {
      toast("error", `Preview failed: ${(e as Error).message}`);
    } finally {
      setPreviewing(null);
    }
  };

  const select = useMutation({
    mutationFn: (voiceId: string) => settingsApi.patch({ tts: { voice: voiceId } }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["settings"], updated);
      toast("success", `Voice set to ${updated.tts.voice} — takes effect on engine restart`);
    },
    onError: (e) => toast("error", e.message),
  });

  if (!running) {
    return (
      <div>
        <h1>Voices</h1>
        <EmptyState>
          Voices are discovered from the loaded speech engine — start the engine (header
          button) to browse and preview them.
        </EmptyState>
      </div>
    );
  }

  return (
    <div>
      <h1>Voices</h1>
      <Card>
        <div className="voice-filters">
          <input
            type="search"
            placeholder="Search voices…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search voices"
          />
          <select value={language} onChange={(e) => setLanguage(e.target.value)} aria-label="Filter by language">
            <option value="">All languages</option>
            {languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
          <select value={style} onChange={(e) => setStyle(e.target.value)} aria-label="Filter by style">
            <option value="">All styles</option>
            {styles.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        {voiceList.isError && (
          <p role="alert">
            {voiceList.error instanceof ApiError && voiceList.error.status === 409
              ? "Engine not running."
              : `Could not load voices: ${voiceList.error?.message}`}
          </p>
        )}
        <div className="voice-grid">
          {filtered.map((voice) => (
            <div
              key={voice.id}
              className={`voice-card ${voice.id === activeVoice ? "voice-active" : ""}`}
            >
              <div>
                <strong>{voice.display_name}</strong>{" "}
                {voice.id === activeVoice && <span className="chip chip-accent">active</span>}
                <div className="model-id">
                  <code>{voice.id}</code>
                </div>
              </div>
              <div>
                <span className="chip">{voice.language}</span>
                {voice.style_tag && <span className="chip">{voice.style_tag}</span>}
              </div>
              <div className="model-actions">
                <button onClick={() => void preview(voice.id)} disabled={previewing !== null}>
                  {previewing === voice.id ? "Playing…" : "Preview"}
                </button>
                {voice.id !== activeVoice && (
                  <button className="primary" onClick={() => select.mutate(voice.id)}>
                    Use
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
        {filtered.length === 0 && voiceList.data && (
          <EmptyState>No voices match the current filters.</EmptyState>
        )}
      </Card>
    </div>
  );
}
