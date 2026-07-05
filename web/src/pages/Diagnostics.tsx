/**
 * Diagnostics (Part 11): live resource/pipeline/queue view, merging the
 * WebSocket snapshot (pushed) with a periodic GET /diagnostics poll as a
 * fallback for the fields the event stream doesn't push every tick (e.g.
 * while idle with no turns in flight).
 */

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { diagnostics, system } from "../api/endpoints";
import { useWsStore } from "../ws/store";
import { Card, EmptyState, Meter } from "../components/common";
import "./diagnostics.css";

const SPARKLINE_POINTS = 40;

function Sparkline({ values, max }: { values: number[]; max: number }) {
  const width = 160;
  const height = 32;
  const points = values
    .map((v, i) => {
      const x = (i / Math.max(1, SPARKLINE_POINTS - 1)) * width;
      const y = height - (Math.min(v, max) / max) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} className="sparkline" role="img" aria-hidden="true">
      <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
    </svg>
  );
}

export function Diagnostics() {
  const snapshot = useWsStore((s) => s.snapshot);
  const eventLog = useWsStore((s) => s.eventLog);
  const poll = useQuery({
    queryKey: ["diagnostics"],
    queryFn: diagnostics.snapshot,
    refetchInterval: 3000,
    retry: false,
  });
  const hardware = useQuery({ queryKey: ["hardware"], queryFn: system.hardware });

  const data = snapshot ?? poll.data;

  const [cpuHistory, setCpuHistory] = useState<number[]>([]);
  const [ramHistory, setRamHistory] = useState<number[]>([]);
  const lastSampledAt = useRef(0);

  useEffect(() => {
    if (!data) return;
    const now = Date.now();
    if (now - lastSampledAt.current < 900) return; // sample at most ~1/s
    lastSampledAt.current = now;
    setCpuHistory((prev) => [...prev.slice(-(SPARKLINE_POINTS - 1)), data.resources.cpu_percent]);
    setRamHistory((prev) => [
      ...prev.slice(-(SPARKLINE_POINTS - 1)),
      (data.resources.ram_used_mb / data.resources.ram_total_mb) * 100,
    ]);
  }, [data]);

  if (!data) {
    return (
      <div>
        <h1>Diagnostics</h1>
        <EmptyState>Waiting for data — start the engine for full metrics.</EmptyState>
      </div>
    );
  }

  return (
    <div>
      <h1>Diagnostics</h1>
      <div className="grid-2">
        <Card title="Resources">
          <Meter label="CPU" value={data.resources.cpu_percent} max={100} unit="%" />
          <Sparkline values={cpuHistory} max={100} />
          <Meter
            label="RAM"
            value={data.resources.ram_used_mb}
            max={data.resources.ram_total_mb}
            unit="MB"
          />
          <Sparkline values={ramHistory} max={100} />
          {data.resources.vram_total_mb !== null && (
            <Meter
              label="VRAM"
              value={data.resources.vram_used_mb ?? 0}
              max={data.resources.vram_total_mb}
              unit="MB"
            />
          )}
          {data.resources.gpu_percent !== null && (
            <Meter label="GPU" value={data.resources.gpu_percent} max={100} unit="%" />
          )}
        </Card>

        <Card title="Hardware">
          {hardware.data ? (
            <table>
              <tbody>
                <tr><th scope="row">Tier</th><td>{hardware.data.tier_name}</td></tr>
                <tr><th scope="row">CPU</th><td>{hardware.data.cpu}</td></tr>
                <tr><th scope="row">GPU</th><td>{hardware.data.gpu ?? "none detected"}</td></tr>
                <tr><th scope="row">VRAM</th><td>{hardware.data.vram_mb} MB</td></tr>
                <tr><th scope="row">RAM</th><td>{hardware.data.ram_mb} MB</td></tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>Loading…</EmptyState>
          )}
        </Card>

        <Card title="Pipeline">
          <table>
            <tbody>
              <tr><th scope="row">State</th><td>{data.state}</td></tr>
              <tr><th scope="row">Epoch</th><td>{data.epoch}</td></tr>
              <tr><th scope="row">Playback active</th><td>{data.playback_active ? "yes" : "no"}</td></tr>
              <tr><th scope="row">Playback queued</th><td>{data.playback_queued_seconds.toFixed(2)} s</td></tr>
              <tr><th scope="row">Input level</th><td>{data.input_level_dbfs.toFixed(1)} dBFS</td></tr>
            </tbody>
          </table>
        </Card>

        <Card title="Queues">
          <table>
            <tbody>
              <tr><th scope="row">Pending audio events</th><td>{data.pending_audio_events}</td></tr>
              <tr><th scope="row">Capture ring depth</th><td>{data.capture_ring_depth}</td></tr>
              <tr><th scope="row">Capture frames dropped</th><td>{data.capture_frames_dropped}</td></tr>
              <tr><th scope="row">Token queue depth</th><td>{data.token_queue_depth}</td></tr>
              <tr><th scope="row">Sentence queue depth</th><td>{data.sentence_queue_depth}</td></tr>
            </tbody>
          </table>
        </Card>

        <Card title="Timing (last turn)">
          {data.last_turn ? (
            <table>
              <tbody>
                <tr><th scope="row">ASR</th><td>{data.last_turn.asr_ms} ms</td></tr>
                <tr><th scope="row">Time to first token</th><td>{data.last_turn.ttft_ms} ms</td></tr>
                <tr><th scope="row">LLM</th><td>{data.last_turn.llm_ms} ms ({data.last_turn.tokens} tokens)</td></tr>
                <tr><th scope="row">TTS first chunk</th><td>{data.last_turn.tts_first_ms} ms</td></tr>
                <tr><th scope="row">Time to first audio</th><td>{data.last_turn.ttfa_ms} ms</td></tr>
                <tr><th scope="row">Total</th><td>{data.last_turn.total_ms} ms</td></tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>No completed turns yet</EmptyState>
          )}
        </Card>

        <Card title="Barge-in">
          <table>
            <tbody>
              <tr><th scope="row">Count (session)</th><td>{data.barge_in_count}</td></tr>
              <tr><th scope="row">Last latency</th><td>{data.last_barge_in_latency_ms ?? "—"} ms</td></tr>
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Event log">
        {eventLog.length === 0 ? (
          <EmptyState>No events yet this session.</EmptyState>
        ) : (
          <div className="event-log">
            {[...eventLog].reverse().map((e, i) => (
              <div key={i} className="event-row">
                <span className="event-time">{new Date(e.at).toLocaleTimeString()}</span>
                <span className="chip">{e.type}</span>
                <span className="event-summary">{e.summary}</span>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
