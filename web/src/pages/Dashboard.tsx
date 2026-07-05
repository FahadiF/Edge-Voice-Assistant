/** Dashboard (Part 3): live overview fed by the WebSocket snapshot/events. */

import { useQuery } from "@tanstack/react-query";
import { engine } from "../api/endpoints";
import { useWsStore } from "../ws/store";
import { Card, EmptyState, Meter, StatusPill, formatBytes } from "../components/common";

function MicLevel({ dbfs }: { dbfs: number }) {
  // -60 dBFS (quiet) .. 0 dBFS (max) mapped to 0..100%.
  const pct = Math.max(0, Math.min(100, ((dbfs + 60) / 60) * 100));
  return (
    <div>
      <div className="meter-head">
        <span>Microphone level</span>
        <span className="meter-value">{dbfs <= -119 ? "silent" : `${dbfs.toFixed(0)} dBFS`}</span>
      </div>
      <div className="meter-track" aria-hidden="true">
        <div className="meter-fill" style={{ width: `${pct}%`, background: "var(--success)" }} />
      </div>
    </div>
  );
}

export function Dashboard() {
  const snapshot = useWsStore((s) => s.snapshot);
  const pipelineState = useWsStore((s) => s.pipelineState);
  const status = useQuery({ queryKey: ["engine-status"], queryFn: engine.status });
  const readiness = useQuery({
    queryKey: ["engine-readiness"],
    queryFn: engine.readiness,
    enabled: !(status.data?.running ?? true),
  });

  const running = status.data?.running ?? false;

  return (
    <div>
      <h1>Dashboard</h1>
      <div className="grid-2">
        <Card title="Assistant">
          <p>
            <StatusPill state={pipelineState} />
          </p>
          {snapshot ? (
            <>
              <MicLevel dbfs={snapshot.input_level_dbfs} />
              <p style={{ color: "var(--text-muted)" }}>
                Turn epoch {snapshot.epoch} · {snapshot.turns_completed} turn(s) completed
                {snapshot.playback_active ? " · playing audio" : ""}
              </p>
            </>
          ) : (
            <EmptyState>Waiting for the live connection…</EmptyState>
          )}
        </Card>

        <Card title="Engine">
          <p>
            Status: <strong>{running ? "running" : "stopped"}</strong>
          </p>
          {!running && readiness.data && !readiness.data.ready && (
            <div>
              <p style={{ color: "var(--warning)" }}>Setup incomplete:</p>
              <ul>
                {readiness.data.problems.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </div>
          )}
          {!running && readiness.data?.ready && (
            <p style={{ color: "var(--text-muted)" }}>
              Ready to start — use the button in the header.
            </p>
          )}
        </Card>

        <Card title="Active models">
          {snapshot ? (
            <table>
              <tbody>
                {Object.entries(snapshot.models).map(([kind, id]) => (
                  <tr key={kind}>
                    <th scope="row">{kind.toUpperCase()}</th>
                    <td>
                      <code>{id}</code>
                    </td>
                    <td style={{ color: "var(--text-muted)" }}>
                      {snapshot.devices[kind] ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState>No snapshot yet</EmptyState>
          )}
        </Card>

        <Card title="Personalization">
          {snapshot ? (
            <table>
              <tbody>
                <tr>
                  <th scope="row">Persona</th>
                  <td>{snapshot.active_persona_id}</td>
                </tr>
                <tr>
                  <th scope="row">User profile</th>
                  <td>{snapshot.active_profile_id ?? "none"}</td>
                </tr>
                <tr>
                  <th scope="row">Voice</th>
                  <td>{snapshot.active_voice}</td>
                </tr>
                <tr>
                  <th scope="row">Language</th>
                  <td>{snapshot.language}</td>
                </tr>
                <tr>
                  <th scope="row">Hardware profile</th>
                  <td>{snapshot.profile}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>No snapshot yet</EmptyState>
          )}
        </Card>

        <Card title="Memory">
          {snapshot ? (
            <table>
              <tbody>
                <tr>
                  <th scope="row">Semantic search</th>
                  <td>{snapshot.memory_enabled ? "enabled" : "keyword-only"}</td>
                </tr>
                <tr>
                  <th scope="row">Stored turns</th>
                  <td>{snapshot.memory_turn_count}</td>
                </tr>
                <tr>
                  <th scope="row">Embedded turns</th>
                  <td>{snapshot.memory_embedding_count}</td>
                </tr>
                <tr>
                  <th scope="row">Database size</th>
                  <td>{formatBytes(snapshot.memory_db_size_bytes)}</td>
                </tr>
                <tr>
                  <th scope="row">Last retrieval</th>
                  <td>
                    {snapshot.last_retrieval_ms !== null
                      ? `${snapshot.last_retrieval_ms} ms (top score ${snapshot.last_retrieval_score_top1?.toFixed(2) ?? "—"})`
                      : "—"}
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>No snapshot yet</EmptyState>
          )}
        </Card>

        <Card title="Latency (last turn)">
          {snapshot?.last_turn ? (
            <table>
              <tbody>
                <tr>
                  <th scope="row">Speech recognition</th>
                  <td>{snapshot.last_turn.asr_ms} ms</td>
                </tr>
                <tr>
                  <th scope="row">Time to first token</th>
                  <td>{snapshot.last_turn.ttft_ms} ms</td>
                </tr>
                <tr>
                  <th scope="row">Time to first audio</th>
                  <td>{snapshot.last_turn.ttfa_ms} ms</td>
                </tr>
                <tr>
                  <th scope="row">Total</th>
                  <td>{snapshot.last_turn.total_ms} ms</td>
                </tr>
                <tr>
                  <th scope="row">Barge-ins (session)</th>
                  <td>
                    {snapshot.barge_in_count}
                    {snapshot.last_barge_in_latency_ms !== null
                      ? ` (last: ${snapshot.last_barge_in_latency_ms} ms)`
                      : ""}
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <EmptyState>No completed turns yet</EmptyState>
          )}
        </Card>

        <Card title="System resources">
          {snapshot ? (
            <>
              <Meter
                label="CPU"
                value={snapshot.resources.cpu_percent}
                max={100}
                unit="%"
              />
              <Meter
                label="RAM"
                value={snapshot.resources.ram_used_mb}
                max={snapshot.resources.ram_total_mb}
                unit="MB"
              />
              {snapshot.resources.vram_total_mb !== null && (
                <Meter
                  label="VRAM"
                  value={snapshot.resources.vram_used_mb ?? 0}
                  max={snapshot.resources.vram_total_mb}
                  unit="MB"
                />
              )}
              {snapshot.resources.gpu_percent !== null && (
                <Meter label="GPU" value={snapshot.resources.gpu_percent} max={100} unit="%" />
              )}
            </>
          ) : (
            <EmptyState>No snapshot yet</EmptyState>
          )}
        </Card>
      </div>
    </div>
  );
}
