/** Models (Part 8): catalog browser with live download progress over WS. */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { models, system } from "../api/endpoints";
import type { HardwareSummary, ModelCard, ModelKind } from "../api/types";
import { useWsStore } from "../ws/store";
import { Card, ConfirmDialog, EmptyState, ProgressBar, toast } from "../components/common";
import "./models.css";

const KIND_LABELS: Record<ModelKind, string> = {
  llm: "Language Models",
  asr: "Speech Recognition",
  tts: "Speech Synthesis",
  vad: "Voice Detection",
  embedding: "Embeddings (Memory)",
};

/** Status vocabulary shown on every model card.
 *
 * Only `active`, `installed`, and `available` can be derived from what the
 * catalog knows today. `ready` (verified usable) and `corrupted` are defined
 * here so the colour language is settled, but nothing emits them yet — they
 * light up when model verification lands (ADR-034 / M1). Deriving them from
 * guesses now would mean showing a green "ready" badge for a model we have
 * never confirmed loads, which is exactly the failure this page had.
 */
type StatusTone = "ready" | "active" | "installed" | "corrupted" | "available";

interface ModelStatusBadge {
  label: string;
  tone: StatusTone;
  title: string;
}

function statusOf(model: ModelCard): ModelStatusBadge {
  if (model.active) {
    return { label: "Active", tone: "active", title: "In use by the engine" };
  }
  if (model.installed) {
    return { label: "Installed", tone: "installed", title: "On disk, not currently active" };
  }
  return { label: "Available", tone: "available", title: "Not downloaded yet — you can install it" };
}

/** "460 MB" / "1.6 GB" — download sizes are catalog data, never hardcoded. */
function formatSize(mb: number): string {
  if (mb <= 0) return "";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

/** Actionable fit message: what the model needs vs what this machine has.
 * Returns null when the model fits, or when we cannot state real numbers —
 * an invented figure would be worse than the generic warning it replaced. */
function fitWarning(model: ModelCard, hardware: HardwareSummary | undefined): string[] | null {
  if (model.compatible || model.vram_mb === 0) return null;
  const lines = [`Requires ~${formatSize(model.vram_mb)} VRAM`];
  if (hardware && hardware.vram_mb > 0) {
    lines.push(`Detected GPU: ${formatSize(hardware.vram_mb)}`);
  } else if (hardware) {
    lines.push("No GPU detected");
  }
  return lines;
}

function ModelRow({ model, hardware }: { model: ModelCard; hardware?: HardwareSummary }) {
  const queryClient = useQueryClient();
  const download = useWsStore((s) => s.downloads[model.id]);
  const clearDownload = useWsStore((s) => s.clearDownload);
  const [confirmRemove, setConfirmRemove] = useState(false);

  // Refresh the catalog when a download for this model finishes.
  useEffect(() => {
    if (!download) return;
    if (download.status === "completed") {
      toast("success", `${model.name} installed`);
      clearDownload(model.id);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    } else if (download.status === "failed") {
      toast("error", `${model.name} download failed: ${download.error}`);
      clearDownload(model.id);
    }
  }, [download, model.id, model.name, clearDownload, queryClient]);

  const startDownload = useMutation({
    mutationFn: () => models.download(model.id),
    onSuccess: (r) => {
      if (r.status === "not_applicable") {
        toast("info", "This model is downloaded automatically by its engine on first use");
      } else if (r.status === "already_running") {
        toast("info", "Download already in progress");
      } else {
        toast("info", `Downloading ${model.name}…`);
      }
    },
    onError: (e) => toast("error", e.message),
  });

  const activate = useMutation({
    mutationFn: () => models.activate(model.id),
    onSuccess: () => {
      toast("success", `${model.name} set as active ${model.kind.toUpperCase()} — takes effect on engine restart`);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onError: (e) => toast("error", e.message),
  });

  const remove = useMutation({
    mutationFn: () => models.remove(model.id),
    onSuccess: () => {
      setConfirmRemove(false);
      toast("success", `${model.name} removed`);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onError: (e) => toast("error", e.message),
  });

  const status = statusOf(model);
  const fit = fitWarning(model, hardware);
  // Download/Remove only exist for manager-managed models. Rather than render
  // buttons that cannot work, say who does own the files — silently omitting
  // the controls just looks broken (M7 UX polish).
  const managedNote =
    model.managed_by === "engine"
      ? model.installed
        ? "managed by its engine — removal arrives with model lifecycle support"
        : "auto-downloads on first use"
      : model.managed_by === "bundled"
        ? "bundled with EVA"
        : "";

  return (
    <div className={`model-card ${model.active ? "model-active" : ""}`}>
      <div className="model-head">
        <div>
          <strong>{model.name}</strong>
          <div className="model-id">
            <code>{model.id}</code>
          </div>
        </div>
        <div className="model-badges">
          <span className={`status-badge status-${status.tone}`} title={status.title}>
            {status.label}
          </span>
        </div>
      </div>
      {model.recommendation && <p className="model-recommendation">{model.recommendation}</p>}
      {fit && (
        <p className="model-fit" role="note">
          {fit.map((line) => (
            <span key={line}>{line}</span>
          ))}
        </p>
      )}
      <dl className="model-facts">
        <div><dt>Provider</dt><dd>{model.provider || "—"}</dd></div>
        <div><dt>License</dt><dd>{model.license || "—"}</dd></div>
        <div><dt>Languages</dt><dd>{model.languages || "—"}</dd></div>
        {model.quantization && <div><dt>Quantization</dt><dd>{model.quantization}</dd></div>}
        {model.context_length !== null && (
          <div><dt>Context</dt><dd>{model.context_length.toLocaleString()} tokens</dd></div>
        )}
        <div><dt>VRAM</dt><dd>{model.vram_mb ? `${model.vram_mb} MB` : "CPU"}</dd></div>
        <div><dt>RAM</dt><dd>{model.ram_mb ? `${model.ram_mb} MB` : "—"}</dd></div>
        <div>
          <dt>Size</dt>
          <dd>{model.installed ? `${model.disk_usage_mb} MB on disk` : model.download_mb ? `${model.download_mb} MB download` : "—"}</dd>
        </div>
      </dl>
      {model.notes && <p className="model-notes">{model.notes}</p>}
      {download?.status === "downloading" && (
        <div className="model-download">
          <ProgressBar
            value={download.bytesDone}
            max={download.bytesTotal}
            label={`Downloading ${model.name}`}
          />
          <span className="model-download-file">{download.filename}</span>
        </div>
      )}
      <div className="model-actions">
        {!model.installed && model.managed_by === "manager" && !download && (
          <button className="primary" onClick={() => startDownload.mutate()} disabled={startDownload.isPending}>
            {model.download_mb ? `Download (${formatSize(model.download_mb)})` : "Download"}
          </button>
        )}
        {managedNote && <span className="field-help">{managedNote}</span>}
        {model.installed && !model.active && (
          <button onClick={() => activate.mutate()} disabled={activate.isPending}>
            Set active
          </button>
        )}
        {model.installed && model.managed_by === "manager" && (
          <button className="danger" onClick={() => setConfirmRemove(true)}>
            Remove
          </button>
        )}
      </div>
      <ConfirmDialog
        open={confirmRemove}
        title={`Remove ${model.name}?`}
        body="The model files are deleted from disk. You can re-download it later."
        confirmLabel="Remove"
        danger
        onConfirm={() => remove.mutate()}
        onCancel={() => setConfirmRemove(false)}
      />
    </div>
  );
}

/** Active first, then installed, then the rest — so the two questions people
 * actually arrive with ("what am I running?", "what do I already have?") are
 * answered by the top of each group instead of by scanning badges. */
const STATUS_ORDER: Record<StatusTone, number> = {
  active: 0,
  ready: 1,
  installed: 2,
  corrupted: 3,
  available: 4,
};

export function Models() {
  const catalog = useQuery({ queryKey: ["models"], queryFn: () => models.list() });
  // Shares the cache with the Dashboard's query — no extra request.
  const hardware = useQuery({ queryKey: ["hardware"], queryFn: system.hardware });

  if (catalog.isLoading) return <p>Loading model catalog…</p>;
  if (!catalog.data) return <p role="alert">Could not load the model catalog.</p>;

  const kinds: ModelKind[] = ["llm", "asr", "tts", "vad", "embedding"];

  return (
    <div>
      <h1>Models</h1>
      {hardware.data && (
        <p className="models-hardware">
          Detected: {hardware.data.gpu ?? "no GPU"}
          {hardware.data.vram_mb > 0 && ` · ${formatSize(hardware.data.vram_mb)} VRAM`}
          {` · ${formatSize(hardware.data.ram_mb)} RAM`} · tier{" "}
          <code>{hardware.data.tier}</code>
        </p>
      )}
      {kinds.map((kind) => {
        const group = catalog.data
          .filter((m) => m.kind === kind)
          .sort(
            (a, b) =>
              STATUS_ORDER[statusOf(a).tone] - STATUS_ORDER[statusOf(b).tone] ||
              a.name.localeCompare(b.name),
          );
        if (group.length === 0) return null;
        const installedCount = group.filter((m) => m.installed).length;
        return (
          <Card key={kind} title={`${KIND_LABELS[kind]} — ${installedCount}/${group.length} installed`}>
            <div className="grid-3">
              {group.map((model) => (
                <ModelRow key={model.id} model={model} hardware={hardware.data} />
              ))}
            </div>
          </Card>
        );
      })}
      {catalog.data.length === 0 && <EmptyState>The model catalog is empty.</EmptyState>}
    </div>
  );
}
