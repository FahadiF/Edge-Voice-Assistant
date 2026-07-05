/** Models (Part 8): catalog browser with live download progress over WS. */

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { models } from "../api/endpoints";
import type { ModelCard, ModelKind } from "../api/types";
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

function ModelRow({ model }: { model: ModelCard }) {
  const queryClient = useQueryClient();
  const download = useWsStore((s) => s.downloads[model.id]);
  const clearDownload = useWsStore((s) => s.clearDownload);
  const [confirmRemove, setConfirmRemove] = useState(false);

  // Refresh the catalog when a download for this model finishes.
  useEffect(() => {
    if (download?.status === "completed") {
      toast("success", `${model.name} installed`);
      clearDownload(model.id);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    } else if (download?.status === "failed") {
      toast("error", `${model.name} download failed: ${download.error}`);
      clearDownload(model.id);
    }
  }, [download?.status, model.id, model.name, clearDownload, queryClient, download?.error]);

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
          {model.active && <span className="chip chip-accent">active</span>}
          {model.installed && !model.active && <span className="chip chip-success">installed</span>}
          {!model.compatible && (
            <span className="chip chip-warning" title={model.compatibility_notes}>
              may not fit
            </span>
          )}
        </div>
      </div>
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
            Download
          </button>
        )}
        {model.managed_by === "engine" && !model.installed && (
          <span className="field-help">auto-downloads on first use</span>
        )}
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

export function Models() {
  const catalog = useQuery({ queryKey: ["models"], queryFn: () => models.list() });

  if (catalog.isLoading) return <p>Loading model catalog…</p>;
  if (!catalog.data) return <p role="alert">Could not load the model catalog.</p>;

  const kinds: ModelKind[] = ["llm", "asr", "tts", "vad", "embedding"];

  return (
    <div>
      <h1>Models</h1>
      {kinds.map((kind) => {
        const group = catalog.data.filter((m) => m.kind === kind);
        if (group.length === 0) return null;
        return (
          <Card key={kind} title={KIND_LABELS[kind]}>
            <div className="grid-3">
              {group.map((model) => (
                <ModelRow key={model.id} model={model} />
              ))}
            </div>
          </Card>
        );
      })}
      {catalog.data.length === 0 && <EmptyState>The model catalog is empty.</EmptyState>}
    </div>
  );
}
