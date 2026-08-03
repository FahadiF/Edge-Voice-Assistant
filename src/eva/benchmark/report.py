"""Benchmark reporting (M8): aggregate `TurnMetrics` into one canonical
report, then project it to JSON, Markdown, or HTML.

**This module collects nothing.** Every number it prints was already measured
elsewhere — by the live orchestrator (`MetricsCollector`, Batch 7) or by
`PipelineBenchmark`. Both hand over `Sequence[TurnMetrics]`, so the aggregator
never learns which produced it: one record type in, one report model out.
That is what makes the same report renderable for a synthetic benchmark run
and for a real conversation.

    MetricsCollector.turns  ─┐
                             ├─► Sequence[TurnMetrics] ─► BenchmarkReport ─┬─► JSON
    PipelineBenchmark rounds ┘                                             ├─► Markdown
                                                                           └─► HTML

`BenchmarkReport` is the single intermediate representation: the three
renderers are projections *of the model*, never of the raw samples, so they
can never disagree about what a run measured.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, ConfigDict

from eva.core.provenance import Environment
from eva.metrics.turn import TurnMetrics

#: Rendered wherever a section has no data *by construction* rather than
#: because the run was fast. See `StageStats.measured`.
NOT_MEASURED = "not measured"

#: Why the WER section is structurally present but always empty today.
WER_DEFERRAL_NOTE = (
    "Speech-recognition accuracy (WER) requires the recorded fixture corpus, "
    "which does not exist yet. No WER figure is reported rather than a "
    "synthetic one, which would not describe real speech."
)


class StageStats(BaseModel):
    """Percentile summary of one measured quantity across a run."""

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str
    samples: int
    p50: float
    p95: float
    minimum: float
    maximum: float
    measured: bool
    """False when every sample was zero.

    A stage can read all-zero for two different reasons — it was never
    exercised (`PipelineBenchmark` never calls `ContextBuilder`, so
    `retrieval_ms` is structurally 0), or it genuinely completed in under a
    millisecond. Those are indistinguishable from the samples alone, and a
    benchmark report that prints "0 ms" for work that never ran is actively
    misleading. Reporting `not measured` is the honest reading of the
    ambiguity; a real sub-millisecond stage is mis-labelled, which is the
    strictly safer error.
    """


class ResourcePeak(BaseModel):
    """High-water marks across whichever turns carried a resource sample."""

    model_config = ConfigDict(frozen=True)

    samples: int
    peak_ram_used_mb: int
    ram_total_mb: int
    peak_cpu_percent: float
    peak_vram_used_mb: int | None = None
    vram_total_mb: int | None = None
    peak_gpu_percent: float | None = None


class ScanSaturation(BaseModel):
    """M1(a) visibility metric: how close retrieval came to `scan_limit`.

    The evidence that decides whether the deferred M1(b) ANN index is ever
    actually needed. Without the configured limit alongside the counts, a raw
    scan count is not actionable, so both travel together.
    """

    model_config = ConfigDict(frozen=True)

    scan_limit: int
    max_scanned: int
    p50_scanned: float
    saturated_turns: int
    saturated_percent: float


class BenchmarkReport(BaseModel):
    """Canonical intermediate representation for every output format."""

    model_config = ConfigDict(frozen=True)

    label: str
    source: str
    """Where the samples came from — `benchmark` or `live-session`. Recorded
    because the two are not comparable: a synthetic run drives TTS-generated
    speech through a fixed prompt, a live session does neither."""
    generated_at: str
    turn_count: int
    completed_count: int
    cancelled_count: int
    stages: tuple[StageStats, ...]
    environment: Environment
    resources: ResourcePeak | None = None
    scan_saturation: ScanSaturation | None = None
    wer: None = None
    """Always None in this batch — see `WER_DEFERRAL_NOTE`. Present so the
    section is structurally in the report rather than silently absent."""
    notes: tuple[str, ...] = ()


# ──────────────────────────── aggregation ────────────────────────────

#: (label, unit, extractor) for every quantity summarised from `TurnMetrics`.
#: Adding a stage here is the *only* thing needed to get it into all three
#: output formats — the renderers iterate the model, never a hardcoded list.
_STAGES: tuple[tuple[str, str, str], ...] = (
    ("Speech recognition", "ms", "asr_ms"),
    ("Memory retrieval", "ms", "retrieval_ms"),
    ("Context composition", "ms", "context_ms"),
    ("Time to first token", "ms", "ttft_ms"),
    ("LLM generation", "ms", "llm_ms"),
    ("First-sentence synthesis", "ms", "tts_first_ms"),
    ("Time to first audio", "ms", "ttfa_ms"),
    ("Total turn", "ms", "total_ms"),
)


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(samples, pct))


def _stage_stats(name: str, unit: str, values: list[float]) -> StageStats:
    if not values:
        return StageStats(
            name=name,
            unit=unit,
            samples=0,
            p50=0.0,
            p95=0.0,
            minimum=0.0,
            maximum=0.0,
            measured=False,
        )
    return StageStats(
        name=name,
        unit=unit,
        samples=len(values),
        p50=round(_percentile(values, 50), 2),
        p95=round(_percentile(values, 95), 2),
        minimum=round(min(values), 2),
        maximum=round(max(values), 2),
        measured=any(v != 0 for v in values),
    )


def _resource_peak(turns: Sequence[TurnMetrics]) -> ResourcePeak | None:
    samples = [t.resources for t in turns if t.resources is not None]
    if not samples:
        return None
    vram = [s.vram_used_mb for s in samples if s.vram_used_mb is not None]
    gpu = [s.gpu_percent for s in samples if s.gpu_percent is not None]
    totals = [s.vram_total_mb for s in samples if s.vram_total_mb is not None]
    return ResourcePeak(
        samples=len(samples),
        peak_ram_used_mb=max(s.ram_used_mb for s in samples),
        ram_total_mb=max(s.ram_total_mb for s in samples),
        peak_cpu_percent=round(max(s.cpu_percent for s in samples), 1),
        peak_vram_used_mb=max(vram) if vram else None,
        vram_total_mb=max(totals) if totals else None,
        peak_gpu_percent=round(max(gpu), 1) if gpu else None,
    )


def _scan_saturation(turns: Sequence[TurnMetrics], scan_limit: int | None) -> ScanSaturation | None:
    if scan_limit is None:
        return None
    counts = [t.retrieval_scan_count for t in turns if t.retrieval_scan_count > 0]
    if not counts:
        return None
    saturated = sum(1 for c in counts if c >= scan_limit)
    return ScanSaturation(
        scan_limit=scan_limit,
        max_scanned=max(counts),
        p50_scanned=round(_percentile([float(c) for c in counts], 50), 1),
        saturated_turns=saturated,
        saturated_percent=round(100.0 * saturated / len(counts), 1),
    )


def aggregate(
    turns: Sequence[TurnMetrics],
    *,
    label: str,
    source: str,
    environment: Environment,
    scan_limit: int | None = None,
    notes: Sequence[str] = (),
) -> BenchmarkReport:
    """Summarise any sequence of `TurnMetrics` into the canonical report.

    Source-agnostic by construction: `turns` may come from
    `MetricsCollector.turns` (a live session) or from `PipelineBenchmark`
    rounds, and this function cannot tell the difference. `source` is recorded
    for the reader's benefit, never branched on.

    Cancelled turns are excluded from the latency percentiles — a barged-in
    turn stopped early by design, and folding its truncated timings into a
    median would understate real latency — but they stay in `turn_count`, so
    the totals still describe everything that happened.
    """
    completed = [t for t in turns if not t.cancelled]
    stages = tuple(
        _stage_stats(name, unit, [float(getattr(t, field)) for t in completed])
        for name, unit, field in _STAGES
    )
    speeds = [t.tokens_per_s for t in completed if t.llm_ms > 0]
    stages += (_stage_stats("LLM speed", "tok/s", speeds),)

    return BenchmarkReport(
        label=label,
        source=source,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        turn_count=len(turns),
        completed_count=len(completed),
        cancelled_count=len(turns) - len(completed),
        stages=stages,
        environment=environment,
        resources=_resource_peak(completed),
        scan_saturation=_scan_saturation(completed, scan_limit),
        notes=tuple(notes),
    )


# ──────────────────────────── rendering ────────────────────────────


def _format_value(value: float, unit: str) -> str:
    return f"{value:.1f} {unit}" if unit == "tok/s" else f"{value:.0f} {unit}"


def to_json(report: BenchmarkReport) -> str:
    """The model verbatim — the machine-readable form other tools diff."""
    return json.dumps(json.loads(report.model_dump_json()), indent=2) + "\n"


def to_markdown(report: BenchmarkReport) -> str:
    """Markdown is the format that diffs usefully in git, which is what makes
    a run comparable to the one before it."""
    e = report.environment
    commit = e.git_commit or "not a git checkout"
    if e.git_dirty:
        commit += " (dirty — uncommitted changes present)"

    lines = [
        f"# Benchmark report — {report.label}",
        "",
        f"- **Source:** {report.source}",
        f"- **Generated:** {report.generated_at}",
        f"- **Turns:** {report.turn_count} "
        f"({report.completed_count} completed, {report.cancelled_count} cancelled)",
        "",
        "## Latency",
        "",
        "| Stage | p50 | p95 | min | max | samples |",
        "|---|---|---|---|---|---|",
    ]
    for s in report.stages:
        if not s.measured:
            lines.append(f"| {s.name} | _{NOT_MEASURED}_ | | | | {s.samples} |")
            continue
        lines.append(
            f"| {s.name} | {_format_value(s.p50, s.unit)} | {_format_value(s.p95, s.unit)} "
            f"| {_format_value(s.minimum, s.unit)} | {_format_value(s.maximum, s.unit)} "
            f"| {s.samples} |"
        )

    lines += ["", "## Speech recognition accuracy", "", f"_{WER_DEFERRAL_NOTE}_"]

    lines += ["", "## Resources", ""]
    r = report.resources
    if r is None:
        lines.append(f"_{NOT_MEASURED} — no turn carried a resource sample._")
    else:
        lines += [
            f"- **Peak RAM:** {r.peak_ram_used_mb} MB of {r.ram_total_mb} MB",
            f"- **Peak CPU:** {r.peak_cpu_percent} %",
        ]
        if r.peak_vram_used_mb is not None:
            total = f" of {r.vram_total_mb} MB" if r.vram_total_mb else ""
            lines.append(f"- **Peak VRAM:** {r.peak_vram_used_mb} MB{total}")
        if r.peak_gpu_percent is not None:
            lines.append(f"- **Peak GPU:** {r.peak_gpu_percent} %")
        lines.append(f"- **Samples:** {r.samples}")

    if report.scan_saturation is not None:
        sat = report.scan_saturation
        lines += [
            "",
            "## Memory scan saturation",
            "",
            f"- **Scan limit:** {sat.scan_limit} candidates",
            f"- **Scanned:** p50 {sat.p50_scanned}, max {sat.max_scanned}",
            f"- **Saturated turns:** {sat.saturated_turns} ({sat.saturated_percent} %)",
        ]

    lines += [
        "",
        "## Environment",
        "",
        f"- **EVA:** {e.eva_version}",
        f"- **Commit:** {commit}",
        f"- **Backend:** {e.backend}",
        f"- **GPU:** {e.gpu_name or 'none detected'}"
        + (f" ({e.gpu_vram_mb} MB)" if e.gpu_vram_mb is not None else "")
        + f", CUDA devices: {e.cuda_device_count}",
        f"- **faster-whisper:** {e.faster_whisper_version} · "
        f"**ctranslate2:** {e.ctranslate2_version}",
        f"- **Python / OS:** {e.python_version} on {e.platform}",
    ]
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    return "\n".join(lines) + "\n"


#: Inline, so a report opens correctly from a file:// URL on a machine with no
#: network — EVA is offline by construction (ADR-017 / the §10 invariant), and
#: a report that phoned a CDN for a stylesheet would quietly break that.
_HTML_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif;
       margin: 2rem auto; max-width: 60rem; padding: 0 1rem; line-height: 1.5; }
h1 { margin-bottom: 0.25rem; }
.sub { color: #667; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }
th, td { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #8884; }
th { font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: #888; font-style: italic; }
.bar { display: block; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.2rem 1rem; }
dt { font-weight: 600; }
dd { margin: 0; }
.wrap { overflow-x: auto; }
"""


def _svg_bar(value: float, peak: float) -> str:
    """One inline SVG bar. Inline because an external chart library is not
    available to an offline report — and is not needed for a single scalar."""
    width = 0 if peak <= 0 else max(1, round(160 * value / peak))
    return (
        f'<svg class="bar" width="160" height="10" viewBox="0 0 160 10" '
        f'role="img" aria-label="{value:.0f}">'
        f'<rect x="0" y="0" width="160" height="10" fill="#8882" rx="2"/>'
        f'<rect x="0" y="0" width="{width}" height="10" fill="#4a90d9" rx="2"/>'
        f"</svg>"
    )


def to_html(report: BenchmarkReport) -> str:
    """A single self-contained file: inline CSS, inline SVG, no scripts, and
    no external references of any kind (asserted by the test suite)."""
    esc = html.escape
    e = report.environment
    commit = e.git_commit or "not a git checkout"
    if e.git_dirty:
        commit += " (dirty — uncommitted changes present)"

    ms_peak = max(
        (s.p50 for s in report.stages if s.measured and s.unit == "ms"),
        default=0.0,
    )
    rows: list[str] = []
    for s in report.stages:
        if not s.measured:
            rows.append(
                f"<tr><td>{esc(s.name)}</td>"
                f'<td colspan="4" class="muted">{NOT_MEASURED}</td>'
                f'<td class="num">{s.samples}</td></tr>'
            )
            continue
        bar = _svg_bar(s.p50, ms_peak) if s.unit == "ms" else ""
        rows.append(
            f"<tr><td>{esc(s.name)}{bar}</td>"
            f'<td class="num">{esc(_format_value(s.p50, s.unit))}</td>'
            f'<td class="num">{esc(_format_value(s.p95, s.unit))}</td>'
            f'<td class="num">{esc(_format_value(s.minimum, s.unit))}</td>'
            f'<td class="num">{esc(_format_value(s.maximum, s.unit))}</td>'
            f'<td class="num">{s.samples}</td></tr>'
        )

    resource_html = f'<p class="muted">{NOT_MEASURED} — no turn carried a resource sample.</p>'
    if report.resources is not None:
        r = report.resources
        items = [
            ("Peak RAM", f"{r.peak_ram_used_mb} MB of {r.ram_total_mb} MB"),
            ("Peak CPU", f"{r.peak_cpu_percent} %"),
        ]
        if r.peak_vram_used_mb is not None:
            total = f" of {r.vram_total_mb} MB" if r.vram_total_mb else ""
            items.append(("Peak VRAM", f"{r.peak_vram_used_mb} MB{total}"))
        if r.peak_gpu_percent is not None:
            items.append(("Peak GPU", f"{r.peak_gpu_percent} %"))
        items.append(("Samples", str(r.samples)))
        resource_html = (
            "<dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in items) + "</dl>"
        )

    saturation_html = ""
    if report.scan_saturation is not None:
        sat = report.scan_saturation
        saturation_html = (
            "<h2>Memory scan saturation</h2><dl>"
            f"<dt>Scan limit</dt><dd>{sat.scan_limit} candidates</dd>"
            f"<dt>Scanned</dt><dd>p50 {sat.p50_scanned}, max {sat.max_scanned}</dd>"
            f"<dt>Saturated turns</dt>"
            f"<dd>{sat.saturated_turns} ({sat.saturated_percent} %)</dd></dl>"
        )

    notes_html = ""
    if report.notes:
        notes_html = (
            "<h2>Notes</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in report.notes) + "</ul>"
        )

    env_items = [
        ("EVA", e.eva_version),
        ("Commit", commit),
        ("Backend", e.backend),
        (
            "GPU",
            (e.gpu_name or "none detected")
            + (f" ({e.gpu_vram_mb} MB)" if e.gpu_vram_mb is not None else "")
            + f", CUDA devices: {e.cuda_device_count}",
        ),
        ("faster-whisper", str(e.faster_whisper_version)),
        ("ctranslate2", str(e.ctranslate2_version)),
        ("Python / OS", f"{e.python_version} on {e.platform}"),
    ]
    env_html = (
        "<dl>" + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in env_items) + "</dl>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVA benchmark — {esc(report.label)}</title>
<style>{_HTML_STYLE}</style></head>
<body>
<h1>Benchmark report</h1>
<p class="sub">{esc(report.label)} · source: {esc(report.source)} · {esc(report.generated_at)}</p>
<p>{report.turn_count} turns ({report.completed_count} completed,
{report.cancelled_count} cancelled)</p>
<h2>Latency</h2>
<div class="wrap"><table>
<thead><tr><th>Stage</th><th>p50</th><th>p95</th><th>min</th><th>max</th><th>n</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
<h2>Speech recognition accuracy</h2>
<p class="muted">{esc(WER_DEFERRAL_NOTE)}</p>
<h2>Resources</h2>
{resource_html}
{saturation_html}
<h2>Environment</h2>
{env_html}
{notes_html}
</body></html>
"""


def render(report: BenchmarkReport, fmt: str) -> str:
    """Project the canonical model into one of the supported formats."""
    renderers = {"json": to_json, "md": to_markdown, "html": to_html}
    if fmt not in renderers:
        raise ValueError(f"Unknown report format {fmt!r} (expected one of {sorted(renderers)})")
    return renderers[fmt](report)


def summary_line(report: BenchmarkReport) -> str:
    """One-line console confirmation after writing a report file."""
    ttfa = next((s for s in report.stages if s.name == "Time to first audio"), None)
    if ttfa is None or not ttfa.measured:
        return f"{report.completed_count} completed turn(s)"
    return f"{report.completed_count} completed turn(s), TTFA p50 {ttfa.p50:.0f} ms"
