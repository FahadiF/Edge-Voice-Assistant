"""Command-line entry point.

M0 provides ``eva diagnose`` and ``eva version``; later milestones add
``eva run`` (voice loop), ``eva serve`` (engine server), and ``eva bench``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import eva
from eva.config import get_app_paths, load_settings
from eva.core.errors import EvaError
from eva.hardware import detect_hardware, recommend_profile
from eva.logging_setup import setup_logging


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"{eva.APP_DISPLAY_NAME} {eva.__version__}")
    return 0


def _cmd_diagnose(_: argparse.Namespace) -> int:
    paths = get_app_paths()
    settings = load_settings(paths.settings_file)
    report = detect_hardware()
    profile = recommend_profile(report)

    def section(title: str) -> None:
        print(f"\n{title}\n{'-' * len(title)}")

    print(f"{eva.APP_DISPLAY_NAME} {eva.__version__} — diagnostics")

    section("System")
    print(f"OS:      {report.os_name} {report.os_version}")
    print(f"Python:  {report.python_version}")

    section("Hardware")
    print(f"CPU:     {report.cpu.name}")
    print(f"Cores:   {report.cpu.physical_cores} physical / {report.cpu.logical_cores} logical")
    print(f"RAM:     {report.memory.total_mb} MB total, {report.memory.available_mb} MB available")
    if report.gpus:
        for gpu in report.gpus:
            vram = f"{gpu.vram_total_mb} MB" if gpu.vram_total_mb else "unknown VRAM"
            driver = f", driver {gpu.driver_version}" if gpu.driver_version else ""
            print(f"GPU:     {gpu.name} ({gpu.backend.upper()}, {vram}{driver})")
    else:
        print("GPU:     none detected (CPU-only)")

    section("Recommended profile")
    print(f"{profile.id} — {profile.display_name}")
    print(f"  {profile.description}")
    print(f"  LLM: {profile.llm_model} | ASR: {profile.asr_model} | TTS: {profile.tts_engine}")

    section("Configuration")
    print(
        f"Settings file: {paths.settings_file}"
        f" ({'present' if paths.settings_file.exists() else 'not created yet — using defaults'})"
    )
    print(f"Active profile setting: {settings.profile}")
    print(f"LLM:  engine={settings.llm.engine} model={settings.llm.model}")
    print(f"ASR:  engine={settings.asr.engine} model={settings.asr.model}")
    print(f"TTS:  engine={settings.tts.engine} voice={settings.tts.voice}")
    print(f"VAD:  engine={settings.vad.engine} threshold={settings.vad.threshold}")

    section("Paths")
    print(f"Config:        {paths.config_dir}")
    print(f"Models:        {paths.models_dir}")
    print(f"Conversations: {paths.conversations_dir}")
    print(f"Logs:          {paths.logs_dir}")
    return 0


def _cmd_devices(_: argparse.Namespace) -> int:
    from eva.audio.devices import list_devices

    inputs, outputs = list_devices()

    def show(title: str, devices: list[object]) -> None:
        print(f"\n{title}\n{'-' * len(title)}")
        for d in devices:
            marker = "*" if getattr(d, "is_default", False) else " "
            print(
                f"{marker} [{getattr(d, 'index', '?'):>3}] {getattr(d, 'name', '?')}"
                f"  ({getattr(d, 'host_api', '?')})"
            )

    show("Input devices (* = default)", list(inputs))
    show("Output devices (* = default)", list(outputs))
    return 0


def _cmd_listen(args: argparse.Namespace) -> int:
    from eva.audio.demo import run_listen

    settings = load_settings(get_app_paths().settings_file)
    return run_listen(settings, seconds=args.seconds)


def _cmd_echo_test(args: argparse.Namespace) -> int:
    from eva.audio.demo import run_echo_test

    settings = load_settings(get_app_paths().settings_file)
    return run_echo_test(settings, record_seconds=args.record_seconds, loops=args.loops)


def _cmd_run(_: argparse.Namespace) -> int:
    from eva.engine import build_assistant, required_models
    from eva.models.manager import ModelManager
    from eva.voice_loop import main_run

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    manager = ModelManager(paths)
    missing = [m for m in required_models(settings) if not manager.is_installed(m)]
    if missing:
        print("Required models are not installed yet:")
        for model_id in missing:
            print(f"  eva models download {model_id}")
        return 1
    return main_run(build_assistant(settings, paths))


def _cmd_models(args: argparse.Namespace) -> int:
    from eva.models.manager import ModelManager

    paths = get_app_paths()
    paths.ensure_exists()
    manager = ModelManager(paths)

    if args.models_command == "list":
        settings = load_settings(paths.settings_file)
        active = {settings.llm.model, settings.asr.model}
        print(f"{'':2}{'id':<32} {'kind':<5} {'license':<12} {'size':>8}  status")
        for info in manager.available():
            installed = manager.is_installed(info.id)
            marker = "*" if info.id in active else " "
            size = sum(f.size_mb for f in info.files)
            size_str = f"{size} MB" if size else "—"
            status = "installed" if installed else "available"
            if info.managed_by == "engine" and not installed:
                status = "auto (on first use)"
            print(
                f"{marker:2}{info.id:<32} {info.kind:<5} {info.license:<12} {size_str:>8}  {status}"
            )
        print("\n* = active in settings")
        return 0

    if args.models_command == "download":
        last_shown = -1

        def progress(filename: str, done: int, total: int) -> None:
            nonlocal last_shown
            pct = int(done * 100 / total) if total else 0
            if pct != last_shown:
                last_shown = pct
                mb = done // 1_048_576
                print(f"\r{filename}: {mb} MB ({pct}%)", end="", flush=True)

        manager.download(args.model_id, progress)
        print(f"\n'{args.model_id}' installed.")
        return 0

    if args.models_command == "remove":
        manager.remove(args.model_id)
        print(f"'{args.model_id}' removed.")
        return 0
    return 2


def _cmd_bench(args: argparse.Namespace) -> int:
    from eva.benchmark.pipeline import PipelineBenchmark
    from eva.engine import build_assistant
    from eva.llm.base import GenerationParams

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    assistant = build_assistant(settings, paths)
    print("Loading models...")
    assistant.preload()
    bench = PipelineBenchmark(
        assistant.asr,
        assistant.llm,
        assistant.tts,
        voice=settings.tts.voice,
        system_prompt=settings.conversation.system_prompt,
        params=GenerationParams(
            temperature=settings.conversation.temperature,
            top_p=settings.conversation.top_p,
            max_tokens=settings.conversation.max_tokens,
        ),
    )
    print(f"\nRunning pipeline benchmark ({args.rounds} round(s))...\n")
    for i in range(args.rounds):
        report = bench.run(args.text)
        print(f"── Round {i + 1} ──")
        print(report.render())
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eva",
        description=f"{eva.APP_DISPLAY_NAME} — fully offline voice assistant",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override configured log level",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Print hardware, configuration, and path report")
    p_diag.set_defaults(func=_cmd_diagnose)

    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=_cmd_version)

    p_dev = sub.add_parser("devices", help="List audio input/output devices")
    p_dev.set_defaults(func=_cmd_devices)

    p_listen = sub.add_parser("listen", help="Live VAD/segmentation monitor")
    p_listen.add_argument("--seconds", type=float, default=30.0)
    p_listen.set_defaults(func=_cmd_listen)

    p_echo = sub.add_parser(
        "echo-test", help="Measure echo cancellation on the speaker/microphone path"
    )
    p_echo.add_argument("--record-seconds", type=float, default=4.0)
    p_echo.add_argument("--loops", type=int, default=2)
    p_echo.set_defaults(func=_cmd_echo_test)

    p_run = sub.add_parser("run", help="Start the voice assistant (interactive)")
    p_run.set_defaults(func=_cmd_run)

    p_models = sub.add_parser("models", help="Manage models")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List available and installed models")
    p_dl = models_sub.add_parser("download", help="Download a model")
    p_dl.add_argument("model_id")
    p_rm = models_sub.add_parser("remove", help="Remove an installed model")
    p_rm.add_argument("model_id")
    p_models.set_defaults(func=_cmd_models)

    p_bench = sub.add_parser("bench", help="Run the end-to-end pipeline benchmark (no mic needed)")
    p_bench.add_argument(
        "--text", default="What is the capital of Finland and what is it known for?"
    )
    p_bench.add_argument("--rounds", type=int, default=1)
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def _force_utf8_stdio() -> None:
    """Windows consoles often default to a legacy code page; our output is UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_stdio()
    args = build_parser().parse_args(argv)

    paths = get_app_paths()
    settings = load_settings(paths.settings_file)
    setup_logging(
        level=args.log_level or settings.developer.log_level,
        logs_dir=paths.logs_dir,
        json_file=settings.developer.log_json,
    )

    try:
        result: int = args.func(args)
    except EvaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":
    sys.exit(main())
