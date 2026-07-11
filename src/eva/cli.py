"""Command-line entry point.

Commands: ``version``, ``diagnose``, ``doctor`` (dependency/model readiness),
``first-run`` (guided setup wizard), ``setup`` (install the LLM runtime),
``devices``, ``listen``, ``echo-test``, ``models``, ``profiles`` (hardware/
model presets), ``config``, ``run`` (voice loop; runs the wizard
automatically when setup is incomplete), ``bench``, ``serve`` (platform API
server — see ``eva.server``, ADR-017); M4 personalization/memory commands:
``personas``, ``users``, ``voices``, ``memory``, ``profile`` (singular —
active *user* profile, not to be confused with the hardware ``profiles``).

The CLI is one client of the same engine services the platform API exposes
(``ModelManager``, ``eva.config.service``, ``eva.onboarding``, …) — it does not
duplicate their logic.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import eva
from eva.config import get_app_paths, load_settings
from eva.config.settings import Settings
from eva.core.errors import EvaError
from eva.hardware import detect_hardware, recommend_profile
from eva.llm.registry import create_llm
from eva.logging_setup import setup_logging
from eva.tts.base import TTSEngine


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


def _cmd_run(args: argparse.Namespace) -> int:
    """Start the assistant, guiding the user through setup if anything is missing."""
    from eva.conversation.personas import (
        register_builtin_personas,
        register_custom_personas,
        resolve_persona,
    )
    from eva.engine import build_assistant
    from eva.onboarding import run_onboarding
    from eva.voice_loop import main_run

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)

    if args.persona:
        register_builtin_personas()
        register_custom_personas(settings)
        settings.conversation.persona = args.persona
        resolve_persona(settings)  # raises RegistryError for an unknown id
        # In-memory only (never saved) — a one-session override for
        # side-by-side persona comparisons without editing settings.json.

    result = run_onboarding(settings, paths, assume_yes=args.yes)
    if not result.ready:
        return 0 if result.declined else 1
    return main_run(build_assistant(settings, paths))


def _cmd_first_run(args: argparse.Namespace) -> int:
    """Launch the onboarding wizard directly, then start the assistant if ready."""
    from eva.engine import build_assistant
    from eva.onboarding import run_onboarding
    from eva.voice_loop import main_run

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)

    result = run_onboarding(settings, paths, assume_yes=args.yes, force=True)
    if not result.ready:
        return 0 if result.declined else 1
    if args.setup_only:
        print("Setup verified. Run `eva run` to start talking.")
        return 0
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

    if args.models_command == "info":
        settings = load_settings(paths.settings_file)
        card = manager.describe(args.model_id, settings)
        width = max(len(k) for k in card)
        for key, value in card.items():
            if value in (None, "", 0) and key not in ("installed", "active", "compatible"):
                continue
            print(f"{key.replace('_', ' '):<{width + 2}}{value}")
        return 0

    if args.models_command == "use":
        from eva.config.settings import save_settings
        from eva.hardware.presets import CUSTOM_PROFILE_ID

        settings = load_settings(paths.settings_file)
        info = manager.info(args.model_id)
        if info.kind == "llm":
            settings.llm.model = info.id
        elif info.kind == "asr":
            settings.asr.model = info.id
        elif info.kind == "tts":
            settings.tts.model = info.id
            settings.tts.engine = info.engine
        elif info.kind == "vad":
            settings.vad.engine = info.engine
        settings.profile = CUSTOM_PROFILE_ID  # manual choice overrides presets
        save_settings(settings, paths.settings_file)
        print(f"Active {info.kind.upper()} set to '{info.id}' (profile: custom). Saved.")
        if not manager.is_installed(info.id):
            print(f"Note: not installed yet — run: eva models download {info.id}")
        return 0
    return 2


def _cmd_profiles(args: argparse.Namespace) -> int:
    from eva.config.settings import save_settings
    from eva.hardware.presets import apply_preset, preset_registry, register_builtin_presets

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    register_builtin_presets()
    tier = recommend_profile(detect_hardware())

    if args.profiles_command == "list":
        print(f"Hardware tier: {tier.id} ({tier.display_name})\n")
        for preset in preset_registry.snapshot().values():
            marker = "*" if preset.id == settings.profile else " "
            models = preset.for_tier(tier.id)
            print(f"{marker} {preset.id:<14} {preset.description}")
            print(f"    LLM {models.llm_model} | ASR {models.asr_model} | TTS {models.tts_model}")
        if settings.profile == "custom":
            print("* custom         (models chosen manually)")
        print("\n* = active profile")
        return 0

    if args.profiles_command == "set":
        apply_preset(settings, args.preset_id, tier.id)
        save_settings(settings, paths.settings_file)
        print(f"Profile '{args.preset_id}' applied for tier '{tier.id}' and saved:")
        print(f"  LLM: {settings.llm.model}")
        print(f"  ASR: {settings.asr.model}")
        print(f"  TTS: {settings.tts.model}")
        from eva.models.manager import ModelManager

        manager = ModelManager(paths)
        from eva.engine import required_models

        missing = [m for m in required_models(settings) if not manager.is_installed(m)]
        for model_id in missing:
            print(f"  Note: '{model_id}' is not installed — run: eva models download {model_id}")
        return 0
    return 2


def _cmd_personas(args: argparse.Namespace) -> int:
    """Manage personas (M4, ADR-022) — settings-backed, no engine needed."""
    from eva.config.settings import PersonaSettingsEntry, save_settings
    from eva.conversation.personas import (
        persona_registry,
        register_builtin_personas,
        register_custom_personas,
    )
    from eva.core.errors import RegistryError

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    register_builtin_personas()
    register_custom_personas(settings)
    custom_ids = {p.id for p in settings.conversation.custom_personas}

    if args.personas_command == "list":
        print(f"{'':2}{'id':<16} {'display name':<20} {'verbosity':<10} tone")
        for persona in persona_registry.snapshot().values():
            marker = "*" if persona.id == settings.conversation.persona else " "
            kind = "custom" if persona.id in custom_ids else "builtin"
            print(
                f"{marker:2}{persona.id:<16} {persona.display_name:<20} "
                f"{persona.verbosity:<10} {persona.tone}  ({kind})"
            )
        print("\n* = active persona (settings.conversation.persona)")
        return 0

    if args.personas_command == "show":
        try:
            persona = persona_registry.get(args.persona_id)
        except RegistryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for key, value in persona.model_dump().items():
            print(f"{key:<20}{value}")
        return 0

    if args.personas_command == "create":
        existing = {p.id for p in persona_registry.snapshot().values()}
        if args.id in existing - custom_ids:
            print(
                f"error: '{args.id}' is a built-in persona id and cannot be overridden",
                file=sys.stderr,
            )
            return 1
        entry = PersonaSettingsEntry(
            id=args.id,
            display_name=args.name,
            system_prompt=args.prompt,
            verbosity=args.verbosity,
            tone=args.tone,
            reasoning_style=args.reasoning_style,
            temperature_override=args.temperature,
        )
        remaining = [p for p in settings.conversation.custom_personas if p.id != args.id]
        settings.conversation.custom_personas = [*remaining, entry]
        save_settings(settings, paths.settings_file)
        print(f"Persona '{args.id}' saved.")
        return 0

    if args.personas_command == "delete":
        if args.persona_id not in custom_ids:
            print(
                f"error: '{args.persona_id}' is not a custom persona (built-ins cannot be deleted)",
                file=sys.stderr,
            )
            return 1
        settings.conversation.custom_personas = [
            p for p in settings.conversation.custom_personas if p.id != args.persona_id
        ]
        save_settings(settings, paths.settings_file)
        persona_registry.unregister(args.persona_id)
        print(f"Persona '{args.persona_id}' deleted.")
        return 0

    if args.personas_command == "use":
        try:
            persona_registry.get(args.persona_id)
        except RegistryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        settings.conversation.persona = args.persona_id
        save_settings(settings, paths.settings_file)
        print(f"Active persona set to '{args.persona_id}' and saved.")
        print("Takes effect on the next `eva run` / `eva serve` start.")
        return 0
    return 2


def _cmd_users(args: argparse.Namespace) -> int:
    """Manage user profiles (M4, ADR-022) — opens the memory database
    directly, the same way `eva models` opens `ModelManager` directly,
    without needing a running `eva serve` engine."""
    import uuid
    from datetime import UTC, datetime

    from eva.config.settings import save_settings
    from eva.memory.models import UserProfile
    from eva.memory.registry import create_stores

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    memory, profiles = create_stores(settings, paths)
    try:
        if args.users_command == "list":
            active = profiles.active()
            active_id = active.id if active else None
            print(f"{'':2}{'id':<36} {'nickname':<16} language  voice")
            for profile in profiles.list():
                marker = "*" if profile.id == active_id else " "
                print(
                    f"{marker:2}{profile.id:<36} {profile.nickname:<16} "
                    f"{profile.preferred_language or '-':<8}  {profile.preferred_voice or '-'}"
                )
            print("\n* = active profile")
            return 0

        if args.users_command == "show":
            profile = profiles.get(args.user_id)
            for key, value in profile.model_dump().items():
                print(f"{key:<22}{value}")
            return 0

        if args.users_command == "create":
            now = datetime.now(UTC)
            kwargs: dict[str, object] = {
                "id": args.id or str(uuid.uuid4()),
                "nickname": args.nickname or "",
                "preferred_language": args.language,
                "preferred_voice": args.voice,
                "preferred_llm_model": args.llm_model,
                "conversation_style": args.style or "",
                "created_at": now,
                "updated_at": now,
            }
            # units/timezone: only set if the flag was passed — otherwise
            # let UserProfile's own field defaults ("metric"/"UTC") apply.
            if args.units is not None:
                kwargs["units"] = args.units
            if args.timezone is not None:
                kwargs["timezone"] = args.timezone
            created = profiles.create(UserProfile(**kwargs))
            print(f"User profile '{created.id}' created.")
            return 0

        if args.users_command == "edit":
            current = profiles.get(args.user_id)
            updates: dict[str, object] = {}
            if args.nickname is not None:
                updates["nickname"] = args.nickname
            if args.language is not None:
                updates["preferred_language"] = args.language
            if args.voice is not None:
                updates["preferred_voice"] = args.voice
            if args.llm_model is not None:
                updates["preferred_llm_model"] = args.llm_model
            if args.style is not None:
                updates["conversation_style"] = args.style
            if args.units is not None:
                updates["units"] = args.units
            if args.timezone is not None:
                updates["timezone"] = args.timezone
            updates["updated_at"] = datetime.now(UTC)
            updated = profiles.update(current.model_copy(update=updates))
            print(f"User profile '{updated.id}' updated.")
            return 0

        if args.users_command == "activate":
            profiles.set_active(args.user_id)
            # Mirror for visibility only — UserProfileStore.active() (DB) is
            # the value ContextBuilder actually reads.
            settings.conversation.active_profile_id = args.user_id
            save_settings(settings, paths.settings_file)
            print(f"User profile '{args.user_id}' activated.")
            return 0

        if args.users_command == "delete":
            profiles.delete(args.user_id)
            print(f"User profile '{args.user_id}' deleted.")
            return 0
        return 2
    finally:
        memory.close()


def _load_tts_engine() -> tuple[TTSEngine, Settings]:
    """Build and load just the TTS engine (no LLM/ASR/audio system) — voice
    discovery/preview only needs this one, CPU-resident model."""
    from eva.models.manager import ModelManager
    from eva.tts.registry import create_tts

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    manager = ModelManager(paths)
    tts = create_tts(settings, manager.files_for(settings.tts.model))
    tts.load()
    return tts, settings


def _cmd_voices(args: argparse.Namespace) -> int:
    """Manage TTS voices (M4, ADR-022)."""
    from eva.config.settings import save_settings
    from eva.tts.voices import preview_text, register_voices_for_engine, voices_for_engine

    if args.voices_command == "use":
        paths = get_app_paths()
        paths.ensure_exists()
        settings = load_settings(paths.settings_file)
        settings.tts.voice = args.voice_id
        save_settings(settings, paths.settings_file)
        print(f"Active voice set to '{args.voice_id}' and saved.")
        print("Takes effect on the next `eva run` / `eva serve` start.")
        return 0

    tts, settings = _load_tts_engine()
    try:
        register_voices_for_engine(settings.tts.engine, tts)
        if args.voices_command == "list":
            print(f"{'':2}{'id':<20} {'language':<10} style")
            for voice in voices_for_engine(settings.tts.engine):
                marker = "*" if voice.id == settings.tts.voice else " "
                print(f"{marker:2}{voice.id:<20} {voice.language:<10} {voice.style_tag}")
            print("\n* = active voice (settings.tts.voice)")
            return 0

        if args.voices_command == "preview":
            import tempfile
            import wave

            from eva.audio.frames import SAMPLE_RATE

            frame = preview_text(tts, args.voice_id, phrase=args.text)
            out_path = Path(tempfile.gettempdir()) / f"eva-voice-preview-{args.voice_id}.wav"
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(frame.tobytes())
            print(f"Preview written to {out_path}")
            return 0
        return 2
    finally:
        tts.unload()


def _cmd_memory(args: argparse.Namespace) -> int:
    """Manage persistent conversation memory (M4, ADR-019) — opens the
    memory database directly, same pattern as `eva users`/`eva models`."""
    import json

    from eva.memory.registry import create_stores

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    memory, _profiles = create_stores(settings, paths)
    try:
        cmd = args.memory_command

        if cmd == "stats":
            stats = memory.stats()
            for key, value in stats.model_dump().items():
                print(f"{key:<22}{value}")
            return 0

        if cmd == "list":
            for conv in memory.all_conversations(include_archived=args.include_archived):
                flag = " [archived]" if conv.archived else ""
                title = conv.title or "(untitled)"
                print(f"{conv.id}  {title:<32}  {conv.started_at}  lang={conv.language}{flag}")
            return 0

        if cmd == "rename":
            memory.set_title(args.conversation_id, args.title)
            print(f"Conversation '{args.conversation_id}' renamed to '{args.title}'.")
            return 0

        if cmd == "show":
            for turn in memory.all_turns(args.conversation_id):
                flags = "".join(f" [{f}]" for f in ("pinned", "favorite") if getattr(turn, f))
                print(f"#{turn.id} [{turn.speaker}] {turn.created_at}{flags}: {turn.text}")
            return 0

        if cmd == "search":
            results = memory.search_text(
                args.query, limit=args.limit, conversation_id=args.conversation_id
            )
            for r in results:
                print(f"#{r.turn.id} score={r.score:.3f} ({r.match_reason}): {r.turn.text}")
            return 0

        if cmd == "forget":
            memory.forget(args.turn_id)
            print(f"Turn #{args.turn_id} forgotten.")
            return 0

        if cmd == "pin":
            memory.pin(args.turn_id, pinned=not args.unset)
            print(f"Turn #{args.turn_id} {'unpinned' if args.unset else 'pinned'}.")
            return 0

        if cmd == "favorite":
            memory.favorite(args.turn_id, favorite=not args.unset)
            print(f"Turn #{args.turn_id} {'unfavorited' if args.unset else 'favorited'}.")
            return 0

        if cmd == "archive":
            memory.archive_conversation(args.conversation_id, archived=not args.unset)
            print(
                f"Conversation '{args.conversation_id}' {'restored' if args.unset else 'archived'}."
            )
            return 0

        if cmd == "delete-conversation":
            memory.delete_conversation(args.conversation_id)
            print(f"Conversation '{args.conversation_id}' deleted.")
            return 0

        if cmd == "merge":
            memory.merge_conversations(args.source_id, args.target_id)
            print(f"Merged '{args.source_id}' into '{args.target_id}'.")
            return 0

        if cmd == "export":
            data = memory.export_json(args.conversation_id)
            text = json.dumps(data, indent=2, default=str)
            if args.out:
                Path(args.out).write_text(text, encoding="utf-8")
                print(f"Exported to {args.out}")
            else:
                print(text)
            return 0

        if cmd == "import":
            payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
            imported = memory.import_json(payload)
            print(f"Imported {imported} turn(s).")
            return 0

        if cmd == "delete-all":
            if not args.yes:
                print(
                    "This permanently deletes ALL conversations. Re-run with --yes.",
                    file=sys.stderr,
                )
                return 1
            memory.delete_all()
            print("All memory deleted.")
            return 0

        if cmd == "summarize":
            from eva.memory.models import MemorySummary
            from eva.memory.summarizer import LLMSummarizer
            from eva.models.manager import ModelManager

            turns = memory.all_turns(args.conversation_id)
            if not turns:
                print("No turns in this conversation.")
                return 0
            manager = ModelManager(paths)
            llm = create_llm(settings, manager.files_for(settings.llm.model)["model"])
            llm.load()
            try:
                text = LLMSummarizer(llm).summarize(turns)
            finally:
                llm.unload()
            first_id, last_id = turns[0].id, turns[-1].id
            assert first_id is not None and last_id is not None
            from datetime import UTC, datetime

            saved = memory.add_summary(
                MemorySummary(
                    conversation_id=args.conversation_id,
                    turn_range_start=first_id,
                    turn_range_end=last_id,
                    text=text,
                    created_at=datetime.now(UTC),
                    model_id=settings.llm.model,
                )
            )
            print(saved.text)
            return 0
        return 2
    finally:
        memory.close()


def _cmd_profile(args: argparse.Namespace) -> int:
    """Quick access to the *active user profile* (M4, ADR-022) — singular,
    distinct from the plural `eva profiles` (hardware/model presets). For
    full user profile CRUD, see `eva users`."""
    from eva.config.settings import save_settings
    from eva.memory.registry import create_stores

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    memory, profiles = create_stores(settings, paths)
    try:
        if args.profile_command == "show":
            active = profiles.active()
            if active is None:
                print("No active user profile. Create one with `eva users create`.")
                return 0
            for key, value in active.model_dump().items():
                print(f"{key:<22}{value}")
            return 0

        if args.profile_command == "use":
            profiles.set_active(args.user_id)
            settings.conversation.active_profile_id = args.user_id
            save_settings(settings, paths.settings_file)
            print(f"User profile '{args.user_id}' activated.")
            return 0
        return 2
    finally:
        memory.close()


def _cmd_bench(args: argparse.Namespace) -> int:
    from eva.benchmark.pipeline import PipelineBenchmark
    from eva.engine import build_assistant
    from eva.llm.base import GenerationParams
    from eva.onboarding import check_readiness, readiness_problems

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    problems = readiness_problems(check_readiness(settings, paths))
    if problems:
        print("Cannot benchmark yet — setup is incomplete:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRun `eva first-run` for guided setup, or `eva doctor` for details.")
        return 1
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


def _cmd_doctor(_: argparse.Namespace) -> int:
    from eva.onboarding import check_readiness

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    items = check_readiness(settings, paths)

    def show(category: str, title: str) -> None:
        print(f"{title}\n{'-' * len(title)}")
        for item in (i for i in items if i.category == category):
            mark = "ok     " if item.ok else "MISSING"
            line = f"[{mark}] {item.name:<28} {item.detail}"
            if not item.ok:
                line += f"\n            -> {item.remedy}"
            print(line)

    show("runtime", "Runtime dependencies")
    print()
    show("model", "Models")

    print()
    if all(i.ok for i in items):
        print("All checks passed. `eva run` is ready.")
        return 0
    print("Setup is incomplete. Run `eva first-run` for guided setup, or fix the items above.")
    return 1


def _cmd_setup(args: argparse.Namespace) -> int:
    from eva.engine import required_models
    from eva.models.manager import ModelManager
    from eva.runtime import choose_variant, install_llama_runtime, llm_runtime_available

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)

    report = detect_hardware()
    variant = choose_variant(report, override=args.variant)
    gpu = report.best_gpu
    print(f"Hardware: {gpu.name if gpu else 'no GPU detected'}")
    print(f"Selected LLM runtime build: {variant}\n")

    if llm_runtime_available() and not args.force:
        print("The llama.cpp runtime is already installed (use --force to reinstall).")
    else:
        code = install_llama_runtime(variant, dry_run=args.dry_run)
        if code != 0:
            print("\nRuntime installation failed; see the pip output above.", file=sys.stderr)
            return code
        if not args.dry_run:
            print("llama.cpp runtime installed.")

    manager = ModelManager(paths)
    missing_models = [m for m in required_models(settings) if not manager.is_installed(m)]
    print("\nNext steps")
    print("----------")
    if missing_models:
        print("Download the required models:")
        for model_id in missing_models:
            print(f"  eva models download {model_id}")
    else:
        print("Required models are already installed.")
    print("Then start the assistant:  eva run")
    print("Verify readiness any time:  eva doctor")
    return 0


def _cmd_start(args: argparse.Namespace) -> int:
    """Start the server as a background process (M5.5, ADR-026)."""
    from eva import service

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    existing = service.read_server_pid(paths)
    if existing is not None:
        print(f"Already running (PID {existing}). Use `eva status` or `eva restart`.")
        return 0
    pid = service.spawn_server(paths, host, port)
    url = service.health_url(host, port)
    print(f"Starting Edge Voice Assistant (PID {pid})...")
    if service.wait_until_healthy(url):
        print(f"Ready: {url.removesuffix('/api/v1/health')}/")
        return 0
    print("error: server did not become healthy — check `eva logs`", file=sys.stderr)
    return 1


def _cmd_stop(_args: argparse.Namespace) -> int:
    from eva import service

    paths = get_app_paths()
    paths.ensure_exists()
    pid = service.read_server_pid(paths)
    if pid is None:
        print("Not running.")
        return 0
    print(f"Stopping (PID {pid})...")
    if service.terminate_server(paths, pid):
        print("Stopped.")
        return 0
    print("error: process did not exit — kill it manually", file=sys.stderr)
    return 1


def _cmd_restart(args: argparse.Namespace) -> int:
    code = _cmd_stop(args)
    if code != 0:
        return code
    return _cmd_start(args)


def _cmd_status(_args: argparse.Namespace) -> int:
    import json
    import urllib.request

    from eva import service

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    pid = service.read_server_pid(paths)
    url = service.health_url(settings.server.host, settings.server.port)

    if pid is None:
        print("Server:  not running")
        return 1
    print(f"Server:  running (PID {pid})")
    if not service.probe_health(url):
        print("API:     not responding (still starting, or a port mismatch)")
        return 1
    print(f"API:     healthy at {url.removesuffix('/api/v1/health')}")
    try:
        status_url = url.replace("/health", "/engine/status")
        with urllib.request.urlopen(status_url, timeout=2) as response:
            body = json.loads(response.read())
        engine_state = f"running ({body['state']})" if body.get("running") else "stopped"
        print(f"Engine:  {engine_state}")
    except Exception:
        print("Engine:  unknown")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    from eva import service

    paths = get_app_paths()
    paths.ensure_exists()
    lines = service.newest_log_lines(paths, args.lines)
    if not lines:
        print(f"No log files in {paths.logs_dir}")
        return 0
    for line in lines:
        print(line)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the platform API server (ADR-017). The CLI's other commands are a
    separate, lighter-weight client of the same underlying services — `eva
    serve` is for the desktop app, a web UI, or any external integration."""
    import webbrowser

    import uvicorn

    from eva.conversation.personas import resolve_persona
    from eva.server import create_app
    from eva.server.static import ui_dist_dir

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    persona = resolve_persona(settings)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Edge Voice Assistant API on http://{host}:{port}")
    print(f"  Docs:      http://{host}:{port}/docs")
    print(f"  WebSocket: ws://{host}:{port}/api/v1/ws")
    print(f"  Persona:   {persona.display_name} ({persona.id})")
    print(f"  Voice:     {settings.tts.voice}")
    print("  (Memory/user-profile stats become available after POST /api/v1/engine/start)")
    if ui_dist_dir() is not None:
        print(f"  Web UI:    http://{display_host}:{port}/")
        if args.open:
            webbrowser.open(f"http://{display_host}:{port}/")
    elif args.open:
        print("  --open requested, but no built web UI was found — skipping.")
    uvicorn.run(create_app(paths), host=host, port=port, log_level="info")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    """Settings inspection/reset — the same `eva.config.service` functions the
    Settings API uses (no duplicated logic between CLI and server)."""
    from eva.config import service as settings_service

    paths = get_app_paths()
    paths.ensure_exists()

    if args.config_command == "show":
        current = load_settings(paths.settings_file)
        print(current.model_dump_json(indent=2))
        return 0

    if args.config_command == "schema":
        import json

        print(json.dumps(settings_service.get_schema(), indent=2))
        return 0

    if args.config_command == "reset":
        settings_service.reset_settings(paths)
        print(f"Settings reset to defaults and saved to {paths.settings_file}")
        return 0
    return 2


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

    p_doctor = sub.add_parser("doctor", help="Report dependency and model readiness")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_setup = sub.add_parser("setup", help="Install the LLM runtime for the detected hardware")
    variant_group = p_setup.add_mutually_exclusive_group()
    variant_group.add_argument(
        "--cpu", dest="variant", action="store_const", const="cpu", help="Force the CPU build"
    )
    variant_group.add_argument(
        "--cuda", dest="variant", action="store_const", const="cuda", help="Force the CUDA build"
    )
    p_setup.set_defaults(variant=None)
    p_setup.add_argument("--force", action="store_true", help="Reinstall even if present")
    p_setup.add_argument(
        "--dry-run", action="store_true", help="Print the install command without running it"
    )
    p_setup.set_defaults(func=_cmd_setup)

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

    p_run = sub.add_parser("run", help="Start the voice assistant (guided setup on first run)")
    p_run.add_argument("--yes", "-y", action="store_true", help="Auto-confirm setup prompts")
    p_run.add_argument(
        "--persona",
        default=None,
        help="Override the active persona for this session only (not saved)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_first = sub.add_parser("first-run", help="Run the guided setup wizard")
    p_first.add_argument("--yes", "-y", action="store_true", help="Auto-confirm setup prompts")
    p_first.add_argument(
        "--setup-only", action="store_true", help="Finish setup without starting the assistant"
    )
    p_first.set_defaults(func=_cmd_first_run)

    p_models = sub.add_parser("models", help="Manage models")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List available and installed models")
    p_dl = models_sub.add_parser("download", help="Download a model")
    p_dl.add_argument("model_id")
    p_rm = models_sub.add_parser("remove", help="Remove an installed model")
    p_rm.add_argument("model_id")
    p_info = models_sub.add_parser("info", help="Show the full model card")
    p_info.add_argument("model_id")
    p_use = models_sub.add_parser("use", help="Set a model as active (persists; profile→custom)")
    p_use.add_argument("model_id")
    p_models.set_defaults(func=_cmd_models)

    p_profiles = sub.add_parser("profiles", help="Model presets (Balanced, Fast, …)")
    profiles_sub = p_profiles.add_subparsers(dest="profiles_command", required=True)
    profiles_sub.add_parser("list", help="List presets and the active one")
    p_pset = profiles_sub.add_parser("set", help="Apply a preset for the detected hardware tier")
    p_pset.add_argument("preset_id")
    p_profiles.set_defaults(func=_cmd_profiles)

    p_personas = sub.add_parser("personas", help="Manage personas (M4)")
    personas_sub = p_personas.add_subparsers(dest="personas_command", required=True)
    personas_sub.add_parser("list", help="List built-in and custom personas")
    p_pshow = personas_sub.add_parser("show", help="Show one persona's full definition")
    p_pshow.add_argument("persona_id")
    p_pcreate = personas_sub.add_parser("create", help="Create or replace a custom persona")
    p_pcreate.add_argument("--id", required=True)
    p_pcreate.add_argument("--name", required=True, help="Display name")
    p_pcreate.add_argument("--prompt", required=True, help="System prompt")
    p_pcreate.add_argument(
        "--verbosity", choices=["minimal", "concise", "normal", "detailed"], default="normal"
    )
    p_pcreate.add_argument("--tone", default="neutral")
    p_pcreate.add_argument("--reasoning-style", dest="reasoning_style", default="direct")
    p_pcreate.add_argument("--temperature", type=float, default=None)
    p_pdel = personas_sub.add_parser("delete", help="Delete a custom persona")
    p_pdel.add_argument("persona_id")
    p_puse = personas_sub.add_parser("use", help="Set the active persona (saved; restart to apply)")
    p_puse.add_argument("persona_id")
    p_personas.set_defaults(func=_cmd_personas)

    p_users = sub.add_parser("users", help="Manage user profiles (M4)")
    users_sub = p_users.add_subparsers(dest="users_command", required=True)
    users_sub.add_parser("list", help="List user profiles")
    p_ushow = users_sub.add_parser("show", help="Show one user profile")
    p_ushow.add_argument("user_id")

    def _add_profile_fields(p: argparse.ArgumentParser, *, id_required: bool) -> None:
        if not id_required:
            p.add_argument("--id", default=None, help="Profile id (generated if omitted)")
        p.add_argument("--nickname", default=None)
        p.add_argument("--language", default=None, help="Preferred language code")
        p.add_argument("--voice", default=None, help="Preferred TTS voice id")
        p.add_argument("--llm-model", dest="llm_model", default=None)
        p.add_argument("--style", default=None, help="Conversation style")
        p.add_argument("--units", choices=["metric", "imperial"], default=None)
        p.add_argument("--timezone", default=None)

    p_ucreate = users_sub.add_parser("create", help="Create a user profile")
    _add_profile_fields(p_ucreate, id_required=False)
    p_uedit = users_sub.add_parser("edit", help="Update fields on an existing user profile")
    p_uedit.add_argument("user_id")
    _add_profile_fields(p_uedit, id_required=True)
    p_uactivate = users_sub.add_parser("activate", help="Set the active user profile")
    p_uactivate.add_argument("user_id")
    p_udel = users_sub.add_parser("delete", help="Delete a user profile")
    p_udel.add_argument("user_id")
    p_users.set_defaults(func=_cmd_users)

    p_voices = sub.add_parser("voices", help="Manage TTS voices (M4)")
    voices_sub = p_voices.add_subparsers(dest="voices_command", required=True)
    voices_sub.add_parser("list", help="List voices for the active TTS engine")
    p_vpreview = voices_sub.add_parser("preview", help="Synthesize a preview phrase to a WAV file")
    p_vpreview.add_argument("voice_id")
    p_vpreview.add_argument("--text", default="Hello, this is a preview of my voice.")
    p_vuse = voices_sub.add_parser("use", help="Set the active voice (saved; restart to apply)")
    p_vuse.add_argument("voice_id")
    p_voices.set_defaults(func=_cmd_voices)

    p_memory = sub.add_parser("memory", help="Manage persistent conversation memory (M4)")
    memory_sub = p_memory.add_subparsers(dest="memory_command", required=True)
    memory_sub.add_parser("stats", help="Aggregate memory statistics")
    p_mlist = memory_sub.add_parser("list", help="List conversations")
    p_mlist.add_argument("--include-archived", action="store_true")
    p_mshow = memory_sub.add_parser("show", help="Show all turns in a conversation")
    p_mshow.add_argument("conversation_id")
    p_mrename = memory_sub.add_parser("rename", help="Set a conversation's title")
    p_mrename.add_argument("conversation_id")
    p_mrename.add_argument("title")
    p_msearch = memory_sub.add_parser("search", help="Keyword search across memory")
    p_msearch.add_argument("query")
    p_msearch.add_argument("--conversation-id", dest="conversation_id", default=None)
    p_msearch.add_argument("--limit", type=int, default=20)
    p_mforget = memory_sub.add_parser("forget", help="Permanently delete one turn")
    p_mforget.add_argument("turn_id", type=int)
    p_mpin = memory_sub.add_parser("pin", help="Pin (boost retrieval) a turn")
    p_mpin.add_argument("turn_id", type=int)
    p_mpin.add_argument("--unset", action="store_true", help="Unpin instead")
    p_mfav = memory_sub.add_parser("favorite", help="Favorite (boost retrieval) a turn")
    p_mfav.add_argument("turn_id", type=int)
    p_mfav.add_argument("--unset", action="store_true", help="Unfavorite instead")
    p_march = memory_sub.add_parser("archive", help="Archive (hide) a conversation")
    p_march.add_argument("conversation_id")
    p_march.add_argument("--unset", action="store_true", help="Restore instead")
    p_mdel = memory_sub.add_parser("delete-conversation", help="Permanently delete a conversation")
    p_mdel.add_argument("conversation_id")
    p_mmerge = memory_sub.add_parser(
        "merge", help="Move all turns from one conversation into another"
    )
    p_mmerge.add_argument("source_id")
    p_mmerge.add_argument("target_id")
    p_mexport = memory_sub.add_parser("export", help="Export memory as JSON")
    p_mexport.add_argument("--conversation-id", dest="conversation_id", default=None)
    p_mexport.add_argument("--out", default=None, help="Write to a file instead of stdout")
    p_mimport = memory_sub.add_parser("import", help="Import a previously exported JSON snapshot")
    p_mimport.add_argument("file")
    p_mdelall = memory_sub.add_parser("delete-all", help="Delete ALL memory (privacy)")
    p_mdelall.add_argument("--yes", action="store_true", help="Confirm the deletion")
    p_msum = memory_sub.add_parser("summarize", help="LLM-summarize a conversation (loads the LLM)")
    p_msum.add_argument("conversation_id")
    p_memory.set_defaults(func=_cmd_memory)

    p_profile = sub.add_parser(
        "profile",
        help="Active user profile shortcut (M4; not to be confused with `profiles`, "
        "the hardware/model presets)",
    )
    profile_sub = p_profile.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("show", help="Show the active user profile")
    p_profuse = profile_sub.add_parser("use", help="Activate a user profile by id")
    p_profuse.add_argument("user_id")
    p_profile.set_defaults(func=_cmd_profile)

    p_bench = sub.add_parser("bench", help="Run the end-to-end pipeline benchmark (no mic needed)")
    p_bench.add_argument(
        "--text", default="What is the capital of Finland and what is it known for?"
    )
    p_bench.add_argument("--rounds", type=int, default=1)
    p_bench.set_defaults(func=_cmd_bench)

    p_serve = sub.add_parser(
        "serve", help="Run the platform API server (desktop/web/plugin backend)"
    )
    p_serve.add_argument("--host", default=None, help="Override settings.server.host")
    p_serve.add_argument("--port", type=int, default=None, help="Override settings.server.port")
    p_serve.add_argument(
        "--open", action="store_true", help="Open the web UI in the default browser once built"
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_start = sub.add_parser("start", help="Start the server in the background (M5.5)")
    p_start.add_argument("--host", default=None, help="Override settings.server.host")
    p_start.add_argument("--port", type=int, default=None, help="Override settings.server.port")
    p_start.set_defaults(func=_cmd_start)

    p_stop = sub.add_parser("stop", help="Stop the background server")
    p_stop.set_defaults(func=_cmd_stop)

    p_restart = sub.add_parser("restart", help="Restart the background server")
    p_restart.add_argument("--host", default=None, help="Override settings.server.host")
    p_restart.add_argument("--port", type=int, default=None, help="Override settings.server.port")
    p_restart.set_defaults(func=_cmd_restart)

    p_status = sub.add_parser("status", help="Show server process, API, and engine status")
    p_status.set_defaults(func=_cmd_status)

    p_logs = sub.add_parser("logs", help="Print the tail of the newest log file")
    p_logs.add_argument("--lines", type=int, default=50, help="How many lines to show")
    p_logs.set_defaults(func=_cmd_logs)

    p_config = sub.add_parser("config", help="Inspect or reset the settings document")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", help="Print the current settings as JSON")
    config_sub.add_parser("schema", help="Print the settings JSON Schema")
    config_sub.add_parser("reset", help="Reset settings to schema defaults")
    p_config.set_defaults(func=_cmd_config)

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
    except KeyboardInterrupt:
        # `run` handles Ctrl+C with its own cleanup/summary; this is the
        # backstop for every other command (bench, downloads, wizard
        # prompts, ...) so none of them ever surface a raw traceback.
        print("\nCancelled.", file=sys.stderr)
        return 130  # conventional exit code for SIGINT
    return result


if __name__ == "__main__":
    sys.exit(main())
