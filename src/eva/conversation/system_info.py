"""Permission-gated local system facts for the prompt (M5.3, ADR-025).

Answers "what time is it?", "how much RAM do I have?", "what OS is this?" —
questions a *local* assistant should trivially answer but a bare LLM cannot.
Each fact appears in the prompt only when its permission toggle
(`settings.permissions`) is on; with a permission off the assistant is told
(via the Context Builder's capability guidance) to attribute the gap to the
user's permission settings, not to inability.

Hardware detection shells out to system probes (`nvidia-smi`, WMI) — done
once per process and cached; hardware doesn't change mid-session. Date/time
is computed fresh on every call.
"""

from __future__ import annotations

import locale as locale_module
import platform
from datetime import datetime
from functools import lru_cache

from eva.config.settings import PermissionsSettings


@lru_cache(maxsize=1)
def _hardware_facts() -> dict[str, str]:
    """CPU/GPU/RAM facts, detected once per process. Failures degrade to an
    empty dict — a probe error must never break prompt composition."""
    try:
        from eva.hardware import detect_hardware

        report = detect_hardware()
        facts = {
            "cpu": f"{report.cpu.name} ({report.cpu.physical_cores} cores)",
            "ram": f"{report.memory.total_mb} MB installed",
        }
        if report.gpus:
            gpu = report.gpus[0]
            vram = f", {gpu.vram_total_mb} MB VRAM" if gpu.vram_total_mb else ""
            facts["gpu"] = f"{gpu.name}{vram}"
        else:
            facts["gpu"] = "no dedicated GPU detected"
        return facts
    except Exception:
        return {}


def system_facts_block(permissions: PermissionsSettings) -> str:
    """The prompt section listing permitted local facts, or "" if nothing is
    permitted. Called per turn — date/time must be current."""
    lines: list[str] = []
    now = datetime.now().astimezone()

    if permissions.date_time:
        lines.append(f"Current local date and time: {now.strftime('%A, %B %d, %Y at %H:%M')}")
    if permissions.timezone:
        lines.append(f"Timezone: {now.tzname() or 'unknown'} (UTC{now.strftime('%z')})")
    if permissions.locale:
        lang, _encoding = locale_module.getlocale()
        if lang:
            lines.append(f"System locale: {lang}")
    if permissions.os:
        lines.append(f"Operating system: {platform.system()} {platform.release()}")

    hardware = _hardware_facts()
    if permissions.cpu and "cpu" in hardware:
        lines.append(f"CPU: {hardware['cpu']}")
    if permissions.gpu and "gpu" in hardware:
        lines.append(f"GPU: {hardware['gpu']}")
    if permissions.ram and "ram" in hardware:
        lines.append(f"RAM: {hardware['ram']}")

    if not lines:
        return ""
    return (
        "Local system information (the user granted permission for each "
        "item below — answer questions about them directly):\n"
        + "\n".join(f"- {line}" for line in lines)
    )
