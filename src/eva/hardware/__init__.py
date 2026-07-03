"""Hardware detection and profile recommendation."""

from eva.hardware.detect import (
    CpuInfo,
    GpuInfo,
    HardwareReport,
    MemoryInfo,
    detect_hardware,
)
from eva.hardware.profiles import PROFILES, HardwareProfile, recommend_profile

__all__ = [
    "PROFILES",
    "CpuInfo",
    "GpuInfo",
    "HardwareProfile",
    "HardwareReport",
    "MemoryInfo",
    "detect_hardware",
    "recommend_profile",
]
