"""Audio device enumeration for settings/UI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.core.errors import AudioError


class AudioDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    is_default: bool
    max_input_channels: int
    max_output_channels: int
    host_api: str


def list_devices() -> tuple[list[AudioDevice], list[AudioDevice]]:
    """Return (input_devices, output_devices)."""
    import sounddevice as sd

    try:
        raw = sd.query_devices()
        host_apis = sd.query_hostapis()
        defaults = sd.default.device  # (input_index, output_index)
    except Exception as exc:
        raise AudioError(f"Cannot enumerate audio devices: {exc}") from exc

    inputs: list[AudioDevice] = []
    outputs: list[AudioDevice] = []
    for index, info in enumerate(raw):
        device = AudioDevice(
            index=index,
            name=str(info["name"]),
            is_default=index in tuple(defaults),
            max_input_channels=int(info["max_input_channels"]),
            max_output_channels=int(info["max_output_channels"]),
            host_api=str(host_apis[int(info["hostapi"])]["name"]),
        )
        if device.max_input_channels > 0:
            inputs.append(device)
        if device.max_output_channels > 0:
            outputs.append(device)
    return inputs, outputs
