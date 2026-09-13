"""Canonical bus package with lazy convenience exports.

The generated ``aim.bus.v1`` modules import through this package. Keeping the
tooling facade lazy avoids a package-initialization cycle while preserving the
existing ``from aim.bus import ...`` API.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ChannelId",
    "ShmHealthFlags",
    "LatestSpscRingSim",
    "SharedMemoryRingInspector",
    "IpcControlHeaderData",
    "IpcSlotHeaderData",
    "FrameDescriptorData",
    "TargetObservationBatchData",
    "TargetObservationData",
    "TrackedTargetBatchData",
    "TrackedTargetData",
    "AimIntentData",
    "ActuationCommandData",
    "ButtonTransitionData",
    "HotLoopTelemetryEventData",
    "StageTimingData",
    "CorrelationHeader",
    "CorrelationFlags",
    "SchemaValidationError",
    "FramePixelFormat",
    "TrackState",
    "AimMode",
    "MouseButton",
    "ButtonAction",
    "PipelineStageCode",
    "Vec2f",
    "Visibility",
    "serialize_frame_descriptor",
    "serialize_actuation_command",
    "deserialize_frame_descriptor",
    "deserialize_actuation_command",
    "validate_file_identifier",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from tools import bus as tooling_bus

    return getattr(tooling_bus, name)
