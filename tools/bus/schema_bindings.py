"""tools/bus/schema_bindings.py
Python FlatBuffers builders, parsers, and validation utilities for OpenPrism Bus v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import struct
from typing import Any, List, cast
import flatbuffers

from aim.bus.v1 import ActuationCommand as FbActuationCommand
from aim.bus.v1 import ButtonTransition as FbButtonTransition
from aim.bus.v1 import CorrelationHeader as FbCorrelationHeader
from aim.bus.v1 import FrameDescriptor as FbFrameDescriptor
from aim.bus.v1 import TargetObservation as FbTargetObservation
from aim.bus.v1 import TargetObservationBatch as FbTargetObservationBatch
from aim.bus.v1 import VelocityHint as FbVelocityHint

try:
    from aim.bus.v1.BBoxf import CreateBBoxf
except ImportError:
    from aim.bus.v1.BBoxf import CreateBboxf as CreateBBoxf  # type: ignore[attr-defined, no-redef]
from aim.bus.v1.Covariance2f import CreateCovariance2f
from aim.bus.v1.Vec2f import CreateVec2f


# ==============================================================================
# Enumerations & Primitives
# ==============================================================================


class Visibility(IntEnum):
    VISIBLE = 0
    PARTIAL = 1
    PREDICTED = 2


class CoordinateSpace(IntEnum):
    PIXEL_TOP_LEFT = 0
    NORMALIZED_CENTER = 1
    CAMERA_SPHERICAL = 2


class MouseButton(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    MIDDLE = 3
    X1 = 4
    X2 = 5


class ButtonAction(IntEnum):
    NONE = 0
    PRESS = 1
    RELEASE = 2
    CLICK = 3


class CorrelationFlags(IntEnum):
    NONE = 0
    SYNTHETIC = 1
    WARMUP = 2
    DROPPED = 4
    TRACE_VERBOSE = 8


class FramePixelFormat(IntEnum):
    UNKNOWN = 0
    B8G8R8A8_UNORM = 1
    R8G8B8A8_UNORM = 2
    R16G16B16A16_FLOAT = 3
    NV12 = 4


class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    OCCLUDED = 2
    COASTING = 3
    LOST = 4


class AimMode(IntEnum):
    IDLE = 0
    TRACKING = 1
    FLICK = 2
    MICRO_CORRECTION = 3
    EMERGENCY_HOLD = 4


class PipelineStageCode(IntEnum):
    CAPTURE_ARRIVAL = 0
    PREPROCESS = 1
    PERCEPTION_INFER = 2
    TRACKING_KALMAN = 3
    AIM_POLICY_PREDICT = 4
    TRAJECTORY_PLAN = 5
    ACTUATION_DISPATCH = 6


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(slots=True)
class Vec2f:
    x: float
    y: float


@dataclass(slots=True)
class BBoxf:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(slots=True)
class Covariance2f:
    xx: float
    xy: float
    yy: float


@dataclass(slots=True)
class CorrelationHeader:
    sequence_id: int
    source_timestamp_ns: int
    pipeline_run_id: int
    flags: int = 0


@dataclass(slots=True)
class FrameDescriptorData:
    header: CorrelationHeader
    frame_id: int
    captured_at_ns: int
    width: int = 1920
    height: int = 1080
    format: FramePixelFormat = FramePixelFormat.B8G8R8A8_UNORM
    pool_slot_index: int = 0
    shared_nt_handle: int = 0
    adapter_luid: int = 0
    is_keyframe: bool = True
    schema_major: int = 1
    schema_minor: int = 0


@dataclass(slots=True)
class TargetObservationData:
    source_id: int
    frame_id: int
    captured_at_ns: int
    center_px: Vec2f
    center_norm: Vec2f
    bbox_px: BBoxf
    effective_radius_px: float
    confidence: float
    covariance_px2: Covariance2f
    velocity_px_per_s: Vec2f = field(default_factory=lambda: Vec2f(0.0, 0.0))
    velocity_confidence: float = 0.0
    visibility: Visibility = Visibility.VISIBLE
    target_value: float = 1.0
    semantic_id: int = 0


@dataclass(slots=True)
class TargetObservationBatchData:
    header: CorrelationHeader
    source_id: int
    frame_id: int
    captured_at_ns: int
    published_at_ns: int
    source_width: int = 1920
    source_height: int = 1080
    targets: List[TargetObservationData] = field(default_factory=list)
    schema_major: int = 1
    schema_minor: int = 0


@dataclass(slots=True)
class TrackedTargetData:
    track_id: int
    state: TrackState
    total_visible_frames: int
    total_missed_frames: int
    filtered_center_px: Vec2f
    filtered_velocity_px_per_s: Vec2f
    filtered_acceleration_px_per_s2: Vec2f
    covariance_px2: Covariance2f
    predicted_center_px: Vec2f
    prediction_time_ns: int
    effective_radius_px: float
    confidence: float
    target_value: float = 1.0
    semantic_id: int = 0


@dataclass(slots=True)
class TrackedTargetBatchData:
    header: CorrelationHeader
    frame_id: int
    timestamp_ns: int
    tracks: List[TrackedTargetData] = field(default_factory=list)
    schema_major: int = 1
    schema_minor: int = 0


@dataclass(slots=True)
class AimIntentData:
    header: CorrelationHeader
    target_track_id: int
    mode: AimMode
    target_aim_px: Vec2f
    target_aim_norm: Vec2f
    lead_offset_px: Vec2f
    error_distance_px: float
    authorize_fire: bool = False
    confidence: float = 1.0
    utility_score: float = 1.0
    command_deadline_ns: int = 0
    schema_major: int = 1
    schema_minor: int = 0


@dataclass(slots=True)
class ButtonTransitionData:
    button: MouseButton = MouseButton.NONE
    action: ButtonAction = ButtonAction.NONE


@dataclass(slots=True)
class ActuationCommandData:
    header: CorrelationHeader
    generated_at_ns: int
    desired_apply_time_ns: int
    delta_x_counts: int
    delta_y_counts: int
    button_transition: ButtonTransitionData = field(default_factory=ButtonTransitionData)
    cancel_superseded: bool = True
    schema_major: int = 1
    schema_minor: int = 0


@dataclass(slots=True)
class StageTimingData:
    stage: PipelineStageCode
    start_ns: int
    end_ns: int
    duration_ns: int


@dataclass(slots=True)
class HotLoopTelemetryEventData:
    header: CorrelationHeader
    capture_arrival_ns: int
    actuation_dispatch_ns: int
    total_latency_ns: int
    is_stale_dropped: bool = False
    queue_depth: int = 0
    stages: List[StageTimingData] = field(default_factory=list)
    schema_major: int = 1
    schema_minor: int = 0


# ==============================================================================
# FlatBuffers Encoders & Decoders (Deterministic Binary Serializers)
# ==============================================================================

FRAME_DESCRIPTOR_IDENTIFIER = b"AFR1"
ACTUATION_COMMAND_IDENTIFIER = b"AAC1"
TARGET_OBSERVATION_BATCH_IDENTIFIER = b"AOB1"
SCHEMA_MAJOR = 1

SCHEMA_MINOR = 0


class SchemaValidationError(ValueError):
    """Raised when a serialized bus message fails envelope or version checks."""


def _build_correlation_header(builder: flatbuffers.Builder, data: CorrelationHeader) -> int:
    FbCorrelationHeader.CorrelationHeaderStart(builder)
    FbCorrelationHeader.CorrelationHeaderAddSequenceId(builder, data.sequence_id)
    FbCorrelationHeader.CorrelationHeaderAddSourceTimestampNs(builder, data.source_timestamp_ns)
    FbCorrelationHeader.CorrelationHeaderAddPipelineRunId(builder, data.pipeline_run_id)
    FbCorrelationHeader.CorrelationHeaderAddFlags(builder, data.flags)
    return cast(int, FbCorrelationHeader.CorrelationHeaderEnd(builder))


def _read_correlation_header(
    header: FbCorrelationHeader.CorrelationHeader | None,
) -> CorrelationHeader:
    if header is None:
        raise SchemaValidationError("required correlation header is missing")
    return CorrelationHeader(
        sequence_id=int(header.SequenceId()),
        source_timestamp_ns=int(header.SourceTimestampNs()),
        pipeline_run_id=int(header.PipelineRunId()),
        flags=int(header.Flags()),
    )


def _validate_version(major: int, minor: int) -> None:
    if major != SCHEMA_MAJOR or minor > SCHEMA_MINOR:
        raise SchemaValidationError(
            f"unsupported schema version {major}.{minor}; host supports {SCHEMA_MAJOR}.{SCHEMA_MINOR}"
        )


def serialize_frame_descriptor(data: FrameDescriptorData) -> bytes:
    """Serializes FrameDescriptor with file identifier 'AFR1'."""
    builder = flatbuffers.Builder(256)
    header = _build_correlation_header(builder, data.header)

    FbFrameDescriptor.FrameDescriptorStart(builder)
    FbFrameDescriptor.FrameDescriptorAddSchemaMajor(builder, data.schema_major)
    FbFrameDescriptor.FrameDescriptorAddSchemaMinor(builder, data.schema_minor)
    FbFrameDescriptor.FrameDescriptorAddHeader(builder, header)
    FbFrameDescriptor.FrameDescriptorAddFrameId(builder, data.frame_id)
    FbFrameDescriptor.FrameDescriptorAddCapturedAtNs(builder, data.captured_at_ns)
    FbFrameDescriptor.FrameDescriptorAddWidth(builder, data.width)
    FbFrameDescriptor.FrameDescriptorAddHeight(builder, data.height)
    FbFrameDescriptor.FrameDescriptorAddFormat(builder, cast(Any, int(data.format)))
    FbFrameDescriptor.FrameDescriptorAddPoolSlotIndex(builder, data.pool_slot_index)
    FbFrameDescriptor.FrameDescriptorAddSharedNtHandle(builder, data.shared_nt_handle)
    FbFrameDescriptor.FrameDescriptorAddAdapterLuid(builder, data.adapter_luid)
    FbFrameDescriptor.FrameDescriptorAddIsKeyframe(builder, data.is_keyframe)
    root = FbFrameDescriptor.FrameDescriptorEnd(builder)

    builder.Finish(root, file_identifier=FRAME_DESCRIPTOR_IDENTIFIER)
    return bytes(builder.Output())


def deserialize_frame_descriptor(buf: bytes) -> FrameDescriptorData:
    """Decode and validate a generated ``FrameDescriptor`` FlatBuffer."""
    if not validate_file_identifier(buf, FRAME_DESCRIPTOR_IDENTIFIER):
        raise SchemaValidationError("invalid FrameDescriptor file identifier")
    try:
        root = FbFrameDescriptor.FrameDescriptor.GetRootAs(buf, 0)
        _validate_version(int(root.SchemaMajor()), int(root.SchemaMinor()))
        return FrameDescriptorData(
            header=_read_correlation_header(root.Header()),
            frame_id=int(root.FrameId()),
            captured_at_ns=int(root.CapturedAtNs()),
            width=int(root.Width()),
            height=int(root.Height()),
            format=FramePixelFormat(int(root.Format())),
            pool_slot_index=int(root.PoolSlotIndex()),
            shared_nt_handle=int(root.SharedNtHandle()),
            adapter_luid=int(root.AdapterLuid()),
            is_keyframe=bool(root.IsKeyframe()),
            schema_major=int(root.SchemaMajor()),
            schema_minor=int(root.SchemaMinor()),
        )
    except (IndexError, struct.error, TypeError, ValueError) as exc:
        if isinstance(exc, SchemaValidationError):
            raise
        raise SchemaValidationError("malformed FrameDescriptor buffer") from exc


def serialize_actuation_command(data: ActuationCommandData) -> bytes:
    """Serializes ActuationCommand with file identifier 'AAC1'."""
    builder = flatbuffers.Builder(256)
    header = _build_correlation_header(builder, data.header)

    FbButtonTransition.ButtonTransitionStart(builder)
    FbButtonTransition.ButtonTransitionAddButton(
        builder, cast(Any, int(data.button_transition.button))
    )
    FbButtonTransition.ButtonTransitionAddAction(
        builder, cast(Any, int(data.button_transition.action))
    )
    transition = FbButtonTransition.ButtonTransitionEnd(builder)

    FbActuationCommand.ActuationCommandStart(builder)
    FbActuationCommand.ActuationCommandAddSchemaMajor(builder, data.schema_major)
    FbActuationCommand.ActuationCommandAddSchemaMinor(builder, data.schema_minor)
    FbActuationCommand.ActuationCommandAddHeader(builder, header)
    FbActuationCommand.ActuationCommandAddGeneratedAtNs(builder, data.generated_at_ns)
    FbActuationCommand.ActuationCommandAddDesiredApplyTimeNs(builder, data.desired_apply_time_ns)
    FbActuationCommand.ActuationCommandAddDeltaXCounts(builder, data.delta_x_counts)
    FbActuationCommand.ActuationCommandAddDeltaYCounts(builder, data.delta_y_counts)
    FbActuationCommand.ActuationCommandAddButtonTransition(builder, transition)
    FbActuationCommand.ActuationCommandAddCancelSuperseded(builder, data.cancel_superseded)
    root = FbActuationCommand.ActuationCommandEnd(builder)

    builder.Finish(root, file_identifier=ACTUATION_COMMAND_IDENTIFIER)
    return bytes(builder.Output())


def deserialize_actuation_command(buf: bytes) -> ActuationCommandData:
    """Decode and validate a generated ``ActuationCommand`` FlatBuffer."""
    if not validate_file_identifier(buf, ACTUATION_COMMAND_IDENTIFIER):
        raise SchemaValidationError("invalid ActuationCommand file identifier")
    try:
        root = FbActuationCommand.ActuationCommand.GetRootAs(buf, 0)
        _validate_version(int(root.SchemaMajor()), int(root.SchemaMinor()))
        transition = root.ButtonTransition()
        if transition is None:
            raise SchemaValidationError("required button transition is missing")
        return ActuationCommandData(
            header=_read_correlation_header(root.Header()),
            generated_at_ns=int(root.GeneratedAtNs()),
            desired_apply_time_ns=int(root.DesiredApplyTimeNs()),
            delta_x_counts=int(root.DeltaXCounts()),
            delta_y_counts=int(root.DeltaYCounts()),
            button_transition=ButtonTransitionData(
                button=MouseButton(int(transition.Button())),
                action=ButtonAction(int(transition.Action())),
            ),
            cancel_superseded=bool(root.CancelSuperseded()),
            schema_major=int(root.SchemaMajor()),
            schema_minor=int(root.SchemaMinor()),
        )
    except (IndexError, struct.error, TypeError, ValueError) as exc:
        if isinstance(exc, SchemaValidationError):
            raise
        raise SchemaValidationError("malformed ActuationCommand buffer") from exc


def serialize_target_observation_batch(data: TargetObservationBatchData) -> bytes:
    """Serializes TargetObservationBatch with file identifier 'AOB1'."""
    builder = flatbuffers.Builder(1024)

    target_offsets: List[int] = []
    for obs in data.targets:
        vel_offset = 0
        if (
            obs.velocity_confidence > 0.0
            or obs.velocity_px_per_s.x != 0.0
            or obs.velocity_px_per_s.y != 0.0
        ):
            FbVelocityHint.VelocityHintStart(builder)
            v_vec = CreateVec2f(builder, obs.velocity_px_per_s.x, obs.velocity_px_per_s.y)
            FbVelocityHint.VelocityHintAddPixelsPerSecond(builder, v_vec)
            FbVelocityHint.VelocityHintAddConfidence(builder, obs.velocity_confidence)
            vel_offset = FbVelocityHint.VelocityHintEnd(builder)

        FbTargetObservation.TargetObservationStart(builder)
        FbTargetObservation.TargetObservationAddSourceId(builder, obs.source_id)
        FbTargetObservation.TargetObservationAddFrameId(builder, obs.frame_id)
        FbTargetObservation.TargetObservationAddCapturedAtNs(builder, obs.captured_at_ns)

        cpx = CreateVec2f(builder, obs.center_px.x, obs.center_px.y)
        FbTargetObservation.TargetObservationAddCenterPx(builder, cpx)

        cnorm = CreateVec2f(builder, obs.center_norm.x, obs.center_norm.y)
        FbTargetObservation.TargetObservationAddCenterNorm(builder, cnorm)

        bbox = CreateBBoxf(
            builder, obs.bbox_px.left, obs.bbox_px.top, obs.bbox_px.right, obs.bbox_px.bottom
        )
        FbTargetObservation.TargetObservationAddBboxPx(builder, bbox)

        FbTargetObservation.TargetObservationAddEffectiveRadiusPx(builder, obs.effective_radius_px)
        FbTargetObservation.TargetObservationAddConfidence(builder, obs.confidence)

        cov = CreateCovariance2f(
            builder, obs.covariance_px2.xx, obs.covariance_px2.xy, obs.covariance_px2.yy
        )
        FbTargetObservation.TargetObservationAddCovariancePx2(builder, cov)

        if vel_offset != 0:
            FbTargetObservation.TargetObservationAddVelocity(builder, vel_offset)

        FbTargetObservation.TargetObservationAddVisibility(builder, cast(Any, int(obs.visibility)))
        FbTargetObservation.TargetObservationAddTargetValue(builder, obs.target_value)
        FbTargetObservation.TargetObservationAddSemanticId(builder, obs.semantic_id)

        target_offsets.append(FbTargetObservation.TargetObservationEnd(builder))

    FbTargetObservationBatch.TargetObservationBatchStartTargetsVector(builder, len(target_offsets))
    for off in reversed(target_offsets):
        builder.PrependUOffsetTRelative(off)
    targets_vec = builder.EndVector()

    header = _build_correlation_header(builder, data.header)

    FbTargetObservationBatch.TargetObservationBatchStart(builder)
    FbTargetObservationBatch.TargetObservationBatchAddSchemaMajor(builder, data.schema_major)
    FbTargetObservationBatch.TargetObservationBatchAddSchemaMinor(builder, data.schema_minor)
    FbTargetObservationBatch.TargetObservationBatchAddHeader(builder, header)
    FbTargetObservationBatch.TargetObservationBatchAddSourceId(builder, data.source_id)
    FbTargetObservationBatch.TargetObservationBatchAddFrameId(builder, data.frame_id)
    FbTargetObservationBatch.TargetObservationBatchAddCapturedAtNs(builder, data.captured_at_ns)
    FbTargetObservationBatch.TargetObservationBatchAddPublishedAtNs(builder, data.published_at_ns)
    FbTargetObservationBatch.TargetObservationBatchAddSourceWidth(builder, data.source_width)
    FbTargetObservationBatch.TargetObservationBatchAddSourceHeight(builder, data.source_height)
    FbTargetObservationBatch.TargetObservationBatchAddTargets(builder, targets_vec)
    root = FbTargetObservationBatch.TargetObservationBatchEnd(builder)

    builder.Finish(root, file_identifier=TARGET_OBSERVATION_BATCH_IDENTIFIER)
    return bytes(builder.Output())


def deserialize_target_observation_batch(buf: bytes) -> TargetObservationBatchData:
    """Decode and validate a generated ``TargetObservationBatch`` FlatBuffer."""
    if not validate_file_identifier(buf, TARGET_OBSERVATION_BATCH_IDENTIFIER):
        raise SchemaValidationError("invalid TargetObservationBatch file identifier")
    try:
        root = FbTargetObservationBatch.TargetObservationBatch.GetRootAs(buf, 0)
        _validate_version(int(root.SchemaMajor()), int(root.SchemaMinor()))

        targets: List[TargetObservationData] = []
        for i in range(root.TargetsLength()):
            t = root.Targets(i)
            if t is None:
                continue

            cpx = t.CenterPx()
            center_px = Vec2f(float(cpx.X()), float(cpx.Y())) if cpx else Vec2f(0.0, 0.0)

            cnorm = t.CenterNorm()
            center_norm = Vec2f(float(cnorm.X()), float(cnorm.Y())) if cnorm else Vec2f(0.0, 0.0)

            bb = t.BboxPx()
            bbox = (
                BBoxf(float(bb.Left()), float(bb.Top()), float(bb.Right()), float(bb.Bottom()))
                if bb
                else BBoxf(0.0, 0.0, 0.0, 0.0)
            )

            cov = t.CovariancePx2()
            covariance = (
                Covariance2f(float(cov.Xx()), float(cov.Xy()), float(cov.Yy()))
                if cov
                else Covariance2f(0.0, 0.0, 0.0)
            )

            vel_hint = t.Velocity()
            if vel_hint is not None and vel_hint.PixelsPerSecond() is not None:
                pps = vel_hint.PixelsPerSecond()
                vel_px = Vec2f(float(pps.X()), float(pps.Y())) if pps else Vec2f(0.0, 0.0)
                vel_conf = float(vel_hint.Confidence())
            else:
                vel_px = Vec2f(0.0, 0.0)
                vel_conf = 0.0

            targets.append(
                TargetObservationData(
                    source_id=int(t.SourceId()),
                    frame_id=int(t.FrameId()),
                    captured_at_ns=int(t.CapturedAtNs()),
                    center_px=center_px,
                    center_norm=center_norm,
                    bbox_px=bbox,
                    effective_radius_px=float(t.EffectiveRadiusPx()),
                    confidence=float(t.Confidence()),
                    covariance_px2=covariance,
                    velocity_px_per_s=vel_px,
                    velocity_confidence=vel_conf,
                    visibility=Visibility(int(t.Visibility())),
                    target_value=float(t.TargetValue()),
                    semantic_id=int(t.SemanticId()),
                )
            )

        return TargetObservationBatchData(
            header=_read_correlation_header(root.Header()),
            source_id=int(root.SourceId()),
            frame_id=int(root.FrameId()),
            captured_at_ns=int(root.CapturedAtNs()),
            published_at_ns=int(root.PublishedAtNs()),
            source_width=int(root.SourceWidth()),
            source_height=int(root.SourceHeight()),
            targets=targets,
            schema_major=int(root.SchemaMajor()),
            schema_minor=int(root.SchemaMinor()),
        )
    except (IndexError, struct.error, TypeError, ValueError) as exc:
        if isinstance(exc, SchemaValidationError):
            raise
        raise SchemaValidationError("malformed TargetObservationBatch buffer") from exc


def validate_file_identifier(buf: bytes, expected_ident: bytes) -> bool:
    """Validates 4-byte FlatBuffers file identifier at offset 4..8."""
    if len(expected_ident) != 4 or len(buf) < 8:
        return False
    return buf[4:8] == expected_ident
