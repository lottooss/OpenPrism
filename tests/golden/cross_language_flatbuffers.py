"""Bidirectional C++/Python golden-message verification for canonical bus v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import flatbuffers

from aim.bus.v1 import ActuationCommand
from aim.bus.v1 import CorrelationHeader
from aim.bus.v1 import FrameDescriptor
from aim.bus.v1 import FramePixelFormat


def build_python_frame() -> bytes:
    builder = flatbuffers.Builder(256)
    CorrelationHeader.CorrelationHeaderStart(builder)
    CorrelationHeader.CorrelationHeaderAddSequenceId(builder, 42)
    CorrelationHeader.CorrelationHeaderAddSourceTimestampNs(builder, 1_000_000_000)
    CorrelationHeader.CorrelationHeaderAddPipelineRunId(builder, 7)
    CorrelationHeader.CorrelationHeaderAddFlags(builder, 1)
    header = CorrelationHeader.CorrelationHeaderEnd(builder)

    FrameDescriptor.FrameDescriptorStart(builder)
    FrameDescriptor.FrameDescriptorAddSchemaMajor(builder, 1)
    FrameDescriptor.FrameDescriptorAddSchemaMinor(builder, 0)
    FrameDescriptor.FrameDescriptorAddHeader(builder, header)
    FrameDescriptor.FrameDescriptorAddFrameId(builder, 101)
    FrameDescriptor.FrameDescriptorAddCapturedAtNs(builder, 1_000_000_000)
    FrameDescriptor.FrameDescriptorAddWidth(builder, 1920)
    FrameDescriptor.FrameDescriptorAddHeight(builder, 1080)
    FrameDescriptor.FrameDescriptorAddFormat(builder, FramePixelFormat.FramePixelFormat.B8G8R8A8_UNORM)
    FrameDescriptor.FrameDescriptorAddPoolSlotIndex(builder, 3)
    FrameDescriptor.FrameDescriptorAddSharedNtHandle(builder, 0x12345678)
    FrameDescriptor.FrameDescriptorAddAdapterLuid(builder, 0x10688)
    FrameDescriptor.FrameDescriptorAddIsKeyframe(builder, False)
    frame = FrameDescriptor.FrameDescriptorEnd(builder)
    builder.Finish(frame, file_identifier=b"AFR1")
    return bytes(builder.Output())


def verify_cpp_command(buffer: bytes) -> None:
    if not ActuationCommand.ActuationCommand.ActuationCommandBufferHasIdentifier(buffer, 0, False):
        raise AssertionError("C++ message has the wrong file identifier")
    command = ActuationCommand.ActuationCommand.GetRootAs(buffer, 0)
    header = command.Header()
    transition = command.ButtonTransition()
    assert command.SchemaMajor() == 1
    assert command.SchemaMinor() == 0
    assert header is not None
    assert header.SequenceId() == 42
    assert header.SourceTimestampNs() == 1_000_000_000
    assert header.PipelineRunId() == 7
    assert header.Flags() == 1
    assert command.GeneratedAtNs() == 1_000_500_000
    assert command.DesiredApplyTimeNs() == 1_001_000_000
    assert command.DeltaXCounts() == 25
    assert command.DeltaYCounts() == -14
    assert transition is not None
    assert transition.Button() == 1
    assert transition.Action() == 1
    assert command.CancelSuperseded() is False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    python_frame_path = args.workspace / "frame-from-python.afr"
    cpp_command_path = args.workspace / "command-from-cpp.aac"

    python_frame = build_python_frame()
    python_frame_path.write_bytes(python_frame)
    subprocess.run(
        [
            str(args.native),
            "--verify-python-frame",
            str(python_frame_path),
            "--write-cpp-command",
            str(cpp_command_path),
        ],
        check=True,
    )

    cpp_command = cpp_command_path.read_bytes()
    verify_cpp_command(cpp_command)
    print(
        json.dumps(
            {
                "python_frame_sha256": hashlib.sha256(python_frame).hexdigest(),
                "cpp_command_sha256": hashlib.sha256(cpp_command).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
