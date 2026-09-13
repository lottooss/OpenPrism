"""tools/bus/shm_observation_publisher.py
High-performance Python Shared Memory Publisher for OpenPrism Milestone M8-01.
Publishes canonical TargetObservationBatch payloads into the versioned lock-free seqlock ring.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from typing import Optional

from tools.bus.ring import (
    SHM_RING_MAGIC,
    SHM_VERSION_MAJOR,
    SHM_VERSION_MINOR,
    ChannelId,
    IpcControlHeaderData,
    SharedMemoryRingInspector,
)
from tools.bus.schema_bindings import (
    TargetObservationBatchData,
)

AOB_MAGIC_IDENTIFIER = 0x31424F41  # 'AOB1'
MAX_OBSERVATIONS = 64
OBSERVATION_SIZE = 104
BATCH_DATA_SIZE = 6752
SLOT_HEADER_SIZE = 64
SLOT_STRIDE = 6848
RING_CAPACITY = 16
HEADER_SIZE = 128
TOTAL_SEGMENT_SIZE = HEADER_SIZE + RING_CAPACITY * SLOT_STRIDE

# Binary struct formats (x64 MSVC ABI layout)
FMT_OBSERVATION = "<QQq16fB3xfI4x"
FMT_BATCH_PREFIX = "<HH12xQqII8xQQqqIII4x"


class SharedMemoryObservationPublisher:
    """Publishes canonical TargetObservationBatch records over Windows shared memory."""

    def __init__(
        self,
        shm_name: str = "aim_target_observations_shm",
        pipeline_run_id: int = 1,
        create: bool = True,
    ) -> None:
        self.shm_name = shm_name
        self.pipeline_run_id = pipeline_run_id
        self._seq = 0
        self._closed = False

        if create:
            self._shm = mmap.mmap(-1, TOTAL_SEGMENT_SIZE, tagname=shm_name)
            # Initialize 128-byte IpcControlHeader
            header_data = IpcControlHeaderData(
                magic=SHM_RING_MAGIC,
                version_major=SHM_VERSION_MAJOR,
                version_minor=SHM_VERSION_MINOR,
                header_size_bytes=HEADER_SIZE,
                channel_id=int(ChannelId.TARGET_OBSERVATION),
                pipeline_run_id=pipeline_run_id,
                writer_pid=os.getpid(),
                ring_capacity=RING_CAPACITY,
                slot_stride_bytes=SLOT_STRIDE,
                max_payload_bytes=BATCH_DATA_SIZE,
                writer_heartbeat_ns=time.perf_counter_ns(),
                producer_head_seq=0,
                consumer_tail_seq=0,
                health_flags=0,
            )
            raw_header = SharedMemoryRingInspector.serialize_control_header(header_data)
            self._shm[0:128] = raw_header
        else:
            self._shm = mmap.mmap(-1, TOTAL_SEGMENT_SIZE, tagname=shm_name)

    def update_heartbeat(self, now_ns: Optional[int] = None) -> None:
        """Pulses writer heartbeat in Cacheline 1 of control header."""
        if self._closed:
            return
        if now_ns is None:
            now_ns = time.perf_counter_ns()
        struct.pack_into("<q", self._shm, 64, now_ns)

    def write_observation_batch(
        self,
        batch: TargetObservationBatchData,
        publish_time_ns: Optional[int] = None,
    ) -> int:
        """Publishes a batch using lock-free seqlock synchronization."""
        if self._closed:
            raise RuntimeError("Cannot write to closed SharedMemoryObservationPublisher")

        if publish_time_ns is None:
            publish_time_ns = time.perf_counter_ns()

        self._seq += 1
        seq = self._seq
        idx = (seq - 1) & (RING_CAPACITY - 1)
        slot_offset = HEADER_SIZE + idx * SLOT_STRIDE
        data_offset = slot_offset + SLOT_HEADER_SIZE

        # 1. Mark slot as writing (seq * 2 - 1 is odd)
        struct.pack_into("<q", self._shm, slot_offset, seq * 2 - 1)

        # 2. Pack batch prefix (96 bytes)
        prefix_bytes = struct.pack(
            FMT_BATCH_PREFIX,
            batch.schema_major,
            batch.schema_minor,
            batch.header.sequence_id,
            batch.header.source_timestamp_ns,
            batch.header.pipeline_run_id,
            batch.header.flags,
            batch.source_id,
            batch.frame_id,
            batch.captured_at_ns,
            publish_time_ns,
            batch.source_width,
            batch.source_height,
            min(len(batch.targets), MAX_OBSERVATIONS),
        )
        self._shm[data_offset : data_offset + 96] = prefix_bytes

        # 3. Pack targets (up to MAX_OBSERVATIONS * 104 bytes)
        targets_offset = data_offset + 96
        for i, obs in enumerate(batch.targets[:MAX_OBSERVATIONS]):
            obs_bytes = struct.pack(
                FMT_OBSERVATION,
                obs.source_id,
                obs.frame_id,
                obs.captured_at_ns,
                obs.center_px.x,
                obs.center_px.y,
                obs.center_norm.x,
                obs.center_norm.y,
                obs.bbox_px.left,
                obs.bbox_px.top,
                obs.bbox_px.right,
                obs.bbox_px.bottom,
                obs.effective_radius_px,
                obs.confidence,
                obs.covariance_px2.xx,
                obs.covariance_px2.xy,
                obs.covariance_px2.yy,
                obs.velocity_px_per_s.x,
                obs.velocity_px_per_s.y,
                obs.velocity_confidence,
                int(obs.visibility),
                obs.target_value,
                obs.semantic_id,
            )
            obs_dest = targets_offset + i * OBSERVATION_SIZE
            self._shm[obs_dest : obs_dest + OBSERVATION_SIZE] = obs_bytes

        # 4. Update slot metadata
        # payload_size (offset + 8), magic_id (offset + 12), published_at_ns (offset + 16)
        struct.pack_into("<IIq", self._shm, slot_offset + 8, BATCH_DATA_SIZE, AOB_MAGIC_IDENTIFIER, publish_time_ns)

        # 5. Mark slot complete (seq * 2 is even)
        # seq_after (offset + 24), seq_before (offset + 0)
        struct.pack_into("<q", self._shm, slot_offset + 24, seq * 2)
        struct.pack_into("<q", self._shm, slot_offset, seq * 2)

        # 6. Update control header (writer_heartbeat_ns at offset 64, producer_head_seq at offset 72)
        struct.pack_into("<qq", self._shm, 64, publish_time_ns, seq)

        return seq

    def close(self) -> None:
        """Closes the shared memory mapping."""
        if not self._closed:
            self._closed = True
            self._shm.close()

    def __enter__(self) -> SharedMemoryObservationPublisher:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
