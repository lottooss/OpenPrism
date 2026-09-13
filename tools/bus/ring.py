"""tools/bus/ring.py
Python Reference Simulation, Concurrency Model, and Shared Memory Inspector for Aim Bus Ring.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")

SHM_RING_MAGIC = 0x524D5341  # 'ASMR'
SHM_VERSION_MAJOR = 1
SHM_VERSION_MINOR = 0


class ChannelId(IntEnum):
    UNKNOWN = 0
    FRAME_DESCRIPTOR = 1
    TARGET_OBSERVATION = 2
    TRACKED_TARGET = 3
    AIM_INTENT = 4
    ACTUATION_COMMAND = 5
    TELEMETRY_EVENT = 6


class ShmHealthFlags(IntEnum):
    OK = 0
    WRITER_ACTIVE = 1 << 0
    WRITER_STALE = 1 << 1
    WRITER_CRASHED = 1 << 2
    BUFFER_OVERRUN = 1 << 3
    SCHEMA_MISMATCH = 1 << 4


@dataclass
class BusSlot(Generic[T]):
    sequence: int = 0
    data: Optional[T] = None


@dataclass
class IpcControlHeaderData:
    magic: int
    version_major: int
    version_minor: int
    header_size_bytes: int
    channel_id: int
    pipeline_run_id: int
    writer_pid: int
    ring_capacity: int
    slot_stride_bytes: int
    max_payload_bytes: int
    writer_heartbeat_ns: int
    producer_head_seq: int
    consumer_tail_seq: int
    health_flags: int


@dataclass
class IpcSlotHeaderData:
    seq_before: int
    payload_size: int
    magic_identifier: int
    published_at_ns: int
    seq_after: int


class LatestSpscRingSim(Generic[T]):
    """Python reference implementation of LatestSpscRing with latest-message-wins semantics."""

    def __init__(self, capacity: int = 16) -> None:
        if capacity < 2 or (capacity & (capacity - 1)) != 0:
            raise ValueError(f"Capacity must be a power of two >= 2, got {capacity}")
        self._capacity = capacity
        self._index_mask = capacity - 1
        self._slots: List[BusSlot[T]] = [BusSlot[T]() for _ in range(capacity)]
        self._producer_seq = 0
        self._consumer_last_seq = 0
        self._total_reads = 0
        self._total_dropped = 0
        self._torn_reads = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def latest_sequence(self) -> int:
        return self._producer_seq

    @property
    def last_consumed_sequence(self) -> int:
        return self._consumer_last_seq

    @property
    def dropped_count(self) -> int:
        return self._total_dropped

    @property
    def total_reads(self) -> int:
        return self._total_reads

    @property
    def total_produced(self) -> int:
        return self._producer_seq

    def empty(self) -> bool:
        return self._producer_seq == 0

    def push(self, item: T) -> int:
        """Publishes an item to the ring, overwriting oldest slot if full."""
        self._producer_seq += 1
        seq = self._producer_seq
        idx = (seq - 1) & self._index_mask

        slot = self._slots[idx]
        slot.sequence = seq
        slot.data = item
        return seq

    def try_read_latest(self) -> Tuple[bool, Optional[T], int, int]:
        """Reads newest message.

        Returns:
            Tuple of (success, item, sequence, dropped_since_last)
        """
        head = self._producer_seq
        if head <= self._consumer_last_seq or head == 0:
            return False, None, self._consumer_last_seq, 0

        idx = (head - 1) & self._index_mask
        slot = self._slots[idx]

        if slot.sequence != head:
            self._torn_reads += 1
            return False, None, self._consumer_last_seq, 0

        dropped = 0
        if self._consumer_last_seq > 0 and head > self._consumer_last_seq + 1:
            dropped = head - self._consumer_last_seq - 1
            self._total_dropped += dropped
        elif self._consumer_last_seq == 0 and head > 1:
            dropped = head - 1
            self._total_dropped += dropped

        self._consumer_last_seq = head
        self._total_reads += 1
        return True, slot.data, head, dropped

    def try_read_sequence(self, target_seq: int) -> Tuple[bool, Optional[T]]:
        """Queries a specific sequence if still within ring window."""
        if target_seq <= 0 or target_seq > self._producer_seq:
            return False, None
        if self._producer_seq >= self._capacity and target_seq <= self._producer_seq - self._capacity:
            return False, None  # Overwritten

        idx = (target_seq - 1) & self._index_mask
        slot = self._slots[idx]
        if slot.sequence != target_seq:
            return False, None

        return True, slot.data

    def reset(self) -> None:
        self._producer_seq = 0
        self._consumer_last_seq = 0
        self._total_reads = 0
        self._total_dropped = 0
        self._torn_reads = 0
        for slot in self._slots:
            slot.sequence = 0
            slot.data = None


# Type alias for API parity with C++ LatestSpscRing
LatestSpscRing = LatestSpscRingSim


class SharedMemoryRingInspector:
    """Inspector for parsing and verifying Windows shared memory ring headers and slots."""

    HEADER_FORMAT = "<IHHIIIIIII28s"  # 64 bytes Cacheline 0
    SYNC_FORMAT = "<qqqI36s"         # 64 bytes Cacheline 1
    SLOT_HDR_FORMAT = "<qIIqq32s"    # 64 bytes Slot Header

    @classmethod
    def parse_control_header(cls, buffer: bytes) -> IpcControlHeaderData:
        """Parses a 128-byte IpcControlHeader binary structure."""
        if len(buffer) < 128:
            raise ValueError(f"Buffer too short for IpcControlHeader: {len(buffer)} < 128")

        c0 = struct.unpack_from(cls.HEADER_FORMAT, buffer, 0)
        magic, v_maj, v_min, hdr_size, chan, run_id, pid, cap, stride, max_payload, _ = c0

        c1 = struct.unpack_from(cls.SYNC_FORMAT, buffer, 64)
        hb_ns, head_seq, tail_seq, flags, _ = c1

        return IpcControlHeaderData(
            magic=magic,
            version_major=v_maj,
            version_minor=v_min,
            header_size_bytes=hdr_size,
            channel_id=chan,
            pipeline_run_id=run_id,
            writer_pid=pid,
            ring_capacity=cap,
            slot_stride_bytes=stride,
            max_payload_bytes=max_payload,
            writer_heartbeat_ns=hb_ns,
            producer_head_seq=head_seq,
            consumer_tail_seq=tail_seq,
            health_flags=flags,
        )

    @classmethod
    def serialize_control_header(cls, hdr: IpcControlHeaderData) -> bytes:
        """Serializes IpcControlHeaderData to exact 128-byte binary layout."""
        c0 = struct.pack(
            cls.HEADER_FORMAT,
            hdr.magic,
            hdr.version_major,
            hdr.version_minor,
            hdr.header_size_bytes,
            hdr.channel_id,
            hdr.pipeline_run_id,
            hdr.writer_pid,
            hdr.ring_capacity,
            hdr.slot_stride_bytes,
            hdr.max_payload_bytes,
            b"\x00" * 28,
        )
        c1 = struct.pack(
            cls.SYNC_FORMAT,
            hdr.writer_heartbeat_ns,
            hdr.producer_head_seq,
            hdr.consumer_tail_seq,
            hdr.health_flags,
            b"\x00" * 36,
        )
        return c0 + c1

    @classmethod
    def parse_slot_header(cls, buffer: bytes, offset: int = 0) -> IpcSlotHeaderData:
        """Parses a 64-byte IpcSlotHeader binary structure."""
        if len(buffer) < offset + 64:
            raise ValueError(f"Buffer too short for IpcSlotHeader at offset {offset}")

        s_before, p_size, magic_id, pub_ns, s_after, _ = struct.unpack_from(cls.SLOT_HDR_FORMAT, buffer, offset)
        return IpcSlotHeaderData(
            seq_before=s_before,
            payload_size=p_size,
            magic_identifier=magic_id,
            published_at_ns=pub_ns,
            seq_after=s_after,
        )

    @classmethod
    def serialize_slot_header(cls, slot: IpcSlotHeaderData) -> bytes:
        """Serializes IpcSlotHeaderData to 64-byte binary layout."""
        return struct.pack(
            cls.SLOT_HDR_FORMAT,
            slot.seq_before,
            slot.payload_size,
            slot.magic_identifier,
            slot.published_at_ns,
            slot.seq_after,
            b"\x00" * 32,
        )
