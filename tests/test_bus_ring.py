"""tests/test_bus_ring.py
Unit tests for OpenPrism Bounded Latest-Wins Bus Rings, IPC Headers, and Concurrency Protocols.
"""

from __future__ import annotations

import unittest
from tools.bus.ring import (
    ChannelId,
    IpcControlHeaderData,
    IpcSlotHeaderData,
    LatestSpscRingSim,
    SharedMemoryRingInspector,
    ShmHealthFlags,
    SHM_RING_MAGIC,
    SHM_VERSION_MAJOR,
    SHM_VERSION_MINOR,
)
from tools.bus.schema_bindings import (
    ActuationCommandData,
    ButtonAction,
    ButtonTransitionData,
    CorrelationHeader,
    FrameDescriptorData,
    FramePixelFormat,
    MouseButton,
    serialize_actuation_command,
    serialize_frame_descriptor,
    validate_file_identifier,
)


class TestBusRing(unittest.TestCase):
    """Test suite for LatestSpscRing semantics, IPC headers, and schema traits."""

    # =========================================================================
    # SECTION 1: LatestSpscRingSim Tests
    # =========================================================================

    def test_ring_power_of_two_enforcement(self) -> None:
        """Verify ring capacity must be power of two >= 2."""
        valid_caps = [2, 4, 8, 16, 32, 64, 128, 1024]
        for cap in valid_caps:
            ring = LatestSpscRingSim[int](capacity=cap)
            self.assertEqual(ring.capacity, cap)

        invalid_caps = [-4, 0, 1, 3, 5, 6, 7, 9, 15, 100]
        for cap in invalid_caps:
            with self.assertRaises(ValueError):
                LatestSpscRingSim[int](capacity=cap)

    def test_basic_push_and_read_latest(self) -> None:
        """Verify basic push and single latest read."""
        ring = LatestSpscRingSim[str](capacity=4)
        self.assertTrue(ring.empty())
        self.assertEqual(ring.latest_sequence, 0)

        seq1 = ring.push("msg1")
        self.assertEqual(seq1, 1)
        self.assertFalse(ring.empty())
        self.assertEqual(ring.latest_sequence, 1)

        ok, data, seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertEqual(data, "msg1")
        self.assertEqual(seq, 1)
        self.assertEqual(drops, 0)
        self.assertEqual(ring.total_reads, 1)

        # Subsequent read should return False (no new message)
        ok2, data2, seq2, drops2 = ring.try_read_latest()
        self.assertFalse(ok2)
        self.assertIsNone(data2)
        self.assertEqual(seq2, 1)
        self.assertEqual(drops2, 0)

    def test_latest_wins_overwrite_and_drop_tracking(self) -> None:
        """Verify producer overwriting slots and exact dropped count accounting."""
        ring = LatestSpscRingSim[int](capacity=4)

        # Producer writes 10 items without consumer reading
        for i in range(1, 11):
            ring.push(i * 10)

        self.assertEqual(ring.latest_sequence, 10)
        self.assertEqual(ring.total_produced, 10)

        # Consumer reads: should jump directly to sequence 10
        ok, data, seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertEqual(data, 100)
        self.assertEqual(seq, 10)
        self.assertEqual(drops, 9)  # Dropped seq 1..9 before initial read
        self.assertEqual(ring.dropped_count, 9)

        # Producer writes 5 more items (seq 11..15)
        for i in range(11, 16):
            ring.push(i * 10)

        ok2, data2, seq2, drops2 = ring.try_read_latest()
        self.assertTrue(ok2)
        self.assertEqual(data2, 150)
        self.assertEqual(seq2, 15)
        self.assertEqual(drops2, 4)  # Dropped seq 11, 12, 13, 14
        self.assertEqual(ring.dropped_count, 13)

    def test_historical_sequence_query(self) -> None:
        """Verify historical sequence queries within and outside capacity window."""
        ring = LatestSpscRingSim[str](capacity=4)

        for i in range(1, 7):
            ring.push(f"val_{i}")

        # Current head is 6, capacity is 4 -> window is [3, 4, 5, 6]
        ok, val = ring.try_read_sequence(6)
        self.assertTrue(ok)
        self.assertEqual(val, "val_6")

        ok, val = ring.try_read_sequence(5)
        self.assertTrue(ok)
        self.assertEqual(val, "val_5")

        ok, val = ring.try_read_sequence(4)
        self.assertTrue(ok)
        self.assertEqual(val, "val_4")

        ok, val = ring.try_read_sequence(3)
        self.assertTrue(ok)
        self.assertEqual(val, "val_3")

        # Overwritten sequences
        self.assertFalse(ring.try_read_sequence(2)[0])
        self.assertFalse(ring.try_read_sequence(1)[0])

        # Invalid/future sequences
        self.assertFalse(ring.try_read_sequence(7)[0])
        self.assertFalse(ring.try_read_sequence(0)[0])
        self.assertFalse(ring.try_read_sequence(-1)[0])

    def test_ring_reset(self) -> None:
        """Verify reset clears all state cleanly."""
        ring = LatestSpscRingSim[int](capacity=8)
        for i in range(20):
            ring.push(i)
        ring.try_read_latest()

        ring.reset()
        self.assertTrue(ring.empty())
        self.assertEqual(ring.latest_sequence, 0)
        self.assertEqual(ring.last_consumed_sequence, 0)
        self.assertEqual(ring.dropped_count, 0)
        self.assertEqual(ring.total_reads, 0)

    # =========================================================================
    # SECTION 2: Shared Memory IPC Header Binary Serialization Tests
    # =========================================================================

    def test_control_header_binary_roundtrip(self) -> None:
        """Verify 128-byte IpcControlHeader binary pack/unpack integrity."""
        hdr = IpcControlHeaderData(
            magic=SHM_RING_MAGIC,
            version_major=SHM_VERSION_MAJOR,
            version_minor=SHM_VERSION_MINOR,
            header_size_bytes=128,
            channel_id=int(ChannelId.TARGET_OBSERVATION),
            pipeline_run_id=42,
            writer_pid=12345,
            ring_capacity=16,
            slot_stride_bytes=4096,
            max_payload_bytes=4032,
            writer_heartbeat_ns=5000000000,
            producer_head_seq=100,
            consumer_tail_seq=95,
            health_flags=int(ShmHealthFlags.WRITER_ACTIVE),
        )

        binary = SharedMemoryRingInspector.serialize_control_header(hdr)
        self.assertEqual(len(binary), 128)

        parsed = SharedMemoryRingInspector.parse_control_header(binary)
        self.assertEqual(parsed.magic, SHM_RING_MAGIC)
        self.assertEqual(parsed.version_major, 1)
        self.assertEqual(parsed.version_minor, 0)
        self.assertEqual(parsed.header_size_bytes, 128)
        self.assertEqual(parsed.channel_id, int(ChannelId.TARGET_OBSERVATION))
        self.assertEqual(parsed.pipeline_run_id, 42)
        self.assertEqual(parsed.writer_pid, 12345)
        self.assertEqual(parsed.ring_capacity, 16)
        self.assertEqual(parsed.slot_stride_bytes, 4096)
        self.assertEqual(parsed.max_payload_bytes, 4032)
        self.assertEqual(parsed.writer_heartbeat_ns, 5000000000)
        self.assertEqual(parsed.producer_head_seq, 100)
        self.assertEqual(parsed.consumer_tail_seq, 95)
        self.assertEqual(parsed.health_flags, int(ShmHealthFlags.WRITER_ACTIVE))

    def test_slot_header_binary_roundtrip(self) -> None:
        """Verify 64-byte IpcSlotHeader binary pack/unpack integrity."""
        slot = IpcSlotHeaderData(
            seq_before=200,
            payload_size=1024,
            magic_identifier=0x31424F41,  # 'AOB1'
            published_at_ns=1234567890,
            seq_after=200,
        )

        binary = SharedMemoryRingInspector.serialize_slot_header(slot)
        self.assertEqual(len(binary), 64)

        parsed = SharedMemoryRingInspector.parse_slot_header(binary)
        self.assertEqual(parsed.seq_before, 200)
        self.assertEqual(parsed.payload_size, 1024)
        self.assertEqual(parsed.magic_identifier, 0x31424F41)
        self.assertEqual(parsed.published_at_ns, 1234567890)
        self.assertEqual(parsed.seq_after, 200)

    def test_control_header_short_buffer_rejection(self) -> None:
        """Verify short binary buffers raise ValueError."""
        with self.assertRaises(ValueError):
            SharedMemoryRingInspector.parse_control_header(b"\x00" * 127)

        with self.assertRaises(ValueError):
            SharedMemoryRingInspector.parse_slot_header(b"\x00" * 63)

    # =========================================================================
    # SECTION 3: Multi-Rate Consumer Simulation
    # =========================================================================

    def test_dual_consumer_different_sampling_rates(self) -> None:
        """Simulate fast producer (1000 Hz), medium consumer (144 Hz), slow consumer (30 Hz)."""
        ring = LatestSpscRingSim[int](capacity=16)

        consumer_144_last_seq = 0
        consumer_30_last_seq = 0
        c144_reads: list[int] = []
        c30_reads: list[int] = []

        # Simulate 1000 ticks
        for tick in range(1, 1001):
            ring.push(tick)

            # 144 Hz consumer samples every ~7 ticks
            if tick % 7 == 0:
                head = ring.latest_sequence
                if head > consumer_144_last_seq:
                    ok, val, seq, drops = ring.try_read_latest()
                    if ok and val is not None:
                        c144_reads.append(val)
                        consumer_144_last_seq = seq

            # 30 Hz consumer samples every ~33 ticks
            if tick % 33 == 0:
                head = ring.latest_sequence
                if head > consumer_30_last_seq:
                    ok, val, seq, drops = ring.try_read_latest()
                    if ok and val is not None:
                        c30_reads.append(val)
                        consumer_30_last_seq = seq

        self.assertGreater(len(c144_reads), 100)
        self.assertGreater(len(c30_reads), 20)

        # Monotonicity check for both consumers
        for i in range(1, len(c144_reads)):
            self.assertGreater(c144_reads[i], c144_reads[i - 1])

        for i in range(1, len(c30_reads)):
            self.assertGreater(c30_reads[i], c30_reads[i - 1])

    # =========================================================================
    # SECTION 4: Canonical Schema Magic Identifier Validation
    # =========================================================================

    def test_schema_magic_identifiers_and_serializers(self) -> None:
        """Verify FrameDescriptor and ActuationCommand serializations."""
        hdr = CorrelationHeader(sequence_id=1, source_timestamp_ns=1000000, pipeline_run_id=1)
        frame_data = FrameDescriptorData(
            header=hdr,
            frame_id=100,
            captured_at_ns=1000000,
            width=1920,
            height=1080,
            format=FramePixelFormat.B8G8R8A8_UNORM,
        )
        frame_buf = serialize_frame_descriptor(frame_data)
        self.assertTrue(validate_file_identifier(frame_buf, b"AFR1"))

        act_data = ActuationCommandData(
            header=hdr,
            generated_at_ns=1000500,
            desired_apply_time_ns=1001000,
            delta_x_counts=15,
            delta_y_counts=-8,
            button_transition=ButtonTransitionData(button=MouseButton.LEFT, action=ButtonAction.CLICK),
        )
        act_buf = serialize_actuation_command(act_data)
        self.assertTrue(validate_file_identifier(act_buf, b"AAC1"))


if __name__ == "__main__":
    unittest.main()
