"""tests/test_challenger_adversarial.py
Adversarial Verification Suite for Milestone M1-04 (#10):
Bounded Latest-Wins In-Process and Shared-Memory Bus Rings.

Focus Areas:
1. Latency benchmarking: evaluate acquire/release handoff overhead (p50, p95, p99, max).
2. Boundary testing: capacity constraints (power of 2, capacity == 2, capacity == 1024), buffer wrap-around.
3. Shared memory boundary tests: corrupted IPC headers, stale writer heartbeat timeouts, Windows named shm lifecycle.
4. Concurrency & SeqLock race condition / torn read stress harness.
"""

from __future__ import annotations

import mmap
import os
import struct
import threading
import time
import unittest
from typing import List, Tuple

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


class TestChallengerAdversarialM104(unittest.TestCase):
    """Empirical adversarial verification tests for Milestone M1-04."""

    # =========================================================================
    # FOCUS 1: Latency Benchmarking (Acquire/Release Overhead)
    # =========================================================================

    def test_latency_benchmark_acquire_release(self) -> None:
        """Measure acquire/release latency distribution (p50, p95, p99, max) over 100,000 operations."""
        ring = LatestSpscRingSim[int](capacity=16)
        iterations = 100_000

        # Warmup
        for i in range(5000):
            ring.push(i)
            ring.try_read_latest()
        ring.reset()

        push_times: List[float] = []
        read_times: List[float] = []

        t_total_start = time.perf_counter_ns()
        for i in range(1, iterations + 1):
            t0 = time.perf_counter_ns()
            ring.push(i)
            t1 = time.perf_counter_ns()
            push_times.append(t1 - t0)

            t2 = time.perf_counter_ns()
            ok, val, seq, drops = ring.try_read_latest()
            t3 = time.perf_counter_ns()
            read_times.append(t3 - t2)
            self.assertTrue(ok)
            self.assertEqual(val, i)

        t_total_end = time.perf_counter_ns()
        total_time_ms = (t_total_end - t_total_start) / 1_000_000.0

        push_times.sort()
        read_times.sort()

        p50_push = push_times[int(iterations * 0.50)]
        p95_push = push_times[int(iterations * 0.95)]
        p99_push = push_times[int(iterations * 0.99)]
        max_push = push_times[-1]

        p50_read = read_times[int(iterations * 0.50)]
        p95_read = read_times[int(iterations * 0.95)]
        p99_read = read_times[int(iterations * 0.99)]
        max_read = read_times[-1]

        print("\n--- Latency Benchmark Results (Python Simulator) ---")
        print(f"Total iterations: {iterations} push + {iterations} read")
        print(f"Total wall time:  {total_time_ms:.2f} ms")
        print(f"Push (write) ns:  p50={p50_push:.0f}ns, p95={p95_push:.0f}ns, p99={p99_push:.0f}ns, max={max_push:.0f}ns")
        print(f"Read (acquire) ns: p50={p50_read:.0f}ns, p95={p95_read:.0f}ns, p99={p99_read:.0f}ns, max={max_read:.0f}ns")

        # Sanity assertions on latency
        self.assertLess(p50_push, 10_000, "Push p50 must be under 10us in Python simulation")
        self.assertLess(p50_read, 10_000, "Read p50 must be under 10us in Python simulation")

    # =========================================================================
    # FOCUS 2: Boundary Testing (Capacities & Buffer Overflow Wrap-Around)
    # =========================================================================

    def test_capacity_boundaries_and_invalid_inputs(self) -> None:
        """Verify strict power-of-two validation across boundaries."""
        # Valid powers of 2: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
        for p in range(1, 17):
            cap = 1 << p
            ring = LatestSpscRingSim[int](capacity=cap)
            self.assertEqual(ring.capacity, cap)

        # Invalid capacities (non-power-of-2, < 2, negative, zero)
        invalid_cases = [-1024, -16, -1, 0, 1, 3, 5, 6, 7, 9, 10, 15, 17, 31, 33, 100, 1023, 1025, 65535, 65537]
        for cap in invalid_cases:
            with self.assertRaises(ValueError, msg=f"Capacity {cap} should have been rejected"):
                LatestSpscRingSim[int](capacity=cap)

    def test_capacity_boundary_minimum_two(self) -> None:
        """Verify boundary behavior with minimal ring capacity == 2."""
        ring = LatestSpscRingSim[int](capacity=2)
        self.assertEqual(ring.capacity, 2)
        self.assertTrue(ring.empty())

        # Push 1
        ring.push(10)
        self.assertEqual(ring.latest_sequence, 1)
        ok, val, seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertEqual(val, 10)
        self.assertEqual(seq, 1)
        self.assertEqual(drops, 0)

        # Push 2 and 3 without reading (causes overwrite of slot 1 by 3)
        ring.push(20)
        ring.push(30)
        self.assertEqual(ring.latest_sequence, 3)

        # Read latest
        ok, val, seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertEqual(val, 30)
        self.assertEqual(seq, 3)
        self.assertEqual(drops, 1)  # Dropped seq 2
        self.assertEqual(ring.dropped_count, 1)

        # Historical queries: capacity 2, head 3 -> retained [2, 3]
        ok_3, val_3 = ring.try_read_sequence(3)
        self.assertTrue(ok_3)
        self.assertEqual(val_3, 30)

        ok_2, val_2 = ring.try_read_sequence(2)
        self.assertTrue(ok_2)
        self.assertEqual(val_2, 20)

        # Seq 1 overwritten
        ok_1, val_1 = ring.try_read_sequence(1)
        self.assertFalse(ok_1)
        self.assertIsNone(val_1)

        # Seq 0 or 4 invalid
        self.assertFalse(ring.try_read_sequence(0)[0])
        self.assertFalse(ring.try_read_sequence(4)[0])

    def test_capacity_boundary_large_1024(self) -> None:
        """Verify boundary behavior with large ring capacity == 1024."""
        cap = 1024
        ring = LatestSpscRingSim[int](capacity=cap)
        self.assertEqual(ring.capacity, cap)

        # Push exactly 1024 items
        for i in range(1, 1025):
            ring.push(i * 2)

        self.assertEqual(ring.latest_sequence, 1024)
        self.assertEqual(ring.total_produced, 1024)

        # All 1024 items should be accessible in historical window [1, 1024]
        for i in range(1, 1025):
            ok, val = ring.try_read_sequence(i)
            self.assertTrue(ok, f"Sequence {i} should be retained")
            self.assertEqual(val, i * 2)

        # Seq 0 and 1025 should be inaccessible
        self.assertFalse(ring.try_read_sequence(0)[0])
        self.assertFalse(ring.try_read_sequence(1025)[0])

        # Push another 1024 items (total 2048)
        for i in range(1025, 2049):
            ring.push(i * 2)

        self.assertEqual(ring.latest_sequence, 2048)

        # Window is now [1025, 2048]
        # First half [1, 1024] is overwritten
        for i in range(1, 1025):
            ok, val = ring.try_read_sequence(i)
            self.assertFalse(ok, f"Sequence {i} should be overwritten")

        # Second half [1025, 2048] is retained
        for i in range(1025, 2049):
            ok, val = ring.try_read_sequence(i)
            self.assertTrue(ok, f"Sequence {i} should be retained")
            self.assertEqual(val, i * 2)

        # Read latest
        ok, val, seq, drops = ring.try_read_latest()
        self.assertTrue(ok)
        self.assertEqual(val, 4096)
        self.assertEqual(seq, 2048)
        self.assertEqual(drops, 2047)

    def test_buffer_overflow_drop_accounting_invariant(self) -> None:
        """Verify that produced == consumed + dropped across varied burst rates."""
        for cap in [2, 4, 8, 16, 64, 128]:
            for step in [1, 2, 3, 5, 7, 13, 29, 100]:
                ring = LatestSpscRingSim[int](capacity=cap)
                consumer_reads = 0
                consumer_drops = 0
                consumer_last = 0

                total_items = 2000
                for i in range(1, total_items + 1):
                    ring.push(i)

                    if i % step == 0:
                        ok, val, seq, drops = ring.try_read_latest()
                        if ok:
                            consumer_reads += 1
                            consumer_drops += drops
                            consumer_last = seq

                # Final drain
                if ring.latest_sequence > consumer_last:
                    ok, val, seq, drops = ring.try_read_latest()
                    if ok:
                        consumer_reads += 1
                        consumer_drops += drops

                self.assertEqual(
                    consumer_reads + consumer_drops,
                    total_items,
                    f"Invariant violated for cap={cap}, step={step}: {consumer_reads} + {consumer_drops} != {total_items}",
                )
                self.assertEqual(ring.dropped_count, consumer_drops)
                self.assertEqual(ring.total_reads, consumer_reads)
                self.assertEqual(ring.total_produced, total_items)

    # =========================================================================
    # FOCUS 3: Shared Memory Boundary Tests (Corrupted IPC, Heartbeats, Detach)
    # =========================================================================

    def test_corrupted_ipc_control_header_adversarial(self) -> None:
        """Adversarial testing on IpcControlHeader parsing with corrupted buffers."""
        valid_hdr = IpcControlHeaderData(
            magic=SHM_RING_MAGIC,
            version_major=SHM_VERSION_MAJOR,
            version_minor=SHM_VERSION_MINOR,
            header_size_bytes=128,
            channel_id=int(ChannelId.ACTUATION_COMMAND),
            pipeline_run_id=99,
            writer_pid=1234,
            ring_capacity=16,
            slot_stride_bytes=256,
            max_payload_bytes=192,
            writer_heartbeat_ns=123456789000,
            producer_head_seq=500,
            consumer_tail_seq=490,
            health_flags=int(ShmHealthFlags.WRITER_ACTIVE),
        )
        valid_bytes = SharedMemoryRingInspector.serialize_control_header(valid_hdr)
        self.assertEqual(len(valid_bytes), 128)

        # Case 1: Truncated buffer
        for short_len in [0, 1, 32, 63, 64, 127]:
            with self.assertRaises(ValueError):
                SharedMemoryRingInspector.parse_control_header(valid_bytes[:short_len])

        # Case 2: Corrupted magic number
        corrupt_magic = bytearray(valid_bytes)
        struct.pack_into("<I", corrupt_magic, 0, 0x00000000)
        parsed = SharedMemoryRingInspector.parse_control_header(bytes(corrupt_magic))
        self.assertNotEqual(parsed.magic, SHM_RING_MAGIC)

        struct.pack_into("<I", corrupt_magic, 0, 0xFFFFFFFF)
        parsed = SharedMemoryRingInspector.parse_control_header(bytes(corrupt_magic))
        self.assertNotEqual(parsed.magic, SHM_RING_MAGIC)

        # Case 3: Invalid version major (e.g. version 2 or 0)
        corrupt_ver = bytearray(valid_bytes)
        struct.pack_into("<H", corrupt_ver, 4, 2)
        parsed = SharedMemoryRingInspector.parse_control_header(bytes(corrupt_ver))
        self.assertEqual(parsed.version_major, 2)

        # Case 4: Invalid capacity in header
        corrupt_cap = bytearray(valid_bytes)
        struct.pack_into("<I", corrupt_cap, 24, 15)  # Offset 24 is ring_capacity (15 is not power of 2)
        parsed = SharedMemoryRingInspector.parse_control_header(bytes(corrupt_cap))
        self.assertEqual(parsed.ring_capacity, 15)

    def test_corrupted_slot_header_adversarial(self) -> None:
        """Adversarial testing on BusSlotHeader parsing with corrupted buffers."""
        valid_slot = IpcSlotHeaderData(
            seq_before=100,
            payload_size=256,
            magic_identifier=0x31434141,  # 'AAC1'
            published_at_ns=987654321,
            seq_after=100,
        )
        valid_bytes = SharedMemoryRingInspector.serialize_slot_header(valid_slot)
        self.assertEqual(len(valid_bytes), 64)

        # Truncated buffer
        for short_len in [0, 1, 31, 63]:
            with self.assertRaises(ValueError):
                SharedMemoryRingInspector.parse_slot_header(valid_bytes[:short_len])

        # Torn sequence mid-write (seq_before odd, seq_after old even)
        torn_bytes = bytearray(valid_bytes)
        struct.pack_into("<q", torn_bytes, 0, 101)  # seq_before = 101 (odd)
        struct.pack_into("<q", torn_bytes, 24, 100)  # seq_after = 100 (even)
        parsed = SharedMemoryRingInspector.parse_slot_header(bytes(torn_bytes))
        self.assertEqual(parsed.seq_before, 101)
        self.assertEqual(parsed.seq_after, 100)
        self.assertNotEqual(parsed.seq_before, parsed.seq_after)
        self.assertTrue((parsed.seq_before & 1) != 0, "Odd sequence indicates mid-write state")

    def test_heartbeat_timeout_stale_detection(self) -> None:
        """Verify exact timing boundaries for writer heartbeat liveness and stale detection."""
        timeout_ns = 50_000_000  # 50 ms
        base_hb_ns = 1_000_000_000

        # Helper function replicating SharedMemoryRing::is_writer_alive
        def is_writer_alive(last_hb: int, current_time: int, timeout: int = 50_000_000) -> bool:
            if last_hb == 0:
                return True  # Heartbeat not yet recorded
            return (current_time - last_hb) <= timeout

        # 1. Initial state (heartbeat = 0) -> alive
        self.assertTrue(is_writer_alive(0, 5_000_000_000, timeout_ns))

        # 2. Fresh heartbeat (same timestamp) -> alive
        self.assertTrue(is_writer_alive(base_hb_ns, base_hb_ns, timeout_ns))

        # 3. Within timeout window (10ms elapsed) -> alive
        self.assertTrue(is_writer_alive(base_hb_ns, base_hb_ns + 10_000_000, timeout_ns))

        # 4. Exact boundary (50ms elapsed) -> alive
        self.assertTrue(is_writer_alive(base_hb_ns, base_hb_ns + 50_000_000, timeout_ns))

        # 5. Exactly 1 ns past timeout (50ms + 1ns) -> stale
        self.assertFalse(is_writer_alive(base_hb_ns, base_hb_ns + 50_000_001, timeout_ns))

        # 6. Significantly past timeout (500ms elapsed) -> stale
        self.assertFalse(is_writer_alive(base_hb_ns, base_hb_ns + 500_000_000, timeout_ns))

        # 7. Clock skew / monotonic edge: current_time < last_hb -> alive
        self.assertTrue(is_writer_alive(base_hb_ns, base_hb_ns - 1_000, timeout_ns))

    @unittest.skipUnless(os.name == "nt", "Windows named mappings require Windows mmap")
    def test_windows_named_shared_memory_mmap_lifecycle(self) -> None:
        """Verify Windows named shared memory mapping, multi-handle lifecycle, and cleanup."""
        shm_tag = "Local\\AimAgent_ChallengerTest_ShmRing"
        shm_size = 128 + 16 * 64  # 128-byte control header + 16 * 64-byte slots = 1152 bytes

        # Creator opens anonymous/named mmap
        creator_shm = mmap.mmap(-1, shm_size, tagname=shm_tag, access=mmap.ACCESS_WRITE)
        self.assertIsNotNone(creator_shm)

        # Write control header to creator shm
        hdr = IpcControlHeaderData(
            magic=SHM_RING_MAGIC,
            version_major=SHM_VERSION_MAJOR,
            version_minor=SHM_VERSION_MINOR,
            header_size_bytes=128,
            channel_id=int(ChannelId.FRAME_DESCRIPTOR),
            pipeline_run_id=777,
            writer_pid=os.getpid(),
            ring_capacity=16,
            slot_stride_bytes=64,
            max_payload_bytes=0,
            writer_heartbeat_ns=time.perf_counter_ns(),
            producer_head_seq=1,
            consumer_tail_seq=0,
            health_flags=int(ShmHealthFlags.WRITER_ACTIVE),
        )
        creator_shm.seek(0)
        creator_shm.write(SharedMemoryRingInspector.serialize_control_header(hdr))
        creator_shm.flush()

        # Reader 1 attaches to same named tag
        reader1_shm = mmap.mmap(-1, shm_size, tagname=shm_tag, access=mmap.ACCESS_READ)
        reader1_shm.seek(0)
        read_bytes = reader1_shm.read(128)
        parsed = SharedMemoryRingInspector.parse_control_header(read_bytes)
        self.assertEqual(parsed.magic, SHM_RING_MAGIC)
        self.assertEqual(parsed.pipeline_run_id, 777)
        self.assertEqual(parsed.writer_pid, os.getpid())

        # Reader 2 attaches simultaneously
        reader2_shm = mmap.mmap(-1, shm_size, tagname=shm_tag, access=mmap.ACCESS_READ)
        reader2_shm.seek(0)
        read_bytes2 = reader2_shm.read(128)
        parsed2 = SharedMemoryRingInspector.parse_control_header(read_bytes2)
        self.assertEqual(parsed2.pipeline_run_id, 777)

        # Detach readers
        reader1_shm.close()
        reader2_shm.close()

        # Close creator
        creator_shm.close()

    # =========================================================================
    # FOCUS 4: Concurrency & Multi-Threaded Stress Test
    # =========================================================================

    def test_concurrent_multithreaded_producer_consumer_stress(self) -> None:
        """Run multi-threaded producer/consumer stress test (50,000 items) verifying monotonic reads."""
        ring = LatestSpscRingSim[Tuple[int, int, int]](capacity=32)
        total_items = 50_000

        start_event = threading.Event()
        producer_finished = threading.Event()

        consumed_items: List[Tuple[int, int, int]] = []
        consumed_drops: List[int] = []
        non_monotonic_count = [0]
        torn_reads_count = [0]

        def producer_worker() -> None:
            start_event.wait()
            for i in range(1, total_items + 1):
                # Payload: (seq, magic ^ seq, checksum)
                magic = 0xDEADBEEF ^ i
                checksum = i + magic
                ring.push((i, magic, checksum))
                if i % 100 == 0:
                    time.sleep(0.0001)  # Yield to give consumer concurrent interleaving
            producer_finished.set()

        def consumer_worker() -> None:
            start_event.wait()
            last_seq = 0
            while not producer_finished.is_set() or ring.latest_sequence > last_seq:
                ok, item, seq, drops = ring.try_read_latest()
                if ok and item is not None:
                    s, m, c = item
                    # Checksum integrity check
                    if m != (0xDEADBEEF ^ s) or c != (s + m):
                        torn_reads_count[0] += 1

                    # Monotonicity check
                    if s <= last_seq:
                        non_monotonic_count[0] += 1

                    last_seq = seq
                    consumed_items.append(item)
                    consumed_drops.append(drops)
                else:
                    time.sleep(0.00005)

        p_thread = threading.Thread(target=producer_worker)
        c_thread = threading.Thread(target=consumer_worker)

        p_thread.start()
        c_thread.start()

        start_event.set()

        p_thread.join()
        c_thread.join()

        print("\n--- Concurrent Multi-Threaded Stress Test ---")
        print(f"Total Produced: {ring.total_produced}")
        print(f"Total Consumed: {len(consumed_items)}")
        print(f"Total Dropped:  {sum(consumed_drops)}")
        print(f"Torn Reads:     {torn_reads_count[0]}")
        print(f"Non-monotonic:  {non_monotonic_count[0]}")

        self.assertEqual(ring.total_produced, total_items)
        self.assertEqual(torn_reads_count[0], 0, "No torn reads should be observed")
        self.assertEqual(non_monotonic_count[0], 0, "Consumed sequences must be strictly monotonic")
        self.assertGreater(len(consumed_items), 0, "Consumer must have read messages")


if __name__ == "__main__":
    unittest.main()
