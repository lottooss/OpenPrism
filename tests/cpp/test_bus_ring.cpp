// tests/cpp/test_bus_ring.cpp
// Comprehensive test suite and latency benchmarks for OpenPrism Bus Rings (Milestone M1-04)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "aim/bus/bus.hpp"
#include "aim/core/clock.hpp"

namespace allocation_probe {

thread_local bool enabled = false;
thread_local std::size_t calls = 0;
thread_local std::size_t bytes = 0;

void record(std::size_t size) noexcept {
    if (enabled) {
        ++calls;
        bytes += size;
    }
}

struct Report final {
    std::size_t calls;
    std::size_t bytes;
};

void begin() noexcept {
    calls = 0;
    bytes = 0;
    enabled = true;
}

[[nodiscard]] Report end() noexcept {
    enabled = false;
    return {.calls = calls, .bytes = bytes};
}

}  // namespace allocation_probe

// GCC cannot see that these replacement operators are a matched pair: it classifies the
// replaced `operator new` as new-like and `std::free` as malloc-like, then reports
// -Wmismatched-new-delete once the vector teardown below gets inlined into the benchmark.
// Every allocation here is malloc/aligned_alloc and every deallocation is the matching
// free, so the pairing is correct and the diagnostic is a false positive.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(std::size_t size) {
    const std::size_t requested = size == 0 ? 1 : size;
    if (void* pointer = std::malloc(requested)) {
        allocation_probe::record(requested);
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    ::operator delete(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    ::operator delete[](pointer);
}

#if defined(__cpp_aligned_new)
void* operator new(std::size_t size, std::align_val_t alignment) {
    const std::size_t requested = size == 0 ? 1 : size;
    const std::size_t alignment_bytes = static_cast<std::size_t>(alignment);
#if defined(_MSC_VER)
    void* pointer = _aligned_malloc(requested, alignment_bytes);
#else
    const std::size_t padded = ((requested + alignment_bytes - 1) / alignment_bytes) * alignment_bytes;
    void* pointer = std::aligned_alloc(alignment_bytes, padded);
#endif
    if (pointer == nullptr) {
        throw std::bad_alloc();
    }
    allocation_probe::record(requested);
    return pointer;
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
#if defined(_MSC_VER)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete[](pointer, alignment);
}
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace aim;
using namespace aim::bus;

// Custom payload for stress testing torn-read detection
struct alignas(64) TestStressPayload {
    std::uint64_t sequence{0};
    std::uint64_t magic_check{0};
    MonotonicNs timestamp_ns{0};
    std::uint64_t checksum{0};
    std::uint8_t payload[432]{0};

    static TestStressPayload create(std::uint64_t seq, MonotonicNs ts) noexcept {
        TestStressPayload p;
        p.sequence = seq;
        p.magic_check = 0xDEADBEEFCAFEBABEULL ^ seq;
        p.timestamp_ns = ts;
        p.checksum = compute_checksum(seq, p.magic_check, ts);
        std::memset(p.payload, static_cast<int>(seq & 0xFF), sizeof(p.payload));
        return p;
    }

    [[nodiscard]] bool validate() const noexcept {
        if (magic_check != (0xDEADBEEFCAFEBABEULL ^ sequence)) return false;
        if (checksum != compute_checksum(sequence, magic_check, timestamp_ns)) return false;
        const std::uint8_t expected_byte = static_cast<std::uint8_t>(sequence & 0xFF);
        for (std::size_t i = 0; i < sizeof(payload); ++i) {
            if (payload[i] != expected_byte) return false;
        }
        return true;
    }

private:
    static constexpr std::uint64_t compute_checksum(std::uint64_t s, std::uint64_t m, MonotonicNs t) noexcept {
        return s * 6364136223846793005ULL + m * 1442695040888963407ULL + static_cast<std::uint64_t>(t);
    }
};

static_assert(sizeof(TestStressPayload) == 512, "TestStressPayload must be 512 bytes");
static_assert(alignof(TestStressPayload) == 64, "TestStressPayload must be 64-byte aligned");

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] ASSERTION FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return false; \
        } \
    } while (false)

// =============================================================================
// Test Cases
// =============================================================================

// TC-BUS-01: Static Alignment & Layout Guarantees
bool test_alignments_and_layouts() {
    std::cout << "[RUN] TC-BUS-01: Verifying cacheline alignments and struct layouts...\n";

    static_assert(sizeof(BusSlotHeader) == 64, "BusSlotHeader must be 64 bytes");
    static_assert(alignof(BusSlotHeader) == 64, "BusSlotHeader must be 64-byte aligned");

    static_assert(sizeof(IpcControlHeader) == 128, "IpcControlHeader must be 128 bytes");
    static_assert(alignof(IpcControlHeader) == 64, "IpcControlHeader must be 64-byte aligned");

    static_assert(alignof(LatestSpscRing<ActuationCommand, 16>::Slot) == 64, "Ring Slot must be 64-byte aligned");
    static_assert(alignof(LatestSpscRing<ActuationCommand, 16>::ProducerState) == 64, "ProducerState must be 64-byte aligned");
    static_assert(alignof(LatestSpscRing<ActuationCommand, 16>::ConsumerState) == 64, "ConsumerState must be 64-byte aligned");

    static_assert(sizeof(LatestSpscRing<ActuationCommand, 16>::ProducerState) == 64, "ProducerState must be 64 bytes");
    static_assert(sizeof(LatestSpscRing<ActuationCommand, 16>::ConsumerState) == 64, "ConsumerState must be 64 bytes");

    std::cout << "  [+] Static alignment assertions verified successfully.\n";
    return true;
}

// TC-BUS-02: Basic Single-Threaded Push & Pop Latest
bool test_basic_push_pop_latest() {
    std::cout << "[RUN] TC-BUS-02: Testing basic push and try_read_latest semantics...\n";

    LatestSpscRing<ActuationCommand, 4> ring;
    TEST_ASSERT(ring.empty(), "Ring should be initially empty");
    TEST_ASSERT(ring.capacity() == 4, "Capacity should equal 4");
    TEST_ASSERT(ring.latest_sequence() == 0, "Latest sequence should be 0");

    ActuationCommand cmd1;
    cmd1.generated_at_ns = 1000000;
    cmd1.delta_x_counts = 10;
    cmd1.delta_y_counts = -5;

    const std::uint64_t seq1 = ring.push(cmd1);
    TEST_ASSERT(seq1 == 1, "First push should return sequence 1");
    TEST_ASSERT(!ring.empty(), "Ring should not be empty after push");
    TEST_ASSERT(ring.latest_sequence() == 1, "Latest sequence should be 1");

    ActuationCommand out_cmd{};
    std::uint64_t last_seq = 0;
    std::uint64_t drops = 0;

    bool ok = ring.try_read_latest(out_cmd, last_seq, &drops);
    TEST_ASSERT(ok, "try_read_latest should succeed");
    TEST_ASSERT(last_seq == 1, "Consumed sequence should be 1");
    TEST_ASSERT(drops == 0, "Drop count should be 0");
    TEST_ASSERT(out_cmd.generated_at_ns == 1000000, "Payload generated_at_ns mismatch");
    TEST_ASSERT(out_cmd.delta_x_counts == 10, "Payload delta_x mismatch");
    TEST_ASSERT(out_cmd.delta_y_counts == -5, "Payload delta_y mismatch");

    // Second read should return false (no new message)
    ok = ring.try_read_latest(out_cmd, last_seq, &drops);
    TEST_ASSERT(!ok, "Subsequent try_read_latest should return false");
    TEST_ASSERT(last_seq == 1, "Sequence should remain 1");

    std::cout << "  [+] Basic push/read passed.\n";
    return true;
}

// TC-BUS-03: Overwriting & Latest-Wins Semantics with Drop Accounting
bool test_latest_wins_overwrite_and_drop_accounting() {
    std::cout << "[RUN] TC-BUS-03: Testing latest-wins overwrite and drop accounting...\n";

    LatestSpscRing<ActuationCommand, 4> ring;

    // Push 10 messages without consumer reading
    for (std::uint64_t i = 1; i <= 10; ++i) {
        ActuationCommand cmd;
        cmd.generated_at_ns = static_cast<MonotonicNs>(i * 1000);
        cmd.delta_x_counts = static_cast<std::int32_t>(i * 5);
        const std::uint64_t seq = ring.push(cmd);
        TEST_ASSERT(seq == i, "Sequence should increment monotonically");
    }

    TEST_ASSERT(ring.latest_sequence() == 10, "Latest sequence should be 10");

    ActuationCommand out_cmd{};
    std::uint64_t last_seq = 0;
    std::uint64_t drops = 0;

    bool ok = ring.try_read_latest(out_cmd, last_seq, &drops);
    TEST_ASSERT(ok, "try_read_latest should succeed");
    TEST_ASSERT(last_seq == 10, "Consumer should jump straight to sequence 10");
    TEST_ASSERT(drops == 9, "Should report 9 dropped messages on initial jump from 0 to 10");
    TEST_ASSERT(ring.dropped_count() == 9, "Ring total dropped count should record 9 drops");
    TEST_ASSERT(out_cmd.generated_at_ns == 10000, "Should read payload of message 10");
    TEST_ASSERT(out_cmd.delta_x_counts == 50, "Should read delta_x of message 10");

    // Push another 5 messages (11..15)
    for (std::uint64_t i = 11; i <= 15; ++i) {
        ActuationCommand cmd;
        cmd.generated_at_ns = static_cast<MonotonicNs>(i * 1000);
        ring.push(cmd);
    }

    ok = ring.try_read_latest(out_cmd, last_seq, &drops);
    TEST_ASSERT(ok, "Second try_read_latest should succeed");
    TEST_ASSERT(last_seq == 15, "Consumer should jump to sequence 15");
    TEST_ASSERT(drops == 4, "Should report exactly 4 dropped messages between 10 and 15");
    TEST_ASSERT(ring.dropped_count() == 13, "Ring total dropped count should record 13 drops (9 + 4)");

    std::cout << "  [+] Overwrite and drop accounting verified.\n";
    return true;
}

// TC-BUS-04: Historical Sequence Addressing (try_read_sequence)
bool test_historical_sequence_addressing() {
    std::cout << "[RUN] TC-BUS-04: Testing historical sequence addressing...\n";

    LatestSpscRing<ActuationCommand, 4> ring;

    for (std::uint64_t i = 1; i <= 6; ++i) {
        ActuationCommand cmd;
        cmd.generated_at_ns = static_cast<MonotonicNs>(i * 100);
        ring.push(cmd);
    }

    // Ring capacity is 4, head is 6. Retained window: [3, 4, 5, 6].
    ActuationCommand out{};

    // Valid queries within window
    TEST_ASSERT(ring.try_read_sequence(6, out) && out.generated_at_ns == 600, "Sequence 6 should be accessible");
    TEST_ASSERT(ring.try_read_sequence(5, out) && out.generated_at_ns == 500, "Sequence 5 should be accessible");
    TEST_ASSERT(ring.try_read_sequence(4, out) && out.generated_at_ns == 400, "Sequence 4 should be accessible");
    TEST_ASSERT(ring.try_read_sequence(3, out) && out.generated_at_ns == 300, "Sequence 3 should be accessible");

    // Overwritten sequences
    TEST_ASSERT(!ring.try_read_sequence(2, out), "Sequence 2 was overwritten, should fail");
    TEST_ASSERT(!ring.try_read_sequence(1, out), "Sequence 1 was overwritten, should fail");

    // Future / unpublished sequences
    TEST_ASSERT(!ring.try_read_sequence(7, out), "Sequence 7 is unpublished, should fail");
    TEST_ASSERT(!ring.try_read_sequence(0, out), "Sequence 0 is invalid, should fail");

    std::cout << "  [+] Historical sequence addressing verified.\n";
    return true;
}

// TC-BUS-05: Concurrent Multi-Threaded Stress Test with Torn-Read Detection
bool test_concurrent_multithreaded_stress() {
    std::cout << "[RUN] TC-BUS-05: Running concurrent multi-threaded stress test (500,000 messages)...\n";

    constexpr std::size_t kRingCap = 16;
    constexpr std::uint64_t kTotalMessages = 500000;

    LatestSpscRing<TestStressPayload, kRingCap> ring;

    std::atomic<bool> start_flag{false};
    std::atomic<bool> producer_done{false};
    std::atomic<std::uint64_t> torn_reads_found{0};
    std::atomic<std::uint64_t> non_monotonic_sequences{0};
    std::atomic<std::uint64_t> total_consumed{0};

    // Consumer Thread
    std::thread consumer_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        std::uint64_t last_seen_seq = 0;
        TestStressPayload item{};

        while (!producer_done.load(std::memory_order_acquire) || ring.latest_sequence() > last_seen_seq) {
            std::uint64_t drops = 0;
            if (ring.try_read_latest(item, last_seen_seq, &drops)) {
                total_consumed.fetch_add(1, std::memory_order_relaxed);

                // Verify payload integrity (torn-read check)
                if (!item.validate()) {
                    torn_reads_found.fetch_add(1, std::memory_order_relaxed);
                }

                // Verify sequence monotonicity
                if (item.sequence <= last_seen_seq - drops - 1 && last_seen_seq > 0) {
                    non_monotonic_sequences.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    });

    // Producer Thread
    std::thread producer_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        for (std::uint64_t seq = 1; seq <= kTotalMessages; ++seq) {
            TestStressPayload p = TestStressPayload::create(seq, static_cast<MonotonicNs>(seq * 10));
            ring.push(p);
        }

        producer_done.store(true, std::memory_order_release);
    });

    // Launch both threads simultaneously
    start_flag.store(true, std::memory_order_release);

    producer_thread.join();
    consumer_thread.join();

    const std::uint64_t produced = ring.total_produced();
    const std::uint64_t consumed = total_consumed.load();
    const std::uint64_t torn = torn_reads_found.load();
    const std::uint64_t non_mono = non_monotonic_sequences.load();

    std::cout << "  [*] Producer published: " << produced << " messages.\n";
    std::cout << "  [*] Consumer read:      " << consumed << " latest messages.\n";
    std::cout << "  [*] Drops recorded:     " << ring.dropped_count() << " messages.\n";
    std::cout << "  [*] Torn reads detected: " << torn << " (must be 0)\n";
    std::cout << "  [*] Non-monotonic seqs:  " << non_mono << " (must be 0)\n";

    TEST_ASSERT(produced == kTotalMessages, "Produced count mismatch");
    TEST_ASSERT(torn == 0, "Zero torn reads must be observed");
    TEST_ASSERT(non_mono == 0, "All consumed sequences must be strictly monotonic");
    TEST_ASSERT(consumed > 0, "Consumer must have read messages");

    std::cout << "  [+] Multi-threaded stress test passed with zero torn reads.\n";
    return true;
}

// TC-BUS-06: Native latency distribution and warmed allocation benchmark
bool test_latency_benchmark() {
    std::cout << "[RUN] TC-BUS-06: Running native latency/allocation benchmark (1,000,000 iterations)...\n";

    LatestSpscRing<ActuationCommand, 16> ring;
    ActuationCommand cmd{};
    cmd.delta_x_counts = 42;
    cmd.delta_y_counts = -42;

    constexpr std::size_t kIterations = 1000000;
    std::vector<std::int64_t> push_latencies_ns(kIterations);
    std::vector<std::int64_t> read_latencies_ns(kIterations);

    // Warmup 50,000 iterations to prime CPU caches
    for (std::size_t i = 0; i < 50000; ++i) {
        ring.push(cmd);
        ActuationCommand out{};
        std::uint64_t last_seq = 0;
        ring.try_read_latest(out, last_seq);
    }
    ring.reset();

    std::uint64_t last_seq = 0;
    ActuationCommand out_cmd{};

    const auto t_start = std::chrono::steady_clock::now();

    bool all_reads_succeeded = true;
    allocation_probe::begin();
    for (std::size_t i = 0; i < kIterations; ++i) {
        const auto push_start = std::chrono::steady_clock::now();
        ring.push(cmd);
        const auto push_end = std::chrono::steady_clock::now();
        const bool read_succeeded = ring.try_read_latest(out_cmd, last_seq);
        const auto read_end = std::chrono::steady_clock::now();
        all_reads_succeeded = all_reads_succeeded && read_succeeded;
        push_latencies_ns[i] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(push_end - push_start).count();
        read_latencies_ns[i] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(read_end - push_end).count();
    }
    const allocation_probe::Report allocation_report = allocation_probe::end();

    const auto t_end = std::chrono::steady_clock::now();
    const auto total_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    std::sort(push_latencies_ns.begin(), push_latencies_ns.end());
    std::sort(read_latencies_ns.begin(), read_latencies_ns.end());

    const auto percentile = [](const std::vector<std::int64_t>& samples, std::size_t percent) {
        return samples[((samples.size() - 1) * percent) / 100];
    };

    const auto push_p50 = percentile(push_latencies_ns, 50);
    const auto push_p95 = percentile(push_latencies_ns, 95);
    const auto push_p99 = percentile(push_latencies_ns, 99);
    const auto push_max = push_latencies_ns.back();
    const auto read_p50 = percentile(read_latencies_ns, 50);
    const auto read_p95 = percentile(read_latencies_ns, 95);
    const auto read_p99 = percentile(read_latencies_ns, 99);
    const auto read_max = read_latencies_ns.back();

    std::cout << "  [*] Total operations: " << kIterations * 2 << " (1M push + 1M pop)\n";
    std::cout << "  [*] Total time:       " << static_cast<double>(total_duration_ns) / 1000000.0 << " ms\n";
    std::cout << "  [*] Push latency ns:  p50=" << push_p50 << " p95=" << push_p95 << " p99=" << push_p99
              << " max=" << push_max << "\n";
    std::cout << "  [*] Read latency ns:  p50=" << read_p50 << " p95=" << read_p95 << " p99=" << read_p99
              << " max=" << read_max << "\n";
    std::cout << "  [*] Warmed allocations: calls=" << allocation_report.calls
              << " bytes=" << allocation_report.bytes << "\n";

    TEST_ASSERT(all_reads_succeeded, "Every same-thread read must observe the just-published message");
    TEST_ASSERT(allocation_report.calls == 0, "Warmed push/read loop must perform zero C++ heap allocations");
    TEST_ASSERT(allocation_report.bytes == 0, "Warmed push/read loop must allocate zero C++ heap bytes");
    // Per-operation timing above brackets each call with two steady_clock reads, and a clock
    // read costs tens of nanoseconds on its own -- so those percentiles are dominated by the
    // instrument and cannot resolve the sub-10ns handoff target. Time a whole batch with a
    // single clock pair to get the real per-operation cost. The accumulators keep the
    // optimizer from eliding the loop body.
    constexpr std::size_t kAmortizedIterations = 1000000;
    ring.reset();

    std::uint64_t push_sequence_sum = 0;
    const auto push_batch_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kAmortizedIterations; ++i) {
        push_sequence_sum += ring.push(cmd);
    }
    const auto push_batch_end = std::chrono::steady_clock::now();

    std::uint64_t read_seq = 0;
    std::int64_t read_payload_sum = 0;
    std::size_t successful_reads = 0;
    ActuationCommand amortized_out{};
    const auto read_batch_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kAmortizedIterations; ++i) {
        ring.push(cmd);
        if (ring.try_read_latest(amortized_out, read_seq)) {
            read_payload_sum += amortized_out.delta_x_counts;
            ++successful_reads;
        }
    }
    const auto read_batch_end = std::chrono::steady_clock::now();

    const double push_ns_per_op =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            push_batch_end - push_batch_start).count()) / static_cast<double>(kAmortizedIterations);
    const double push_read_ns_per_op =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            read_batch_end - read_batch_start).count()) / static_cast<double>(kAmortizedIterations);

    std::cout << "  [*] Amortized push:   " << push_ns_per_op << " ns/op (single clock pair over "
              << kAmortizedIterations << " ops)" << '\n';
    std::cout << "  [*] Amortized push+read: " << push_read_ns_per_op << " ns/op" << '\n';

    TEST_ASSERT(push_sequence_sum > 0, "Amortized push loop was optimized away");
    TEST_ASSERT(read_payload_sum != 0 || successful_reads == 0, "Amortized read loop was optimized away");
    TEST_ASSERT(successful_reads == kAmortizedIterations, "Every amortized same-thread read must succeed");
    TEST_ASSERT(push_p50 < 10000, "Native push p50 must remain below the 10us sanity ceiling");
    TEST_ASSERT(read_p50 < 10000, "Native read p50 must remain below the 10us sanity ceiling");
#ifdef _DEBUG
    TEST_ASSERT(push_ns_per_op < 300.0, "Amortized push must stay inside the debug sanity budget");
    TEST_ASSERT(push_read_ns_per_op < 500.0, "Amortized push+read must stay inside the debug sanity budget");
#else
    TEST_ASSERT(push_ns_per_op < 100.0, "Amortized push must stay well inside the hot-path budget");
    TEST_ASSERT(push_read_ns_per_op < 200.0, "Amortized push+read must stay well inside the hot-path budget");
#endif

    std::cout << "  [+] Latency benchmark completed successfully.\n";
    return true;
}

// TC-BUS-07: Shared Memory Ring Named IPC & Heartbeat Tracking
bool test_shared_memory_ring_ipc() {
    std::cout << "[RUN] TC-BUS-07: Testing SharedMemoryRing named IPC & heartbeat...\n";

    const std::string channel_name = "Local\\AimAgent_Test_ActuationChannel";
    auto writer = SharedMemoryRing<ActuationCommand, 8>::create(channel_name, 1001);
    TEST_ASSERT(writer.is_valid(), "Writer shared memory creation should succeed");
    TEST_ASSERT(writer.capacity() == 8, "Writer capacity should be 8");

    auto reader = SharedMemoryRing<ActuationCommand, 8>::open_read_only(channel_name);
    TEST_ASSERT(reader.is_valid(), "Reader attaching to shared memory should succeed");

    // Freshness must fail closed: a writer that has never published a heartbeat is not
    // "alive by default". The previous behaviour returned true here, which would have let a
    // consumer act on a channel whose producer had not started.
    TEST_ASSERT(!reader.is_writer_alive(1000000LL, 50000000LL),
                "A writer with no recorded heartbeat must be reported dead");

    // Test heartbeat tracking
    MonotonicNs now_ns = 5000000000LL; // 5.0 seconds
    writer.update_heartbeat(now_ns);
    TEST_ASSERT(reader.is_writer_alive(now_ns, 50000000LL), "Writer should be alive at current timestamp");
    TEST_ASSERT(!reader.is_writer_alive(now_ns + 100000000LL, 50000000LL), "Writer should be marked stale after timeout");

    // Write commands through writer
    for (std::uint64_t i = 1; i <= 5; ++i) {
        ActuationCommand cmd;
        cmd.generated_at_ns = static_cast<MonotonicNs>(i * 1000);
        cmd.delta_x_counts = static_cast<std::int32_t>(i * 10);
        cmd.delta_y_counts = static_cast<std::int32_t>(-static_cast<std::int64_t>(i * 5));
        const auto res = writer.write_latest(cmd, now_ns + static_cast<MonotonicNs>(i * 1000));
        TEST_ASSERT(res == BusWriteResult::ok, "write_latest should return ok");
    }

    // Read latest from reader
    ActuationCommand out_cmd{};
    std::uint64_t last_seq = 0;
    std::uint64_t drops = 0;

    const auto read_res = reader.try_read_latest(out_cmd, last_seq, &drops);
    TEST_ASSERT(read_res == BusReadResult::ok, "reader.try_read_latest should succeed");
    TEST_ASSERT(last_seq == 5, "Reader should read latest sequence 5");
    TEST_ASSERT(out_cmd.generated_at_ns == 5000, "Payload generated_at_ns should match sequence 5");
    TEST_ASSERT(out_cmd.delta_x_counts == 50, "Payload delta_x should match sequence 5");
    TEST_ASSERT(out_cmd.delta_y_counts == -25, "Payload delta_y should match sequence 5");

    // Attaching a second read-write handle must not claim creation. On POSIX the previous
    // code set is_creator on every read-write open, so this attach re-initialized the live
    // producer's control header and unlinked the channel name when it closed.
    {
        auto attached = SharedMemoryRing<ActuationCommand, 8>::create(channel_name, 2002);
        TEST_ASSERT(attached.is_valid(), "Attaching a second writer handle should succeed");
        TEST_ASSERT(attached.latest_sequence() == 5,
                    "Attaching must not reset the live producer's published sequence");
    }

    ActuationCommand post_attach_cmd{};
    std::uint64_t post_attach_seq = 0;
    TEST_ASSERT(reader.try_read_latest(post_attach_cmd, post_attach_seq) == BusReadResult::ok,
                "Channel must survive a detaching non-creator handle");
    TEST_ASSERT(post_attach_seq == 5, "Sequence must be intact after the attach/detach cycle");

    std::cout << "  [+] SharedMemoryRing IPC and heartbeat verified.\n";
    return true;
}

// TC-BUS-08: Canonical Message Payload Traits & Struct Conformance
bool test_canonical_message_traits() {
    std::cout << "[RUN] TC-BUS-08: Verifying canonical message traits and magic identifiers...\n";

    // FrameDescriptor
    TEST_ASSERT(BusPayloadTraits<FrameDescriptor>::kMagicIdentifier == 0x31524641, "FrameDescriptor magic 'AFR1'");
    TEST_ASSERT(BusPayloadTraits<FrameDescriptor>::kChannelId == ChannelId::frame_descriptor, "ChannelId::frame_descriptor");

    // TargetObservationBatch
    TEST_ASSERT(BusPayloadTraits<TargetObservationBatch>::kMagicIdentifier == 0x31424F41, "TargetObservationBatch magic 'AOB1'");
    TargetObservationBatch obs_batch;
    TargetObservation obs;
    obs.center_px = {100.0f, 200.0f};
    obs.confidence = 0.95f;
    TEST_ASSERT(obs_batch.add_target(obs), "add_target should succeed");
    TEST_ASSERT(obs_batch.items().size() == 1, "items().size() should be 1");
    TEST_ASSERT(obs_batch.items()[0].confidence == 0.95f, "Target confidence should match");

    // TrackedTargetBatch
    TEST_ASSERT(BusPayloadTraits<TrackedTargetBatch>::kMagicIdentifier == 0x31545441, "TrackedTargetBatch magic 'ATT1'");
    TrackedTargetBatch track_batch;
    TrackedTarget track;
    track.track_id = 42;
    track.state = TrackState::confirmed;
    TEST_ASSERT(track_batch.add_track(track), "add_track should succeed");
    TEST_ASSERT(track_batch.items().size() == 1, "tracks.size() should be 1");
    TEST_ASSERT(track_batch.items()[0].track_id == 42, "Track ID should match");

    // AimIntent
    TEST_ASSERT(BusPayloadTraits<AimIntent>::kMagicIdentifier == 0x31494141, "AimIntent magic 'AAI1'");

    // ActuationCommand
    TEST_ASSERT(BusPayloadTraits<ActuationCommand>::kMagicIdentifier == 0x31434141, "ActuationCommand magic 'AAC1'");

    // HotLoopTelemetryEvent
    TEST_ASSERT(BusPayloadTraits<HotLoopTelemetryEvent>::kMagicIdentifier == 0x31455441, "HotLoopTelemetryEvent magic 'ATE1'");
    HotLoopTelemetryEvent telem;
    TEST_ASSERT(telem.add_stage(PipelineStage::preprocess, 1000, 1500), "add_stage should succeed");
    TEST_ASSERT(telem.stage_items().size() == 1, "stage_items() size should be 1");
    TEST_ASSERT(telem.stage_items()[0].duration_ns == 500, "Stage duration should be 500ns");

    std::cout << "  [+] Canonical message traits verified.\n";
    return true;
}

// =============================================================================
// Main Entrypoint
// =============================================================================

int main() {
    std::cout << "=================================================================\n";
    std::cout << "OpenPrism C++ Bus Ring Unit Test Suite (Milestone M1-04 #10)\n";
    std::cout << "=================================================================\n";

    bool all_passed = true;
    all_passed &= test_alignments_and_layouts();
    all_passed &= test_basic_push_pop_latest();
    all_passed &= test_latest_wins_overwrite_and_drop_accounting();
    all_passed &= test_historical_sequence_addressing();
    all_passed &= test_concurrent_multithreaded_stress();
    all_passed &= test_latency_benchmark();
    all_passed &= test_shared_memory_ring_ipc();
    all_passed &= test_canonical_message_traits();

    std::cout << "=================================================================\n";
    if (all_passed) {
        std::cout << "[+] ALL BUS RING TESTS PASSED SUCCESSFULLY (8/8)\n";
        std::cout << "=================================================================\n";
        return 0;
    } else {
        std::cerr << "[-] SOME BUS RING TESTS FAILED\n";
        std::cout << "=================================================================\n";
        return 1;
    }
}
