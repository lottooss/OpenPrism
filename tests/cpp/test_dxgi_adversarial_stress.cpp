// tests/cpp/test_dxgi_adversarial_stress.cpp
// Adversarial Empirical Challenge Suite for Milestone M2-01 (#13 DXGI Capture Loop & Surface Leases)

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/clock.hpp"

namespace aim {
inline std::ostream& operator<<(std::ostream& os, FrameFormat f) {
    return os << static_cast<std::uint32_t>(f);
}
}

namespace aim::bus {
inline std::ostream& operator<<(std::ostream& os, FramePixelFormat f) {
    return os << static_cast<std::uint32_t>(f);
}
}

namespace aim::capture {
inline std::ostream& operator<<(std::ostream& os, CaptureState s) {
    return os << static_cast<std::uint32_t>(s);
}
}

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion FAILED: " #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        std::cerr << "Assertion FAILED: !" #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "Assertion FAILED: " #a " == " #b " (" << (a) << " != " << (b) << ") at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

namespace {

// =============================================================================
// 1. Heavy Alternating Fault Injection Stress Test (100 multi-fault cycles)
// =============================================================================
int challenge_1_heavy_alternating_fault_injection() {
    std::cout << "  [Challenge 1] Heavy Alternating Fault Injection Stress Test..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(1'000'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.timeout_ms = 16;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);

    // Sequence of 100 fault-injection cycles:
    // Pattern per cycle: S_OK -> TIMEOUT -> ACCESS_LOST -> retry(fail) -> retry(recover) -> E_ACCESSDENIED -> retry(recover) -> DEVICE_REMOVED -> retry(recover)
    for (std::size_t cycle = 0; cycle < 100; ++cycle) {
        // 1. Normal S_OK frame
        {
            backend_ptr->queue_acquire_result(S_OK, 10'000'000 + cycle * 100'000);
            aim::FrameLease lease1;
            ASSERT_TRUE(source.try_acquire_latest(lease1));
            ASSERT_TRUE(lease1.is_valid());
            ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
        }

        // 2. DXGI_ERROR_WAIT_TIMEOUT (cadence pause)
        {
            backend_ptr->queue_acquire_result(DXGI_ERROR_WAIT_TIMEOUT);
            aim::FrameLease lease_timeout;
            ASSERT_FALSE(source.try_acquire_latest(lease_timeout));
            ASSERT_FALSE(lease_timeout.is_valid());
            ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
            ASSERT_FALSE(source.health().is_access_lost);
        }

        // 3. DXGI_ERROR_ACCESS_LOST
        {
            backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
            aim::FrameLease lease_lost;
            ASSERT_FALSE(source.try_acquire_latest(lease_lost));
            ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);
            ASSERT_TRUE(source.health().is_access_lost);
            ASSERT_FALSE(source.health().is_active);

            // During backoff (< 10 ms), attempts fail fast
            clock->advance_ms(5.0);
            ASSERT_FALSE(source.try_acquire_latest(lease_lost));

            // Reinitialization fails on first try
            clock->advance_ms(6.0); // Total 11 ms
            backend_ptr->set_create_duplication_result(false);
            ASSERT_FALSE(source.try_acquire_latest(lease_lost));
            ASSERT_EQ(source.state(), aim::capture::CaptureState::backoff_wait);

            // Advance 21 ms (backoff was doubled to 20ms), succeed reinit and acquire S_OK
            clock->advance_ms(21.0);
            backend_ptr->set_create_duplication_result(true);
            backend_ptr->queue_acquire_result(S_OK, 20'000'000 + cycle * 100'000);
            aim::FrameLease lease_rec1;
            ASSERT_TRUE(source.try_acquire_latest(lease_rec1));
            ASSERT_TRUE(lease_rec1.is_valid());
            ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
            ASSERT_TRUE(source.health().is_active);
            ASSERT_FALSE(source.health().is_access_lost);
        }

        // 4. E_ACCESSDENIED (UAC prompt / secure desktop switch)
        {
            backend_ptr->queue_acquire_result(E_ACCESSDENIED);
            aim::FrameLease lease_denied;
            ASSERT_FALSE(source.try_acquire_latest(lease_denied));
            ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);
            ASSERT_TRUE(source.health().is_access_lost);

            // Recover from E_ACCESSDENIED after 11 ms
            clock->advance_ms(11.0);
            backend_ptr->set_create_duplication_result(true);
            backend_ptr->queue_acquire_result(S_OK, 30'000'000 + cycle * 100'000);
            aim::FrameLease lease_rec2;
            ASSERT_TRUE(source.try_acquire_latest(lease_rec2));
            ASSERT_TRUE(lease_rec2.is_valid());
            ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
        }

        // 5. DXGI_ERROR_DEVICE_REMOVED (TDR / driver crash)
        {
            backend_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED);
            aim::FrameLease lease_dev_lost;
            ASSERT_FALSE(source.try_acquire_latest(lease_dev_lost));
            ASSERT_EQ(source.state(), aim::capture::CaptureState::device_lost);
            ASSERT_FALSE(source.health().is_active);

            // Advance 12 ms, recover device
            clock->advance_ms(12.0);
            backend_ptr->queue_acquire_result(S_OK, 40'000'000 + cycle * 100'000);
            aim::FrameLease lease_rec3;
            ASSERT_TRUE(source.try_acquire_latest(lease_rec3));
            ASSERT_TRUE(lease_rec3.is_valid());
            ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
        }
    }

    ASSERT_EQ(source.health().total_timeouts, 100u);
    ASSERT_EQ(source.health().total_access_loss_events, 200u); // 100 ACCESS_LOST + 100 E_ACCESSDENIED
    ASSERT_EQ(source.health().total_frames_acquired, 400u);    // 4 S_OK per cycle * 100 cycles

    return 0;
}

// =============================================================================
// 2. Exponential Backoff Timing Precision & Non-Spinning Stress Test
// =============================================================================
int challenge_2_backoff_timing_precision_and_non_spinning() {
    std::cout << "  [Challenge 2] Exponential Backoff Timing & Non-Spinning Test..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Inject ACCESS_LOST
    backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
    aim::FrameLease lease;
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);

    const std::size_t base_duplication_calls = backend_ptr->create_duplication_call_count();

    // Verify exact exponential backoff doubling: 10ms, 20ms, 40ms, 80ms, 160ms, 250ms (capped), 250ms
    const std::vector<std::int64_t> expected_backoffs_ms = {10, 20, 40, 80, 160, 250, 250, 250};

    backend_ptr->set_create_duplication_result(false); // Fail reinit continuously

    std::size_t expected_retries = 0;
    for (std::size_t step = 0; step < expected_backoffs_ms.size(); ++step) {
        const std::int64_t backoff_ms = expected_backoffs_ms[step];

        // Hammer try_acquire_latest with 10,000 calls during the backoff window
        const auto wall_start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10'000; ++i) {
            ASSERT_FALSE(source.try_acquire_latest(lease));
        }
        const auto wall_duration = std::chrono::steady_clock::now() - wall_start;
        const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_duration).count();

        // Ensure 10,000 rejected calls complete quickly -> proof of zero busy-spin/blocking
        ASSERT_TRUE(wall_ns < 500'000'000LL);

        // Verify NO new create_duplication calls were made during sub-window calls
        ASSERT_EQ(backend_ptr->create_duplication_call_count(), base_duplication_calls + expected_retries);

        // Advance fake clock just before backoff threshold (backoff_ms - 1 ms)
        clock->advance_ms(static_cast<double>(backoff_ms - 1));
        ASSERT_FALSE(source.try_acquire_latest(lease));
        ASSERT_EQ(backend_ptr->create_duplication_call_count(), base_duplication_calls + expected_retries);

        // Advance 2 ms to cross the threshold -> triggers exactly ONE reinitialization retry
        clock->advance_ms(2.0);
        ASSERT_FALSE(source.try_acquire_latest(lease));
        ++expected_retries;
        ASSERT_EQ(backend_ptr->create_duplication_call_count(), base_duplication_calls + expected_retries);
    }

    // Now restore access and verify immediate clean recovery and backoff reset
    clock->advance_ms(251.0);
    backend_ptr->set_create_duplication_result(true);
    backend_ptr->queue_acquire_result(S_OK, 999'999'999);

    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
    ASSERT_FALSE(source.health().is_access_lost);
    ASSERT_TRUE(source.health().is_active);

    return 0;
}

// =============================================================================
// 3. Monotonic Timestamping & Correlation ID Verification
// =============================================================================
int challenge_3_timestamp_monotonicity_and_correlation_ids() {
    std::cout << "  [Challenge 3] Timestamp Monotonicity & Correlation ID Verification..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(500'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Configure pipeline run ID for correlation tracking
    const std::uint32_t kTestRunId = 0xCAFEBABE;
    source.set_pipeline_run_id(kTestRunId);

    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);

    std::uint64_t last_seq = 0;
    aim::MonotonicNs last_ts = 0;

    // Publish 5,000 continuous frames with simulated jittery QPC timestamps
    std::uint64_t current_qpc = 1'000'000ULL;
    for (std::size_t i = 1; i <= 5'000; ++i) {
        // Variable frame intervals: 144 Hz (69444 ticks), 240 Hz (41666 ticks), 60 Hz (166666 ticks)
        std::uint64_t delta_ticks = 69444ULL;
        if (i % 3 == 0) delta_ticks = 41666ULL;
        if (i % 7 == 0) delta_ticks = 166666ULL;

        current_qpc += delta_ticks;
        backend_ptr->queue_acquire_result(S_OK, current_qpc);

        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));
        ASSERT_TRUE(lease.is_valid());

        // 1. Verify FrameLease internal correlation
        ASSERT_EQ(lease.frame_id(), i);
        const aim::MonotonicNs expected_ns = source.is_10mhz()
            ? static_cast<aim::MonotonicNs>(current_qpc * 100ULL)
            : aim::qpc_to_ns_128(current_qpc, source.qpf());
        ASSERT_EQ(lease.captured_at_ns(), expected_ns);

        // 2. Read published FrameDescriptor from ring
        aim::bus::FrameDescriptor desc{};
        ASSERT_TRUE(ring.try_read_latest(desc));

        // 3. Verify strict monotonicity
        ASSERT_TRUE(desc.header.sequence_id > last_seq);
        ASSERT_TRUE(desc.header.source_timestamp_ns > last_ts);
        last_seq = desc.header.sequence_id;
        last_ts = desc.header.source_timestamp_ns;

        // 4. Verify cross-packet correlation IDs
        ASSERT_EQ(desc.header.sequence_id, i);
        ASSERT_EQ(desc.frame_id, i);
        ASSERT_EQ(desc.header.pipeline_run_id, kTestRunId);
        ASSERT_EQ(desc.header.source_timestamp_ns, expected_ns);
        ASSERT_EQ(desc.captured_at_ns, expected_ns);
        ASSERT_EQ(desc.pool_slot_index, lease.pool_slot_index());
        ASSERT_EQ(desc.adapter_luid, backend_ptr->adapter_luid());
        ASSERT_EQ(desc.width, 1920u);
        ASSERT_EQ(desc.height, 1080u);
        ASSERT_EQ(desc.format, aim::bus::FramePixelFormat::b8g8r8a8_unorm);
        ASSERT_TRUE(desc.is_keyframe);
    }

    ASSERT_EQ(source.health().total_frames_acquired, 5'000u);
    ASSERT_EQ(source.health().last_frame_timestamp_ns, last_ts);

    return 0;
}

// =============================================================================
// 4. Graceful Shutdown & Leak Verification Across All States
// =============================================================================
int challenge_4_shutdown_from_all_states_and_leak_audit() {
    std::cout << "  [Challenge 4] Shutdown & Leak Verification Across All States..." << std::endl;

    // Subtest 4.1: Shutdown from uninitialized state
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend));
            ASSERT_EQ(src.state(), aim::capture::CaptureState::uninitialized);
            // Destruct without initializing
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.2: Shutdown from stopped state
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend));
            aim::FrameSourceConfig cfg{};
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_EQ(src.state(), aim::capture::CaptureState::stopped);
            src.stop(); // Double stop
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.3: Shutdown from running state with active leases
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        auto* backend_ptr = mock_backend.get();
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend));
            aim::FrameSourceConfig cfg{};
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_TRUE(src.start());

            backend_ptr->queue_acquire_result(S_OK, 100'000);
            aim::FrameLease surviving_lease;
            ASSERT_TRUE(src.try_acquire_latest(surviving_lease));
            ASSERT_TRUE(surviving_lease.is_valid());
            ASSERT_EQ(src.surface_pool().active_lease_count(), 1u);

            // Stop source while lease is held
            src.stop();
            ASSERT_EQ(src.state(), aim::capture::CaptureState::stopped);
            ASSERT_FALSE(src.health().is_active);

            // Release lease before source destruction
            surviving_lease.reset();
            ASSERT_FALSE(surviving_lease.is_valid());
            ASSERT_EQ(src.surface_pool().active_lease_count(), 0u);
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.4: Shutdown from access_lost state
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        auto* backend_ptr = mock_backend.get();
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend));
            aim::FrameSourceConfig cfg{};
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_TRUE(src.start());

            backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
            aim::FrameLease l;
            ASSERT_FALSE(src.try_acquire_latest(l));
            ASSERT_EQ(src.state(), aim::capture::CaptureState::access_lost);
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.5: Shutdown from backoff_wait state
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        auto* backend_ptr = mock_backend.get();
        auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend), clock);
            aim::FrameSourceConfig cfg{};
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_TRUE(src.start());

            backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
            aim::FrameLease l;
            ASSERT_FALSE(src.try_acquire_latest(l));

            clock->advance_ms(11.0);
            backend_ptr->set_create_duplication_result(false);
            ASSERT_FALSE(src.try_acquire_latest(l));
            ASSERT_EQ(src.state(), aim::capture::CaptureState::backoff_wait);
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.6: Shutdown from device_lost state
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        auto* backend_ptr = mock_backend.get();
        {
            aim::capture::DxgiFrameSource src(std::move(mock_backend));
            aim::FrameSourceConfig cfg{};
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_TRUE(src.start());

            backend_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED);
            aim::FrameLease l;
            ASSERT_FALSE(src.try_acquire_latest(l));
            ASSERT_EQ(src.state(), aim::capture::CaptureState::device_lost);
        }
        ASSERT_EQ(tracker->load(), 0u);
    }

    // Subtest 4.7: Rapid 200 initialize/start/stop cycles
    {
        auto tracker = std::make_shared<std::atomic<std::size_t>>(0);
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(tracker);
        auto* backend_ptr = mock_backend.get();
        aim::capture::DxgiFrameSource src(std::move(mock_backend));
        aim::FrameSourceConfig cfg{};
        for (int cycle = 0; cycle < 200; ++cycle) {
            ASSERT_TRUE(src.initialize(cfg));
            ASSERT_TRUE(src.start());
            src.stop();
        }
        ASSERT_EQ(backend_ptr->active_staging_texture_count(), 4u); // Initialized with 4
        src.stop();
    }

    return 0;
}

// =============================================================================
// 5. Dynamic Resolution Switching Under Concurrent Faults
// =============================================================================
int challenge_5_resolution_switching_under_faults() {
    std::cout << "  [Challenge 5] Resolution Switching Under Concurrent Faults..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Switch resolutions dynamically: 1080p -> 1440p -> 4K -> 720p with faults interleaved
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> resolutions = {
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {1280, 720},
        {1920, 1080}
    };

    for (const auto& [w, h] : resolutions) {
        backend_ptr->set_output_dimensions(w, h);

        // Inject access lost right during mode switch
        backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
        aim::FrameLease lease;
        ASSERT_FALSE(source.try_acquire_latest(lease));
        ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);

        // Recover after backoff
        clock->advance_ms(15.0);
        backend_ptr->queue_acquire_result(S_OK, 100'000);

        ASSERT_TRUE(source.try_acquire_latest(lease));
        ASSERT_TRUE(lease.is_valid());
        ASSERT_EQ(lease.width_px(), w);
        ASSERT_EQ(lease.height_px(), h);
        ASSERT_EQ(source.surface_pool().width(), w);
        ASSERT_EQ(source.surface_pool().height(), h);
    }

    return 0;
}

// =============================================================================
// 6. Empirical Vulnerability Probe: Stale Lease Underflow Across Device Resets
// =============================================================================
int challenge_6_stale_lease_release_underflow_repro() {
    std::cout << "  [Challenge 6] Empirical Vulnerability Probe: Stale Lease Across Device Reset..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // 1. Acquire Lease 0
    backend_ptr->queue_acquire_result(S_OK, 100'000);
    aim::FrameLease lease0;
    ASSERT_TRUE(source.try_acquire_latest(lease0));
    ASSERT_EQ(lease0.pool_slot_index(), 0u);
    ASSERT_EQ(source.surface_pool().ref_count(0), 1u);

    // 2. Trigger DEVICE_REMOVED (causes surface_pool_.release_all())
    backend_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED);
    aim::FrameLease dummy;
    ASSERT_FALSE(source.try_acquire_latest(dummy));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::device_lost);

    // 3. Advance clock and recover device (causes surface_pool_.initialize())
    clock->advance_ms(15.0);
    backend_ptr->queue_acquire_result(S_OK, 200'000);
    aim::FrameLease new_lease;
    ASSERT_TRUE(source.try_acquire_latest(new_lease));
    ASSERT_TRUE(new_lease.is_valid());

    // 4. Check ref_count of Slot 0 in new pool
    // In newly initialized pool, slot 0 ref_count should be 1 (held by new_lease)
    ASSERT_EQ(source.surface_pool().ref_count(0), 1u);

    // 5. If downstream consumer now resets the OLD lease0 created BEFORE device reset:
    // This demonstrates whether release_surface bounds-checks / protects against underflow.
    const std::uint32_t ref_before = source.surface_pool().ref_count(0);
    lease0.reset();
    const std::uint32_t ref_after = source.surface_pool().ref_count(0);

    std::cout << "      [Vulnerability Observation] Slot 0 ref_count before stale release: "
              << ref_before << ", after stale release: " << ref_after << std::endl;

    return 0;
}

} // namespace

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Running Challenger 2 Adversarial Stress Suite for Milestone M2-01 (#13)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    if (challenge_1_heavy_alternating_fault_injection() != 0) return 1;
    if (challenge_2_backoff_timing_precision_and_non_spinning() != 0) return 1;
    if (challenge_3_timestamp_monotonicity_and_correlation_ids() != 0) return 1;
    if (challenge_4_shutdown_from_all_states_and_leak_audit() != 0) return 1;
    if (challenge_5_resolution_switching_under_faults() != 0) return 1;
    if (challenge_6_stale_lease_release_underflow_repro() != 0) return 1;

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALL 6 ADVERSARIAL CHALLENGES COMPLETED." << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
