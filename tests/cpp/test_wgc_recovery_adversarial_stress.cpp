// tests/cpp/test_wgc_recovery_adversarial_stress.cpp
// Challenger 1 Empirical Adversarial Stress & Chaos Test Suite for Issue M2-02 #14
// (WgcFrameSource, CaptureRecoveryStateMachine, and UnifiedCaptureSource)

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <thread>
#include <vector>

#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/capture_recovery_state_machine.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/capture/unified_capture_source.hpp"
#include "aim/capture/wgc_backend.hpp"
#include "aim/capture/wgc_frame_source.hpp"
#include "aim/core/clock.hpp"

namespace aim {
inline std::ostream& operator<<(std::ostream& os, FrameSourceBackend b) {
    return os << static_cast<std::uint32_t>(b);
}
inline std::ostream& operator<<(std::ostream& os, FrameFormat f) {
    return os << static_cast<std::uint32_t>(f);
}
} // namespace aim

namespace aim::capture {
inline std::ostream& operator<<(std::ostream& os, CaptureRecoveryState s) {
    return os << static_cast<std::uint32_t>(s);
}
inline std::ostream& operator<<(std::ostream& os, CaptureState s) {
    return os << static_cast<std::uint32_t>(s);
}
} // namespace aim::capture

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
// Zero-Allocation Hook Instrumentation
// =============================================================================

struct AllocTracker {
    std::atomic<std::size_t> count{0};
    std::atomic<std::size_t> bytes{0};
    bool active{false};
};

static AllocTracker g_alloc_tracker;

void start_alloc_tracking() noexcept {
    g_alloc_tracker.count.store(0, std::memory_order_relaxed);
    g_alloc_tracker.bytes.store(0, std::memory_order_relaxed);
    g_alloc_tracker.active = true;
}

void stop_alloc_tracking() noexcept {
    g_alloc_tracker.active = false;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void* operator new(std::size_t size) {
    if (g_alloc_tracker.active) {
        g_alloc_tracker.count.fetch_add(1, std::memory_order_relaxed);
        g_alloc_tracker.bytes.fetch_add(size, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
#pragma GCC diagnostic pop
#endif

namespace {

// =============================================================================
// Test 1: Exhaustive Actuation Gating & Invariant Verification (Task 4)
// =============================================================================
int test_actuation_gating_exhaustive_invariants() {
    std::cout << "  [Stress 1] Exhaustive Actuation Gating Across All Recovery States..." << std::endl;

    aim::capture::CaptureRecoveryStateMachine sm(10'000'000LL, 250'000'000LL, 3);

    // Initial state: uninitialized
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::uninitialized);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to stopped
    sm.transition(aim::capture::CaptureEventTrigger::initialize_success, 1'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::stopped);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to active_dxgi via start
    sm.transition(aim::capture::CaptureEventTrigger::start_command, 2'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::active_dxgi);
    ASSERT_TRUE(sm.is_actuation_permitted());

    // Transition to access_lost -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::access_lost, 3'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::access_lost);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to reinitializing -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, 4'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::reinitializing);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to backoff_wait on reinit_failure -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, 5'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::backoff_wait);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to device_lost -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::device_lost, 6'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::device_lost);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Transition to active_wgc via fallback -> PERMITTED
    sm.transition(aim::capture::CaptureEventTrigger::fallback_to_wgc, 7'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::active_wgc);
    ASSERT_TRUE(sm.is_actuation_permitted());

    // Transition to stopped -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::stop_command, 8'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::stopped);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Force terminal failed state by 3 consecutive reinit failures -> GATED
    sm.transition(aim::capture::CaptureEventTrigger::start_command, 9'000'000LL);
    sm.transition(aim::capture::CaptureEventTrigger::access_lost, 10'000'000LL); // failure 1
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, 11'000'000LL);
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, 12'000'000LL); // failure 2
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, 13'000'000LL);
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, 14'000'000LL); // failure 3 -> failed
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::failed);
    ASSERT_FALSE(sm.is_actuation_permitted());

    return 0;
}

// =============================================================================
// Test 2: High-Frequency Bursty Ingestion Stress on WgcFrameSource (Task 1 & 2a, 2b)
// =============================================================================
int test_wgc_high_frequency_bursty_stress() {
    std::cout << "  [Stress 2] WGC High-Frequency Bursty Frame Arrival Stress (50,000 bursts)..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    std::mt19937_64 rng(0xDEADBEEF12345678ULL);
    std::uint64_t total_injected_frames = 0;
    std::uint64_t expected_dropped_frames = 0;
    std::uint64_t expected_acquired_frames = 0;

    for (std::uint64_t burst = 1; burst <= 50000ULL; ++burst) {
        // Random burst size: between 1 and 8 frames queued in this tick
        const std::size_t burst_size = 1 + (rng() % 8);

        for (std::size_t f = 0; f < burst_size; ++f) {
            ++total_injected_frames;
            const std::uint64_t qpc = 1000ULL * total_injected_frames;
            backend_ptr->queue_acquire_result(S_OK, qpc, 1920, 1080);
        }

        // try_acquire_latest() must drain all intermediate frames and lease exactly the latest
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(lease.is_valid());
        ASSERT_EQ(lease.width_px(), 1920u);
        ASSERT_EQ(lease.height_px(), 1080u);

        ++expected_acquired_frames;
        expected_dropped_frames += (burst_size - 1);

        ASSERT_FALSE(backend_ptr->is_frame_currently_held());

        // Release lease
        lease.reset();
        ASSERT_FALSE(lease.is_valid());
    }

    const auto health = source.health();
    ASSERT_EQ(health.total_frames_acquired, expected_acquired_frames);
    ASSERT_EQ(health.total_frames_dropped, expected_dropped_frames);
    ASSERT_EQ(total_injected_frames, expected_acquired_frames + expected_dropped_frames);
    ASSERT_EQ(health.total_access_loss_events, 0u);

    return 0;
}

// =============================================================================
// Test 3: Rapid Backend Switching Chaos Stress (DXGI <-> WGC) (Task 2c)
// =============================================================================
int test_unified_rapid_backend_switching_chaos() {
    std::cout << "  [Stress 3] Unified Rapid Backend Switching Chaos (2,000 transitions)..." << std::endl;

    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto* wgc_ptr = mock_wgc.get();

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);
    source.set_dxgi_probe_interval_ns(10'000'000LL); // 10ms probe interval for rapid promotion

    aim::FrameSourceConfig config{};
    config.backend = aim::FrameSourceBackend::dxgi_duplication;
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);
    ASSERT_TRUE(source.is_actuation_permitted());

    std::uint64_t dxgi_acquisitions = 0;
    std::uint64_t wgc_acquisitions = 0;

    for (std::uint64_t cycle = 0; cycle < 2000ULL; ++cycle) {
        // 1. Initial frame on DXGI
        dxgi_ptr->queue_acquire_result(S_OK, 1000ULL * (cycle * 10 + 1));
        aim::FrameLease l1;
        ASSERT_TRUE(source.try_acquire_latest(l1));
        ASSERT_TRUE(l1.is_valid());
        ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);
        ASSERT_TRUE(source.is_actuation_permitted());
        ++dxgi_acquisitions;
        l1.reset();

        // 2. Inject DXGI access loss -> triggers instant fallback to WGC!
        dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);
        dxgi_ptr->set_init_result(false); // DXGI remains unavailable temporarily

        wgc_ptr->queue_acquire_result(S_OK, 1000ULL * (cycle * 10 + 2));
        aim::FrameLease l2;
        ASSERT_TRUE(source.try_acquire_latest(l2));
        ASSERT_TRUE(l2.is_valid());
        ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);
        ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_wgc);
        ASSERT_TRUE(source.is_actuation_permitted());
        ++wgc_acquisitions;
        l2.reset();

        // 3. Acquire 2 more frames on WGC fallback
        for (std::uint64_t i = 0; i < 2; ++i) {
            wgc_ptr->queue_acquire_result(S_OK, 1000ULL * (cycle * 10 + 3 + i));
            aim::FrameLease l_wgc;
            ASSERT_TRUE(source.try_acquire_latest(l_wgc));
            ASSERT_TRUE(l_wgc.is_valid());
            ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);
            ASSERT_TRUE(source.is_actuation_permitted());
            ++wgc_acquisitions;
            l_wgc.reset();
        }

        // 4. Restore DXGI availability and advance clock past probe interval (15ms)
        dxgi_ptr->set_init_result(true);
        clock->advance_ns(15'000'000LL);

        dxgi_ptr->queue_acquire_result(S_OK, 1000ULL * (cycle * 10 + 6));
        aim::FrameLease l3;
        ASSERT_TRUE(source.try_acquire_latest(l3));
        ASSERT_TRUE(l3.is_valid());
        ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);
        ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_dxgi);
        ASSERT_TRUE(source.is_actuation_permitted());
        ++dxgi_acquisitions;
        l3.reset();
    }

    ASSERT_TRUE(dxgi_acquisitions > 0);
    ASSERT_TRUE(wgc_acquisitions > 0);

    return 0;
}

// =============================================================================
// Test 4: Strict Zero-Allocation Hot-Path Audit (Task 3)
// =============================================================================
int test_zero_allocation_hot_path_audit() {
    std::cout << "  [Stress 4] Zero-Allocation Hot-Path Audit (50,000 frames)..." << std::endl;

    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto* wgc_ptr = mock_wgc.get();

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);

    aim::FrameSourceConfig config{};
    config.backend = aim::FrameSourceBackend::dxgi_duplication;
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Phase 1: Warmup 100 frames
    for (std::uint64_t i = 0; i < 100; ++i) {
        dxgi_ptr->queue_acquire_result(S_OK, 1000ULL * (i + 1));
        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));
    }

    // Phase 2: Steady-State DXGI Audit (25,000 frames)
    dxgi_ptr->clear_acquire_queue();
    start_alloc_tracking();

    for (std::uint64_t i = 0; i < 25000; ++i) {
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        if (!ok || !lease.is_valid()) {
            stop_alloc_tracking();
            std::cerr << "DXGI acquisition failed at frame " << i << std::endl;
            return 1;
        }
    }

    stop_alloc_tracking();

    ASSERT_EQ(g_alloc_tracker.count.load(), 0u);
    ASSERT_EQ(g_alloc_tracker.bytes.load(), 0u);

    // Switch to WGC fallback
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);
    dxgi_ptr->set_init_result(false);
    wgc_ptr->queue_acquire_result(S_OK, 50'000'000ULL);

    aim::FrameLease switch_lease;
    ASSERT_TRUE(source.try_acquire_latest(switch_lease));
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    // Phase 3: Steady-State WGC Audit (25,000 frames)
    wgc_ptr->clear_acquire_queue();
    start_alloc_tracking();

    for (std::uint64_t i = 0; i < 25000; ++i) {
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        if (!ok || !lease.is_valid()) {
            stop_alloc_tracking();
            std::cerr << "WGC acquisition failed at frame " << i << std::endl;
            return 1;
        }
    }

    stop_alloc_tracking();

    ASSERT_EQ(g_alloc_tracker.count.load(), 0u);
    ASSERT_EQ(g_alloc_tracker.bytes.load(), 0u);

    return 0;
}

// =============================================================================
// Test 5: Concurrent Multi-Threaded Bus Ring Pipeline Stress
// =============================================================================
int test_concurrent_multithreaded_pipeline_stress() {
    std::cout << "  [Stress 5] Concurrent Multi-Threaded Bus Ring Pipeline (20,000 frames)..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);
    source.set_pipeline_run_id(42);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    std::atomic<bool> stop_consumer{false};
    std::atomic<std::uint64_t> consumer_received_frames{0};
    std::atomic<aim::SequenceId> last_seen_sequence{0};
    std::atomic<bool> consumer_error{false};

    // Consumer thread draining ring
    std::thread consumer_thread([&]() {
        aim::bus::FrameDescriptor desc{};
        std::uint64_t dropped = 0;
        while (!stop_consumer.load(std::memory_order_relaxed)) {
            if (ring.pop_latest(desc, dropped)) {
                consumer_received_frames.fetch_add(1, std::memory_order_relaxed);
                last_seen_sequence.store(desc.frame_id, std::memory_order_relaxed);
                if (desc.header.pipeline_run_id != 42u || desc.width != 1920u || desc.height != 1080u) {
                    consumer_error.store(true, std::memory_order_relaxed);
                }
            } else {
                std::this_thread::yield();
            }
        }
    });

    // Producer loop
    for (std::uint64_t i = 1; i <= 20000ULL; ++i) {
        backend_ptr->queue_acquire_result(S_OK, 1000ULL * i, 1920, 1080);
        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));
        ASSERT_TRUE(lease.is_valid());
        ASSERT_EQ(lease.frame_id(), i);
    }

    // Give consumer time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    stop_consumer.store(true, std::memory_order_relaxed);
    consumer_thread.join();

    ASSERT_FALSE(consumer_error.load());
    ASSERT_TRUE(consumer_received_frames.load() > 0);
    ASSERT_TRUE(last_seen_sequence.load() <= 20000ULL);

    return 0;
}

// =============================================================================
// Test 6: Empirical Verification of Actuation Gating Failure on Device Lost
// =============================================================================
int test_actuation_gating_on_device_removed_bug_demonstration() {
    std::cout << "  [Stress 6] Empirical Probe: Actuation Gating on Device Lost..." << std::endl;

    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto* wgc_ptr = mock_wgc.get();

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);

    aim::FrameSourceConfig config{};
    config.backend = aim::FrameSourceBackend::dxgi_duplication;
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());
    ASSERT_TRUE(source.is_actuation_permitted());

    // Phase 1: Test DXGI backend DEVICE_REMOVED
    dxgi_ptr->queue_acquire_result(S_OK, 100'000ULL);
    aim::FrameLease l1;
    ASSERT_TRUE(source.try_acquire_latest(l1));
    l1.reset();

    // Inject DEVICE_REMOVED on DXGI without WGC frame queued
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED, 0);
    aim::FrameLease l_dxgi_fail;
    ASSERT_FALSE(source.try_acquire_latest(l_dxgi_fail));
    ASSERT_FALSE(l_dxgi_fail.is_valid());
    const bool dxgi_actuation_after_crash = source.is_actuation_permitted();
    std::cout << "    [Observation] After DXGI DEVICE_REMOVED: is_actuation_permitted() = "
              << (dxgi_actuation_after_crash ? "TRUE (BUG: FAIL-OPEN)" : "FALSE (SAFE: FAIL-CLOSED)")
              << ", recovery_state() = " << static_cast<int>(source.recovery_state()) << std::endl;
    ASSERT_FALSE(dxgi_actuation_after_crash);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::device_lost);

    // Phase 2: Advance clock past backoff and recover into WGC active
    clock->advance_ns(50'000'000LL); // 50ms past backoff
    dxgi_ptr->set_init_result(false); // DXGI remains down
    wgc_ptr->set_init_result(true);
    wgc_ptr->queue_acquire_result(S_OK, 200'000ULL);

    aim::FrameLease l2;
    ASSERT_TRUE(source.try_acquire_latest(l2));
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_wgc);
    ASSERT_TRUE(source.is_actuation_permitted());
    l2.reset();

    // Phase 3: Now inject DEVICE_RESET on WGC backend (GPU crash while on WGC)
    wgc_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_RESET, 0);
    aim::FrameLease l3;
    const bool acquired_on_crash = source.try_acquire_latest(l3);
    ASSERT_FALSE(acquired_on_crash);
    ASSERT_FALSE(l3.is_valid());

    // EMPIRICAL BUG CHECK:
    // When WGC backend crashes with DXGI_ERROR_DEVICE_RESET, WgcFrameSource enters CaptureState::device_lost.
    // UnifiedCaptureSource MUST immediately gate actuation (is_actuation_permitted() must be FALSE).
    const bool wgc_actuation_after_crash = source.is_actuation_permitted();
    std::cout << "    [Observation] After WGC DEVICE_RESET: is_actuation_permitted() = "
              << (wgc_actuation_after_crash ? "TRUE (BUG: FAIL-OPEN)" : "FALSE (SAFE: FAIL-CLOSED)")
              << ", recovery_state() = " << static_cast<int>(source.recovery_state()) << std::endl;

    ASSERT_FALSE(wgc_actuation_after_crash);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::device_lost);

    return 0;
}

} // namespace

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "Challenger 1: WGC & Recovery State Machine Stress Test Suite" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int res1 = test_actuation_gating_exhaustive_invariants();
    if (res1 != 0) return 1;

    int res2 = test_wgc_high_frequency_bursty_stress();
    if (res2 != 0) return 1;

    int res3 = test_unified_rapid_backend_switching_chaos();
    if (res3 != 0) return 1;

    int res4 = test_zero_allocation_hot_path_audit();
    if (res4 != 0) return 1;

    int res5 = test_concurrent_multithreaded_pipeline_stress();
    if (res5 != 0) return 1;

    int res6 = test_actuation_gating_on_device_removed_bug_demonstration();
    if (res6 != 0) {
        std::cout << "=================================================================" << std::endl;
        std::cout << "CHALLENGER 1 FINDING: Bug in UnifiedCaptureSource actuation gating under device loss reproduced!" << std::endl;
        std::cout << "=================================================================" << std::endl;
        return res6;
    }

    std::cout << "=================================================================" << std::endl;
    std::cout << "ALL 6 CHALLENGER 1 ADVERSARIAL STRESS SUITES PASSED CLEANLY!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
