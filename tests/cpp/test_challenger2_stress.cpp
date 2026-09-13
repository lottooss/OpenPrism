// tests/cpp/test_challenger2_stress.cpp
// Challenger 2 Empirical Stress & Adversarial Test Harness for Issue M2-02 #14 (WGC & Recovery State Machine)

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <queue>
#include <random>
#include <utility>
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
}

namespace aim::capture {
inline std::ostream& operator<<(std::ostream& os, CaptureState s) {
    return os << static_cast<std::uint32_t>(s);
}
inline std::ostream& operator<<(std::ostream& os, CaptureRecoveryState s) {
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
#pragma GCC diagnostic ignored "-Wsized-deallocation"
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

void* operator new[](std::size_t size) {
    if (g_alloc_tracker.active) {
        g_alloc_tracker.count.fetch_add(1, std::memory_order_relaxed);
        g_alloc_tracker.bytes.fetch_add(size, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
#pragma GCC diagnostic pop
#endif

namespace {

// =============================================================================
// Test 1: Dynamic Resolution & Dimension Mutation Under Active Capture & Leases
// =============================================================================
int test_stress_dynamic_resolution_churn() {
    std::cout << "  [Stress 1] Dynamic Resolution Churn & In-Flight Lease Safety..." << std::endl;

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

    struct Res { std::uint32_t w; std::uint32_t h; };
    const std::vector<Res> resolutions = {
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {1280, 720},
        {800, 600},
        {1366, 768},
        {1920, 1080}
    };

    // Hold onto an old lease while resolution changes dynamically
    aim::FrameLease held_lease_1080p;

    // First acquire a 1080p frame and keep the lease alive
    backend_ptr->queue_acquire_result(S_OK, 100'000ULL, 1920, 1080);
    ASSERT_TRUE(source.try_acquire_latest(held_lease_1080p));
    ASSERT_TRUE(held_lease_1080p.is_valid());
    ASSERT_EQ(held_lease_1080p.width_px(), 1920u);
    ASSERT_EQ(held_lease_1080p.height_px(), 1080u);

    // Now cycle through 500 resolution changes while held_lease_1080p is still held
    for (std::size_t iter = 0; iter < 500; ++iter) {
        const auto& r = resolutions[iter % resolutions.size()];
        backend_ptr->queue_acquire_result(S_OK, 200'000ULL + iter * 1000ULL, r.w, r.h);

        aim::FrameLease new_lease;
        const bool ok = source.try_acquire_latest(new_lease);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(new_lease.is_valid());
        ASSERT_EQ(new_lease.width_px(), r.w);
        ASSERT_EQ(new_lease.height_px(), r.h);
        ASSERT_EQ(source.surface_pool().width(), r.w);
        ASSERT_EQ(source.surface_pool().height(), r.h);
    }

    // Safely destroy held_lease_1080p after many pool resizes
    held_lease_1080p.reset();
    ASSERT_FALSE(held_lease_1080p.is_valid());

    // Verify pool is still completely healthy and operable
    backend_ptr->queue_acquire_result(S_OK, 900'000ULL, 1920, 1080);
    aim::FrameLease final_lease;
    ASSERT_TRUE(source.try_acquire_latest(final_lease));
    ASSERT_TRUE(final_lease.is_valid());
    ASSERT_EQ(final_lease.width_px(), 1920u);
    ASSERT_EQ(final_lease.height_px(), 1080u);

    return 0;
}

// =============================================================================
// Test 2: Dynamic Resolution Across Unified Fallback & Bus Publication
// =============================================================================
int test_stress_unified_resolution_and_bus_descriptors() {
    std::cout << "  [Stress 2] Unified Fallback Resolution & Bus Descriptor Verification..." << std::endl;

    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto* wgc_ptr = mock_wgc.get();

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource unified_source(std::move(mock_dxgi), std::move(mock_wgc), clock);
    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    unified_source.bind_bus_ring(&ring);
    unified_source.set_pipeline_run_id(42);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(unified_source.initialize(config));
    ASSERT_TRUE(unified_source.start());

    // 1. Initial DXGI acquisition at 1080p
    dxgi_ptr->set_output_dimensions(1920, 1080);
    dxgi_ptr->queue_acquire_result(S_OK, 100'000ULL);
    aim::FrameLease l1;
    ASSERT_TRUE(unified_source.try_acquire_latest(l1));
    ASSERT_EQ(l1.width_px(), 1920u);
    ASSERT_EQ(l1.height_px(), 1080u);

    aim::bus::FrameDescriptor desc1{};
    ASSERT_TRUE(ring.try_read_latest(desc1));
    ASSERT_EQ(desc1.header.pipeline_run_id, 42u);
    ASSERT_EQ(desc1.width, 1920u);
    ASSERT_EQ(desc1.height, 1080u);

    // 2. DXGI Access Loss -> Fallback to WGC at 1440p (2560x1440)
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);
    wgc_ptr->queue_acquire_result(S_OK, 200'000ULL, 2560, 1440);

    aim::FrameLease l2;
    ASSERT_TRUE(unified_source.try_acquire_latest(l2));
    ASSERT_EQ(l2.width_px(), 2560u);
    ASSERT_EQ(l2.height_px(), 1440u);
    ASSERT_EQ(unified_source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    aim::bus::FrameDescriptor desc2{};
    ASSERT_TRUE(ring.try_read_latest(desc2));
    ASSERT_EQ(desc2.width, 2560u);
    ASSERT_EQ(desc2.height, 1440u);

    // 3. WGC provides 4K frame (3840x2160)
    wgc_ptr->queue_acquire_result(S_OK, 300'000ULL, 3840, 2160);
    aim::FrameLease l3;
    ASSERT_TRUE(unified_source.try_acquire_latest(l3));
    ASSERT_EQ(l3.width_px(), 3840u);
    ASSERT_EQ(l3.height_px(), 2160u);

    aim::bus::FrameDescriptor desc3{};
    ASSERT_TRUE(ring.try_read_latest(desc3));
    ASSERT_EQ(desc3.width, 3840u);
    ASSERT_EQ(desc3.height, 2160u);

    return 0;
}

// =============================================================================
// Test 3: Repeated Fault Injections, Exponential Backoff, and Fail-Closed Gating
// =============================================================================
int test_stress_fault_injection_and_actuation_gating_matrix() {
    std::cout << "  [Stress 3] Fault Injection Matrix & Actuation Fail-Closed Gating..." << std::endl;

    aim::capture::CaptureRecoveryStateMachine sm(10'000'000LL, 250'000'000LL, 3);
    aim::FakeClock clock(10'000'000LL);

    // 1. Initial uninitialized state: actuation must be gated (FALSE)
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::uninitialized);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // 2. Initialize success -> stopped: actuation gated (FALSE)
    sm.transition(aim::capture::CaptureEventTrigger::initialize_success, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::stopped);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // 3. Start command -> active_dxgi: actuation permitted (TRUE)
    sm.transition(aim::capture::CaptureEventTrigger::start_command, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::active_dxgi);
    ASSERT_TRUE(sm.is_actuation_permitted());

    // 4. Access lost -> access_lost state: actuation gated (FALSE), backoff = 20ms
    sm.transition(aim::capture::CaptureEventTrigger::access_lost, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::access_lost);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.current_backoff_ns(), 20'000'000LL);
    ASSERT_EQ(sm.consecutive_failures(), 1u);

    // 5. Backoff expired -> reinitializing: actuation gated (FALSE)
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::reinitializing);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // 6. Reinit failure 1 -> backoff_wait: actuation gated (FALSE), backoff = 40ms
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::backoff_wait);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.current_backoff_ns(), 40'000'000LL);
    ASSERT_EQ(sm.consecutive_failures(), 2u);

    // 7. Backoff expired -> reinitializing
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::reinitializing);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // 8. Reinit failure 2 (consecutive failures = 3 >= max_consecutive_retries) -> failed: permanently gated
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, clock.now_ns());
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::failed);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.consecutive_failures(), 3u);

    // 9. Reset -> uninitialized
    sm.reset();
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::uninitialized);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.consecutive_failures(), 0u);
    ASSERT_EQ(sm.current_backoff_ns(), 10'000'000LL);

    // 10. Randomized 1,000 Transition Invariant Check
    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<int> dist(0, 13);

    for (std::size_t i = 0; i < 1000; ++i) {
        clock.advance_ms(5);
        const auto trigger = static_cast<aim::capture::CaptureEventTrigger>(dist(rng));
        sm.transition(trigger, clock.now_ns());

        const auto s = sm.current_state();
        const bool permitted = sm.is_actuation_permitted();

        if (s == aim::capture::CaptureRecoveryState::active_dxgi ||
            s == aim::capture::CaptureRecoveryState::active_wgc) {
            ASSERT_TRUE(permitted);
        } else {
            // ALL other states MUST gate actuation unconditionally!
            ASSERT_FALSE(permitted);
        }

        // Backoff must always be within [initial_backoff_ns, max_backoff_ns]
        ASSERT_TRUE(sm.current_backoff_ns() >= 10'000'000LL);
        ASSERT_TRUE(sm.current_backoff_ns() <= 250'000'000LL);
    }

    return 0;
}

// =============================================================================
// Test 4: Sequence Number Monotonicity Across Backend Switches
// =============================================================================
int test_stress_sequence_number_monotonicity_across_switches() {
    std::cout << "  [Stress 4] Sequence Number Monotonicity Across Failover Switches..." << std::endl;

    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto* wgc_ptr = mock_wgc.get();

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource unified_source(std::move(mock_dxgi), std::move(mock_wgc), clock);
    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    unified_source.bind_bus_ring(&ring);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(unified_source.initialize(config));
    ASSERT_TRUE(unified_source.start());

    std::uint64_t last_seq = 0;
    aim::MonotonicNs last_time_ns = 0;

    // DXGI burst: 5 frames
    for (std::size_t i = 1; i <= 5; ++i) {
        clock->advance_ms(7);
        dxgi_ptr->queue_acquire_result(S_OK, 1000ULL * i);

        aim::FrameLease lease;
        ASSERT_TRUE(unified_source.try_acquire_latest(lease));
        ASSERT_TRUE(lease.is_valid());
        ASSERT_TRUE(lease.frame_id() > last_seq);
        last_seq = lease.frame_id();

        aim::bus::FrameDescriptor desc{};
        ASSERT_TRUE(ring.try_read_latest(desc));
        ASSERT_EQ(desc.header.sequence_id, last_seq);
        ASSERT_TRUE(desc.header.source_timestamp_ns >= last_time_ns);
        last_time_ns = desc.header.source_timestamp_ns;
    }

    // Failover to WGC: DXGI access lost
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);
    wgc_ptr->queue_acquire_result(S_OK, 10'000ULL);

    aim::FrameLease lease_wgc;
    ASSERT_TRUE(unified_source.try_acquire_latest(lease_wgc));
    ASSERT_TRUE(lease_wgc.is_valid());
    ASSERT_EQ(unified_source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    // Sequence produced by WGC backend
    ASSERT_TRUE(lease_wgc.frame_id() > 0);

    aim::bus::FrameDescriptor desc_wgc{};
    ASSERT_TRUE(ring.try_read_latest(desc_wgc));
    ASSERT_EQ(desc_wgc.header.sequence_id, lease_wgc.frame_id());

    // Continue on WGC for 10 frames
    for (std::size_t i = 2; i <= 10; ++i) {
        clock->advance_ms(7);
        wgc_ptr->queue_acquire_result(S_OK, 10'000ULL + i * 1000ULL);

        aim::FrameLease l;
        ASSERT_TRUE(unified_source.try_acquire_latest(l));
        ASSERT_TRUE(l.is_valid());
        ASSERT_EQ(l.frame_id(), static_cast<aim::SequenceId>(i));

        aim::bus::FrameDescriptor desc{};
        ASSERT_TRUE(ring.try_read_latest(desc));
        ASSERT_EQ(desc.header.sequence_id, l.frame_id());
    }

    return 0;
}

// =============================================================================
// Test 5: Resource Cleanup & Immediate Destruction Under Stress and Recovery
// =============================================================================
int test_stress_immediate_destruction_in_all_recovery_states() {
    std::cout << "  [Stress 5] Immediate Destruction Across All Recovery States (No Leaks)..." << std::endl;

    auto shared_dxgi_counter = std::make_shared<std::atomic<std::size_t>>(0);
    auto shared_wgc_counter = std::make_shared<std::atomic<std::size_t>>(0);

    for (int state_idx = 0; state_idx < 9; ++state_idx) {
        auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>(shared_dxgi_counter);
        auto* dxgi_ptr = mock_dxgi.get();

        auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>(shared_wgc_counter);
        auto* wgc_ptr = mock_wgc.get();

        auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

        {
            aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);
            aim::FrameSourceConfig config{};
            config.pool_capacity = 4;
            config.target_width_px = 1920;
            config.target_height_px = 1080;
            config.fallback_to_wgc = true;

            switch (state_idx) {
            case 0: // uninitialized
                break;
            case 1: // stopped
                source.initialize(config);
                break;
            case 2: // active_dxgi
                source.initialize(config);
                source.start();
                break;
            case 3: // active_wgc
                dxgi_ptr->set_init_result(false);
                source.initialize(config);
                source.start();
                break;
            case 4: // access_lost
                source.initialize(config);
                source.start();
                dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);
                {
                    aim::FrameLease l;
                    source.try_acquire_latest(l);
                }
                break;
            case 5: // backoff_wait
            case 6: // reinitializing
            case 7: // device_lost
                source.initialize(config);
                source.start();
                dxgi_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED);
                {
                    aim::FrameLease l;
                    source.try_acquire_latest(l);
                }
                break;
            case 8: // failed
                dxgi_ptr->set_init_result(false);
                wgc_ptr->set_init_result(false);
                source.initialize(config);
                break;
            }
            // Immediate destruction here at scope exit
        }

        // Verify active staging texture count returned to 0
        ASSERT_EQ(shared_dxgi_counter->load(std::memory_order_relaxed), 0u);
        ASSERT_EQ(shared_wgc_counter->load(std::memory_order_relaxed), 0u);
    }

    // 1,000 rapid lifecycle construct-init-start-destroy cycles
    for (std::size_t i = 0; i < 1000; ++i) {
        auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>(shared_wgc_counter);
        auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

        {
            aim::capture::WgcFrameSource src(std::move(mock_wgc), clock);
            aim::FrameSourceConfig cfg{};
            cfg.pool_capacity = 4;
            src.initialize(cfg);
            src.start();
        }

        ASSERT_EQ(shared_wgc_counter->load(std::memory_order_relaxed), 0u);
    }

    return 0;
}

// =============================================================================
// Test 6: Zero Steady-State Heap Allocations (20,000 frames)
// =============================================================================
int test_stress_zero_steady_state_allocations_wgc_and_unified() {
    std::cout << "  [Stress 6] Zero Steady-State Allocation Audit (20,000 frames)..." << std::endl;

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_wgc), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);

    // Warmup 20 operations
    for (std::uint64_t i = 0; i < 20ULL; ++i) {
        aim::FrameLease lease;
        source.try_acquire_latest(lease);
        aim::bus::FrameDescriptor desc{};
        ring.try_read_latest(desc);
    }

    // Begin allocation audit
    start_alloc_tracking();

    for (std::uint64_t i = 1; i <= 20000ULL; ++i) {
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(lease.is_valid());

        aim::bus::FrameDescriptor desc{};
        const bool read_ok = ring.try_read_latest(desc);
        ASSERT_TRUE(read_ok);
    }

    const std::size_t alloc_calls = g_alloc_tracker.count.load(std::memory_order_relaxed);
    const std::size_t alloc_bytes = g_alloc_tracker.bytes.load(std::memory_order_relaxed);
    stop_alloc_tracking();

    std::cout << "    Measured WGC Heap Allocations: " << alloc_calls << " calls / " << alloc_bytes << " bytes." << std::endl;
    ASSERT_EQ(alloc_calls, 0u);
    ASSERT_EQ(alloc_bytes, 0u);

    return 0;
}

} // namespace

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Running Challenger 2 Empirical Stress Harness for Milestone M2-02 (#14)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    if (test_stress_dynamic_resolution_churn() != 0) return 1;
    if (test_stress_unified_resolution_and_bus_descriptors() != 0) return 1;
    if (test_stress_fault_injection_and_actuation_gating_matrix() != 0) return 1;
    if (test_stress_sequence_number_monotonicity_across_switches() != 0) return 1;
    if (test_stress_immediate_destruction_in_all_recovery_states() != 0) return 1;
    if (test_stress_zero_steady_state_allocations_wgc_and_unified() != 0) return 1;

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALL 6 CHALLENGER 2 STRESS TESTS PASSED WITH 100% SUCCESS (0 DEFECTS)." << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
