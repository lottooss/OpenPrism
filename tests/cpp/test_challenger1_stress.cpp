// tests/cpp/test_challenger1_stress.cpp
// Challenger 1 Empirical Stress & Adversarial Test Harness for Milestone M2-01 (#13)

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
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/clock.hpp"

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
// Test 1: High-Frequency Lease Acquisition, Churn, and Move Semantics Stress
// =============================================================================
int test_stress_pool_and_lease_churn() {
    std::cout << "  [Stress 1] High-Frequency Pool and Lease Churn (100,000 iterations)..." << std::endl;

    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;

    const std::uint32_t kCapacity = 8;
    ASSERT_TRUE(pool.initialize(backend.get(), kCapacity, 1920, 1080));

    std::mt19937_64 rng(0xC0FFEE12345678ULL);

    // Maintain a rotating queue of active leases
    std::vector<aim::FrameLease> active_leases;
    active_leases.reserve(kCapacity);

    for (std::uint64_t i = 1; i <= 100000ULL; ++i) {
        const std::uint32_t slot = pool.acquire_free_slot();

        if (slot != aim::capture::GpuSurfacePool::kInvalidSlot) {
            // Create lease
            const aim::MonotonicNs ts_ns = static_cast<aim::MonotonicNs>(i * 69444ULL);
            aim::FrameLease lease = pool.create_lease(slot, i, ts_ns);
            ASSERT_TRUE(lease.is_valid());
            ASSERT_EQ(lease.frame_id(), i);
            ASSERT_EQ(lease.pool_slot_index(), slot);
            ASSERT_TRUE(pool.ref_count(slot) >= 1u);

            // Exercise move constructor and move assignment
            aim::FrameLease moved_lease = std::move(lease);
            ASSERT_FALSE(lease.is_valid());
            ASSERT_TRUE(moved_lease.is_valid());

            aim::FrameLease moved_assign;
            moved_assign = std::move(moved_lease);
            ASSERT_FALSE(moved_lease.is_valid());
            ASSERT_TRUE(moved_assign.is_valid());

            active_leases.push_back(std::move(moved_assign));
        } else {
            // Pool is full: verify active lease count is exactly capacity
            ASSERT_EQ(pool.active_lease_count(), kCapacity);
        }

        // Randomly release an active lease to simulate asynchronous consumer completion
        if (!active_leases.empty() && (active_leases.size() >= kCapacity || (rng() % 3 == 0))) {
            const std::size_t idx_to_remove = rng() % active_leases.size();
            const std::uint32_t released_slot = active_leases[idx_to_remove].pool_slot_index();
            active_leases[idx_to_remove].reset();
            ASSERT_FALSE(active_leases[idx_to_remove].is_valid());
            ASSERT_EQ(pool.ref_count(released_slot), 0u);

            active_leases.erase(active_leases.begin() + static_cast<std::ptrdiff_t>(idx_to_remove));
        }
    }

    // Release all remaining leases
    active_leases.clear();
    ASSERT_EQ(pool.active_lease_count(), 0u);
    for (std::uint32_t s = 0; s < kCapacity; ++s) {
        ASSERT_EQ(pool.ref_count(s), 0u);
    }

    return 0;
}

// =============================================================================
// Test 2: Slot Exhaustion, Refcount Integrity, and Move Chains
// =============================================================================
int test_stress_slot_exhaustion_and_move_chains() {
    std::cout << "  [Stress 2] Slot Exhaustion and Complex Move Chains..." << std::endl;

    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;

    const std::uint32_t kCapacity = 4;
    ASSERT_TRUE(pool.initialize(backend.get(), kCapacity, 1920, 1080));

    std::vector<aim::FrameLease> leases;
    for (std::uint32_t s = 0; s < kCapacity; ++s) {
        const std::uint32_t slot = pool.acquire_free_slot();
        ASSERT_EQ(slot, s);
        leases.push_back(pool.create_lease(slot, s + 1, static_cast<aim::MonotonicNs>((s + 1) * 1000ULL)));
        ASSERT_EQ(pool.ref_count(s), 1u);
    }
    ASSERT_EQ(pool.active_lease_count(), kCapacity);

    // Acquire on full pool -> returns kInvalidSlot
    ASSERT_EQ(pool.acquire_free_slot(), aim::capture::GpuSurfacePool::kInvalidSlot);

    // Deep move chain through a queue
    std::queue<aim::FrameLease> lease_queue;
    for (auto& l : leases) {
        lease_queue.push(std::move(l));
        ASSERT_FALSE(l.is_valid());
    }
    leases.clear();

    // Refcounts must remain strictly 1 per slot
    for (std::uint32_t s = 0; s < kCapacity; ++s) {
        ASSERT_EQ(pool.ref_count(s), 1u);
    }
    ASSERT_EQ(pool.active_lease_count(), kCapacity);

    // Pop from queue and reset one by one
    std::uint32_t popped_count = 0;
    while (!lease_queue.empty()) {
        aim::FrameLease l = std::move(lease_queue.front());
        lease_queue.pop();
        ASSERT_TRUE(l.is_valid());

        const std::uint32_t s = l.pool_slot_index();
        ASSERT_EQ(pool.ref_count(s), 1u);
        l.reset();
        ASSERT_EQ(pool.ref_count(s), 0u);
        ++popped_count;
        ASSERT_EQ(pool.active_lease_count(), kCapacity - popped_count);
    }

    ASSERT_EQ(pool.active_lease_count(), 0u);

    return 0;
}

// =============================================================================
// Test 3: Rapid Repeated Start/Stop/Acquire Cycles & State Transitions
// =============================================================================
int test_stress_rapid_start_stop_acquire_cycles() {
    std::cout << "  [Stress 3] Rapid Repeated Start/Stop/Acquire Cycles (500 cycles)..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10000000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.timeout_ms = 16;
    ASSERT_TRUE(source.initialize(config));

    for (std::uint64_t cycle = 0; cycle < 500ULL; ++cycle) {
        ASSERT_TRUE(source.start());
        ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
        ASSERT_TRUE(source.health().is_active);

        // Ingest 5 frames
        for (std::uint64_t f = 0; f < 5ULL; ++f) {
            backend_ptr->queue_acquire_result(S_OK, 100000ULL + cycle * 10000ULL + f * 1000ULL);
            aim::FrameLease lease;
            ASSERT_TRUE(source.try_acquire_latest(lease));
            ASSERT_TRUE(lease.is_valid());
        }

        source.stop();
        ASSERT_EQ(source.state(), aim::capture::CaptureState::stopped);
        ASSERT_FALSE(source.health().is_active);

        // try_acquire_latest on stopped source returns false
        aim::FrameLease lease_stopped;
        ASSERT_FALSE(source.try_acquire_latest(lease_stopped));
        ASSERT_FALSE(lease_stopped.is_valid());
    }

    ASSERT_EQ(source.health().total_frames_acquired, 2500u);
    ASSERT_EQ(source.health().total_frames_dropped, 0u);

    return 0;
}

// =============================================================================
// Test 4: Device Removal & TDR Storm Recovery Cycles (100 device resets)
// =============================================================================
int test_stress_device_removal_recovery_storm() {
    std::cout << "  [Stress 4] Device Removal and TDR Storm Recovery (100 resets)..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10000000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    for (std::uint64_t reset_idx = 0; reset_idx < 100ULL; ++reset_idx) {
        // Normal acquisition
        backend_ptr->queue_acquire_result(S_OK, 10000ULL + reset_idx * 100000ULL);
        aim::FrameLease lease_ok;
        ASSERT_TRUE(source.try_acquire_latest(lease_ok));
        ASSERT_TRUE(lease_ok.is_valid());

        // Device removal crash (alternating DEVICE_REMOVED and DEVICE_RESET)
        const HRESULT hr_crash = (reset_idx % 2 == 0) ? DXGI_ERROR_DEVICE_REMOVED : DXGI_ERROR_DEVICE_RESET;
        backend_ptr->queue_acquire_result(hr_crash);

        aim::FrameLease lease_crash;
        ASSERT_FALSE(source.try_acquire_latest(lease_crash));
        ASSERT_EQ(source.state(), aim::capture::CaptureState::device_lost);
        ASSERT_FALSE(source.health().is_active);

        // During backoff (< 10ms), try_acquire returns false without reallocating
        clock->advance_ms(5);
        ASSERT_FALSE(source.try_acquire_latest(lease_crash));

        // Advance past 10ms -> trigger full device & surface pool recovery
        clock->advance_ms(6);
        backend_ptr->queue_acquire_result(S_OK, 20000ULL + reset_idx * 100000ULL);

        aim::FrameLease lease_recovered;
        ASSERT_TRUE(source.try_acquire_latest(lease_recovered));
        ASSERT_TRUE(lease_recovered.is_valid());
        ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
        ASSERT_TRUE(source.health().is_active);
        ASSERT_FALSE(source.health().is_access_lost);
    }

    // Verify staging texture count is strictly bounded to pool capacity (no COM leaks)
    ASSERT_EQ(backend_ptr->active_staging_texture_count(), 4u);

    return 0;
}

// =============================================================================
// Test 5: Zero Steady-State Allocations Across 50,000 Operations
// =============================================================================
int test_stress_zero_steady_state_allocations() {
    std::cout << "  [Stress 5] Zero Steady-State Allocation Audit (50,000 operations)..." << std::endl;

    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10000000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
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

    for (std::uint64_t i = 1; i <= 50000ULL; ++i) {
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

    std::cout << "    Measured Heap Allocations: " << alloc_calls << " calls / " << alloc_bytes << " bytes." << std::endl;
    ASSERT_EQ(alloc_calls, 0u);
    ASSERT_EQ(alloc_bytes, 0u);

    return 0;
}

// =============================================================================
// Test 6: Invalid Config, Null Backend, and Boundary Edge Cases
// =============================================================================
int test_stress_boundary_and_invalid_configurations() {
    std::cout << "  [Stress 6] Boundary and Invalid Configuration Hardening..." << std::endl;

    // Subtest 6.1: GpuSurfacePool invalid capacities
    {
        auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
        aim::capture::GpuSurfacePool pool;

        ASSERT_FALSE(pool.initialize(nullptr, 4, 1920, 1080));
        ASSERT_FALSE(pool.initialize(backend.get(), 0, 1920, 1080));
        ASSERT_FALSE(pool.initialize(backend.get(), 1, 1920, 1080));
        ASSERT_FALSE(pool.initialize(backend.get(), 17, 1920, 1080));
        ASSERT_FALSE(pool.initialize(backend.get(), 4, 0, 1080));
        ASSERT_FALSE(pool.initialize(backend.get(), 4, 1920, 0));
        ASSERT_FALSE(pool.is_initialized());

        // Operations on uninitialized pool
        ASSERT_EQ(pool.acquire_free_slot(), aim::capture::GpuSurfacePool::kInvalidSlot);
        ASSERT_EQ(pool.get_texture(0), nullptr);
        ASSERT_EQ(pool.get_shared_handle(0), 0u);
        ASSERT_EQ(pool.ref_count(0), 0u);

        aim::FrameLease l = pool.create_lease(0, 1, 100);
        ASSERT_FALSE(l.is_valid());
    }

    // Subtest 6.2: DxgiFrameSource invalid configs
    {
        auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
        aim::capture::DxgiFrameSource src(std::move(mock_backend));

        aim::FrameSourceConfig bad_cfg{};
        bad_cfg.pool_capacity = 1;
        ASSERT_FALSE(src.initialize(bad_cfg));
        ASSERT_EQ(src.state(), aim::capture::CaptureState::failed);

        aim::FrameLease l;
        ASSERT_FALSE(src.try_acquire_latest(l));
        ASSERT_FALSE(l.is_valid());
    }

    // Subtest 6.3: FrameLease self-move assignment
    {
        auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
        aim::capture::GpuSurfacePool pool;
        ASSERT_TRUE(pool.initialize(backend.get(), 2, 1920, 1080));

        const std::uint32_t slot = pool.acquire_free_slot();
        aim::FrameLease l = pool.create_lease(slot, 100, 5000);
        ASSERT_TRUE(l.is_valid());
        ASSERT_EQ(pool.ref_count(slot), 1u);

        // Self move assign
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
        l = std::move(l);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

        ASSERT_TRUE(l.is_valid());
        ASSERT_EQ(pool.ref_count(slot), 1u);
        ASSERT_EQ(l.pool_slot_index(), slot);

        l.reset();
        ASSERT_FALSE(l.is_valid());
        ASSERT_EQ(pool.ref_count(slot), 0u);
    }

    return 0;
}

} // namespace

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Running Challenger 1 Empirical Stress Harness for Milestone M2-01 (#13)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    if (test_stress_pool_and_lease_churn() != 0) return 1;
    if (test_stress_slot_exhaustion_and_move_chains() != 0) return 1;
    if (test_stress_rapid_start_stop_acquire_cycles() != 0) return 1;
    if (test_stress_device_removal_recovery_storm() != 0) return 1;
    if (test_stress_zero_steady_state_allocations() != 0) return 1;
    if (test_stress_boundary_and_invalid_configurations() != 0) return 1;

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALL 6 CHALLENGER 1 STRESS TESTS PASSED WITH 100% SUCCESS (0 DEFECTS)." << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
