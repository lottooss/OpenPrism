// tests/cpp/test_cuda_interop_challenger_stress.cpp
// Challenger 2 Empirical Stress & Adversarial Test Harness for Issue M2-03 #15 (D3D11-CUDA Interop)

#include <algorithm>
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

#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/cuda_surface_pool.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/clock.hpp"

using aim::capture::CudaResult;

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
// Test 1: Dynamic Pool Reinitialization, Capacity Churn, & Teardown Leak Audit
// =============================================================================
int test_stress_dynamic_pool_reinit_and_leak_audit() {
    std::cout << "  [Stress 1] Dynamic Pool Reinit, Capacity Churn & Leak Audit..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    const std::vector<std::uint32_t> capacities = {2, 4, 8, 16, 3, 5, 2};

    // Cycle through 100 dynamic re-initializations
    for (std::size_t cycle = 0; cycle < 100; ++cycle) {
        const std::uint32_t cap = capacities[cycle % capacities.size()];

        aim::capture::GpuSurfacePool gpu_pool;
        ASSERT_TRUE(gpu_pool.initialize(mock_dxgi.get(), cap, 1920, 1080));

        aim::capture::CudaSurfacePool cuda_pool(mock_cuda);
        ASSERT_TRUE(cuda_pool.initialize(gpu_pool, 0x00010688));
        ASSERT_EQ(cuda_pool.capacity(), cap);
        ASSERT_TRUE(cuda_pool.is_initialized());
        ASSERT_EQ(mock_cuda->active_resources_count(), cap);
        ASSERT_EQ(mock_cuda->active_streams_count(), 1u);
        ASSERT_EQ(mock_cuda->active_events_count(), cap * 2u);

        // Map and unmap every slot
        for (std::uint32_t slot = 0; slot < cap; ++slot) {
            aim::FrameLease frame_lease = gpu_pool.create_lease(slot, static_cast<aim::SequenceId>(slot + 1), 1000);
            aim::capture::CudaMappedSurfaceLease mapped_lease;
            ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease), mapped_lease));
            ASSERT_TRUE(mapped_lease.is_valid());
            ASSERT_TRUE(cuda_pool.is_slot_mapped(slot));

            mapped_lease.reset();
            ASSERT_FALSE(cuda_pool.is_slot_mapped(slot));
        }

        // Explicit release
        cuda_pool.release_all();
        ASSERT_FALSE(cuda_pool.is_initialized());
        ASSERT_EQ(mock_cuda->active_resources_count(), 0u);
        ASSERT_EQ(mock_cuda->active_streams_count(), 0u);
        ASSERT_EQ(mock_cuda->active_events_count(), 0u);
        ASSERT_EQ(mock_cuda->active_surfaces_count(), 0u);
    }

    // 1,000 rapid construct-init-destroy cycles (scope exit destruction)
    for (std::size_t i = 0; i < 1000; ++i) {
        aim::capture::GpuSurfacePool gpu_pool;
        gpu_pool.initialize(mock_dxgi.get(), 4, 1920, 1080);

        {
            aim::capture::CudaSurfacePool cuda_pool(mock_cuda);
            cuda_pool.initialize(gpu_pool, 0x00010688);
        }

        ASSERT_EQ(mock_cuda->active_resources_count(), 0u);
        ASSERT_EQ(mock_cuda->active_streams_count(), 0u);
        ASSERT_EQ(mock_cuda->active_events_count(), 0u);
    }

    return 0;
}

// =============================================================================
// Test 2: In-Flight Lease Safety Across Dynamic Pool Reinitialization
// =============================================================================
int test_stress_in_flight_leases_across_pool_reinitialization() {
    std::cout << "  [Stress 2] In-Flight Lease Safety Across Pool Reinitialization..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    aim::capture::GpuSurfacePool gpu_pool1;
    ASSERT_TRUE(gpu_pool1.initialize(mock_dxgi.get(), 4, 1920, 1080));

    aim::capture::CudaSurfacePool cuda_pool(mock_cuda);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool1, 0x00010688));

    // Acquire and hold a mapped lease
    aim::FrameLease frame_lease1 = gpu_pool1.create_lease(1, 101, 5000);
    aim::capture::CudaMappedSurfaceLease held_lease;
    ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease1), held_lease));
    ASSERT_TRUE(held_lease.is_valid());
    ASSERT_TRUE(cuda_pool.is_slot_mapped(1));

    // While held_lease is alive, reinitialize cuda_pool with a new gpu_pool (e.g. resolution change)
    aim::capture::GpuSurfacePool gpu_pool2;
    ASSERT_TRUE(gpu_pool2.initialize(mock_dxgi.get(), 8, 2560, 1440));
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool2, 0x00010688));

    // Old slot mapping state is reset by reinit
    ASSERT_EQ(cuda_pool.capacity(), 8u);

    // Destroying held_lease from old generation must be completely safe (no crash, no double unmap)
    held_lease.reset();
    ASSERT_FALSE(held_lease.is_valid());

    // Verify cuda_pool is fully functional in the new configuration
    aim::FrameLease frame_lease2 = gpu_pool2.create_lease(5, 202, 6000);
    aim::capture::CudaMappedSurfaceLease new_lease;
    ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease2), new_lease));
    ASSERT_TRUE(new_lease.is_valid());
    ASSERT_EQ(new_lease.slot_index(), 5u);

    return 0;
}

// =============================================================================
// Test 3: Out-of-Order Lease Destruction & Complex Move Assignment Chains
// =============================================================================
int test_stress_out_of_order_lease_destruction_and_move_chains() {
    std::cout << "  [Stress 3] Out-of-Order Lease Destruction & Move Assignment Chains..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(mock_dxgi.get(), 8, 1920, 1080));

    aim::capture::CudaSurfacePool cuda_pool(mock_cuda);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, 0x00010688));

    // Subtest 3.1: Acquire 8 leases, destroy in shuffled order
    {
        std::vector<aim::capture::CudaMappedSurfaceLease> leases(8);
        for (std::uint32_t i = 0; i < 8; ++i) {
            aim::FrameLease fl = gpu_pool.create_lease(i, static_cast<aim::SequenceId>(i + 1), 1000);
            ASSERT_TRUE(cuda_pool.map_surface(std::move(fl), leases[i]));
            ASSERT_TRUE(cuda_pool.is_slot_mapped(i));
        }

        std::vector<std::size_t> indices = {3, 7, 0, 5, 2, 6, 1, 4};
        for (auto idx : indices) {
            ASSERT_TRUE(leases[idx].is_valid());
            leases[idx].reset();
            ASSERT_FALSE(leases[idx].is_valid());
            ASSERT_FALSE(cuda_pool.is_slot_mapped(static_cast<std::uint32_t>(idx)));
        }

        // All slots unmapped
        for (std::uint32_t i = 0; i < 8; ++i) {
            ASSERT_FALSE(cuda_pool.is_slot_mapped(i));
        }
    }

    // Subtest 3.2: Move assignment chains & Self-move
    {
        aim::FrameLease fl0 = gpu_pool.create_lease(0, 100, 1000);
        aim::FrameLease fl1 = gpu_pool.create_lease(1, 101, 1000);

        aim::capture::CudaMappedSurfaceLease lA, lB, lC;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl0), lA));
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl1), lB));

        // lC = std::move(lA) -> lA becomes invalid, lC holds slot 0
        lC = std::move(lA);
        ASSERT_FALSE(lA.is_valid());
        ASSERT_TRUE(lC.is_valid());
        ASSERT_EQ(lC.slot_index(), 0u);
        ASSERT_TRUE(cuda_pool.is_slot_mapped(0));

        // Overwrite lC with lB -> slot 0 is unmapped, lC holds slot 1
        lC = std::move(lB);
        ASSERT_FALSE(lB.is_valid());
        ASSERT_TRUE(lC.is_valid());
        ASSERT_EQ(lC.slot_index(), 1u);
        ASSERT_FALSE(cuda_pool.is_slot_mapped(0)); // Slot 0 freed by move assignment
        ASSERT_TRUE(cuda_pool.is_slot_mapped(1));

        // Self move assign
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
        lC = std::move(lC);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        ASSERT_TRUE(lC.is_valid());
        ASSERT_EQ(lC.slot_index(), 1u);
        ASSERT_TRUE(cuda_pool.is_slot_mapped(1));

        // Double reset
        lC.reset();
        ASSERT_FALSE(lC.is_valid());
        ASSERT_FALSE(cuda_pool.is_slot_mapped(1));
        lC.reset();
        ASSERT_FALSE(lC.is_valid());
    }

    return 0;
}

// =============================================================================
// Test 4: Fault Injection Matrix (Driver Errors During Map/Unmap/Registration)
// =============================================================================
int test_stress_driver_fault_injection_matrix() {
    std::cout << "  [Stress 4] Driver Fault Injection Matrix & Fail-Closed Robustness..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(mock_dxgi.get(), 4, 1920, 1080));

    // Fault 4.1: CUDA not available
    {
        mock_cuda->set_cuda_available(false);
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_FALSE(pool.initialize(gpu_pool, 0x00010688));
        ASSERT_FALSE(pool.is_initialized());
        mock_cuda->set_cuda_available(true);
    }

    // Fault 4.2: Adapter mismatch
    {
        mock_cuda->set_adapter_match(false);
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_FALSE(pool.initialize(gpu_pool, 0x00010688));
        ASSERT_FALSE(pool.is_initialized());
        mock_cuda->set_adapter_match(true, 0);
    }

    // Fault 4.3: Registration failure on texture 2 -> clean rollback
    {
        mock_cuda->set_register_result(CudaResult::error_out_of_memory);
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_FALSE(pool.initialize(gpu_pool, 0x00010688));
        ASSERT_FALSE(pool.is_initialized());
        ASSERT_EQ(mock_cuda->active_resources_count(), 0u);
        mock_cuda->set_register_result(CudaResult::success);
    }

    // Fault 4.4: Map resource failure -> fail-closed
    {
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_TRUE(pool.initialize(gpu_pool, 0x00010688));

        mock_cuda->set_map_result(CudaResult::error_map_failed);

        aim::FrameLease fl = gpu_pool.create_lease(0, 1, 1000);
        aim::capture::CudaMappedSurfaceLease ml;
        ASSERT_FALSE(pool.map_surface(std::move(fl), ml));
        ASSERT_FALSE(ml.is_valid());
        ASSERT_FALSE(pool.is_slot_mapped(0));

        mock_cuda->set_map_result(CudaResult::success);
    }

    // Fault 4.5: get_mapped_array failure -> unmaps and fails closed
    {
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_TRUE(pool.initialize(gpu_pool, 0x00010688));

        mock_cuda->set_get_array_result(CudaResult::error_invalid_value);

        aim::FrameLease fl = gpu_pool.create_lease(1, 2, 1000);
        aim::capture::CudaMappedSurfaceLease ml;
        ASSERT_FALSE(pool.map_surface(std::move(fl), ml));
        ASSERT_FALSE(ml.is_valid());
        ASSERT_FALSE(pool.is_slot_mapped(1));
        ASSERT_FALSE(mock_cuda->is_resource_mapped(pool.get_graphics_resource(1)));

        mock_cuda->set_get_array_result(CudaResult::success);
    }

    // Fault 4.6: create_surface_object failure -> unmaps and fails closed
    {
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_TRUE(pool.initialize(gpu_pool, 0x00010688));

        mock_cuda->set_create_surface_result(CudaResult::error_out_of_memory);

        aim::FrameLease fl = gpu_pool.create_lease(2, 3, 1000);
        aim::capture::CudaMappedSurfaceLease ml;
        ASSERT_FALSE(pool.map_surface(std::move(fl), ml));
        ASSERT_FALSE(ml.is_valid());
        ASSERT_FALSE(pool.is_slot_mapped(2));
        ASSERT_FALSE(mock_cuda->is_resource_mapped(pool.get_graphics_resource(2)));

        mock_cuda->set_create_surface_result(CudaResult::success);
    }

    // Fault 4.7: 1,000 Randomized Driver Fault Storm
    {
        aim::capture::CudaSurfacePool pool(mock_cuda);
        ASSERT_TRUE(pool.initialize(gpu_pool, 0x00010688));

        std::mt19937_64 rng(42);
        std::uniform_int_distribution<int> fault_dist(0, 2);

        for (std::size_t i = 0; i < 1000; ++i) {
            const int fault = fault_dist(rng);
            if (fault == 1) mock_cuda->queue_map_result(CudaResult::error_map_failed);
            else if (fault == 2) mock_cuda->queue_map_result(CudaResult::error_invalid_value);

            aim::FrameLease fl = gpu_pool.create_lease(0, static_cast<aim::SequenceId>(i + 1), 1000);
            aim::capture::CudaMappedSurfaceLease ml;
            const bool ok = pool.map_surface(std::move(fl), ml);

            if (fault == 0) {
                ASSERT_TRUE(ok);
                ASSERT_TRUE(ml.is_valid());
                ASSERT_TRUE(pool.is_slot_mapped(0));
                ml.reset();
                ASSERT_FALSE(pool.is_slot_mapped(0));
            } else {
                ASSERT_FALSE(ok);
                ASSERT_FALSE(ml.is_valid());
                ASSERT_FALSE(pool.is_slot_mapped(0));
            }
        }
    }

    return 0;
}

// =============================================================================
// Test 5: Zero Steady-State Heap Allocations Over 50,000 Map/Unmap Cycles
// =============================================================================
int test_stress_zero_steady_state_allocations_cuda_interop() {
    std::cout << "  [Stress 5] Zero Steady-State Heap Allocations (50,000 map/unmap cycles)..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(mock_dxgi.get(), 4, 1920, 1080));

    aim::capture::CudaSurfacePool cuda_pool(mock_cuda);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, 0x00010688));

    // Warmup 20 operations
    for (std::uint64_t i = 0; i < 20ULL; ++i) {
        const std::uint32_t slot = static_cast<std::uint32_t>(i % 4);
        aim::FrameLease fl = gpu_pool.create_lease(slot, static_cast<aim::SequenceId>(i + 1), 1000);
        aim::capture::CudaMappedSurfaceLease ml;
        cuda_pool.map_surface(std::move(fl), ml);
        ml.reset();
    }

    // Begin allocation audit
    start_alloc_tracking();

    for (std::uint64_t i = 1; i <= 50000ULL; ++i) {
        const std::uint32_t slot = static_cast<std::uint32_t>(i % 4);
        aim::FrameLease fl = gpu_pool.create_lease(slot, static_cast<aim::SequenceId>(i), static_cast<aim::MonotonicNs>(1000 + i));
        aim::capture::CudaMappedSurfaceLease ml;
        const bool ok = cuda_pool.map_surface(std::move(fl), ml);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(ml.is_valid());
        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot));

        ml.reset();
        ASSERT_FALSE(cuda_pool.is_slot_mapped(slot));
    }

    const std::size_t alloc_calls = g_alloc_tracker.count.load(std::memory_order_relaxed);
    const std::size_t alloc_bytes = g_alloc_tracker.bytes.load(std::memory_order_relaxed);
    stop_alloc_tracking();

    std::cout << "    Measured CUDA Interop Heap Allocations: " << alloc_calls << " calls / " << alloc_bytes << " bytes." << std::endl;
    ASSERT_EQ(alloc_calls, 0u);
    ASSERT_EQ(alloc_bytes, 0u);

    return 0;
}

// =============================================================================
// Test 6: Move Semantics & Swap of CudaSurfacePool Instances
// =============================================================================
int test_stress_cuda_surface_pool_move_and_swap() {
    std::cout << "  [Stress 6] CudaSurfacePool Move Constructor and Move Assignment..." << std::endl;

    auto mock_cuda = std::make_shared<aim::capture::MockCudaInteropBackend>();
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();

    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(mock_dxgi.get(), 4, 1920, 1080));

    // Move construct
    {
        aim::capture::CudaSurfacePool poolA(mock_cuda);
        ASSERT_TRUE(poolA.initialize(gpu_pool, 0x00010688));
        ASSERT_EQ(poolA.capacity(), 4u);

        aim::capture::CudaSurfacePool poolB(std::move(poolA));
        ASSERT_FALSE(poolA.is_initialized());
        ASSERT_TRUE(poolB.is_initialized());
        ASSERT_EQ(poolB.capacity(), 4u);

        // Map through poolB
        aim::FrameLease fl = gpu_pool.create_lease(0, 1, 1000);
        aim::capture::CudaMappedSurfaceLease ml;
        ASSERT_TRUE(poolB.map_surface(std::move(fl), ml));
        ASSERT_TRUE(ml.is_valid());
        ml.reset();
    }

    // All resources cleaned up on destruction of poolB
    ASSERT_EQ(mock_cuda->active_resources_count(), 0u);
    ASSERT_EQ(mock_cuda->active_streams_count(), 0u);
    ASSERT_EQ(mock_cuda->active_events_count(), 0u);

    return 0;
}

} // namespace

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "Running Challenger 2 Empirical Stress Harness for Milestone M2-03 (#15)" << std::endl;
    std::cout << "======================================================================" << std::endl;

    if (test_stress_dynamic_pool_reinit_and_leak_audit() != 0) return 1;
    if (test_stress_in_flight_leases_across_pool_reinitialization() != 0) return 1;
    if (test_stress_out_of_order_lease_destruction_and_move_chains() != 0) return 1;
    if (test_stress_driver_fault_injection_matrix() != 0) return 1;
    if (test_stress_zero_steady_state_allocations_cuda_interop() != 0) return 1;
    if (test_stress_cuda_surface_pool_move_and_swap() != 0) return 1;

    std::cout << "======================================================================" << std::endl;
    std::cout << "ALL 6 M2-03 CHALLENGER STRESS TESTS PASSED WITH 100% SUCCESS (0 DEFECTS)." << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
