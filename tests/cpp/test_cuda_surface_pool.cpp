// tests/cpp/test_cuda_surface_pool.cpp
// Comprehensive Test Suite for CudaSurfacePool and CudaMappedSurfaceLease

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/cuda_surface_pool.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/clock.hpp"

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
// Test 1: Initialization and Pre-Registration
// =============================================================================
int test_initialization_and_pre_registration() {
    std::cout << "  [Test 1] Initialization & Pre-Registration of All Slots..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);

    ASSERT_FALSE(cuda_pool.is_initialized());
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));
    ASSERT_TRUE(cuda_pool.is_initialized());
    ASSERT_EQ(cuda_pool.capacity(), 4u);
    ASSERT_EQ(cuda_pool.cuda_device_id(), 0);
    ASSERT_TRUE(cuda_pool.processing_stream() != nullptr);

    // Verify all 4 slots are pre-registered with CUDA
    for (std::uint32_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(cuda_pool.is_slot_registered(i));
        ASSERT_FALSE(cuda_pool.is_slot_mapped(i));
        ASSERT_TRUE(cuda_pool.get_graphics_resource(i) != nullptr);
        ASSERT_TRUE(cuda_pool.get_ready_event(i) != nullptr);
        ASSERT_TRUE(cuda_pool.get_complete_event(i) != nullptr);
    }

    ASSERT_EQ(cuda_backend->register_count(), 4u);
    ASSERT_EQ(cuda_backend->stream_create_count(), 1u);
    ASSERT_EQ(cuda_backend->event_create_count(), 8u); // 2 events per slot * 4 slots

    return 0;
}

// =============================================================================
// Test 2: Adapter Mismatch Rejection
// =============================================================================
int test_adapter_mismatch_rejection() {
    std::cout << "  [Test 2] Fail-Closed Adapter Mismatch Rejection..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    cuda_backend->set_adapter_match(false); // Simulate mismatch (e.g. Intel iGPU vs RTX 4060)

    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_FALSE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));
    ASSERT_FALSE(cuda_pool.is_initialized());
    ASSERT_EQ(cuda_pool.capacity(), 0u);
    ASSERT_EQ(cuda_backend->register_count(), 0u);

    return 0;
}

// =============================================================================
// Test 3: Zero-Allocation Hot Path Mapping Audit (10,000 Cycles)
// =============================================================================
int test_zero_allocation_hot_path_audit() {
    std::cout << "  [Test 3] Zero-Allocation Hot Path Audit (10,000 cycles)..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));

    // Warmup 100 cycles to ensure any static initialization is settled
    for (std::uint64_t i = 0; i < 100; ++i) {
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        ASSERT_TRUE(slot != aim::capture::GpuSurfacePool::kInvalidSlot);
        const aim::MonotonicNs ts = static_cast<aim::MonotonicNs>(1000ULL * (i + 1));
        aim::FrameLease frame_lease = gpu_pool.create_lease(slot, i + 1, ts);
        aim::capture::CudaMappedSurfaceLease mapped_lease;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease), mapped_lease));
        ASSERT_TRUE(mapped_lease.is_valid());
        mapped_lease.reset();
    }

    // Steady-state 10,000-cycle audit
    start_alloc_tracking();

    for (std::uint64_t i = 0; i < 10000; ++i) {
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        if (slot == aim::capture::GpuSurfacePool::kInvalidSlot) {
            stop_alloc_tracking();
            std::cerr << "Slot acquisition failed at cycle " << i << std::endl;
            return 1;
        }

        const aim::MonotonicNs ts = static_cast<aim::MonotonicNs>(1000ULL * (i + 101));
        aim::FrameLease frame_lease = gpu_pool.create_lease(slot, i + 101, ts);
        aim::capture::CudaMappedSurfaceLease mapped_lease;
        const bool mapped = cuda_pool.map_surface(std::move(frame_lease), mapped_lease);
        if (!mapped || !mapped_lease.is_valid()) {
            stop_alloc_tracking();
            std::cerr << "Mapping failed at cycle " << i << std::endl;
            return 1;
        }

        if (mapped_lease.slot_index() != slot || mapped_lease.frame_id() != i + 101) {
            stop_alloc_tracking();
            std::cerr << "Lease metadata mismatch at cycle " << i << std::endl;
            return 1;
        }

        // Release mapped surface
        mapped_lease.reset();
    }

    stop_alloc_tracking();

    std::cout << "    [Audit Result] 10,000 mapping cycles: "
              << g_alloc_tracker.count.load() << " heap allocations, "
              << g_alloc_tracker.bytes.load() << " bytes" << std::endl;

    ASSERT_EQ(g_alloc_tracker.count.load(), 0u);
    ASSERT_EQ(g_alloc_tracker.bytes.load(), 0u);

    return 0;
}

// =============================================================================
// Test 4: Mapped Lease RAII Lifecycle and Move Semantics
// =============================================================================
int test_mapped_lease_raii_lifecycle() {
    std::cout << "  [Test 4] CudaMappedSurfaceLease RAII Lifecycle & Move Semantics..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));

    // 1. Scope-based destruction
    {
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        aim::FrameLease frame_lease = gpu_pool.create_lease(slot, 42, 1000'000LL);
        aim::capture::CudaMappedSurfaceLease mapped_lease;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease), mapped_lease));
        ASSERT_TRUE(mapped_lease.is_valid());
        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot));
        ASSERT_EQ(gpu_pool.ref_count(slot), 1u);
    }
    // Exited scope: should be automatically unmapped and frame lease released
    ASSERT_FALSE(cuda_pool.is_slot_mapped(0));
    ASSERT_EQ(gpu_pool.ref_count(0), 0u);

    // 2. Move Construction
    {
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        aim::FrameLease frame_lease = gpu_pool.create_lease(slot, 100, 2000'000LL);
        aim::capture::CudaMappedSurfaceLease lease1;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease), lease1));
        ASSERT_TRUE(lease1.is_valid());

        // Move construct lease2 from lease1
        aim::capture::CudaMappedSurfaceLease lease2(std::move(lease1));
        ASSERT_FALSE(lease1.is_valid());
        ASSERT_TRUE(lease2.is_valid());
        ASSERT_EQ(lease2.frame_id(), 100u);
        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot));
    }
    ASSERT_FALSE(cuda_pool.is_slot_mapped(0));

    // 3. Move Assignment
    {
        std::uint32_t slot1 = gpu_pool.acquire_free_slot();
        std::uint32_t slot2 = gpu_pool.acquire_free_slot();

        aim::FrameLease fl1 = gpu_pool.create_lease(slot1, 201, 3000'000LL);
        aim::FrameLease fl2 = gpu_pool.create_lease(slot2, 202, 3000'000LL);

        aim::capture::CudaMappedSurfaceLease l1;
        aim::capture::CudaMappedSurfaceLease l2;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl1), l1));
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl2), l2));

        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot1));
        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot2));

        // Assigning l2 to l1 should unmap slot1 and take over slot2
        l1 = std::move(l2);
        ASSERT_FALSE(cuda_pool.is_slot_mapped(slot1));
        ASSERT_TRUE(cuda_pool.is_slot_mapped(slot2));
        ASSERT_FALSE(l2.is_valid());
        ASSERT_TRUE(l1.is_valid());
        ASSERT_EQ(l1.frame_id(), 202u);
    }
    ASSERT_FALSE(cuda_pool.is_slot_mapped(1));

    return 0;
}

// =============================================================================
// Test 5: Concurrent Multi-Slot Mapping
// =============================================================================
int test_concurrent_multi_slot_mapping() {
    std::cout << "  [Test 5] Concurrent Multi-Slot Mapping..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));

    std::vector<aim::capture::CudaMappedSurfaceLease> leases(4);
    for (std::uint32_t i = 0; i < 4; ++i) {
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        ASSERT_EQ(slot, i);
        const aim::MonotonicNs ts = static_cast<aim::MonotonicNs>(1000ULL * (i + 1));
        aim::FrameLease fl = gpu_pool.create_lease(slot, i + 1, ts);
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl), leases[i]));
        ASSERT_TRUE(leases[i].is_valid());
        ASSERT_TRUE(cuda_pool.is_slot_mapped(i));
    }

    // All 4 slots should be mapped simultaneously with unique surface handles
    for (std::uint32_t i = 0; i < 4; ++i) {
        for (std::uint32_t j = i + 1; j < 4; ++j) {
            ASSERT_TRUE(leases[i].mapped_array() != leases[j].mapped_array());
            ASSERT_TRUE(leases[i].surface_object() != leases[j].surface_object());
        }
    }

    // Resetting one slot frees only that slot
    leases[1].reset();
    ASSERT_FALSE(cuda_pool.is_slot_mapped(1));
    ASSERT_TRUE(cuda_pool.is_slot_mapped(0));
    ASSERT_TRUE(cuda_pool.is_slot_mapped(2));
    ASSERT_TRUE(cuda_pool.is_slot_mapped(3));

    // Can re-map slot 1
    std::uint32_t re_slot = gpu_pool.acquire_free_slot();
    ASSERT_EQ(re_slot, 1u);
    aim::FrameLease fl_re = gpu_pool.create_lease(re_slot, 999, 999000LL);
    ASSERT_TRUE(cuda_pool.map_surface(std::move(fl_re), leases[1]));
    ASSERT_TRUE(cuda_pool.is_slot_mapped(1));

    return 0;
}

// =============================================================================
// Test 6: Map Failure Handling & Fail-Closed Behavior
// =============================================================================
int test_map_failure_handling() {
    std::cout << "  [Test 6] Map Failure Handling & Fail-Closed Behavior..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));

    // 1. Inject mapping error
    cuda_backend->queue_map_result(aim::capture::CudaResult::error_map_failed);

    std::uint32_t slot = gpu_pool.acquire_free_slot();
    aim::FrameLease fl = gpu_pool.create_lease(slot, 1, 1000LL);
    aim::capture::CudaMappedSurfaceLease mapped_lease;

    ASSERT_FALSE(cuda_pool.map_surface(std::move(fl), mapped_lease));
    ASSERT_FALSE(mapped_lease.is_valid());
    ASSERT_FALSE(cuda_pool.is_slot_mapped(slot));

    // 2. Next map succeeds
    std::uint32_t slot2 = gpu_pool.acquire_free_slot();
    aim::FrameLease fl2 = gpu_pool.create_lease(slot2, 2, 2000LL);
    ASSERT_TRUE(cuda_pool.map_surface(std::move(fl2), mapped_lease));
    ASSERT_TRUE(mapped_lease.is_valid());
    ASSERT_TRUE(cuda_pool.is_slot_mapped(slot2));

    return 0;
}

// =============================================================================
// Test 7: Clean Teardown and Reinitialization
// =============================================================================
int test_clean_teardown_and_reinitialization() {
    std::cout << "  [Test 7] Clean Teardown & Reinitialization..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);

    for (int cycle = 0; cycle < 5; ++cycle) {
        ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));
        ASSERT_TRUE(cuda_pool.is_initialized());
        ASSERT_EQ(cuda_backend->active_resources_count(), 4u);
        ASSERT_EQ(cuda_backend->active_streams_count(), 1u);
        ASSERT_EQ(cuda_backend->active_events_count(), 8u);

        // Map and unmap 2 slots
        std::uint32_t slot = gpu_pool.acquire_free_slot();
        aim::FrameLease fl = gpu_pool.create_lease(slot, 1, 1000LL);
        aim::capture::CudaMappedSurfaceLease ml;
        ASSERT_TRUE(cuda_pool.map_surface(std::move(fl), ml));
        ml.reset();

        cuda_pool.release_all();
        ASSERT_FALSE(cuda_pool.is_initialized());
        ASSERT_EQ(cuda_backend->active_resources_count(), 0u);
        ASSERT_EQ(cuda_backend->active_streams_count(), 0u);
        ASSERT_EQ(cuda_backend->active_events_count(), 0u);
    }

    return 0;
}

// =============================================================================
// Test 8: Asynchronous release ownership and source-pool reconfiguration
// =============================================================================
int test_deferred_release_and_reconfiguration() {
    std::cout << "  [Test 8] Deferred Release & Reconfiguration Safety..." << std::endl;

    auto dxgi_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool gpu_pool;
    ASSERT_TRUE(gpu_pool.initialize(dxgi_backend.get(), 4, 1920, 1080));

    auto cuda_backend = std::make_shared<aim::capture::MockCudaInteropBackend>();
    aim::capture::CudaSurfacePool cuda_pool(cuda_backend);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));

    const std::uint32_t slot = gpu_pool.acquire_free_slot();
    aim::FrameLease frame_lease = gpu_pool.create_lease(slot, 77, 77'000LL);
    aim::capture::CudaMappedSurfaceLease mapped_lease;
    ASSERT_TRUE(cuda_pool.map_surface(std::move(frame_lease), mapped_lease));

    cuda_backend->queue_event_query_result(aim::capture::CudaResult::error_not_ready);
    mapped_lease.reset();
    ASSERT_TRUE(cuda_pool.is_slot_release_pending(slot));
    ASSERT_EQ(gpu_pool.ref_count(slot), 1u);

    cuda_pool.reap_completed_releases();
    ASSERT_FALSE(cuda_pool.is_slot_release_pending(slot));
    ASSERT_EQ(gpu_pool.ref_count(slot), 0u);

    const std::uint64_t old_generation = gpu_pool.generation();
    ASSERT_TRUE(gpu_pool.resize(dxgi_backend.get(), 2560, 1440));
    ASSERT_TRUE(gpu_pool.generation() > old_generation);
    ASSERT_FALSE(cuda_pool.is_initialized());
    ASSERT_EQ(cuda_backend->active_resources_count(), 0u);
    ASSERT_TRUE(cuda_pool.initialize(gpu_pool, dxgi_backend->adapter_luid()));
    ASSERT_EQ(cuda_backend->active_resources_count(), 4u);

    return 0;
}

} // namespace

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "Test Suite: CudaSurfacePool & CudaMappedSurfaceLease" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int res = 0;
    res = test_initialization_and_pre_registration();
    if (res != 0) return res;

    res = test_adapter_mismatch_rejection();
    if (res != 0) return res;

    res = test_zero_allocation_hot_path_audit();
    if (res != 0) return res;

    res = test_mapped_lease_raii_lifecycle();
    if (res != 0) return res;

    res = test_concurrent_multi_slot_mapping();
    if (res != 0) return res;

    res = test_map_failure_handling();
    if (res != 0) return res;

    res = test_clean_teardown_and_reinitialization();
    if (res != 0) return res;

    res = test_deferred_release_and_reconfiguration();
    if (res != 0) return res;

    std::cout << "=================================================================" << std::endl;
    std::cout << "ALL 8 CUDA SURFACE POOL TESTS PASSED CLEANLY!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
