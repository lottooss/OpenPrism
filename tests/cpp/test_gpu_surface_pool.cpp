// tests/cpp/test_gpu_surface_pool.cpp
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/gpu_surface_pool.hpp"

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

int test_pool_initialization() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;

    ASSERT_FALSE(pool.is_initialized());
    ASSERT_EQ(pool.capacity(), 0u);

    // Invalid capacity < 2
    ASSERT_FALSE(pool.initialize(backend.get(), 1, 1920, 1080));
    // Invalid dimensions 0
    ASSERT_FALSE(pool.initialize(backend.get(), 4, 0, 1080));
    ASSERT_FALSE(pool.initialize(backend.get(), 4, 1920, 0));

    // Valid initialization
    ASSERT_TRUE(pool.initialize(backend.get(), 4, 1920, 1080));
    ASSERT_TRUE(pool.is_initialized());
    ASSERT_EQ(pool.capacity(), 4u);
    ASSERT_EQ(pool.width(), 1920u);
    ASSERT_EQ(pool.height(), 1080u);
    ASSERT_EQ(pool.active_lease_count(), 0u);
    ASSERT_EQ(backend->active_staging_texture_count(), 4u);

    return 0;
}

int test_pool_slot_acquisition_and_lease_raii() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;
    ASSERT_TRUE(pool.initialize(backend.get(), 3, 1920, 1080));

    // Acquire Slot 0
    const std::uint32_t s0 = pool.acquire_free_slot();
    ASSERT_EQ(s0, 0u);
    ASSERT_EQ(pool.ref_count(0), 0u);

    {
        aim::FrameLease lease0 = pool.create_lease(s0, 1001, 10'000'000LL);
        ASSERT_TRUE(lease0.is_valid());
        ASSERT_EQ(lease0.frame_id(), 1001u);
        ASSERT_EQ(lease0.captured_at_ns(), 10'000'000LL);
        ASSERT_EQ(lease0.pool_slot_index(), 0u);
        ASSERT_EQ(lease0.width_px(), 1920u);
        ASSERT_EQ(lease0.height_px(), 1080u);
        ASSERT_EQ(pool.ref_count(0), 1u);
        ASSERT_EQ(pool.active_lease_count(), 1u);

        // While Slot 0 is leased, acquire next slot (should be Slot 1)
        const std::uint32_t s1 = pool.acquire_free_slot();
        ASSERT_EQ(s1, 1u);

        aim::FrameLease lease1 = pool.create_lease(s1, 1002, 16'944'444LL);
        ASSERT_TRUE(lease1.is_valid());
        ASSERT_EQ(lease1.pool_slot_index(), 1u);
        ASSERT_EQ(pool.ref_count(1), 1u);
        ASSERT_EQ(pool.active_lease_count(), 2u);

        // Test move semantics
        aim::FrameLease moved_lease1(std::move(lease1));
        ASSERT_TRUE(moved_lease1.is_valid());
        ASSERT_FALSE(lease1.is_valid());
        ASSERT_EQ(pool.ref_count(1), 1u); // Ref count stays 1 across move

        // Test move assignment
        aim::FrameLease moved_assign;
        moved_assign = std::move(moved_lease1);
        ASSERT_TRUE(moved_assign.is_valid());
        ASSERT_FALSE(moved_lease1.is_valid());
        ASSERT_EQ(pool.ref_count(1), 1u);
    }

    // Both leases went out of scope -> ref counts returned to 0
    ASSERT_EQ(pool.ref_count(0), 0u);
    ASSERT_EQ(pool.ref_count(1), 0u);
    ASSERT_EQ(pool.active_lease_count(), 0u);

    return 0;
}

int test_pool_exhaustion_and_contention() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;
    // Capacity 2
    ASSERT_TRUE(pool.initialize(backend.get(), 2, 1920, 1080));

    const std::uint32_t s0 = pool.acquire_free_slot();
    ASSERT_EQ(s0, 0u);
    aim::FrameLease lease0 = pool.create_lease(s0, 1, 1000);

    const std::uint32_t s1 = pool.acquire_free_slot();
    ASSERT_EQ(s1, 1u);
    aim::FrameLease lease1 = pool.create_lease(s1, 2, 2000);

    ASSERT_EQ(pool.active_lease_count(), 2u);

    // All slots are now leased -> acquire_free_slot must return kInvalidSlot
    const std::uint32_t s2 = pool.acquire_free_slot();
    ASSERT_EQ(s2, aim::capture::GpuSurfacePool::kInvalidSlot);

    // Release lease0 explicitly
    lease0.reset();
    ASSERT_FALSE(lease0.is_valid());
    ASSERT_EQ(pool.ref_count(0), 0u);
    ASSERT_EQ(pool.active_lease_count(), 1u);

    // Slot 0 is now available again!
    const std::uint32_t s3 = pool.acquire_free_slot();
    ASSERT_EQ(s3, 0u);

    return 0;
}

int test_pool_stale_lease_release_no_underflow() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;
    ASSERT_TRUE(pool.initialize(backend.get(), 4, 1920, 1080));

    // 1. Direct release_surface on slots with 0 refcount must not underflow
    for (std::uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(pool.ref_count(i), 0u);
        pool.release_surface(i);
        ASSERT_EQ(pool.ref_count(i), 0u);
    }

    // 2. Simulate device reset with outstanding lease
    aim::FrameLease stale_lease;
    {
        const std::uint32_t slot = pool.acquire_free_slot();
        ASSERT_EQ(slot, 0u);
        stale_lease = pool.create_lease(slot, 42, 1'000'000LL);
        ASSERT_EQ(pool.ref_count(0), 1u);

        // Device reset occurs: re-initialize the pool
        ASSERT_TRUE(pool.initialize(backend.get(), 4, 1920, 1080));
        ASSERT_EQ(pool.ref_count(0), 0u);
    }

    // Dropping stale_lease invokes release_surface(0) on re-initialized pool
    stale_lease.reset();
    ASSERT_EQ(pool.ref_count(0), 0u);

    // Slot 0 must remain available and healthy
    const std::uint32_t new_slot = pool.acquire_free_slot();
    ASSERT_EQ(new_slot, 0u);

    return 0;
}

int test_pool_resize() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    aim::capture::GpuSurfacePool pool;
    ASSERT_TRUE(pool.initialize(backend.get(), 3, 1920, 1080));

    ASSERT_EQ(pool.width(), 1920u);
    ASSERT_EQ(pool.height(), 1080u);

    // Resize to 2560x1440
    ASSERT_TRUE(pool.resize(backend.get(), 2560, 1440));
    ASSERT_EQ(pool.width(), 2560u);
    ASSERT_EQ(pool.height(), 1440u);
    ASSERT_EQ(pool.capacity(), 3u);

    // Same size -> no-op returning true
    ASSERT_TRUE(pool.resize(backend.get(), 2560, 1440));

    return 0;
}

int test_pool_clean_teardown() {
    auto backend = std::make_unique<aim::capture::MockDxgiBackend>();
    {
        aim::capture::GpuSurfacePool pool;
        ASSERT_TRUE(pool.initialize(backend.get(), 4, 1920, 1080));
        ASSERT_EQ(backend->active_staging_texture_count(), 4u);
    }
    // Destructor called release_all
    ASSERT_EQ(backend->active_staging_texture_count(), 0u);

    return 0;
}

} // namespace

int main() {
    std::cout << "Running Native C++20 GPU Surface Pool Test Suite..." << std::endl;

    if (test_pool_initialization() != 0) return 1;
    if (test_pool_slot_acquisition_and_lease_raii() != 0) return 1;
    if (test_pool_exhaustion_and_contention() != 0) return 1;
    if (test_pool_stale_lease_release_no_underflow() != 0) return 1;
    if (test_pool_resize() != 0) return 1;
    if (test_pool_clean_teardown() != 0) return 1;

    std::cout << "All GPU Surface Pool tests PASSED successfully." << std::endl;
    return 0;
}
