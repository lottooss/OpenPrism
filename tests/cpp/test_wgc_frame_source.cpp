// tests/cpp/test_wgc_frame_source.cpp
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/wgc_backend.hpp"
#include "aim/capture/wgc_frame_source.hpp"
#include "aim/core/clock.hpp"

namespace aim {
inline std::ostream& operator<<(std::ostream& os, FrameFormat f) {
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
// Zero-Allocation Probe Setup
// =============================================================================

struct AllocationStats {
    std::atomic<std::size_t> call_count{0};
    std::atomic<std::size_t> bytes_allocated{0};
    bool track_enabled{false};
};

static AllocationStats g_alloc_stats;

void enable_allocation_tracking() noexcept {
    g_alloc_stats.call_count.store(0, std::memory_order_relaxed);
    g_alloc_stats.bytes_allocated.store(0, std::memory_order_relaxed);
    g_alloc_stats.track_enabled = true;
}

void disable_allocation_tracking() noexcept {
    g_alloc_stats.track_enabled = false;
}

} // namespace

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void* operator new(std::size_t size) {
    if (g_alloc_stats.track_enabled) {
        g_alloc_stats.call_count.fetch_add(1, std::memory_order_relaxed);
        g_alloc_stats.bytes_allocated.fetch_add(size, std::memory_order_relaxed);
    }
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}
#pragma GCC diagnostic pop
#endif

namespace {

// =============================================================================
// Test Cases
// =============================================================================

int test_wgc_successful_acquisition_and_timestamping() {
    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

    aim::FrameSourceConfig config{};
    config.display_index = 0;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.pool_capacity = 4;
    config.timeout_ms = 16;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::stopped);
    ASSERT_FALSE(source.health().is_active);

    ASSERT_TRUE(source.start());
    ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
    ASSERT_TRUE(source.health().is_active);

    backend_ptr->queue_acquire_result(S_OK, 100'000ULL, 1920, 1080);

    aim::FrameLease lease;
    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(lease.frame_id(), 1u);
    ASSERT_EQ(lease.width_px(), 1920u);
    ASSERT_EQ(lease.height_px(), 1080u);
    ASSERT_EQ(lease.captured_at_ns(), 10'000'000LL);
    ASSERT_EQ(lease.format(), aim::FrameFormat::b8g8r8a8_unorm);
    ASSERT_TRUE(lease.native_texture_ptr() != nullptr);

    ASSERT_EQ(source.health().total_frames_acquired, 1u);
    ASSERT_EQ(source.health().total_frames_dropped, 0u);
    ASSERT_EQ(backend_ptr->copy_after_release_count(), 0u);

    return 0;
}

int test_wgc_zero_allocation_hot_path() {
    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Warmup phase
    for (int i = 0; i < 10; ++i) {
        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));
    }

    // Steady state audit over 10,000 frames
    enable_allocation_tracking();

    for (int i = 0; i < 10000; ++i) {
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        if (!ok || !lease.is_valid()) {
            disable_allocation_tracking();
            std::cerr << "Failed steady state frame acquisition at index " << i << std::endl;
            return 1;
        }
    }

    disable_allocation_tracking();

    ASSERT_EQ(g_alloc_stats.call_count.load(), 0u);
    ASSERT_EQ(g_alloc_stats.bytes_allocated.load(), 0u);
    ASSERT_EQ(source.health().total_frames_acquired, 10010u);

    return 0;
}

int test_wgc_surface_pool_exhaustion_and_drop() {
    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 2; // Fixed capacity 2
    config.target_width_px = 1920;
    config.target_height_px = 1080;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Acquire and hold lease 1
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_TRUE(lease1.is_valid());

    // Acquire and hold lease 2
    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_TRUE(lease2.is_valid());

    // Pool is fully exhausted (2/2 leased). 3rd acquisition must safely drop!
    aim::FrameLease lease3;
    ASSERT_FALSE(source.try_acquire_latest(lease3));
    ASSERT_FALSE(lease3.is_valid());
    ASSERT_EQ(source.health().total_frames_dropped, 1u);

    // Release lease1 back to pool
    lease1.reset();
    ASSERT_FALSE(lease1.is_valid());

    // Now acquisition succeeds again!
    ASSERT_TRUE(source.try_acquire_latest(lease3));
    ASSERT_TRUE(lease3.is_valid());

    return 0;
}

int test_wgc_dynamic_resolution_resize() {
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

    // First frame: 1080p
    backend_ptr->queue_acquire_result(S_OK, 100'000ULL, 1920, 1080);
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_EQ(lease1.width_px(), 1920u);
    ASSERT_EQ(lease1.height_px(), 1080u);

    // Dynamic resolution switch: 1440p
    backend_ptr->queue_acquire_result(S_OK, 200'000ULL, 2560, 1440);
    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_EQ(lease2.width_px(), 2560u);
    ASSERT_EQ(lease2.height_px(), 1440u);
    ASSERT_EQ(source.surface_pool().width(), 2560u);
    ASSERT_EQ(source.surface_pool().height(), 1440u);

    // Dynamic resolution switch: 4K (3840x2160)
    backend_ptr->queue_acquire_result(S_OK, 300'000ULL, 3840, 2160);
    aim::FrameLease lease3;
    ASSERT_TRUE(source.try_acquire_latest(lease3));
    ASSERT_EQ(lease3.width_px(), 3840u);
    ASSERT_EQ(lease3.height_px(), 2160u);
    ASSERT_EQ(source.surface_pool().width(), 3840u);
    ASSERT_EQ(source.surface_pool().height(), 2160u);

    return 0;
}

int test_wgc_draining_stale_frames() {
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

    // Queue 5 frames in backend
    backend_ptr->queue_acquire_result(S_OK, 100'000ULL, 1920, 1080);
    backend_ptr->queue_acquire_result(S_OK, 200'000ULL, 1920, 1080);
    backend_ptr->queue_acquire_result(S_OK, 300'000ULL, 1920, 1080);
    backend_ptr->queue_acquire_result(S_OK, 400'000ULL, 1920, 1080);
    backend_ptr->queue_acquire_result(S_OK, 500'000ULL, 1920, 1080);

    // Single try_acquire_latest() drains 4 older frames and acquires the 5th
    aim::FrameLease lease;
    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_EQ(lease.captured_at_ns(), 50'000'000LL);
    ASSERT_EQ(source.health().total_frames_dropped, 4u);
    ASSERT_EQ(source.health().total_frames_acquired, 1u);
    ASSERT_EQ(backend_ptr->copy_after_release_count(), 0u);
    ASSERT_FALSE(backend_ptr->is_frame_currently_held());

    return 0;
}

int test_wgc_timeout_retains_latest_texture() {
    aim::capture::MockWgcBackend backend;
    ASSERT_TRUE(backend.initialize({}));
    ASSERT_TRUE(backend.create_capture_session());
    backend.queue_acquire_result(S_OK, 100'000ULL);
    backend.queue_acquire_result(DXGI_ERROR_WAIT_TIMEOUT);
    aim::capture::CapturedRawFrame first{};
    ASSERT_EQ(backend.try_get_next_frame(first), S_OK);
    ASSERT_TRUE(backend.is_frame_currently_held());
    aim::capture::CapturedRawFrame empty{};
    ASSERT_EQ(backend.try_get_next_frame(empty), DXGI_ERROR_WAIT_TIMEOUT);
    ASSERT_TRUE(backend.is_frame_currently_held());
    backend.copy_texture(nullptr, first.raw_texture);
    ASSERT_EQ(backend.copy_after_release_count(), 0u);
    backend.release_frame();
    ASSERT_FALSE(backend.is_frame_currently_held());
    return 0;
}

int test_wgc_bus_ring_publication() {
    auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>();
    auto* backend_ptr = mock_backend.get();
    backend_ptr->set_adapter_luid(0xDEADCAFE);
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::WgcFrameSource source(std::move(mock_backend), clock);
    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> bus_ring;
    source.bind_bus_ring(&bus_ring);
    source.set_pipeline_run_id(777);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    backend_ptr->queue_acquire_result(S_OK, 100'000ULL, 1920, 1080);

    aim::FrameLease lease;
    ASSERT_TRUE(source.try_acquire_latest(lease));

    aim::bus::FrameDescriptor desc{};
    ASSERT_TRUE(bus_ring.try_read_latest(desc));
    ASSERT_EQ(desc.schema_major, 1u);
    ASSERT_EQ(desc.header.sequence_id, 1u);
    ASSERT_EQ(desc.header.pipeline_run_id, 777u);
    ASSERT_EQ(desc.header.source_timestamp_ns, 10'000'000LL);
    ASSERT_EQ(desc.width, 1920u);
    ASSERT_EQ(desc.height, 1080u);
    ASSERT_EQ(desc.adapter_luid, 0xDEADCAFEULL);
    ASSERT_EQ(desc.pool_slot_index, lease.pool_slot_index());

    return 0;
}

int test_wgc_access_loss_and_recovery() {
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

    // Inject access lost
    backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);

    aim::FrameLease lease;
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);
    ASSERT_TRUE(source.health().is_access_lost);
    ASSERT_FALSE(source.health().is_active);
    ASSERT_EQ(source.health().total_access_loss_events, 1u);

    return 0;
}

int test_wgc_clean_shutdown_and_resource_leak_audit() {
    auto external_counter = std::make_shared<std::atomic<std::size_t>>(0);

    {
        auto mock_backend = std::make_unique<aim::capture::MockWgcBackend>(external_counter);
        auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

        aim::capture::WgcFrameSource source(std::move(mock_backend), clock);

        aim::FrameSourceConfig config{};
        config.pool_capacity = 4;
        config.target_width_px = 1920;
        config.target_height_px = 1080;

        ASSERT_TRUE(source.initialize(config));
        ASSERT_EQ(external_counter->load(), 4u);

        ASSERT_TRUE(source.start());

        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));

        source.stop();
        ASSERT_EQ(source.state(), aim::capture::CaptureState::stopped);
    }

    // Source and backend destroyed -> all GPU staging textures freed!
    ASSERT_EQ(external_counter->load(), 0u);

    return 0;
}

} // namespace

int main() {
    std::cout << "Running WGC frame source test suite..." << std::endl;

    if (test_wgc_successful_acquisition_and_timestamping() != 0) return 1;
    if (test_wgc_timeout_retains_latest_texture() != 0) return 1;
    if (test_wgc_zero_allocation_hot_path() != 0) return 1;
    if (test_wgc_surface_pool_exhaustion_and_drop() != 0) return 1;
    if (test_wgc_dynamic_resolution_resize() != 0) return 1;
    if (test_wgc_draining_stale_frames() != 0) return 1;
    if (test_wgc_bus_ring_publication() != 0) return 1;
    if (test_wgc_access_loss_and_recovery() != 0) return 1;
    if (test_wgc_clean_shutdown_and_resource_leak_audit() != 0) return 1;

    std::cout << "All WGC frame source tests PASSED!" << std::endl;
    return 0;
}
