// tests/cpp/test_dxgi_frame_source.cpp
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <vector>
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
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
// Hook global operator new / delete for allocation audit on GCC/Clang
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

int test_dxgi_successful_acquisition_and_timestamping() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL); // 10 ms start

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);

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

    // Bind Bus Ring
    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);

    // Acquire 10 frames
    for (std::uint64_t i = 1; i <= 10; ++i) {
        const std::uint64_t synthetic_qpc = 100'000ULL + i * 69444ULL;
        backend_ptr->queue_acquire_result(S_OK, synthetic_qpc);

        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));
        ASSERT_TRUE(lease.is_valid());
        ASSERT_EQ(lease.frame_id(), i);
        ASSERT_EQ(lease.width_px(), 1920u);
        ASSERT_EQ(lease.height_px(), 1080u);
        ASSERT_EQ(lease.format(), aim::FrameFormat::b8g8r8a8_unorm);
        ASSERT_TRUE(lease.native_texture_ptr() != nullptr);

        // Verify monotonic timestamp calculation (10 MHz QPF fast-path / qpc_to_ns_128)
        const aim::MonotonicNs expected_ns = source.is_10mhz()
            ? static_cast<aim::MonotonicNs>(synthetic_qpc * 100ULL)
            : aim::qpc_to_ns_128(synthetic_qpc, source.qpf());
        ASSERT_EQ(lease.captured_at_ns(), expected_ns);

        // Verify Bus Ring Publication
        aim::bus::FrameDescriptor desc{};
        ASSERT_TRUE(ring.try_read_latest(desc));
        ASSERT_EQ(desc.frame_id, i);
        ASSERT_EQ(desc.captured_at_ns, expected_ns);
        ASSERT_EQ(desc.width, 1920u);
        ASSERT_EQ(desc.height, 1080u);
        ASSERT_EQ(desc.pool_slot_index, lease.pool_slot_index());
        ASSERT_EQ(desc.adapter_luid, backend_ptr->adapter_luid());
    }

    ASSERT_EQ(source.health().total_frames_acquired, 10u);
    ASSERT_EQ(source.health().total_frames_dropped, 0u);
    ASSERT_EQ(backend_ptr->release_frame_call_count(), 10u);

    return 0;
}

int test_dxgi_zero_allocation_hot_path() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);

    aim::FrameSourceConfig config{};
    config.pool_capacity = 4;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    aim::bus::LatestSpscRing<aim::bus::FrameDescriptor, 16> ring;
    source.bind_bus_ring(&ring);

    // Warmup 10 frames
    for (std::uint64_t i = 0; i < 10; ++i) {
        aim::FrameLease lease;
        source.try_acquire_latest(lease);
    }

    // Enable Zero-Allocation Audit
    enable_allocation_tracking();

    for (std::uint64_t i = 0; i < 10'000; ++i) {
        aim::FrameLease lease;
        const bool ok = source.try_acquire_latest(lease);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(lease.is_valid());

        aim::bus::FrameDescriptor desc{};
        const bool read_ok = ring.try_read_latest(desc);
        ASSERT_TRUE(read_ok);
    }

    const std::size_t alloc_count = g_alloc_stats.call_count.load(std::memory_order_relaxed);
    const std::size_t alloc_bytes = g_alloc_stats.bytes_allocated.load(std::memory_order_relaxed);
    disable_allocation_tracking();

    // Verify 0 calls / 0 bytes allocated in steady state
    ASSERT_EQ(alloc_count, 0u);
    ASSERT_EQ(alloc_bytes, 0u);

    return 0;
}

int test_dxgi_pool_exhaustion_and_drop_handling() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);

    // Pool capacity of 2
    aim::FrameSourceConfig config{};
    config.pool_capacity = 2;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Acquire Lease 1 and hold it
    backend_ptr->queue_acquire_result(S_OK, 100'000);
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_TRUE(lease1.is_valid());

    // Acquire Lease 2 and hold it
    backend_ptr->queue_acquire_result(S_OK, 200'000);
    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_TRUE(lease2.is_valid());

    ASSERT_EQ(source.surface_pool().active_lease_count(), 2u);

    // 3rd frame arrives while both leases are held by downstream consumer
    backend_ptr->queue_acquire_result(S_OK, 300'000);
    aim::FrameLease lease3;
    const bool acquired = source.try_acquire_latest(lease3);

    // Must safely drop frame without overwriting leased textures
    ASSERT_FALSE(acquired);
    ASSERT_FALSE(lease3.is_valid());
    ASSERT_EQ(source.health().total_frames_dropped, 1u);
    ASSERT_EQ(backend_ptr->release_frame_call_count(), 3u); // ReleaseFrame called on dropped frame

    // Release lease1
    lease1.reset();
    ASSERT_EQ(source.surface_pool().active_lease_count(), 1u);

    // 4th frame arrives -> succeeds by acquiring freed Slot 0
    backend_ptr->queue_acquire_result(S_OK, 400'000);
    aim::FrameLease lease4;
    ASSERT_TRUE(source.try_acquire_latest(lease4));
    ASSERT_TRUE(lease4.is_valid());
    ASSERT_EQ(lease4.pool_slot_index(), 0u);

    return 0;
}

int test_dxgi_wait_timeout_cadence_handling() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Inject DXGI_ERROR_WAIT_TIMEOUT
    backend_ptr->queue_acquire_result(DXGI_ERROR_WAIT_TIMEOUT);

    aim::FrameLease lease;
    const bool ok = source.try_acquire_latest(lease);

    ASSERT_FALSE(ok);
    ASSERT_FALSE(lease.is_valid());
    ASSERT_EQ(source.health().total_timeouts, 1u);
    // Timeout is benign: state remains running and access is NOT lost
    ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
    ASSERT_FALSE(source.health().is_access_lost);

    // Subsequent normal frame acquires cleanly
    backend_ptr->queue_acquire_result(S_OK, 100'000);
    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());

    return 0;
}

int test_dxgi_access_loss_and_exponential_backoff_recovery() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Inject DXGI_ERROR_ACCESS_LOST
    backend_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST);

    aim::FrameLease lease;
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::access_lost);
    ASSERT_TRUE(source.health().is_access_lost);
    ASSERT_FALSE(source.health().is_active);
    ASSERT_EQ(source.health().total_access_loss_events, 1u);

    // Immediately calling try_acquire_latest during backoff window (10ms) returns false without retry
    clock->advance_ms(5);
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(backend_ptr->create_duplication_call_count(), 1u); // No new create_duplication attempt yet

    // Advance clock past initial 10 ms backoff -> trigger reinitialization attempt
    clock->advance_ms(6); // Total 11 ms
    // Simulate reinitialization failure (e.g. desktop still locked)
    backend_ptr->set_create_duplication_result(false);
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::backoff_wait);
    ASSERT_EQ(backend_ptr->create_duplication_call_count(), 2u);

    // Backoff doubled to 20 ms. Advance only 10 ms -> still waiting
    clock->advance_ms(10);
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(backend_ptr->create_duplication_call_count(), 2u);

    // Advance remaining 11 ms -> attempt reinitialization with success!
    clock->advance_ms(11);
    backend_ptr->set_create_duplication_result(true);
    backend_ptr->queue_acquire_result(S_OK, 500'000);

    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
    ASSERT_FALSE(source.health().is_access_lost);
    ASSERT_TRUE(source.health().is_active);

    return 0;
}

int test_dxgi_dynamic_resolution_switch() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Acquire 1080p frame
    backend_ptr->queue_acquire_result(S_OK, 100'000);
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_EQ(lease1.width_px(), 1920u);
    ASSERT_EQ(lease1.height_px(), 1080u);

    // Resolution switches to 2560x1440
    backend_ptr->set_output_dimensions(2560, 1440);
    backend_ptr->queue_acquire_result(S_OK, 200'000);

    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_EQ(lease2.width_px(), 2560u);
    ASSERT_EQ(lease2.height_px(), 1440u);
    ASSERT_EQ(source.surface_pool().width(), 2560u);
    ASSERT_EQ(source.surface_pool().height(), 1440u);

    return 0;
}

int test_dxgi_device_removed_recovery() {
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
    aim::FrameSourceConfig config{};
    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());

    // Inject DXGI_ERROR_DEVICE_REMOVED (TDR / driver crash)
    backend_ptr->queue_acquire_result(DXGI_ERROR_DEVICE_REMOVED);

    aim::FrameLease lease;
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_EQ(source.state(), aim::capture::CaptureState::device_lost);
    ASSERT_FALSE(source.health().is_active);

    // Advance clock past backoff and recover device
    clock->advance_ms(15);
    backend_ptr->queue_acquire_result(S_OK, 300'000);

    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(source.state(), aim::capture::CaptureState::running);
    ASSERT_TRUE(source.health().is_active);

    return 0;
}

int test_dxgi_clean_shutdown_and_teardown() {
    auto texture_tracker = std::make_shared<std::atomic<std::size_t>>(0);
    auto mock_backend = std::make_unique<aim::capture::MockDxgiBackend>(texture_tracker);
    auto* backend_ptr = mock_backend.get();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    {
        aim::capture::DxgiFrameSource source(std::move(mock_backend), clock);
        aim::FrameSourceConfig config{};
        ASSERT_TRUE(source.initialize(config));
        ASSERT_TRUE(source.start());

        backend_ptr->queue_acquire_result(S_OK, 100'000);
        aim::FrameLease lease;
        ASSERT_TRUE(source.try_acquire_latest(lease));

        source.stop();
        ASSERT_EQ(source.state(), aim::capture::CaptureState::stopped);
        ASSERT_FALSE(source.health().is_active);
    }

    // After destruction, all staging textures and COM interfaces released (verified via surviving tracker)
    ASSERT_EQ(texture_tracker->load(), 0u);

    return 0;
}

} // namespace

int main() {
    std::cout << "Running Native C++20 DXGI Frame Source Test Suite..." << std::endl;

    if (test_dxgi_successful_acquisition_and_timestamping() != 0) return 1;
    if (test_dxgi_zero_allocation_hot_path() != 0) return 1;
    if (test_dxgi_pool_exhaustion_and_drop_handling() != 0) return 1;
    if (test_dxgi_wait_timeout_cadence_handling() != 0) return 1;
    if (test_dxgi_access_loss_and_exponential_backoff_recovery() != 0) return 1;
    if (test_dxgi_dynamic_resolution_switch() != 0) return 1;
    if (test_dxgi_device_removed_recovery() != 0) return 1;
    if (test_dxgi_clean_shutdown_and_teardown() != 0) return 1;

    std::cout << "All DXGI Frame Source tests PASSED successfully." << std::endl;
    return 0;
}
