// tests/cpp/test_capture_recovery.cpp
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/capture_recovery_state_machine.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
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

int test_state_machine_exponential_backoff_and_gating() {
    aim::capture::CaptureRecoveryStateMachine sm(10'000'000LL, 250'000'000LL, 3);

    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::uninitialized);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Start -> active_dxgi
    sm.transition(aim::capture::CaptureEventTrigger::start_command, 10'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::active_dxgi);
    ASSERT_TRUE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.current_backoff_ns(), 10'000'000LL);

    // 1st Access lost -> backoff 20ms, actuation gated!
    sm.transition(aim::capture::CaptureEventTrigger::access_lost, 20'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::access_lost);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.current_backoff_ns(), 20'000'000LL);
    ASSERT_EQ(sm.consecutive_failures(), 1u);

    // Backoff expired -> reinitializing
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, 40'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::reinitializing);
    ASSERT_FALSE(sm.is_actuation_permitted());

    // Reinit failure 1 -> backoff_wait, backoff doubles to 40ms
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, 45'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::backoff_wait);
    ASSERT_FALSE(sm.is_actuation_permitted());
    ASSERT_EQ(sm.current_backoff_ns(), 40'000'000LL);
    ASSERT_EQ(sm.consecutive_failures(), 2u);

    // Backoff expired -> reinitializing
    sm.transition(aim::capture::CaptureEventTrigger::backoff_expired, 85'000'000LL);

    // Reinit failure 2 (consecutive failures = 3 >= max_consecutive_retries) -> transitions to FAILED!
    sm.transition(aim::capture::CaptureEventTrigger::reinit_failure, 90'000'000LL);
    ASSERT_EQ(sm.current_state(), aim::capture::CaptureRecoveryState::failed);
    ASSERT_FALSE(sm.is_actuation_permitted());

    return 0;
}

int test_recovery_dxgi_primary_selection() {
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
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

    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_dxgi);
    ASSERT_TRUE(source.is_actuation_permitted());

    aim::FrameLease lease;
    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(lease.width_px(), 1920u);
    ASSERT_EQ(lease.height_px(), 1080u);

    return 0;
}

int test_recovery_fallback_to_wgc_on_unsupported_dxgi() {
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    mock_dxgi->set_init_result(false); // DXGI fails to initialize (e.g. Hybrid GPU / Unsupported)

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
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

    // Automatically switched to WGC fallback!
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_wgc);
    ASSERT_TRUE(source.is_actuation_permitted());

    aim::FrameLease lease;
    ASSERT_TRUE(source.try_acquire_latest(lease));
    ASSERT_TRUE(lease.is_valid());

    return 0;
}

int test_recovery_dxgi_access_loss_and_wgc_fallback_switch() {
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
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
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);

    // Initial frame on DXGI
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_TRUE(lease1.is_valid());

    // Inject DXGI ACCESS LOST (e.g. UAC secure desktop or fullscreen switch)
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);

    // try_acquire_latest() detects access loss and falls back to WGC immediately!
    aim::FrameLease lease2;
    const bool ok = source.try_acquire_latest(lease2);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(lease2.is_valid());
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_wgc);
    ASSERT_TRUE(source.is_actuation_permitted());

    return 0;
}

int test_recovery_wgc_to_dxgi_promotion() {
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    auto* dxgi_ptr = mock_dxgi.get();

    // Start with DXGI failing so we begin in WGC fallback
    dxgi_ptr->set_init_result(false);

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);
    source.set_dxgi_probe_interval_ns(1'000'000'000LL); // 1.0s probe interval

    aim::FrameSourceConfig config{};
    config.backend = aim::FrameSourceBackend::dxgi_duplication;
    config.pool_capacity = 4;
    config.target_width_px = 1920;
    config.target_height_px = 1080;
    config.fallback_to_wgc = true;

    ASSERT_TRUE(source.initialize(config));
    ASSERT_TRUE(source.start());
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    // Acquire frame on WGC
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    // DXGI becomes available again!
    dxgi_ptr->set_init_result(true);

    // Advance clock by 2.0 seconds past probe interval
    clock->advance_ns(2'000'000'000LL);

    // Next acquisition triggers promotion probe and seamlessly switches back to DXGI!
    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::dxgi_duplication);
    ASSERT_EQ(source.recovery_state(), aim::capture::CaptureRecoveryState::active_dxgi);
    ASSERT_TRUE(source.is_actuation_permitted());

    return 0;
}

int test_recovery_all_backends_fail_closed() {
    auto mock_dxgi = std::make_unique<aim::capture::MockDxgiBackend>();
    mock_dxgi->set_init_result(false);

    auto mock_wgc = std::make_unique<aim::capture::MockWgcBackend>();
    mock_wgc->set_init_result(false);

    auto clock = std::make_shared<aim::FakeClock>(10'000'000LL);

    aim::capture::UnifiedCaptureSource source(std::move(mock_dxgi), std::move(mock_wgc), clock);

    aim::FrameSourceConfig config{};
    config.backend = aim::FrameSourceBackend::dxgi_duplication;
    config.fallback_to_wgc = true;

    // Both backends failed -> initialization fails closed!
    ASSERT_FALSE(source.initialize(config));
    ASSERT_FALSE(source.is_actuation_permitted());

    aim::FrameLease lease;
    ASSERT_FALSE(source.try_acquire_latest(lease));
    ASSERT_FALSE(source.is_actuation_permitted());

    return 0;
}

int test_recovery_dynamic_resolution_across_fallback() {
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

    // 1st Frame on DXGI: 1080p
    dxgi_ptr->queue_acquire_result(S_OK, 100'000ULL);
    aim::FrameLease lease1;
    ASSERT_TRUE(source.try_acquire_latest(lease1));
    ASSERT_EQ(lease1.width_px(), 1920u);
    ASSERT_EQ(lease1.height_px(), 1080u);

    // DXGI fails with access loss and WGC provides 1440p frame
    dxgi_ptr->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0);
    wgc_ptr->queue_acquire_result(S_OK, 200'000ULL, 2560, 1440);

    aim::FrameLease lease2;
    ASSERT_TRUE(source.try_acquire_latest(lease2));
    ASSERT_EQ(lease2.width_px(), 2560u);
    ASSERT_EQ(lease2.height_px(), 1440u);
    ASSERT_EQ(source.active_backend(), aim::FrameSourceBackend::windows_graphics_capture);

    return 0;
}

} // namespace

int main() {
    std::cout << "Running capture recovery and state machine test suite..." << std::endl;

    if (test_state_machine_exponential_backoff_and_gating() != 0) return 1;
    if (test_recovery_dxgi_primary_selection() != 0) return 1;
    if (test_recovery_fallback_to_wgc_on_unsupported_dxgi() != 0) return 1;
    if (test_recovery_dxgi_access_loss_and_wgc_fallback_switch() != 0) return 1;
    if (test_recovery_wgc_to_dxgi_promotion() != 0) return 1;
    if (test_recovery_all_backends_fail_closed() != 0) return 1;
    if (test_recovery_dynamic_resolution_across_fallback() != 0) return 1;

    std::cout << "All capture recovery tests PASSED!" << std::endl;
    return 0;
}
