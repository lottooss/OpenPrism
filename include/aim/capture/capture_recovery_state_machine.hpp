// include/aim/capture/capture_recovery_state_machine.hpp
#pragma once

#include <cstdint>
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"

namespace aim::capture {

enum class CaptureRecoveryState : std::uint8_t {
    uninitialized       = 0,
    stopped             = 1,
    active_dxgi         = 2,  // DXGI Desktop Duplication active
    active_wgc          = 3,  // WGC fallback active
    access_lost         = 4,  // Access lost, actuation gated
    backoff_wait        = 5,  // Exponential backoff wait, actuation gated
    reinitializing      = 6,  // Attempting reinitialization / fallback
    device_lost         = 7,  // D3D11 device removed / reset
    failed              = 8   // Fatal terminal failure, actuation permanently gated
};

enum class CaptureEventTrigger : std::uint8_t {
    initialize_success  = 0,
    initialize_failure  = 1,
    start_command       = 2,
    stop_command        = 3,
    frame_acquired_ok   = 4,
    frame_timeout       = 5,
    access_lost         = 6,
    device_lost         = 7,
    backoff_expired     = 8,
    reinit_success      = 9,
    reinit_failure      = 10,
    fallback_to_wgc     = 11,
    promote_to_dxgi     = 12,
    dimension_changed   = 13
};

class CaptureRecoveryStateMachine {
public:
    explicit CaptureRecoveryStateMachine(MonotonicNs initial_backoff_ns = 10'000'000LL,
                                         MonotonicNs max_backoff_ns = 250'000'000LL,
                                         std::uint32_t max_consecutive_retries = 3) noexcept;

    [[nodiscard]] CaptureRecoveryState current_state() const noexcept { return current_state_; }
    [[nodiscard]] FrameSourceBackend active_backend() const noexcept { return active_backend_; }
    [[nodiscard]] bool is_actuation_permitted() const noexcept;
    [[nodiscard]] MonotonicNs current_backoff_ns() const noexcept { return current_backoff_ns_; }
    [[nodiscard]] MonotonicNs last_state_change_ns() const noexcept { return last_state_change_ns_; }
    [[nodiscard]] std::uint32_t consecutive_failures() const noexcept { return consecutive_failures_; }
    [[nodiscard]] std::uint32_t max_consecutive_retries() const noexcept { return max_consecutive_retries_; }

    void transition(CaptureEventTrigger trigger, MonotonicNs now_ns) noexcept;
    void reset() noexcept;

private:
    CaptureRecoveryState current_state_{CaptureRecoveryState::uninitialized};
    FrameSourceBackend active_backend_{FrameSourceBackend::dxgi_duplication};
    MonotonicNs initial_backoff_ns_{10'000'000LL};
    MonotonicNs max_backoff_ns_{250'000'000LL};
    MonotonicNs current_backoff_ns_{10'000'000LL};
    MonotonicNs last_state_change_ns_{0};
    std::uint32_t consecutive_failures_{0};
    std::uint32_t max_consecutive_retries_{3};
};

} // namespace aim::capture
