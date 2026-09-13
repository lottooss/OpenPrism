// src/capture/capture_recovery_state_machine.cpp
#include "aim/capture/capture_recovery_state_machine.hpp"
#include <algorithm>

namespace aim::capture {

CaptureRecoveryStateMachine::CaptureRecoveryStateMachine(MonotonicNs initial_backoff_ns,
                                                         MonotonicNs max_backoff_ns,
                                                         std::uint32_t max_consecutive_retries) noexcept
    : initial_backoff_ns_(initial_backoff_ns),
      max_backoff_ns_(max_backoff_ns),
      current_backoff_ns_(initial_backoff_ns),
      max_consecutive_retries_(max_consecutive_retries) {}

bool CaptureRecoveryStateMachine::is_actuation_permitted() const noexcept {
    return (current_state_ == CaptureRecoveryState::active_dxgi ||
            current_state_ == CaptureRecoveryState::active_wgc);
}

void CaptureRecoveryStateMachine::reset() noexcept {
    current_state_ = CaptureRecoveryState::uninitialized;
    active_backend_ = FrameSourceBackend::dxgi_duplication;
    current_backoff_ns_ = initial_backoff_ns_;
    last_state_change_ns_ = 0;
    consecutive_failures_ = 0;
}

void CaptureRecoveryStateMachine::transition(CaptureEventTrigger trigger, MonotonicNs now_ns) noexcept {
    last_state_change_ns_ = now_ns;

    switch (trigger) {
    case CaptureEventTrigger::initialize_success:
        current_state_ = CaptureRecoveryState::stopped;
        consecutive_failures_ = 0;
        current_backoff_ns_ = initial_backoff_ns_;
        break;

    case CaptureEventTrigger::initialize_failure:
        ++consecutive_failures_;
        if (consecutive_failures_ >= max_consecutive_retries_) {
            current_state_ = CaptureRecoveryState::failed;
        } else {
            current_state_ = CaptureRecoveryState::backoff_wait;
            current_backoff_ns_ = std::min(current_backoff_ns_ * 2, max_backoff_ns_);
        }
        break;

    case CaptureEventTrigger::start_command:
        if (current_state_ == CaptureRecoveryState::stopped ||
            current_state_ == CaptureRecoveryState::uninitialized) {
            current_state_ = (active_backend_ == FrameSourceBackend::dxgi_duplication)
                                 ? CaptureRecoveryState::active_dxgi
                                 : CaptureRecoveryState::active_wgc;
            consecutive_failures_ = 0;
            current_backoff_ns_ = initial_backoff_ns_;
        }
        break;

    case CaptureEventTrigger::stop_command:
        current_state_ = CaptureRecoveryState::stopped;
        break;

    case CaptureEventTrigger::frame_acquired_ok:
        current_state_ = (active_backend_ == FrameSourceBackend::dxgi_duplication)
                             ? CaptureRecoveryState::active_dxgi
                             : CaptureRecoveryState::active_wgc;
        consecutive_failures_ = 0;
        current_backoff_ns_ = initial_backoff_ns_;
        break;

    case CaptureEventTrigger::frame_timeout:
        // Transient timeout does not automatically change state, but keeps active state
        break;

    case CaptureEventTrigger::access_lost:
        ++consecutive_failures_;
        current_state_ = CaptureRecoveryState::access_lost;
        current_backoff_ns_ = std::min(current_backoff_ns_ * 2, max_backoff_ns_);
        break;

    case CaptureEventTrigger::device_lost:
        ++consecutive_failures_;
        current_state_ = CaptureRecoveryState::device_lost;
        current_backoff_ns_ = std::min(current_backoff_ns_ * 2, max_backoff_ns_);
        break;

    case CaptureEventTrigger::backoff_expired:
        current_state_ = CaptureRecoveryState::reinitializing;
        break;

    case CaptureEventTrigger::reinit_success:
        current_state_ = (active_backend_ == FrameSourceBackend::dxgi_duplication)
                             ? CaptureRecoveryState::active_dxgi
                             : CaptureRecoveryState::active_wgc;
        consecutive_failures_ = 0;
        current_backoff_ns_ = initial_backoff_ns_;
        break;

    case CaptureEventTrigger::reinit_failure:
        ++consecutive_failures_;
        if (consecutive_failures_ >= max_consecutive_retries_) {
            current_state_ = CaptureRecoveryState::failed;
        } else {
            current_state_ = CaptureRecoveryState::backoff_wait;
            current_backoff_ns_ = std::min(current_backoff_ns_ * 2, max_backoff_ns_);
        }
        break;

    case CaptureEventTrigger::fallback_to_wgc:
        active_backend_ = FrameSourceBackend::windows_graphics_capture;
        current_state_ = CaptureRecoveryState::active_wgc;
        consecutive_failures_ = 0;
        current_backoff_ns_ = initial_backoff_ns_;
        break;

    case CaptureEventTrigger::promote_to_dxgi:
        active_backend_ = FrameSourceBackend::dxgi_duplication;
        current_state_ = CaptureRecoveryState::active_dxgi;
        consecutive_failures_ = 0;
        current_backoff_ns_ = initial_backoff_ns_;
        break;

    case CaptureEventTrigger::dimension_changed:
        // Dimensions changed, stays in active capture state
        break;
    }
}

} // namespace aim::capture
