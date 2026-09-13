// src/safety/safety_supervisor.cpp
#include "aim/safety/safety_supervisor.hpp"

namespace aim::safety {

SafetySupervisor::SafetySupervisor(const SafetySupervisorConfig& config) noexcept
    : config_(config) {}

void SafetySupervisor::set_focus_callback(FocusCheckFn fn) noexcept {
    focus_check_ = std::move(fn);
}

void SafetySupervisor::set_calibration_callback(CalibrationCheckFn fn) noexcept {
    calibration_check_ = std::move(fn);
}

bool SafetySupervisor::check_actuation_safety(const ActuationCommand& command, MonotonicNs now_ns) noexcept {
    total_checked_.fetch_add(1, std::memory_order_relaxed);

    // Gate 1: Check emergency stop latch
    if (latch_.is_latched()) {
        last_reason_.store(latch_.reason(), std::memory_order_release);
        total_rejected_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Gate 2: Foreground focus verification
    if (config_.require_foreground) {
        if (!focus_check_ || !focus_check_()) {
            trigger_emergency_stop(SafetyReason::lost_focus);
            total_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    // Gate 3: Calibration validity check
    if (config_.require_valid_calibration) {
        if (!calibration_check_ || !calibration_check_()) {
            trigger_emergency_stop(SafetyReason::invalid_calibration);
            total_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    // Gate 4: Monotonic sequence enforcement
    if (command.sequence_id == 0) {
        last_reason_.store(SafetyReason::stale_data, std::memory_order_release);
        total_rejected_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Gate 5: Freshness deadline check
    // Positive ordered timestamps make subtraction representable near INT64_MAX.
    // Configuration cannot relax the architecture's hard 10 ms stale cutoff.
    if (config_.max_command_age_ns <= 0 || config_.max_command_age_ns > 10'000'000LL ||
        command.generated_at_ns <= 0 || now_ns <= 0 || command.generated_at_ns > now_ns ||
        now_ns - command.generated_at_ns > config_.max_command_age_ns) {
        last_reason_.store(SafetyReason::stale_data, std::memory_order_release);
        total_rejected_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Gate 6: Count bounds and limit checking
    const auto limit = static_cast<std::int64_t>(config_.max_single_step_counts);
    if (limit <= 0 || static_cast<std::int64_t>(command.delta_x_counts) > limit ||
        static_cast<std::int64_t>(command.delta_x_counts) < -limit ||
        static_cast<std::int64_t>(command.delta_y_counts) > limit ||
        static_cast<std::int64_t>(command.delta_y_counts) < -limit) {
        trigger_emergency_stop(SafetyReason::out_of_bounds);
        total_rejected_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Rejecting malformed input must not consume its sequence and block a later
    // valid command. CAS also prevents concurrent duplicate-sequence acceptance.
    if (config_.enforce_monotonic_sequence) {
        auto previous = last_sequence_id_.load(std::memory_order_acquire);
        if (command.sequence_id <= previous || !last_sequence_id_.compare_exchange_strong(
                previous, command.sequence_id, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // One bounded attempt: contention is rejected, never waited out in
            // the warmed dispatch path.
            last_reason_.store(SafetyReason::stale_data, std::memory_order_release);
            total_rejected_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    last_reason_.store(SafetyReason::none, std::memory_order_release);
    total_passed_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void SafetySupervisor::trigger_emergency_stop(SafetyReason reason) noexcept {
    latch_.trigger(reason);
    last_reason_.store(reason, std::memory_order_release);
    e_stops_triggered_.fetch_add(1, std::memory_order_relaxed);
}

bool SafetySupervisor::is_latched() const noexcept {
    return latch_.is_latched();
}

bool SafetySupervisor::try_reset(const ResetToken& token) noexcept {
    if (!token.is_valid || token.token_id != config_.authorized_reset_token) {
        return false;
    }
    if (latch_.try_reset(token)) {
        last_reason_.store(SafetyReason::none, std::memory_order_release);
        return true;
    }
    return false;
}

SafetyReason SafetySupervisor::last_reason() const noexcept {
    return last_reason_.load(std::memory_order_acquire);
}

SafetySupervisorStats SafetySupervisor::stats() const noexcept {
    return SafetySupervisorStats{
        .total_commands_checked = total_checked_.load(std::memory_order_relaxed),
        .total_commands_passed = total_passed_.load(std::memory_order_relaxed),
        .total_commands_rejected = total_rejected_.load(std::memory_order_relaxed),
        .emergency_stops_triggered = e_stops_triggered_.load(std::memory_order_relaxed),
        .last_safety_reason = last_reason_.load(std::memory_order_relaxed)
    };
}

void SafetySupervisor::reset_stats() noexcept {
    total_checked_.store(0, std::memory_order_relaxed);
    total_passed_.store(0, std::memory_order_relaxed);
    total_rejected_.store(0, std::memory_order_relaxed);
    e_stops_triggered_.store(0, std::memory_order_relaxed);
    last_reason_.store(SafetyReason::none, std::memory_order_relaxed);
}

} // namespace aim::safety
