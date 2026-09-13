// include/aim/safety/safety_supervisor.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

#include "aim/core/actuator.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"
#include "aim/interfaces/safety_supervisor.hpp"

namespace aim::safety {

struct SafetySupervisorConfig {
    MonotonicNs max_command_age_ns{10'000'000LL};  // 10 ms hard cutoff for staleness
    std::int32_t max_single_step_counts{250};      // Absolute max counts allowed in one step
    bool require_foreground{true};
    bool enforce_monotonic_sequence{true};
    bool require_valid_calibration{true};
    std::string authorized_reset_token{"authorized_operator_token"};
};

struct SafetySupervisorStats {
    std::uint64_t total_commands_checked{0};
    std::uint64_t total_commands_passed{0};
    std::uint64_t total_commands_rejected{0};
    std::uint64_t emergency_stops_triggered{0};
    SafetyReason last_safety_reason{SafetyReason::none};
};

/// @brief Centralized fail-closed Safety Supervisor (Milestone M6-05).
/// Enforces emergency stop, required focus/calibration callbacks, command
/// sequence/freshness, and maximum count gates before any actuator can dispatch.
/// Source heartbeat/provenance and motion derivative limits are checked by the
/// pipeline/scheduler. A rejecting caller must cancel pending actuator output.
class SafetySupervisor final : public ISafetySupervisor {
public:
    using FocusCheckFn = std::function<bool()>;
    using CalibrationCheckFn = std::function<bool()>;

    explicit SafetySupervisor(const SafetySupervisorConfig& config = {}) noexcept;

    void set_focus_callback(FocusCheckFn fn) noexcept;
    void set_calibration_callback(CalibrationCheckFn fn) noexcept;

    bool check_actuation_safety(const ActuationCommand& command, MonotonicNs now_ns) noexcept override;
    void trigger_emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept override;
    [[nodiscard]] bool is_latched() const noexcept override;
    bool try_reset(const ResetToken& token) noexcept override;

    [[nodiscard]] SafetyReason last_reason() const noexcept;
    [[nodiscard]] SafetySupervisorStats stats() const noexcept;
    void reset_stats() noexcept;

private:
    SafetySupervisorConfig config_{};
    EmergencyStopLatch latch_{};
    FocusCheckFn focus_check_{nullptr};
    CalibrationCheckFn calibration_check_{nullptr};

    std::atomic<std::uint64_t> last_sequence_id_{0};
    std::atomic<SafetyReason> last_reason_{SafetyReason::none};

    std::atomic<std::uint64_t> total_checked_{0};
    std::atomic<std::uint64_t> total_passed_{0};
    std::atomic<std::uint64_t> total_rejected_{0};
    std::atomic<std::uint64_t> e_stops_triggered_{0};
};

} // namespace aim::safety
