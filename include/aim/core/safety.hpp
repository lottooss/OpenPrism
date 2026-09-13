// include/aim/core/safety.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

enum class SafetyReason : std::uint32_t {
    none = 0,
    emergency_stop_triggered = 1,
    stale_data = 2,
    lost_focus = 3,
    invalid_calibration = 4,
    out_of_bounds = 5,
    driver_error = 6,
    operator_requested = 7
};

[[nodiscard]] constexpr std::string_view safety_reason_to_string(SafetyReason reason) noexcept {
    switch (reason) {
        case SafetyReason::none: return "none";
        case SafetyReason::emergency_stop_triggered: return "emergency_stop_triggered";
        case SafetyReason::stale_data: return "stale_data";
        case SafetyReason::lost_focus: return "lost_focus";
        case SafetyReason::invalid_calibration: return "invalid_calibration";
        case SafetyReason::out_of_bounds: return "out_of_bounds";
        case SafetyReason::driver_error: return "driver_error";
        case SafetyReason::operator_requested: return "operator_requested";
    }
    return "unknown";
}

struct ResetToken {
    std::string token_id{};
    MonotonicNs generated_at_ns{0};
    bool is_valid{false};
};

/// @brief Thread-safe latched emergency-stop state machine.
/// Once triggered, remains latched until explicitly reset with an approved valid token.
class EmergencyStopLatch {
public:
    constexpr EmergencyStopLatch() noexcept = default;

    /// @brief Trigger and lock the emergency stop latch.
    void trigger(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept {
        const SafetyReason effective_reason = (reason != SafetyReason::none) ? reason : SafetyReason::emergency_stop_triggered;
        reason_.store(effective_reason, std::memory_order_release);
    }

    /// @brief Check if the latch is currently engaged.
    [[nodiscard]] bool is_latched() const noexcept {
        return reason_.load(std::memory_order_acquire) != SafetyReason::none;
    }

    /// @brief Query the reason the latch was engaged.
    [[nodiscard]] SafetyReason reason() const noexcept {
        return reason_.load(std::memory_order_acquire);
    }

    /// @brief Attempt to reset the latch with an explicit valid reset token.
    [[nodiscard]] bool try_reset(const ResetToken& token) noexcept {
        if (!token.is_valid || token.token_id.empty()) {
            return false;
        }
        reason_.store(SafetyReason::none, std::memory_order_release);
        return true;
    }

private:
    std::atomic<SafetyReason> reason_{SafetyReason::none};
};

} // namespace aim
