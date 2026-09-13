// include/aim/core/actuator.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

enum class MouseButton : std::uint8_t {
    none = 0,
    left = 1,
    right = 2,
    middle = 3,
    extra1 = 4,
    extra2 = 5
};

enum class ButtonAction : std::uint8_t {
    none = 0,
    press = 1,
    release = 2,
    click = 3
};

struct ButtonTransition {
    MouseButton button{MouseButton::none};
    ButtonAction action{ButtonAction::none};
};

struct ActuationCommand {
    SequenceId sequence_id{0};
    CorrelationId correlation_id{};
    MonotonicNs generated_at_ns{0};
    MonotonicNs desired_apply_time_ns{0};
    std::int32_t delta_x_counts{0};
    std::int32_t delta_y_counts{0};
    ButtonTransition button_transition{};
};

enum class SubmitResult : std::uint8_t {
    submitted = 0,
    rejected_latched = 1,
    rejected_stale = 2,
    rejected_uninitialized = 3,
    rejected_foreground = 4,
    rejected_dispatch_failed = 5
};

struct ActuatorConfig {
    std::string backend{"null"};
    std::uint32_t scheduler_hz{1000};
    bool relative_counts{true};
    bool cancel_superseded{true};
    bool require_emergency_stop{true};
};

struct ActuatorHealth {
    bool is_active{false};
    bool is_latched{false};
    std::uint64_t total_commands_submitted{0};
    std::uint64_t total_commands_rejected{0};
    std::uint64_t total_commands_cancelled{0};
    MonotonicNs last_dispatch_ns{0};
    std::uint32_t pressed_buttons_mask{0};
};

/// @brief Primary Actuator interface.
class IActuator {
public:
    virtual ~IActuator() = default;
    virtual bool initialize(const ActuatorConfig& config) noexcept = 0;
    virtual bool start() noexcept = 0;
    virtual SubmitResult submit_latest(const ActuationCommand& command) noexcept = 0;
    virtual void cancel_pending() noexcept = 0;
    virtual void emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept = 0;
    virtual bool reset_emergency_stop(const ResetToken& token) noexcept = 0;
    virtual void shutdown() noexcept = 0;
    [[nodiscard]] virtual ActuatorHealth health() const noexcept = 0;
};

/// @brief Non-actuating NullActuator recording commands without OS input.
/// Provides deterministic execution and latched emergency-stop testing.
class NullActuator final : public IActuator {
public:
    static constexpr std::size_t kMaxRecordedCommands = 1024;

    NullActuator() noexcept = default;
    ~NullActuator() noexcept override {
        shutdown();
    }

    bool initialize(const ActuatorConfig& config) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
        is_initialized_ = true;
        return true;
    }

    bool start() noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_initialized_) return false;
        if (latch_.is_latched()) return false;
        is_active_ = true;
        return true;
    }

    SubmitResult submit_latest(const ActuationCommand& command) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_initialized_ || !is_active_) {
            ++total_rejected_;
            return SubmitResult::rejected_uninitialized;
        }
        if (latch_.is_latched()) {
            ++total_rejected_;
            return SubmitResult::rejected_latched;
        }

        // Apply simulated button state safely without UB shift
        const auto& btn = command.button_transition;
        const auto btn_val = static_cast<std::uint32_t>(btn.button);
        if (btn.button != MouseButton::none && btn_val < 32u) {
            const std::uint32_t bit = 1u << btn_val;
            if (btn.action == ButtonAction::press || btn.action == ButtonAction::click) {
                pressed_buttons_mask_ |= bit;
            }
            if (btn.action == ButtonAction::release || btn.action == ButtonAction::click) {
                pressed_buttons_mask_ &= ~bit;
            }
        }

        pending_command_ = command;
        if (recorded_count_ < kMaxRecordedCommands) {
            recorded_commands_[recorded_count_++] = command;
        } else {
            // Saturated ring overwrite
            recorded_commands_[(recorded_count_++) % kMaxRecordedCommands] = command;
        }
        last_dispatch_ns_ = command.desired_apply_time_ns > 0 ? command.desired_apply_time_ns : command.generated_at_ns;
        ++total_submitted_;
        return SubmitResult::submitted;
    }

    void cancel_pending() noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_command_.has_value()) {
            pending_command_.reset();
            ++total_cancelled_;
        }
        // Match the real actuator contract: cancellation cannot leave a
        // logically held button behind.
        pressed_buttons_mask_ = 0;
    }

    void emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        latch_.trigger(reason);
        if (pending_command_.has_value()) {
            pending_command_.reset();
            ++total_cancelled_;
        }
        // Fail-safe: release all simulated pressed buttons immediately
        pressed_buttons_mask_ = 0;
    }

    bool reset_emergency_stop(const ResetToken& token) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        return latch_.try_reset(token);
    }

    void shutdown() noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        is_active_ = false;
        pending_command_.reset();
        // Fail-safe: release all button states on shutdown
        pressed_buttons_mask_ = 0;
    }

    [[nodiscard]] ActuatorHealth health() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        return ActuatorHealth{
            .is_active = is_active_,
            .is_latched = latch_.is_latched(),
            .total_commands_submitted = total_submitted_,
            .total_commands_rejected = total_rejected_,
            .total_commands_cancelled = total_cancelled_,
            .last_dispatch_ns = last_dispatch_ns_,
            .pressed_buttons_mask = pressed_buttons_mask_
        };
    }

    [[nodiscard]] std::vector<ActuationCommand> recorded_commands() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t valid_count = (recorded_count_ < kMaxRecordedCommands) ? recorded_count_ : kMaxRecordedCommands;
        return std::vector<ActuationCommand>(recorded_commands_.begin(), recorded_commands_.begin() + valid_count);
    }

    void clear_recorded_commands() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        recorded_count_ = 0;
    }

private:
    mutable std::mutex mutex_;
    ActuatorConfig config_{};
    EmergencyStopLatch latch_{};
    bool is_initialized_{false};
    bool is_active_{false};
    std::optional<ActuationCommand> pending_command_{std::nullopt};
    std::array<ActuationCommand, kMaxRecordedCommands> recorded_commands_{};
    std::size_t recorded_count_{0};
    std::uint64_t total_submitted_{0};
    std::uint64_t total_rejected_{0};
    std::uint64_t total_cancelled_{0};
    MonotonicNs last_dispatch_ns_{0};
    std::uint32_t pressed_buttons_mask_{0};
};

} // namespace aim
