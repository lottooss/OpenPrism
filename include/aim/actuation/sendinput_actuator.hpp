// include/aim/actuation/sendinput_actuator.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "aim/core/actuator.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::actuation {

struct SendInputConfig {
    ActuatorConfig base_config{};
    MonotonicNs max_tolerated_lag_ns{10'000'000LL}; // 10 ms hard stale cutoff
    std::string target_window_title{};
    // Optional exact foreground pin. When either value is supplied, the
    // production backend requires both the same HWND and process ID on every
    // foreground check. Zero values preserve title-only compatibility for
    // injected/generic backends that do not have a window identity.
    std::uintptr_t target_window_handle{0};
    std::uint32_t target_process_id{0};
    bool verify_foreground{true};
    bool enforce_monotonic_sequence{true};
    float mouse_speed_multiplier{1.0f};
};

struct SendInputStats {
    std::uint64_t total_dispatches{0};
    std::uint64_t stale_dropped{0};
    std::uint64_t out_of_order_dropped{0};
    std::uint64_t foreground_rejected{0};
    std::uint64_t dispatch_failures{0};
    std::uint64_t partial_dispatch_failures{0};
    std::uint64_t release_failures{0};
    std::uint64_t button_presses{0};
    std::uint64_t button_releases{0};
    std::int64_t cumulative_counts_x{0};
    std::int64_t cumulative_counts_y{0};
    std::uint32_t last_platform_error{0};
    std::uint64_t contention_rejections{0};
};

struct InputDispatchResult {
    std::uint32_t requested_events{0};
    std::uint32_t dispatched_events{0};
    std::uint32_t platform_error{0};

    [[nodiscard]] constexpr bool complete() const noexcept {
        return requested_events == dispatched_events;
    }

    [[nodiscard]] constexpr bool partial() const noexcept {
        return dispatched_events > 0 && dispatched_events < requested_events;
    }
};

/// OS boundary used by the production Win32 backend and deterministic tests.
/// Injected backends are non-owning and must outlive the actuator.
class ISendInputBackend {
public:
    virtual ~ISendInputBackend() = default;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual bool foreground_matches(std::string_view expected_title) const noexcept = 0;
    // Backward-compatible extension for exact OS window/process pinning.
    // Generic or non-window test backends retain the old title-only behavior
    // when no identity is requested; a requested identity fails closed unless
    // the backend implements this check.
    [[nodiscard]] virtual bool foreground_identity_matches(
        std::string_view expected_title, std::uintptr_t expected_window_handle,
        std::uint32_t expected_process_id) const noexcept {
        return expected_window_handle == 0 && expected_process_id == 0 &&
               foreground_matches(expected_title);
    }
    [[nodiscard]] virtual InputDispatchResult dispatch(const ActuationCommand& command) noexcept = 0;
    [[nodiscard]] virtual InputDispatchResult release_buttons(std::uint32_t pressed_buttons_mask) noexcept = 0;
};

/// @brief Production SendInput relative count & button actuator (Milestone M6-01).
/// Guarantees fail-closed safety, foreground verification, deadline checking,
/// monotonic sequence validation, and paired button state release.
/// Hot operations never wait for another caller. A concurrent stop invalidates
/// submissions immediately; an already-entered OS call is released by its owner
/// before that call returns. Dependencies/destruction require quiescent callers.
class SendInputActuator final : public IActuator {
public:
    static constexpr std::size_t kMaxRecordedDispatches = 1024;

    SendInputActuator() noexcept;
    explicit SendInputActuator(ISendInputBackend* backend) noexcept;
    ~SendInputActuator() noexcept override {
        shutdown();
    }

    bool initialize(const ActuatorConfig& config) noexcept override;
    bool initialize(const SendInputConfig& config) noexcept;
    bool start() noexcept override;
    SubmitResult submit_latest(const ActuationCommand& command) noexcept override;
    void cancel_pending() noexcept override;
    void emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept override;
    bool reset_emergency_stop(const ResetToken& token) noexcept override;
    void shutdown() noexcept override;
    [[nodiscard]] ActuatorHealth health() const noexcept override;
    [[nodiscard]] SendInputStats stats() const noexcept;

    /// @brief Check if target foreground window matches configured expectations
    [[nodiscard]] bool check_foreground_window() const noexcept;

    /// @brief Get history of recorded dispatches (for deterministic test validation)
    [[nodiscard]] std::vector<ActuationCommand> recorded_dispatches() const;
    void clear_recorded_dispatches() noexcept;

private:
    static constexpr std::uint32_t kOwned = 1u;
    static constexpr std::uint32_t kCancelRequested = 2u;
    struct OwnershipScope {
        SendInputActuator& owner;
        ~OwnershipScope() { owner.release_ownership(); }
    };
    [[nodiscard]] bool try_ownership() const noexcept;
    void release_ownership() noexcept;
    void cancel_owned() noexcept;
    [[nodiscard]] bool check_foreground_window_unsafe() const noexcept;
    [[nodiscard]] bool release_all_held_buttons_unsafe() noexcept;
    void conservatively_track_partial_button_dispatch(const ActuationCommand& command) noexcept;

    mutable std::atomic<std::uint32_t> ownership_{0};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<std::uint64_t> shutdown_epoch_{0};
    std::atomic<std::uint64_t> emergency_epoch_{0};
    std::atomic<std::uint64_t> contention_rejections_{0};
    SendInputConfig config_{};
    EmergencyStopLatch latch_{};
    bool is_initialized_{false};
    bool is_active_{false};
    ISendInputBackend* backend_{nullptr};

    SequenceId last_sequence_id_{0};
    std::optional<ActuationCommand> pending_command_{std::nullopt};
    std::array<ActuationCommand, kMaxRecordedDispatches> recorded_dispatches_{};
    std::size_t recorded_count_{0};

    std::uint64_t total_submitted_{0};
    std::uint64_t total_rejected_{0};
    std::uint64_t total_cancelled_{0};
    MonotonicNs last_dispatch_ns_{0};
    std::uint32_t pressed_buttons_mask_{0};
    SendInputStats stats_{};
};

} // namespace aim::actuation
