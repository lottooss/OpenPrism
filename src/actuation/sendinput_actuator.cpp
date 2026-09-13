// src/actuation/sendinput_actuator.cpp
#include "aim/actuation/sendinput_actuator.hpp"

#include <chrono>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace aim::actuation {
namespace {

constexpr std::uint32_t button_bit(MouseButton button) noexcept {
    const auto value = static_cast<std::uint32_t>(button);
    return button != MouseButton::none && value < 32u ? (1u << value) : 0u;
}

class PlatformSendInputBackend final : public ISendInputBackend {
public:
    [[nodiscard]] bool available() const noexcept override {
#if defined(_WIN32)
        return true;
#else
        return false;
#endif
    }

    [[nodiscard]] bool foreground_matches([[maybe_unused]] std::string_view expected_title) const noexcept override {
#if defined(_WIN32)
        const HWND window = ::GetForegroundWindow();
        return title_matches(window, expected_title);
#else
        return false;
#endif
    }

    [[nodiscard]] bool foreground_identity_matches(
        std::string_view expected_title, std::uintptr_t expected_window_handle,
        std::uint32_t expected_process_id) const noexcept override {
#if defined(_WIN32)
        if (expected_window_handle == 0 || expected_process_id == 0) {
            return false;
        }
        const HWND window = ::GetForegroundWindow();
        if (reinterpret_cast<std::uintptr_t>(window) != expected_window_handle) {
            return false;
        }
        DWORD process_id{};
        if (::GetWindowThreadProcessId(window, &process_id) == 0 ||
            process_id != expected_process_id) {
            return false;
        }
        return title_matches(window, expected_title);
#else
        (void)expected_title;
        (void)expected_window_handle;
        (void)expected_process_id;
        return false;
#endif
    }

    [[nodiscard]] InputDispatchResult dispatch([[maybe_unused]] const ActuationCommand& command) noexcept override {
#if defined(_WIN32)
        INPUT inputs[3]{};
        UINT count = 0;

        if (command.delta_x_counts != 0 || command.delta_y_counts != 0) {
            INPUT& input = inputs[count++];
            input.type = INPUT_MOUSE;
            input.mi.dx = static_cast<LONG>(command.delta_x_counts);
            input.mi.dy = static_cast<LONG>(command.delta_y_counts);
            input.mi.dwFlags = MOUSEEVENTF_MOVE;
        }

        const auto append_button = [&inputs, &count](MouseButton button, bool down) noexcept {
            INPUT& input = inputs[count++];
            input.type = INPUT_MOUSE;
            switch (button) {
                case MouseButton::left:
                    input.mi.dwFlags = down ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
                    break;
                case MouseButton::right:
                    input.mi.dwFlags = down ? MOUSEEVENTF_RIGHTDOWN : MOUSEEVENTF_RIGHTUP;
                    break;
                case MouseButton::middle:
                    input.mi.dwFlags = down ? MOUSEEVENTF_MIDDLEDOWN : MOUSEEVENTF_MIDDLEUP;
                    break;
                case MouseButton::extra1:
                    input.mi.dwFlags = down ? MOUSEEVENTF_XDOWN : MOUSEEVENTF_XUP;
                    input.mi.mouseData = XBUTTON1;
                    break;
                case MouseButton::extra2:
                    input.mi.dwFlags = down ? MOUSEEVENTF_XDOWN : MOUSEEVENTF_XUP;
                    input.mi.mouseData = XBUTTON2;
                    break;
                case MouseButton::none:
                    --count;
                    break;
            }
        };

        const auto& transition = command.button_transition;
        if (transition.action == ButtonAction::press || transition.action == ButtonAction::click) {
            append_button(transition.button, true);
        }
        if (transition.action == ButtonAction::release || transition.action == ButtonAction::click) {
            append_button(transition.button, false);
        }

        if (count == 0) {
            return {};
        }

        ::SetLastError(ERROR_SUCCESS);
        const UINT sent = ::SendInput(count, inputs, sizeof(INPUT));
        const DWORD error = sent == count ? ERROR_SUCCESS : ::GetLastError();
        return InputDispatchResult{
            .requested_events = count,
            .dispatched_events = sent,
            .platform_error = static_cast<std::uint32_t>(error),
        };
#else
        return InputDispatchResult{.requested_events = 1, .dispatched_events = 0, .platform_error = 1};
#endif
    }

    [[nodiscard]] InputDispatchResult release_buttons([[maybe_unused]] std::uint32_t mask) noexcept override {
#if defined(_WIN32)
        INPUT inputs[5]{};
        UINT count = 0;

        const auto append_release = [&inputs, &count](MouseButton button) noexcept {
            INPUT& input = inputs[count++];
            input.type = INPUT_MOUSE;
            switch (button) {
                case MouseButton::left:
                    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                    break;
                case MouseButton::right:
                    input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
                    break;
                case MouseButton::middle:
                    input.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
                    break;
                case MouseButton::extra1:
                    input.mi.dwFlags = MOUSEEVENTF_XUP;
                    input.mi.mouseData = XBUTTON1;
                    break;
                case MouseButton::extra2:
                    input.mi.dwFlags = MOUSEEVENTF_XUP;
                    input.mi.mouseData = XBUTTON2;
                    break;
                case MouseButton::none:
                    --count;
                    break;
            }
        };

        for (const MouseButton button : {MouseButton::left,
                                         MouseButton::right,
                                         MouseButton::middle,
                                         MouseButton::extra1,
                                         MouseButton::extra2}) {
            if ((mask & button_bit(button)) != 0) {
                append_release(button);
            }
        }

        if (count == 0) {
            return {};
        }

        ::SetLastError(ERROR_SUCCESS);
        const UINT sent = ::SendInput(count, inputs, sizeof(INPUT));
        const DWORD error = sent == count ? ERROR_SUCCESS : ::GetLastError();
        return InputDispatchResult{
            .requested_events = count,
            .dispatched_events = sent,
            .platform_error = static_cast<std::uint32_t>(error),
        };
#else
        return InputDispatchResult{.requested_events = 1, .dispatched_events = 0, .platform_error = 1};
#endif
    }

private:
#if defined(_WIN32)
    [[nodiscard]] static bool title_matches(HWND window,
                                            std::string_view expected_title) noexcept {
        if (window == nullptr || expected_title.empty()) {
            return false;
        }

        char title[256]{};
        const int length =
            ::GetWindowTextA(window, title, static_cast<int>(sizeof(title)));
        if (length <= 0) {
            return false;
        }
        return std::string_view{title, static_cast<std::size_t>(length)}.find(
                   expected_title) != std::string_view::npos;
    }
#endif
};

ISendInputBackend* platform_backend() noexcept {
    static PlatformSendInputBackend backend{};
    return &backend;
}

MonotonicNs monotonic_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

SendInputActuator::SendInputActuator() noexcept : backend_(platform_backend()) {}

SendInputActuator::SendInputActuator(ISendInputBackend* backend) noexcept : backend_(backend) {}

bool SendInputActuator::try_ownership() const noexcept {
    std::uint32_t expected = 0;
    return ownership_.compare_exchange_strong(expected, kOwned, std::memory_order_acquire);
}

void SendInputActuator::cancel_owned() noexcept {
    if (shutdown_requested_.load(std::memory_order_acquire)) is_active_ = false;
    if (pending_command_) {
        pending_command_.reset();
        ++total_cancelled_;
    }
    if (!release_all_held_buttons_unsafe()) latch_.trigger(SafetyReason::driver_error);
}

void SendInputActuator::release_ownership() noexcept {
    std::uint32_t expected = kOwned;
    if (ownership_.compare_exchange_strong(expected, 0, std::memory_order_release)) return;
    // A cancellation raced this owner. No new dispatch can enter while we
    // release. Further cancel requests cover the same already-cleared state;
    // shutdown/latch flags remain independently visible after ownership ends.
    cancel_owned();
    ownership_.store(0, std::memory_order_release);
}

bool SendInputActuator::initialize(const ActuatorConfig& config) noexcept {
    SendInputConfig send_config{};
    send_config.base_config = config;
    return initialize(send_config);
}

bool SendInputActuator::initialize(const SendInputConfig& config) noexcept {
    if (!try_ownership()) return false;
    const OwnershipScope release{*this};
    is_initialized_ = false;
    is_active_ = false;

    if (!release_all_held_buttons_unsafe()) {
        latch_.trigger(SafetyReason::driver_error);
        return false;
    }
    if (backend_ == nullptr || !backend_->available() || !config.verify_foreground ||
        config.target_window_title.empty() || !config.base_config.relative_counts ||
        !config.base_config.require_emergency_stop || !config.enforce_monotonic_sequence ||
        config.max_tolerated_lag_ns <= 0 || config.max_tolerated_lag_ns > 10'000'000LL) {
        return false;
    }

    config_ = config;
    last_sequence_id_ = 0;
    pending_command_.reset();
    recorded_count_ = 0;
    total_submitted_ = 0;
    total_rejected_ = 0;
    total_cancelled_ = 0;
    last_dispatch_ns_ = 0;
    stats_ = SendInputStats{};
    contention_rejections_.store(0);
    is_initialized_ = true;
    return true;
}

bool SendInputActuator::start() noexcept {
    const auto stop_epoch = shutdown_epoch_.load(std::memory_order_acquire);
    if (!try_ownership()) return false;
    const OwnershipScope release{*this};
    shutdown_requested_.store(false, std::memory_order_release);
    if (!is_initialized_ || latch_.is_latched() || !check_foreground_window_unsafe()) {
        return false;
    }
    if (shutdown_epoch_.load(std::memory_order_acquire) != stop_epoch ||
        (ownership_.load(std::memory_order_acquire) & kCancelRequested) != 0) {
        shutdown_requested_.store(true, std::memory_order_release);
        return false;
    }
    is_active_ = true;
    return true;
}

bool SendInputActuator::check_foreground_window_unsafe() const noexcept {
    if (backend_ == nullptr || !backend_->available() ||
        !config_.verify_foreground || config_.target_window_title.empty()) {
        return false;
    }
    if (config_.target_window_handle != 0 || config_.target_process_id != 0) {
        return backend_->foreground_identity_matches(
            config_.target_window_title, config_.target_window_handle,
            config_.target_process_id);
    }
    return backend_->foreground_matches(config_.target_window_title);
}

bool SendInputActuator::check_foreground_window() const noexcept {
    if (!try_ownership()) return false;
    const OwnershipScope release{*const_cast<SendInputActuator*>(this)};
    return check_foreground_window_unsafe();
}

bool SendInputActuator::release_all_held_buttons_unsafe() noexcept {
    if (pressed_buttons_mask_ == 0) {
        return true;
    }

    const InputDispatchResult result = backend_ != nullptr
                                           ? backend_->release_buttons(pressed_buttons_mask_)
                                           : InputDispatchResult{.requested_events = 1};
    if (result.platform_error != 0) {
        stats_.last_platform_error = result.platform_error;
    }
    if (!result.complete()) {
        ++stats_.release_failures;
        return false;
    }

    pressed_buttons_mask_ = 0;
    return true;
}

void SendInputActuator::conservatively_track_partial_button_dispatch(
    const ActuationCommand& command) noexcept {
    const auto& transition = command.button_transition;
    if (transition.action == ButtonAction::press || transition.action == ButtonAction::click) {
        pressed_buttons_mask_ |= button_bit(transition.button);
    }
}

SubmitResult SendInputActuator::submit_latest(const ActuationCommand& command) noexcept {
    if (!try_ownership()) {
        ++contention_rejections_;
        cancel_pending();
        return latch_.is_latched() ? SubmitResult::rejected_latched : SubmitResult::rejected_dispatch_failed;
    }
    const OwnershipScope release{*this};
    if (!is_initialized_ || !is_active_ || shutdown_requested_.load(std::memory_order_acquire)) {
        ++total_rejected_;
        return SubmitResult::rejected_uninitialized;
    }
    if (latch_.is_latched()) {
        ++total_rejected_;
        return SubmitResult::rejected_latched;
    }

    if (command.sequence_id == 0 || command.sequence_id <= last_sequence_id_) {
        ++total_rejected_;
        ++stats_.out_of_order_dropped;
        cancel_owned();
        latch_.trigger(SafetyReason::stale_data);
        return SubmitResult::rejected_stale;
    }

    const MonotonicNs now_ns = monotonic_now_ns();
    const MonotonicNs target_time =
        command.desired_apply_time_ns > 0 ? command.desired_apply_time_ns : command.generated_at_ns;
    const MonotonicNs source_time = command.correlation_id.source_timestamp_ns;
    // A fresh desired time must never disguise an old captured observation.
    // Order comparisons before subtraction to avoid signed timestamp overflow.
    if (source_time <= 0 || command.generated_at_ns <= 0 ||
        source_time > command.generated_at_ns || command.generated_at_ns > now_ns ||
        target_time < source_time || target_time > now_ns ||
        now_ns - source_time > config_.max_tolerated_lag_ns) {
        if (command.sequence_id > 0) {
            last_sequence_id_ = command.sequence_id;
        }
        ++total_rejected_;
        ++stats_.stale_dropped;
        if (!release_all_held_buttons_unsafe()) {
            latch_.trigger(SafetyReason::driver_error);
        } else {
            latch_.trigger(SafetyReason::stale_data);
        }
        return SubmitResult::rejected_stale;
    }

    if (!check_foreground_window_unsafe()) {
        if (command.sequence_id > 0) {
            last_sequence_id_ = command.sequence_id;
        }
        ++total_rejected_;
        ++stats_.foreground_rejected;
        if (!release_all_held_buttons_unsafe()) {
            latch_.trigger(SafetyReason::driver_error);
        } else {
            latch_.trigger(SafetyReason::lost_focus);
        }
        return SubmitResult::rejected_foreground;
    }

    // Foreground/backend callbacks may request stop reentrantly. Revalidate
    // immediately before entering the OS boundary; never dispatch after that.
    if (latch_.is_latched() || shutdown_requested_.load(std::memory_order_acquire) ||
        (ownership_.load(std::memory_order_acquire) & kCancelRequested) != 0) {
        ++total_rejected_;
        cancel_owned();
        return SubmitResult::rejected_latched;
    }
    const InputDispatchResult result = backend_->dispatch(command);
    if (command.sequence_id > 0) {
        // A platform call may have partially taken effect. Consuming the
        // sequence prevents a caller retry from duplicating it.
        last_sequence_id_ = command.sequence_id;
    }

    if (result.platform_error != 0) {
        stats_.last_platform_error = result.platform_error;
    }
    if (!result.complete()) {
        ++total_rejected_;
        ++stats_.dispatch_failures;
        if (result.partial()) {
            ++stats_.partial_dispatch_failures;
            conservatively_track_partial_button_dispatch(command);
        }
        (void)release_all_held_buttons_unsafe();
        latch_.trigger(SafetyReason::driver_error);
        return SubmitResult::rejected_dispatch_failed;
    }

    const auto& transition = command.button_transition;
    const std::uint32_t bit = button_bit(transition.button);
    if (transition.action == ButtonAction::press) {
        pressed_buttons_mask_ |= bit;
        ++stats_.button_presses;
    } else if (transition.action == ButtonAction::release) {
        pressed_buttons_mask_ &= ~bit;
        ++stats_.button_releases;
    } else if (transition.action == ButtonAction::click) {
        ++stats_.button_presses;
        ++stats_.button_releases;
    }

    // A call already inside SendInput cannot be recalled. Record the possible
    // held state first, release it, and reject the receipt if cancellation won.
    if (latch_.is_latched() || shutdown_requested_.load(std::memory_order_acquire) ||
        (ownership_.load(std::memory_order_acquire) & kCancelRequested) != 0) {
        ++total_rejected_;
        cancel_owned();
        return SubmitResult::rejected_latched;
    }

    // Receipt commit point: the final cancellation-word read above succeeded
    // after the OS reported complete dispatch. A cancellation published after
    // that read releases held state in OwnershipScope, but cannot undo already
    // executed input; retain its accepted receipt to prevent duplicate clicks.
    // Latch/shutdown flags independently reject subsequent submissions while
    // the owning call finishes this bounded accounting section.
    last_dispatch_ns_ = now_ns;
    pending_command_ = command;
    if (recorded_count_ < kMaxRecordedDispatches) {
        recorded_dispatches_[recorded_count_++] = command;
    } else {
        recorded_dispatches_[(recorded_count_++) % kMaxRecordedDispatches] = command;
    }

    stats_.cumulative_counts_x += command.delta_x_counts;
    stats_.cumulative_counts_y += command.delta_y_counts;
    ++stats_.total_dispatches;
    ++total_submitted_;
    return SubmitResult::submitted;
}

void SendInputActuator::cancel_pending() noexcept {
    ownership_.fetch_or(kCancelRequested, std::memory_order_acq_rel);
    if ((ownership_.fetch_or(kOwned, std::memory_order_acq_rel) & kOwned) != 0) return;
    // release_ownership handles the cancellation once, including requests
    // arriving from a backend release callback. No recursion or retry loop.
    const OwnershipScope release{*this};
}

void SendInputActuator::emergency_stop(SafetyReason reason) noexcept {
    emergency_epoch_.fetch_add(1, std::memory_order_acq_rel);
    latch_.trigger(reason);
    cancel_pending();
}

bool SendInputActuator::reset_emergency_stop(const ResetToken& token) noexcept {
    const auto epoch = emergency_epoch_.load(std::memory_order_acquire);
    if (!try_ownership()) return false;
    const OwnershipScope release{*this};
    if (pressed_buttons_mask_ != 0) {
        return false;
    }
    const bool reset = latch_.try_reset(token);
    if (emergency_epoch_.load(std::memory_order_acquire) != epoch) {
        latch_.trigger(SafetyReason::emergency_stop_triggered);
        return false;
    }
    return reset;
}

void SendInputActuator::shutdown() noexcept {
    shutdown_epoch_.fetch_add(1, std::memory_order_acq_rel);
    shutdown_requested_.store(true, std::memory_order_release);
    cancel_pending();
}

ActuatorHealth SendInputActuator::health() const noexcept {
    if (!try_ownership()) return {.is_active = false, .is_latched = true};
    const OwnershipScope release{*const_cast<SendInputActuator*>(this)};
    return ActuatorHealth{
        .is_active = is_active_ && !shutdown_requested_.load(std::memory_order_acquire),
        .is_latched = latch_.is_latched(),
        .total_commands_submitted = total_submitted_,
        .total_commands_rejected = total_rejected_ + contention_rejections_.load(),
        .total_commands_cancelled = total_cancelled_,
        .last_dispatch_ns = last_dispatch_ns_,
        .pressed_buttons_mask = pressed_buttons_mask_,
    };
}

SendInputStats SendInputActuator::stats() const noexcept {
    if (!try_ownership()) return {};
    const OwnershipScope release{*const_cast<SendInputActuator*>(this)};
    auto snapshot = stats_;
    snapshot.contention_rejections = contention_rejections_.load();
    return snapshot;
}

std::vector<ActuationCommand> SendInputActuator::recorded_dispatches() const {
    if (!try_ownership()) return {};
    const OwnershipScope release{*const_cast<SendInputActuator*>(this)};
    const std::size_t valid =
        recorded_count_ < kMaxRecordedDispatches ? recorded_count_ : kMaxRecordedDispatches;
    return std::vector<ActuationCommand>(recorded_dispatches_.begin(), recorded_dispatches_.begin() + valid);
}

void SendInputActuator::clear_recorded_dispatches() noexcept {
    if (!try_ownership()) return;
    const OwnershipScope release{*this};
    recorded_count_ = 0;
}

} // namespace aim::actuation
