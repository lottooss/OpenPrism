#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <atomic>
#include <thread>

#include "aim/actuation/sendinput_actuator.hpp"

using namespace aim;
using namespace aim::actuation;

#define TEST_ASSERT(cond)                                                                            \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (false)

namespace {

ActuationCommand fresh_command() {
    ActuationCommand command{};
    command.generated_at_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    command.correlation_id.source_timestamp_ns = command.generated_at_ns;
    command.desired_apply_time_ns = command.generated_at_ns;
    return command;
}

class MockSendInputBackend final : public ISendInputBackend {
public:
    bool is_available{true};
    bool foreground_matches_result{true};
    bool foreground_identity_matches_result{true};
    std::uintptr_t current_window_handle{0x1234U};
    std::uint32_t current_process_id{4242U};
    InputDispatchResult dispatch_result{.requested_events = 1, .dispatched_events = 1};
    InputDispatchResult release_result{.requested_events = 1, .dispatched_events = 1};
    std::uint64_t dispatch_calls{0};
    std::uint64_t release_calls{0};
    mutable std::uint64_t identity_checks{0};
    std::uint32_t last_release_mask{0};

    [[nodiscard]] bool available() const noexcept override {
        return is_available;
    }

    [[nodiscard]] bool foreground_matches(std::string_view expected_title) const noexcept override {
        return foreground_matches_result && expected_title == "Aim Lab";
    }

    [[nodiscard]] bool foreground_identity_matches(
        std::string_view expected_title, std::uintptr_t expected_window_handle,
        std::uint32_t expected_process_id) const noexcept override {
        ++identity_checks;
        return foreground_identity_matches_result &&
               expected_title == "Aim Lab" &&
               current_window_handle == expected_window_handle &&
               current_process_id == expected_process_id;
    }

    [[nodiscard]] InputDispatchResult dispatch(const ActuationCommand&) noexcept override {
        ++dispatch_calls;
        return dispatch_result;
    }

    [[nodiscard]] InputDispatchResult release_buttons(std::uint32_t mask) noexcept override {
        ++release_calls;
        last_release_mask = mask;
        return release_result;
    }
};

SendInputConfig valid_config() {
    SendInputConfig config{};
    config.target_window_title = "Aim Lab";
    config.verify_foreground = true;
    config.enforce_monotonic_sequence = true;
    return config;
}

ResetToken valid_reset_token() {
    return ResetToken{.token_id = "authorized-test-reset", .generated_at_ns = 1, .is_valid = true};
}

void test_initialization_fails_closed() {
    MockSendInputBackend backend{};
    SendInputActuator actuator{&backend};

    SendInputConfig config = valid_config();
    config.verify_foreground = false;
    TEST_ASSERT(!actuator.initialize(config));

    config = valid_config();
    config.target_window_title.clear();
    TEST_ASSERT(!actuator.initialize(config));

    config = valid_config();
    backend.is_available = false;
    TEST_ASSERT(!actuator.initialize(config));

    backend.is_available = true;
    TEST_ASSERT(actuator.initialize(config = valid_config()));
    backend.foreground_matches_result = false;
    TEST_ASSERT(!actuator.start());
}

void test_monotonic_and_deadline_dispatch() {
    MockSendInputBackend backend{};
    SendInputActuator actuator{&backend};
    SendInputConfig config = valid_config();
    config.max_tolerated_lag_ns = 5'000'000LL;
    TEST_ASSERT(actuator.initialize(config));
    TEST_ASSERT(actuator.start());

    ActuationCommand first = fresh_command();
    first.sequence_id = 1;
    first.delta_x_counts = 10;
    first.delta_y_counts = 5;
    TEST_ASSERT(actuator.submit_latest(first) == SubmitResult::submitted);

    TEST_ASSERT(actuator.submit_latest(first) == SubmitResult::rejected_stale);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));

    const MonotonicNs now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count();
    ActuationCommand expired{};
    expired.sequence_id = 2;
    expired.generated_at_ns = now_ns - 50'000'000LL;
    expired.correlation_id.source_timestamp_ns = expired.generated_at_ns;
    expired.desired_apply_time_ns = expired.generated_at_ns;
    expired.delta_x_counts = 100;
    TEST_ASSERT(actuator.submit_latest(expired) == SubmitResult::rejected_stale);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));
    TEST_ASSERT(actuator.submit_latest(expired) == SubmitResult::rejected_stale);

    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));

    ActuationCommand fresh = fresh_command();
    fresh.sequence_id = 3;
    fresh.delta_x_counts = -5;
    fresh.delta_y_counts = 15;
    TEST_ASSERT(actuator.submit_latest(fresh) == SubmitResult::submitted);

    const SendInputStats stats = actuator.stats();
    TEST_ASSERT(stats.total_dispatches == 2);
    TEST_ASSERT(stats.stale_dropped == 1);
    TEST_ASSERT(stats.out_of_order_dropped == 2);
    TEST_ASSERT(stats.cumulative_counts_x == 5);
    TEST_ASSERT(stats.cumulative_counts_y == 20);
    TEST_ASSERT(backend.dispatch_calls == 2);
}

void test_foreground_mismatch_never_dispatches() {
    MockSendInputBackend backend{};
    SendInputActuator actuator{&backend};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    ActuationCommand press = fresh_command();
    press.sequence_id = 1;
    press.button_transition = {MouseButton::left, ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(press) == SubmitResult::submitted);

    backend.foreground_matches_result = false;
    ActuationCommand command = fresh_command();
    command.sequence_id = 2;
    command.delta_x_counts = 20;
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_foreground);
    TEST_ASSERT(backend.dispatch_calls == 1);
    TEST_ASSERT(backend.release_calls == 1);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.stats().foreground_rejected == 1);
    TEST_ASSERT(actuator.recorded_dispatches().size() == 1);
}

void test_pinned_foreground_identity_rejects_switch() {
    MockSendInputBackend backend{};
    SendInputActuator actuator{&backend};
    SendInputConfig config = valid_config();
    config.target_window_handle = backend.current_window_handle;
    config.target_process_id = backend.current_process_id;
    // The exact identity path must be used when pinned; title-only matching is
    // deliberately disabled to model a same-title window in another process.
    backend.foreground_matches_result = false;
    TEST_ASSERT(actuator.initialize(config));
    TEST_ASSERT(actuator.start());

    backend.current_process_id += 1;
    ActuationCommand command = fresh_command();
    command.sequence_id = 1;
    command.delta_x_counts = 20;
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_foreground);
    TEST_ASSERT(backend.dispatch_calls == 0);
    TEST_ASSERT(backend.identity_checks >= 2);
    TEST_ASSERT(actuator.health().is_latched);
}

void test_dispatch_failure_latches_and_consumes_sequence() {
    MockSendInputBackend backend{};
    backend.dispatch_result = {
        .requested_events = 1,
        .dispatched_events = 0,
        .platform_error = 5,
    };
    SendInputActuator actuator{&backend};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    ActuationCommand command = fresh_command();
    command.sequence_id = 7;
    command.delta_x_counts = 40;
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_dispatch_failed);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.health().total_commands_submitted == 0);
    TEST_ASSERT(actuator.recorded_dispatches().empty());
    TEST_ASSERT(actuator.stats().dispatch_failures == 1);
    TEST_ASSERT(actuator.stats().last_platform_error == 5);

    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));
    backend.dispatch_result = {.requested_events = 1, .dispatched_events = 1};
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_stale);
    TEST_ASSERT(backend.dispatch_calls == 1);
}

void test_partial_button_dispatch_is_cleaned_up() {
    MockSendInputBackend backend{};
    backend.dispatch_result = {
        .requested_events = 2,
        .dispatched_events = 1,
        .platform_error = 87,
    };
    SendInputActuator actuator{&backend};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    ActuationCommand click = fresh_command();
    click.sequence_id = 1;
    click.button_transition = {MouseButton::left, ButtonAction::click};
    TEST_ASSERT(actuator.submit_latest(click) == SubmitResult::rejected_dispatch_failed);

    const std::uint32_t left_mask = 1u << static_cast<std::uint32_t>(MouseButton::left);
    TEST_ASSERT(backend.release_calls == 1);
    TEST_ASSERT(backend.last_release_mask == left_mask);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.stats().partial_dispatch_failures == 1);
}

void test_release_failure_remains_visible_and_blocks_reset() {
    MockSendInputBackend backend{};
    SendInputActuator actuator{&backend};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    ActuationCommand press = fresh_command();
    press.sequence_id = 1;
    press.button_transition = {MouseButton::right, ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(press) == SubmitResult::submitted);

    backend.release_result = {
        .requested_events = 1,
        .dispatched_events = 0,
        .platform_error = 5,
    };
    actuator.cancel_pending();
    const std::uint32_t right_mask = 1u << static_cast<std::uint32_t>(MouseButton::right);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == right_mask);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.stats().release_failures == 1);
    TEST_ASSERT(!actuator.reset_emergency_stop(valid_reset_token()));

    backend.release_result = {.requested_events = 1, .dispatched_events = 1};
    actuator.shutdown();
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));
}

template <typename Actuator>
void assert_common_button_cancellation_contract(Actuator& actuator) {
    ActuationCommand press = fresh_command();
    press.sequence_id = 1;
    press.button_transition = {MouseButton::left, ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(press) == SubmitResult::submitted);
    TEST_ASSERT(actuator.health().pressed_buttons_mask != 0);
    actuator.cancel_pending();
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.health().total_commands_cancelled == 1);
}

void test_null_actuator_contract_parity() {
    ActuatorConfig null_config{};
    NullActuator null_actuator{};
    TEST_ASSERT(null_actuator.initialize(null_config));
    TEST_ASSERT(null_actuator.start());
    assert_common_button_cancellation_contract(null_actuator);

    MockSendInputBackend backend{};
    SendInputActuator sendinput_actuator{&backend};
    TEST_ASSERT(sendinput_actuator.initialize(valid_config()));
    TEST_ASSERT(sendinput_actuator.start());
    assert_common_button_cancellation_contract(sendinput_actuator);
    TEST_ASSERT(backend.release_calls == 1);
}

void test_source_freshness_cannot_be_retimed() {
    for (int fault = 0; fault < 5; ++fault) {
        MockSendInputBackend backend;
        SendInputActuator actuator(&backend);
        TEST_ASSERT(actuator.initialize(valid_config()));
        TEST_ASSERT(actuator.start());
        auto command = fresh_command();
        command.sequence_id = 1;
        command.delta_x_counts = 1;
        if (fault == 0) command.correlation_id.source_timestamp_ns -= 50'000'000LL;
        if (fault == 1) command.correlation_id.source_timestamp_ns = 0;
        if (fault == 2) command.generated_at_ns = 0;
        if (fault == 3) command.desired_apply_time_ns += 1'000'000'000LL;
        if (fault == 4) command.sequence_id = 0;
        TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_stale);
        TEST_ASSERT(backend.dispatch_calls == 0);
    }
    MockSendInputBackend backend;
    SendInputActuator actuator(&backend);
    auto config = valid_config();
    config.max_tolerated_lag_ns = 10'000'001LL;
    TEST_ASSERT(!actuator.initialize(config));
    config = valid_config();
    config.enforce_monotonic_sequence = false;
    TEST_ASSERT(!actuator.initialize(config));
}

void test_duplicate_sequence_releases_held_button() {
    MockSendInputBackend backend;
    SendInputActuator actuator(&backend);
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());
    auto command = fresh_command();
    command.sequence_id = 1;
    command.button_transition = {MouseButton::left, ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::submitted);
    TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_stale);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(backend.dispatch_calls == 1);
    TEST_ASSERT(backend.release_calls == 1);
}

class ReentrantBackend final : public ISendInputBackend {
public:
    SendInputActuator* actuator{nullptr};
    int mode{0};
    mutable bool armed{false};
    unsigned dispatch_calls{0};
    unsigned release_calls{0};
    bool available() const noexcept override { return true; }
    bool foreground_matches(std::string_view) const noexcept override {
        if (armed && mode == 0) actuator->shutdown();
        return true;
    }
    InputDispatchResult dispatch(const ActuationCommand&) noexcept override {
        ++dispatch_calls;
        if (mode == 1) actuator->emergency_stop();
        if (mode == 2) actuator->cancel_pending();
        if (mode == 3) {
            auto nested = fresh_command();
            nested.sequence_id = 2;
            TEST_ASSERT(actuator->submit_latest(nested) == SubmitResult::rejected_dispatch_failed);
        }
        return {.requested_events = 1, .dispatched_events = 1};
    }
    InputDispatchResult release_buttons(std::uint32_t) noexcept override {
        ++release_calls;
        // Cancellation during cleanup is coalesced, never recursive.
        actuator->cancel_pending();
        return {.requested_events = 1, .dispatched_events = 1};
    }
};

void test_reentrant_stop_and_submission_are_bounded() {
    for (int mode = 0; mode < 4; ++mode) {
        ReentrantBackend backend;
        SendInputActuator actuator(&backend);
        backend.actuator = &actuator;
        backend.mode = mode;
        TEST_ASSERT(actuator.initialize(valid_config()));
        TEST_ASSERT(actuator.start());
        backend.armed = true;
        auto command = fresh_command();
        command.sequence_id = 1;
        command.button_transition = {MouseButton::left, ButtonAction::press};
        TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::rejected_latched);
        TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
        TEST_ASSERT(actuator.health().total_commands_submitted == 0);
        TEST_ASSERT(backend.dispatch_calls == (mode == 0 ? 0u : 1u));
        TEST_ASSERT(backend.release_calls == (mode == 0 ? 0u : 1u));
    }
}

class PausedDispatchBackend final : public ISendInputBackend {
public:
    std::atomic<bool> entered{false};
    std::atomic<bool> resume{false};
    std::atomic<bool> timed_out{false};
    std::atomic<unsigned> release_calls{0};
    bool available() const noexcept override { return true; }
    bool foreground_matches(std::string_view) const noexcept override { return true; }
    InputDispatchResult dispatch(const ActuationCommand&) noexcept override {
        entered.store(true, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!resume.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (!resume.load(std::memory_order_acquire)) {
            timed_out.store(true, std::memory_order_release);
            return {.requested_events = 1, .dispatched_events = 0};
        }
        return {.requested_events = 1, .dispatched_events = 1};
    }
    InputDispatchResult release_buttons(std::uint32_t) noexcept override {
        ++release_calls;
        return {.requested_events = 1, .dispatched_events = 1};
    }
};

void test_concurrent_shutdown_releases_in_flight_press() {
    PausedDispatchBackend backend;
    SendInputActuator actuator(&backend);
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());
    SubmitResult result{};
    std::atomic<bool> owner_completed{false};
    std::thread owner([&] {
        auto press = fresh_command();
        press.sequence_id = 1;
        press.button_transition = {MouseButton::left, ButtonAction::press};
        result = actuator.submit_latest(press);
        owner_completed.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!backend.entered.load(std::memory_order_acquire) &&
           !owner_completed.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool reached_dispatch = backend.entered.load(std::memory_order_acquire);
    // These return while the owner is deliberately paused inside dispatch.
    actuator.shutdown();
    actuator.emergency_stop();
    auto next = fresh_command();
    next.sequence_id = 2;
    const auto rejected_next = actuator.submit_latest(next);
    const bool inactive = !actuator.health().is_active;
    // Always unblock and join before asserting. A worker that rejects its
    // command before dispatch (e.g. source expired) must fail, never hang.
    backend.resume.store(true, std::memory_order_release);
    owner.join();
    TEST_ASSERT(reached_dispatch);
    TEST_ASSERT(!backend.timed_out.load(std::memory_order_acquire));
    TEST_ASSERT(rejected_next == SubmitResult::rejected_latched);
    TEST_ASSERT(inactive);
    TEST_ASSERT(result == SubmitResult::rejected_latched);
    TEST_ASSERT(backend.release_calls.load() == 1);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(actuator.health().total_commands_submitted == 0);
    TEST_ASSERT(actuator.stats().contention_rejections == 1);
}

} // namespace

int main() {
    test_initialization_fails_closed();
    test_monotonic_and_deadline_dispatch();
    test_foreground_mismatch_never_dispatches();
    test_pinned_foreground_identity_rejects_switch();
    test_dispatch_failure_latches_and_consumes_sequence();
    test_partial_button_dispatch_is_cleaned_up();
    test_release_failure_remains_visible_and_blocks_reset();
    test_null_actuator_contract_parity();
    test_source_freshness_cannot_be_retimed();
    test_duplicate_sequence_releases_held_button();
    test_reentrant_stop_and_submission_are_bounded();
    test_concurrent_shutdown_releases_in_flight_press();
    std::cout << "All M6-01 SendInput actuator tests passed." << std::endl;
    return 0;
}
