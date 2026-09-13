// tests/cpp/test_safety_supervisor.cpp
#include <iostream>
#include <limits>

#include "aim/safety/safety_supervisor.hpp"

using namespace aim;
using namespace aim::safety;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

ActuationCommand valid_command() {
    ActuationCommand command{};
    command.sequence_id = 1;
    command.generated_at_ns = 1'000'000'000LL;
    command.delta_x_counts = 20;
    return command;
}

void install_valid_checks(SafetySupervisor& supervisor) {
    supervisor.set_focus_callback([] { return true; });
    supervisor.set_calibration_callback([] { return true; });
}

void test_missing_callbacks_reject() {
    auto command = valid_command();
    for (int missing = 0; missing < 2; ++missing) {
        SafetySupervisor supervisor{};
        if (missing == 0) {
            supervisor.set_calibration_callback([] { return true; });
        } else {
            supervisor.set_focus_callback([] { return true; });
        }
        TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));
        TEST_ASSERT(supervisor.is_latched());
        TEST_ASSERT(supervisor.last_reason() == (missing == 0 ? SafetyReason::lost_focus
                                                              : SafetyReason::invalid_calibration));
        TEST_ASSERT(supervisor.stats().total_commands_passed == 0);
    }
    // Explicit offline configurations may disable their corresponding checks.
    SafetySupervisorConfig offline{};
    offline.require_foreground = false;
    offline.require_valid_calibration = false;
    SafetySupervisor supervisor{offline};
    TEST_ASSERT(supervisor.check_actuation_safety(command, command.generated_at_ns));
}

void test_temporal_validation_without_overflow() {
    const auto max_time = std::numeric_limits<MonotonicNs>::max();
    struct TimingCase { MonotonicNs generated; MonotonicNs now; bool accepted; };
    const TimingCase cases[] = {
        {0, 1, false}, {-1, 1, false}, {1, 0, false}, {1, -1, false},
        {2, 1, false}, {1, 10'000'001, true}, {1, 10'000'002, false},
        {max_time - 1, max_time, true}, {1, max_time, false},
        {std::numeric_limits<MonotonicNs>::min(), max_time, false}
    };
    for (const auto& test : cases) {
        SafetySupervisor supervisor{};
        install_valid_checks(supervisor);
        auto command = valid_command();
        command.generated_at_ns = test.generated;
        TEST_ASSERT(supervisor.check_actuation_safety(command, test.now) == test.accepted);
        TEST_ASSERT(supervisor.stats().total_commands_passed == (test.accepted ? 1u : 0u));
        TEST_ASSERT(supervisor.last_reason() == (test.accepted ? SafetyReason::none : SafetyReason::stale_data));
    }
    for (const MonotonicNs invalid_age : {MonotonicNs{0}, MonotonicNs{-1}, MonotonicNs{10'000'001}, max_time}) {
        SafetySupervisorConfig config{};
        config.max_command_age_ns = invalid_age;
        SafetySupervisor supervisor{config};
        install_valid_checks(supervisor);
        const auto command = valid_command();
        TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));
    }
}

void test_zero_and_rejected_sequence_cannot_bypass_or_poison() {
    SafetySupervisor supervisor{};
    install_valid_checks(supervisor);
    auto command = valid_command();
    command.sequence_id = 0;
    TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));
    command.sequence_id = std::numeric_limits<SequenceId>::max();
    TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns + 10'000'001));
    command.sequence_id = 1;
    TEST_ASSERT(supervisor.check_actuation_safety(command, command.generated_at_ns));
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::none);
    TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));

    SafetySupervisorConfig unordered{};
    unordered.enforce_monotonic_sequence = false;
    SafetySupervisor without_ordering{unordered};
    install_valid_checks(without_ordering);
    command.sequence_id = 0;
    TEST_ASSERT(!without_ordering.check_actuation_safety(command, command.generated_at_ns));
}

void test_extreme_counts_and_rejection_cancel_buttons() {
    for (const auto extreme : {std::numeric_limits<std::int32_t>::min(),
                               std::numeric_limits<std::int32_t>::max()}) {
        for (int axis = 0; axis < 2; ++axis) {
            SafetySupervisor supervisor{};
            install_valid_checks(supervisor);
            NullActuator actuator{};
            TEST_ASSERT(actuator.initialize({}));
            TEST_ASSERT(actuator.start());
            auto command = valid_command();
            command.button_transition = {MouseButton::left, ButtonAction::press};
            TEST_ASSERT(supervisor.check_actuation_safety(command, command.generated_at_ns));
            TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::submitted);
            TEST_ASSERT(actuator.health().pressed_buttons_mask != 0);
            command.sequence_id = 2;
            if (axis == 0) command.delta_x_counts = extreme;
            else command.delta_y_counts = extreme;
            TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));
            // The dispatch owner pairs a rejected check with cancellation.
            actuator.cancel_pending();
            TEST_ASSERT(supervisor.last_reason() == SafetyReason::out_of_bounds);
            TEST_ASSERT(supervisor.is_latched());
            TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
            TEST_ASSERT(actuator.health().total_commands_submitted == 1);
        }
    }
    for (const std::int32_t limit : {0, -1, std::numeric_limits<std::int32_t>::min()}) {
        SafetySupervisorConfig config{};
        config.max_single_step_counts = limit;
        SafetySupervisor supervisor{config};
        install_valid_checks(supervisor);
        const auto command = valid_command();
        TEST_ASSERT(!supervisor.check_actuation_safety(command, command.generated_at_ns));
        TEST_ASSERT(supervisor.last_reason() == SafetyReason::out_of_bounds);
    }
}

void test_all_safety_gates() {
    std::cout << "[Test 1] Testing all centralized safety gates..." << std::endl;

    SafetySupervisorConfig config{};
    config.max_command_age_ns = 10'000'000LL; // 10ms
    config.max_single_step_counts = 100;
    config.authorized_reset_token = "secret_reset_token";

    SafetySupervisor supervisor{config};
    bool focus_active = true;
    bool calib_valid = true;

    supervisor.set_focus_callback([&]() { return focus_active; });
    supervisor.set_calibration_callback([&]() { return calib_valid; });

    // 1. Valid command passes
    ActuationCommand cmd1{};
    cmd1.sequence_id = 1;
    cmd1.generated_at_ns = 1'000'000'000LL;
    cmd1.delta_x_counts = 25;
    cmd1.delta_y_counts = -30;
    TEST_ASSERT(supervisor.check_actuation_safety(cmd1, 1'005'000'000LL));

    // 2. Duplicate sequence fails closed
    TEST_ASSERT(!supervisor.check_actuation_safety(cmd1, 1'006'000'000LL));
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::stale_data);

    // 3. Stale command (> 10ms old) fails closed
    ActuationCommand cmd2{};
    cmd2.sequence_id = 2;
    cmd2.generated_at_ns = 1'000'000'000LL;
    cmd2.delta_x_counts = 10;
    TEST_ASSERT(!supervisor.check_actuation_safety(cmd2, 1'020'000'000LL)); // 20ms old!
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::stale_data);

    // 4. Oversized command (> 100 counts) fails closed and triggers emergency stop
    ActuationCommand cmd3{};
    cmd3.sequence_id = 3;
    cmd3.generated_at_ns = 1'030'000'000LL;
    cmd3.delta_x_counts = 150; // Exceeds limit
    TEST_ASSERT(!supervisor.check_actuation_safety(cmd3, 1'031'000'000LL));
    TEST_ASSERT(supervisor.is_latched());
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::out_of_bounds);

    // 5. Try invalid reset token -> fails
    ResetToken bad_token{.token_id = "wrong_token", .is_valid = true};
    TEST_ASSERT(!supervisor.try_reset(bad_token));
    TEST_ASSERT(supervisor.is_latched());

    // 6. Valid reset token succeeds
    ResetToken good_token{.token_id = "secret_reset_token", .is_valid = true};
    TEST_ASSERT(supervisor.try_reset(good_token));
    TEST_ASSERT(!supervisor.is_latched());

    // 7. Focus loss fails closed
    focus_active = false;
    ActuationCommand cmd4{};
    cmd4.sequence_id = 4;
    cmd4.generated_at_ns = 1'040'000'000LL;
    cmd4.delta_x_counts = 10;
    TEST_ASSERT(!supervisor.check_actuation_safety(cmd4, 1'041'000'000LL));
    TEST_ASSERT(supervisor.is_latched());
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::lost_focus);

    // Reset and test calibration failure
    focus_active = true;
    TEST_ASSERT(supervisor.try_reset(good_token));

    calib_valid = false;
    ActuationCommand cmd5{};
    cmd5.sequence_id = 5;
    cmd5.generated_at_ns = 1'050'000'000LL;
    cmd5.delta_x_counts = 10;
    TEST_ASSERT(!supervisor.check_actuation_safety(cmd5, 1'051'000'000LL));
    TEST_ASSERT(supervisor.is_latched());
    TEST_ASSERT(supervisor.last_reason() == SafetyReason::invalid_calibration);

    const auto stats = supervisor.stats();
    TEST_ASSERT(stats.total_commands_checked >= 6);
    TEST_ASSERT(stats.total_commands_passed == 1);
    TEST_ASSERT(stats.emergency_stops_triggered >= 3);

    std::cout << "  -> All centralized safety gates passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M6-05 Safety Supervisor Unit Tests           " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_all_safety_gates();
    test_missing_callbacks_reject();
    test_temporal_validation_without_overflow();
    test_zero_and_rejected_sequence_cannot_bypass_or_poison();
    test_extreme_counts_and_rejection_cancel_buttons();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M6-05 Safety Supervisor Tests Passed Successfully!         " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
