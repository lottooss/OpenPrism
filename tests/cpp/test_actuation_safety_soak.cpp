// tests/cpp/test_actuation_safety_soak.cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "aim/actuation/actuator_scheduler.hpp"
#include "aim/actuation/sendinput_actuator.hpp"
#include "aim/calibration/calibration_model.hpp"

using namespace aim;
using namespace aim::actuation;
using namespace aim::calibration;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

ActuationCommand fresh_command() {
    ActuationCommand command{};
    command.generated_at_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    command.correlation_id.source_timestamp_ns = command.generated_at_ns;
    command.desired_apply_time_ns = command.generated_at_ns;
    return command;
}

class SoakInputBackend final : public ISendInputBackend {
public:
    [[nodiscard]] bool available() const noexcept override { return true; }
    [[nodiscard]] bool foreground_matches(std::string_view title) const noexcept override {
        return title == "Aim Lab";
    }
    [[nodiscard]] InputDispatchResult dispatch(const ActuationCommand&) noexcept override {
        return {.requested_events = 1, .dispatched_events = 1};
    }
    [[nodiscard]] InputDispatchResult release_buttons(std::uint32_t) noexcept override {
        return {.requested_events = 1, .dispatched_events = 1};
    }
};

struct SoakMetrics {
    double p50_delay_us{0.0};
    double p95_delay_us{0.0};
    double p99_delay_us{0.0};
    double max_delay_us{0.0};
    std::uint64_t total_commands{0};
    std::uint64_t estop_events{0};
    std::uint64_t estop_rejections{0};
    std::uint64_t successful_resets{0};
};

SoakMetrics run_actuation_soak_test() {
    // Fake-clock timing must terminate at NullActuator, whose clock is supplied
    // by this replay. SendInput uses real QPC and correctly rejects these times.
    NullActuator actuator{};
    ActuatorConfig act_config{};

    TEST_ASSERT(actuator.initialize(act_config));
    TEST_ASSERT(actuator.start());

    ActuatorScheduler scheduler{&actuator};
    SchedulerConfig sched_config{};
    sched_config.target_frequency_hz = 1000;
    TEST_ASSERT(scheduler.initialize(sched_config, &actuator));

    CalibrationProfile cal_prof{};
    cal_prof.counts_per_pixel_x = 1.0f;
    cal_prof.counts_per_pixel_y = 1.0f;
    CalibrationModel cal_model{cal_prof};

    const std::size_t kTotalIterations = 10000;
    std::uint64_t estop_count = 0;
    std::uint64_t reset_count = 0;

    MonotonicNs sim_time_ns = 1'000'000'000LL;

    for (std::size_t i = 0; i < kTotalIterations; ++i) {
        sim_time_ns += 1'000'000LL; // 1 ms step

        // Every 20 steps, inject a new TrajectoryPlan. Its unrefreshed tail
        // expires after the hard 10ms source deadline.
        if (i % 20 == 0) {
            TrajectoryPlan plan{};
            plan.plan_id = i / 20 + 1;
            plan.correlation_id = {plan.plan_id, sim_time_ns, 6, 0};
            plan.start_time_ns = sim_time_ns;
            plan.end_time_ns = sim_time_ns + 15'000'000LL;
            plan.point_count = 16;
            for (std::uint32_t p = 0; p < 16; ++p) {
                plan.points[p].target_time_ns = sim_time_ns + static_cast<MonotonicNs>(p * 1'000'000LL);
                const auto counts = cal_model.pixels_to_counts(static_cast<float>(p + 1), 0.0f);
                plan.points[p].step_delta_x_counts = counts.counts_x;
                plan.points[p].step_delta_y_counts = counts.counts_y;
            }
            scheduler.submit_plan(plan);
        }

        // Every 2,000 steps, trigger Emergency Stop
        if (i % 2000 == 500) {
            actuator.emergency_stop(SafetyReason::emergency_stop_triggered);
            estop_count++;
            TEST_ASSERT(actuator.health().is_latched);
            TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
        }

        // 100 steps after Emergency Stop, perform authorized reset
        if (i % 2000 == 600) {
            ResetToken token{};
            token.token_id = "soak_test_master_token";
            token.is_valid = true;
            TEST_ASSERT(actuator.reset_emergency_stop(token));
            TEST_ASSERT(!actuator.health().is_latched);
            reset_count++;
        }

        // Tick scheduler step
        scheduler.tick_step(sim_time_ns);
    }

    const auto jitter = scheduler.jitter_stats();
    const auto act_health = actuator.health();
    TEST_ASSERT(jitter.steps_dispatched == act_health.total_commands_submitted);
    TEST_ASSERT(jitter.steps_dispatched > 5000);

    SoakMetrics metrics{};
    metrics.p50_delay_us = jitter.p50_delay_us;
    metrics.p95_delay_us = jitter.p95_delay_us;
    metrics.p99_delay_us = jitter.p99_delay_us;
    metrics.max_delay_us = jitter.max_delay_us;
    metrics.total_commands = jitter.steps_dispatched;
    metrics.estop_events = estop_count;
    metrics.estop_rejections = act_health.total_commands_rejected;
    metrics.successful_resets = reset_count;

    return metrics;
}

void run_fault_campaign_tests() {
    std::cout << "  Running M6-06 Fault Injection Campaign..." << std::endl;

    // Fault 1: Stale data and sequence inversion
    {
        SoakInputBackend backend{};
        SendInputActuator actuator{&backend};
        SendInputConfig cfg{};
        cfg.target_window_title = "Aim Lab";
        cfg.max_tolerated_lag_ns = 5'000'000LL;
        TEST_ASSERT(actuator.initialize(cfg));
        TEST_ASSERT(actuator.start());

        ActuationCommand c1 = fresh_command();
        c1.sequence_id = 10;
        c1.delta_x_counts = 10;
        TEST_ASSERT(actuator.submit_latest(c1) == SubmitResult::submitted);

        // Inverted sequence
        ActuationCommand c_old{};
        c_old.sequence_id = 9;
        c_old.delta_x_counts = 10;
        TEST_ASSERT(actuator.submit_latest(c_old) == SubmitResult::rejected_stale);
    }

    // Fault 2: Lost focus during active button press
    {
        SoakInputBackend backend{};
        SendInputActuator actuator{&backend};
        SendInputConfig cfg{};
        cfg.target_window_title = "Aim Lab";
        TEST_ASSERT(actuator.initialize(cfg));
        TEST_ASSERT(actuator.start());

        ActuationCommand press = fresh_command();
        press.sequence_id = 1;
        press.button_transition = {MouseButton::left, ButtonAction::press};
        TEST_ASSERT(actuator.submit_latest(press) == SubmitResult::submitted);
        TEST_ASSERT(actuator.health().pressed_buttons_mask != 0);

        // Emergency stop due to focus loss
        actuator.emergency_stop(SafetyReason::lost_focus);
        TEST_ASSERT(actuator.health().is_latched);
        TEST_ASSERT(actuator.health().pressed_buttons_mask == 0); // Must be cleanly released!
    }

    // Fault 3: Invalid calibration profile
    {
        CalibrationProfile bad_prof{};
        bad_prof.counts_per_pixel_x = -5.0f; // Negative gain illegal!
        CalibrationModel model{bad_prof};
        TEST_ASSERT(!model.is_valid());
        const auto counts = model.pixels_to_counts(50.0f, 50.0f);
        TEST_ASSERT(counts.counts_x == 0 && counts.counts_y == 0); // Fails safe to zero movement
    }

    // Fault 4: Shutdown during button transition
    {
        SoakInputBackend backend{};
        SendInputActuator actuator{&backend};
        SendInputConfig cfg{};
        cfg.target_window_title = "Aim Lab";
        TEST_ASSERT(actuator.initialize(cfg));
        TEST_ASSERT(actuator.start());

        ActuationCommand press = fresh_command();
        press.sequence_id = 1;
        press.button_transition = {MouseButton::right, ButtonAction::press};
        TEST_ASSERT(actuator.submit_latest(press) == SubmitResult::submitted);
        TEST_ASSERT(actuator.health().pressed_buttons_mask != 0);

        // Shutdown cleanly releases held buttons
        actuator.shutdown();
        TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    }

    std::cout << "  -> M6-06 Fault Injection Campaign passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M6 Actuation & Safety Soak Benchmark         " << std::endl;
    std::cout << "================================================================" << std::endl;

    const SoakMetrics m = run_actuation_soak_test();

    std::cout << "  Soak Execution Summary (10,000 continuous microsteps):" << std::endl;
    std::cout << "    Total Steps Dispatched:   " << m.total_commands << std::endl;
    std::cout << "    Emergency Stop Events:    " << m.estop_events << std::endl;
    std::cout << "    Latched Safety Rejections:" << m.estop_rejections << std::endl;
    std::cout << "    Successful Token Resets:  " << m.successful_resets << std::endl;
    std::cout << std::endl;

    std::cout << "  1 kHz Scheduler Jitter Telemetry:" << std::endl;
    std::cout << "    p50 Delay: " << m.p50_delay_us << " us (" << (m.p50_delay_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    p95 Delay: " << m.p95_delay_us << " us (" << (m.p95_delay_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    p99 Delay: " << m.p99_delay_us << " us (" << (m.p99_delay_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    Max Delay: " << m.max_delay_us << " us (" << (m.max_delay_us / 1000.0) << " ms)" << std::endl;

    TEST_ASSERT(m.total_commands > 5000);
    TEST_ASSERT(m.estop_events == 5);
    TEST_ASSERT(m.successful_resets == 5);
    TEST_ASSERT(m.estop_rejections > 0); // Proof of fail-closed latching

    run_fault_campaign_tests();

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M6 Actuation & Safety Acceptance Passed!         " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
