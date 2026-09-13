// tests/cpp/test_actuator_scheduler.cpp
#include <chrono>
#include <iostream>
#include <thread>

#include "aim/actuation/actuator_scheduler.hpp"
#include "aim/core/actuator.hpp"

using namespace aim;
using namespace aim::actuation;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {
void stamp_plan(TrajectoryPlan& plan) {
    plan.start_time_ns = plan.points[0].target_time_ns - 1'000'000LL;
    plan.correlation_id = {42, plan.start_time_ns, 999, 7};
    plan.end_time_ns = plan.points[plan.point_count - 1].target_time_ns;
}
void test_due_source_failure_and_sync_stop() {
    NullActuator actuator;
    TEST_ASSERT(actuator.initialize({}));
    TEST_ASSERT(actuator.start());
    ActuatorScheduler scheduler{&actuator};
    TrajectoryPlan plan{};
    plan.plan_id = 1;
    plan.point_count = 2;
    plan.points[0].target_time_ns = 1'001'000'000LL;
    plan.points[1].target_time_ns = 1'009'000'000LL;
    plan.points[0].step_delta_x_counts = 1;
    plan.points[1].step_delta_x_counts = 8;
    stamp_plan(plan);
    TEST_ASSERT(scheduler.try_submit_plan(plan));
    TEST_ASSERT(!scheduler.tick_step(1'000'999'999LL));
    TEST_ASSERT(actuator.health().total_commands_submitted == 0);
    TEST_ASSERT(scheduler.tick_step(1'001'000'000LL));
    const auto cmd = actuator.recorded_commands().back();
    TEST_ASSERT(cmd.generated_at_ns == plan.correlation_id.source_timestamp_ns);
    TEST_ASSERT(cmd.correlation_id.sequence_id == 42);
    TEST_ASSERT(cmd.correlation_id.pipeline_run_id == 999);
    TEST_ASSERT(cmd.correlation_id.flags == 7);
    TEST_ASSERT(cmd.desired_apply_time_ns == plan.points[0].target_time_ns);
    // Only two milliseconds late for the point, but eleven milliseconds old
    // for the originating observation: reject the full remainder and release.
    TEST_ASSERT(!scheduler.tick_step(1'011'000'000LL));
    TEST_ASSERT(scheduler.jitter_stats().steps_dropped_expired == 1);
    TEST_ASSERT(!scheduler.tick_step(1'009'000'000LL));
    TEST_ASSERT(scheduler.try_submit_plan(plan));
    actuator.shutdown();
    TEST_ASSERT(!scheduler.tick_step(1'001'000'000LL));
    TEST_ASSERT(scheduler.jitter_stats().steps_dispatched == 1);
    TEST_ASSERT(scheduler.jitter_stats().steps_rejected == 1);
    TEST_ASSERT(actuator.start());
    TEST_ASSERT(scheduler.try_submit_plan(plan));
    ActuationCommand held{};
    held.button_transition = {MouseButton::left, ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(held) == SubmitResult::submitted);
    scheduler.stop();
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);
    TEST_ASSERT(!scheduler.tick_step(1'001'000'000LL));
    plan.point_count = static_cast<std::uint32_t>(kMaxTrajectoryPoints + 1);
    TEST_ASSERT(!scheduler.try_submit_plan(plan));
    plan.point_count = 2;
    plan.correlation_id.source_timestamp_ns = 0;
    TEST_ASSERT(!scheduler.try_submit_plan(plan));
}

void test_synchronous_plan_ticking() {
    std::cout << "[Test 1] Synchronous plan execution and count dispatch..." << std::endl;

    NullActuator actuator{};
    ActuatorConfig act_config{};
    TEST_ASSERT(actuator.initialize(act_config));
    TEST_ASSERT(actuator.start());

    ActuatorScheduler scheduler{&actuator};
    SchedulerConfig sched_config{};
    TEST_ASSERT(scheduler.initialize(sched_config, &actuator));

    // Create a 5-step trajectory plan
    TrajectoryPlan plan{};
    plan.plan_id = 101;
    plan.point_count = 5;
    MonotonicNs sim_time = 1'000'000'000LL;

    for (std::uint32_t i = 0; i < 5; ++i) {
        sim_time += 1'000'000LL; // 1 ms step
        plan.points[i].target_time_ns = sim_time;
        plan.points[i].step_delta_x_counts = static_cast<std::int32_t>(i + 1);
        plan.points[i].step_delta_y_counts = 0;
    }

    stamp_plan(plan);
    scheduler.submit_plan(plan);

    // Tick 5 steps
    MonotonicNs cur_time = 1'000'000'000LL;
    for (std::uint32_t i = 0; i < 5; ++i) {
        cur_time += 1'000'000LL;
        TEST_ASSERT(scheduler.tick_step(cur_time));
    }

    // 6th tick should return false (plan completed)
    TEST_ASSERT(!scheduler.tick_step(cur_time + 1'000'000LL));

    const auto recorded = actuator.recorded_commands();
    TEST_ASSERT(recorded.size() == 5);
    TEST_ASSERT(recorded[0].delta_x_counts == 1);
    TEST_ASSERT(recorded[4].delta_x_counts == 5);

    const auto stats = scheduler.jitter_stats();
    TEST_ASSERT(stats.steps_dispatched == 5);
    TEST_ASSERT(stats.steps_dropped_expired == 0);

    std::cout << "  -> Synchronous plan execution passed." << std::endl;
}

void test_plan_superseding_no_fifo_backlog() {
    std::cout << "[Test 2] Plan superseding (no FIFO backlog)..." << std::endl;

    NullActuator actuator{};
    ActuatorConfig act_config{};
    TEST_ASSERT(actuator.initialize(act_config));
    TEST_ASSERT(actuator.start());

    ActuatorScheduler scheduler{&actuator};
    SchedulerConfig sched_config{};
    TEST_ASSERT(scheduler.initialize(sched_config, &actuator));

    // Submit Plan 1: 10 steps
    TrajectoryPlan plan1{};
    plan1.plan_id = 1;
    plan1.point_count = 10;
    for (std::uint32_t i = 0; i < 10; ++i) {
        plan1.points[i].target_time_ns = 1'000'000'000LL + static_cast<MonotonicNs>(i * 1'000'000LL);
        plan1.points[i].step_delta_x_counts = 10;
    }
    stamp_plan(plan1);
    scheduler.submit_plan(plan1);

    // Execute first 3 steps of Plan 1
    for (std::uint32_t i = 0; i < 3; ++i) {
        TEST_ASSERT(scheduler.tick_step(1'000'000'000LL + static_cast<MonotonicNs>(i * 1'000'000LL)));
    }

    // Submit Plan 2: 4 steps (Supercedes Plan 1!)
    TrajectoryPlan plan2{};
    plan2.plan_id = 2;
    plan2.point_count = 4;
    for (std::uint32_t i = 0; i < 4; ++i) {
        plan2.points[i].target_time_ns = 2'000'000'000LL + static_cast<MonotonicNs>(i * 1'000'000LL);
        plan2.points[i].step_delta_x_counts = 99;
    }
    stamp_plan(plan2);
    scheduler.submit_plan(plan2);

    // Execute all 4 steps of Plan 2
    for (std::uint32_t i = 0; i < 4; ++i) {
        TEST_ASSERT(scheduler.tick_step(2'000'000'000LL + static_cast<MonotonicNs>(i * 1'000'000LL)));
    }

    // Next tick returns false
    TEST_ASSERT(!scheduler.tick_step(2'005'000'000LL));

    const auto recorded = actuator.recorded_commands();
    // Must contain 3 steps from Plan 1 + 4 steps from Plan 2 = 7 steps total (remaining 7 steps of Plan 1 dropped!)
    TEST_ASSERT(recorded.size() == 7);
    TEST_ASSERT(recorded[2].delta_x_counts == 10);
    TEST_ASSERT(recorded[3].delta_x_counts == 99);

    const auto stats = scheduler.jitter_stats();
    TEST_ASSERT(stats.plans_superseded == 1);
    TEST_ASSERT(stats.steps_dispatched == 7);

    std::cout << "  -> Plan superseding passed." << std::endl;
}

void test_live_worker_thread_lifecycle() {
    std::cout << "[Test 3] Live worker thread start and stop lifecycle..." << std::endl;

    NullActuator actuator{};
    ActuatorConfig act_config{};
    TEST_ASSERT(actuator.initialize(act_config));
    TEST_ASSERT(actuator.start());

    ActuatorScheduler scheduler{&actuator};
    SchedulerConfig sched_config{};
    sched_config.target_frequency_hz = 1000;
    TEST_ASSERT(scheduler.initialize(sched_config, &actuator));

    TEST_ASSERT(scheduler.start());
    TEST_ASSERT(scheduler.is_running());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    scheduler.stop();
    TEST_ASSERT(!scheduler.is_running());

    std::cout << "  -> Live worker thread lifecycle passed." << std::endl;
}

void test_stop_cancel_race_conditions() {
    std::cout << "[Test 4] Stop / cancel / submit concurrent race stress..." << std::endl;

    NullActuator actuator{};
    ActuatorConfig act_config{};
    TEST_ASSERT(actuator.initialize(act_config));
    TEST_ASSERT(actuator.start());

    ActuatorScheduler scheduler{&actuator};
    SchedulerConfig sched_config{};
    sched_config.target_frequency_hz = 1000;
    sched_config.enable_busy_spin_fine_sleep = false;
    TEST_ASSERT(scheduler.initialize(sched_config, &actuator));
    TEST_ASSERT(scheduler.start());

    std::atomic<bool> run_stress{true};

    // Thread 1: Submit plans rapidly
    std::thread submitter([&]() {
        std::uint64_t pid = 1;
        while (run_stress.load(std::memory_order_relaxed)) {
            TrajectoryPlan plan{};
            plan.plan_id = pid++;
            plan.point_count = 10;
            for (std::uint32_t i = 0; i < 10; ++i) {
                plan.points[i].target_time_ns = 1'000'000'000LL + static_cast<MonotonicNs>(i * 1'000'000LL);
                plan.points[i].step_delta_x_counts = 5;
            }
            stamp_plan(plan);
    scheduler.submit_plan(plan);
            std::this_thread::yield();
        }
    });

    // Thread 2: Cancel active plan rapidly
    std::thread canceller([&]() {
        while (run_stress.load(std::memory_order_relaxed)) {
            scheduler.cancel_active_plan();
            std::this_thread::yield();
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    run_stress.store(false, std::memory_order_relaxed);

    submitter.join();
    canceller.join();

    scheduler.stop();
    TEST_ASSERT(!scheduler.is_running());

    std::cout << "  -> Stop/cancel race conditions passed." << std::endl;
}

void test_scheduler_jitter_distribution() {
    std::cout << "[Test 5] Deterministic injected-delay histogram (not OS jitter)..." << std::endl;
    NullActuator actuator;
    TEST_ASSERT(actuator.initialize({}));
    TEST_ASSERT(actuator.start());
    ActuatorScheduler scheduler(&actuator);
    for (std::uint32_t i = 0; i < 10000; ++i) {
        TrajectoryPlan plan{};
        plan.plan_id = i + 1;
        plan.point_count = 1;
        plan.points[0].target_time_ns = 1'001'000'000LL + static_cast<MonotonicNs>(i) * 1'000'000LL;
        plan.points[0].step_delta_x_counts = 1;
        stamp_plan(plan);
        TEST_ASSERT(scheduler.try_submit_plan(plan));
        TEST_ASSERT(scheduler.tick_step(plan.points[0].target_time_ns + static_cast<MonotonicNs>(i % 5) * 100'000LL));
    }
    const auto stats = scheduler.jitter_stats();
    TEST_ASSERT(stats.steps_dispatched == 10000);
    TEST_ASSERT(stats.p50_delay_us == 200.0);
    TEST_ASSERT(stats.p95_delay_us == 400.0);
    TEST_ASSERT(stats.p99_delay_us == 400.0);
    TEST_ASSERT(stats.max_delay_us == 400.0);
}
} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M6-02 Actuator Scheduler Unit Tests          " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_due_source_failure_and_sync_stop();
    test_synchronous_plan_ticking();
    test_plan_superseding_no_fifo_backlog();
    test_live_worker_thread_lifecycle();
    test_stop_cancel_race_conditions();
    test_scheduler_jitter_distribution();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M6-02 Actuator Scheduler Tests Passed Successfully!        " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
