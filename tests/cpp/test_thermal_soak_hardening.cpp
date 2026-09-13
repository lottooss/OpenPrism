// tests/cpp/test_thermal_soak_hardening.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#include "aim/core/actuator.hpp"
#include "aim/pipeline/closed_loop_pipeline.hpp"
#include "aim/policy/utility_policy.hpp"
#include "aim/safety/safety_supervisor.hpp"
#include "aim/scenario/scenario_adapter.hpp"
#include "aim/tracking/tracking_engine.hpp"
#include "aim/trajectory/trajectory_planner.hpp"

using namespace aim;
using namespace aim::pipeline;
using namespace aim::policy;
using namespace aim::safety;
using namespace aim::scenario;
using namespace aim::tracking;
using namespace aim::trajectory;
using namespace aim::calibration;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

std::size_t get_process_working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M7-04 Thermal Soak & Hardening Benchmark     " << std::endl;
    std::cout << "================================================================" << std::endl;

    TrackingEngine tracker{};
    aim::TrackerConfig t_cfg{};
    TEST_ASSERT(tracker.initialize(t_cfg));

    UtilityAimPolicy policy{};
    PolicyManifest p_man{};
    TEST_ASSERT(policy.initialize(p_man));

    TrajectoryPlanner planner{};
    NullActuator actuator{};
    ActuatorConfig a_cfg{};
    TEST_ASSERT(actuator.initialize(a_cfg));
    TEST_ASSERT(actuator.start());

    SafetySupervisorConfig s_cfg{};
    s_cfg.authorized_reset_token = "soak_authorized_token";
    SafetySupervisor safety{s_cfg};
    // Explicit simulated environment; production requires real callbacks.
    safety.set_focus_callback([] { return true; });
    safety.set_calibration_callback([] { return true; });

    ScenarioAdapter scenario{};
    TEST_ASSERT(scenario.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);

    CalibrationProfile cal_prof{};
    CalibrationModel cal_model{cal_prof};

    ClosedLoopPipeline pipeline{
        &tracker, &policy, &planner, &actuator, &safety, &scenario, cal_model
    };

    const std::size_t kWarmupFrames = 1000;
    const std::size_t kSoakFrames = 25000;

    MonotonicNs sim_time_ns = 1'000'000'000LL;
    ObservableScenarioState state{};
    state.foreground_confirmed = true;

    bus::TargetObservationBatch batch{};
    batch.captured_at_ns = sim_time_ns;
    bus::TargetObservation obs{};
    obs.source_id = 1;
    obs.center_px = PixelPoint{960.0f, 540.0f};
    obs.effective_radius_px = 15.0f;
    obs.confidence = 0.99f;
    obs.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
    batch.add_target(obs);

    // Warmup
    for (std::size_t i = 0; i < kWarmupFrames; ++i) {
        sim_time_ns += 1'000'000LL;
        batch.captured_at_ns = sim_time_ns;
        pipeline.process_frame(batch, state, sim_time_ns + 100'000LL);
    }

    const std::size_t initial_memory = get_process_working_set_bytes();
    std::cout << "  Post-warmup Working Set: " << (initial_memory / 1024 / 1024) << " MB" << std::endl;

    std::vector<double> latencies_us;
    latencies_us.reserve(kSoakFrames);

    std::uint64_t injected_faults = 0;
    std::uint64_t recovered_faults = 0;

    for (std::size_t i = 0; i < kSoakFrames; ++i) {
        sim_time_ns += 1'000'000LL;
        batch.header.sequence_id = i + 1;
        batch.header.source_timestamp_ns = sim_time_ns;
        batch.captured_at_ns = sim_time_ns;

        // Periodic fault injection (every 5000 frames)
        if (i % 5000 == 1000) {
            safety.trigger_emergency_stop(SafetyReason::operator_requested);
            injected_faults++;
        }

        // Periodic recovery & revalidation (100 frames after fault)
        if (i % 5000 == 1100) {
            TEST_ASSERT(safety.is_latched()); // Must remain disabled until authorized reset!
            ResetToken token{.token_id = "soak_authorized_token", .is_valid = true};
            TEST_ASSERT(safety.try_reset(token));
            TEST_ASSERT(!safety.is_latched());
            recovered_faults++;
        }

        const auto t0 = std::chrono::steady_clock::now();
        pipeline.process_frame(batch, state, sim_time_ns + 100'000LL);
        const auto t1 = std::chrono::steady_clock::now();

        const double dur_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies_us.push_back(dur_us);
    }

    const std::size_t final_memory = get_process_working_set_bytes();
    const double memory_growth_mb = (final_memory > initial_memory)
        ? static_cast<double>(final_memory - initial_memory) / (1024.0 * 1024.0)
        : 0.0;

    std::sort(latencies_us.begin(), latencies_us.end());
    const double p50_us = latencies_us[static_cast<std::size_t>(latencies_us.size() * 0.50)];
    const double p95_us = latencies_us[static_cast<std::size_t>(latencies_us.size() * 0.95)];
    const double p99_us = latencies_us[static_cast<std::size_t>(latencies_us.size() * 0.99)];
    const double max_us = latencies_us.back();

    std::cout << "  Sustained Soak Summary (" << kSoakFrames << " continuous iterations):" << std::endl;
    std::cout << "    Initial Memory:          " << (initial_memory / 1024 / 1024) << " MB" << std::endl;
    std::cout << "    Final Memory:            " << (final_memory / 1024 / 1024) << " MB" << std::endl;
    std::cout << "    Memory Growth:           " << memory_growth_mb << " MB (budget <= 5.0 MB)" << std::endl;
    std::cout << "    Injected Faults:         " << injected_faults << std::endl;
    std::cout << "    Recovered Faults:        " << recovered_faults << std::endl;
    std::cout << std::endl;
    std::cout << "  Soak Latency Distribution:" << std::endl;
    std::cout << "    p50 Latency:             " << p50_us << " us" << std::endl;
    std::cout << "    p95 Latency:             " << p95_us << " us" << std::endl;
    std::cout << "    p99 Latency:             " << p99_us << " us (" << (p99_us / 1000.0) << " ms)" << std::endl;
    std::cout << "    Max Latency:             " << max_us << " us (" << (max_us / 1000.0) << " ms)" << std::endl;

    // Hardening assertions
    TEST_ASSERT(memory_growth_mb <= 5.0);
    TEST_ASSERT(p99_us <= 6000.0);
    TEST_ASSERT(max_us <= 10000.0);
    TEST_ASSERT(injected_faults == 5);
    TEST_ASSERT(recovered_faults == 5);

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M7-04 Thermal Soak & Hardening Tests Passed!     " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
