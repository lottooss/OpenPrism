// tests/cpp/test_core_contracts.cpp
// Native C++20 deterministic unit tests for OpenPrism Core headers

#include <cassert>
#include <iostream>
#include <string_view>
#include "aim/core/actuator.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/stage_timer.hpp"
#include "aim/core/telemetry.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            return 1; \
        } \
    } while (false)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

int test_time_conversions() {
    // 10 MHz standard invariant frequency
    const std::uint64_t qpf_10mhz = 10'000'000ULL;
    const std::uint64_t ticks = 123'456'789ULL;
    const aim::MonotonicNs expected_ns = static_cast<aim::MonotonicNs>(ticks * 100ULL);

    ASSERT_EQ(aim::qpc_to_ns_euclidean(ticks, qpf_10mhz), expected_ns);
    ASSERT_EQ(aim::qpc_to_ns_128(ticks, qpf_10mhz), expected_ns);

    // Zero frequency guard
    ASSERT_EQ(aim::qpc_to_ns_euclidean(ticks, 0), 0);
    ASSERT_EQ(aim::qpc_to_ns_128(ticks, 0), 0);

    // Overflow safety at 10 years uptime: 10 * 365.25 * 86400 * 10,000,000 = 3,155,760,000,000,000 ticks
    const std::uint64_t high_ticks = 3'155'760'000'000'000ULL;
    const aim::MonotonicNs high_ns = aim::qpc_to_ns_euclidean(high_ticks, qpf_10mhz);
    ASSERT_TRUE(high_ns > 0);

    // Non-10 MHz arbitrary frequency
    const std::uint64_t custom_qpf = 3'579'545ULL; // ACPI timer frequency
    const aim::MonotonicNs custom_ns = aim::qpc_to_ns_euclidean(custom_qpf, custom_qpf);
    ASSERT_EQ(custom_ns, 1'000'000'000LL);

    // Unit conversions
    ASSERT_EQ(aim::ms_to_ns(1.5), 1'500'000LL);
    ASSERT_EQ(aim::ns_to_ms(1'500'000LL), 1.5);

    return 0;
}

int test_clocks() {
    // Fake Clock
    aim::FakeClock fake(1'000'000'000LL);
    ASSERT_EQ(fake.now_ns(), 1'000'000'000LL);

    fake.advance_ns(500'000LL);
    ASSERT_EQ(fake.now_ns(), 1'000'500'000LL);

    fake.advance_ms(1.5);
    ASSERT_EQ(fake.now_ns(), 1'002'000'000LL);

    fake.set_ns(2'000'000'000LL);
    ASSERT_EQ(fake.now_ns(), 2'000'000'000LL);

    // Real QPC Clock
    aim::QpcClock qpc;
    const auto t0 = qpc.now_ns();
    const auto t1 = qpc.now_ns();
    ASSERT_TRUE(t1 >= t0);

    return 0;
}

int test_stage_timer_and_buffer() {
    aim::FakeClock fake(1'000'000'000LL);
    aim::FixedTelemetryBuffer<4> buffer;
    aim::CorrelationId cid{
        .sequence_id = 42,
        .source_timestamp_ns = 1'000'000'000LL,
        .pipeline_run_id = 1,
        .flags = static_cast<std::uint32_t>(aim::CorrelationFlags::synthetic)
    };

    ASSERT_EQ(buffer.size(), 0);

    {
        aim::ScopedStageTimer<4> timer(aim::PipelineStage::perception_infer, cid, fake, buffer);
        fake.advance_ns(2'500'000LL); // 2.5 ms
    }

    ASSERT_EQ(buffer.size(), 1);
    const auto events = buffer.events();
    ASSERT_EQ(events.size(), 1);
    ASSERT_EQ(events[0].duration_ns(), 2'500'000LL);
    ASSERT_EQ(events[0].duration_ms(), 2.5);

    // Fill buffer to capacity
    for (int i = 0; i < 5; ++i) {
        aim::StageTimestampEvent evt{
            .correlation_id = cid,
            .stage = aim::PipelineStage::tracking_kalman,
            .start_ns = 1'000'000LL,
            .end_ns = 2'000'000LL
        };
        buffer.record(evt);
    }

    ASSERT_EQ(buffer.size(), 4); // Saturated without allocation or overflow
    return 0;
}

int test_safety_and_null_actuator() {
    aim::NullActuator actuator;
    aim::ActuatorConfig config{
        .backend = "null",
        .scheduler_hz = 1000,
        .relative_counts = true,
        .cancel_superseded = true,
        .require_emergency_stop = true
    };

    aim::CorrelationId cid{
        .sequence_id = 100,
        .source_timestamp_ns = 1'000'000'000LL,
        .pipeline_run_id = 1,
        .flags = static_cast<std::uint32_t>(aim::CorrelationFlags::none)
    };

    aim::ActuationCommand cmd{
        .sequence_id = 100,
        .correlation_id = cid,
        .generated_at_ns = 1'000'000'000LL,
        .desired_apply_time_ns = 1'000'500'000LL,
        .delta_x_counts = 15,
        .delta_y_counts = -10,
        .button_transition = {
            .button = aim::MouseButton::left,
            .action = aim::ButtonAction::press
        }
    };

    // Uninitialized submit must be rejected
    ASSERT_EQ(actuator.submit_latest(cmd), aim::SubmitResult::rejected_uninitialized);

    ASSERT_TRUE(actuator.initialize(config));
    ASSERT_TRUE(actuator.start());

    // Submit valid command
    ASSERT_EQ(actuator.submit_latest(cmd), aim::SubmitResult::submitted);
    const auto records = actuator.recorded_commands();
    ASSERT_EQ(records.size(), 1);
    ASSERT_EQ(records[0].delta_x_counts, 15);
    ASSERT_EQ(records[0].delta_y_counts, -10);

    const auto h1 = actuator.health();
    ASSERT_TRUE(h1.is_active);
    ASSERT_FALSE(h1.is_latched);
    ASSERT_EQ(h1.pressed_buttons_mask, (1u << static_cast<std::uint32_t>(aim::MouseButton::left)));

    // Trigger Emergency Stop
    actuator.emergency_stop(aim::SafetyReason::emergency_stop_triggered);

    const auto h2 = actuator.health();
    ASSERT_TRUE(h2.is_latched);
    ASSERT_EQ(h2.pressed_buttons_mask, 0u); // Fail-safe: button mask cleared immediately

    // Start must fail while latched
    ASSERT_FALSE(actuator.start());

    // Subsequent submissions must be rejected
    ASSERT_EQ(actuator.submit_latest(cmd), aim::SubmitResult::rejected_latched);

    // Reset with invalid token must fail
    aim::ResetToken bad_token{.token_id = "", .generated_at_ns = 0, .is_valid = true};
    ASSERT_FALSE(actuator.reset_emergency_stop(bad_token));
    ASSERT_TRUE(actuator.health().is_latched);

    // Reset with valid token must succeed
    aim::ResetToken good_token{.token_id = "TOKEN_VALID_OK", .generated_at_ns = 1'000LL, .is_valid = true};
    ASSERT_TRUE(actuator.reset_emergency_stop(good_token));
    ASSERT_FALSE(actuator.health().is_latched);

    actuator.shutdown();
    ASSERT_FALSE(actuator.health().is_active);

    return 0;
}

int test_frame_lease_raii() {
    aim::FrameLease empty;
    ASSERT_FALSE(empty.is_valid());

    aim::FrameLease lease(
        101,
        1'000'000'000LL,
        1920,
        1080,
        aim::FrameFormat::b8g8r8a8_unorm,
        reinterpret_cast<void*>(0xDEADBEEFULL),
        2
    );

    ASSERT_TRUE(lease.is_valid());
    ASSERT_EQ(lease.frame_id(), 101);
    ASSERT_EQ(lease.width_px(), 1920);
    ASSERT_EQ(lease.height_px(), 1080);
    ASSERT_EQ(lease.pool_slot_index(), 2);

    // Test move construction
    aim::FrameLease moved(std::move(lease));
    ASSERT_TRUE(moved.is_valid());
    ASSERT_FALSE(lease.is_valid());
    ASSERT_EQ(moved.frame_id(), 101);

    return 0;
}

int main() {
    std::cout << "Running Native C++20 Core Contracts Test Suite..." << std::endl;

    if (test_time_conversions() != 0) return 1;
    if (test_clocks() != 0) return 1;
    if (test_stage_timer_and_buffer() != 0) return 1;
    if (test_safety_and_null_actuator() != 0) return 1;
    if (test_frame_lease_raii() != 0) return 1;

    std::cout << "All C++20 core contract tests PASSED successfully." << std::endl;
    return 0;
}
