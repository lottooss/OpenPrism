// M1 exit gate: canonical observations drive every downstream module without images.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "aim/core/actuator.hpp"
#include "aim/interfaces/aim_policy.hpp"
#include "aim/interfaces/tracking_engine.hpp"
#include "aim/interfaces/trajectory_planner.hpp"
#include "aim/sim/sim.hpp"

namespace {

using namespace aim;

constexpr std::size_t kFrameCount = 32;

// Recorded field-canonical FNV-1a digest of the 32 commands the fixed-seed pipeline emits.
// Verified identical under GCC 15.2 at -O0/-O2/-O3/-march=native/-ffast-math. Comparing two
// in-process runs only proves the run is repeatable; pinning the value is what detects a
// change in simulator, tracker, policy, planner, or toolchain behaviour between builds.
constexpr std::uint64_t kExpectedCommandDigest = 0xb6731de8286178d2ULL;

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "[-] ASSERTION FAILED: " << message << " (" << __FILE__ << ':' << __LINE__ << ")\n"; \
            return false; \
        } \
    } while (false)

class ReferenceTracker final : public ITrackingEngine {
public:
    bool initialize(const TrackerConfig& config) noexcept override {
        static_cast<void>(config);
        initialized_ = true;
        health_ = {};
        return true;
    }

    void reset() noexcept override {
        health_ = {};
    }

    bool update(const bus::TargetObservationBatch& observations,
                MonotonicNs prediction_target_time_ns,
                bus::TrackedTargetBatch& out_tracks) noexcept override {
        if (!initialized_ || prediction_target_time_ns < observations.captured_at_ns) {
            return false;
        }

        out_tracks.clear();
        out_tracks.header = observations.header;
        out_tracks.frame_id = observations.frame_id;
        out_tracks.timestamp_ns = prediction_target_time_ns;

        for (const auto& observation : observations.items()) {
            const double prediction_seconds = static_cast<double>(
                prediction_target_time_ns - observation.captured_at_ns) / 1'000'000'000.0;
            const auto prediction_seconds_f = static_cast<float>(prediction_seconds);

            bus::TrackedTarget track{};
            track.track_id = observation.source_id;
            track.state = observation.visibility == Visibility::predicted
                ? bus::TrackState::occluded
                : bus::TrackState::confirmed;
            track.total_visible_frames = observation.visibility == Visibility::predicted ? 0U : 1U;
            track.total_missed_frames = observation.visibility == Visibility::predicted ? 1U : 0U;
            track.filtered_center_px = observation.center_px;
            track.filtered_velocity_px_per_s = observation.velocity.pixels_per_second;
            track.covariance_px2 = observation.covariance_px2;
            track.predicted_center_px = PixelPoint{
                observation.center_px.x +
                    (observation.velocity.pixels_per_second.x_per_s * prediction_seconds_f),
                observation.center_px.y +
                    (observation.velocity.pixels_per_second.y_per_s * prediction_seconds_f)};
            track.prediction_time_ns = prediction_target_time_ns;
            track.effective_radius_px = observation.effective_radius_px;
            track.confidence = observation.confidence;
            track.target_value = observation.target_value;
            track.semantic_id = observation.semantic_id;

            if (!out_tracks.add_track(track)) {
                break;
            }
        }

        health_.active_tracks = out_tracks.track_count;
        health_.total_tracks_created += out_tracks.track_count;
        health_.total_associations += out_tracks.track_count;
        return true;
    }

    [[nodiscard]] TrackerHealth health() const noexcept override {
        return health_;
    }

private:
    bool initialized_{false};
    TrackerHealth health_{};
};

class ReferencePolicy final : public IAimPolicy {
public:
    [[nodiscard]] PolicyContract contract() const noexcept override {
        return PolicyContract{
            .policy_name = "m1_reference_policy",
            .version = "1.0.0",
            .fov_horizontal_deg = 103.0F,
            .max_angular_speed_deg_s = 720.0F};
    }

    bool initialize(const PolicyManifest& manifest) noexcept override {
        static_cast<void>(manifest);
        initialized_ = true;
        return true;
    }

    bool choose(const PolicyInput& input, bus::AimIntent& out_intent) noexcept override {
        if (!initialized_ || input.tracks == nullptr || input.tracks->track_count == 0U) {
            return false;
        }

        const bus::TrackedTarget* selected = &input.tracks->tracks[0];
        for (std::size_t index = 1; index < input.tracks->track_count; ++index) {
            const auto& candidate = input.tracks->tracks[index];
            const float candidate_utility = candidate.confidence * candidate.target_value;
            const float selected_utility = selected->confidence * selected->target_value;
            if (candidate_utility > selected_utility ||
                (candidate_utility == selected_utility && candidate.track_id < selected->track_id)) {
                selected = &candidate;
            }
        }

        const float error_x = selected->predicted_center_px.x - input.crosshair.center_px.x;
        const float error_y = selected->predicted_center_px.y - input.crosshair.center_px.y;

        out_intent = {};
        out_intent.header = input.correlation_id;
        out_intent.target_track_id = selected->track_id;
        out_intent.mode = bus::AimMode::tracking;
        out_intent.target_aim_px = selected->predicted_center_px;
        out_intent.target_aim_norm = sim::pixel_to_normalized(selected->predicted_center_px);
        out_intent.error_distance_px = std::sqrt((error_x * error_x) + (error_y * error_y));
        out_intent.authorize_fire = false;
        out_intent.confidence = selected->confidence;
        out_intent.utility_score = selected->confidence * selected->target_value;
        out_intent.command_deadline_ns = input.decision_time_ns + 10'000'000LL;
        return true;
    }

    void reset() noexcept override {}

private:
    bool initialized_{false};
};

class ReferencePlanner final : public ITrajectoryPlanner {
public:
    bool plan(const bus::AimIntent& intent,
              const ControlState& state,
              const PlannerLimits& limits,
              TrajectoryPlan& out_plan) noexcept override {
        static_cast<void>(limits);
        const float delta_x = intent.target_aim_px.x - state.current_crosshair_px.x;
        const float delta_y = intent.target_aim_px.y - state.current_crosshair_px.y;

        out_plan = {};
        out_plan.plan_id = ++next_plan_id_;
        out_plan.correlation_id = intent.header;
        out_plan.start_time_ns = intent.header.source_timestamp_ns + 3'000'000LL;
        out_plan.end_time_ns = out_plan.start_time_ns + 1'000'000LL;
        out_plan.point_count = 1;
        out_plan.points[0] = TrajectoryPoint{
            .target_time_ns = out_plan.end_time_ns,
            .position_px = intent.target_aim_px,
            .velocity_px_s = PixelVelocity{delta_x * 1'000.0F, delta_y * 1'000.0F},
            .step_delta_x_counts = static_cast<std::int32_t>(std::lround(delta_x)),
            .step_delta_y_counts = static_cast<std::int32_t>(std::lround(delta_y))};
        return true;
    }

    void cancel() noexcept override {}

    void reset() noexcept override {
        next_plan_id_ = 0;
    }

private:
    SequenceId next_plan_id_{0};
};

[[nodiscard]] ActuationCommand command_from_plan(const TrajectoryPlan& plan) noexcept {
    const auto& point = plan.points[0];
    return ActuationCommand{
        .sequence_id = plan.plan_id,
        .correlation_id = plan.correlation_id,
        .generated_at_ns = plan.start_time_ns,
        .desired_apply_time_ns = point.target_time_ns,
        .delta_x_counts = point.step_delta_x_counts,
        .delta_y_counts = point.step_delta_y_counts,
        .button_transition = ButtonTransition{MouseButton::none, ButtonAction::none}};
}

[[nodiscard]] bool commands_equal(const ActuationCommand& lhs, const ActuationCommand& rhs) noexcept {
    return lhs.sequence_id == rhs.sequence_id &&
        lhs.correlation_id.sequence_id == rhs.correlation_id.sequence_id &&
        lhs.correlation_id.source_timestamp_ns == rhs.correlation_id.source_timestamp_ns &&
        lhs.correlation_id.pipeline_run_id == rhs.correlation_id.pipeline_run_id &&
        lhs.correlation_id.flags == rhs.correlation_id.flags &&
        lhs.generated_at_ns == rhs.generated_at_ns &&
        lhs.desired_apply_time_ns == rhs.desired_apply_time_ns &&
        lhs.delta_x_counts == rhs.delta_x_counts &&
        lhs.delta_y_counts == rhs.delta_y_counts &&
        lhs.button_transition.button == rhs.button_transition.button &&
        lhs.button_transition.action == rhs.button_transition.action;
}

void hash_u64(std::uint64_t value, std::uint64_t& hash) noexcept {
    constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;
    for (std::size_t byte_index = 0; byte_index < sizeof(value); ++byte_index) {
        hash ^= value & 0xFFU;
        hash *= kFnvPrime;
        value >>= 8U;
    }
}

[[nodiscard]] std::uint64_t command_digest(
    const std::array<ActuationCommand, kFrameCount>& commands) noexcept {
    std::uint64_t hash = 14'695'981'039'346'656'037ULL;
    for (const auto& command : commands) {
        hash_u64(command.sequence_id, hash);
        hash_u64(command.correlation_id.sequence_id, hash);
        hash_u64(static_cast<std::uint64_t>(command.correlation_id.source_timestamp_ns), hash);
        hash_u64(command.correlation_id.pipeline_run_id, hash);
        hash_u64(command.correlation_id.flags, hash);
        hash_u64(static_cast<std::uint64_t>(command.generated_at_ns), hash);
        hash_u64(static_cast<std::uint64_t>(command.desired_apply_time_ns), hash);
        hash_u64(static_cast<std::uint32_t>(command.delta_x_counts), hash);
        hash_u64(static_cast<std::uint32_t>(command.delta_y_counts), hash);
        hash_u64(static_cast<std::uint8_t>(command.button_transition.button), hash);
        hash_u64(static_cast<std::uint8_t>(command.button_transition.action), hash);
    }
    return hash;
}

bool run_reference_pipeline(std::array<ActuationCommand, kFrameCount>& output_commands) {
    const auto scenario = sim::ScenarioBuilder::make_strafe_track_preset(0xA11CEULL);
    sim::TargetSimulator simulator(scenario);
    ReferenceTracker tracker;
    ReferencePolicy policy;
    ReferencePlanner planner;
    NullActuator actuator;

    TEST_ASSERT(tracker.initialize(TrackerConfig{}), "Reference tracker initialization failed");
    TEST_ASSERT(policy.initialize(PolicyManifest{}), "Reference policy initialization failed");
    TEST_ASSERT(actuator.initialize(ActuatorConfig{}), "NullActuator initialization failed");
    TEST_ASSERT(actuator.start(), "NullActuator start failed");

    ControlState control_state{};
    MonotonicNs previous_apply_time_ns = 0;
    std::size_t nonzero_motion_count = 0;

    for (std::size_t frame_index = 0; frame_index < kFrameCount; ++frame_index) {
        const bus::TargetObservationBatch observations = simulator.step();
        TEST_ASSERT(observations.target_count > 0U, "Simulator produced an empty canonical batch");

        bus::TrackedTargetBatch tracks{};
        const MonotonicNs prediction_time_ns = observations.published_at_ns + 500'000LL;
        TEST_ASSERT(
            tracker.update(observations, prediction_time_ns, tracks),
            "Reference tracker rejected canonical observations");
        TEST_ASSERT(tracks.header.sequence_id == observations.header.sequence_id, "Tracker lost sequence ID");
        TEST_ASSERT(tracks.header.source_timestamp_ns == observations.header.source_timestamp_ns,
                    "Tracker lost source timestamp");

        const PolicyInput policy_input{
            .correlation_id = tracks.header,
            .decision_time_ns = prediction_time_ns,
            .crosshair = CrosshairState{
                .center_px = control_state.current_crosshair_px,
                .center_norm = sim::pixel_to_normalized(control_state.current_crosshair_px),
                .is_recoil_active = false},
            .tracks = &tracks};

        bus::AimIntent intent{};
        TEST_ASSERT(policy.choose(policy_input, intent), "Reference policy did not produce an intent");
        TEST_ASSERT(intent.header.sequence_id == observations.header.sequence_id, "Policy lost sequence ID");
        TEST_ASSERT(!intent.authorize_fire, "Reference policy must remain non-firing");

        TrajectoryPlan plan{};
        TEST_ASSERT(planner.plan(intent, control_state, PlannerLimits{}, plan), "Reference planner failed");
        TEST_ASSERT(plan.point_count == 1U, "Reference plan must contain one complete point");
        TEST_ASSERT(plan.correlation_id.sequence_id == observations.header.sequence_id,
                    "Planner lost sequence ID");

        const ActuationCommand command = command_from_plan(plan);
        TEST_ASSERT(command.correlation_id.is_synthetic(), "Synthetic provenance was lost");
        TEST_ASSERT(command.desired_apply_time_ns > previous_apply_time_ns,
                    "Command apply timestamps must be strictly monotonic");
        TEST_ASSERT(actuator.submit_latest(command) == SubmitResult::submitted,
                    "NullActuator rejected a complete synthetic command");

        output_commands[frame_index] = command;
        previous_apply_time_ns = command.desired_apply_time_ns;
        if (command.delta_x_counts != 0 || command.delta_y_counts != 0) {
            ++nonzero_motion_count;
        }
        control_state.current_crosshair_px = plan.points[0].position_px;
        control_state.current_velocity_px_s = plan.points[0].velocity_px_s;
    }

    const auto recorded_commands = actuator.recorded_commands();
    TEST_ASSERT(recorded_commands.size() == kFrameCount, "NullActuator did not record every plan");
    for (std::size_t index = 0; index < kFrameCount; ++index) {
        TEST_ASSERT(commands_equal(recorded_commands[index], output_commands[index]),
                    "NullActuator command differs from submitted plan");
    }
    TEST_ASSERT(nonzero_motion_count > 0U, "Synthetic pipeline produced no aiming motion");
    TEST_ASSERT(actuator.health().total_commands_submitted == kFrameCount,
                "NullActuator health count is incomplete");
    actuator.shutdown();
    return true;
}

bool test_image_free_pipeline_exit_gate() {
    std::cout << "[RUN] TC-SIM-09: Driving tracker -> policy -> planner -> NullActuator without images...\n";

    std::array<ActuationCommand, kFrameCount> first_run{};
    std::array<ActuationCommand, kFrameCount> second_run{};
    TEST_ASSERT(run_reference_pipeline(first_run), "First deterministic pipeline run failed");
    TEST_ASSERT(run_reference_pipeline(second_run), "Second deterministic pipeline run failed");

    for (std::size_t index = 0; index < kFrameCount; ++index) {
        TEST_ASSERT(commands_equal(first_run[index], second_run[index]),
                    "Fixed-seed pipeline output is not byte-field stable");
        TEST_ASSERT(first_run[index].sequence_id == index + 1U, "Plan sequence is not contiguous");
        TEST_ASSERT(first_run[index].correlation_id.sequence_id == index + 1U,
                    "Canonical correlation sequence is not contiguous");
    }

    const std::uint64_t first_digest = command_digest(first_run);
    const std::uint64_t second_digest = command_digest(second_run);
    TEST_ASSERT(first_digest == second_digest, "Fixed-seed command digest changed between runs");

    std::cout << "  [*] Frames delivered: " << kFrameCount << '\n';
    std::cout << "  [*] Stable command digest (FNV-1a): 0x" << std::hex << first_digest << std::dec << '\n';

    if (first_digest != kExpectedCommandDigest) {
        std::cerr << "[-] ASSERTION FAILED: command digest does not match the recorded value\n"
                  << "      expected 0x" << std::hex << kExpectedCommandDigest
                  << "\n      actual   0x" << first_digest << std::dec << '\n'
                  << "    Either a pipeline stage changed behaviour, or this toolchain does not\n"
                  << "    reproduce the recorded run. Both are findings; neither is noise.\n";
        return false;
    }
    std::cout << "  [+] Image-free canonical pipeline and NullActuator delivery verified.\n";
    return true;
}

} // namespace

int main() {
    std::cout << "===============================================================\n";
    std::cout << "M1 Image-Free Canonical Pipeline Exit Gate\n";
    std::cout << "===============================================================\n";

    if (!test_image_free_pipeline_exit_gate()) {
        std::cerr << "[-] M1 IMAGE-FREE PIPELINE EXIT GATE FAILED\n";
        return 1;
    }

    std::cout << "[+] M1 IMAGE-FREE PIPELINE EXIT GATE PASSED\n";
    return 0;
}
