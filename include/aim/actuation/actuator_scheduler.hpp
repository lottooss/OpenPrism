#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include "aim/core/actuator.hpp"
#include "aim/interfaces/safety_supervisor.hpp"
#include "aim/interfaces/trajectory_planner.hpp"

namespace aim::actuation {
struct SchedulerConfig {
    std::uint32_t target_frequency_hz{1000};
    MonotonicNs max_step_delay_ns{5'000'000LL};
    bool drop_expired_steps{true};
    bool enable_busy_spin_fine_sleep{false};
    // May tighten the hard cutoff to reserve time for validation/OS dispatch.
    MonotonicNs max_source_age_ns{10'000'000LL};
};
struct SchedulerJitterStats {
    double p50_delay_us{0.0};
    double p95_delay_us{0.0};
    double p99_delay_us{0.0};
    double max_delay_us{0.0};
    double mean_delay_us{0.0};
    std::uint64_t steps_dispatched{0};
    std::uint64_t steps_dropped_expired{0};
    std::uint64_t plans_superseded{0};
    std::uint64_t steps_rejected{0};
    std::uint64_t publication_rejections{0};
};
enum class SchedulerTickResult : std::uint8_t { idle, zero_step, submitted, rejected, expired };

/// Bounded latest-plan scheduler. One tick owner (worker OR synchronous host).
/// Publication/cancellation may be concurrent. Ownership uses one bounded atomic
/// attempt: contention cancels old work instead of waiting or queueing it.
/// An already-entered backend dispatch completes before cancellation is applied.
class ActuatorScheduler {
public:
    explicit ActuatorScheduler(IActuator* actuator = nullptr) noexcept;
    ~ActuatorScheduler();
    bool initialize(const SchedulerConfig& config, IActuator* actuator,
                    ISafetySupervisor* safety = nullptr) noexcept;
    bool start() noexcept;
    void stop() noexcept;
    [[nodiscard]] bool try_submit_plan(const TrajectoryPlan& plan) noexcept;
    void submit_plan(const TrajectoryPlan& plan) noexcept { (void)try_submit_plan(plan); }
    void cancel_active_plan() noexcept;
    bool tick_step(MonotonicNs current_time_ns) noexcept;
    SchedulerTickResult tick(MonotonicNs now_ns, ActuationCommand& accepted_command,
                             std::uint32_t& point_index) noexcept;
    [[nodiscard]] bool is_running() const noexcept { return is_running_.load(std::memory_order_acquire); }
    [[nodiscard]] SchedulerJitterStats jitter_stats() const noexcept;
    SequenceId reserve_command_sequence() noexcept { return next_command_sequence_.fetch_add(1) + 1; }
    void reset_stats() noexcept;
private:
    static constexpr std::uint32_t kOwned = 1u;
    static constexpr std::uint32_t kCancelRequested = 2u;
    struct OwnershipRelease {
        ActuatorScheduler& owner;
        ~OwnershipRelease() { owner.release_ownership(); }
    };
    bool try_ownership() noexcept;
    void release_ownership() noexcept;
    void loop_thread_func() noexcept;
    void cancel_owned() noexcept;
    SchedulerConfig config_{};
    // Non-owning; host keeps dependencies alive through stop()/destruction.
    IActuator* actuator_{nullptr};
    ISafetySupervisor* safety_{nullptr};
    std::atomic<bool> is_running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<std::uint32_t> ownership_{0};
    TrajectoryPlan active_plan_{};
    std::uint32_t current_step_index_{0};
    bool has_active_plan_{false};
    std::atomic<SequenceId> next_command_sequence_{0};
    static constexpr std::size_t kMaxRecordedDelays = 8192;
    std::array<std::atomic<MonotonicNs>, kMaxRecordedDelays> delays_ns_{};
    std::atomic<std::uint64_t> delay_count_{0};
    std::atomic<std::uint64_t> steps_dispatched_{0};
    std::atomic<std::uint64_t> steps_dropped_expired_{0};
    std::atomic<std::uint64_t> plans_superseded_{0};
    std::atomic<std::uint64_t> steps_rejected_{0};
    std::atomic<std::uint64_t> publication_rejections_{0};
    std::thread worker_thread_{};
};
} // namespace aim::actuation
