#include "aim/actuation/actuator_scheduler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include "aim/core/clock.hpp"

namespace aim::actuation {
namespace {
constexpr MonotonicNs kHardSourceAgeNs = 10'000'000LL;
bool valid_plan(const TrajectoryPlan& plan) noexcept {
    if (plan.point_count > kMaxTrajectoryPoints) return false;
    if (plan.point_count == 0) return true;
    if (plan.correlation_id.source_timestamp_ns <= 0 ||
        plan.start_time_ns < plan.correlation_id.source_timestamp_ns) return false;
    MonotonicNs previous = plan.start_time_ns;
    for (const auto& point : plan.items()) {
        if (point.target_time_ns < previous || point.target_time_ns <= 0 ||
            !std::isfinite(point.position_px.x) || !std::isfinite(point.position_px.y) ||
            !std::isfinite(point.velocity_px_s.x_per_s) || !std::isfinite(point.velocity_px_s.y_per_s)) return false;
        previous = point.target_time_ns;
    }
    return plan.end_time_ns == previous;
}
} // namespace
ActuatorScheduler::ActuatorScheduler(IActuator* actuator) noexcept : actuator_(actuator) {}
ActuatorScheduler::~ActuatorScheduler() { stop(); }
bool ActuatorScheduler::try_ownership() noexcept {
    std::uint32_t expected = 0;
    return ownership_.compare_exchange_strong(expected, kOwned, std::memory_order_acquire);
}
void ActuatorScheduler::release_ownership() noexcept {
    std::uint32_t expected = kOwned;
    if (ownership_.compare_exchange_strong(expected, 0, std::memory_order_release)) return;
    cancel_owned();
    ownership_.store(0, std::memory_order_release);
}
bool ActuatorScheduler::initialize(const SchedulerConfig& config, IActuator* actuator,
                                   ISafetySupervisor* safety) noexcept {
    stop();
    if (config.target_frequency_hz == 0 || config.target_frequency_hz > 1000 ||
        config.max_source_age_ns <= 0 || config.max_source_age_ns > kHardSourceAgeNs ||
        config.max_step_delay_ns < 0 || config.max_step_delay_ns > kHardSourceAgeNs ||
        !config.drop_expired_steps) return false;
    config_ = config;
    actuator_ = actuator;
    safety_ = safety;
    reset_stats();
    return actuator_ != nullptr;
}
bool ActuatorScheduler::start() noexcept {
    if (is_running()) return true;
    if (!actuator_) return false;
    stop_requested_.store(false, std::memory_order_release);
    try { worker_thread_ = std::thread(&ActuatorScheduler::loop_thread_func, this); }
    catch (...) { return false; }
    is_running_.store(true, std::memory_order_release);
    return true;
}
void ActuatorScheduler::stop() noexcept {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_thread_.joinable()) worker_thread_.join();
    is_running_.store(false, std::memory_order_release);
    cancel_active_plan(); // Also cancels synchronous plans without a worker.
}
void ActuatorScheduler::cancel_owned() noexcept {
    has_active_plan_ = false;
    current_step_index_ = 0;
    if (actuator_) actuator_->cancel_pending();
}
bool ActuatorScheduler::try_submit_plan(const TrajectoryPlan& plan) noexcept {
    if (!valid_plan(plan) || !try_ownership()) {
        ++publication_rejections_;
        cancel_active_plan();
        return false;
    }
    const OwnershipRelease release{*this};
    if (has_active_plan_) ++plans_superseded_;
    active_plan_ = plan;
    current_step_index_ = 0;
    has_active_plan_ = plan.point_count != 0;
    if (!has_active_plan_) cancel_owned();
    return true;
}
void ActuatorScheduler::cancel_active_plan() noexcept {
    ownership_.fetch_or(kCancelRequested, std::memory_order_acq_rel);
    if ((ownership_.fetch_or(kOwned, std::memory_order_acq_rel) & kOwned) != 0) return;
    const OwnershipRelease release{*this};
}
SchedulerTickResult ActuatorScheduler::tick(MonotonicNs now_ns, ActuationCommand& accepted,
                                           std::uint32_t& point_index) noexcept {
    accepted = {};
    if (!try_ownership()) return SchedulerTickResult::idle;
    const OwnershipRelease release{*this};
    if ((ownership_.load(std::memory_order_acquire) & kCancelRequested) != 0) {
        cancel_owned();
        return SchedulerTickResult::rejected;
    }
    if (!has_active_plan_) return SchedulerTickResult::idle;
    const auto& point = active_plan_.points[current_step_index_];
    const auto source_ns = active_plan_.correlation_id.source_timestamp_ns;
    if (now_ns < source_ns || now_ns - source_ns > config_.max_source_age_ns ||
        (now_ns >= point.target_time_ns && now_ns - point.target_time_ns > config_.max_step_delay_ns)) {
        ++steps_dropped_expired_;
        cancel_owned();
        return SchedulerTickResult::expired;
    }
    if (now_ns < point.target_time_ns) return SchedulerTickResult::idle;
    ActuationCommand command{};
    command.sequence_id = reserve_command_sequence();
    command.correlation_id = active_plan_.correlation_id;
    command.generated_at_ns = source_ns;
    command.desired_apply_time_ns = point.target_time_ns;
    command.delta_x_counts = point.step_delta_x_counts;
    command.delta_y_counts = point.step_delta_y_counts;
    const bool zero = command.delta_x_counts == 0 && command.delta_y_counts == 0;
    if (!actuator_ || (safety_ && !safety_->check_actuation_safety(command, now_ns)) ||
        (!zero && actuator_->submit_latest(command) != SubmitResult::submitted)) {
        ++steps_rejected_;
        cancel_owned();
        return SchedulerTickResult::rejected;
    }
    point_index = current_step_index_++;
    has_active_plan_ = current_step_index_ < active_plan_.point_count;
    if (!zero) {
        accepted = command;
        ++steps_dispatched_;
        const auto index = delay_count_.fetch_add(1, std::memory_order_relaxed);
        delays_ns_[index % kMaxRecordedDelays].store(now_ns - point.target_time_ns, std::memory_order_relaxed);
    }
    return zero ? SchedulerTickResult::zero_step : SchedulerTickResult::submitted;
}
bool ActuatorScheduler::tick_step(MonotonicNs now_ns) noexcept {
    ActuationCommand command{};
    std::uint32_t index{};
    const auto result = tick(now_ns, command, index);
    return result == SchedulerTickResult::submitted || result == SchedulerTickResult::zero_step;
}
void ActuatorScheduler::loop_thread_func() noexcept {
    QpcClock clock;
    const auto interval = std::chrono::nanoseconds(1'000'000'000LL / config_.target_frequency_hz);
    auto deadline = std::chrono::steady_clock::now();
    while (!stop_requested_.load(std::memory_order_acquire)) {
        deadline += interval;
        if (config_.enable_busy_spin_fine_sleep) {
            std::this_thread::sleep_until(deadline - std::chrono::microseconds(100));
            while (!stop_requested_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        } else {
            std::this_thread::sleep_until(deadline);
        }
        if (stop_requested_.load(std::memory_order_acquire)) break;
        (void)tick_step(clock.now_ns());
        const auto now = std::chrono::steady_clock::now();
        if (deadline + interval < now) deadline = now;
    }
}
SchedulerJitterStats ActuatorScheduler::jitter_stats() const noexcept {
    SchedulerJitterStats stats{};
    stats.steps_dispatched = steps_dispatched_.load();
    stats.steps_dropped_expired = steps_dropped_expired_.load();
    stats.plans_superseded = plans_superseded_.load();
    stats.steps_rejected = steps_rejected_.load();
    stats.publication_rejections = publication_rejections_.load();
    const auto count = static_cast<std::size_t>((std::min)(delay_count_.load(), static_cast<std::uint64_t>(kMaxRecordedDelays)));
    if (count == 0) return stats;
    std::array<double, kMaxRecordedDelays> sorted{};
    for (std::size_t i = 0; i < count; ++i) sorted[i] = static_cast<double>(delays_ns_[i].load()) / 1000.0;
    std::sort(sorted.begin(), sorted.begin() + count);
    stats.p50_delay_us = sorted[count * 50 / 100];
    stats.p95_delay_us = sorted[count * 95 / 100];
    stats.p99_delay_us = sorted[count * 99 / 100];
    stats.max_delay_us = sorted[count - 1];
    stats.mean_delay_us = std::accumulate(sorted.begin(), sorted.begin() + count, 0.0) / static_cast<double>(count);
    return stats;
}
void ActuatorScheduler::reset_stats() noexcept {
    delay_count_.store(0);
    steps_dispatched_.store(0);
    steps_dropped_expired_.store(0);
    plans_superseded_.store(0);
    steps_rejected_.store(0);
    publication_rejections_.store(0);
}
} // namespace aim::actuation
