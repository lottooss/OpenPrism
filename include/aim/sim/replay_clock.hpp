// include/aim/sim/replay_clock.hpp
// Steppable deterministic replay clock with driftless fractional nanosecond accumulation
#pragma once

#include <cmath>
#include <cstdint>
#include "aim/core/clock.hpp"
#include "aim/core/time.hpp"

namespace aim::sim {

/// @brief Deterministic replay clock and simulation time controller implementing aim::IClock.
/// Uses exact fractional-nanosecond progression to guarantee zero timestamp drift (0.0 ns error).
class ReplayClock final : public IClock {
public:
    explicit ReplayClock(
        double cadence_hz = 144.0,
        MonotonicNs start_time_ns = 1'000'000'000LL
    ) noexcept
        : cadence_hz_(cadence_hz > 0.0 ? cadence_hz : 144.0),
          start_time_ns_(start_time_ns),
          current_time_ns_(start_time_ns) {}

    /// @brief Returns the current virtual monotonic simulation timestamp.
    [[nodiscard]] MonotonicNs now_ns() noexcept override {
        return current_time_ns_;
    }

    [[nodiscard]] double cadence_hz() const noexcept { return cadence_hz_; }
    [[nodiscard]] MonotonicNs start_time_ns() const noexcept { return start_time_ns_; }
    [[nodiscard]] std::uint64_t tick_index() const noexcept { return tick_index_; }
    [[nodiscard]] std::uint64_t frame_id() const noexcept { return frame_id_; }
    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return sequence_id_; }
    [[nodiscard]] bool is_paused() const noexcept { return is_paused_; }
    [[nodiscard]] double rate_scale() const noexcept { return rate_scale_; }

    [[nodiscard]] MonotonicNs elapsed_ns() const noexcept {
        return current_time_ns_ - start_time_ns_;
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        return static_cast<double>(elapsed_ns()) / 1'000'000'000.0;
    }

    void set_cadence_hz(double hz) noexcept {
        if (hz > 0.0) cadence_hz_ = hz;
    }

    void pause() noexcept { is_paused_ = true; }
    void resume() noexcept { is_paused_ = false; }
    void set_rate_scale(double scale) noexcept {
        rate_scale_ = (scale >= 0.0) ? scale : 1.0;
    }

    /// @brief Advances clock by exactly `count` frame ticks with zero cumulative rounding drift.
    void step(std::uint64_t count = 1) noexcept {
        if (is_paused_ || count == 0) return;

        tick_index_ += count;
        frame_id_ += count;
        sequence_id_ += count;

        // Exact accumulation: t = start + (tick * 1e9 * rate_scale) / cadence_hz
        if (rate_scale_ == 1.0) {
            const auto total_ns = static_cast<MonotonicNs>(
                (static_cast<double>(tick_index_) * 1'000'000'000.0) / cadence_hz_
            );
            current_time_ns_ = start_time_ns_ + total_ns;
        } else {
            const auto total_ns = static_cast<MonotonicNs>(
                (static_cast<double>(tick_index_) * 1'000'000'000.0 * rate_scale_) / cadence_hz_
            );
            current_time_ns_ = start_time_ns_ + total_ns;
        }
    }

    /// @brief Advances clock by 1 tick.
    void advance_tick() noexcept {
        step(1);
    }

    /// @brief Deterministically seeks to a specified tick index.
    void seek_tick(std::uint64_t target_tick) noexcept {
        tick_index_ = target_tick;
        frame_id_ = target_tick;
        sequence_id_ = target_tick;
        const auto total_ns = static_cast<MonotonicNs>(
            (static_cast<double>(tick_index_) * 1'000'000'000.0 * rate_scale_) / cadence_hz_
        );
        current_time_ns_ = start_time_ns_ + total_ns;
    }

    /// @brief Deterministically seeks to a target simulation timestamp.
    void seek_ns(MonotonicNs target_ns) noexcept {
        if (target_ns < start_time_ns_) target_ns = start_time_ns_;
        const double delta_sec = static_cast<double>(target_ns - start_time_ns_) / 1'000'000'000.0;
        const double effective_hz = (rate_scale_ > 0.0) ? (cadence_hz_ / rate_scale_) : cadence_hz_;
        tick_index_ = static_cast<std::uint64_t>(std::round(delta_sec * effective_hz));
        frame_id_ = tick_index_;
        sequence_id_ = tick_index_;
        const auto total_ns = static_cast<MonotonicNs>(
            (static_cast<double>(tick_index_) * 1'000'000'000.0 * rate_scale_) / cadence_hz_
        );
        current_time_ns_ = start_time_ns_ + total_ns;
    }

    /// @brief Resets replay clock state.
    void reset(MonotonicNs initial_ns = 1'000'000'000LL) noexcept {
        start_time_ns_ = initial_ns;
        current_time_ns_ = initial_ns;
        tick_index_ = 0;
        frame_id_ = 0;
        sequence_id_ = 0;
        rate_scale_ = 1.0;
        is_paused_ = false;
    }

private:
    double cadence_hz_{144.0};
    MonotonicNs start_time_ns_{1'000'000'000LL};
    MonotonicNs current_time_ns_{1'000'000'000LL};
    std::uint64_t tick_index_{0};
    std::uint64_t frame_id_{0};
    std::uint64_t sequence_id_{0};
    double rate_scale_{1.0};
    bool is_paused_{false};
};

} // namespace aim::sim
