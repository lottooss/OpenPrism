// include/aim/interfaces/tracking_engine.hpp
#pragma once

#include <cstdint>
#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

struct TrackerConfig {
    float gate_distance_px{50.0f};
    float min_init_confidence{0.7f};
    std::uint32_t max_coasting_frames{3};
    float position_process_noise{10.0f};
    float velocity_process_noise{50.0f};
    float measurement_noise{2.0f};
};

struct TrackerHealth {
    std::uint32_t active_tracks{0};
    std::uint64_t total_tracks_created{0};
    std::uint64_t total_tracks_lost{0};
    std::uint64_t total_associations{0};
};

/// @brief Primary tracking interface implementing adaptive Kalman bank and Hungarian gating.
class ITrackingEngine {
public:
    virtual ~ITrackingEngine() = default;
    virtual bool initialize(const TrackerConfig& config) noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual bool update(const bus::TargetObservationBatch& observations,
                        MonotonicNs prediction_target_time_ns,
                        bus::TrackedTargetBatch& out_tracks) noexcept = 0;
    [[nodiscard]] virtual TrackerHealth health() const noexcept = 0;
};

} // namespace aim
