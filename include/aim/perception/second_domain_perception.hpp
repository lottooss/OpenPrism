// include/aim/perception/second_domain_perception.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/interfaces/perception_engine.hpp"

namespace aim::perception {

struct HumanoidTargetDef {
    std::uint64_t source_id{1};
    PixelPoint center_px{960.0f, 540.0f};
    float width_px{24.0f};
    float height_px{72.0f}; // 1:3 aspect ratio (tall humanoid silhouette)
    float confidence{0.95f};
    PixelVelocity velocity_px_per_s{0.0f, 0.0f};
    float velocity_confidence{0.8f};
    std::uint32_t semantic_id{2}; // 2 = humanoid / tactical target
};

struct SecondDomainConfig {
    std::string domain_name{"tactical_humanoid"};
    std::uint32_t frame_width{1920};
    std::uint32_t frame_height{1080};
    float target_value{1.0f};
};

/// @brief Perception engine and observation source for the second domain (Milestone M8-02).
/// Emits canonical TargetObservationBatch batches containing non-circular humanoid targets
/// with anisotropic covariance to prove that tracking, policy, and actuation are domain-agnostic.
class SecondDomainObservationSource final : public IObservationSource, public IPerceptionEngine {
public:
    explicit SecondDomainObservationSource(SecondDomainConfig config = {}) noexcept;
    ~SecondDomainObservationSource() override = default;

    // IObservationSource implementation
    bool start() noexcept override;
    bool try_read_latest(bus::TargetObservationBatch& out_batch) noexcept override;
    void stop() noexcept override;

    // IPerceptionEngine implementation
    [[nodiscard]] ModelContract contract() const noexcept override;
    PerceptionStatus initialize(const ModelManifest& manifest) noexcept override;
    PerceptionStatus warmup(std::uint32_t iterations) noexcept override;
    PerceptionStatus enqueue(const PerceptionRequest& request, InferenceTicket& out_ticket) noexcept override;
    PollResult try_collect(const InferenceTicket& ticket, bus::TargetObservationBatch& out_batch) noexcept override;
    void shutdown() noexcept override;

    // Target configuration / injection for testing and replay
    void set_targets(const std::vector<HumanoidTargetDef>& targets);
    void advance_simulation(MonotonicNs step_dt_ns) noexcept;
    void set_current_time_ns(MonotonicNs now_ns) noexcept { current_time_ns_ = now_ns; }

    /// @brief Converts a HumanoidTargetDef into a canonical TargetObservation.
    [[nodiscard]] static bus::TargetObservation make_humanoid_observation(
        const HumanoidTargetDef& def,
        std::uint64_t frame_id,
        MonotonicNs timestamp_ns,
        std::uint32_t width = 1920,
        std::uint32_t height = 1080
    ) noexcept;

private:
    SecondDomainConfig config_;
    std::vector<HumanoidTargetDef> targets_{};
    MonotonicNs current_time_ns_{1'000'000'000LL};
    std::uint64_t current_frame_id_{0};
    std::uint64_t ticket_counter_{0};
    bool is_running_{false};
    bool is_initialized_{false};
};

} // namespace aim::perception
