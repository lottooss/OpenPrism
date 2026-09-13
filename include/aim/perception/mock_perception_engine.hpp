#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "aim/interfaces/perception_engine.hpp"
#include "aim/perception/tensorrt_engine_runner.hpp"

namespace aim::perception {

struct DecodedDetection {
    float center_x_px{0.0f};
    float center_y_px{0.0f};
    float width_px{0.0f};
    float height_px{0.0f};
    float radius_px{0.0f};
    float confidence{0.0f};
    std::uint32_t class_id{0};
};

/// @brief Explicit deterministic test double. Never used as a production backend.
class MockTensorRtEngineRunner final : public IPerceptionEngine {
public:
    explicit MockTensorRtEngineRunner(TensorRtRunnerConfig config = {}) noexcept;

    [[nodiscard]] ModelContract contract() const noexcept override;
    PerceptionStatus initialize(const ModelManifest& manifest) noexcept override;
    PerceptionStatus warmup(std::uint32_t iterations) noexcept override;
    PerceptionStatus enqueue(const PerceptionRequest& request,
                             InferenceTicket& out_ticket) noexcept override;
    PollResult try_collect(const InferenceTicket& ticket,
                           bus::TargetObservationBatch& out_batch) noexcept override;
    void shutdown() noexcept override;

    [[nodiscard]] bool set_detections(std::span<const DecodedDetection> detections) noexcept;
    void inject_device_fault() noexcept { has_device_fault_ = true; }
    void clear_device_fault() noexcept { has_device_fault_ = false; }

    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }
    [[nodiscard]] bool is_warmed_up() const noexcept { return is_warmed_up_; }
    [[nodiscard]] bool has_device_fault() const noexcept { return has_device_fault_; }
    [[nodiscard]] std::uint64_t total_inferences() const noexcept { return total_inferences_; }
    [[nodiscard]] std::uint64_t total_faults() const noexcept { return total_faults_; }

private:
    TensorRtRunnerConfig config_{};
    std::array<DecodedDetection, bus::kMaxObservations> detections_{};
    std::size_t detection_count_{0};
    InferenceTicket active_ticket_{};
    PerceptionRequest active_request_{};
    std::uint64_t next_ticket_id_{1};
    std::uint64_t total_inferences_{0};
    std::uint64_t total_faults_{0};
    bool is_initialized_{false};
    bool is_warmed_up_{false};
    bool has_device_fault_{false};
    bool pending_inference_{false};
};

} // namespace aim::perception
