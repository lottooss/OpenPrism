#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "aim/interfaces/perception_engine.hpp"
#include "aim/perception/engine_artifact.hpp"

namespace aim::perception {

struct TensorRtRunnerConfig {
    std::string model_name{"yolo11n_aimlabs_reference"};
    std::string model_version{"1.0.0"};
    std::uint32_t input_width{640};
    std::uint32_t input_height{384};
    std::uint32_t input_channels{3};
    std::uint32_t num_anchors{5040};
    std::uint32_t num_classes{1};
    std::uint32_t max_detections{64};
    float confidence_threshold{0.25f};
    float nms_iou_threshold{0.45f};
    bool enable_cuda_graph{true};
    std::size_t max_engine_size_bytes{256U * 1024U * 1024U};
};

/// @brief Production TensorRT engine boundary. It never provides simulated inference.
class TensorRtEngineRunner final : public IPerceptionEngine {
public:
    explicit TensorRtEngineRunner(TensorRtRunnerConfig config = {});
    ~TensorRtEngineRunner() override;

    TensorRtEngineRunner(const TensorRtEngineRunner&) = delete;
    TensorRtEngineRunner& operator=(const TensorRtEngineRunner&) = delete;
    TensorRtEngineRunner(TensorRtEngineRunner&&) noexcept;
    TensorRtEngineRunner& operator=(TensorRtEngineRunner&&) noexcept;

    [[nodiscard]] ModelContract contract() const noexcept override;
    PerceptionStatus initialize(const ModelManifest& manifest) noexcept override;
    PerceptionStatus warmup(std::uint32_t iterations) noexcept override;
    PerceptionStatus enqueue(const PerceptionRequest& request,
                             InferenceTicket& out_ticket) noexcept override;
    PollResult try_collect(const InferenceTicket& ticket,
                           bus::TargetObservationBatch& out_batch) noexcept override;
    void shutdown() noexcept override;

    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }
    [[nodiscard]] bool is_warmed_up() const noexcept { return is_warmed_up_; }
    [[nodiscard]] bool is_graph_captured() const noexcept { return is_graph_captured_; }
    [[nodiscard]] bool has_device_fault() const noexcept { return has_device_fault_; }
    [[nodiscard]] std::uint64_t total_inferences() const noexcept { return total_inferences_; }
    [[nodiscard]] std::uint64_t total_faults() const noexcept { return total_faults_; }
    [[nodiscard]] static bool is_backend_compiled() noexcept;
    /// Measured GPU inference duration of the latest collected ticket, excluding transfers.
    [[nodiscard]] std::optional<float> last_inference_ms() const noexcept { return last_inference_ms_; }

private:
    [[nodiscard]] bool manifest_matches_contract(const ModelManifest& manifest) const noexcept;

    struct Impl;
    std::unique_ptr<Impl> impl_{nullptr};

    TensorRtRunnerConfig config_{};
    ModelManifest manifest_{};
    std::optional<float> last_inference_ms_{};
    bool is_initialized_{false};
    bool is_warmed_up_{false};
    bool is_graph_captured_{false};
    bool has_device_fault_{false};
    std::uint64_t total_inferences_{0};
    std::uint64_t total_faults_{0};
};

} // namespace aim::perception
