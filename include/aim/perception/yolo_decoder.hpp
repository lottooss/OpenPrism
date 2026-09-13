// include/aim/perception/yolo_decoder.hpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aim/bus/bus_traits.hpp"
#include "aim/bus/types.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/coordinate_transform.hpp"

namespace aim::perception {

struct YoloDecoderConfig {
    float confidence_threshold{0.25f};
    float nms_iou_threshold{0.45f};
    std::uint32_t num_anchors{5040};
    std::uint32_t num_classes{1};
    std::uint32_t model_width{640};
    std::uint32_t model_height{384};
    std::uint32_t source_width{1920};
    std::uint32_t source_height{1080};
    std::uint32_t max_detections{64};
    AffineTransform2D transform{1.0f / 3.0f, 1.0f / 3.0f, 0.0f, 12.0f};
};

struct RawBoxCandidate {
    float center_x_model{0.0f};
    float center_y_model{0.0f};
    float width_model{0.0f};
    float height_model{0.0f};
    float confidence{0.0f};
    std::uint32_t class_id{0};
    float xmin_model{0.0f};
    float ymin_model{0.0f};
    float xmax_model{0.0f};
    float ymax_model{0.0f};
};

class YoloDecoder {
public:
    explicit YoloDecoder(YoloDecoderConfig config = {});

    /// @brief Decodes raw tensor outputs (shape [1, 5, num_anchors]) into canonical TargetObservationBatch.
    /// Zero heap allocations on hot path!
    void decode(
        const float* raw_tensor_ptr,
        std::size_t num_elements,
        const CorrelationId& correlation_id,
        MonotonicNs timestamp_ns,
        bus::TargetObservationBatch& out_batch) noexcept;

    [[nodiscard]] std::uint64_t overflow_count() const noexcept { return overflow_count_; }
    [[nodiscard]] std::uint64_t total_decoded_frames() const noexcept { return total_decoded_frames_; }

    static float compute_box_iou(const RawBoxCandidate& a, const RawBoxCandidate& b) noexcept;

private:
    YoloDecoderConfig config_;
    std::uint64_t overflow_count_{0};
    std::uint64_t total_decoded_frames_{0};

    // Pre-allocated candidate pool (zero allocations on hot path)
    std::vector<RawBoxCandidate> candidates_{};
    std::vector<bool> suppressed_{};
};

} // namespace aim::perception
