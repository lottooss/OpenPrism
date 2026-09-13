// src/perception/yolo_decoder.cpp
#include "aim/perception/yolo_decoder.hpp"

#include <algorithm>
#include <cmath>

namespace aim::perception {

YoloDecoder::YoloDecoder(YoloDecoderConfig config)
    : config_(std::move(config)) {
    // Pre-allocate fixed working buffers (zero hot-path allocations)
    candidates_.reserve(config_.num_anchors);
    suppressed_.reserve(config_.num_anchors);
}

float YoloDecoder::compute_box_iou(const RawBoxCandidate& a, const RawBoxCandidate& b) noexcept {
    const float ix1 = std::max(a.xmin_model, b.xmin_model);
    const float iy1 = std::max(a.ymin_model, b.ymin_model);
    const float ix2 = std::min(a.xmax_model, b.xmax_model);
    const float iy2 = std::min(a.ymax_model, b.ymax_model);

    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float intersection = iw * ih;

    const float area_a = (a.xmax_model - a.xmin_model) * (a.ymax_model - a.ymin_model);
    const float area_b = (b.xmax_model - b.xmin_model) * (b.ymax_model - b.ymin_model);
    const float union_area = area_a + area_b - intersection;

    return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
}

void YoloDecoder::decode(
    const float* raw_tensor_ptr,
    std::size_t num_elements,
    const CorrelationId& correlation_id,
    MonotonicNs timestamp_ns,
    bus::TargetObservationBatch& out_batch) noexcept {
    ++total_decoded_frames_;
    candidates_.clear();

    out_batch.header = correlation_id;
    out_batch.frame_id = correlation_id.sequence_id;
    out_batch.captured_at_ns = timestamp_ns;
    out_batch.published_at_ns = timestamp_ns;
    out_batch.source_width = config_.source_width;
    out_batch.source_height = config_.source_height;
    out_batch.clear();

    if (raw_tensor_ptr == nullptr || num_elements < 5 * config_.num_anchors) {
        return;
    }

    const std::uint32_t num_anchors = config_.num_anchors;
    // Tensor layout: [1, 5, num_anchors]
    // channel 0: cx, channel 1: cy, channel 2: w, channel 3: h, channel 4: conf
    const float* cx_ptr = raw_tensor_ptr;
    const float* cy_ptr = raw_tensor_ptr + num_anchors;
    const float* w_ptr = raw_tensor_ptr + 2 * num_anchors;
    const float* h_ptr = raw_tensor_ptr + 3 * num_anchors;
    const float* conf_ptr = raw_tensor_ptr + 4 * num_anchors;

    // 1. Filter candidates above confidence threshold
    for (std::uint32_t i = 0; i < num_anchors; ++i) {
        const float conf = conf_ptr[i];
        if (!std::isfinite(conf) || conf > 1.0F || conf < config_.confidence_threshold) {
            continue;
        }

        const float cx = cx_ptr[i];
        const float cy = cy_ptr[i];
        const float w = w_ptr[i];
        const float h = h_ptr[i];

        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || w <= 0.0F || h <= 0.0F ||
            cx < 0.0F || cy < 0.0F || cx > static_cast<float>(config_.model_width) ||
            cy > static_cast<float>(config_.model_height) ||
            w > static_cast<float>(config_.model_width) || h > static_cast<float>(config_.model_height)) {
            continue;
        }
        RawBoxCandidate cand{};
        cand.center_x_model = cx;
        cand.center_y_model = cy;
        cand.width_model = w;
        cand.height_model = h;
        cand.confidence = conf;
        cand.class_id = 0;
        cand.xmin_model = cx - w * 0.5f;
        cand.ymin_model = cy - h * 0.5f;
        cand.xmax_model = cx + w * 0.5f;
        cand.ymax_model = cy + h * 0.5f;

        candidates_.push_back(cand);
    }

    if (candidates_.empty()) {
        return;
    }

    // 2. Deterministic Sort: descending by confidence
    std::sort(candidates_.begin(), candidates_.end(), [](const RawBoxCandidate& a, const RawBoxCandidate& b) {
        return a.confidence > b.confidence;
    });

    // 3. Non-Maximum Suppression (NMS)
    suppressed_.assign(candidates_.size(), false);
    const float iou_thresh = config_.nms_iou_threshold;

    std::uint32_t kept_count = 0;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        if (suppressed_[i]) {
            continue;
        }

        const auto& kept = candidates_[i];
        if (kept_count >= std::min<std::size_t>(config_.max_detections, bus::kMaxObservations)) {
            ++overflow_count_;
            break;
        }

        // Map model coordinates to full 1920x1080 source FOV
        const PixelPoint model_center{kept.center_x_model, kept.center_y_model};
        const PixelPoint src_center = config_.transform.inverse_point(model_center);
        const float src_radius = config_.transform.inverse_radius((kept.width_model + kept.height_model) * 0.25f);

        if (!std::isfinite(src_center.x) || !std::isfinite(src_center.y) ||
            !std::isfinite(src_radius) || src_radius <= 0.0F ||
            src_center.x < 0.0F || src_center.y < 0.0F ||
            src_center.x > static_cast<float>(config_.source_width) ||
            src_center.y > static_cast<float>(config_.source_height)) {
            continue; // Predictions inside letterbox padding are not source-frame observations.
        }
        bus::TargetObservation target{};
        target.source_id = kept_count + 1;
        target.frame_id = correlation_id.sequence_id;
        target.captured_at_ns = timestamp_ns;
        target.semantic_id = kept.class_id;
        target.center_px = src_center;
        target.center_norm = NormalizedPoint{
            (src_center.x / (static_cast<float>(config_.source_width) * 0.5f)) - 1.0f,
            (src_center.y / (static_cast<float>(config_.source_height) * 0.5f)) - 1.0f
        };
        target.effective_radius_px = src_radius;
        target.bbox_px = BoundingBox{
            src_center.x - src_radius,
            src_center.y - src_radius,
            src_center.x + src_radius,
            src_center.y + src_radius
        };
        target.confidence = kept.confidence;
        target.covariance_px2 = Covariance2D{
            .xx = 1.0f / (config_.transform.sx * config_.transform.sx),
            .xy = 0.0f,
            .yy = 1.0f / (config_.transform.sy * config_.transform.sy)
        };
        target.visibility = Visibility::visible;
        target.target_value = 1.0f;

        if (out_batch.add_target(target)) {
            ++kept_count;
        }

        // Suppress overlapping boxes
        for (std::size_t j = i + 1; j < candidates_.size(); ++j) {
            if (!suppressed_[j]) {
                if (compute_box_iou(kept, candidates_[j]) > iou_thresh) {
                    suppressed_[j] = true;
                }
            }
        }
    }
}

} // namespace aim::perception
