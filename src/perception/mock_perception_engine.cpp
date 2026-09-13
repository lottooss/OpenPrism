#include "aim/perception/mock_perception_engine.hpp"

#include <algorithm>
#include <utility>

namespace aim::perception {

MockTensorRtEngineRunner::MockTensorRtEngineRunner(TensorRtRunnerConfig config) noexcept
    : config_(std::move(config)) {}

ModelContract MockTensorRtEngineRunner::contract() const noexcept {
    ModelContract result{};
    result.model_name = config_.model_name + "_mock";
    result.model_version = config_.model_version;
    result.input_width_px = config_.input_width;
    result.input_height_px = config_.input_height;
    result.input_channels = config_.input_channels;
    result.max_detections = config_.max_detections;
    result.confidence_threshold = config_.confidence_threshold;
    return result;
}

PerceptionStatus MockTensorRtEngineRunner::initialize(const ModelManifest& manifest) noexcept {
    shutdown();
    if (manifest.runtime_backend != "mock") {
        return PerceptionStatus::unsupported_format;
    }
    is_initialized_ = true;
    return PerceptionStatus::ok;
}

PerceptionStatus MockTensorRtEngineRunner::warmup(std::uint32_t /*iterations*/) noexcept {
    if (!is_initialized_) {
        return PerceptionStatus::uninitialized;
    }
    if (has_device_fault_) {
        return PerceptionStatus::device_error;
    }
    is_warmed_up_ = true;
    return PerceptionStatus::ok;
}

PerceptionStatus MockTensorRtEngineRunner::enqueue(
    const PerceptionRequest& request,
    InferenceTicket& out_ticket) noexcept {
    out_ticket = InferenceTicket{};
    if (!is_initialized_) {
        return PerceptionStatus::uninitialized;
    }
    if (has_device_fault_) {
        ++total_faults_;
        return PerceptionStatus::device_error;
    }

    out_ticket.ticket_id = next_ticket_id_++;
    out_ticket.submitted_at_ns = request.captured_at_ns;
    active_ticket_ = out_ticket;
    active_request_ = request;
    pending_inference_ = true;
    ++total_inferences_;
    return PerceptionStatus::ok;
}

PollResult MockTensorRtEngineRunner::try_collect(
    const InferenceTicket& ticket,
    bus::TargetObservationBatch& out_batch) noexcept {
    if (!is_initialized_ || has_device_fault_) {
        return PollResult::error;
    }
    if (!pending_inference_ || ticket.ticket_id != active_ticket_.ticket_id) {
        return PollResult::empty;
    }

    out_batch.clear();
    out_batch.header = active_request_.correlation_id;
    out_batch.frame_id = active_request_.frame_id;
    out_batch.captured_at_ns = active_request_.captured_at_ns;
    out_batch.published_at_ns = active_request_.captured_at_ns;
    out_batch.source_width = 1920;
    out_batch.source_height = 1080;

    for (std::size_t index = 0; index < detection_count_; ++index) {
        const DecodedDetection& detection = detections_[index];
        if (detection.confidence < config_.confidence_threshold) {
            continue;
        }

        const float radius = detection.radius_px > 0.0F
            ? detection.radius_px
            : (detection.width_px + detection.height_px) * 0.25F;
        bus::TargetObservation target{};
        target.source_id = static_cast<std::uint64_t>(index + 1U);
        target.frame_id = active_request_.frame_id;
        target.captured_at_ns = active_request_.captured_at_ns;
        target.semantic_id = detection.class_id;
        target.center_px = PixelPoint{detection.center_x_px, detection.center_y_px};
        target.center_norm = NormalizedPoint{
            (detection.center_x_px / 960.0F) - 1.0F,
            (detection.center_y_px / 540.0F) - 1.0F};
        target.effective_radius_px = radius;
        target.bbox_px = BoundingBox{
            detection.center_x_px - detection.width_px * 0.5F,
            detection.center_y_px - detection.height_px * 0.5F,
            detection.center_x_px + detection.width_px * 0.5F,
            detection.center_y_px + detection.height_px * 0.5F};
        target.confidence = detection.confidence;
        target.covariance_px2 = Covariance2D{1.0F, 0.0F, 1.0F};
        target.visibility = Visibility::visible;
        target.target_value = 1.0F;
        if (!out_batch.add_target(target)) {
            break;
        }
    }

    pending_inference_ = false;
    return PollResult::ready;
}

bool MockTensorRtEngineRunner::set_detections(
    std::span<const DecodedDetection> detections) noexcept {
    if (detections.size() > detections_.size() ||
        detections.size() > static_cast<std::size_t>(config_.max_detections)) {
        return false;
    }
    std::copy(detections.begin(), detections.end(), detections_.begin());
    detection_count_ = detections.size();
    return true;
}

void MockTensorRtEngineRunner::shutdown() noexcept {
    is_initialized_ = false;
    is_warmed_up_ = false;
    has_device_fault_ = false;
    pending_inference_ = false;
    detection_count_ = 0;
    active_ticket_ = InferenceTicket{};
    active_request_ = PerceptionRequest{};
}

} // namespace aim::perception
