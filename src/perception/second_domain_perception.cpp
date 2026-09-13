// src/perception/second_domain_perception.cpp
#include "aim/perception/second_domain_perception.hpp"

#include <algorithm>
#include <cmath>

namespace aim::perception {

SecondDomainObservationSource::SecondDomainObservationSource(SecondDomainConfig config) noexcept
    : config_(std::move(config)) {}

bool SecondDomainObservationSource::start() noexcept {
    is_running_ = true;
    return true;
}

void SecondDomainObservationSource::stop() noexcept {
    is_running_ = false;
}

void SecondDomainObservationSource::set_targets(const std::vector<HumanoidTargetDef>& targets) {
    targets_ = targets;
}

void SecondDomainObservationSource::advance_simulation(MonotonicNs step_dt_ns) noexcept {
    const float dt_sec = static_cast<float>(step_dt_ns) / 1'000'000'000.0f;
    current_time_ns_ += step_dt_ns;
    ++current_frame_id_;

    for (auto& tgt : targets_) {
        tgt.center_px.x += tgt.velocity_px_per_s.x_per_s * dt_sec;
        tgt.center_px.y += tgt.velocity_px_per_s.y_per_s * dt_sec;

        // Bounce within screen boundaries
        if (tgt.center_px.x < 100.0f || tgt.center_px.x > static_cast<float>(config_.frame_width) - 100.0f) {
            tgt.velocity_px_per_s.x_per_s = -tgt.velocity_px_per_s.x_per_s;
        }
        if (tgt.center_px.y < 100.0f || tgt.center_px.y > static_cast<float>(config_.frame_height) - 100.0f) {
            tgt.velocity_px_per_s.y_per_s = -tgt.velocity_px_per_s.y_per_s;
        }
    }
}

bus::TargetObservation SecondDomainObservationSource::make_humanoid_observation(
    const HumanoidTargetDef& def,
    std::uint64_t frame_id,
    MonotonicNs timestamp_ns,
    std::uint32_t width,
    std::uint32_t height
) noexcept {
    bus::TargetObservation obs{};
    obs.source_id = def.source_id;
    obs.frame_id = frame_id;
    obs.captured_at_ns = timestamp_ns;
    obs.center_px = def.center_px;

    // Convert pixel coordinates to normalized [-1, 1] coordinates
    const float half_w = static_cast<float>(width) * 0.5f;
    const float half_h = static_cast<float>(height) * 0.5f;
    obs.center_norm.x = (def.center_px.x - half_w) / half_w;
    obs.center_norm.y = (def.center_px.y - half_h) / half_h;

    // Tall humanoid bounding box (height >> width)
    const float half_box_w = def.width_px * 0.5f;
    const float half_box_h = def.height_px * 0.5f;
    obs.bbox_px.left = def.center_px.x - half_box_w;
    obs.bbox_px.right = def.center_px.x + half_box_w;
    obs.bbox_px.top = def.center_px.y - half_box_h;
    obs.bbox_px.bottom = def.center_px.y + half_box_h;

    // Effective radius: hit tolerance for tall target is based on half width (narrowest cross-section)
    obs.effective_radius_px = std::min(def.width_px, def.height_px) * 0.5f;
    obs.confidence = def.confidence;

    // Anisotropic covariance: vertical position uncertainty is typically greater than horizontal for humanoids
    const float var_x = (def.width_px * 0.1f) * (def.width_px * 0.1f);
    const float var_y = (def.height_px * 0.1f) * (def.height_px * 0.1f);
    obs.covariance_px2.xx = std::max(1.0f, var_x);
    obs.covariance_px2.xy = 0.0f;
    obs.covariance_px2.yy = std::max(4.0f, var_y);

    obs.velocity.pixels_per_second = def.velocity_px_per_s;
    obs.velocity.confidence = def.velocity_confidence;
    obs.visibility = Visibility::visible;
    obs.target_value = 1.0f;
    obs.semantic_id = def.semantic_id;

    return obs;
}

bool SecondDomainObservationSource::try_read_latest(bus::TargetObservationBatch& out_batch) noexcept {
    if (!is_running_) return false;

    out_batch = {};
    out_batch.schema_major = 1;
    out_batch.schema_minor = 0;
    out_batch.header.sequence_id = current_frame_id_;
    out_batch.header.source_timestamp_ns = current_time_ns_;
    out_batch.header.pipeline_run_id = 888;
    out_batch.source_id = 1;
    out_batch.frame_id = current_frame_id_;
    out_batch.captured_at_ns = current_time_ns_;
    out_batch.published_at_ns = current_time_ns_;
    out_batch.source_width = config_.frame_width;
    out_batch.source_height = config_.frame_height;

    for (const auto& tgt : targets_) {
        if (!out_batch.add_target(make_humanoid_observation(
            tgt,
            current_frame_id_,
            current_time_ns_,
            config_.frame_width,
            config_.frame_height
        ))) {
            break;
        }
    }

    return true;
}

ModelContract SecondDomainObservationSource::contract() const noexcept {
    ModelContract c{};
    c.model_name = "tactical_humanoid_detector";
    c.model_version = "1.0.0";
    c.input_width_px = 640;
    c.input_height_px = 384;
    c.input_channels = 3;
    c.max_detections = 64;
    c.confidence_threshold = 0.5f;
    return c;
}

PerceptionStatus SecondDomainObservationSource::initialize(const ModelManifest&) noexcept {
    is_initialized_ = true;
    return PerceptionStatus::ok;
}

PerceptionStatus SecondDomainObservationSource::warmup(std::uint32_t) noexcept {
    return PerceptionStatus::ok;
}

PerceptionStatus SecondDomainObservationSource::enqueue(const PerceptionRequest& request, InferenceTicket& out_ticket) noexcept {
    if (!is_initialized_) return PerceptionStatus::uninitialized;
    out_ticket.ticket_id = ++ticket_counter_;
    out_ticket.submitted_at_ns = request.captured_at_ns;
    return PerceptionStatus::ok;
}

PollResult SecondDomainObservationSource::try_collect(const InferenceTicket&, bus::TargetObservationBatch& out_batch) noexcept {
    if (try_read_latest(out_batch)) {
        return PollResult::ready;
    }
    return PollResult::empty;
}

void SecondDomainObservationSource::shutdown() noexcept {
    stop();
    is_initialized_ = false;
}

} // namespace aim::perception
