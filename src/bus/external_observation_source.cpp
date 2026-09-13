// src/bus/external_observation_source.cpp
#include "aim/bus/external_observation_source.hpp"

#include <chrono>
#include <cmath>

namespace aim::bus {

namespace {

inline MonotonicNs get_current_time_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

} // namespace

ExternalObservationSource::ExternalObservationSource(ExternalObservationSourceConfig config) noexcept
    : config_(std::move(config)) {}

ExternalObservationSource::~ExternalObservationSource() {
    stop();
}

bool ExternalObservationSource::start() noexcept {
    stop();

    ring_ = SharedMemoryRing<bus::TargetObservationBatch, 16>::open_read_only(config_.shm_name);
    if (!ring_.is_valid()) {
        telemetry_.last_status = ObservationSourceStatus::uninitialized;
        return false;
    }

    const auto* hdr = ring_.control_header();
    if (hdr == nullptr) {
        telemetry_.last_status = ObservationSourceStatus::uninitialized;
        return false;
    }

    if (config_.validate_schema) {
        if (hdr->magic != kShmRingMagic) {
            telemetry_.last_status = ObservationSourceStatus::schema_mismatch;
            return false;
        }
        if (hdr->version_major != kShmVersionMajor) {
            telemetry_.last_status = ObservationSourceStatus::schema_mismatch;
            return false;
        }
        if (hdr->channel_id != static_cast<std::uint32_t>(BusPayloadTraits<bus::TargetObservationBatch>::kChannelId)) {
            telemetry_.last_status = ObservationSourceStatus::schema_mismatch;
            return false;
        }
        if (config_.expected_pipeline_run_id != 0 && hdr->pipeline_run_id != config_.expected_pipeline_run_id) {
            telemetry_.last_status = ObservationSourceStatus::schema_mismatch;
            return false;
        }
    }

    last_consumed_seq_ = 0;
    is_running_ = true;
    telemetry_.last_status = ObservationSourceStatus::ok;
    return true;
}

void ExternalObservationSource::stop() noexcept {
    is_running_ = false;
    ring_ = {};
    last_consumed_seq_ = 0;
}

bool ExternalObservationSource::try_read_latest(bus::TargetObservationBatch& out_batch) noexcept {
    return try_read_latest_at(out_batch, get_current_time_ns());
}

bool ExternalObservationSource::try_read_latest_at(bus::TargetObservationBatch& out_batch, MonotonicNs now_ns) noexcept {
    if (!is_running_ || !ring_.is_valid()) {
        telemetry_.last_status = ObservationSourceStatus::uninitialized;
        return false;
    }

    // Heartbeat / Producer Liveness Gate
    if (config_.require_heartbeat) {
        if (!ring_.is_writer_alive(now_ns, config_.max_heartbeat_age_ns)) {
            telemetry_.last_status = ObservationSourceStatus::producer_dead;
            ++telemetry_.total_heartbeat_timeouts;
            return false;
        }
    }

    const auto* hdr = ring_.control_header();
    if (hdr != nullptr) {
        telemetry_.last_producer_heartbeat_ns = hdr->writer_heartbeat_ns.load(std::memory_order_relaxed);
    }

    // Attempt Seqlock Read
    std::uint64_t dropped = 0;
    const auto res = ring_.try_read_latest(out_batch, last_consumed_seq_, &dropped);
    telemetry_.total_dropped_overrun += dropped;

    if (res == BusReadResult::no_new_data) {
        telemetry_.last_status = ObservationSourceStatus::no_new_data;
        return false;
    }

    if (res == BusReadResult::torn_or_overrun) {
        telemetry_.last_status = ObservationSourceStatus::malformed_payload;
        ++telemetry_.total_torn_reads;
        return false;
    }

    if (res != BusReadResult::ok) {
        telemetry_.last_status = ObservationSourceStatus::malformed_payload;
        return false;
    }

    // Structural and semantic payload integrity verification
    if (!validate_batch(out_batch)) {
        telemetry_.last_status = ObservationSourceStatus::malformed_payload;
        ++telemetry_.total_malformed_rejected;
        return false;
    }

    telemetry_.last_status = ObservationSourceStatus::ok;
    telemetry_.last_read_timestamp_ns = now_ns;
    ++telemetry_.total_batches_read;
    return true;
}

bool ExternalObservationSource::validate_batch(const bus::TargetObservationBatch& batch) noexcept {
    if (batch.schema_major != 1) {
        return false;
    }

    if (batch.target_count > bus::kMaxObservations) {
        return false;
    }

    for (const auto& target : batch.items()) {
        if (!std::isfinite(target.center_px.x) || !std::isfinite(target.center_px.y)) {
            return false;
        }
        if (!std::isfinite(target.center_norm.x) || !std::isfinite(target.center_norm.y)) {
            return false;
        }
        if (!std::isfinite(target.effective_radius_px) || target.effective_radius_px < 0.0f) {
            return false;
        }
        if (!std::isfinite(target.confidence) || target.confidence < 0.0f || target.confidence > 1.0f) {
            return false;
        }
        if (!std::isfinite(target.covariance_px2.xx) ||
            !std::isfinite(target.covariance_px2.xy) ||
            !std::isfinite(target.covariance_px2.yy)) {
            return false;
        }
        if (target.covariance_px2.xx < 0.0f || target.covariance_px2.yy < 0.0f) {
            return false;
        }
    }

    return true;
}

} // namespace aim::bus
