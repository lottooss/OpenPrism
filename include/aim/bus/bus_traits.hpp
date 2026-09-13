// include/aim/bus/bus_traits.hpp
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include "aim/bus/types.hpp"
#include "aim/core/actuator.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::bus {

constexpr std::size_t kMaxObservations = 64;
constexpr std::size_t kMaxTrackedTargets = 16;
constexpr std::size_t kMaxTelemetryStages = 8;

enum class FramePixelFormat : std::uint8_t {
    unknown             = 0,
    b8g8r8a8_unorm      = 1,
    r8g8b8a8_unorm      = 2,
    r16g16b16a16_float  = 3,
    nv12                = 4
};

enum class TrackState : std::uint8_t {
    tentative   = 0,
    confirmed   = 1,
    occluded    = 2,
    coasting    = 3,
    lost        = 4
};

enum class AimMode : std::uint8_t {
    idle                = 0,
    tracking            = 1,
    flick               = 2,
    micro_correction    = 3,
    emergency_hold      = 4
};

struct FrameDescriptor {
    std::uint16_t schema_major{1};
    std::uint16_t schema_minor{0};
    CorrelationId header{};
    std::uint64_t frame_id{0};
    MonotonicNs captured_at_ns{0};
    std::uint32_t width{1920};
    std::uint32_t height{1080};
    FramePixelFormat format{FramePixelFormat::b8g8r8a8_unorm};
    std::uint32_t pool_slot_index{0};
    std::uint64_t shared_nt_handle{0};
    std::uint64_t adapter_luid{0};
    bool is_keyframe{true};
};

struct VelocityHint {
    PixelVelocity pixels_per_second{};
    float confidence{0.0f};
};

struct TargetObservation {
    std::uint64_t source_id{0};
    std::uint64_t frame_id{0};
    MonotonicNs captured_at_ns{0};
    PixelPoint center_px{};
    NormalizedPoint center_norm{};
    BoundingBox bbox_px{};
    float effective_radius_px{0.0f};
    float confidence{0.0f};
    Covariance2D covariance_px2{};
    VelocityHint velocity{};
    Visibility visibility{Visibility::visible};
    float target_value{1.0f};
    std::uint32_t semantic_id{0};
};

struct TargetObservationBatch {
    std::uint16_t schema_major{1};
    std::uint16_t schema_minor{0};
    CorrelationId header{};
    std::uint64_t source_id{0};
    std::uint64_t frame_id{0};
    MonotonicNs captured_at_ns{0};
    MonotonicNs published_at_ns{0};
    std::uint32_t source_width{1920};
    std::uint32_t source_height{1080};
    std::uint32_t target_count{0};
    std::array<TargetObservation, kMaxObservations> targets{};

    [[nodiscard]] std::span<const TargetObservation> items() const noexcept {
        return {targets.data(), target_count};
    }

    bool add_target(const TargetObservation& obs) noexcept {
        if (target_count >= kMaxObservations) return false;
        targets[target_count++] = obs;
        return true;
    }

    void clear() noexcept {
        target_count = 0;
    }
};

struct TrackedTarget {
    TrackId track_id{0};
    TrackState state{TrackState::confirmed};
    std::uint32_t total_visible_frames{1};
    std::uint32_t total_missed_frames{0};
    PixelPoint filtered_center_px{};
    PixelVelocity filtered_velocity_px_per_s{};
    PixelAcceleration filtered_acceleration_px_per_s2{};
    Covariance2D covariance_px2{};
    PixelPoint predicted_center_px{};
    MonotonicNs prediction_time_ns{0};
    float effective_radius_px{0.0f};
    float confidence{0.0f};
    float target_value{1.0f};
    std::uint32_t semantic_id{0};
};

struct TrackedTargetBatch {
    std::uint16_t schema_major{1};
    std::uint16_t schema_minor{0};
    CorrelationId header{};
    std::uint64_t frame_id{0};
    MonotonicNs timestamp_ns{0};
    std::uint32_t track_count{0};
    std::array<TrackedTarget, kMaxTrackedTargets> tracks{};

    [[nodiscard]] std::span<const TrackedTarget> items() const noexcept {
        return {tracks.data(), track_count};
    }

    bool add_track(const TrackedTarget& track) noexcept {
        if (track_count >= kMaxTrackedTargets) return false;
        tracks[track_count++] = track;
        return true;
    }

    void clear() noexcept {
        track_count = 0;
    }
};

struct AimIntent {
    std::uint16_t schema_major{1};
    std::uint16_t schema_minor{0};
    CorrelationId header{};
    TrackId target_track_id{0};
    AimMode mode{AimMode::tracking};
    PixelPoint target_aim_px{};
    NormalizedPoint target_aim_norm{};
    PixelPoint lead_offset_px{};
    float error_distance_px{0.0f};
    bool authorize_fire{false};
    float confidence{1.0f};
    float utility_score{1.0f};
    MonotonicNs command_deadline_ns{0};
};

struct StageTiming {
    PipelineStage stage{PipelineStage::capture_arrival};
    MonotonicNs start_ns{0};
    MonotonicNs end_ns{0};
    MonotonicNs duration_ns{0};
};

struct alignas(64) HotLoopTelemetryEvent {
    std::uint16_t schema_major{1};
    std::uint16_t schema_minor{0};
    CorrelationId header{};
    MonotonicNs capture_arrival_ns{0};
    MonotonicNs actuation_dispatch_ns{0};
    MonotonicNs total_latency_ns{0};
    bool is_stale_dropped{false};
    std::uint32_t queue_depth{0};
    std::uint32_t stage_count{0};
    std::array<StageTiming, kMaxTelemetryStages> stages{};

    [[nodiscard]] std::span<const StageTiming> stage_items() const noexcept {
        return {stages.data(), stage_count};
    }

    bool add_stage(PipelineStage stage, MonotonicNs start, MonotonicNs end) noexcept {
        if (stage_count >= kMaxTelemetryStages) return false;
        stages[stage_count++] = StageTiming{
            .stage = stage,
            .start_ns = start,
            .end_ns = end,
            .duration_ns = (end >= start) ? (end - start) : 0
        };
        return true;
    }

    void clear() noexcept {
        stage_count = 0;
    }
};

// =============================================================================
// Bus Payload Traits & FourCC Magic Identifiers
// =============================================================================

template <typename T>
struct BusPayloadTraits;

template <>
struct BusPayloadTraits<FrameDescriptor> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31524641; // 'AFR1'
    static constexpr ChannelId kChannelId = ChannelId::frame_descriptor;
    static constexpr std::string_view kChannelName = "FrameDescriptor";
    static constexpr std::string_view kFileExtension = ".afr";
};

template <>
struct BusPayloadTraits<TargetObservationBatch> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31424F41; // 'AOB1'
    static constexpr ChannelId kChannelId = ChannelId::target_observation;
    static constexpr std::string_view kChannelName = "TargetObservationBatch";
    static constexpr std::string_view kFileExtension = ".aob";
};

template <>
struct BusPayloadTraits<TrackedTargetBatch> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31545441; // 'ATT1'
    static constexpr ChannelId kChannelId = ChannelId::tracked_target;
    static constexpr std::string_view kChannelName = "TrackedTargetBatch";
    static constexpr std::string_view kFileExtension = ".att";
};

template <>
struct BusPayloadTraits<AimIntent> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31494141; // 'AAI1'
    static constexpr ChannelId kChannelId = ChannelId::aim_intent;
    static constexpr std::string_view kChannelName = "AimIntent";
    static constexpr std::string_view kFileExtension = ".aai";
};

template <>
struct BusPayloadTraits<ActuationCommand> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31434141; // 'AAC1'
    static constexpr ChannelId kChannelId = ChannelId::actuation_command;
    static constexpr std::string_view kChannelName = "ActuationCommand";
    static constexpr std::string_view kFileExtension = ".aac";
};

template <>
struct BusPayloadTraits<HotLoopTelemetryEvent> {
    static constexpr std::uint32_t kMagicIdentifier = 0x31455441; // 'ATE1'
    static constexpr ChannelId kChannelId = ChannelId::telemetry_event;
    static constexpr std::string_view kChannelName = "HotLoopTelemetryEvent";
    static constexpr std::string_view kFileExtension = ".ate";
};

} // namespace aim::bus
