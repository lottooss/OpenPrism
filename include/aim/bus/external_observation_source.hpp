// include/aim/bus/external_observation_source.hpp
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

#include "aim/bus/bus_traits.hpp"
#include "aim/bus/shared_memory_ring.hpp"
#include "aim/core/time.hpp"
#include "aim/interfaces/perception_engine.hpp"

namespace aim::bus {

enum class ObservationSourceStatus : std::uint32_t {
    ok = 0,
    uninitialized = 1,
    producer_dead = 2,
    heartbeat_timeout = 3,
    schema_mismatch = 4,
    malformed_payload = 5,
    no_new_data = 6
};

[[nodiscard]] constexpr std::string_view to_string(ObservationSourceStatus status) noexcept {
    switch (status) {
        case ObservationSourceStatus::ok: return "ok";
        case ObservationSourceStatus::uninitialized: return "uninitialized";
        case ObservationSourceStatus::producer_dead: return "producer_dead";
        case ObservationSourceStatus::heartbeat_timeout: return "heartbeat_timeout";
        case ObservationSourceStatus::schema_mismatch: return "schema_mismatch";
        case ObservationSourceStatus::malformed_payload: return "malformed_payload";
        case ObservationSourceStatus::no_new_data: return "no_new_data";
    }
    return "unknown";
}

struct ExternalObservationSourceConfig {
    std::string shm_name{"aim_target_observations_shm"};
    MonotonicNs max_heartbeat_age_ns{50'000'000LL}; // 50 ms max heartbeat delay before producer deemed dead
    bool require_heartbeat{true};
    bool validate_schema{true};
    std::uint32_t expected_pipeline_run_id{0}; // 0 = match any run id
};

struct ExternalObservationSourceTelemetry {
    std::uint64_t total_batches_read{0};
    std::uint64_t total_dropped_overrun{0};
    std::uint64_t total_torn_reads{0};
    std::uint64_t total_malformed_rejected{0};
    std::uint64_t total_heartbeat_timeouts{0};
    ObservationSourceStatus last_status{ObservationSourceStatus::uninitialized};
    MonotonicNs last_read_timestamp_ns{0};
    MonotonicNs last_producer_heartbeat_ns{0};
};

/// @brief External Observation Source over versioned shared-memory ring (Milestone M8-01).
/// Enables external processes or alternative perception models to feed canonical observations
/// without contaminating core tracking, prediction, aim policy, or actuation.
class ExternalObservationSource final : public IObservationSource {
public:
    explicit ExternalObservationSource(ExternalObservationSourceConfig config = {}) noexcept;
    ~ExternalObservationSource() override;

    ExternalObservationSource(const ExternalObservationSource&) = delete;
    ExternalObservationSource& operator=(const ExternalObservationSource&) = delete;

    ExternalObservationSource(ExternalObservationSource&&) noexcept = default;
    ExternalObservationSource& operator=(ExternalObservationSource&&) noexcept = default;

    /// @brief Attach to the named shared memory ring in read-only mode.
    bool start() noexcept override;

    /// @brief Read the newest available observation batch from shared memory.
    /// Fails closed on stale publisher, dead process, malformed batch, or schema mismatch.
    bool try_read_latest(bus::TargetObservationBatch& out_batch) noexcept override;

    /// @brief Read with explicit caller timestamp for deterministic replay/testing.
    bool try_read_latest_at(bus::TargetObservationBatch& out_batch, MonotonicNs now_ns) noexcept;

    void stop() noexcept override;

    [[nodiscard]] bool is_running() const noexcept { return is_running_; }
    [[nodiscard]] ObservationSourceStatus last_status() const noexcept { return telemetry_.last_status; }
    [[nodiscard]] const ExternalObservationSourceTelemetry& telemetry() const noexcept { return telemetry_; }
    void reset_telemetry() noexcept { telemetry_ = {}; }

    /// @brief Validate observation batch integrity (bounds, finite coordinates, confidence).
    [[nodiscard]] static bool validate_batch(const bus::TargetObservationBatch& batch) noexcept;

private:
    ExternalObservationSourceConfig config_;
    SharedMemoryRing<bus::TargetObservationBatch, 16> ring_{};
    ExternalObservationSourceTelemetry telemetry_{};
    std::uint64_t last_consumed_seq_{0};
    bool is_running_{false};
};

} // namespace aim::bus
