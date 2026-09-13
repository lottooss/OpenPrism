// include/aim/interfaces/perception_engine.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include "aim/bus/bus_traits.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

enum class PerceptionStatus : std::uint32_t {
    ok = 0,
    invalid_argument = 1,
    uninitialized = 2,
    device_error = 3,
    inference_timeout = 4,
    unsupported_format = 5,
    artifact_not_found = 6,
    integrity_error = 7,
    backend_unavailable = 8
};

enum class PollResult : std::uint8_t {
    ready = 0,
    pending = 1,
    empty = 2,
    error = 3
};

struct ModelContract {
    std::string model_name{};
    std::string model_version{};
    std::uint32_t input_width_px{640};
    std::uint32_t input_height_px{384};
    std::uint32_t input_channels{3};
    std::uint32_t max_detections{64};
    float confidence_threshold{0.5f};
};

struct ModelManifest {
    std::string model_path{};
    std::string engine_sha256{};
    std::string runtime_backend{"tensorrt"};
    std::string precision{"fp16"};
    std::uint32_t input_width{640};
    std::uint32_t input_height{384};
    std::uint32_t input_channels{3};
    std::uint32_t target_gpu_device_id{0};
    std::size_t workspace_size_bytes{128 * 1024 * 1024};
};

struct PerceptionRequest {
    SequenceId frame_id{0};
    CorrelationId correlation_id{};
    MonotonicNs captured_at_ns{0};
    // Producer retains ownership and keeps the device tensor immutable/alive
    // until ticket collection or engine shutdown. GPU production must complete
    // before enqueue(); this native request carries no producer event.
    const void* gpu_tensor_ptr{nullptr};
    std::size_t tensor_size_bytes{0};
};

struct InferenceTicket {
    std::uint64_t ticket_id{0};
    MonotonicNs submitted_at_ns{0};
    // Non-owning backend handle valid until collection or engine shutdown.
    void* native_gpu_event{nullptr};
};

/// @brief Primary abstraction for visual neural perception engines (e.g. YOLO11n TensorRT).
class IPerceptionEngine {
public:
    virtual ~IPerceptionEngine() = default;
    [[nodiscard]] virtual ModelContract contract() const noexcept = 0;
    virtual PerceptionStatus initialize(const ModelManifest& manifest) noexcept = 0;
    virtual PerceptionStatus warmup(std::uint32_t iterations) noexcept = 0;
    virtual PerceptionStatus enqueue(const PerceptionRequest& request, InferenceTicket& out_ticket) noexcept = 0;
    virtual PollResult try_collect(const InferenceTicket& ticket, bus::TargetObservationBatch& out_batch) noexcept = 0;
    virtual void shutdown() noexcept = 0;
};

/// @brief Abstraction for direct external observation streams (e.g. synthetic simulator or IPC).
class IObservationSource {
public:
    virtual ~IObservationSource() = default;
    virtual bool start() noexcept = 0;
    virtual bool try_read_latest(bus::TargetObservationBatch& out_batch) noexcept = 0;
    virtual void stop() noexcept = 0;
};

} // namespace aim
