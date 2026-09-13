// include/aim/perception/capture_to_tensor_pipeline.hpp
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "aim/bus/bus.hpp"
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/unified_capture_source.hpp"
#include "aim/capture/cuda_surface_pool.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/gpu_preprocessor.hpp"
#include "aim/perception/preprocess_types.hpp"

namespace aim::perception {

/// @brief Configuration for the combined capture-to-tensor execution pipeline.
struct CaptureToTensorConfig {
    aim::FrameSourceConfig capture_config{};
    PreprocessConfig preprocess_config{};
    MonotonicNs stale_threshold_ns{10'000'000LL}; // 10ms hard cutoff from blueprint
};

/// @brief Statistical summary of latency samples in milliseconds.
struct LatencyStats {
    std::uint64_t sample_count{0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
    double max_ms{0.0};
    double min_ms{0.0};
    double avg_ms{0.0};
};

/// @brief Comprehensive runtime telemetry metrics for the capture-to-tensor pipeline.
struct CaptureToTensorMetrics {
    std::uint64_t total_frames_processed{0};
    std::uint64_t frames_acquired{0};
    std::uint64_t frames_dropped_stale{0};
    std::uint64_t frames_dropped_invalid{0};
    std::uint64_t recovery_events_count{0};

    LatencyStats capture_to_tensor_latency{}; // surface arrival -> tensor ready
    LatencyStats stage_acquire_latency{};     // frame lease acquire duration
    LatencyStats stage_preprocess_latency{};  // GPU preprocess kernel duration
};

/// @brief Complete, unified capture-to-tensor pipeline coordinating frame acquisition,
/// GPU surface interop, fused preprocessing, and publishing to downstream perception/inference rings.
class CaptureToTensorPipeline {
public:
    explicit CaptureToTensorPipeline(
        std::unique_ptr<aim::IFrameSource> frame_source = nullptr,
        std::unique_ptr<IGpuPreprocessor> preprocessor = nullptr,
        std::shared_ptr<IClock> clock = std::make_shared<QpcClock>()) noexcept;

    ~CaptureToTensorPipeline() noexcept;

    CaptureToTensorPipeline(const CaptureToTensorPipeline&) = delete;
    CaptureToTensorPipeline& operator=(const CaptureToTensorPipeline&) = delete;
    CaptureToTensorPipeline(CaptureToTensorPipeline&&) = delete;
    CaptureToTensorPipeline& operator=(CaptureToTensorPipeline&&) = delete;

    /// @brief Initializes capture source, CUDA interop, preprocessor pool, and internal ring buffers.
    bool initialize(const CaptureToTensorConfig& config,
                    std::shared_ptr<capture::ICudaInteropBackend> cuda_backend = nullptr) noexcept;

    /// @brief Warms up GPU pipelines, texture staging, CUDA kernels, and memory pools.
    bool warmup(std::uint32_t warmup_iterations = 10) noexcept;

    /// @brief Starts frame acquisition loop.
    bool start() noexcept;

    /// @brief Executes a single iteration of the capture-to-tensor pipeline.
    /// Allocation-free after warmup.
    /// @param out_tensor_lease Target RAII lease receiving preprocessed tensor upon success.
    /// @return True if a fresh valid tensor was produced; false if no frame, stale, or recovery active.
    bool process_frame(PreprocessedTensorLease& out_tensor_lease) noexcept;

    /// @brief Stops capture and releases pipeline resources.
    void stop() noexcept;

    /// @brief Returns current accumulated latency and drop metrics.
    [[nodiscard]] CaptureToTensorMetrics metrics() const noexcept;

    /// @brief Resets all latency and drop metrics.
    void reset_metrics() noexcept;

    /// @brief Binds downstream bus ring buffer to automatically push preprocessed tensor descriptors.
    void bind_tensor_bus_ring(bus::LatestSpscRing<PreprocessedTensorDescriptor, 16>* ring) noexcept;

    /// @brief Returns pointer to underlying frame source.
    [[nodiscard]] aim::IFrameSource* frame_source() noexcept { return frame_source_.get(); }

    /// @brief Returns pointer to underlying GPU preprocessor.
    [[nodiscard]] IGpuPreprocessor* preprocessor() noexcept { return preprocessor_.get(); }

    /// @brief Returns whether downstream actuation is currently permitted.
    [[nodiscard]] bool is_actuation_permitted() const noexcept;

    /// @brief Returns whether capture source and preprocessor are currently healthy.
    [[nodiscard]] bool is_healthy() const noexcept;

    /// @brief Returns active capture backend.
    [[nodiscard]] aim::FrameSourceBackend active_capture_backend() const noexcept;

private:
    static LatencyStats calculate_stats(const std::vector<double>& samples_ms) noexcept;
    bool refresh_cuda_surface_pool() noexcept;

    std::unique_ptr<aim::IFrameSource> frame_source_{nullptr};
    std::unique_ptr<IGpuPreprocessor> preprocessor_{nullptr};
    std::shared_ptr<IClock> clock_{nullptr};
    CaptureToTensorConfig config_{};
    std::shared_ptr<capture::ICudaInteropBackend> cuda_backend_{nullptr};
    std::unique_ptr<capture::CudaSurfacePool> cuda_surface_pool_{nullptr};
    const capture::GpuSurfacePool* registered_surface_pool_{nullptr};
    std::uint64_t registered_surface_generation_{0};
    std::uint64_t registered_adapter_luid_{0};
    bool uses_hardware_preprocessing_{false};

    bus::LatestSpscRing<PreprocessedTensorDescriptor, 16>* bus_ring_{nullptr};

    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};

    // Telemetry and latency sample accumulators (preallocated for benchmarking)
    mutable std::vector<double> capture_to_tensor_samples_ms_{};
    mutable std::vector<double> acquire_samples_ms_{};
    mutable std::vector<double> preprocess_samples_ms_{};

    std::uint64_t total_frames_processed_{0};
    std::uint64_t frames_acquired_{0};
    std::uint64_t frames_dropped_stale_{0};
    std::uint64_t frames_dropped_invalid_{0};
    std::uint64_t recovery_events_count_{0};

    aim::capture::CaptureRecoveryState last_recovery_state_{aim::capture::CaptureRecoveryState::uninitialized};
};

} // namespace aim::perception
