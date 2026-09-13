// include/aim/capture/dxgi_frame_source.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include "aim/bus/bus.hpp"
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"

namespace aim::capture {

/// @brief Operational state of the DXGI Desktop Duplication capture loop.
enum class CaptureState : std::uint8_t {
    uninitialized  = 0,
    stopped        = 1,
    running        = 2,
    access_lost    = 3,
    reinitializing = 4,
    backoff_wait   = 5,
    device_lost    = 6,
    failed         = 7
};

/// @brief Primary visual capture frame source conforming to IFrameSource with zero-allocation pooling.
class DxgiFrameSource final : public IFrameSource {
public:
    explicit DxgiFrameSource(std::unique_ptr<IDxgiBackend> backend = nullptr,
                             std::shared_ptr<IClock> clock = std::make_shared<QpcClock>()) noexcept;
    ~DxgiFrameSource() noexcept override;

    DxgiFrameSource(const DxgiFrameSource&) = delete;
    DxgiFrameSource& operator=(const DxgiFrameSource&) = delete;
    DxgiFrameSource(DxgiFrameSource&&) noexcept;
    DxgiFrameSource& operator=(DxgiFrameSource&&) noexcept;

    // IFrameSource interface
    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool start() noexcept override;
    bool try_acquire_latest(FrameLease& out_lease) noexcept override;
    void stop() noexcept override;
    [[nodiscard]] FrameSourceHealth health() const noexcept override;

    // Subsystem inspection and configuration
    [[nodiscard]] CaptureState state() const noexcept { return state_.load(std::memory_order_acquire); }
    [[nodiscard]] const GpuSurfacePool& surface_pool() const noexcept { return surface_pool_; }
    [[nodiscard]] GpuSurfacePool& surface_pool() noexcept { return surface_pool_; }
    [[nodiscard]] const IDxgiBackend* backend() const noexcept { return backend_.get(); }
    [[nodiscard]] IDxgiBackend* backend() noexcept { return backend_.get(); }
    [[nodiscard]] std::uint64_t qpf() const noexcept { return qpf_; }
    [[nodiscard]] bool is_10mhz() const noexcept { return is_10mhz_; }

    /// @brief Overrides QPF for deterministic timestamp unit testing.
    void set_qpf_for_testing(std::uint64_t qpf) noexcept {
        qpf_ = qpf;
        is_10mhz_ = (qpf == 10'000'000ULL);
    }

    /// @brief Binds the in-process bus ring to publish FrameDescriptor messages directly.
    void bind_bus_ring(bus::LatestSpscRing<bus::FrameDescriptor, 16>* ring) noexcept {
        bus_ring_ = ring;
    }

    /// @brief Sets the pipeline run ID for correlation tracking in published FrameDescriptors.
    void set_pipeline_run_id(std::uint32_t run_id) noexcept {
        pipeline_run_id_ = run_id;
    }

private:
    bool attempt_reinitialization() noexcept;
    bool handle_device_lost() noexcept;
    void publish_frame_descriptor(std::uint32_t pool_slot_index,
                                  SequenceId frame_id,
                                  MonotonicNs captured_at_ns) noexcept;

    static constexpr MonotonicNs kInitialBackoffNs = 10'000'000LL;  // 10 ms
    static constexpr MonotonicNs kMaxBackoffNs     = 250'000'000LL; // 250 ms

    std::unique_ptr<IDxgiBackend> backend_{nullptr};
    std::shared_ptr<IClock> clock_{nullptr};
    FrameSourceConfig config_{};
    mutable FrameSourceHealth health_{};
    std::atomic<CaptureState> state_{CaptureState::uninitialized};

    GpuSurfacePool surface_pool_{};
    bus::LatestSpscRing<bus::FrameDescriptor, 16>* bus_ring_{nullptr};

    SequenceId frame_sequence_{0};
    std::uint32_t pipeline_run_id_{1};
    MonotonicNs last_reinit_attempt_ns_{0};
    MonotonicNs current_backoff_ns_{kInitialBackoffNs};
    MonotonicNs last_fps_sample_time_ns_{0};
    std::uint64_t last_fps_sample_frame_count_{0};
    std::uint64_t qpf_{10'000'000ULL};
    bool is_10mhz_{true};
};

} // namespace aim::capture
