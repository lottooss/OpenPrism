// include/aim/capture/unified_capture_source.hpp
#pragma once

#include <atomic>
#include <memory>
#include "aim/bus/bus.hpp"
#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/capture_recovery_state_machine.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/dxgi_frame_source.hpp"
#include "aim/capture/wgc_backend.hpp"
#include "aim/capture/wgc_frame_source.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"

namespace aim::capture {

/// @brief Unified multi-backend capture source with fail-closed recovery and automatic fallback between DXGI and WGC.
class UnifiedCaptureSource final : public IFrameSource {
public:
    explicit UnifiedCaptureSource(std::unique_ptr<IFrameSource> dxgi_source = nullptr,
                                  std::unique_ptr<IFrameSource> wgc_source = nullptr,
                                  std::shared_ptr<IClock> clock = std::make_shared<QpcClock>()) noexcept;

    UnifiedCaptureSource(std::unique_ptr<IDxgiBackend> dxgi_backend,
                         std::unique_ptr<IWgcBackend> wgc_backend,
                         std::shared_ptr<IClock> clock = std::make_shared<QpcClock>()) noexcept;

    ~UnifiedCaptureSource() noexcept override;

    UnifiedCaptureSource(const UnifiedCaptureSource&) = delete;
    UnifiedCaptureSource& operator=(const UnifiedCaptureSource&) = delete;
    UnifiedCaptureSource(UnifiedCaptureSource&&) noexcept;
    UnifiedCaptureSource& operator=(UnifiedCaptureSource&&) noexcept;

    // IFrameSource interface
    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool start() noexcept override;
    bool try_acquire_latest(FrameLease& out_lease) noexcept override;
    void stop() noexcept override;
    [[nodiscard]] FrameSourceHealth health() const noexcept override;

    // State inspection
    [[nodiscard]] CaptureRecoveryState recovery_state() const noexcept { return state_machine_.current_state(); }
    [[nodiscard]] FrameSourceBackend active_backend() const noexcept { return state_machine_.active_backend(); }
    [[nodiscard]] bool is_actuation_permitted() const noexcept { return state_machine_.is_actuation_permitted(); }
    [[nodiscard]] GpuSurfacePool* active_surface_pool() noexcept;
    [[nodiscard]] const GpuSurfacePool* active_surface_pool() const noexcept;
    [[nodiscard]] std::uint64_t active_adapter_luid() const noexcept;

    void bind_bus_ring(bus::LatestSpscRing<bus::FrameDescriptor, 16>* ring) noexcept;
    void set_pipeline_run_id(std::uint32_t run_id) noexcept;
    void set_dxgi_probe_interval_ns(MonotonicNs interval_ns) noexcept { dxgi_probe_interval_ns_ = interval_ns; }

private:
    bool attempt_recovery() noexcept;
    bool probe_dxgi_promotion() noexcept;

    std::unique_ptr<IFrameSource> dxgi_source_{nullptr};
    std::unique_ptr<IFrameSource> wgc_source_{nullptr};
    std::shared_ptr<IClock> clock_{nullptr};
    FrameSourceConfig config_{};
    CaptureRecoveryStateMachine state_machine_{};

    bus::LatestSpscRing<bus::FrameDescriptor, 16>* bus_ring_{nullptr};
    std::uint32_t pipeline_run_id_{1};

    MonotonicNs last_dxgi_probe_ns_{0};
    MonotonicNs dxgi_probe_interval_ns_{5'000'000'000LL}; // 5.0 seconds
};

} // namespace aim::capture
