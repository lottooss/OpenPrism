// src/capture/unified_capture_source.cpp
#include "aim/capture/unified_capture_source.hpp"
#include <utility>

namespace aim::capture {

UnifiedCaptureSource::UnifiedCaptureSource(std::unique_ptr<IFrameSource> dxgi_source,
                                           std::unique_ptr<IFrameSource> wgc_source,
                                           std::shared_ptr<IClock> clock) noexcept
    : dxgi_source_(std::move(dxgi_source)),
      wgc_source_(std::move(wgc_source)),
      clock_(std::move(clock)) {
    if (!clock_) {
        clock_ = std::make_shared<QpcClock>();
    }
}

UnifiedCaptureSource::UnifiedCaptureSource(std::unique_ptr<IDxgiBackend> dxgi_backend,
                                           std::unique_ptr<IWgcBackend> wgc_backend,
                                           std::shared_ptr<IClock> clock) noexcept
    : clock_(std::move(clock)) {
    if (!clock_) {
        clock_ = std::make_shared<QpcClock>();
    }
    dxgi_source_ = std::make_unique<DxgiFrameSource>(std::move(dxgi_backend), clock_);
    wgc_source_ = std::make_unique<WgcFrameSource>(std::move(wgc_backend), clock_);
}

UnifiedCaptureSource::~UnifiedCaptureSource() noexcept {
    stop();
}

UnifiedCaptureSource::UnifiedCaptureSource(UnifiedCaptureSource&& other) noexcept
    : dxgi_source_(std::move(other.dxgi_source_)),
      wgc_source_(std::move(other.wgc_source_)),
      clock_(std::move(other.clock_)),
      config_(other.config_),
      state_machine_(other.state_machine_),
      bus_ring_(other.bus_ring_),
      pipeline_run_id_(other.pipeline_run_id_),
      last_dxgi_probe_ns_(other.last_dxgi_probe_ns_),
      dxgi_probe_interval_ns_(other.dxgi_probe_interval_ns_) {
    other.bus_ring_ = nullptr;
}

UnifiedCaptureSource& UnifiedCaptureSource::operator=(UnifiedCaptureSource&& other) noexcept {
    if (this != &other) {
        stop();

        dxgi_source_ = std::move(other.dxgi_source_);
        wgc_source_ = std::move(other.wgc_source_);
        clock_ = std::move(other.clock_);
        config_ = other.config_;
        state_machine_ = other.state_machine_;
        bus_ring_ = other.bus_ring_;
        pipeline_run_id_ = other.pipeline_run_id_;
        last_dxgi_probe_ns_ = other.last_dxgi_probe_ns_;
        dxgi_probe_interval_ns_ = other.dxgi_probe_interval_ns_;

        other.bus_ring_ = nullptr;
    }
    return *this;
}

bool UnifiedCaptureSource::initialize(const FrameSourceConfig& config) noexcept {
    config_ = config;

    if (!dxgi_source_) {
        dxgi_source_ = std::make_unique<DxgiFrameSource>(nullptr, clock_);
    }
    if (!wgc_source_) {
        wgc_source_ = std::make_unique<WgcFrameSource>(nullptr, clock_);
    }

    if (bus_ring_) {
        bind_bus_ring(bus_ring_);
    }
    set_pipeline_run_id(pipeline_run_id_);

    state_machine_.reset();

    if (config_.backend == FrameSourceBackend::windows_graphics_capture) {
        if (wgc_source_->initialize(config_)) {
            state_machine_.transition(CaptureEventTrigger::fallback_to_wgc, clock_->now_ns());
            state_machine_.transition(CaptureEventTrigger::initialize_success, clock_->now_ns());
            return true;
        }
        state_machine_.transition(CaptureEventTrigger::initialize_failure, clock_->now_ns());
        return false;
    }

    // Default: try DXGI first
    if (dxgi_source_->initialize(config_)) {
        state_machine_.transition(CaptureEventTrigger::promote_to_dxgi, clock_->now_ns());
        state_machine_.transition(CaptureEventTrigger::initialize_success, clock_->now_ns());
        return true;
    }

    // DXGI failed, try fallback to WGC if enabled
    if (config_.fallback_to_wgc && wgc_source_->initialize(config_)) {
        state_machine_.transition(CaptureEventTrigger::fallback_to_wgc, clock_->now_ns());
        state_machine_.transition(CaptureEventTrigger::initialize_success, clock_->now_ns());
        return true;
    }

    state_machine_.transition(CaptureEventTrigger::initialize_failure, clock_->now_ns());
    return false;
}

bool UnifiedCaptureSource::start() noexcept {
    const auto s = state_machine_.current_state();
    if (s == CaptureRecoveryState::active_dxgi || s == CaptureRecoveryState::active_wgc) {
        return true;
    }

    if (s == CaptureRecoveryState::uninitialized || s == CaptureRecoveryState::failed) {
        if (!initialize(config_)) {
            return false;
        }
    }

    bool started = false;
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        started = dxgi_source_->start();
    } else {
        started = wgc_source_->start();
    }

    if (started) {
        state_machine_.transition(CaptureEventTrigger::start_command, clock_->now_ns());
        last_dxgi_probe_ns_ = clock_->now_ns();
        return true;
    }

    // If starting DXGI failed, try fallback
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication && config_.fallback_to_wgc) {
        if (wgc_source_->initialize(config_) && wgc_source_->start()) {
            state_machine_.transition(CaptureEventTrigger::fallback_to_wgc, clock_->now_ns());
            last_dxgi_probe_ns_ = clock_->now_ns();
            return true;
        }
    }

    state_machine_.transition(CaptureEventTrigger::reinit_failure, clock_->now_ns());
    return false;
}

void UnifiedCaptureSource::stop() noexcept {
    if (dxgi_source_) {
        dxgi_source_->stop();
    }
    if (wgc_source_) {
        wgc_source_->stop();
    }
    state_machine_.transition(CaptureEventTrigger::stop_command, clock_->now_ns());
}

FrameSourceHealth UnifiedCaptureSource::health() const noexcept {
    FrameSourceHealth h{};
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication && dxgi_source_) {
        h = dxgi_source_->health();
    } else if (wgc_source_) {
        h = wgc_source_->health();
    }

    h.is_active = is_actuation_permitted();
    h.is_access_lost = (state_machine_.current_state() == CaptureRecoveryState::access_lost ||
                        state_machine_.current_state() == CaptureRecoveryState::backoff_wait);
    return h;
}

const GpuSurfacePool* UnifiedCaptureSource::active_surface_pool() const noexcept {
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        const auto* dxgi = dynamic_cast<const DxgiFrameSource*>(dxgi_source_.get());
        return dxgi != nullptr ? &dxgi->surface_pool() : nullptr;
    }
    const auto* wgc = dynamic_cast<const WgcFrameSource*>(wgc_source_.get());
    return wgc != nullptr ? &wgc->surface_pool() : nullptr;
}

GpuSurfacePool* UnifiedCaptureSource::active_surface_pool() noexcept {
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        auto* dxgi = dynamic_cast<DxgiFrameSource*>(dxgi_source_.get());
        return dxgi != nullptr ? &dxgi->surface_pool() : nullptr;
    }
    auto* wgc = dynamic_cast<WgcFrameSource*>(wgc_source_.get());
    return wgc != nullptr ? &wgc->surface_pool() : nullptr;
}

std::uint64_t UnifiedCaptureSource::active_adapter_luid() const noexcept {
    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        const auto* dxgi = dynamic_cast<const DxgiFrameSource*>(dxgi_source_.get());
        return dxgi != nullptr && dxgi->backend() != nullptr ? dxgi->backend()->adapter_luid() : 0;
    }
    const auto* wgc = dynamic_cast<const WgcFrameSource*>(wgc_source_.get());
    return wgc != nullptr && wgc->backend() != nullptr ? wgc->backend()->adapter_luid() : 0;
}

void UnifiedCaptureSource::bind_bus_ring(bus::LatestSpscRing<bus::FrameDescriptor, 16>* ring) noexcept {
    bus_ring_ = ring;
    if (auto* dxgi = dynamic_cast<DxgiFrameSource*>(dxgi_source_.get())) {
        dxgi->bind_bus_ring(ring);
    }
    if (auto* wgc = dynamic_cast<WgcFrameSource*>(wgc_source_.get())) {
        wgc->bind_bus_ring(ring);
    }
}

void UnifiedCaptureSource::set_pipeline_run_id(std::uint32_t run_id) noexcept {
    pipeline_run_id_ = run_id;
    if (auto* dxgi = dynamic_cast<DxgiFrameSource*>(dxgi_source_.get())) {
        dxgi->set_pipeline_run_id(run_id);
    }
    if (auto* wgc = dynamic_cast<WgcFrameSource*>(wgc_source_.get())) {
        wgc->set_pipeline_run_id(run_id);
    }
}

bool UnifiedCaptureSource::attempt_recovery() noexcept {
    const MonotonicNs now_ns = clock_->now_ns();
    state_machine_.transition(CaptureEventTrigger::backoff_expired, now_ns);

    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        if (dxgi_source_->initialize(config_) && dxgi_source_->start()) {
            state_machine_.transition(CaptureEventTrigger::reinit_success, now_ns);
            return true;
        }

        if (config_.fallback_to_wgc) {
            if (wgc_source_->initialize(config_) && wgc_source_->start()) {
                state_machine_.transition(CaptureEventTrigger::fallback_to_wgc, now_ns);
                return true;
            }
        }
    } else {
        if (wgc_source_->initialize(config_) && wgc_source_->start()) {
            state_machine_.transition(CaptureEventTrigger::reinit_success, now_ns);
            return true;
        }
    }

    state_machine_.transition(CaptureEventTrigger::reinit_failure, now_ns);
    return false;
}

bool UnifiedCaptureSource::probe_dxgi_promotion() noexcept {
    if (!dxgi_source_ || config_.backend == FrameSourceBackend::windows_graphics_capture) {
        return false;
    }

    if (dxgi_source_->start()) {
        FrameLease probe_lease;
        if (dxgi_source_->try_acquire_latest(probe_lease)) {
            wgc_source_->stop();
            state_machine_.transition(CaptureEventTrigger::promote_to_dxgi, clock_->now_ns());
            state_machine_.transition(CaptureEventTrigger::frame_acquired_ok, clock_->now_ns());
            return true;
        }
    }

    return false;
}

bool UnifiedCaptureSource::try_acquire_latest(FrameLease& out_lease) noexcept {
    const auto s = state_machine_.current_state();

    if (s == CaptureRecoveryState::uninitialized ||
        s == CaptureRecoveryState::stopped ||
        s == CaptureRecoveryState::failed) {
        out_lease.reset();
        return false;
    }

    if (s == CaptureRecoveryState::access_lost ||
        s == CaptureRecoveryState::backoff_wait ||
        s == CaptureRecoveryState::reinitializing ||
        s == CaptureRecoveryState::device_lost) {
        const MonotonicNs now_ns = clock_->now_ns();
        if (now_ns > state_machine_.last_state_change_ns() &&
            (now_ns - state_machine_.last_state_change_ns() < state_machine_.current_backoff_ns())) {
            out_lease.reset();
            return false;
        }

        if (!attempt_recovery()) {
            out_lease.reset();
            return false;
        }
    }

    if (state_machine_.active_backend() == FrameSourceBackend::dxgi_duplication) {
        if (dxgi_source_->try_acquire_latest(out_lease)) {
            state_machine_.transition(CaptureEventTrigger::frame_acquired_ok, clock_->now_ns());
            return true;
        }

        const auto dxgi_health = dxgi_source_->health();
        if (dxgi_health.is_access_lost) {
            state_machine_.transition(CaptureEventTrigger::access_lost, clock_->now_ns());

            if (config_.fallback_to_wgc && wgc_source_) {
                if (wgc_source_->initialize(config_) && wgc_source_->start()) {
                    if (wgc_source_->try_acquire_latest(out_lease)) {
                        state_machine_.transition(CaptureEventTrigger::fallback_to_wgc, clock_->now_ns());
                        state_machine_.transition(CaptureEventTrigger::frame_acquired_ok, clock_->now_ns());
                        return true;
                    }
                }
            }
        } else if (!dxgi_health.is_active) {
            state_machine_.transition(CaptureEventTrigger::device_lost, clock_->now_ns());
        }

        out_lease.reset();
        return false;
    }

    if (state_machine_.active_backend() == FrameSourceBackend::windows_graphics_capture) {
        const MonotonicNs now_ns = clock_->now_ns();
        if (now_ns - last_dxgi_probe_ns_ >= dxgi_probe_interval_ns_) {
            last_dxgi_probe_ns_ = now_ns;
            if (dxgi_source_->start()) {
                if (dxgi_source_->try_acquire_latest(out_lease)) {
                    wgc_source_->stop();
                    state_machine_.transition(CaptureEventTrigger::promote_to_dxgi, now_ns);
                    state_machine_.transition(CaptureEventTrigger::frame_acquired_ok, now_ns);
                    return true;
                }
            }
        }

        if (wgc_source_->try_acquire_latest(out_lease)) {
            state_machine_.transition(CaptureEventTrigger::frame_acquired_ok, clock_->now_ns());
            return true;
        }

        const auto wgc_health = wgc_source_->health();
        if (wgc_health.is_access_lost) {
            state_machine_.transition(CaptureEventTrigger::access_lost, clock_->now_ns());
        } else if (!wgc_health.is_active) {
            state_machine_.transition(CaptureEventTrigger::device_lost, clock_->now_ns());
        }

        out_lease.reset();
        return false;
    }

    out_lease.reset();
    return false;
}

} // namespace aim::capture
