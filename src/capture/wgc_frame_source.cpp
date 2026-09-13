// src/capture/wgc_frame_source.cpp
#include "aim/capture/wgc_frame_source.hpp"
#include <algorithm>
#include <cstdio>

namespace aim::capture {

WgcFrameSource::WgcFrameSource(std::unique_ptr<IWgcBackend> backend,
                               std::shared_ptr<IClock> clock) noexcept
    : backend_(std::move(backend)),
      clock_(std::move(clock)) {
    if (!backend_) {
#if defined(_WIN32) || defined(_MSC_VER)
        backend_ = std::make_unique<RealWgcBackend>();
#else
        backend_ = std::make_unique<MockWgcBackend>();
#endif
    }
    if (!clock_) {
        clock_ = std::make_shared<QpcClock>();
    }
#if defined(_WIN32) || defined(_MSC_VER)
    LARGE_INTEGER freq;
    if (QueryPerformanceFrequency(&freq)) {
        qpf_ = static_cast<std::uint64_t>(freq.QuadPart);
        is_10mhz_ = (qpf_ == 10'000'000ULL);
    }
#else
    qpf_ = 10'000'000ULL;
    is_10mhz_ = true;
#endif
}

WgcFrameSource::~WgcFrameSource() noexcept {
    stop();
    surface_pool_.release_all();
    if (backend_) {
        backend_->release_all();
    }
}

WgcFrameSource::WgcFrameSource(WgcFrameSource&& other) noexcept
    : backend_(std::move(other.backend_)),
      clock_(std::move(other.clock_)),
      config_(other.config_),
      health_(other.health_),
      state_(other.state_.load(std::memory_order_relaxed)),
      surface_pool_(std::move(other.surface_pool_)),
      bus_ring_(other.bus_ring_),
      frame_sequence_(other.frame_sequence_),
      pipeline_run_id_(other.pipeline_run_id_),
      last_fps_sample_time_ns_(other.last_fps_sample_time_ns_),
      last_fps_sample_frame_count_(other.last_fps_sample_frame_count_),
      qpf_(other.qpf_),
      is_10mhz_(other.is_10mhz_) {
    other.state_.store(CaptureState::uninitialized, std::memory_order_relaxed);
    other.bus_ring_ = nullptr;
}

WgcFrameSource& WgcFrameSource::operator=(WgcFrameSource&& other) noexcept {
    if (this != &other) {
        stop();
        surface_pool_.release_all();
        if (backend_) {
            backend_->release_all();
        }

        backend_ = std::move(other.backend_);
        clock_ = std::move(other.clock_);
        config_ = other.config_;
        health_ = other.health_;
        state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        surface_pool_ = std::move(other.surface_pool_);
        bus_ring_ = other.bus_ring_;
        frame_sequence_ = other.frame_sequence_;
        pipeline_run_id_ = other.pipeline_run_id_;
        last_fps_sample_time_ns_ = other.last_fps_sample_time_ns_;
        last_fps_sample_frame_count_ = other.last_fps_sample_frame_count_;
        qpf_ = other.qpf_;
        is_10mhz_ = other.is_10mhz_;

        other.state_.store(CaptureState::uninitialized, std::memory_order_relaxed);
        other.bus_ring_ = nullptr;
    }
    return *this;
}

bool WgcFrameSource::initialize(const FrameSourceConfig& config) noexcept {
    if (config.pool_capacity < 2 || config.target_width_px == 0 || config.target_height_px == 0) {
        state_.store(CaptureState::failed, std::memory_order_release);
        return false;
    }

    config_ = config;

    // CUDA registrations observe this reset before backend reinitialization.
    surface_pool_.release_all();

    if (!backend_) {
#if defined(_WIN32) || defined(_MSC_VER)
        backend_ = std::make_unique<RealWgcBackend>();
#else
        backend_ = std::make_unique<MockWgcBackend>();
#endif
    }

    if (!backend_->initialize(config_)) {
        state_.store(CaptureState::failed, std::memory_order_release);
        health_.is_active = false;
        return false;
    }

    std::uint32_t width = config_.target_width_px;
    std::uint32_t height = config_.target_height_px;
    backend_->get_output_dimensions(width, height);
    if (width == 0 || height == 0) {
        width = config_.target_width_px;
        height = config_.target_height_px;
    }

    if (!surface_pool_.initialize(backend_.get(),
                                 config_.pool_capacity,
                                 width,
                                 height,
                                 FrameFormat::b8g8r8a8_unorm)) {
        state_.store(CaptureState::failed, std::memory_order_release);
        health_.is_active = false;
        return false;
    }

    state_.store(CaptureState::stopped, std::memory_order_release);
    health_.is_active = false;
    health_.is_access_lost = false;
    health_.is_cross_adapter = config_.allow_cross_adapter_copy;

    const std::uint64_t luid = backend_->adapter_luid();
    std::snprintf(health_.active_adapter_luid.data(),
                  health_.active_adapter_luid.size(),
                  "0x%016llX",
                  static_cast<unsigned long long>(luid));

    last_fps_sample_time_ns_ = clock_->now_ns();
    last_fps_sample_frame_count_ = 0;

    return true;
}

bool WgcFrameSource::start() noexcept {
    CaptureState current = state_.load(std::memory_order_acquire);
    if (current == CaptureState::running) {
        return true;
    }

    if (current == CaptureState::uninitialized || current == CaptureState::failed) {
        if (!initialize(config_)) {
            return false;
        }
    }

    if (!backend_->is_session_active()) {
        if (!backend_->create_capture_session()) {
            state_.store(CaptureState::failed, std::memory_order_release);
            health_.is_active = false;
            return false;
        }
    }

    state_.store(CaptureState::running, std::memory_order_release);
    health_.is_active = true;
    health_.is_access_lost = false;
    last_fps_sample_time_ns_ = clock_->now_ns();
    last_fps_sample_frame_count_ = health_.total_frames_acquired;
    return true;
}

void WgcFrameSource::stop() noexcept {
    state_.store(CaptureState::stopped, std::memory_order_release);
    health_.is_active = false;
    if (backend_) {
        backend_->release_capture_session();
    }
}

FrameSourceHealth WgcFrameSource::health() const noexcept {
    return health_;
}

void WgcFrameSource::publish_frame_descriptor(std::uint32_t pool_slot_index,
                                             SequenceId frame_id,
                                             MonotonicNs captured_at_ns) noexcept {
    if (!bus_ring_) {
        return;
    }

    bus::FrameDescriptor desc{};
    desc.schema_major = 1;
    desc.schema_minor = 0;
    desc.header.sequence_id = frame_id;
    desc.header.source_timestamp_ns = captured_at_ns;
    desc.header.pipeline_run_id = pipeline_run_id_;
    desc.header.flags = 0;
    desc.frame_id = frame_id;
    desc.captured_at_ns = captured_at_ns;
    desc.width = surface_pool_.width();
    desc.height = surface_pool_.height();
    desc.format = bus::FramePixelFormat::b8g8r8a8_unorm;
    desc.pool_slot_index = pool_slot_index;
    desc.shared_nt_handle = surface_pool_.get_shared_handle(pool_slot_index);
    desc.adapter_luid = backend_ ? backend_->adapter_luid() : 0;
    desc.is_keyframe = true;

    bus_ring_->push(desc);
}

bool WgcFrameSource::try_acquire_latest(FrameLease& out_lease) noexcept {
    CaptureState s = state_.load(std::memory_order_acquire);

    if (s != CaptureState::running) {
        out_lease.reset();
        return false;
    }

    CapturedRawFrame raw_frame{};
    CapturedRawFrame latest_raw_frame{};
    bool has_frame = false;

    while (true) {
        const HRESULT hr = backend_->try_get_next_frame(raw_frame);
        if (hr == S_OK) {
            if (has_frame) {
                // The backend has already replaced/released the old frame;
                // releasing here would close the newly acquired candidate.
                ++health_.total_frames_dropped;
            }
            latest_raw_frame = raw_frame;
            has_frame = true;
        } else if (hr == DXGI_ERROR_WAIT_TIMEOUT || hr == S_FALSE) {
            break;
        } else if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_ACCESS_DENIED || hr == E_ACCESSDENIED) {
            health_.is_access_lost = true;
            health_.is_active = false;
            ++health_.total_access_loss_events;
            backend_->release_capture_session();
            state_.store(CaptureState::access_lost, std::memory_order_release);
            out_lease.reset();
            return false;
        } else if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            health_.is_active = false;
            surface_pool_.release_all();
            backend_->release_all();
            state_.store(CaptureState::device_lost, std::memory_order_release);
            out_lease.reset();
            return false;
        } else {
            break;
        }
    }

    if (!has_frame) {
        ++health_.total_timeouts;
        out_lease.reset();
        return false;
    }

    if (latest_raw_frame.width != 0 && latest_raw_frame.height != 0 &&
        (latest_raw_frame.width != surface_pool_.width() || latest_raw_frame.height != surface_pool_.height())) {
        surface_pool_.resize(backend_.get(), latest_raw_frame.width, latest_raw_frame.height);
    }

    const std::uint32_t slot = surface_pool_.acquire_free_slot();
    if (slot == GpuSurfacePool::kInvalidSlot) {
        ++health_.total_frames_dropped;
        backend_->release_frame();
        out_lease.reset();
        return false;
    }

    void* staging_tex = surface_pool_.get_texture(slot);
    backend_->copy_texture(staging_tex, latest_raw_frame.raw_texture);
    backend_->release_frame();

    MonotonicNs captured_at_ns = 0;
    if (latest_raw_frame.last_present_time_qpc != 0) {
        captured_at_ns = is_10mhz_
            ? static_cast<MonotonicNs>(latest_raw_frame.last_present_time_qpc * 100ULL)
            : qpc_to_ns_128(latest_raw_frame.last_present_time_qpc, qpf_);
    } else {
        captured_at_ns = clock_->now_ns();
    }

    const SequenceId seq = ++frame_sequence_;
    out_lease = surface_pool_.create_lease(slot, seq, captured_at_ns);

    ++health_.total_frames_acquired;
    health_.last_frame_timestamp_ns = captured_at_ns;

    const MonotonicNs now_ns = clock_->now_ns();
    if (now_ns - last_fps_sample_time_ns_ >= 500'000'000LL) {
        const double elapsed_s = static_cast<double>(now_ns - last_fps_sample_time_ns_) / 1e9;
        const double frames = static_cast<double>(health_.total_frames_acquired - last_fps_sample_frame_count_);
        health_.current_cadence_fps = (elapsed_s > 0.0) ? (frames / elapsed_s) : 0.0;
        last_fps_sample_time_ns_ = now_ns;
        last_fps_sample_frame_count_ = health_.total_frames_acquired;
    }

    publish_frame_descriptor(slot, seq, captured_at_ns);
    return true;
}

} // namespace aim::capture
