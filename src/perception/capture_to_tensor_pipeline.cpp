// src/perception/capture_to_tensor_pipeline.cpp
#include "aim/perception/capture_to_tensor_pipeline.hpp"

namespace aim::perception {

CaptureToTensorPipeline::CaptureToTensorPipeline(
    std::unique_ptr<aim::IFrameSource> frame_source,
    std::unique_ptr<IGpuPreprocessor> preprocessor,
    std::shared_ptr<IClock> clock) noexcept
    : frame_source_(std::move(frame_source)),
      preprocessor_(std::move(preprocessor)),
      clock_(std::move(clock)) {
    if (!clock_) {
        clock_ = std::make_shared<QpcClock>();
    }
    if (!frame_source_) {
        frame_source_ = std::make_unique<capture::UnifiedCaptureSource>(
            std::unique_ptr<aim::IFrameSource>{nullptr},
            std::unique_ptr<aim::IFrameSource>{nullptr},
            clock_);
    }
    if (!preprocessor_) {
        preprocessor_ = std::make_unique<GpuPreprocessor>();
    }

    // Pre-reserve telemetry capacity for zero-allocation benchmark runs
    capture_to_tensor_samples_ms_.reserve(50000);
    acquire_samples_ms_.reserve(50000);
    preprocess_samples_ms_.reserve(50000);
}

CaptureToTensorPipeline::~CaptureToTensorPipeline() noexcept {
    stop();
}

bool CaptureToTensorPipeline::initialize(
    const CaptureToTensorConfig& config,
    std::shared_ptr<capture::ICudaInteropBackend> cuda_backend) noexcept {
    stop();
    config_ = config;
    cuda_backend_ = std::move(cuda_backend);
    if (!cuda_backend_) {
        cuda_backend_ = std::make_shared<capture::RealCudaInteropBackend>();
    }

    if (!frame_source_->initialize(config.capture_config)) {
        return false;
    }

    // Register capture textures first so the CUDA device/context is selected before
    // the production preprocessor allocates its device tensor pool.
    static_cast<void>(refresh_cuda_surface_pool());

    if (!preprocessor_->initialize(config.preprocess_config, cuda_backend_)) {
        if (cuda_surface_pool_) {
            cuda_surface_pool_->release_all();
        }
        cuda_backend_.reset();
        return false;
    }

    uses_hardware_preprocessing_ = preprocessor_->health().is_hardware_accelerated;
    if (uses_hardware_preprocessing_ &&
        (!cuda_surface_pool_ || !cuda_surface_pool_->is_initialized())) {
        preprocessor_->shutdown();
        cuda_backend_.reset();
        return false;
    }

    is_initialized_.store(true, std::memory_order_release);
    return true;
}

bool CaptureToTensorPipeline::refresh_cuda_surface_pool() noexcept {
    auto* unified = dynamic_cast<capture::UnifiedCaptureSource*>(frame_source_.get());
    if (unified == nullptr || cuda_backend_ == nullptr) {
        return false;
    }

    capture::GpuSurfacePool* active_pool = unified->active_surface_pool();
    const std::uint64_t adapter_luid = unified->active_adapter_luid();
    if (active_pool == nullptr || !active_pool->is_initialized() || adapter_luid == 0) {
        return false;
    }

    if (cuda_surface_pool_ && cuda_surface_pool_->is_initialized() &&
        registered_surface_pool_ == active_pool &&
        registered_surface_generation_ == active_pool->generation() &&
        registered_adapter_luid_ == adapter_luid) {
        cuda_surface_pool_->reap_completed_releases();
        return true;
    }

    if (!cuda_surface_pool_) {
        cuda_surface_pool_ = std::make_unique<capture::CudaSurfacePool>(cuda_backend_);
    }
    if (!cuda_surface_pool_->initialize(*active_pool, adapter_luid)) {
        registered_surface_pool_ = nullptr;
        registered_surface_generation_ = 0;
        registered_adapter_luid_ = 0;
        return false;
    }

    registered_surface_pool_ = active_pool;
    registered_surface_generation_ = active_pool->generation();
    registered_adapter_luid_ = adapter_luid;
    return true;
}

bool CaptureToTensorPipeline::warmup(std::uint32_t warmup_iterations) noexcept {
    if (!is_initialized_.load(std::memory_order_acquire)) {
        return false;
    }

    if (!preprocessor_->warmup(warmup_iterations)) {
        return false;
    }

    // Reset metric counters after warmup to clear startup transients
    reset_metrics();
    return true;
}

bool CaptureToTensorPipeline::start() noexcept {
    if (!is_initialized_.load(std::memory_order_acquire)) {
        return false;
    }

    if (!frame_source_->start()) {
        return false;
    }

    is_running_.store(true, std::memory_order_release);
    return true;
}

bool CaptureToTensorPipeline::process_frame(PreprocessedTensorLease& out_tensor_lease) noexcept {
    out_tensor_lease.reset();
    ++total_frames_processed_;

    if (!is_running_.load(std::memory_order_acquire)) {
        return false;
    }

    // Check capture health & access loss
    auto health_state = frame_source_->health();
    if (!health_state.is_active || health_state.is_access_lost) {
        ++frames_dropped_invalid_;
        return false;
    }

    MonotonicNs t_start_acquire_ns = clock_->now_ns();
    aim::FrameLease frame_lease{};
    if (!frame_source_->try_acquire_latest(frame_lease)) {
        return false; // No new frame available on this tick
    }

    MonotonicNs t_end_acquire_ns = clock_->now_ns();
    ++frames_acquired_;

    if (!frame_lease.is_valid()) {
        ++frames_dropped_invalid_;
        return false;
    }

    // Stale frame cutoff check
    MonotonicNs captured_at_ns = frame_lease.captured_at_ns();
    if (t_end_acquire_ns > captured_at_ns &&
        (t_end_acquire_ns - captured_at_ns) > config_.stale_threshold_ns) {
        ++frames_dropped_stale_;
        return false;
    }

    // Execute the production CUDA surface path, or the explicit CI mock path.
    MonotonicNs t_start_preprocess_ns = clock_->now_ns();
    CorrelationId corr{frame_lease.frame_id(), captured_at_ns, 1, 0};
    bool preprocessed = false;
    if (uses_hardware_preprocessing_) {
        if (!refresh_cuda_surface_pool()) {
            ++frames_dropped_invalid_;
            return false;
        }
        capture::CudaMappedSurfaceLease mapped_surface{};
        if (!cuda_surface_pool_->map_surface(std::move(frame_lease), mapped_surface)) {
            ++frames_dropped_invalid_;
            return false;
        }
        preprocessed = preprocessor_->preprocess_cuda_surface(mapped_surface, out_tensor_lease);
    } else {
        const std::uint32_t stride_bytes =
            frame_lease.width_px() * (frame_lease.format() == FrameFormat::nv12 ? 1U : 4U);
        preprocessed = preprocessor_->preprocess_raw_memory(
            frame_lease.native_texture_ptr(), frame_lease.width_px(), frame_lease.height_px(),
            stride_bytes, frame_lease.format(), frame_lease.frame_id(), corr,
            captured_at_ns, out_tensor_lease);
    }

    if (!preprocessed) {
        ++frames_dropped_invalid_;
        return false;
    }

    MonotonicNs t_end_preprocess_ns = clock_->now_ns();

    // Attach high-resolution timing metadata
    out_tensor_lease.descriptor().preprocessed_at_ns = t_end_preprocess_ns;

    // Publish to downstream bus ring if connected
    if (bus_ring_ != nullptr) {
        bus_ring_->push(out_tensor_lease.descriptor());
    }

    // Record stage latencies
    double capture_to_tensor_ms = static_cast<double>(t_end_preprocess_ns - t_start_acquire_ns) / 1'000'000.0;
    double acquire_ms = static_cast<double>(t_end_acquire_ns - t_start_acquire_ns) / 1'000'000.0;
    double preprocess_ms = static_cast<double>(t_end_preprocess_ns - t_start_preprocess_ns) / 1'000'000.0;

    if (capture_to_tensor_samples_ms_.size() < capture_to_tensor_samples_ms_.capacity()) {
        capture_to_tensor_samples_ms_.push_back(capture_to_tensor_ms);
        acquire_samples_ms_.push_back(acquire_ms);
        preprocess_samples_ms_.push_back(preprocess_ms);
    }

    return true;
}

void CaptureToTensorPipeline::stop() noexcept {
    is_running_.store(false, std::memory_order_release);
    is_initialized_.store(false, std::memory_order_release);
    if (preprocessor_) {
        preprocessor_->shutdown();
    }
    if (cuda_surface_pool_) {
        cuda_surface_pool_->release_all();
    }
    if (frame_source_) {
        frame_source_->stop();
    }
    registered_surface_pool_ = nullptr;
    registered_surface_generation_ = 0;
    registered_adapter_luid_ = 0;
    uses_hardware_preprocessing_ = false;
    cuda_backend_.reset();
}

void CaptureToTensorPipeline::reset_metrics() noexcept {
    total_frames_processed_ = 0;
    frames_acquired_ = 0;
    frames_dropped_stale_ = 0;
    frames_dropped_invalid_ = 0;
    recovery_events_count_ = 0;

    capture_to_tensor_samples_ms_.clear();
    acquire_samples_ms_.clear();
    preprocess_samples_ms_.clear();
}

void CaptureToTensorPipeline::bind_tensor_bus_ring(
    bus::LatestSpscRing<PreprocessedTensorDescriptor, 16>* ring) noexcept {
    bus_ring_ = ring;
}

LatencyStats CaptureToTensorPipeline::calculate_stats(const std::vector<double>& samples_ms) noexcept {
    if (samples_ms.empty()) {
        return LatencyStats{};
    }

    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());

    std::size_t n = sorted.size();
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);

    LatencyStats stats{};
    stats.sample_count = static_cast<std::uint64_t>(n);
    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();
    stats.avg_ms = sum / static_cast<double>(n);

    stats.p50_ms = sorted[static_cast<std::size_t>(static_cast<double>(n) * 0.50)];
    stats.p95_ms = sorted[static_cast<std::size_t>(static_cast<double>(n) * 0.95)];
    stats.p99_ms = sorted[static_cast<std::size_t>(static_cast<double>(n) * 0.99)];

    return stats;
}

CaptureToTensorMetrics CaptureToTensorPipeline::metrics() const noexcept {
    CaptureToTensorMetrics m{};
    m.total_frames_processed = total_frames_processed_;
    m.frames_acquired = frames_acquired_;
    m.frames_dropped_stale = frames_dropped_stale_;
    m.frames_dropped_invalid = frames_dropped_invalid_;
    m.recovery_events_count = recovery_events_count_;

    m.capture_to_tensor_latency = calculate_stats(capture_to_tensor_samples_ms_);
    m.stage_acquire_latency = calculate_stats(acquire_samples_ms_);
    m.stage_preprocess_latency = calculate_stats(preprocess_samples_ms_);

    return m;
}

bool CaptureToTensorPipeline::is_healthy() const noexcept {
    if (!frame_source_ || !preprocessor_) {
        return false;
    }
    const auto capture_health = frame_source_->health();
    const auto preprocess_health = preprocessor_->health();
    return capture_health.is_active && !capture_health.is_access_lost &&
        preprocess_health.is_initialized;
}

bool CaptureToTensorPipeline::is_actuation_permitted() const noexcept {
    if (!frame_source_ || !preprocessor_) {
        return false;
    }
    if (!preprocessor_->health().is_initialized) {
        return false;
    }
    auto* unified = dynamic_cast<capture::UnifiedCaptureSource*>(frame_source_.get());
    if (unified) {
        return unified->is_actuation_permitted();
    }
    return frame_source_->health().is_active && !frame_source_->health().is_access_lost;
}

aim::FrameSourceBackend CaptureToTensorPipeline::active_capture_backend() const noexcept {
    if (!frame_source_) {
        return aim::FrameSourceBackend::dxgi_duplication;
    }
    auto* unified = dynamic_cast<capture::UnifiedCaptureSource*>(frame_source_.get());
    if (unified) {
        return unified->active_backend();
    }
    return aim::FrameSourceBackend::dxgi_duplication;
}

} // namespace aim::perception
