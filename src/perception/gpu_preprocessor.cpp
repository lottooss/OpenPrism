// src/perception/gpu_preprocessor.cpp
#include "aim/perception/gpu_preprocessor.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace aim::perception {

static MonotonicNs get_current_time_ns() noexcept {
    return static_cast<MonotonicNs>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

// =============================================================================
// GpuPreprocessor Implementation
// =============================================================================

GpuPreprocessor::GpuPreprocessor(std::uint32_t pool_capacity) noexcept
    : pool_capacity_(std::clamp(pool_capacity, 1U, kMaxCapacity)) {}

GpuPreprocessor::~GpuPreprocessor() noexcept {
    shutdown();
}

GpuPreprocessor::GpuPreprocessor(GpuPreprocessor&& other) noexcept
    : config_(other.config_),
      backend_(std::move(other.backend_)),
      pool_capacity_(other.pool_capacity_),
      is_initialized_(other.is_initialized_),
      stream_(other.stream_) {
    other.is_initialized_ = false;
    other.stream_ = nullptr;
    for (std::size_t i = 0; i < kMaxCapacity; ++i) {
        slots_[i].device_buffer = other.slots_[i].device_buffer;
        slots_[i].size_bytes = other.slots_[i].size_bytes;
        slots_[i].ready_event = other.slots_[i].ready_event;
        slots_[i].is_leased.store(other.slots_[i].is_leased.load(std::memory_order_relaxed), std::memory_order_relaxed);
        slots_[i].has_pending_work.store(
            other.slots_[i].has_pending_work.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        other.slots_[i].device_buffer = nullptr;
        other.slots_[i].size_bytes = 0;
        other.slots_[i].ready_event = nullptr;
        other.slots_[i].is_leased.store(false, std::memory_order_relaxed);
        other.slots_[i].has_pending_work.store(false, std::memory_order_relaxed);
    }
}

GpuPreprocessor& GpuPreprocessor::operator=(GpuPreprocessor&& other) noexcept {
    if (this != &other) {
        shutdown();

        config_ = other.config_;
        backend_ = std::move(other.backend_);
        pool_capacity_ = other.pool_capacity_;
        is_initialized_ = other.is_initialized_;
        stream_ = other.stream_;

        other.is_initialized_ = false;
        other.stream_ = nullptr;

        for (std::size_t i = 0; i < kMaxCapacity; ++i) {
            slots_[i].device_buffer = other.slots_[i].device_buffer;
            slots_[i].size_bytes = other.slots_[i].size_bytes;
            slots_[i].ready_event = other.slots_[i].ready_event;
            slots_[i].is_leased.store(other.slots_[i].is_leased.load(std::memory_order_relaxed), std::memory_order_relaxed);
            slots_[i].has_pending_work.store(
                other.slots_[i].has_pending_work.load(std::memory_order_relaxed),
                std::memory_order_relaxed);
            other.slots_[i].device_buffer = nullptr;
            other.slots_[i].size_bytes = 0;
            other.slots_[i].ready_event = nullptr;
            other.slots_[i].is_leased.store(false, std::memory_order_relaxed);
            other.slots_[i].has_pending_work.store(false, std::memory_order_relaxed);
        }
    }
    return *this;
}

bool GpuPreprocessor::initialize(const PreprocessConfig& config,
                                 std::shared_ptr<capture::ICudaInteropBackend> backend) noexcept {
    shutdown();

    config_ = config;
    backend_ = std::move(backend);
    const std::size_t tensor_bytes = config_.tensor_size_bytes();

    const bool config_supported = tensor_bytes > 0 &&
        config_.source_format == FrameFormat::b8g8r8a8_unorm &&
        config_.precision == TensorPrecision::fp16 &&
        config_.layout == TensorLayout::nchw &&
        config_.batch_size == 1 && config_.channels == 3 &&
        config_.target_width_px > 0 && config_.target_height_px > 0 &&
        config_.scaled_width_px > 0 && config_.scaled_height_px > 0;
    if (!config_supported || backend_ == nullptr || !backend_->is_cuda_available() ||
        !FusedPreprocessKernel::is_cuda_compiled()) {
        backend_.reset();
        return false;
    }

    if (backend_->create_stream(stream_, capture::CudaStreamFlags::non_blocking) !=
        capture::CudaResult::success) {
        stream_ = nullptr;
        backend_.reset();
        return false;
    }

    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        slots_[i].is_leased.store(false, std::memory_order_release);
        slots_[i].has_pending_work.store(false, std::memory_order_release);
        slots_[i].size_bytes = tensor_bytes;
        if (!FusedPreprocessKernel::allocate_device_tensor(&slots_[i].device_buffer, tensor_bytes) ||
            backend_->create_event(slots_[i].ready_event,
                                   capture::CudaEventFlags::disable_timing) !=
                capture::CudaResult::success) {
            shutdown();
            return false;
        }
    }

    total_frames_.store(0, std::memory_order_relaxed);
    dropped_frames_.store(0, std::memory_order_relaxed);
    total_latency_ns_.store(0, std::memory_order_relaxed);
    max_latency_ns_.store(0, std::memory_order_relaxed);
    last_latency_ns_.store(0, std::memory_order_relaxed);

    is_initialized_ = true;
    return true;
}

bool GpuPreprocessor::warmup(std::uint32_t iterations) noexcept {
    if (!is_initialized_ || stream_ == nullptr) {
        return false;
    }

    for (std::uint32_t i = 0; i < iterations; ++i) {
        TensorSlot& slot = slots_[i % pool_capacity_];
        if (!FusedPreprocessKernel::warmup_device_tensor(
                slot.device_buffer, slot.size_bytes, stream_)) {
            return false;
        }
    }

    if (backend_->synchronize_stream(stream_) != capture::CudaResult::success) {
        return false;
    }

    total_frames_.store(0, std::memory_order_relaxed);
    dropped_frames_.store(0, std::memory_order_relaxed);
    total_latency_ns_.store(0, std::memory_order_relaxed);
    max_latency_ns_.store(0, std::memory_order_relaxed);
    last_latency_ns_.store(0, std::memory_order_relaxed);

    return true;
}

std::uint32_t GpuPreprocessor::acquire_free_slot() noexcept {
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        if (slots_[i].has_pending_work.load(std::memory_order_acquire)) {
            if (backend_ == nullptr || slots_[i].ready_event == nullptr ||
                backend_->query_event(slots_[i].ready_event) != capture::CudaResult::success) {
                continue;
            }
            slots_[i].has_pending_work.store(false, std::memory_order_release);
        }
        bool expected = false;
        if (slots_[i].is_leased.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return i;
        }
    }
    return kInvalidSlot;
}

bool GpuPreprocessor::preprocess_cuda_surface(const capture::CudaMappedSurfaceLease& surface_lease,
                                              PreprocessedTensorLease& out_lease) noexcept {
    if (!is_initialized_ || !surface_lease.is_valid() || surface_lease.surface_object() == 0 ||
        surface_lease.stream() == nullptr) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const auto& frame_lease = surface_lease.frame_lease();
    if (frame_lease.format() != FrameFormat::b8g8r8a8_unorm) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const std::uint32_t slot_index = acquire_free_slot();
    if (slot_index == kInvalidSlot) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    TensorSlot& slot = slots_[slot_index];
    const MonotonicNs start_ns = get_current_time_ns();
    FusedKernelParams params{};
    params.src_width = frame_lease.width_px();
    params.src_height = frame_lease.height_px();
    params.src_stride_bytes = frame_lease.width_px() * 4U;
    params.src_format = frame_lease.format();
    params.dst_tensor = slot.device_buffer;
    params.dst_width = config_.target_width_px;
    params.dst_height = config_.target_height_px;
    params.scaled_width = config_.scaled_width_px;
    params.scaled_height = config_.scaled_height_px;
    params.pad_x = config_.pad_x;
    params.pad_y = config_.pad_y;
    params.pad_value_normalized = config_.pad_value_normalized;
    params.pad_half_bits = HalfFloat(config_.pad_value_normalized).bits;
    params.scale_x = static_cast<float>(frame_lease.width_px()) /
        static_cast<float>(config_.scaled_width_px);
    params.scale_y = static_cast<float>(frame_lease.height_px()) /
        static_cast<float>(config_.scaled_height_px);

    const capture::CudaStreamHandle processing_stream = surface_lease.stream();
    const bool launched = FusedPreprocessKernel::launch_cuda_surface(
        surface_lease.surface_object(), params, processing_stream);
    if (!launched || backend_->record_event(slot.ready_event, processing_stream) !=
                         capture::CudaResult::success) {
        slot.is_leased.store(false, std::memory_order_release);
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    slot.has_pending_work.store(true, std::memory_order_release);

    const MonotonicNs end_ns = get_current_time_ns();
    const auto submission_latency_ns = static_cast<std::uint64_t>(
        std::max(std::int64_t{0}, end_ns - start_ns));
    last_latency_ns_.store(submission_latency_ns, std::memory_order_relaxed);
    total_latency_ns_.fetch_add(submission_latency_ns, std::memory_order_relaxed);
    total_frames_.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t current_max = max_latency_ns_.load(std::memory_order_relaxed);
    while (submission_latency_ns > current_max &&
           !max_latency_ns_.compare_exchange_weak(
               current_max, submission_latency_ns, std::memory_order_relaxed)) {
    }

    PreprocessedTensorDescriptor descriptor{};
    descriptor.frame_id = frame_lease.frame_id();
    descriptor.correlation_id = CorrelationId{
        frame_lease.frame_id(), frame_lease.captured_at_ns(), 1, 0};
    descriptor.captured_at_ns = frame_lease.captured_at_ns();
    descriptor.preprocessed_at_ns = end_ns;
    descriptor.width_px = config_.target_width_px;
    descriptor.height_px = config_.target_height_px;
    descriptor.channels = config_.channels;
    descriptor.size_bytes = slot.size_bytes;
    descriptor.precision = config_.precision;
    descriptor.layout = config_.layout;
    descriptor.affine_transform = AffineTransform2D::create_letterbox(
        frame_lease.width_px(), frame_lease.height_px(),
        config_.target_width_px, config_.target_height_px);
    descriptor.gpu_tensor_ptr = slot.device_buffer;
    descriptor.cpu_tensor_ptr = nullptr;
    descriptor.native_ready_event = slot.ready_event;
    descriptor.pool_slot_index = slot_index;
    descriptor.is_valid = true;
    out_lease = PreprocessedTensorLease(descriptor, this);
    return true;
}

bool GpuPreprocessor::preprocess_raw_memory(const void* src_pixels,
                                            std::uint32_t width,
                                            std::uint32_t height,
                                            std::uint32_t stride_bytes,
                                            FrameFormat format,
                                            SequenceId frame_id,
                                            const CorrelationId& correlation_id,
                                            MonotonicNs captured_at_ns,
                                            PreprocessedTensorLease& out_lease) noexcept {
    static_cast<void>(src_pixels);
    static_cast<void>(width);
    static_cast<void>(height);
    static_cast<void>(stride_bytes);
    static_cast<void>(format);
    static_cast<void>(frame_id);
    static_cast<void>(correlation_id);
    static_cast<void>(captured_at_ns);
    out_lease.reset();
    dropped_frames_.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void GpuPreprocessor::release_tensor_slot(std::uint32_t slot_index) noexcept {
    if (slot_index < pool_capacity_) {
        slots_[slot_index].is_leased.store(false, std::memory_order_release);
    }
}

std::uint32_t GpuPreprocessor::available_slots() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        if (!slots_[i].is_leased.load(std::memory_order_acquire) &&
            !slots_[i].has_pending_work.load(std::memory_order_acquire)) {
            ++count;
        }
    }
    return count;
}

bool GpuPreprocessor::is_slot_leased(std::uint32_t slot_index) const noexcept {
    if (slot_index >= pool_capacity_) {
        return false;
    }
    return slots_[slot_index].is_leased.load(std::memory_order_acquire);
}

PreprocessHealth GpuPreprocessor::health() const noexcept {
    PreprocessHealth h{};
    h.is_initialized = is_initialized_;
    h.is_hardware_accelerated = is_initialized_ && FusedPreprocessKernel::is_cuda_compiled();
    h.total_frames_preprocessed = total_frames_.load(std::memory_order_relaxed);
    h.total_frames_dropped = dropped_frames_.load(std::memory_order_relaxed);
    h.last_latency_us = static_cast<double>(last_latency_ns_.load(std::memory_order_relaxed)) / 1000.0;
    if (h.total_frames_preprocessed > 0) {
        h.avg_latency_us = static_cast<double>(total_latency_ns_.load(std::memory_order_relaxed)) /
                           (static_cast<double>(h.total_frames_preprocessed) * 1000.0);
    }
    h.max_latency_us = static_cast<double>(max_latency_ns_.load(std::memory_order_relaxed)) / 1000.0;
    return h;
}

void GpuPreprocessor::shutdown() noexcept {
    is_initialized_ = false;

    if (backend_ && backend_->is_cuda_available()) {
        if (stream_) {
            for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
                if (slots_[i].has_pending_work.load(std::memory_order_acquire) &&
                    slots_[i].ready_event != nullptr) {
                    static_cast<void>(backend_->stream_wait_event(stream_, slots_[i].ready_event));
                }
            }
            static_cast<void>(backend_->synchronize_stream(stream_));
        }
        if (stream_) {
            backend_->destroy_stream(stream_);
            stream_ = nullptr;
        }
        for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
            if (slots_[i].ready_event) {
                backend_->destroy_event(slots_[i].ready_event);
                slots_[i].ready_event = nullptr;
            }
        }
    }

    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        FusedPreprocessKernel::free_device_tensor(slots_[i].device_buffer);
        slots_[i].device_buffer = nullptr;
        slots_[i].size_bytes = 0;
        slots_[i].is_leased.store(false, std::memory_order_relaxed);
        slots_[i].has_pending_work.store(false, std::memory_order_relaxed);
    }
    backend_.reset();
}

// =============================================================================
// MockGpuPreprocessor Implementation
// =============================================================================

MockGpuPreprocessor::MockGpuPreprocessor(std::uint32_t pool_capacity) noexcept
    : pool_capacity_(std::clamp(pool_capacity, 1U, kMaxCapacity)) {}

MockGpuPreprocessor::~MockGpuPreprocessor() noexcept {
    shutdown();
}

bool MockGpuPreprocessor::initialize(const PreprocessConfig& config,
                                     std::shared_ptr<capture::ICudaInteropBackend> backend) noexcept {
    shutdown();
    config_ = config;
    backend_ = std::move(backend);

    const std::size_t tensor_bytes = config_.tensor_size_bytes();
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        slots_[i].buffer.resize(tensor_bytes, 0);
        slots_[i].is_leased.store(false, std::memory_order_release);
    }

    total_frames_.store(0, std::memory_order_relaxed);
    dropped_frames_.store(0, std::memory_order_relaxed);
    total_latency_ns_.store(0, std::memory_order_relaxed);
    max_latency_ns_.store(0, std::memory_order_relaxed);
    last_latency_ns_.store(0, std::memory_order_relaxed);

    is_initialized_ = true;
    return true;
}

bool MockGpuPreprocessor::warmup(std::uint32_t iterations) noexcept {
    if (!is_initialized_) return false;
    std::vector<std::uint8_t> dummy(static_cast<std::size_t>(config_.source_width_px) * config_.source_height_px * 4, 100);
    CorrelationId dummy_corr{};
    for (std::uint32_t i = 0; i < iterations; ++i) {
        PreprocessedTensorLease lease{};
        if (!preprocess_raw_memory(dummy.data(), config_.source_width_px, config_.source_height_px,
                                   config_.source_width_px * 4, config_.source_format, i, dummy_corr,
                                   get_current_time_ns(), lease)) {
            return false;
        }
    }
    total_frames_.store(0, std::memory_order_relaxed);
    dropped_frames_.store(0, std::memory_order_relaxed);
    total_latency_ns_.store(0, std::memory_order_relaxed);
    max_latency_ns_.store(0, std::memory_order_relaxed);
    last_latency_ns_.store(0, std::memory_order_relaxed);
    return true;
}

std::uint32_t MockGpuPreprocessor::acquire_free_slot() noexcept {
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        bool expected = false;
        if (slots_[i].is_leased.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return i;
        }
    }
    return GpuPreprocessor::kInvalidSlot;
}

bool MockGpuPreprocessor::preprocess_cuda_surface(const capture::CudaMappedSurfaceLease& surface_lease,
                                                  PreprocessedTensorLease& out_lease) noexcept {
    if (!is_initialized_ || !surface_lease.is_valid()) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto& f = surface_lease.frame_lease();
    return preprocess_raw_memory(f.native_texture_ptr(), f.width_px(), f.height_px(), f.width_px() * 4,
                                f.format(), f.frame_id(), CorrelationId{f.frame_id(), f.captured_at_ns(), 1, 0},
                                f.captured_at_ns(), out_lease);
}

bool MockGpuPreprocessor::preprocess_raw_memory(const void* src_pixels,
                                                std::uint32_t width,
                                                std::uint32_t height,
                                                std::uint32_t stride_bytes,
                                                FrameFormat format,
                                                SequenceId frame_id,
                                                const CorrelationId& correlation_id,
                                                MonotonicNs captured_at_ns,
                                                PreprocessedTensorLease& out_lease) noexcept {
    if (!is_initialized_ || !src_pixels || width == 0 || height == 0) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    std::uint32_t slot = acquire_free_slot();
    if (slot == GpuPreprocessor::kInvalidSlot) {
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const MonotonicNs start_ns = get_current_time_ns();

    FusedKernelParams params{};
    params.src_pixels = src_pixels;
    params.src_width = width;
    params.src_height = height;
    params.src_stride_bytes = stride_bytes;
    params.src_format = format;

    params.dst_tensor = slots_[slot].buffer.data();
    params.dst_width = config_.target_width_px;
    params.dst_height = config_.target_height_px;
    params.scaled_width = config_.scaled_width_px;
    params.scaled_height = config_.scaled_height_px;
    params.pad_x = config_.pad_x;
    params.pad_y = config_.pad_y;
    params.pad_value_normalized = config_.pad_value_normalized;
    params.pad_half_bits = HalfFloat(config_.pad_value_normalized).bits;
    params.scale_x = static_cast<float>(width) / static_cast<float>(config_.scaled_width_px);
    params.scale_y = static_cast<float>(height) / static_cast<float>(config_.scaled_height_px);

    bool success = FusedPreprocessKernel::execute_software(params);
    if (!success) {
        slots_[slot].is_leased.store(false, std::memory_order_release);
        out_lease.reset();
        dropped_frames_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const MonotonicNs end_ns = get_current_time_ns();
    const auto latency_ns = static_cast<std::uint64_t>(std::max(std::int64_t{0}, end_ns - start_ns));

    last_latency_ns_.store(latency_ns, std::memory_order_relaxed);
    total_latency_ns_.fetch_add(latency_ns, std::memory_order_relaxed);
    total_frames_.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t current_max = max_latency_ns_.load(std::memory_order_relaxed);
    while (latency_ns > current_max &&
           !max_latency_ns_.compare_exchange_weak(
               current_max, latency_ns, std::memory_order_relaxed)) {
    }

    PreprocessedTensorDescriptor descriptor{};
    descriptor.frame_id = frame_id;
    descriptor.correlation_id = correlation_id;
    descriptor.captured_at_ns = captured_at_ns;
    descriptor.preprocessed_at_ns = end_ns;
    descriptor.width_px = config_.target_width_px;
    descriptor.height_px = config_.target_height_px;
    descriptor.channels = config_.channels;
    descriptor.size_bytes = config_.tensor_size_bytes();
    descriptor.precision = config_.precision;
    descriptor.layout = config_.layout;
    descriptor.affine_transform = AffineTransform2D::create_letterbox(width, height, config_.target_width_px, config_.target_height_px);
    descriptor.gpu_tensor_ptr = slots_[slot].buffer.data();
    descriptor.cpu_tensor_ptr = slots_[slot].buffer.data();
    descriptor.pool_slot_index = slot;
    descriptor.is_valid = true;

    out_lease = PreprocessedTensorLease(descriptor, this);
    return true;
}

void MockGpuPreprocessor::release_tensor_slot(std::uint32_t slot_index) noexcept {
    if (slot_index < pool_capacity_) {
        slots_[slot_index].is_leased.store(false, std::memory_order_release);
    }
}

std::uint32_t MockGpuPreprocessor::available_slots() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        if (!slots_[i].is_leased.load(std::memory_order_acquire)) {
            ++count;
        }
    }
    return count;
}

PreprocessHealth MockGpuPreprocessor::health() const noexcept {
    PreprocessHealth h{};
    h.is_initialized = is_initialized_;
    h.total_frames_preprocessed = total_frames_.load(std::memory_order_relaxed);
    h.total_frames_dropped = dropped_frames_.load(std::memory_order_relaxed);
    h.last_latency_us = static_cast<double>(last_latency_ns_.load(std::memory_order_relaxed)) / 1000.0;
    if (h.total_frames_preprocessed > 0) {
        h.avg_latency_us = static_cast<double>(total_latency_ns_.load(std::memory_order_relaxed)) /
                           (static_cast<double>(h.total_frames_preprocessed) * 1000.0);
    }
    h.max_latency_us = static_cast<double>(max_latency_ns_.load(std::memory_order_relaxed)) / 1000.0;
    return h;
}

void MockGpuPreprocessor::shutdown() noexcept {
    is_initialized_ = false;
    for (std::uint32_t i = 0; i < pool_capacity_; ++i) {
        slots_[i].buffer.clear();
        slots_[i].buffer.shrink_to_fit();
        slots_[i].is_leased.store(false, std::memory_order_relaxed);
    }
}

} // namespace aim::perception
