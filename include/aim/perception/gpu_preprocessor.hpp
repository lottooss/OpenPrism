// include/aim/perception/gpu_preprocessor.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/cuda_surface_pool.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/fused_kernel.hpp"
#include "aim/perception/preprocess_types.hpp"

namespace aim::perception {

class GpuPreprocessor;

/// @brief Non-allocating callback interface for releasing a tensor slot back to the preprocessor pool.
class ITensorPoolReleaser {
public:
    virtual ~ITensorPoolReleaser() = default;
    virtual void release_tensor_slot(std::uint32_t slot_index) noexcept = 0;
};

/// @brief Move-only RAII lease for a preprocessed GPU/Device tensor buffer.
class PreprocessedTensorLease {
public:
    constexpr PreprocessedTensorLease() noexcept = default;

    PreprocessedTensorLease(PreprocessedTensorDescriptor descriptor,
                            ITensorPoolReleaser* releaser) noexcept
        : descriptor_(descriptor),
          releaser_(releaser),
          is_valid_(descriptor.is_valid) {}

    ~PreprocessedTensorLease() noexcept {
        reset();
    }

    PreprocessedTensorLease(const PreprocessedTensorLease&) = delete;
    PreprocessedTensorLease& operator=(const PreprocessedTensorLease&) = delete;

    PreprocessedTensorLease(PreprocessedTensorLease&& other) noexcept
        : descriptor_(other.descriptor_),
          releaser_(other.releaser_),
          is_valid_(other.is_valid_) {
        other.is_valid_ = false;
        other.releaser_ = nullptr;
        other.descriptor_.is_valid = false;
        other.descriptor_.gpu_tensor_ptr = nullptr;
        other.descriptor_.cpu_tensor_ptr = nullptr;
    }

    PreprocessedTensorLease& operator=(PreprocessedTensorLease&& other) noexcept {
        if (this != &other) {
            reset();
            descriptor_ = other.descriptor_;
            releaser_ = other.releaser_;
            is_valid_ = other.is_valid_;

            other.is_valid_ = false;
            other.releaser_ = nullptr;
            other.descriptor_.is_valid = false;
            other.descriptor_.gpu_tensor_ptr = nullptr;
            other.descriptor_.cpu_tensor_ptr = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (is_valid_ && releaser_ != nullptr) {
            releaser_->release_tensor_slot(descriptor_.pool_slot_index);
        }
        is_valid_ = false;
        releaser_ = nullptr;
        descriptor_.is_valid = false;
        descriptor_.gpu_tensor_ptr = nullptr;
        descriptor_.cpu_tensor_ptr = nullptr;
    }

    [[nodiscard]] bool is_valid() const noexcept { return is_valid_ && descriptor_.is_valid; }
    [[nodiscard]] explicit operator bool() const noexcept { return is_valid(); }
    [[nodiscard]] const PreprocessedTensorDescriptor& descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] PreprocessedTensorDescriptor& descriptor() noexcept { return descriptor_; }
    [[nodiscard]] SequenceId frame_id() const noexcept { return descriptor_.frame_id; }
    [[nodiscard]] MonotonicNs captured_at_ns() const noexcept { return descriptor_.captured_at_ns; }
    [[nodiscard]] MonotonicNs preprocessed_at_ns() const noexcept { return descriptor_.preprocessed_at_ns; }
    [[nodiscard]] void* gpu_tensor_ptr() const noexcept { return descriptor_.gpu_tensor_ptr; }
    [[nodiscard]] const void* cpu_tensor_ptr() const noexcept { return descriptor_.cpu_tensor_ptr; }
    [[nodiscard]] const AffineTransform2D& affine_transform() const noexcept { return descriptor_.affine_transform; }

private:
    PreprocessedTensorDescriptor descriptor_{};
    ITensorPoolReleaser* releaser_{nullptr};
    bool is_valid_{false};
};

/// @brief Preprocessing health and telemetry metrics.
struct PreprocessHealth {
    bool is_initialized{false};
    bool is_hardware_accelerated{false};
    std::uint64_t total_frames_preprocessed{0};
    std::uint64_t total_frames_dropped{0};
    double last_latency_us{0.0};
    double avg_latency_us{0.0};
    double max_latency_us{0.0};
};

/// @brief Abstract interface for GPU-accelerated frame preprocessing.
class IGpuPreprocessor {
public:
    virtual ~IGpuPreprocessor() = default;

    virtual bool initialize(const PreprocessConfig& config,
                            std::shared_ptr<capture::ICudaInteropBackend> backend) noexcept = 0;
    virtual bool warmup(std::uint32_t iterations) noexcept = 0;
    virtual bool preprocess_cuda_surface(const capture::CudaMappedSurfaceLease& surface_lease,
                                        PreprocessedTensorLease& out_lease) noexcept = 0;
    virtual bool preprocess_raw_memory(const void* src_pixels,
                                      std::uint32_t width,
                                      std::uint32_t height,
                                      std::uint32_t stride_bytes,
                                      FrameFormat format,
                                      SequenceId frame_id,
                                      const CorrelationId& correlation_id,
                                      MonotonicNs captured_at_ns,
                                      PreprocessedTensorLease& out_lease) noexcept = 0;
    virtual void shutdown() noexcept = 0;
    [[nodiscard]] virtual PreprocessHealth health() const noexcept = 0;
};

/// @brief High-performance fused GPU preprocessor managing pre-allocated buffer pools and zero allocations.
class GpuPreprocessor final : public IGpuPreprocessor, public ITensorPoolReleaser {
public:
    static constexpr std::uint32_t kDefaultPoolCapacity = 4;
    static constexpr std::uint32_t kMaxCapacity = 16;
    static constexpr std::uint32_t kInvalidSlot = 0xFFFFFFFFU;

    struct TensorSlot {
        void* device_buffer{nullptr};
        std::size_t size_bytes{0};
        capture::CudaEventHandle ready_event{nullptr};
        std::atomic<bool> is_leased{false};
        std::atomic<bool> has_pending_work{false};
    };

    explicit GpuPreprocessor(std::uint32_t pool_capacity = kDefaultPoolCapacity) noexcept;
    ~GpuPreprocessor() noexcept override;

    GpuPreprocessor(const GpuPreprocessor&) = delete;
    GpuPreprocessor& operator=(const GpuPreprocessor&) = delete;
    GpuPreprocessor(GpuPreprocessor&& other) noexcept;
    GpuPreprocessor& operator=(GpuPreprocessor&& other) noexcept;

    bool initialize(const PreprocessConfig& config,
                    std::shared_ptr<capture::ICudaInteropBackend> backend) noexcept override;
    bool warmup(std::uint32_t iterations) noexcept override;

    bool preprocess_cuda_surface(const capture::CudaMappedSurfaceLease& surface_lease,
                                PreprocessedTensorLease& out_lease) noexcept override;

    bool preprocess_raw_memory(const void* src_pixels,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::uint32_t stride_bytes,
                              FrameFormat format,
                              SequenceId frame_id,
                              const CorrelationId& correlation_id,
                              MonotonicNs captured_at_ns,
                              PreprocessedTensorLease& out_lease) noexcept override;

    void shutdown() noexcept override;
    [[nodiscard]] PreprocessHealth health() const noexcept override;

    // ITensorPoolReleaser
    void release_tensor_slot(std::uint32_t slot_index) noexcept override;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return pool_capacity_; }
    [[nodiscard]] std::uint32_t available_slots() const noexcept;
    [[nodiscard]] bool is_slot_leased(std::uint32_t slot_index) const noexcept;

private:
    std::uint32_t acquire_free_slot() noexcept;

    PreprocessConfig config_{};
    std::shared_ptr<capture::ICudaInteropBackend> backend_{nullptr};
    std::uint32_t pool_capacity_{kDefaultPoolCapacity};
    bool is_initialized_{false};

    capture::CudaStreamHandle stream_{nullptr};
    std::array<TensorSlot, kMaxCapacity> slots_{};

    // Telemetry & metrics
    mutable std::atomic<std::uint64_t> total_frames_{0};
    mutable std::atomic<std::uint64_t> dropped_frames_{0};
    mutable std::atomic<std::uint64_t> total_latency_ns_{0};
    mutable std::atomic<std::uint64_t> max_latency_ns_{0};
    mutable std::atomic<std::uint64_t> last_latency_ns_{0};
};

/// @brief Mock GPU preprocessor for headless CI environments.
class MockGpuPreprocessor final : public IGpuPreprocessor, public ITensorPoolReleaser {
public:
    static constexpr std::uint32_t kDefaultPoolCapacity = 4;
    static constexpr std::uint32_t kMaxCapacity = 16;

    struct MockTensorSlot {
        std::vector<std::uint8_t> buffer{};
        std::atomic<bool> is_leased{false};
    };

    explicit MockGpuPreprocessor(std::uint32_t pool_capacity = kDefaultPoolCapacity) noexcept;
    ~MockGpuPreprocessor() noexcept override;

    bool initialize(const PreprocessConfig& config,
                    std::shared_ptr<capture::ICudaInteropBackend> backend) noexcept override;
    bool warmup(std::uint32_t iterations) noexcept override;

    bool preprocess_cuda_surface(const capture::CudaMappedSurfaceLease& surface_lease,
                                PreprocessedTensorLease& out_lease) noexcept override;

    bool preprocess_raw_memory(const void* src_pixels,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::uint32_t stride_bytes,
                              FrameFormat format,
                              SequenceId frame_id,
                              const CorrelationId& correlation_id,
                              MonotonicNs captured_at_ns,
                              PreprocessedTensorLease& out_lease) noexcept override;

    void shutdown() noexcept override;
    [[nodiscard]] PreprocessHealth health() const noexcept override;

    void release_tensor_slot(std::uint32_t slot_index) noexcept override;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return pool_capacity_; }
    [[nodiscard]] std::uint32_t available_slots() const noexcept;

private:
    std::uint32_t acquire_free_slot() noexcept;

    PreprocessConfig config_{};
    std::shared_ptr<capture::ICudaInteropBackend> backend_{nullptr};
    std::uint32_t pool_capacity_{kDefaultPoolCapacity};
    bool is_initialized_{false};
    std::array<MockTensorSlot, kMaxCapacity> slots_{};

    mutable std::atomic<std::uint64_t> total_frames_{0};
    mutable std::atomic<std::uint64_t> dropped_frames_{0};
    mutable std::atomic<std::uint64_t> total_latency_ns_{0};
    mutable std::atomic<std::uint64_t> max_latency_ns_{0};
    mutable std::atomic<std::uint64_t> last_latency_ns_{0};
};

} // namespace aim::perception
