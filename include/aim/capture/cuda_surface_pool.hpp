// include/aim/capture/cuda_surface_pool.hpp
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/gpu_surface_pool.hpp"
#include "aim/core/frame_source.hpp"

namespace aim::capture {

class CudaSurfacePool;

/// @brief Move-only RAII lease for a CUDA-mapped Direct3D 11 staging texture surface.
class CudaMappedSurfaceLease {
public:
    CudaMappedSurfaceLease() noexcept = default;
    ~CudaMappedSurfaceLease() noexcept { reset(); }

    CudaMappedSurfaceLease(const CudaMappedSurfaceLease&) = delete;
    CudaMappedSurfaceLease& operator=(const CudaMappedSurfaceLease&) = delete;

    CudaMappedSurfaceLease(CudaMappedSurfaceLease&& other) noexcept
        : pool_(other.pool_),
          slot_index_(other.slot_index_),
          frame_lease_(std::move(other.frame_lease_)),
          mapped_array_(other.mapped_array_),
          surface_object_(other.surface_object_),
          stream_(other.stream_) {
        other.pool_ = nullptr;
        other.slot_index_ = GpuSurfacePool::kInvalidSlot;
        other.mapped_array_ = nullptr;
        other.surface_object_ = 0;
        other.stream_ = nullptr;
    }

    CudaMappedSurfaceLease& operator=(CudaMappedSurfaceLease&& other) noexcept {
        if (this != &other) {
            reset();
            pool_ = other.pool_;
            slot_index_ = other.slot_index_;
            frame_lease_ = std::move(other.frame_lease_);
            mapped_array_ = other.mapped_array_;
            surface_object_ = other.surface_object_;
            stream_ = other.stream_;

            other.pool_ = nullptr;
            other.slot_index_ = GpuSurfacePool::kInvalidSlot;
            other.mapped_array_ = nullptr;
            other.surface_object_ = 0;
            other.stream_ = nullptr;
        }
        return *this;
    }

    void reset() noexcept;

    [[nodiscard]] bool is_valid() const noexcept {
        return pool_ != nullptr && slot_index_ != GpuSurfacePool::kInvalidSlot && frame_lease_.is_valid();
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return is_valid();
    }

    [[nodiscard]] std::uint32_t slot_index() const noexcept { return slot_index_; }
    [[nodiscard]] SequenceId frame_id() const noexcept { return frame_lease_.frame_id(); }
    [[nodiscard]] MonotonicNs captured_at_ns() const noexcept { return frame_lease_.captured_at_ns(); }
    [[nodiscard]] CudaArrayHandle mapped_array() const noexcept { return mapped_array_; }
    [[nodiscard]] CudaSurfaceObjectHandle surface_object() const noexcept { return surface_object_; }
    [[nodiscard]] CudaStreamHandle stream() const noexcept { return stream_; }
    [[nodiscard]] const FrameLease& frame_lease() const noexcept { return frame_lease_; }

private:
    friend class CudaSurfacePool;

    CudaMappedSurfaceLease(CudaSurfacePool* pool,
                           std::uint32_t slot_index,
                           FrameLease frame_lease,
                           CudaArrayHandle mapped_array,
                           CudaSurfaceObjectHandle surface_object,
                           CudaStreamHandle stream) noexcept
        : pool_(pool),
          slot_index_(slot_index),
          frame_lease_(std::move(frame_lease)),
          mapped_array_(mapped_array),
          surface_object_(surface_object),
          stream_(stream) {}

    CudaSurfacePool* pool_{nullptr};
    std::uint32_t slot_index_{GpuSurfacePool::kInvalidSlot};
    FrameLease frame_lease_{};
    CudaArrayHandle mapped_array_{nullptr};
    CudaSurfaceObjectHandle surface_object_{0};
    CudaStreamHandle stream_{nullptr};
};

/// @brief Pre-registered Direct3D 11 - CUDA buffer pool managing zero-copy GPU mapping and streams.
class CudaSurfacePool final : public IGpuSurfacePoolObserver {
public:
    static constexpr std::uint32_t kMaxCapacity = GpuSurfacePool::kMaxCapacity;

    struct SlotResource {
        CudaGraphicsResourceHandle graphics_resource{nullptr};
        CudaArrayHandle mapped_array{nullptr};
        CudaSurfaceObjectHandle surface_object{0};
        CudaEventHandle ready_event{nullptr};
        CudaEventHandle complete_event{nullptr};
        FrameLease pending_frame_lease{};
        bool is_registered{false};
        bool is_mapped{false};
        bool is_release_pending{false};
    };

    explicit CudaSurfacePool(std::shared_ptr<ICudaInteropBackend> backend) noexcept;
    ~CudaSurfacePool() noexcept;

    CudaSurfacePool(const CudaSurfacePool&) = delete;
    CudaSurfacePool& operator=(const CudaSurfacePool&) = delete;
    CudaSurfacePool(CudaSurfacePool&& other) noexcept;
    CudaSurfacePool& operator=(CudaSurfacePool&& other) noexcept;

    /// @brief Pre-registers all staging textures in the GpuSurfacePool with CUDA.
    /// @param gpu_pool The underlying Direct3D 11 surface pool.
    /// @param dxgi_adapter_luid Adapter LUID driving capture for CUDA device affinity validation.
    /// @return true if all slots registered and CUDA stream/events created; false on mismatch or failure.
    bool initialize(GpuSurfacePool& gpu_pool, std::uint64_t dxgi_adapter_luid) noexcept;

    /// @brief Maps the specified slot for CUDA preprocessing, returning an RAII lease.
    /// @param frame_lease Active FrameLease for the slot.
    /// @param out_mapped_lease Output move-only RAII lease containing mapped array and surface.
    /// @return true on success, false if map fails (fail-closed).
    bool map_surface(FrameLease frame_lease, CudaMappedSurfaceLease& out_mapped_lease) noexcept;

    /// @brief Enqueues unmap and retains the capture lease until CUDA reports completion.
    void unmap_surface(std::uint32_t slot_index, FrameLease frame_lease) noexcept;

    /// @brief Non-blocking completion poll that returns finished capture leases to their source pool.
    void reap_completed_releases() noexcept;

    /// @brief Unregisters all CUDA resources and destroys streams and events.
    void release_all() noexcept;

    // Accessors
    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] int cuda_device_id() const noexcept { return cuda_device_id_; }
    [[nodiscard]] CudaStreamHandle processing_stream() const noexcept { return processing_stream_; }
    [[nodiscard]] ICudaInteropBackend* backend() const noexcept { return backend_.get(); }
    [[nodiscard]] bool is_slot_mapped(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] bool is_slot_registered(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] bool is_slot_release_pending(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] CudaGraphicsResourceHandle get_graphics_resource(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] CudaEventHandle get_ready_event(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] CudaEventHandle get_complete_event(std::uint32_t slot_index) const noexcept;

    void before_surface_pool_reset(GpuSurfacePool& pool) noexcept override;

private:
    std::shared_ptr<ICudaInteropBackend> backend_{nullptr};
    std::uint32_t capacity_{0};
    int cuda_device_id_{-1};
    bool is_initialized_{false};
    GpuSurfacePool* source_pool_{nullptr};

    CudaStreamHandle processing_stream_{nullptr};
    std::array<SlotResource, kMaxCapacity> slots_{};
};

} // namespace aim::capture
