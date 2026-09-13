// include/aim/capture/gpu_surface_pool.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include "aim/capture/dxgi_backend.hpp"
#include "aim/core/frame_source.hpp"

namespace aim::capture {

class GpuSurfacePool;

class IGpuSurfacePoolObserver {
public:
    virtual ~IGpuSurfacePoolObserver() = default;
    virtual void before_surface_pool_reset(GpuSurfacePool& pool) noexcept = 0;
};

/// @brief Preallocated pool of GPU staging surfaces with RAII lease tracking and zero hot-path allocations.
class GpuSurfacePool final : public ISurfacePoolReleaser {
public:
    static constexpr std::uint32_t kInvalidSlot = 0xFFFFFFFF;
    static constexpr std::uint32_t kMaxCapacity = 16;
    static constexpr std::uint32_t kDefaultCapacity = 4;

    struct alignas(64) Slot {
        void* texture{nullptr};
        std::uint64_t shared_handle{0};
        alignas(64) std::atomic<std::uint32_t> ref_count{0};
        std::atomic<std::uint64_t> active_frame_id{0};
        std::atomic<MonotonicNs> captured_at_ns{0};
        std::uint32_t slot_index{0};

        Slot() noexcept = default;
        ~Slot() noexcept = default;

        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;

        Slot(Slot&& other) noexcept
            : texture(other.texture),
              shared_handle(other.shared_handle),
              ref_count(other.ref_count.load(std::memory_order_relaxed)),
              active_frame_id(other.active_frame_id.load(std::memory_order_relaxed)),
              captured_at_ns(other.captured_at_ns.load(std::memory_order_relaxed)),
              slot_index(other.slot_index) {
            other.texture = nullptr;
            other.shared_handle = 0;
            other.ref_count.store(0, std::memory_order_relaxed);
            other.active_frame_id.store(0, std::memory_order_relaxed);
            other.captured_at_ns.store(0, std::memory_order_relaxed);
            other.slot_index = 0;
        }

        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                texture = other.texture;
                shared_handle = other.shared_handle;
                ref_count.store(other.ref_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
                active_frame_id.store(other.active_frame_id.load(std::memory_order_relaxed), std::memory_order_relaxed);
                captured_at_ns.store(other.captured_at_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
                slot_index = other.slot_index;

                other.texture = nullptr;
                other.shared_handle = 0;
                other.ref_count.store(0, std::memory_order_relaxed);
                other.active_frame_id.store(0, std::memory_order_relaxed);
                other.captured_at_ns.store(0, std::memory_order_relaxed);
                other.slot_index = 0;
            }
            return *this;
        }
    };

    GpuSurfacePool() noexcept = default;
    ~GpuSurfacePool() noexcept override;

    GpuSurfacePool(const GpuSurfacePool&) = delete;
    GpuSurfacePool& operator=(const GpuSurfacePool&) = delete;
    GpuSurfacePool(GpuSurfacePool&&) noexcept;
    GpuSurfacePool& operator=(GpuSurfacePool&&) noexcept;

    /// @brief Preallocates a fixed pool of N staging textures on the GPU.
    bool initialize(IGpuTextureAllocator* backend,
                    std::uint32_t capacity,
                    std::uint32_t width,
                    std::uint32_t height,
                    FrameFormat format = FrameFormat::b8g8r8a8_unorm) noexcept;

    /// @brief Acquires the next available unleased slot for writing.
    /// @return Slot index [0, capacity-1] or kInvalidSlot if all slots are currently leased.
    [[nodiscard]] std::uint32_t acquire_free_slot() noexcept;

    /// @brief Creates an RAII FrameLease claiming exclusive read ownership of the specified slot.
    [[nodiscard]] FrameLease create_lease(std::uint32_t slot_index,
                                          SequenceId frame_id,
                                          MonotonicNs captured_at_ns) noexcept;

    /// @brief ISurfacePoolReleaser callback invoked upon FrameLease destruction or reset.
    void release_surface(std::uint32_t pool_slot_index) noexcept override;

    /// @brief Releases all allocated textures and resets the pool.
    void release_all() noexcept;

    /// @brief Reallocates textures if output resolution changes.
    bool resize(IGpuTextureAllocator* backend, std::uint32_t new_width, std::uint32_t new_height) noexcept;

    // Accessors
    [[nodiscard]] void* get_texture(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] std::uint64_t get_shared_handle(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] std::uint32_t ref_count(std::uint32_t slot_index) const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::uint32_t width() const noexcept { return width_; }
    [[nodiscard]] std::uint32_t height() const noexcept { return height_; }
    [[nodiscard]] FrameFormat format() const noexcept { return format_; }
    [[nodiscard]] std::uint32_t active_lease_count() const noexcept;
    [[nodiscard]] bool is_initialized() const noexcept { return is_initialized_; }
    [[nodiscard]] IGpuTextureAllocator* backend() const noexcept { return backend_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    void bind_observer(IGpuSurfacePoolObserver* observer) noexcept { observer_ = observer; }
    void unbind_observer(const IGpuSurfacePoolObserver* observer) noexcept {
        if (observer_ == observer) {
            observer_ = nullptr;
        }
    }

private:
    IGpuTextureAllocator* backend_{nullptr};
    std::uint32_t capacity_{0};
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    FrameFormat format_{FrameFormat::unknown};
    bool is_initialized_{false};
    std::uint64_t generation_{0};
    IGpuSurfacePoolObserver* observer_{nullptr};

    alignas(64) std::atomic<std::uint32_t> write_index_{0};
    std::array<Slot, kMaxCapacity> slots_{};
};

} // namespace aim::capture
