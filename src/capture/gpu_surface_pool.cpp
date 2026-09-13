// src/capture/gpu_surface_pool.cpp
#include "aim/capture/gpu_surface_pool.hpp"
#include <algorithm>

namespace aim::capture {

GpuSurfacePool::~GpuSurfacePool() noexcept {
    release_all();
}

GpuSurfacePool::GpuSurfacePool(GpuSurfacePool&& other) noexcept
    : backend_(other.backend_),
      capacity_(other.capacity_),
      width_(other.width_),
      height_(other.height_),
      format_(other.format_),
      is_initialized_(other.is_initialized_),
      generation_(other.generation_),
      write_index_(other.write_index_.load(std::memory_order_relaxed)),
      slots_(std::move(other.slots_)) {
    other.backend_ = nullptr;
    other.capacity_ = 0;
    other.width_ = 0;
    other.height_ = 0;
    other.format_ = FrameFormat::unknown;
    other.is_initialized_ = false;
    other.generation_ = 0;
}

GpuSurfacePool& GpuSurfacePool::operator=(GpuSurfacePool&& other) noexcept {
    if (this != &other) {
        release_all();

        backend_ = other.backend_;
        capacity_ = other.capacity_;
        width_ = other.width_;
        height_ = other.height_;
        format_ = other.format_;
        is_initialized_ = other.is_initialized_;
        generation_ = other.generation_;
        write_index_.store(other.write_index_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        slots_ = std::move(other.slots_);

        other.backend_ = nullptr;
        other.capacity_ = 0;
        other.width_ = 0;
        other.height_ = 0;
        other.format_ = FrameFormat::unknown;
        other.is_initialized_ = false;
        other.generation_ = 0;
    }
    return *this;
}

bool GpuSurfacePool::initialize(IGpuTextureAllocator* backend,
                               std::uint32_t capacity,
                               std::uint32_t width,
                               std::uint32_t height,
                               FrameFormat format) noexcept {
    if (is_initialized_) {
        release_all();
    }

    if (!backend || capacity < 2 || capacity > kMaxCapacity || width == 0 || height == 0) {
        return false;
    }

    backend_ = backend;
    capacity_ = capacity;
    width_ = width;
    height_ = height;
    format_ = format;

    for (std::uint32_t i = 0; i < capacity_; ++i) {
        slots_[i].slot_index = i;
        slots_[i].ref_count.store(0, std::memory_order_relaxed);
        slots_[i].active_frame_id.store(0, std::memory_order_relaxed);
        slots_[i].captured_at_ns.store(0, std::memory_order_relaxed);

        void* tex = nullptr;
        std::uint64_t handle = 0;
        if (!backend_->allocate_staging_texture(width, height, format, &tex, &handle)) {
            release_all();
            return false;
        }
        slots_[i].texture = tex;
        slots_[i].shared_handle = handle;
    }

    write_index_.store(0, std::memory_order_relaxed);
    is_initialized_ = true;
    ++generation_;
    return true;
}

std::uint32_t GpuSurfacePool::acquire_free_slot() noexcept {
    if (!is_initialized_ || capacity_ == 0) {
        return kInvalidSlot;
    }

    const std::uint32_t start_idx = write_index_.fetch_add(1, std::memory_order_relaxed);
    for (std::uint32_t attempt = 0; attempt < capacity_; ++attempt) {
        const std::uint32_t slot = (start_idx + attempt) % capacity_;
        if (slots_[slot].ref_count.load(std::memory_order_acquire) == 0) {
            return slot;
        }
    }

    return kInvalidSlot;
}

FrameLease GpuSurfacePool::create_lease(std::uint32_t slot_index,
                                       SequenceId frame_id,
                                       MonotonicNs captured_at_ns) noexcept {
    if (slot_index >= capacity_ || !is_initialized_) {
        return {};
    }

    slots_[slot_index].ref_count.fetch_add(1, std::memory_order_acq_rel);
    slots_[slot_index].active_frame_id.store(frame_id, std::memory_order_release);
    slots_[slot_index].captured_at_ns.store(captured_at_ns, std::memory_order_release);

    return FrameLease(
        frame_id,
        captured_at_ns,
        width_,
        height_,
        format_,
        slots_[slot_index].texture,
        slot_index,
        this
    );
}

void GpuSurfacePool::release_surface(std::uint32_t pool_slot_index) noexcept {
    if (pool_slot_index < capacity_) {
        std::uint32_t current = slots_[pool_slot_index].ref_count.load(std::memory_order_relaxed);
        while (current > 0) {
            if (slots_[pool_slot_index].ref_count.compare_exchange_weak(
                    current, current - 1, std::memory_order_release, std::memory_order_relaxed)) {
                break;
            }
        }
    }
}

void GpuSurfacePool::release_all() noexcept {
    const bool had_resources = is_initialized_ || capacity_ != 0;
    if (had_resources && observer_ != nullptr) {
        IGpuSurfacePoolObserver* observer = observer_;
        observer_ = nullptr;
        observer->before_surface_pool_reset(*this);
    }
    if (backend_) {
        for (std::uint32_t i = 0; i < capacity_; ++i) {
            if (slots_[i].texture) {
                backend_->free_staging_texture(slots_[i].texture, slots_[i].shared_handle);
                slots_[i].texture = nullptr;
                slots_[i].shared_handle = 0;
            }
            slots_[i].ref_count.store(0, std::memory_order_relaxed);
        }
    }
    capacity_ = 0;
    width_ = 0;
    height_ = 0;
    is_initialized_ = false;
    if (had_resources) {
        ++generation_;
    }
}

bool GpuSurfacePool::resize(IGpuTextureAllocator* backend, std::uint32_t new_width, std::uint32_t new_height) noexcept {
    if (width_ == new_width && height_ == new_height && is_initialized_) {
        return true;
    }
    const std::uint32_t cap = (capacity_ > 0) ? capacity_ : kDefaultCapacity;
    const FrameFormat fmt = (format_ != FrameFormat::unknown) ? format_ : FrameFormat::b8g8r8a8_unorm;
    IGpuTextureAllocator* b = backend ? backend : backend_;
    return initialize(b, cap, new_width, new_height, fmt);
}

void* GpuSurfacePool::get_texture(std::uint32_t slot_index) const noexcept {
    return (slot_index < capacity_) ? slots_[slot_index].texture : nullptr;
}

std::uint64_t GpuSurfacePool::get_shared_handle(std::uint32_t slot_index) const noexcept {
    return (slot_index < capacity_) ? slots_[slot_index].shared_handle : 0;
}

std::uint32_t GpuSurfacePool::ref_count(std::uint32_t slot_index) const noexcept {
    return (slot_index < capacity_) ? slots_[slot_index].ref_count.load(std::memory_order_acquire) : 0;
}

std::uint32_t GpuSurfacePool::active_lease_count() const noexcept {
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < capacity_; ++i) {
        if (slots_[i].ref_count.load(std::memory_order_acquire) > 0) {
            ++count;
        }
    }
    return count;
}

} // namespace aim::capture
