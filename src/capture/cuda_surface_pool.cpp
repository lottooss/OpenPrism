// src/capture/cuda_surface_pool.cpp
#include "aim/capture/cuda_surface_pool.hpp"

#include <utility>

namespace aim::capture {

// =============================================================================
// CudaMappedSurfaceLease Implementation
// =============================================================================

void CudaMappedSurfaceLease::reset() noexcept {
    if (pool_ && slot_index_ != GpuSurfacePool::kInvalidSlot) {
        pool_->unmap_surface(slot_index_, std::move(frame_lease_));
    }
    pool_ = nullptr;
    slot_index_ = GpuSurfacePool::kInvalidSlot;
    frame_lease_.reset();
    mapped_array_ = nullptr;
    surface_object_ = 0;
    stream_ = nullptr;
}

// =============================================================================
// CudaSurfacePool Implementation
// =============================================================================

CudaSurfacePool::CudaSurfacePool(std::shared_ptr<ICudaInteropBackend> backend) noexcept
    : backend_(std::move(backend)) {}

CudaSurfacePool::~CudaSurfacePool() noexcept {
    release_all();
}

CudaSurfacePool::CudaSurfacePool(CudaSurfacePool&& other) noexcept
    : backend_(std::move(other.backend_)),
      capacity_(other.capacity_),
      cuda_device_id_(other.cuda_device_id_),
      is_initialized_(other.is_initialized_),
      source_pool_(other.source_pool_),
      processing_stream_(other.processing_stream_),
      slots_(std::move(other.slots_)) {
    if (source_pool_ != nullptr) {
        source_pool_->bind_observer(this);
    }
    other.capacity_ = 0;
    other.cuda_device_id_ = -1;
    other.is_initialized_ = false;
    other.source_pool_ = nullptr;
    other.processing_stream_ = nullptr;
    for (auto& s : other.slots_) {
        s = SlotResource{};
    }
}

CudaSurfacePool& CudaSurfacePool::operator=(CudaSurfacePool&& other) noexcept {
    if (this != &other) {
        release_all();
        backend_ = std::move(other.backend_);
        capacity_ = other.capacity_;
        cuda_device_id_ = other.cuda_device_id_;
        is_initialized_ = other.is_initialized_;
        source_pool_ = other.source_pool_;
        processing_stream_ = other.processing_stream_;
        slots_ = std::move(other.slots_);

        if (source_pool_ != nullptr) {
            source_pool_->bind_observer(this);
        }

        other.capacity_ = 0;
        other.cuda_device_id_ = -1;
        other.is_initialized_ = false;
        other.source_pool_ = nullptr;
        other.processing_stream_ = nullptr;
        for (auto& s : other.slots_) {
            s = SlotResource{};
        }
    }
    return *this;
}

bool CudaSurfacePool::initialize(GpuSurfacePool& gpu_pool, std::uint64_t dxgi_adapter_luid) noexcept {
    release_all();

    if (!backend_ || !backend_->is_cuda_available()) {
        return false;
    }

    if (!gpu_pool.is_initialized() || gpu_pool.capacity() == 0 || gpu_pool.capacity() > kMaxCapacity) {
        return false;
    }

    int dev_id = -1;
    if (!backend_->validate_adapter_match(dxgi_adapter_luid, dev_id)) {
        return false;
    }

    if (backend_->set_device(dev_id) != CudaResult::success) {
        return false;
    }
    cuda_device_id_ = dev_id;

    if (backend_->create_stream(processing_stream_, CudaStreamFlags::non_blocking) != CudaResult::success) {
        release_all();
        return false;
    }

    const std::uint32_t cap = gpu_pool.capacity();
    for (std::uint32_t i = 0; i < cap; ++i) {
        void* tex = gpu_pool.get_texture(i);
        if (!tex) {
            release_all();
            return false;
        }

        CudaGraphicsResourceHandle res = nullptr;
        CudaResult r = backend_->register_d3d11_texture(
            tex, CudaGraphicsRegisterFlags::surface_load_store, res);
        if (r != CudaResult::success || !res) {
            release_all();
            return false;
        }

        slots_[i].graphics_resource = res;
        slots_[i].is_registered = true;
        slots_[i].is_mapped = false;
        slots_[i].mapped_array = nullptr;
        slots_[i].surface_object = 0;

        if (backend_->create_event(slots_[i].ready_event, CudaEventFlags::disable_timing) != CudaResult::success ||
            backend_->create_event(slots_[i].complete_event, CudaEventFlags::disable_timing) != CudaResult::success) {
            release_all();
            return false;
        }
    }

    capacity_ = cap;
    is_initialized_ = true;
    source_pool_ = &gpu_pool;
    source_pool_->bind_observer(this);
    return true;
}

bool CudaSurfacePool::map_surface(FrameLease frame_lease, CudaMappedSurfaceLease& out_mapped_lease) noexcept {
    reap_completed_releases();

    if (!is_initialized_ || !frame_lease.is_valid()) {
        out_mapped_lease.reset();
        return false;
    }

    const std::uint32_t slot = frame_lease.pool_slot_index();
    if (slot >= capacity_ || !slots_[slot].is_registered || slots_[slot].is_release_pending) {
        out_mapped_lease.reset();
        return false;
    }

    if (slots_[slot].is_mapped) {
        out_mapped_lease.reset();
        return false;
    }

    CudaResult r = backend_->map_resources(&slots_[slot].graphics_resource, 1, processing_stream_);
    if (r != CudaResult::success) {
        out_mapped_lease.reset();
        return false;
    }

    r = backend_->get_mapped_array(slots_[slot].graphics_resource, 0, 0, slots_[slot].mapped_array);
    if (r != CudaResult::success || !slots_[slot].mapped_array) {
        backend_->unmap_resources(&slots_[slot].graphics_resource, 1, processing_stream_);
        out_mapped_lease.reset();
        return false;
    }

    if (slots_[slot].surface_object == 0) {
        r = backend_->create_surface_object(slots_[slot].mapped_array, slots_[slot].surface_object);
        if (r != CudaResult::success || slots_[slot].surface_object == 0) {
            backend_->unmap_resources(&slots_[slot].graphics_resource, 1, processing_stream_);
            out_mapped_lease.reset();
            return false;
        }
    }

    slots_[slot].is_mapped = true;

    out_mapped_lease = CudaMappedSurfaceLease(
        this,
        slot,
        std::move(frame_lease),
        slots_[slot].mapped_array,
        slots_[slot].surface_object,
        processing_stream_
    );

    return true;
}

void CudaSurfacePool::unmap_surface(std::uint32_t slot_index, FrameLease frame_lease) noexcept {
    if (!is_initialized_ || slot_index >= capacity_) {
        return;
    }

    auto& slot = slots_[slot_index];
    if (!slot.is_registered || !slot.is_mapped || !frame_lease.is_valid()) {
        return;
    }

    slot.pending_frame_lease = std::move(frame_lease);
    slot.is_release_pending = true;
    const CudaResult unmap_result =
        backend_->unmap_resources(&slot.graphics_resource, 1, processing_stream_);
    if (unmap_result != CudaResult::success) {
        return;
    }
    slot.is_mapped = false;
    if (backend_->record_event(slot.complete_event, processing_stream_) != CudaResult::success) {
        return;
    }

    if (backend_->query_event(slot.complete_event) == CudaResult::success) {
        slot.pending_frame_lease.reset();
        slot.is_release_pending = false;
    }
}

void CudaSurfacePool::reap_completed_releases() noexcept {
    if (!is_initialized_ || !backend_) {
        return;
    }
    for (std::uint32_t i = 0; i < capacity_; ++i) {
        auto& slot = slots_[i];
        if (slot.is_release_pending && slot.complete_event != nullptr &&
            backend_->query_event(slot.complete_event) == CudaResult::success) {
            slot.pending_frame_lease.reset();
            slot.is_release_pending = false;
        }
    }
}

void CudaSurfacePool::release_all() noexcept {
    if (source_pool_ != nullptr) {
        source_pool_->unbind_observer(this);
        source_pool_ = nullptr;
    }
    if (backend_) {
        bool has_outstanding_work = false;
        for (std::uint32_t i = 0; i < kMaxCapacity; ++i) {
            auto& slot = slots_[i];
            if (slot.is_mapped && slot.graphics_resource) {
                if (backend_->unmap_resources(
                        &slot.graphics_resource, 1, processing_stream_) == CudaResult::success) {
                    slot.is_mapped = false;
                }
                has_outstanding_work = true;
            }
            has_outstanding_work = has_outstanding_work || slot.is_release_pending;
        }
        if (has_outstanding_work && processing_stream_) {
            static_cast<void>(backend_->synchronize_stream(processing_stream_));
        }
        for (std::uint32_t i = 0; i < kMaxCapacity; ++i) {
            auto& slot = slots_[i];
            if (slot.surface_object != 0) {
                backend_->destroy_surface_object(slot.surface_object);
                slot.surface_object = 0;
            }
            if (slot.is_registered && slot.graphics_resource) {
                backend_->unregister_resource(slot.graphics_resource);
                slot.graphics_resource = nullptr;
                slot.is_registered = false;
            }
            if (slot.ready_event) {
                backend_->destroy_event(slot.ready_event);
                slot.ready_event = nullptr;
            }
            if (slot.complete_event) {
                backend_->destroy_event(slot.complete_event);
                slot.complete_event = nullptr;
            }
            slot.mapped_array = nullptr;
            slot.pending_frame_lease.reset();
            slot.is_release_pending = false;
        }

        if (processing_stream_) {
            backend_->destroy_stream(processing_stream_);
            processing_stream_ = nullptr;
        }
    }

    capacity_ = 0;
    cuda_device_id_ = -1;
    is_initialized_ = false;
}

void CudaSurfacePool::before_surface_pool_reset(GpuSurfacePool& pool) noexcept {
    if (source_pool_ == &pool) {
        source_pool_ = nullptr;
        release_all();
    }
}

bool CudaSurfacePool::is_slot_mapped(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return false;
    return slots_[slot_index].is_mapped;
}

bool CudaSurfacePool::is_slot_registered(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return false;
    return slots_[slot_index].is_registered;
}

bool CudaSurfacePool::is_slot_release_pending(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return false;
    return slots_[slot_index].is_release_pending;
}

CudaGraphicsResourceHandle CudaSurfacePool::get_graphics_resource(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return nullptr;
    return slots_[slot_index].graphics_resource;
}

CudaEventHandle CudaSurfacePool::get_ready_event(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return nullptr;
    return slots_[slot_index].ready_event;
}

CudaEventHandle CudaSurfacePool::get_complete_event(std::uint32_t slot_index) const noexcept {
    if (slot_index >= capacity_) return nullptr;
    return slots_[slot_index].complete_event;
}

} // namespace aim::capture
