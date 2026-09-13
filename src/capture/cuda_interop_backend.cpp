// src/capture/cuda_interop_backend.cpp
#include "aim/capture/cuda_interop_backend.hpp"

#include <cstring>
#include <iostream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#endif

namespace aim::capture {

// =============================================================================
// CUDA Driver API Type Definitions & Function Pointer Signatures
// =============================================================================

using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUevent = struct CUevent_st*;
using CUarray = struct CUarray_st*;
using CUmipmappedArray = struct CUmipmappedArray_st*;
using CUgraphicsResource = struct CUgraphicsResource_st*;
using CUsurfObject = unsigned long long;
using CUdeviceptr = unsigned long long;

enum CUresult_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_UNKNOWN = 999
};
using CUresult = CUresult_enum;

enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY           = 0x00,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
    CU_RESOURCE_TYPE_LINEAR          = 0x02,
    CU_RESOURCE_TYPE_PITCH2D         = 0x03
};
using CUresourcetype = CUresourcetype_enum;

enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8  = 0x01,
    CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    CU_AD_FORMAT_SIGNED_INT8    = 0x08,
    CU_AD_FORMAT_SIGNED_INT16   = 0x09,
    CU_AD_FORMAT_SIGNED_INT32   = 0x0a,
    CU_AD_FORMAT_HALF           = 0x10,
    CU_AD_FORMAT_FLOAT          = 0x20
};
using CUarray_format = CUarray_format_enum;

struct CUDA_RESOURCE_DESC {
    CUresourcetype resType;
    union {
        struct {
            CUarray hArray;
        } array;
        struct {
            CUmipmappedArray hMipmappedArray;
        } mipmap;
        struct {
            CUdeviceptr devPtr;
            CUarray_format format;
            unsigned int numChannels;
            size_t sizeInBytes;
        } linear;
        struct {
            CUdeviceptr devPtr;
            CUarray_format format;
            unsigned int numChannels;
            size_t width;
            size_t height;
            size_t pitchInBytes;
        } pitch2D;
        struct {
            int reserved[32];
        } reserved;
    } res;
    unsigned int flags;
};

using PFN_cuInit = CUresult (*)(unsigned int flags);
using PFN_cuDeviceGetCount = CUresult (*)(int* count);
using PFN_cuDeviceGet = CUresult (*)(CUdevice* device, int ordinal);
using PFN_cuDeviceGetName = CUresult (*)(char* name, int len, CUdevice dev);
using PFN_cuDeviceGetLuid = CUresult (*)(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
using PFN_cuDevicePrimaryCtxRetain = CUresult (*)(CUcontext* pctx, CUdevice dev);
using PFN_cuDevicePrimaryCtxRelease = CUresult (*)(CUdevice dev);
using PFN_cuCtxSetCurrent = CUresult (*)(CUcontext ctx);
using PFN_cuCtxGetCurrent = CUresult (*)(CUcontext* pctx);
using PFN_cuGraphicsD3D11RegisterResource = CUresult (*)(CUgraphicsResource* pCudaResource, void* pD3DResource, unsigned int flags);
using PFN_cuGraphicsUnregisterResource = CUresult (*)(CUgraphicsResource resource);
using PFN_cuGraphicsMapResources = CUresult (*)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
using PFN_cuGraphicsUnmapResources = CUresult (*)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
using PFN_cuGraphicsSubResourceGetMappedArray = CUresult (*)(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
using PFN_cuSurfObjectCreate = CUresult (*)(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
using PFN_cuSurfObjectDestroy = CUresult (*)(CUsurfObject surfObject);
using PFN_cuStreamCreate = CUresult (*)(CUstream* phStream, unsigned int flags);
using PFN_cuStreamDestroy_v2 = CUresult (*)(CUstream hStream);
using PFN_cuEventCreate = CUresult (*)(CUevent* phEvent, unsigned int flags);
using PFN_cuEventDestroy_v2 = CUresult (*)(CUevent hEvent);
using PFN_cuEventRecord = CUresult (*)(CUevent hEvent, CUstream hStream);
using PFN_cuStreamWaitEvent = CUresult (*)(CUstream hStream, CUevent hEvent, unsigned int flags);
using PFN_cuEventQuery = CUresult (*)(CUevent hEvent);
using PFN_cuStreamSynchronize = CUresult (*)(CUstream hStream);
using PFN_cuGetErrorString = CUresult (*)(CUresult error, const char** pStr);
using PFN_cuGetErrorName = CUresult (*)(CUresult error, const char** pStr);

static inline CudaResult to_cuda_result(CUresult r) noexcept {
    switch (r) {
        case CUDA_SUCCESS: return CudaResult::success;
        case CUDA_ERROR_INVALID_VALUE: return CudaResult::error_invalid_value;
        case CUDA_ERROR_OUT_OF_MEMORY: return CudaResult::error_out_of_memory;
        case CUDA_ERROR_NOT_INITIALIZED: return CudaResult::error_not_initialized;
        case CUDA_ERROR_DEINITIALIZED: return CudaResult::error_deinitialized;
        case CUDA_ERROR_NO_DEVICE: return CudaResult::error_no_device;
        case CUDA_ERROR_INVALID_DEVICE: return CudaResult::error_invalid_device;
        case CUDA_ERROR_INVALID_IMAGE: return CudaResult::error_invalid_image;
        case CUDA_ERROR_INVALID_CONTEXT: return CudaResult::error_invalid_context;
        case CUDA_ERROR_MAP_FAILED: return CudaResult::error_map_failed;
        case CUDA_ERROR_UNMAP_FAILED: return CudaResult::error_unmap_failed;
        case CUDA_ERROR_ARRAY_IS_MAPPED: return CudaResult::error_array_is_mapped;
        case CUDA_ERROR_ALREADY_MAPPED: return CudaResult::error_already_mapped;
        case CUDA_ERROR_NO_BINARY_FOR_GPU: return CudaResult::error_no_binary_for_gpu;
        case CUDA_ERROR_ALREADY_ACQUIRED: return CudaResult::error_already_acquired;
        case CUDA_ERROR_NOT_MAPPED: return CudaResult::error_not_mapped;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return CudaResult::error_not_mapped_as_array;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return CudaResult::error_not_mapped_as_pointer;
        case CUDA_ERROR_ECC_UNCORRECTABLE: return CudaResult::error_ecc_uncorrectable;
        case CUDA_ERROR_UNSUPPORTED_LIMIT: return CudaResult::error_unsupported_limit;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return CudaResult::error_context_already_in_use;
        case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: return CudaResult::error_peer_access_unsupported;
        case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: return CudaResult::error_invalid_graphics_context;
        case CUDA_ERROR_NOT_READY: return CudaResult::error_not_ready;
        default: return CudaResult::error_unknown;
    }
}

// =============================================================================
// RealCudaInteropBackend Implementation
// =============================================================================

struct RealCudaInteropBackend::Impl {
    bool cuda_available{false};
    int active_device_id{-1};
    CUdevice active_device{-1};
    CUcontext primary_ctx{nullptr};
    bool primary_ctx_retained{false};

#if defined(_WIN32)
    HMODULE nvcuda_module{nullptr};
#endif

    // Function pointers
    PFN_cuInit fn_cuInit{nullptr};
    PFN_cuDeviceGetCount fn_cuDeviceGetCount{nullptr};
    PFN_cuDeviceGet fn_cuDeviceGet{nullptr};
    PFN_cuDeviceGetName fn_cuDeviceGetName{nullptr};
    PFN_cuDeviceGetLuid fn_cuDeviceGetLuid{nullptr};
    PFN_cuDevicePrimaryCtxRetain fn_cuDevicePrimaryCtxRetain{nullptr};
    PFN_cuDevicePrimaryCtxRelease fn_cuDevicePrimaryCtxRelease{nullptr};
    PFN_cuCtxSetCurrent fn_cuCtxSetCurrent{nullptr};
    PFN_cuCtxGetCurrent fn_cuCtxGetCurrent{nullptr};
    PFN_cuGraphicsD3D11RegisterResource fn_cuGraphicsD3D11RegisterResource{nullptr};
    PFN_cuGraphicsUnregisterResource fn_cuGraphicsUnregisterResource{nullptr};
    PFN_cuGraphicsMapResources fn_cuGraphicsMapResources{nullptr};
    PFN_cuGraphicsUnmapResources fn_cuGraphicsUnmapResources{nullptr};
    PFN_cuGraphicsSubResourceGetMappedArray fn_cuGraphicsSubResourceGetMappedArray{nullptr};
    PFN_cuSurfObjectCreate fn_cuSurfObjectCreate{nullptr};
    PFN_cuSurfObjectDestroy fn_cuSurfObjectDestroy{nullptr};
    PFN_cuStreamCreate fn_cuStreamCreate{nullptr};
    PFN_cuStreamDestroy_v2 fn_cuStreamDestroy{nullptr};
    PFN_cuEventCreate fn_cuEventCreate{nullptr};
    PFN_cuEventDestroy_v2 fn_cuEventDestroy{nullptr};
    PFN_cuEventRecord fn_cuEventRecord{nullptr};
    PFN_cuStreamWaitEvent fn_cuStreamWaitEvent{nullptr};
    PFN_cuEventQuery fn_cuEventQuery{nullptr};
    PFN_cuStreamSynchronize fn_cuStreamSynchronize{nullptr};
    PFN_cuGetErrorString fn_cuGetErrorString{nullptr};
    PFN_cuGetErrorName fn_cuGetErrorName{nullptr};

    Impl() noexcept {
#if defined(_WIN32)
        nvcuda_module = LoadLibraryW(L"nvcuda.dll");
        if (!nvcuda_module) {
            cuda_available = false;
            return;
        }

        auto load_sym = [this](const char* name) -> FARPROC {
            return GetProcAddress(nvcuda_module, name);
        };

        fn_cuInit = reinterpret_cast<PFN_cuInit>(load_sym("cuInit"));
        fn_cuDeviceGetCount = reinterpret_cast<PFN_cuDeviceGetCount>(load_sym("cuDeviceGetCount"));
        fn_cuDeviceGet = reinterpret_cast<PFN_cuDeviceGet>(load_sym("cuDeviceGet"));
        fn_cuDeviceGetName = reinterpret_cast<PFN_cuDeviceGetName>(load_sym("cuDeviceGetName"));
        fn_cuDeviceGetLuid = reinterpret_cast<PFN_cuDeviceGetLuid>(load_sym("cuDeviceGetLuid"));
        fn_cuDevicePrimaryCtxRetain = reinterpret_cast<PFN_cuDevicePrimaryCtxRetain>(load_sym("cuDevicePrimaryCtxRetain"));
        fn_cuDevicePrimaryCtxRelease = reinterpret_cast<PFN_cuDevicePrimaryCtxRelease>(load_sym("cuDevicePrimaryCtxRelease"));
        fn_cuCtxSetCurrent = reinterpret_cast<PFN_cuCtxSetCurrent>(load_sym("cuCtxSetCurrent"));
        fn_cuCtxGetCurrent = reinterpret_cast<PFN_cuCtxGetCurrent>(load_sym("cuCtxGetCurrent"));
        fn_cuGraphicsD3D11RegisterResource = reinterpret_cast<PFN_cuGraphicsD3D11RegisterResource>(load_sym("cuGraphicsD3D11RegisterResource"));
        fn_cuGraphicsUnregisterResource = reinterpret_cast<PFN_cuGraphicsUnregisterResource>(load_sym("cuGraphicsUnregisterResource"));
        fn_cuGraphicsMapResources = reinterpret_cast<PFN_cuGraphicsMapResources>(load_sym("cuGraphicsMapResources"));
        fn_cuGraphicsUnmapResources = reinterpret_cast<PFN_cuGraphicsUnmapResources>(load_sym("cuGraphicsUnmapResources"));
        fn_cuGraphicsSubResourceGetMappedArray = reinterpret_cast<PFN_cuGraphicsSubResourceGetMappedArray>(load_sym("cuGraphicsSubResourceGetMappedArray"));
        fn_cuSurfObjectCreate = reinterpret_cast<PFN_cuSurfObjectCreate>(load_sym("cuSurfObjectCreate"));
        fn_cuSurfObjectDestroy = reinterpret_cast<PFN_cuSurfObjectDestroy>(load_sym("cuSurfObjectDestroy"));
        fn_cuStreamCreate = reinterpret_cast<PFN_cuStreamCreate>(load_sym("cuStreamCreate"));
        fn_cuStreamDestroy = reinterpret_cast<PFN_cuStreamDestroy_v2>(load_sym("cuStreamDestroy_v2"));
        if (!fn_cuStreamDestroy) {
            fn_cuStreamDestroy = reinterpret_cast<PFN_cuStreamDestroy_v2>(load_sym("cuStreamDestroy"));
        }
        fn_cuEventCreate = reinterpret_cast<PFN_cuEventCreate>(load_sym("cuEventCreate"));
        fn_cuEventDestroy = reinterpret_cast<PFN_cuEventDestroy_v2>(load_sym("cuEventDestroy_v2"));
        if (!fn_cuEventDestroy) {
            fn_cuEventDestroy = reinterpret_cast<PFN_cuEventDestroy_v2>(load_sym("cuEventDestroy"));
        }
        fn_cuEventRecord = reinterpret_cast<PFN_cuEventRecord>(load_sym("cuEventRecord"));
        fn_cuStreamWaitEvent = reinterpret_cast<PFN_cuStreamWaitEvent>(load_sym("cuStreamWaitEvent"));
        fn_cuEventQuery = reinterpret_cast<PFN_cuEventQuery>(load_sym("cuEventQuery"));
        fn_cuStreamSynchronize = reinterpret_cast<PFN_cuStreamSynchronize>(load_sym("cuStreamSynchronize"));
        fn_cuGetErrorString = reinterpret_cast<PFN_cuGetErrorString>(load_sym("cuGetErrorString"));
        fn_cuGetErrorName = reinterpret_cast<PFN_cuGetErrorName>(load_sym("cuGetErrorName"));

        if (!fn_cuInit || !fn_cuDeviceGetCount || !fn_cuDeviceGet || !fn_cuDeviceGetLuid ||
            !fn_cuDevicePrimaryCtxRetain || !fn_cuDevicePrimaryCtxRelease || !fn_cuCtxSetCurrent ||
            !fn_cuGraphicsD3D11RegisterResource || !fn_cuGraphicsUnregisterResource ||
            !fn_cuGraphicsMapResources || !fn_cuGraphicsUnmapResources ||
            !fn_cuGraphicsSubResourceGetMappedArray || !fn_cuSurfObjectCreate || !fn_cuSurfObjectDestroy ||
            !fn_cuStreamCreate || !fn_cuStreamDestroy || !fn_cuEventCreate || !fn_cuEventDestroy ||
            !fn_cuEventRecord || !fn_cuStreamWaitEvent || !fn_cuEventQuery || !fn_cuStreamSynchronize) {
            cuda_available = false;
            return;
        }

        CUresult r = fn_cuInit(0);
        cuda_available = (r == CUDA_SUCCESS);
#else
        cuda_available = false;
#endif
    }

    ~Impl() noexcept {
#if defined(_WIN32)
        if (primary_ctx_retained && active_device_id >= 0 && fn_cuDevicePrimaryCtxRelease) {
            fn_cuDevicePrimaryCtxRelease(active_device);
            primary_ctx_retained = false;
            primary_ctx = nullptr;
        }
        if (nvcuda_module) {
            FreeLibrary(nvcuda_module);
            nvcuda_module = nullptr;
        }
#endif
    }
};

RealCudaInteropBackend::RealCudaInteropBackend() noexcept
    : impl_(std::make_unique<Impl>()) {}

RealCudaInteropBackend::~RealCudaInteropBackend() noexcept = default;

bool RealCudaInteropBackend::is_cuda_available() const noexcept {
    return impl_ && impl_->cuda_available;
}

bool RealCudaInteropBackend::validate_adapter_match(std::uint64_t dxgi_adapter_luid,
                                                    int& out_cuda_device_id) noexcept {
    out_cuda_device_id = -1;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuDeviceGetCount ||
        !impl_->fn_cuDeviceGet || !impl_->fn_cuDeviceGetLuid) {
        return false;
    }

    int count = 0;
    if (impl_->fn_cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        CUdevice dev = -1;
        if (impl_->fn_cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            continue;
        }

        char luid_bytes[8]{0};
        unsigned int node_mask = 0;
        if (impl_->fn_cuDeviceGetLuid(luid_bytes, &node_mask, dev) != CUDA_SUCCESS) {
            continue;
        }

        std::uint64_t cuda_luid = 0;
        std::memcpy(&cuda_luid, luid_bytes, sizeof(cuda_luid));
        if (cuda_luid == dxgi_adapter_luid) {
            out_cuda_device_id = i;
            return true;
        }
    }

    return false;
}

CudaResult RealCudaInteropBackend::set_device(int device_id) noexcept {
    if (!impl_ || !impl_->cuda_available) {
        return CudaResult::error_no_device;
    }
    if (device_id < 0) {
        return CudaResult::error_invalid_device;
    }

    int count = 0;
    if (impl_->fn_cuDeviceGetCount(&count) != CUDA_SUCCESS || device_id >= count) {
        return CudaResult::error_invalid_device;
    }

    if (impl_->primary_ctx_retained && impl_->active_device_id == device_id) {
        if (impl_->primary_ctx && impl_->fn_cuCtxSetCurrent) {
            impl_->fn_cuCtxSetCurrent(impl_->primary_ctx);
        }
        return CudaResult::success;
    }

    if (impl_->primary_ctx_retained && impl_->fn_cuDevicePrimaryCtxRelease) {
        impl_->fn_cuDevicePrimaryCtxRelease(impl_->active_device);
        impl_->primary_ctx_retained = false;
        impl_->primary_ctx = nullptr;
    }

    CUdevice dev = -1;
    CUresult r = impl_->fn_cuDeviceGet(&dev, device_id);
    if (r != CUDA_SUCCESS) {
        return to_cuda_result(r);
    }

    CUcontext ctx = nullptr;
    r = impl_->fn_cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS) {
        return to_cuda_result(r);
    }

    r = impl_->fn_cuCtxSetCurrent(ctx);
    if (r != CUDA_SUCCESS) {
        impl_->fn_cuDevicePrimaryCtxRelease(dev);
        return to_cuda_result(r);
    }

    impl_->active_device = dev;
    impl_->active_device_id = device_id;
    impl_->primary_ctx = ctx;
    impl_->primary_ctx_retained = true;

    return CudaResult::success;
}

CudaResult RealCudaInteropBackend::register_d3d11_texture(void* d3d11_texture,
                                                         CudaGraphicsRegisterFlags flags,
                                                         CudaGraphicsResourceHandle& out_resource) noexcept {
    out_resource = nullptr;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuGraphicsD3D11RegisterResource) {
        return CudaResult::error_not_initialized;
    }
    if (!d3d11_texture) {
        return CudaResult::error_invalid_value;
    }

    unsigned int cu_flags = 0;
    const auto uflags = static_cast<std::uint32_t>(flags);
    if ((uflags & static_cast<std::uint32_t>(CudaGraphicsRegisterFlags::read_only)) != 0) {
        cu_flags |= 0x1;
    }
    if ((uflags & static_cast<std::uint32_t>(CudaGraphicsRegisterFlags::write_discard)) != 0) {
        cu_flags |= 0x2;
    }
    if ((uflags & static_cast<std::uint32_t>(CudaGraphicsRegisterFlags::surface_load_store)) != 0) {
        cu_flags |= 0x4;
    }
    if ((uflags & static_cast<std::uint32_t>(CudaGraphicsRegisterFlags::texture_gather)) != 0) {
        cu_flags |= 0x8;
    }

    CUgraphicsResource res = nullptr;
    CUresult r = impl_->fn_cuGraphicsD3D11RegisterResource(&res, d3d11_texture, cu_flags);
    if (r == CUDA_SUCCESS) {
        out_resource = reinterpret_cast<CudaGraphicsResourceHandle>(res);
    }
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::unregister_resource(CudaGraphicsResourceHandle resource) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuGraphicsUnregisterResource) {
        return CudaResult::error_not_initialized;
    }
    if (!resource) {
        return CudaResult::error_invalid_value;
    }

    auto res = reinterpret_cast<CUgraphicsResource>(resource);
    CUresult r = impl_->fn_cuGraphicsUnregisterResource(res);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::map_resources(CudaGraphicsResourceHandle* resources,
                                                std::size_t count,
                                                CudaStreamHandle stream) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuGraphicsMapResources) {
        return CudaResult::error_not_initialized;
    }
    if (!resources || count == 0) {
        return CudaResult::error_invalid_value;
    }

    auto* cu_resources = reinterpret_cast<CUgraphicsResource*>(resources);
    auto cu_stream = reinterpret_cast<CUstream>(stream);
    CUresult r = impl_->fn_cuGraphicsMapResources(static_cast<unsigned int>(count), cu_resources, cu_stream);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::unmap_resources(CudaGraphicsResourceHandle* resources,
                                                  std::size_t count,
                                                  CudaStreamHandle stream) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuGraphicsUnmapResources) {
        return CudaResult::error_not_initialized;
    }
    if (!resources || count == 0) {
        return CudaResult::error_invalid_value;
    }

    auto* cu_resources = reinterpret_cast<CUgraphicsResource*>(resources);
    auto cu_stream = reinterpret_cast<CUstream>(stream);
    CUresult r = impl_->fn_cuGraphicsUnmapResources(static_cast<unsigned int>(count), cu_resources, cu_stream);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::get_mapped_array(CudaGraphicsResourceHandle resource,
                                                   unsigned int array_index,
                                                   unsigned int mip_level,
                                                   CudaArrayHandle& out_array) noexcept {
    out_array = nullptr;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuGraphicsSubResourceGetMappedArray) {
        return CudaResult::error_not_initialized;
    }
    if (!resource) {
        return CudaResult::error_invalid_value;
    }

    auto res = reinterpret_cast<CUgraphicsResource>(resource);
    CUarray arr = nullptr;
    CUresult r = impl_->fn_cuGraphicsSubResourceGetMappedArray(&arr, res, array_index, mip_level);
    if (r == CUDA_SUCCESS) {
        out_array = reinterpret_cast<CudaArrayHandle>(arr);
    }
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::create_surface_object(CudaArrayHandle array,
                                                        CudaSurfaceObjectHandle& out_surface_object) noexcept {
    out_surface_object = 0;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuSurfObjectCreate) {
        return CudaResult::error_not_initialized;
    }
    if (!array) {
        return CudaResult::error_invalid_value;
    }

    CUDA_RESOURCE_DESC desc{};
    desc.resType = CU_RESOURCE_TYPE_ARRAY;
    desc.res.array.hArray = reinterpret_cast<CUarray>(array);
    desc.flags = 0;

    CUsurfObject surf = 0;
    CUresult r = impl_->fn_cuSurfObjectCreate(&surf, &desc);
    if (r == CUDA_SUCCESS) {
        out_surface_object = static_cast<CudaSurfaceObjectHandle>(surf);
    }
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::destroy_surface_object(CudaSurfaceObjectHandle surface_object) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuSurfObjectDestroy) {
        return CudaResult::error_not_initialized;
    }
    if (surface_object == 0) {
        return CudaResult::error_invalid_value;
    }

    CUresult r = impl_->fn_cuSurfObjectDestroy(static_cast<CUsurfObject>(surface_object));
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::create_stream(CudaStreamHandle& out_stream,
                                                CudaStreamFlags flags) noexcept {
    out_stream = nullptr;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuStreamCreate) {
        return CudaResult::error_not_initialized;
    }

    unsigned int cu_flags = (flags == CudaStreamFlags::non_blocking) ? 1 : 0;
    CUstream st = nullptr;
    CUresult r = impl_->fn_cuStreamCreate(&st, cu_flags);
    if (r == CUDA_SUCCESS) {
        out_stream = reinterpret_cast<CudaStreamHandle>(st);
    }
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::destroy_stream(CudaStreamHandle stream) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuStreamDestroy) {
        return CudaResult::error_not_initialized;
    }
    if (!stream) {
        return CudaResult::error_invalid_value;
    }

    auto st = reinterpret_cast<CUstream>(stream);
    CUresult r = impl_->fn_cuStreamDestroy(st);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::create_event(CudaEventHandle& out_event,
                                               CudaEventFlags flags) noexcept {
    out_event = nullptr;
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuEventCreate) {
        return CudaResult::error_not_initialized;
    }

    unsigned int cu_flags = 0;
    const auto uflags = static_cast<std::uint32_t>(flags);
    if ((uflags & static_cast<std::uint32_t>(CudaEventFlags::blocking_sync)) != 0) {
        cu_flags |= 0x1;
    }
    if ((uflags & static_cast<std::uint32_t>(CudaEventFlags::disable_timing)) != 0) {
        cu_flags |= 0x2;
    }
    if ((uflags & static_cast<std::uint32_t>(CudaEventFlags::interprocess)) != 0) {
        cu_flags |= 0x4;
    }

    CUevent ev = nullptr;
    CUresult r = impl_->fn_cuEventCreate(&ev, cu_flags);
    if (r == CUDA_SUCCESS) {
        out_event = reinterpret_cast<CudaEventHandle>(ev);
    }
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::destroy_event(CudaEventHandle event) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuEventDestroy) {
        return CudaResult::error_not_initialized;
    }
    if (!event) {
        return CudaResult::error_invalid_value;
    }

    auto ev = reinterpret_cast<CUevent>(event);
    CUresult r = impl_->fn_cuEventDestroy(ev);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::record_event(CudaEventHandle event, CudaStreamHandle stream) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuEventRecord) {
        return CudaResult::error_not_initialized;
    }
    if (!event || !stream) {
        return CudaResult::error_invalid_value;
    }

    auto ev = reinterpret_cast<CUevent>(event);
    auto st = reinterpret_cast<CUstream>(stream);
    CUresult r = impl_->fn_cuEventRecord(ev, st);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::stream_wait_event(CudaStreamHandle stream,
                                                    CudaEventHandle event,
                                                    unsigned int flags) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuStreamWaitEvent) {
        return CudaResult::error_not_initialized;
    }
    if (!stream || !event) {
        return CudaResult::error_invalid_value;
    }

    auto st = reinterpret_cast<CUstream>(stream);
    auto ev = reinterpret_cast<CUevent>(event);
    CUresult r = impl_->fn_cuStreamWaitEvent(st, ev, flags);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::query_event(CudaEventHandle event) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuEventQuery) {
        return CudaResult::error_not_initialized;
    }
    if (!event) {
        return CudaResult::error_invalid_value;
    }

    auto ev = reinterpret_cast<CUevent>(event);
    CUresult r = impl_->fn_cuEventQuery(ev);
    return to_cuda_result(r);
}

CudaResult RealCudaInteropBackend::synchronize_stream(CudaStreamHandle stream) noexcept {
    if (!impl_ || !impl_->cuda_available || !impl_->fn_cuStreamSynchronize) {
        return CudaResult::error_not_initialized;
    }
    if (!stream) {
        return CudaResult::error_invalid_value;
    }

    auto st = reinterpret_cast<CUstream>(stream);
    CUresult r = impl_->fn_cuStreamSynchronize(st);
    return to_cuda_result(r);
}

const char* RealCudaInteropBackend::get_error_string(CudaResult result) const noexcept {
    switch (result) {
        case CudaResult::success: return "cudaSuccess";
        case CudaResult::error_invalid_value: return "cudaErrorInvalidValue";
        case CudaResult::error_out_of_memory: return "cudaErrorMemoryAllocation";
        case CudaResult::error_not_initialized: return "cudaErrorInitializationError";
        case CudaResult::error_deinitialized: return "cudaErrorDeinitialized";
        case CudaResult::error_no_device: return "cudaErrorNoDevice";
        case CudaResult::error_invalid_device: return "cudaErrorInvalidDevice";
        case CudaResult::error_invalid_image: return "cudaErrorInvalidImage";
        case CudaResult::error_invalid_context: return "cudaErrorInvalidContext";
        case CudaResult::error_map_failed: return "cudaErrorMapBufferObjectFailed";
        case CudaResult::error_unmap_failed: return "cudaErrorUnmapBufferObjectFailed";
        case CudaResult::error_array_is_mapped: return "cudaErrorArrayIsMapped";
        case CudaResult::error_already_mapped: return "cudaErrorAlreadyMapped";
        case CudaResult::error_no_binary_for_gpu: return "cudaErrorNoBinaryForGpu";
        case CudaResult::error_already_acquired: return "cudaErrorAlreadyAcquired";
        case CudaResult::error_not_mapped: return "cudaErrorNotMapped";
        case CudaResult::error_not_mapped_as_array: return "cudaErrorNotMappedAsArray";
        case CudaResult::error_not_mapped_as_pointer: return "cudaErrorNotMappedAsPointer";
        case CudaResult::error_ecc_uncorrectable: return "cudaErrorECCUncorrectable";
        case CudaResult::error_unsupported_limit: return "cudaErrorUnsupportedLimit";
        case CudaResult::error_context_already_in_use: return "cudaErrorContextAlreadyInUse";
        case CudaResult::error_peer_access_unsupported: return "cudaErrorPeerAccessUnsupported";
        case CudaResult::error_invalid_graphics_context: return "cudaErrorInvalidGraphicsContext";
        case CudaResult::error_not_ready: return "cudaErrorNotReady";
        case CudaResult::error_adapter_mismatch: return "cudaErrorAdapterMismatch";
        default: return "cudaErrorUnknown";
    }
}

// =============================================================================
// MockCudaInteropBackend Implementation
// =============================================================================

MockCudaInteropBackend::~MockCudaInteropBackend() noexcept {
    resources_.clear();
}

bool MockCudaInteropBackend::validate_adapter_match(std::uint64_t dxgi_adapter_luid,
                                                   int& out_cuda_device_id) noexcept {
    if (!cuda_available_ || !adapter_match_) {
        out_cuda_device_id = -1;
        return false;
    }
    if (expected_adapter_luid_ != 0 && dxgi_adapter_luid != expected_adapter_luid_) {
        out_cuda_device_id = -1;
        return false;
    }
    out_cuda_device_id = mock_device_id_;
    return true;
}

CudaResult MockCudaInteropBackend::set_device(int device_id) noexcept {
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (device_id < 0 || (adapter_match_ && device_id != mock_device_id_)) {
        return CudaResult::error_invalid_device;
    }
    current_device_id_ = device_id;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::register_d3d11_texture(void* d3d11_texture,
                                                         CudaGraphicsRegisterFlags flags,
                                                         CudaGraphicsResourceHandle& out_resource) noexcept {
    ++register_count_;
    if (!cuda_available_) {
        out_resource = nullptr;
        return CudaResult::error_no_device;
    }
    if (register_result_ != CudaResult::success) {
        out_resource = nullptr;
        return register_result_;
    }
    if (!d3d11_texture) {
        out_resource = nullptr;
        return CudaResult::error_invalid_value;
    }

    auto handle = reinterpret_cast<CudaGraphicsResourceHandle>(++resource_handle_seq_);
    MockResourceInfo info{};
    info.d3d11_texture = d3d11_texture;
    info.register_flags = flags;
    info.mapped_array = reinterpret_cast<CudaArrayHandle>(++array_handle_seq_);
    info.surface_object = ++surface_handle_seq_;
    info.is_mapped = false;
    info.mapped_stream = nullptr;

    resources_[handle] = info;
    out_resource = handle;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::unregister_resource(CudaGraphicsResourceHandle resource) noexcept {
    ++unregister_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    auto it = resources_.find(resource);
    if (it == resources_.end()) {
        return CudaResult::error_invalid_value;
    }
    if (it->second.is_mapped) {
        return CudaResult::error_already_mapped;
    }
    resources_.erase(it);
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::map_resources(CudaGraphicsResourceHandle* resources,
                                                std::size_t count,
                                                CudaStreamHandle stream) noexcept {
    ++map_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!queued_map_results_.empty()) {
        auto res = queued_map_results_.front();
        queued_map_results_.pop();
        if (res != CudaResult::success) {
            return res;
        }
    }
    if (map_result_ != CudaResult::success) {
        return map_result_;
    }
    if (!resources || count == 0) {
        return CudaResult::error_invalid_value;
    }

    for (std::size_t i = 0; i < count; ++i) {
        auto it = resources_.find(resources[i]);
        if (it == resources_.end()) {
            return CudaResult::error_invalid_value;
        }
        if (it->second.is_mapped) {
            return CudaResult::error_already_mapped;
        }
        it->second.is_mapped = true;
        it->second.mapped_stream = stream;
    }

    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::unmap_resources(CudaGraphicsResourceHandle* resources,
                                                  std::size_t count,
                                                  CudaStreamHandle /*stream*/) noexcept {
    ++unmap_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (unmap_result_ != CudaResult::success) {
        return unmap_result_;
    }
    if (!resources || count == 0) {
        return CudaResult::error_invalid_value;
    }

    for (std::size_t i = 0; i < count; ++i) {
        auto it = resources_.find(resources[i]);
        if (it == resources_.end()) {
            return CudaResult::error_invalid_value;
        }
        if (!it->second.is_mapped) {
            return CudaResult::error_not_mapped;
        }
        it->second.is_mapped = false;
        it->second.mapped_stream = nullptr;
    }

    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::get_mapped_array(CudaGraphicsResourceHandle resource,
                                                   unsigned int /*array_index*/,
                                                   unsigned int /*mip_level*/,
                                                   CudaArrayHandle& out_array) noexcept {
    ++get_array_count_;
    if (!cuda_available_) {
        out_array = nullptr;
        return CudaResult::error_no_device;
    }
    if (get_array_result_ != CudaResult::success) {
        out_array = nullptr;
        return get_array_result_;
    }
    auto it = resources_.find(resource);
    if (it == resources_.end()) {
        out_array = nullptr;
        return CudaResult::error_invalid_value;
    }
    if (!it->second.is_mapped) {
        out_array = nullptr;
        return CudaResult::error_not_mapped;
    }
    out_array = it->second.mapped_array;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::create_surface_object(CudaArrayHandle array,
                                                        CudaSurfaceObjectHandle& out_surface_object) noexcept {
    ++create_surface_count_;
    if (!cuda_available_) {
        out_surface_object = 0;
        return CudaResult::error_no_device;
    }
    if (create_surface_result_ != CudaResult::success) {
        out_surface_object = 0;
        return create_surface_result_;
    }
    if (!array) {
        out_surface_object = 0;
        return CudaResult::error_invalid_value;
    }
    out_surface_object = ++surface_handle_seq_;
    ++active_surfaces_;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::destroy_surface_object(CudaSurfaceObjectHandle surface_object) noexcept {
    ++destroy_surface_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (destroy_surface_result_ != CudaResult::success) {
        return destroy_surface_result_;
    }
    if (surface_object == 0) {
        return CudaResult::error_invalid_value;
    }
    if (active_surfaces_ > 0) {
        --active_surfaces_;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::create_stream(CudaStreamHandle& out_stream,
                                                CudaStreamFlags /*flags*/) noexcept {
    ++stream_create_count_;
    if (!cuda_available_) {
        out_stream = nullptr;
        return CudaResult::error_no_device;
    }
    out_stream = reinterpret_cast<CudaStreamHandle>(++stream_handle_seq_);
    ++active_streams_;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::destroy_stream(CudaStreamHandle stream) noexcept {
    ++stream_destroy_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!stream) {
        return CudaResult::error_invalid_value;
    }
    if (active_streams_ > 0) {
        --active_streams_;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::create_event(CudaEventHandle& out_event,
                                               CudaEventFlags /*flags*/) noexcept {
    ++event_create_count_;
    if (!cuda_available_) {
        out_event = nullptr;
        return CudaResult::error_no_device;
    }
    out_event = reinterpret_cast<CudaEventHandle>(++event_handle_seq_);
    ++active_events_;
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::destroy_event(CudaEventHandle event) noexcept {
    ++event_destroy_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!event) {
        return CudaResult::error_invalid_value;
    }
    if (active_events_ > 0) {
        --active_events_;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::record_event(CudaEventHandle event, CudaStreamHandle stream) noexcept {
    ++event_record_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!event || !stream) {
        return CudaResult::error_invalid_value;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::stream_wait_event(CudaStreamHandle stream,
                                                    CudaEventHandle event,
                                                    unsigned int /*flags*/) noexcept {
    ++stream_wait_event_count_;
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!stream || !event) {
        return CudaResult::error_invalid_value;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::query_event(CudaEventHandle event) noexcept {
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!event) {
        return CudaResult::error_invalid_value;
    }
    if (!queued_event_query_results_.empty()) {
        const CudaResult result = queued_event_query_results_.front();
        queued_event_query_results_.pop();
        return result;
    }
    return CudaResult::success;
}

CudaResult MockCudaInteropBackend::synchronize_stream(CudaStreamHandle stream) noexcept {
    if (!cuda_available_) {
        return CudaResult::error_no_device;
    }
    if (!stream) {
        return CudaResult::error_invalid_value;
    }
    return CudaResult::success;
}

bool MockCudaInteropBackend::is_resource_mapped(CudaGraphicsResourceHandle res) const noexcept {
    auto it = resources_.find(res);
    if (it == resources_.end()) return false;
    return it->second.is_mapped;
}

const char* MockCudaInteropBackend::get_error_string(CudaResult result) const noexcept {
    switch (result) {
        case CudaResult::success: return "cudaSuccess";
        case CudaResult::error_invalid_value: return "cudaErrorInvalidValue";
        case CudaResult::error_out_of_memory: return "cudaErrorMemoryAllocation";
        case CudaResult::error_not_initialized: return "cudaErrorInitializationError";
        case CudaResult::error_deinitialized: return "cudaErrorDeinitialized";
        case CudaResult::error_no_device: return "cudaErrorNoDevice";
        case CudaResult::error_invalid_device: return "cudaErrorInvalidDevice";
        case CudaResult::error_invalid_image: return "cudaErrorInvalidImage";
        case CudaResult::error_invalid_context: return "cudaErrorInvalidContext";
        case CudaResult::error_map_failed: return "cudaErrorMapBufferObjectFailed";
        case CudaResult::error_unmap_failed: return "cudaErrorUnmapBufferObjectFailed";
        case CudaResult::error_array_is_mapped: return "cudaErrorArrayIsMapped";
        case CudaResult::error_already_mapped: return "cudaErrorAlreadyMapped";
        case CudaResult::error_no_binary_for_gpu: return "cudaErrorNoBinaryForGpu";
        case CudaResult::error_already_acquired: return "cudaErrorAlreadyAcquired";
        case CudaResult::error_not_mapped: return "cudaErrorNotMapped";
        case CudaResult::error_not_mapped_as_array: return "cudaErrorNotMappedAsArray";
        case CudaResult::error_not_mapped_as_pointer: return "cudaErrorNotMappedAsPointer";
        case CudaResult::error_ecc_uncorrectable: return "cudaErrorECCUncorrectable";
        case CudaResult::error_unsupported_limit: return "cudaErrorUnsupportedLimit";
        case CudaResult::error_context_already_in_use: return "cudaErrorContextAlreadyInUse";
        case CudaResult::error_peer_access_unsupported: return "cudaErrorPeerAccessUnsupported";
        case CudaResult::error_invalid_graphics_context: return "cudaErrorInvalidGraphicsContext";
        case CudaResult::error_adapter_mismatch: return "cudaErrorAdapterMismatch";
        default: return "cudaErrorUnknown";
    }
}

} // namespace aim::capture
