// include/aim/perception/fused_kernel.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/perception/preprocess_types.hpp"

namespace aim::perception {

/// @brief Parameter bundle passed directly into the fused preprocessing kernel.
struct FusedKernelParams {
    const void* src_pixels{nullptr};
    std::uint32_t src_width{1920};
    std::uint32_t src_height{1080};
    std::uint32_t src_stride_bytes{1920 * 4};
    FrameFormat src_format{FrameFormat::b8g8r8a8_unorm};

    void* dst_tensor{nullptr};
    std::uint32_t dst_width{640};
    std::uint32_t dst_height{384};
    std::uint32_t scaled_width{640};
    std::uint32_t scaled_height{360};

    float pad_x{0.0f};
    float pad_y{12.0f};
    float pad_value_normalized{114.0f / 255.0f};
    std::uint16_t pad_half_bits{0x3727}; // HalfFloat(114.0f / 255.0f).bits

    float scale_x{3.0f}; // 1920 / 640
    float scale_y{3.0f}; // 1080 / 360
};

/// @brief Host launcher for the fused single-pass GPU preprocessing kernel.
class FusedPreprocessKernel {
public:
    /// @brief Returns whether the CUDA kernel was compiled into this binary.
    [[nodiscard]] static bool is_cuda_compiled() noexcept;

    /// @brief Allocates a device-resident output tensor during initialization.
    static bool allocate_device_tensor(void** out_device_ptr, std::size_t size_bytes) noexcept;

    /// @brief Releases a tensor allocated by allocate_device_tensor during teardown.
    static void free_device_tensor(void* device_ptr) noexcept;

    /// @brief Warms the CUDA runtime and output buffer without consuming a capture surface.
    static bool warmup_device_tensor(void* device_ptr,
                                     std::size_t size_bytes,
                                     capture::CudaStreamHandle stream) noexcept;

    /// @brief Launches the fused kernel against a mapped D3D11 CUDA surface object.
    static bool launch_cuda_surface(capture::CudaSurfaceObjectHandle surface_object,
                                    const FusedKernelParams& params,
                                    capture::CudaStreamHandle stream) noexcept;

    /// @brief Highly optimized single-pass software execution matching GPU kernel logic.
    static bool execute_software(const FusedKernelParams& params) noexcept;
};

} // namespace aim::perception
