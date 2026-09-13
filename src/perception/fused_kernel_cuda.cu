#include "aim/perception/fused_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace aim::perception {
namespace {

__global__ void fused_bgra8_to_rgb_fp16_nchw(
    cudaSurfaceObject_t source,
    __half* destination,
    std::uint32_t src_width,
    std::uint32_t src_height,
    std::uint32_t dst_width,
    std::uint32_t dst_height,
    std::uint32_t scaled_width,
    std::uint32_t scaled_height,
    std::uint32_t pad_x,
    std::uint32_t pad_y,
    float scale_x,
    float scale_y,
    __half pad_value) {
    const std::uint32_t xt = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t yt = blockIdx.y * blockDim.y + threadIdx.y;
    if (xt >= dst_width || yt >= dst_height) {
        return;
    }

    const std::size_t plane_size = static_cast<std::size_t>(dst_width) * dst_height;
    const std::size_t output_index = static_cast<std::size_t>(yt) * dst_width + xt;

    if (xt < pad_x || yt < pad_y || xt >= pad_x + scaled_width || yt >= pad_y + scaled_height) {
        destination[output_index] = pad_value;
        destination[plane_size + output_index] = pad_value;
        destination[2U * plane_size + output_index] = pad_value;
        return;
    }

    const float source_x = (static_cast<float>(xt - pad_x) + 0.5F) * scale_x - 0.5F;
    const float source_y = (static_cast<float>(yt - pad_y) + 0.5F) * scale_y - 0.5F;
    const int x0 = max(0, min(static_cast<int>(src_width) - 1, static_cast<int>(floorf(source_x))));
    const int y0 = max(0, min(static_cast<int>(src_height) - 1, static_cast<int>(floorf(source_y))));
    const int x1 = min(x0 + 1, static_cast<int>(src_width) - 1);
    const int y1 = min(y0 + 1, static_cast<int>(src_height) - 1);
    const float wx = source_x - floorf(source_x);
    const float wy = source_y - floorf(source_y);

    const uchar4 p00 = surf2Dread<uchar4>(source, x0 * static_cast<int>(sizeof(uchar4)), y0,
                                          cudaBoundaryModeClamp);
    const uchar4 p10 = surf2Dread<uchar4>(source, x1 * static_cast<int>(sizeof(uchar4)), y0,
                                          cudaBoundaryModeClamp);
    const uchar4 p01 = surf2Dread<uchar4>(source, x0 * static_cast<int>(sizeof(uchar4)), y1,
                                          cudaBoundaryModeClamp);
    const uchar4 p11 = surf2Dread<uchar4>(source, x1 * static_cast<int>(sizeof(uchar4)), y1,
                                          cudaBoundaryModeClamp);

    const float w00 = (1.0F - wx) * (1.0F - wy);
    const float w10 = wx * (1.0F - wy);
    const float w01 = (1.0F - wx) * wy;
    const float w11 = wx * wy;
    constexpr float kNormalize = 1.0F / 255.0F;

    const float red = (w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z) * kNormalize;
    const float green = (w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y) * kNormalize;
    const float blue = (w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x) * kNormalize;

    destination[output_index] = __float2half_rn(red);
    destination[plane_size + output_index] = __float2half_rn(green);
    destination[2U * plane_size + output_index] = __float2half_rn(blue);
}

} // namespace

bool FusedPreprocessKernel::is_cuda_compiled() noexcept {
    return true;
}

bool FusedPreprocessKernel::allocate_device_tensor(void** out_device_ptr,
                                                   std::size_t size_bytes) noexcept {
    if (out_device_ptr == nullptr || size_bytes == 0) {
        return false;
    }
    *out_device_ptr = nullptr;
    return cudaMalloc(out_device_ptr, size_bytes) == cudaSuccess;
}

void FusedPreprocessKernel::free_device_tensor(void* device_ptr) noexcept {
    if (device_ptr != nullptr) {
        static_cast<void>(cudaFree(device_ptr));
    }
}

bool FusedPreprocessKernel::warmup_device_tensor(void* device_ptr,
                                                 std::size_t size_bytes,
                                                 capture::CudaStreamHandle stream) noexcept {
    if (device_ptr == nullptr || size_bytes == 0 || stream == nullptr) {
        return false;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return cudaMemsetAsync(device_ptr, 0, size_bytes, cuda_stream) == cudaSuccess;
}

bool FusedPreprocessKernel::launch_cuda_surface(
    capture::CudaSurfaceObjectHandle surface_object,
    const FusedKernelParams& params,
    capture::CudaStreamHandle stream) noexcept {
    if (surface_object == 0 || params.dst_tensor == nullptr || stream == nullptr ||
        params.src_format != FrameFormat::b8g8r8a8_unorm || params.src_width == 0 ||
        params.src_height == 0 || params.dst_width == 0 || params.dst_height == 0 ||
        params.scaled_width == 0 || params.scaled_height == 0) {
        return false;
    }

    const dim3 block(16U, 16U);
    const dim3 grid((params.dst_width + block.x - 1U) / block.x,
                    (params.dst_height + block.y - 1U) / block.y);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fused_bgra8_to_rgb_fp16_nchw<<<grid, block, 0, cuda_stream>>>(
        static_cast<cudaSurfaceObject_t>(surface_object),
        static_cast<__half*>(params.dst_tensor),
        params.src_width,
        params.src_height,
        params.dst_width,
        params.dst_height,
        params.scaled_width,
        params.scaled_height,
        static_cast<std::uint32_t>(params.pad_x),
        static_cast<std::uint32_t>(params.pad_y),
        params.scale_x,
        params.scale_y,
        __float2half_rn(params.pad_value_normalized));
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace aim::perception
