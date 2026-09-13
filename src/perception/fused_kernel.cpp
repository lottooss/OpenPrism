// src/perception/fused_kernel.cpp
#include "aim/perception/fused_kernel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace aim::perception {

bool FusedPreprocessKernel::execute_software(const FusedKernelParams& params) noexcept {
    if (!params.src_pixels || !params.dst_tensor || params.src_width == 0 || params.src_height == 0 ||
        params.dst_width == 0 || params.dst_height == 0) {
        return false;
    }

    auto* out_fp16 = static_cast<std::uint16_t*>(params.dst_tensor);
    const auto* src_bytes = static_cast<const std::uint8_t*>(params.src_pixels);

    const std::uint32_t dst_w = params.dst_width;
    const std::uint32_t dst_h = params.dst_height;
    const std::uint32_t scaled_w = params.scaled_width;
    const std::uint32_t scaled_h = params.scaled_height;
    const float pad_x = params.pad_x;
    const float pad_y = params.pad_y;
    const std::uint16_t pad_half_bits = params.pad_half_bits;

    const std::size_t plane_stride = static_cast<std::size_t>(dst_w) * dst_h;
    std::uint16_t* r_plane = out_fp16;
    std::uint16_t* g_plane = out_fp16 + plane_stride;
    std::uint16_t* b_plane = out_fp16 + 2 * plane_stride;

    const float scale_x = static_cast<float>(params.src_width) / static_cast<float>(scaled_w);
    const float scale_y = static_cast<float>(params.src_height) / static_cast<float>(scaled_h);

    const std::uint8_t* nv12_y = src_bytes;
    const std::uint8_t* nv12_uv = src_bytes + static_cast<std::size_t>(params.src_stride_bytes) * params.src_height;

    for (std::uint32_t yt = 0; yt < dst_h; ++yt) {
        const float yt_f = static_cast<float>(yt);
        if (yt_f < pad_y || yt_f >= (pad_y + static_cast<float>(scaled_h))) {
            const std::size_t row_offset = static_cast<std::size_t>(yt) * dst_w;
            for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
                const std::size_t idx = row_offset + xt;
                r_plane[idx] = pad_half_bits;
                g_plane[idx] = pad_half_bits;
                b_plane[idx] = pad_half_bits;
            }
            continue;
        }

        const float y_act = yt_f - pad_y;
        const float v = (y_act + 0.5f) * scale_y - 0.5f;
        const float v_clamped = std::clamp(v, 0.0f, static_cast<float>(params.src_height - 1));
        const auto y0 = static_cast<std::uint32_t>(v_clamped);
        const std::uint32_t y1 = std::min(y0 + 1, params.src_height - 1);
        const float beta = v_clamped - static_cast<float>(y0);

        const std::size_t row_offset = static_cast<std::size_t>(yt) * dst_w;

        for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
            const float xt_f = static_cast<float>(xt);
            if (xt_f < pad_x || xt_f >= (pad_x + static_cast<float>(scaled_w))) {
                const std::size_t idx = row_offset + xt;
                r_plane[idx] = pad_half_bits;
                g_plane[idx] = pad_half_bits;
                b_plane[idx] = pad_half_bits;
                continue;
            }

            const float x_act = xt_f - pad_x;
            const float u = (x_act + 0.5f) * scale_x - 0.5f;
            const float u_clamped = std::clamp(u, 0.0f, static_cast<float>(params.src_width - 1));
            const auto x0 = static_cast<std::uint32_t>(u_clamped);
            const std::uint32_t x1 = std::min(x0 + 1, params.src_width - 1);
            const float alpha = u_clamped - static_cast<float>(x0);

            const float w00 = (1.0f - alpha) * (1.0f - beta);
            const float w10 = alpha * (1.0f - beta);
            const float w01 = (1.0f - alpha) * beta;
            const float w11 = alpha * beta;

            float r00 = 0.0f, g00 = 0.0f, b00 = 0.0f;
            float r10 = 0.0f, g10 = 0.0f, b10 = 0.0f;
            float r01 = 0.0f, g01 = 0.0f, b01 = 0.0f;
            float r11 = 0.0f, g11 = 0.0f, b11 = 0.0f;

            if (params.src_format == FrameFormat::b8g8r8a8_unorm) {
                const std::size_t off00 = static_cast<std::size_t>(y0) * params.src_stride_bytes + x0 * 4;
                const std::size_t off10 = static_cast<std::size_t>(y0) * params.src_stride_bytes + x1 * 4;
                const std::size_t off01 = static_cast<std::size_t>(y1) * params.src_stride_bytes + x0 * 4;
                const std::size_t off11 = static_cast<std::size_t>(y1) * params.src_stride_bytes + x1 * 4;

                b00 = static_cast<float>(src_bytes[off00 + 0]);
                g00 = static_cast<float>(src_bytes[off00 + 1]);
                r00 = static_cast<float>(src_bytes[off00 + 2]);

                b10 = static_cast<float>(src_bytes[off10 + 0]);
                g10 = static_cast<float>(src_bytes[off10 + 1]);
                r10 = static_cast<float>(src_bytes[off10 + 2]);

                b01 = static_cast<float>(src_bytes[off01 + 0]);
                g01 = static_cast<float>(src_bytes[off01 + 1]);
                r01 = static_cast<float>(src_bytes[off01 + 2]);

                b11 = static_cast<float>(src_bytes[off11 + 0]);
                g11 = static_cast<float>(src_bytes[off11 + 1]);
                r11 = static_cast<float>(src_bytes[off11 + 2]);
            } else if (params.src_format == FrameFormat::r8g8b8a8_unorm) {
                const std::size_t off00 = static_cast<std::size_t>(y0) * params.src_stride_bytes + x0 * 4;
                const std::size_t off10 = static_cast<std::size_t>(y0) * params.src_stride_bytes + x1 * 4;
                const std::size_t off01 = static_cast<std::size_t>(y1) * params.src_stride_bytes + x0 * 4;
                const std::size_t off11 = static_cast<std::size_t>(y1) * params.src_stride_bytes + x1 * 4;

                r00 = static_cast<float>(src_bytes[off00 + 0]);
                g00 = static_cast<float>(src_bytes[off00 + 1]);
                b00 = static_cast<float>(src_bytes[off00 + 2]);

                r10 = static_cast<float>(src_bytes[off10 + 0]);
                g10 = static_cast<float>(src_bytes[off10 + 1]);
                b10 = static_cast<float>(src_bytes[off10 + 2]);

                r01 = static_cast<float>(src_bytes[off01 + 0]);
                g01 = static_cast<float>(src_bytes[off01 + 1]);
                b01 = static_cast<float>(src_bytes[off01 + 2]);

                r11 = static_cast<float>(src_bytes[off11 + 0]);
                g11 = static_cast<float>(src_bytes[off11 + 1]);
                b11 = static_cast<float>(src_bytes[off11 + 2]);
            } else if (params.src_format == FrameFormat::nv12) {
                auto sample_nv12_pixel = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t y_off = static_cast<std::size_t>(y) * params.src_stride_bytes + x;
                    const float y_val = static_cast<float>(nv12_y[y_off]);
                    const std::size_t uv_off = static_cast<std::size_t>(y / 2) * params.src_stride_bytes + (x / 2) * 2;
                    const float u_val = static_cast<float>(nv12_uv[uv_off + 0]);
                    const float v_val = static_cast<float>(nv12_uv[uv_off + 1]);

                    r = std::clamp(y_val + 1.402f * (v_val - 128.0f), 0.0f, 255.0f);
                    g = std::clamp(y_val - 0.344136f * (u_val - 128.0f) - 0.714136f * (v_val - 128.0f), 0.0f, 255.0f);
                    b = std::clamp(y_val + 1.772f * (u_val - 128.0f), 0.0f, 255.0f);
                };
                sample_nv12_pixel(x0, y0, r00, g00, b00);
                sample_nv12_pixel(x1, y0, r10, g10, b10);
                sample_nv12_pixel(x0, y1, r01, g01, b01);
                sample_nv12_pixel(x1, y1, r11, g11, b11);
            } else {
                return false;
            }

            const float r_val = (w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11) * (1.0f / 255.0f);
            const float g_val = (w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11) * (1.0f / 255.0f);
            const float b_val = (w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11) * (1.0f / 255.0f);

            const std::size_t idx = row_offset + xt;
            r_plane[idx] = HalfFloat(r_val).bits;
            g_plane[idx] = HalfFloat(g_val).bits;
            b_plane[idx] = HalfFloat(b_val).bits;
        }
    }

    return true;
}

#if !defined(AIM_HAS_CUDA_PREPROCESS) || AIM_HAS_CUDA_PREPROCESS == 0

bool FusedPreprocessKernel::is_cuda_compiled() noexcept {
    return false;
}

bool FusedPreprocessKernel::allocate_device_tensor(void** out_device_ptr,
                                                   std::size_t /*size_bytes*/) noexcept {
    if (out_device_ptr != nullptr) {
        *out_device_ptr = nullptr;
    }
    return false;
}

void FusedPreprocessKernel::free_device_tensor(void* /*device_ptr*/) noexcept {}

bool FusedPreprocessKernel::warmup_device_tensor(void* /*device_ptr*/,
                                                 std::size_t /*size_bytes*/,
                                                 capture::CudaStreamHandle /*stream*/) noexcept {
    return false;
}

bool FusedPreprocessKernel::launch_cuda_surface(
    capture::CudaSurfaceObjectHandle /*surface_object*/,
    const FusedKernelParams& /*params*/,
    capture::CudaStreamHandle /*stream*/) noexcept {
    return false;
}

#endif

} // namespace aim::perception
