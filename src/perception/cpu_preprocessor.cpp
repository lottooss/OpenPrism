// src/perception/cpu_preprocessor.cpp
#include "aim/perception/cpu_preprocessor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

#include <chrono>

namespace aim::perception {

static MonotonicNs get_current_time_ns() noexcept {
    return static_cast<MonotonicNs>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

CpuReferencePreprocessor::CpuReferencePreprocessor(const PreprocessConfig& config) noexcept
    : config_(config) {}

bool CpuReferencePreprocessor::preprocess(
    const void* src_pixels,
    std::uint32_t src_width,
    std::uint32_t src_height,
    std::uint32_t src_stride_bytes,
    FrameFormat format,
    SequenceId frame_id,
    const CorrelationId& correlation_id,
    MonotonicNs captured_at_ns,
    void* out_tensor_data,
    PreprocessedTensorDescriptor& out_descriptor) const noexcept {

    if (!src_pixels || !out_tensor_data || src_width == 0 || src_height == 0) {
        out_descriptor = {};
        return false;
    }

    auto* out_fp16 = static_cast<std::uint16_t*>(out_tensor_data);
    const auto* src_bytes = static_cast<const std::uint8_t*>(src_pixels);

    const std::uint32_t dst_w = config_.target_width_px;
    const std::uint32_t dst_h = config_.target_height_px;
    const std::uint32_t scaled_w = config_.scaled_width_px;
    const std::uint32_t scaled_h = config_.scaled_height_px;
    const float pad_x = config_.pad_x;
    const float pad_y = config_.pad_y;
    const float pad_norm = config_.pad_value_normalized;
    const std::uint16_t pad_half_bits = HalfFloat(pad_norm).bits;

    const std::size_t plane_stride = static_cast<std::size_t>(dst_w) * dst_h;
    std::uint16_t* r_plane = out_fp16;
    std::uint16_t* g_plane = out_fp16 + plane_stride;
    std::uint16_t* b_plane = out_fp16 + 2 * plane_stride;

    const float scale_x = static_cast<float>(src_width) / static_cast<float>(scaled_w);
    const float scale_y = static_cast<float>(src_height) / static_cast<float>(scaled_h);

    const std::uint8_t* nv12_y = src_bytes;
    const std::uint8_t* nv12_uv = src_bytes + static_cast<std::size_t>(src_stride_bytes) * src_height;

    for (std::uint32_t yt = 0; yt < dst_h; ++yt) {
        const float yt_f = static_cast<float>(yt);
        if (yt_f < pad_y || yt_f >= (pad_y + static_cast<float>(scaled_h))) {
            for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
                const std::size_t idx = yt * dst_w + xt;
                r_plane[idx] = pad_half_bits;
                g_plane[idx] = pad_half_bits;
                b_plane[idx] = pad_half_bits;
            }
            continue;
        }

        const float y_act = yt_f - pad_y;
        const float v = (y_act + 0.5f) * scale_y - 0.5f;
        const float v_clamped = std::clamp(v, 0.0f, static_cast<float>(src_height - 1));
        const auto y0 = static_cast<std::uint32_t>(v_clamped);
        const std::uint32_t y1 = std::min(y0 + 1, src_height - 1);
        const float beta = v_clamped - static_cast<float>(y0);

        for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
            const float xt_f = static_cast<float>(xt);
            if (xt_f < pad_x || xt_f >= (pad_x + static_cast<float>(scaled_w))) {
                const std::size_t idx = yt * dst_w + xt;
                r_plane[idx] = pad_half_bits;
                g_plane[idx] = pad_half_bits;
                b_plane[idx] = pad_half_bits;
                continue;
            }

            const float x_act = xt_f - pad_x;
            const float u = (x_act + 0.5f) * scale_x - 0.5f;
            const float u_clamped = std::clamp(u, 0.0f, static_cast<float>(src_width - 1));
            const auto x0 = static_cast<std::uint32_t>(u_clamped);
            const std::uint32_t x1 = std::min(x0 + 1, src_width - 1);
            const float alpha = u_clamped - static_cast<float>(x0);

            const float w00 = (1.0f - alpha) * (1.0f - beta);
            const float w10 = alpha * (1.0f - beta);
            const float w01 = (1.0f - alpha) * beta;
            const float w11 = alpha * beta;

            float r00 = 0.0f, g00 = 0.0f, b00 = 0.0f;
            float r10 = 0.0f, g10 = 0.0f, b10 = 0.0f;
            float r01 = 0.0f, g01 = 0.0f, b01 = 0.0f;
            float r11 = 0.0f, g11 = 0.0f, b11 = 0.0f;

            if (format == FrameFormat::b8g8r8a8_unorm) {
                auto sample_bgra = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t offset = static_cast<std::size_t>(y) * src_stride_bytes + x * 4;
                    b = static_cast<float>(src_bytes[offset + 0]);
                    g = static_cast<float>(src_bytes[offset + 1]);
                    r = static_cast<float>(src_bytes[offset + 2]);
                };
                sample_bgra(x0, y0, r00, g00, b00);
                sample_bgra(x1, y0, r10, g10, b10);
                sample_bgra(x0, y1, r01, g01, b01);
                sample_bgra(x1, y1, r11, g11, b11);
            } else if (format == FrameFormat::r8g8b8a8_unorm) {
                auto sample_rgba = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t offset = static_cast<std::size_t>(y) * src_stride_bytes + x * 4;
                    r = static_cast<float>(src_bytes[offset + 0]);
                    g = static_cast<float>(src_bytes[offset + 1]);
                    b = static_cast<float>(src_bytes[offset + 2]);
                };
                sample_rgba(x0, y0, r00, g00, b00);
                sample_rgba(x1, y0, r10, g10, b10);
                sample_rgba(x0, y1, r01, g01, b01);
                sample_rgba(x1, y1, r11, g11, b11);
            } else if (format == FrameFormat::nv12) {
                auto sample_nv12 = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t y_offset = static_cast<std::size_t>(y) * src_stride_bytes + x;
                    const float y_val = static_cast<float>(nv12_y[y_offset]);
                    const std::size_t uv_offset = static_cast<std::size_t>(y / 2) * src_stride_bytes + (x / 2) * 2;
                    const float u_val = static_cast<float>(nv12_uv[uv_offset + 0]);
                    const float v_val = static_cast<float>(nv12_uv[uv_offset + 1]);

                    r = std::clamp(y_val + 1.402f * (v_val - 128.0f), 0.0f, 255.0f);
                    g = std::clamp(y_val - 0.344136f * (u_val - 128.0f) - 0.714136f * (v_val - 128.0f), 0.0f, 255.0f);
                    b = std::clamp(y_val + 1.772f * (u_val - 128.0f), 0.0f, 255.0f);
                };
                sample_nv12(x0, y0, r00, g00, b00);
                sample_nv12(x1, y0, r10, g10, b10);
                sample_nv12(x0, y1, r01, g01, b01);
                sample_nv12(x1, y1, r11, g11, b11);
            } else {
                return false;
            }

            const float r_val = (w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11) * (1.0f / 255.0f);
            const float g_val = (w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11) * (1.0f / 255.0f);
            const float b_val = (w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11) * (1.0f / 255.0f);

            const std::size_t idx = yt * dst_w + xt;
            r_plane[idx] = HalfFloat(r_val).bits;
            g_plane[idx] = HalfFloat(g_val).bits;
            b_plane[idx] = HalfFloat(b_val).bits;
        }
    }

    out_descriptor.frame_id = frame_id;
    out_descriptor.correlation_id = correlation_id;
    out_descriptor.captured_at_ns = captured_at_ns;
    out_descriptor.preprocessed_at_ns = get_current_time_ns();
    out_descriptor.width_px = dst_w;
    out_descriptor.height_px = dst_h;
    out_descriptor.channels = config_.channels;
    out_descriptor.size_bytes = config_.tensor_size_bytes();
    out_descriptor.precision = config_.precision;
    out_descriptor.layout = config_.layout;
    out_descriptor.affine_transform = AffineTransform2D::create_letterbox(src_width, src_height, dst_w, dst_h);
    out_descriptor.gpu_tensor_ptr = nullptr;
    out_descriptor.cpu_tensor_ptr = out_tensor_data;
    out_descriptor.is_valid = true;

    return true;
}

bool CpuReferencePreprocessor::preprocess_fp32(
    const void* src_pixels,
    std::uint32_t src_width,
    std::uint32_t src_height,
    std::uint32_t src_stride_bytes,
    FrameFormat format,
    SequenceId frame_id,
    const CorrelationId& correlation_id,
    MonotonicNs captured_at_ns,
    float* out_tensor_data,
    PreprocessedTensorDescriptor& out_descriptor) const noexcept {

    if (!src_pixels || !out_tensor_data || src_width == 0 || src_height == 0) {
        out_descriptor = {};
        return false;
    }

    const auto* src_bytes = static_cast<const std::uint8_t*>(src_pixels);

    const std::uint32_t dst_w = config_.target_width_px;
    const std::uint32_t dst_h = config_.target_height_px;
    const std::uint32_t scaled_w = config_.scaled_width_px;
    const std::uint32_t scaled_h = config_.scaled_height_px;
    const float pad_x = config_.pad_x;
    const float pad_y = config_.pad_y;
    const float pad_norm = config_.pad_value_normalized;

    const std::size_t plane_stride = static_cast<std::size_t>(dst_w) * dst_h;
    float* r_plane = out_tensor_data;
    float* g_plane = out_tensor_data + plane_stride;
    float* b_plane = out_tensor_data + 2 * plane_stride;

    const float scale_x = static_cast<float>(src_width) / static_cast<float>(scaled_w);
    const float scale_y = static_cast<float>(src_height) / static_cast<float>(scaled_h);

    const std::uint8_t* nv12_y = src_bytes;
    const std::uint8_t* nv12_uv = src_bytes + static_cast<std::size_t>(src_stride_bytes) * src_height;

    for (std::uint32_t yt = 0; yt < dst_h; ++yt) {
        const float yt_f = static_cast<float>(yt);
        if (yt_f < pad_y || yt_f >= (pad_y + static_cast<float>(scaled_h))) {
            for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
                const std::size_t idx = yt * dst_w + xt;
                r_plane[idx] = pad_norm;
                g_plane[idx] = pad_norm;
                b_plane[idx] = pad_norm;
            }
            continue;
        }

        const float y_act = yt_f - pad_y;
        const float v = (y_act + 0.5f) * scale_y - 0.5f;
        const float v_clamped = std::clamp(v, 0.0f, static_cast<float>(src_height - 1));
        const auto y0 = static_cast<std::uint32_t>(v_clamped);
        const std::uint32_t y1 = std::min(y0 + 1, src_height - 1);
        const float beta = v_clamped - static_cast<float>(y0);

        for (std::uint32_t xt = 0; xt < dst_w; ++xt) {
            const float xt_f = static_cast<float>(xt);
            if (xt_f < pad_x || xt_f >= (pad_x + static_cast<float>(scaled_w))) {
                const std::size_t idx = yt * dst_w + xt;
                r_plane[idx] = pad_norm;
                g_plane[idx] = pad_norm;
                b_plane[idx] = pad_norm;
                continue;
            }

            const float x_act = xt_f - pad_x;
            const float u = (x_act + 0.5f) * scale_x - 0.5f;
            const float u_clamped = std::clamp(u, 0.0f, static_cast<float>(src_width - 1));
            const auto x0 = static_cast<std::uint32_t>(u_clamped);
            const std::uint32_t x1 = std::min(x0 + 1, src_width - 1);
            const float alpha = u_clamped - static_cast<float>(x0);

            const float w00 = (1.0f - alpha) * (1.0f - beta);
            const float w10 = alpha * (1.0f - beta);
            const float w01 = (1.0f - alpha) * beta;
            const float w11 = alpha * beta;

            float r00 = 0.0f, g00 = 0.0f, b00 = 0.0f;
            float r10 = 0.0f, g10 = 0.0f, b10 = 0.0f;
            float r01 = 0.0f, g01 = 0.0f, b01 = 0.0f;
            float r11 = 0.0f, g11 = 0.0f, b11 = 0.0f;

            if (format == FrameFormat::b8g8r8a8_unorm) {
                auto sample_bgra = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t offset = static_cast<std::size_t>(y) * src_stride_bytes + x * 4;
                    b = static_cast<float>(src_bytes[offset + 0]);
                    g = static_cast<float>(src_bytes[offset + 1]);
                    r = static_cast<float>(src_bytes[offset + 2]);
                };
                sample_bgra(x0, y0, r00, g00, b00);
                sample_bgra(x1, y0, r10, g10, b10);
                sample_bgra(x0, y1, r01, g01, b01);
                sample_bgra(x1, y1, r11, g11, b11);
            } else if (format == FrameFormat::r8g8b8a8_unorm) {
                auto sample_rgba = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t offset = static_cast<std::size_t>(y) * src_stride_bytes + x * 4;
                    r = static_cast<float>(src_bytes[offset + 0]);
                    g = static_cast<float>(src_bytes[offset + 1]);
                    b = static_cast<float>(src_bytes[offset + 2]);
                };
                sample_rgba(x0, y0, r00, g00, b00);
                sample_rgba(x1, y0, r10, g10, b10);
                sample_rgba(x0, y1, r01, g01, b01);
                sample_rgba(x1, y1, r11, g11, b11);
            } else if (format == FrameFormat::nv12) {
                auto sample_nv12 = [&](std::uint32_t x, std::uint32_t y, float& r, float& g, float& b) {
                    const std::size_t y_offset = static_cast<std::size_t>(y) * src_stride_bytes + x;
                    const float y_val = static_cast<float>(nv12_y[y_offset]);
                    const std::size_t uv_offset = static_cast<std::size_t>(y / 2) * src_stride_bytes + (x / 2) * 2;
                    const float u_val = static_cast<float>(nv12_uv[uv_offset + 0]);
                    const float v_val = static_cast<float>(nv12_uv[uv_offset + 1]);

                    r = std::clamp(y_val + 1.402f * (v_val - 128.0f), 0.0f, 255.0f);
                    g = std::clamp(y_val - 0.344136f * (u_val - 128.0f) - 0.714136f * (v_val - 128.0f), 0.0f, 255.0f);
                    b = std::clamp(y_val + 1.772f * (u_val - 128.0f), 0.0f, 255.0f);
                };
                sample_nv12(x0, y0, r00, g00, b00);
                sample_nv12(x1, y0, r10, g10, b10);
                sample_nv12(x0, y1, r01, g01, b01);
                sample_nv12(x1, y1, r11, g11, b11);
            } else {
                return false;
            }

            const float r_val = (w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11) * (1.0f / 255.0f);
            const float g_val = (w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11) * (1.0f / 255.0f);
            const float b_val = (w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11) * (1.0f / 255.0f);

            const std::size_t idx = yt * dst_w + xt;
            r_plane[idx] = r_val;
            g_plane[idx] = g_val;
            b_plane[idx] = b_val;
        }
    }

    out_descriptor.frame_id = frame_id;
    out_descriptor.correlation_id = correlation_id;
    out_descriptor.captured_at_ns = captured_at_ns;
    out_descriptor.preprocessed_at_ns = get_current_time_ns();
    out_descriptor.width_px = dst_w;
    out_descriptor.height_px = dst_h;
    out_descriptor.channels = config_.channels;
    out_descriptor.size_bytes = static_cast<std::size_t>(dst_w) * dst_h * config_.channels * sizeof(float);
    out_descriptor.precision = TensorPrecision::fp32;
    out_descriptor.layout = config_.layout;
    out_descriptor.affine_transform = AffineTransform2D::create_letterbox(src_width, src_height, dst_w, dst_h);
    out_descriptor.gpu_tensor_ptr = nullptr;
    out_descriptor.cpu_tensor_ptr = out_tensor_data;
    out_descriptor.is_valid = true;

    return true;
}

double CpuReferencePreprocessor::compute_mae(
    const std::uint16_t* actual,
    const std::uint16_t* golden,
    std::size_t element_count) noexcept {
    if (!actual || !golden || element_count == 0) {
        return 0.0;
    }
    double sum_error = 0.0;
    for (std::size_t i = 0; i < element_count; ++i) {
        float a = HalfFloat(actual[i]).to_float();
        float g = HalfFloat(golden[i]).to_float();
        sum_error += std::abs(static_cast<double>(a) - static_cast<double>(g));
    }
    return sum_error / static_cast<double>(element_count);
}

float CpuReferencePreprocessor::compute_max_diff(
    const std::uint16_t* actual,
    const std::uint16_t* golden,
    std::size_t element_count) noexcept {
    if (!actual || !golden || element_count == 0) {
        return 0.0f;
    }
    float max_d = 0.0f;
    for (std::size_t i = 0; i < element_count; ++i) {
        float a = HalfFloat(actual[i]).to_float();
        float g = HalfFloat(golden[i]).to_float();
        float d = std::abs(a - g);
        if (d > max_d) {
            max_d = d;
        }
    }
    return max_d;
}

bool CpuReferencePreprocessor::compare_parity(
    const std::uint16_t* actual,
    const std::uint16_t* golden,
    std::size_t element_count,
    double max_mae,
    float max_diff) noexcept {
    if (!actual || !golden || element_count == 0) {
        return false;
    }
    double mae = compute_mae(actual, golden, element_count);
    float max_d = compute_max_diff(actual, golden, element_count);
    return (mae <= max_mae) && (max_d <= max_diff);
}

} // namespace aim::perception
