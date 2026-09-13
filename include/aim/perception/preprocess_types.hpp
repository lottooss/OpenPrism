// include/aim/perception/preprocess_types.hpp
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/coordinate_transform.hpp"

namespace aim::perception {

enum class TensorPrecision : std::uint8_t {
    fp16 = 0,
    fp32 = 1,
    int8 = 2
};

enum class TensorLayout : std::uint8_t {
    nchw = 0, // Planar [Batch, Channels, Height, Width]
    nhwc = 1  // Interleaved [Batch, Height, Width, Channels]
};

/// @brief Preprocessing configuration specifying input, output, scaling, and padding parameters.
struct PreprocessConfig {
    std::uint32_t source_width_px{1920};
    std::uint32_t source_height_px{1080};
    FrameFormat source_format{FrameFormat::b8g8r8a8_unorm};

    std::uint32_t target_width_px{640};
    std::uint32_t target_height_px{384};
    std::uint32_t scaled_width_px{640};
    std::uint32_t scaled_height_px{360};

    float pad_x{0.0f};
    float pad_y{12.0f};
    std::uint8_t pad_value_u8{114};
    float pad_value_normalized{114.0f / 255.0f};

    float scale_factor_x{1.0f / 3.0f};
    float scale_factor_y{1.0f / 3.0f};

    TensorPrecision precision{TensorPrecision::fp16};
    TensorLayout layout{TensorLayout::nchw};
    std::uint32_t batch_size{1};
    std::uint32_t channels{3};

    [[nodiscard]] constexpr std::size_t element_count() const noexcept {
        return static_cast<std::size_t>(batch_size) * channels * target_height_px * target_width_px;
    }

    [[nodiscard]] constexpr std::size_t bytes_per_element() const noexcept {
        return (precision == TensorPrecision::fp16) ? 2 : (precision == TensorPrecision::fp32 ? 4 : 1);
    }

    [[nodiscard]] constexpr std::size_t tensor_size_bytes() const noexcept {
        return element_count() * bytes_per_element();
    }
};

/// @brief IEEE-754 binary16 (FP16) half-precision floating point utility.
struct HalfFloat {
    std::uint16_t bits{0};

    constexpr HalfFloat() noexcept = default;
    constexpr explicit HalfFloat(std::uint16_t raw_bits) noexcept : bits(raw_bits) {}
    explicit HalfFloat(float f) noexcept : bits(float_to_half_bits(f)) {}

    [[nodiscard]] float to_float() const noexcept {
        return half_bits_to_float(bits);
    }

    [[nodiscard]] explicit operator float() const noexcept {
        return to_float();
    }

    [[nodiscard]] bool operator==(const HalfFloat& other) const noexcept {
        return bits == other.bits;
    }

    static std::uint16_t float_to_half_bits(float f) noexcept {
        std::uint32_t x = 0;
        std::memcpy(&x, &f, sizeof(x));
        std::uint32_t sign = (x >> 16) & 0x8000U;
        int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFU) - 127 + 15;
        uint32_t mant = x & 0x007FFFFFU;

        if (exp <= 0) {
            if (exp < -10) {
                return static_cast<std::uint16_t>(sign);
            }
            mant = (mant | 0x00800000U) >> (1 - exp);
            if ((mant & 0x00001000U) != 0U) {
                mant += 0x00002000U;
            }
            return static_cast<std::uint16_t>(sign | (mant >> 13));
        } else if (exp >= 31) {
            if (exp == 143 && mant != 0U) {
                return static_cast<std::uint16_t>(sign | 0x7E00U | (mant >> 13));
            }
            return static_cast<std::uint16_t>(sign | 0x7C00U);
        }

        if ((mant & 0x00001000U) != 0U) {
            mant += 0x00001FFFU + ((mant >> 13) & 1U);
        }
        if ((mant & 0x00800000U) != 0U) {
            mant = 0U;
            exp++;
            if (exp >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7C00U);
            }
        }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
    }

    static float half_bits_to_float(std::uint16_t h) noexcept {
        std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000U) << 16;
        std::uint32_t exp = (static_cast<std::uint32_t>(h) >> 10) & 0x1FU;
        std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x03FFU;

        std::uint32_t f_bits = 0U;
        if (exp == 0U) {
            if (mant == 0U) {
                f_bits = sign;
            } else {
                while ((mant & 0x0400U) == 0U) {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= ~0x0400U;
                f_bits = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
            }
        } else if (exp == 31U) {
            f_bits = sign | 0x7F800000U | (mant << 13);
        } else {
            f_bits = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
        }

        float result = 0.0f;
        std::memcpy(&result, &f_bits, sizeof(result));
        return result;
    }
};

/// @brief Preprocessed tensor metadata and descriptor accompanying inference requests.
struct PreprocessedTensorDescriptor {
    SequenceId frame_id{0};
    CorrelationId correlation_id{};
    MonotonicNs captured_at_ns{0};
    MonotonicNs preprocessed_at_ns{0};

    std::uint32_t width_px{640};
    std::uint32_t height_px{384};
    std::uint32_t channels{3};
    std::size_t size_bytes{1474560}; // 640 * 384 * 3 * 2 for FP16 NCHW

    TensorPrecision precision{TensorPrecision::fp16};
    TensorLayout layout{TensorLayout::nchw};
    AffineTransform2D affine_transform{};

    void* gpu_tensor_ptr{nullptr};
    const void* cpu_tensor_ptr{nullptr};
    void* native_ready_event{nullptr};
    std::uint32_t pool_slot_index{0};
    bool is_valid{false};
};

} // namespace aim::perception
