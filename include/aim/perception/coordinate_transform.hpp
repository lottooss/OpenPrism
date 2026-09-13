// include/aim/perception/coordinate_transform.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "aim/core/types.hpp"

namespace aim::perception {

/// @brief Canonical 2D affine transformation representing forward and inverse letterbox/scaling mappings.
///
/// Coordinate conventions:
/// - Source pixels: [0, src_width_px] x [0, src_height_px], origin top-left, +x right, +y down.
/// - Model pixels: [0, dst_width_px] x [0, dst_height_px], origin top-left, +x right, +y down.
/// - Normalized screen: [-1, 1] x [-1, 1], origin screen center (0, 0), top-left (-1, -1), bottom-right (+1, +1).
struct AffineTransform2D {
    float sx{1.0f / 3.0f};
    float sy{1.0f / 3.0f};
    float padx{0.0f};
    float pady{12.0f};

    // Forward & Inverse Matrix Representations (Row-major 2x3 and 3x3)
    [[nodiscard]] constexpr std::array<float, 6> forward_matrix_2x3() const noexcept {
        return {sx, 0.0f, padx, 0.0f, sy, pady};
    }

    [[nodiscard]] constexpr std::array<float, 9> forward_matrix_3x3() const noexcept {
        return {sx, 0.0f, padx, 0.0f, sy, pady, 0.0f, 0.0f, 1.0f};
    }

    [[nodiscard]] constexpr std::array<float, 6> inverse_matrix_2x3() const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return {inv_sx, 0.0f, -padx * inv_sx, 0.0f, inv_sy, -pady * inv_sy};
    }

    [[nodiscard]] constexpr std::array<float, 9> inverse_matrix_3x3() const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return {inv_sx, 0.0f, -padx * inv_sx, 0.0f, inv_sy, -pady * inv_sy, 0.0f, 0.0f, 1.0f};
    }

    // Point Projections
    [[nodiscard]] constexpr PixelPoint forward_point(const PixelPoint& pt) const noexcept {
        return PixelPoint{pt.x * sx + padx, pt.y * sy + pady};
    }

    [[nodiscard]] constexpr PixelPoint inverse_point(const PixelPoint& pt) const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return PixelPoint{(pt.x - padx) * inv_sx, (pt.y - pady) * inv_sy};
    }

    // Bounding Box Projections
    [[nodiscard]] constexpr BoundingBox forward_box(const BoundingBox& box) const noexcept {
        return BoundingBox{
            box.left * sx + padx,
            box.top * sy + pady,
            box.right * sx + padx,
            box.bottom * sy + pady
        };
    }

    [[nodiscard]] constexpr BoundingBox inverse_box(const BoundingBox& box) const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return BoundingBox{
            (box.left - padx) * inv_sx,
            (box.top - pady) * inv_sy,
            (box.right - padx) * inv_sx,
            (box.bottom - pady) * inv_sy
        };
    }

    // Radius / Scalar Extent Scaling
    [[nodiscard]] inline float forward_radius(float r) const noexcept {
        // Uniform aspect ratio scaling: r' = s * r
        return r * std::sqrt(sx * sy);
    }

    [[nodiscard]] inline float inverse_radius(float r) const noexcept {
        float scale = std::sqrt(sx * sy);
        return (scale != 0.0f) ? (r / scale) : 0.0f;
    }

    // Kinematic Projections (Translations do not affect velocity or acceleration)
    [[nodiscard]] constexpr PixelVelocity forward_velocity(const PixelVelocity& vel) const noexcept {
        return PixelVelocity{vel.x_per_s * sx, vel.y_per_s * sy};
    }

    [[nodiscard]] constexpr PixelVelocity inverse_velocity(const PixelVelocity& vel) const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return PixelVelocity{vel.x_per_s * inv_sx, vel.y_per_s * inv_sy};
    }

    [[nodiscard]] constexpr PixelAcceleration forward_acceleration(const PixelAcceleration& acc) const noexcept {
        return PixelAcceleration{acc.x_per_s2 * sx, acc.y_per_s2 * sy};
    }

    [[nodiscard]] constexpr PixelAcceleration inverse_acceleration(const PixelAcceleration& acc) const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return PixelAcceleration{acc.x_per_s2 * inv_sx, acc.y_per_s2 * inv_sy};
    }

    // Covariance 2D Matrix Scaling (Sigma' = A * Sigma * A^T)
    [[nodiscard]] constexpr Covariance2D forward_covariance(const Covariance2D& cov) const noexcept {
        return Covariance2D{
            cov.xx * (sx * sx),
            cov.xy * (sx * sy),
            cov.yy * (sy * sy)
        };
    }

    [[nodiscard]] constexpr Covariance2D inverse_covariance(const Covariance2D& cov) const noexcept {
        float inv_sx = (sx != 0.0f) ? (1.0f / sx) : 0.0f;
        float inv_sy = (sy != 0.0f) ? (1.0f / sy) : 0.0f;
        return Covariance2D{
            cov.xx * (inv_sx * inv_sx),
            cov.xy * (inv_sx * inv_sy),
            cov.yy * (inv_sy * inv_sy)
        };
    }

    // Letterbox Factory for arbitrary source and target dimensions
    static constexpr AffineTransform2D create_letterbox(
        std::uint32_t src_w, std::uint32_t src_h,
        std::uint32_t dst_w, std::uint32_t dst_h) noexcept {
        if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
            return AffineTransform2D{1.0f, 1.0f, 0.0f, 0.0f};
        }

        float scale_x = static_cast<float>(dst_w) / static_cast<float>(src_w);
        float scale_y = static_cast<float>(dst_h) / static_cast<float>(src_h);
        float scale = (scale_x < scale_y) ? scale_x : scale_y;

        float scaled_w = static_cast<float>(src_w) * scale;
        float scaled_h = static_cast<float>(src_h) * scale;

        float pad_x = (static_cast<float>(dst_w) - scaled_w) * 0.5f;
        float pad_y = (static_cast<float>(dst_h) - scaled_h) * 0.5f;

        return AffineTransform2D{scale, scale, pad_x, pad_y};
    }
};

/// @brief Utility converting between pixel coordinates and canonical [-1, 1] normalized screen space.
class CoordinateConverter {
public:
    /// @brief Converts source screen pixels to normalized [-1, 1] coordinate space.
    /// (0, 0) is screen center, (-1, -1) is top-left, (+1, +1) is bottom-right.
    [[nodiscard]] static constexpr NormalizedPoint pixel_to_normalized(
        const PixelPoint& pt, float screen_width, float screen_height) noexcept {
        if (screen_width <= 0.0f || screen_height <= 0.0f) {
            return NormalizedPoint{0.0f, 0.0f};
        }
        float norm_x = (pt.x / (screen_width * 0.5f)) - 1.0f;
        float norm_y = (pt.y / (screen_height * 0.5f)) - 1.0f;
        return NormalizedPoint{norm_x, norm_y};
    }

    /// @brief Converts normalized [-1, 1] coordinate space to source screen pixels.
    [[nodiscard]] static constexpr PixelPoint normalized_to_pixel(
        const NormalizedPoint& npt, float screen_width, float screen_height) noexcept {
        if (screen_width <= 0.0f || screen_height <= 0.0f) {
            return PixelPoint{0.0f, 0.0f};
        }
        float px_x = (npt.x + 1.0f) * (screen_width * 0.5f);
        float px_y = (npt.y + 1.0f) * (screen_height * 0.5f);
        return PixelPoint{px_x, px_y};
    }

    /// @brief Calculates center point of a bounding box.
    [[nodiscard]] static constexpr PixelPoint box_center(const BoundingBox& box) noexcept {
        return PixelPoint{
            (box.left + box.right) * 0.5f,
            (box.top + box.bottom) * 0.5f
        };
    }

    /// @brief Calculates width and height of a bounding box.
    [[nodiscard]] static constexpr PixelPoint box_extent(const BoundingBox& box) noexcept {
        float w = box.right - box.left;
        float h = box.bottom - box.top;
        return PixelPoint{
            (w >= 0.0f) ? w : -w,
            (h >= 0.0f) ? h : -h
        };
    }

    /// @brief Directly converts a model detection tensor pixel coordinate into normalized screen coordinates [-1, 1].
    [[nodiscard]] static constexpr NormalizedPoint model_to_normalized_screen(
        const PixelPoint& model_pt,
        const AffineTransform2D& transform,
        float screen_width,
        float screen_height) noexcept {
        PixelPoint src_pt = transform.inverse_point(model_pt);
        return pixel_to_normalized(src_pt, screen_width, screen_height);
    }

    /// @brief Directly converts normalized screen coordinates [-1, 1] into model detection tensor pixel coordinates.
    [[nodiscard]] static constexpr PixelPoint normalized_screen_to_model(
        const NormalizedPoint& norm_pt,
        const AffineTransform2D& transform,
        float screen_width,
        float screen_height) noexcept {
        PixelPoint src_pt = normalized_to_pixel(norm_pt, screen_width, screen_height);
        return transform.forward_point(src_pt);
    }
};

} // namespace aim::perception
