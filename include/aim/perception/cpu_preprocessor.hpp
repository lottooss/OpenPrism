// include/aim/perception/cpu_preprocessor.hpp
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>
#include "aim/perception/preprocess_types.hpp"

namespace aim::perception {

/// @brief Golden CPU reference implementation for bilinear letterbox preprocessing.
class CpuReferencePreprocessor {
public:
    explicit CpuReferencePreprocessor(const PreprocessConfig& config = {}) noexcept;
    ~CpuReferencePreprocessor() noexcept = default;

    /// @brief Executes CPU reference preprocessing from source image to planar FP16 tensor.
    /// @param src_pixels Pointer to raw source pixels (BGRA8, RGBA8, or NV12).
    /// @param src_width Source image width (e.g. 1920).
    /// @param src_height Source image height (e.g. 1080).
    /// @param src_stride_bytes Source row stride in bytes (e.g. 1920 * 4 for BGRA8).
    /// @param format Format of the source pixels.
    /// @param frame_id Frame sequence identifier.
    /// @param correlation_id Monotonic correlation header.
    /// @param captured_at_ns Capture timestamp.
    /// @param out_tensor_data Output buffer for preprocessed tensor (size >= config.tensor_size_bytes()).
    /// @param out_descriptor Attached preprocessed tensor descriptor with affine matrices.
    /// @return true on success, false if pointers invalid or dimensions mismatch.
    bool preprocess(const void* src_pixels,
                    std::uint32_t src_width,
                    std::uint32_t src_height,
                    std::uint32_t src_stride_bytes,
                    FrameFormat format,
                    SequenceId frame_id,
                    const CorrelationId& correlation_id,
                    MonotonicNs captured_at_ns,
                    void* out_tensor_data,
                    PreprocessedTensorDescriptor& out_descriptor) const noexcept;

    /// @brief Preprocess with FP32 planar output.
    bool preprocess_fp32(const void* src_pixels,
                         std::uint32_t src_width,
                         std::uint32_t src_height,
                         std::uint32_t src_stride_bytes,
                         FrameFormat format,
                         SequenceId frame_id,
                         const CorrelationId& correlation_id,
                         MonotonicNs captured_at_ns,
                         float* out_tensor_data,
                         PreprocessedTensorDescriptor& out_descriptor) const noexcept;

    [[nodiscard]] const PreprocessConfig& config() const noexcept { return config_; }

    // Parity and verification helpers
    static double compute_mae(const std::uint16_t* actual,
                              const std::uint16_t* golden,
                              std::size_t element_count) noexcept;

    static float compute_max_diff(const std::uint16_t* actual,
                                  const std::uint16_t* golden,
                                  std::size_t element_count) noexcept;

    static bool compare_parity(const std::uint16_t* actual,
                               const std::uint16_t* golden,
                               std::size_t element_count,
                               double max_mae = 1e-3,
                               float max_diff = 0.002f) noexcept;

private:
    PreprocessConfig config_{};
};

} // namespace aim::perception
