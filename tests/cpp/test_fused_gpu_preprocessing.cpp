// tests/cpp/test_fused_gpu_preprocessing.cpp
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/cpu_preprocessor.hpp"
#include "aim/perception/fused_kernel.hpp"
#include "aim/perception/gpu_preprocessor.hpp"
#include "aim/perception/preprocess_types.hpp"

using namespace aim;
using namespace aim::perception;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

static MonotonicNs get_current_time_ns() noexcept {
    return static_cast<MonotonicNs>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

void generate_uniform_frame(std::vector<std::uint8_t>& buf,
                            std::uint32_t width,
                            std::uint32_t height,
                            std::uint8_t r,
                            std::uint8_t g,
                            std::uint8_t b,
                            FrameFormat format) {
    if (format == FrameFormat::b8g8r8a8_unorm) {
        buf.resize(static_cast<std::size_t>(width) * height * 4);
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
            buf[i * 4 + 0] = b;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = r;
            buf[i * 4 + 3] = 255;
        }
    } else if (format == FrameFormat::r8g8b8a8_unorm) {
        buf.resize(static_cast<std::size_t>(width) * height * 4);
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
            buf[i * 4 + 0] = r;
            buf[i * 4 + 1] = g;
            buf[i * 4 + 2] = b;
            buf[i * 4 + 3] = 255;
        }
    } else if (format == FrameFormat::nv12) {
        buf.resize(static_cast<std::size_t>(width) * height + static_cast<std::size_t>(width) * (height / 2));
        float y_f = 0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) + 0.114f * static_cast<float>(b);
        float u_f = -0.169f * static_cast<float>(r) - 0.331f * static_cast<float>(g) + 0.500f * static_cast<float>(b) + 128.0f;
        float v_f = 0.500f * static_cast<float>(r) - 0.419f * static_cast<float>(g) - 0.081f * static_cast<float>(b) + 128.0f;

        auto y_val = static_cast<std::uint8_t>(std::clamp(y_f, 0.0f, 255.0f));
        auto u_val = static_cast<std::uint8_t>(std::clamp(u_f, 0.0f, 255.0f));
        auto v_val = static_cast<std::uint8_t>(std::clamp(v_f, 0.0f, 255.0f));

        std::fill_n(buf.data(), static_cast<std::size_t>(width) * height, y_val);
        auto* uv = buf.data() + static_cast<std::size_t>(width) * height;
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * (height / 2); i += 2) {
            uv[i + 0] = u_val;
            uv[i + 1] = v_val;
        }
    }
}

void generate_gradient_frame(std::vector<std::uint8_t>& buf,
                             std::uint32_t width,
                             std::uint32_t height,
                             FrameFormat format) {
    buf.resize(static_cast<std::size_t>(width) * height * 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            std::size_t idx = (static_cast<std::size_t>(y) * width + x) * 4;
            auto r = static_cast<std::uint8_t>((x * 255) / width);
            auto g = static_cast<std::uint8_t>((y * 255) / height);
            auto b = static_cast<std::uint8_t>(((x + y) * 255) / (width + height));
            if (format == FrameFormat::b8g8r8a8_unorm) {
                buf[idx + 0] = b;
                buf[idx + 1] = g;
                buf[idx + 2] = r;
                buf[idx + 3] = 255;
            } else {
                buf[idx + 0] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
                buf[idx + 3] = 255;
            }
        }
    }
}

void generate_checkerboard_frame(std::vector<std::uint8_t>& buf,
                                 std::uint32_t width,
                                 std::uint32_t height,
                                 std::uint32_t square_size,
                                 FrameFormat format) {
    buf.resize(static_cast<std::size_t>(width) * height * 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            std::size_t idx = (static_cast<std::size_t>(y) * width + x) * 4;
            bool check = ((x / square_size) + (y / square_size)) % 2 == 0;
            std::uint8_t val = check ? 240 : 15;
            if (format == FrameFormat::b8g8r8a8_unorm) {
                buf[idx + 0] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
                buf[idx + 3] = 255;
            } else {
                buf[idx + 0] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
                buf[idx + 3] = 255;
            }
        }
    }
}

void generate_noise_frame(std::vector<std::uint8_t>& buf,
                          std::uint32_t width,
                          std::uint32_t height,
                          std::uint32_t seed,
                          FrameFormat format) {
    buf.resize(static_cast<std::size_t>(width) * height * 4);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
        auto val = static_cast<std::uint8_t>(dist(rng));
        if (format == FrameFormat::b8g8r8a8_unorm) {
            buf[i * 4 + 0] = val;
            buf[i * 4 + 1] = static_cast<std::uint8_t>(dist(rng));
            buf[i * 4 + 2] = static_cast<std::uint8_t>(dist(rng));
            buf[i * 4 + 3] = 255;
        } else {
            buf[i * 4 + 0] = val;
            buf[i * 4 + 1] = static_cast<std::uint8_t>(dist(rng));
            buf[i * 4 + 2] = static_cast<std::uint8_t>(dist(rng));
            buf[i * 4 + 3] = 255;
        }
    }
}

// -----------------------------------------------------------------------------
// Test 1: Half Float IEEE-754 Conversion Parity
// -----------------------------------------------------------------------------
void test_half_float_conversions() {
    std::cout << "[Test 1] Testing HalfFloat IEEE-754 conversions..." << std::endl;

    TEST_ASSERT(HalfFloat(0.0f).to_float() == 0.0f);
    TEST_ASSERT(HalfFloat(1.0f).to_float() == 1.0f);
    TEST_ASSERT(HalfFloat(-1.0f).to_float() == -1.0f);

    float pad_norm = 114.0f / 255.0f;
    HalfFloat pad_half(pad_norm);
    float pad_rec = pad_half.to_float();
    float diff = std::abs(pad_norm - pad_rec);
    TEST_ASSERT(diff < 0.0001f);
    TEST_ASSERT(pad_half.bits == 0x3727);

    // Verify monotonicity across [0, 1]
    for (int i = 0; i < 256; ++i) {
        float f = static_cast<float>(i) / 255.0f;
        HalfFloat h(f);
        float f_back = h.to_float();
        TEST_ASSERT(std::abs(f - f_back) <= 0.001f);
    }
    std::cout << "  -> HalfFloat passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: Affine Transformation 2D Mathematical & Subpixel Parity
// -----------------------------------------------------------------------------
void test_affine_transform_2d() {
    std::cout << "[Test 2] Testing AffineTransform2D math & subpixel round-trips..." << std::endl;

    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);
    TEST_ASSERT(std::abs(tf.sx - (1.0f / 3.0f)) < 1e-6f);
    TEST_ASSERT(std::abs(tf.sy - (1.0f / 3.0f)) < 1e-6f);
    TEST_ASSERT(tf.padx == 0.0f);
    TEST_ASSERT(tf.pady == 12.0f);

    // Test matrix elements
    auto m_fwd = tf.forward_matrix_2x3();
    TEST_ASSERT(m_fwd[0] == tf.sx && m_fwd[2] == 0.0f && m_fwd[4] == tf.sy && m_fwd[5] == 12.0f);

    auto m_inv = tf.inverse_matrix_2x3();
    TEST_ASSERT(std::abs(m_inv[0] - 3.0f) < 1e-6f);
    TEST_ASSERT(std::abs(m_inv[4] - 3.0f) < 1e-6f);
    TEST_ASSERT(std::abs(m_inv[5] - (-36.0f)) < 1e-5f);

    // Point forward & inverse round trips
    std::vector<PixelPoint> test_points = {
        {0.0f, 0.0f},
        {960.0f, 540.0f},
        {1920.0f, 1080.0f},
        {123.456f, 789.123f},
        {0.5f, 0.5f},
        {1919.5f, 1079.5f}
    };

    for (const auto& pt : test_points) {
        PixelPoint pt_fwd = tf.forward_point(pt);
        PixelPoint pt_inv = tf.inverse_point(pt_fwd);
        TEST_ASSERT(std::abs(pt.x - pt_inv.x) < 0.001f);
        TEST_ASSERT(std::abs(pt.y - pt_inv.y) < 0.001f);
    }

    // Bounding box round trip
    BoundingBox box{100.25f, 200.5f, 500.75f, 600.25f};
    BoundingBox box_fwd = tf.forward_box(box);
    BoundingBox box_inv = tf.inverse_box(box_fwd);
    TEST_ASSERT(std::abs(box.left - box_inv.left) < 0.001f);
    TEST_ASSERT(std::abs(box.top - box_inv.top) < 0.001f);
    TEST_ASSERT(std::abs(box.right - box_inv.right) < 0.001f);
    TEST_ASSERT(std::abs(box.bottom - box_inv.bottom) < 0.001f);

    // Velocity vector round trip
    PixelVelocity vel{300.0f, -150.0f};
    PixelVelocity vel_fwd = tf.forward_velocity(vel);
    PixelVelocity vel_inv = tf.inverse_velocity(vel_fwd);
    TEST_ASSERT(std::abs(vel.x_per_s - vel_inv.x_per_s) < 0.001f);
    TEST_ASSERT(std::abs(vel.y_per_s - vel_inv.y_per_s) < 0.001f);

    // Covariance matrix round trip
    Covariance2D cov{9.0f, 1.5f, 4.0f};
    Covariance2D cov_fwd = tf.forward_covariance(cov);
    Covariance2D cov_inv = tf.inverse_covariance(cov_fwd);
    TEST_ASSERT(std::abs(cov.xx - cov_inv.xx) < 0.001f);
    TEST_ASSERT(std::abs(cov.xy - cov_inv.xy) < 0.001f);
    TEST_ASSERT(std::abs(cov.yy - cov_inv.yy) < 0.001f);

    std::cout << "  -> AffineTransform2D passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 3: Numerical Parity against CPU Golden Reference
// -----------------------------------------------------------------------------
void test_numerical_parity() {
    std::cout << "[Test 3] Testing numerical parity across diverse image patterns..." << std::endl;

    PreprocessConfig config{};
    config.source_width_px = 1920;
    config.source_height_px = 1080;
    config.target_width_px = 640;
    config.target_height_px = 384;
    config.scaled_width_px = 640;
    config.scaled_height_px = 360;
    config.pad_x = 0.0f;
    config.pad_y = 12.0f;

    CpuReferencePreprocessor cpu_ref(config);
    MockGpuPreprocessor mock_gpu(4);
    auto mock_backend = std::make_shared<capture::MockCudaInteropBackend>();
    mock_gpu.initialize(config, mock_backend);

    std::vector<std::uint8_t> frame_buf;
    std::vector<std::uint16_t> golden_tensor(config.element_count());

    auto run_check = [&](const char* name, FrameFormat fmt) {
        PreprocessedTensorDescriptor golden_desc{};
        CorrelationId corr{100, get_current_time_ns(), 1, 0};
        bool ok_cpu = cpu_ref.preprocess(
            frame_buf.data(), 1920, 1080, 1920 * (fmt == FrameFormat::nv12 ? 1 : 4), fmt,
            100, corr, get_current_time_ns(), golden_tensor.data(), golden_desc
        );
        TEST_ASSERT(ok_cpu);

        PreprocessedTensorLease gpu_lease{};
        bool ok_gpu = mock_gpu.preprocess_raw_memory(
            frame_buf.data(), 1920, 1080, 1920 * (fmt == FrameFormat::nv12 ? 1 : 4), fmt,
            100, corr, get_current_time_ns(), gpu_lease
        );
        TEST_ASSERT(ok_gpu);
        TEST_ASSERT(gpu_lease.is_valid());

        const auto* actual_data = static_cast<const std::uint16_t*>(gpu_lease.cpu_tensor_ptr());
        double mae = CpuReferencePreprocessor::compute_mae(actual_data, golden_tensor.data(), config.element_count());
        float max_diff = CpuReferencePreprocessor::compute_max_diff(actual_data, golden_tensor.data(), config.element_count());

        std::cout << "  Pattern [" << name << "]: MAE = " << mae << ", MaxDiff = " << max_diff << std::endl;
        TEST_ASSERT(mae <= 1e-3);
        TEST_ASSERT(max_diff <= 2e-3f);

        // Verify letterbox padding value in output (12 px top/bottom = 0x3727)
        std::uint16_t expected_pad = HalfFloat(114.0f / 255.0f).bits;
        for (std::uint32_t c = 0; c < 3; ++c) {
            const std::size_t plane_offset = c * (640 * 384);
            // Top pad (y < 12)
            for (std::uint32_t y = 0; y < 12; ++y) {
                for (std::uint32_t x = 0; x < 640; ++x) {
                    TEST_ASSERT(actual_data[plane_offset + y * 640 + x] == expected_pad);
                }
            }
            // Bottom pad (y >= 372)
            for (std::uint32_t y = 372; y < 384; ++y) {
                for (std::uint32_t x = 0; x < 640; ++x) {
                    TEST_ASSERT(actual_data[plane_offset + y * 640 + x] == expected_pad);
                }
            }
        }
    };

    // 1. Uniform Red BGRA
    generate_uniform_frame(frame_buf, 1920, 1080, 255, 0, 0, FrameFormat::b8g8r8a8_unorm);
    run_check("Uniform Red BGRA", FrameFormat::b8g8r8a8_unorm);

    // 2. Uniform Green BGRA
    generate_uniform_frame(frame_buf, 1920, 1080, 0, 255, 0, FrameFormat::b8g8r8a8_unorm);
    run_check("Uniform Green BGRA", FrameFormat::b8g8r8a8_unorm);

    // 3. Uniform Blue BGRA
    generate_uniform_frame(frame_buf, 1920, 1080, 0, 0, 255, FrameFormat::b8g8r8a8_unorm);
    run_check("Uniform Blue BGRA", FrameFormat::b8g8r8a8_unorm);

    // 4. Gradient BGRA
    generate_gradient_frame(frame_buf, 1920, 1080, FrameFormat::b8g8r8a8_unorm);
    run_check("Color Gradient BGRA", FrameFormat::b8g8r8a8_unorm);

    // 5. Fine Checkerboard 2x2
    generate_checkerboard_frame(frame_buf, 1920, 1080, 2, FrameFormat::b8g8r8a8_unorm);
    run_check("Checkerboard 2x2 BGRA", FrameFormat::b8g8r8a8_unorm);

    // 6. Noise BGRA
    generate_noise_frame(frame_buf, 1920, 1080, 0xA11CE, FrameFormat::b8g8r8a8_unorm);
    run_check("Random Noise BGRA", FrameFormat::b8g8r8a8_unorm);

    // 7. NV12 Uniform
    generate_uniform_frame(frame_buf, 1920, 1080, 128, 128, 128, FrameFormat::nv12);
    run_check("Uniform Grey NV12", FrameFormat::nv12);

    std::cout << "  -> Numerical parity passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4: Software reference soak (10,000 frames)
// -----------------------------------------------------------------------------
void test_software_reference_soak() {
    std::cout << "[Test 4] Testing software reference stability across 10,000 frames..." << std::endl;

    PreprocessConfig config{};
    MockGpuPreprocessor preprocessor(4);
    auto mock_backend = std::make_shared<capture::MockCudaInteropBackend>();
    bool init_ok = preprocessor.initialize(config, mock_backend);
    TEST_ASSERT(init_ok);

    // Warm up
    bool warmup_ok = preprocessor.warmup(20);
    TEST_ASSERT(warmup_ok);

    std::vector<std::uint8_t> frame_buf;
    generate_gradient_frame(frame_buf, 1920, 1080, FrameFormat::b8g8r8a8_unorm);

    CorrelationId corr{0, 0, 1, 0};
#if defined(NDEBUG)
    const std::size_t kIterations = 10000;
#else
    const std::size_t kIterations = 500;
#endif

    auto t_start = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < kIterations; ++i) {
        PreprocessedTensorLease lease{};
        corr.sequence_id = i + 1;
        corr.source_timestamp_ns = get_current_time_ns();

        bool ok = preprocessor.preprocess_raw_memory(
            frame_buf.data(), 1920, 1080, 1920 * 4, FrameFormat::b8g8r8a8_unorm,
            i + 1, corr, corr.source_timestamp_ns, lease
        );

        TEST_ASSERT(ok);
        TEST_ASSERT(lease.is_valid());
        TEST_ASSERT(lease.frame_id() == i + 1);
        TEST_ASSERT(lease.descriptor().affine_transform.pady == 12.0f);

        // Auto-released on loop iteration
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double per_frame_ms = total_ms / static_cast<double>(kIterations);

    auto h = preprocessor.health();
    std::cout << "  Processed " << h.total_frames_preprocessed << " frames in "
              << total_ms << " ms (" << per_frame_ms << " ms/frame average)." << std::endl;
    std::cout << "  Avg latency: " << h.avg_latency_us << " us, Max latency: " << h.max_latency_us << " us." << std::endl;

    TEST_ASSERT(h.total_frames_preprocessed == kIterations);
    TEST_ASSERT(h.total_frames_dropped == 0);
    TEST_ASSERT(preprocessor.available_slots() == 4);

    TEST_ASSERT(!h.is_hardware_accelerated);
    std::cout << "  -> Software reference soak passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 5: RAII Lease Lifecycles, Move Semantics & Pool Exhaustion
// -----------------------------------------------------------------------------
void test_lease_lifecycles_and_pool_exhaustion() {
    std::cout << "[Test 5] Testing RAII lease lifecycle and pool exhaustion..." << std::endl;

    PreprocessConfig config{};
    MockGpuPreprocessor preprocessor(2); // Pool capacity 2
    auto mock_backend = std::make_shared<capture::MockCudaInteropBackend>();
    preprocessor.initialize(config, mock_backend);

    std::vector<std::uint8_t> frame_buf;
    generate_gradient_frame(frame_buf, 1920, 1080, FrameFormat::b8g8r8a8_unorm);
    CorrelationId corr{1, 0, 1, 0};

    PreprocessedTensorLease lease1{};
    bool ok1 = preprocessor.preprocess_raw_memory(frame_buf.data(), 1920, 1080, 1920 * 4,
                                                 FrameFormat::b8g8r8a8_unorm, 1, corr, 1000, lease1);
    TEST_ASSERT(ok1 && lease1.is_valid());
    TEST_ASSERT(preprocessor.available_slots() == 1);

    PreprocessedTensorLease lease2{};
    bool ok2 = preprocessor.preprocess_raw_memory(frame_buf.data(), 1920, 1080, 1920 * 4,
                                                 FrameFormat::b8g8r8a8_unorm, 2, corr, 2000, lease2);
    TEST_ASSERT(ok2 && lease2.is_valid());
    TEST_ASSERT(preprocessor.available_slots() == 0);

    // 3rd acquisition should fail safely (pool exhausted)
    PreprocessedTensorLease lease3{};
    bool ok3 = preprocessor.preprocess_raw_memory(frame_buf.data(), 1920, 1080, 1920 * 4,
                                                 FrameFormat::b8g8r8a8_unorm, 3, corr, 3000, lease3);
    TEST_ASSERT(!ok3);
    TEST_ASSERT(!lease3.is_valid());
    TEST_ASSERT(preprocessor.health().total_frames_dropped == 1);

    // Move semantics test
    PreprocessedTensorLease moved_lease(std::move(lease1));
    TEST_ASSERT(!lease1.is_valid());
    TEST_ASSERT(moved_lease.is_valid());
    TEST_ASSERT(moved_lease.frame_id() == 1);

    // Release moved lease
    moved_lease.reset();
    TEST_ASSERT(!moved_lease.is_valid());
    TEST_ASSERT(preprocessor.available_slots() == 1);

    // Now slot is free
    bool ok4 = preprocessor.preprocess_raw_memory(frame_buf.data(), 1920, 1080, 1920 * 4,
                                                 FrameFormat::b8g8r8a8_unorm, 4, corr, 4000, lease3);
    TEST_ASSERT(ok4 && lease3.is_valid());
    TEST_ASSERT(lease3.frame_id() == 4);

    std::cout << "  -> Lease lifecycles & pool exhaustion passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 6: Fail-closed & Error Handling
// -----------------------------------------------------------------------------
void test_fail_closed_error_handling() {
    std::cout << "[Test 6] Testing fail-closed safety and invalid inputs..." << std::endl;

    GpuPreprocessor uninitialized_prep(4);
    PreprocessedTensorLease lease{};
    std::vector<std::uint8_t> dummy(100, 0);
    CorrelationId corr{};

    // 1. Uninitialized preprocessor
    TEST_ASSERT(!uninitialized_prep.preprocess_raw_memory(dummy.data(), 1920, 1080, 1920 * 4,
                                                    FrameFormat::b8g8r8a8_unorm, 1, corr, 0, lease));
    TEST_ASSERT(!lease.is_valid());

    // 2. Null input pointer
    PreprocessConfig config{};
    GpuPreprocessor prep(4);
    TEST_ASSERT(!prep.initialize(config, nullptr));
    TEST_ASSERT(!prep.preprocess_raw_memory(nullptr, 1920, 1080, 1920 * 4,
                                      FrameFormat::b8g8r8a8_unorm, 1, corr, 0, lease));

    // 3. Zero dimensions
    TEST_ASSERT(!prep.preprocess_raw_memory(dummy.data(), 0, 1080, 0,
                                      FrameFormat::b8g8r8a8_unorm, 1, corr, 0, lease));
    TEST_ASSERT(!prep.preprocess_raw_memory(dummy.data(), 1920, 0, 1920 * 4,
                                      FrameFormat::b8g8r8a8_unorm, 1, corr, 0, lease));

    // 4. Unsupported format
    TEST_ASSERT(!prep.preprocess_raw_memory(dummy.data(), 1920, 1080, 1920 * 4,
                                      FrameFormat::unknown, 1, corr, 0, lease));

    // 5. A production preprocessor never silently falls back to the CPU.
    auto unavailable_backend = std::make_shared<capture::MockCudaInteropBackend>();
    unavailable_backend->set_cuda_available(false);
    TEST_ASSERT(!prep.initialize(config, unavailable_backend));
    TEST_ASSERT(!prep.health().is_hardware_accelerated);

    std::cout << "  -> Fail-closed error handling passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running OpenPrism M2-04 Preprocessing Test Suite   " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_half_float_conversions();
    test_affine_transform_2d();
    test_numerical_parity();
    test_software_reference_soak();
    test_lease_lifecycles_and_pool_exhaustion();
    test_fail_closed_error_handling();

    std::cout << "====================================================" << std::endl;
    std::cout << " All M2-04 Preprocessing Tests Passed Successfully! " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
