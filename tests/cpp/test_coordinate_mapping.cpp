// tests/cpp/test_coordinate_mapping.cpp
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "aim/core/types.hpp"
#include "aim/perception/coordinate_transform.hpp"

using namespace aim;
using namespace aim::perception;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

// -----------------------------------------------------------------------------
// Test 1: Standard & Non-Standard Letterbox Geometry Configurations
// -----------------------------------------------------------------------------
void test_letterbox_geometries() {
    std::cout << "[Test 1] Testing letterbox geometry calculations across aspect ratios..." << std::endl;

    struct TestCase {
        std::uint32_t src_w;
        std::uint32_t src_h;
        std::uint32_t dst_w;
        std::uint32_t dst_h;
        float exp_scale;
        float exp_padx;
        float exp_pady;
    };

    std::vector<TestCase> cases = {
        // 16:9 Standard
        {1920, 1080, 640, 384, 1.0f / 3.0f, 0.0f, 12.0f},
        {2560, 1440, 640, 384, 0.25f, 0.0f, 12.0f},
        {3840, 2160, 640, 384, 1.0f / 6.0f, 0.0f, 12.0f},
        {1280, 720, 640, 384, 0.5f, 0.0f, 12.0f},

        // 16:10 Aspect Ratio -> Horizontal letterbox columns (pillarbox)
        {1920, 1200, 640, 384, 0.32f, 12.8f, 0.0f},

        // 21:9 Ultrawide Aspect Ratio -> Vertical letterbox bars
        {3440, 1440, 640, 384, 640.0f / 3440.0f, 0.0f, (384.0f - 1440.0f * (640.0f / 3440.0f)) * 0.5f},

        // 4:3 Aspect Ratio -> Horizontal pillarbox
        {800, 600, 640, 384, 0.64f, 64.0f, 0.0f},

        // 1:1 Aspect Ratio -> Horizontal pillarbox
        {1000, 1000, 640, 384, 0.384f, 128.0f, 0.0f},

        // Degenerate inputs fail closed with identity
        {0, 1080, 640, 384, 1.0f, 0.0f, 0.0f},
        {1920, 0, 640, 384, 1.0f, 0.0f, 0.0f},
        {1920, 1080, 0, 384, 1.0f, 0.0f, 0.0f},
        {1920, 1080, 640, 0, 1.0f, 0.0f, 0.0f},
    };

    for (const auto& c : cases) {
        auto tf = AffineTransform2D::create_letterbox(c.src_w, c.src_h, c.dst_w, c.dst_h);
        TEST_ASSERT(std::abs(tf.sx - c.exp_scale) < 1e-4f);
        TEST_ASSERT(std::abs(tf.sy - c.exp_scale) < 1e-4f);
        TEST_ASSERT(std::abs(tf.padx - c.exp_padx) < 1e-4f);
        TEST_ASSERT(std::abs(tf.pady - c.exp_pady) < 1e-4f);
    }

    std::cout << "  -> Letterbox geometries passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: Subpixel Round-Trip Accuracy
// -----------------------------------------------------------------------------
void test_subpixel_roundtrips() {
    std::cout << "[Test 2] Testing subpixel round-trip accuracy (error <= 0.001 px)..." << std::endl;

    std::vector<std::pair<std::uint32_t, std::uint32_t>> resolutions = {
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {1280, 720},
        {1920, 1200},
        {3440, 1440},
        {800, 600},
        {1000, 1000},
    };

    for (const auto& [w, h] : resolutions) {
        auto tf = AffineTransform2D::create_letterbox(w, h, 640, 384);

        std::vector<PixelPoint> pts = {
            {0.0f, 0.0f},
            {static_cast<float>(w), static_cast<float>(h)},
            {static_cast<float>(w) * 0.5f, static_cast<float>(h) * 0.5f},
            {0.12345f, 0.6789f},
            {static_cast<float>(w) - 0.5f, static_cast<float>(h) - 0.5f},
            {123.4567f, 789.0123f},
            {-50.0f, -50.0f}, // Negative coordinates outside frame
            {static_cast<float>(w) + 100.0f, static_cast<float>(h) + 100.0f}, // Out of bounds
        };

        for (const auto& pt : pts) {
            PixelPoint fwd = tf.forward_point(pt);
            PixelPoint inv = tf.inverse_point(fwd);
            TEST_ASSERT(std::abs(pt.x - inv.x) < 0.001f);
            TEST_ASSERT(std::abs(pt.y - inv.y) < 0.001f);
        }
    }

    std::cout << "  -> Subpixel roundtrips passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 3: Bounding Box, Extent, and Center Round-Trips
// -----------------------------------------------------------------------------
void test_bounding_boxes_and_centers() {
    std::cout << "[Test 3] Testing bounding box transformations and center preservation..." << std::endl;

    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);

    std::vector<BoundingBox> boxes = {
        {100.25f, 200.5f, 300.75f, 400.25f},
        {0.0f, 0.0f, 1920.0f, 1080.0f},
        {950.123f, 530.456f, 969.876f, 549.543f},
        {0.5f, 0.5f, 1.5f, 1.5f},
    };

    for (const auto& box : boxes) {
        BoundingBox box_fwd = tf.forward_box(box);
        BoundingBox box_inv = tf.inverse_box(box_fwd);

        TEST_ASSERT(std::abs(box.left - box_inv.left) < 0.001f);
        TEST_ASSERT(std::abs(box.top - box_inv.top) < 0.001f);
        TEST_ASSERT(std::abs(box.right - box_inv.right) < 0.001f);
        TEST_ASSERT(std::abs(box.bottom - box_inv.bottom) < 0.001f);

        // Center round-trip
        PixelPoint c_orig = CoordinateConverter::box_center(box);
        PixelPoint c_fwd = CoordinateConverter::box_center(box_fwd);
        PixelPoint c_inv = tf.inverse_point(c_fwd);
        TEST_ASSERT(std::abs(c_orig.x - c_inv.x) < 0.001f);
        TEST_ASSERT(std::abs(c_orig.y - c_inv.y) < 0.001f);

        // Extent scaling
        PixelPoint ext_orig = CoordinateConverter::box_extent(box);
        PixelPoint ext_fwd = CoordinateConverter::box_extent(box_fwd);
        TEST_ASSERT(std::abs(ext_orig.x * tf.sx - ext_fwd.x) < 0.001f);
        TEST_ASSERT(std::abs(ext_orig.y * tf.sy - ext_fwd.y) < 0.001f);
    }

    std::cout << "  -> Bounding boxes and centers passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4: Covariance 2D Matrix Scaling & Inversion (Sigma' = A * Sigma * A^T)
// -----------------------------------------------------------------------------
void test_covariance_scaling() {
    std::cout << "[Test 4] Testing covariance matrix scaling and analytical invertibility..." << std::endl;

    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);
    Covariance2D cov{16.0f, 2.5f, 9.0f};

    Covariance2D cov_fwd = tf.forward_covariance(cov);
    Covariance2D cov_inv = tf.inverse_covariance(cov_fwd);

    // Forward: scale^2 = (1/3)^2 = 1/9
    TEST_ASSERT(std::abs(cov_fwd.xx - (16.0f / 9.0f)) < 1e-5f);
    TEST_ASSERT(std::abs(cov_fwd.xy - (2.5f / 9.0f)) < 1e-5f);
    TEST_ASSERT(std::abs(cov_fwd.yy - (9.0f / 9.0f)) < 1e-5f);

    // Inverted back to source
    TEST_ASSERT(std::abs(cov.xx - cov_inv.xx) < 0.001f);
    TEST_ASSERT(std::abs(cov.xy - cov_inv.xy) < 0.001f);
    TEST_ASSERT(std::abs(cov.yy - cov_inv.yy) < 0.001f);

    std::cout << "  -> Covariance scaling passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 5: Kinematics (Velocity & Acceleration)
// -----------------------------------------------------------------------------
void test_kinematics() {
    std::cout << "[Test 5] Testing velocity and acceleration transformations..." << std::endl;

    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);

    PixelVelocity vel{300.0f, -150.0f};
    PixelVelocity vel_fwd = tf.forward_velocity(vel);
    PixelVelocity vel_inv = tf.inverse_velocity(vel_fwd);

    TEST_ASSERT(std::abs(vel_fwd.x_per_s - 100.0f) < 1e-5f);
    TEST_ASSERT(std::abs(vel_fwd.y_per_s - (-50.0f)) < 1e-5f);
    TEST_ASSERT(std::abs(vel.x_per_s - vel_inv.x_per_s) < 0.001f);
    TEST_ASSERT(std::abs(vel.y_per_s - vel_inv.y_per_s) < 0.001f);

    PixelAcceleration acc{-60.0f, 120.0f};
    PixelAcceleration acc_fwd = tf.forward_acceleration(acc);
    PixelAcceleration acc_inv = tf.inverse_acceleration(acc_fwd);

    TEST_ASSERT(std::abs(acc_fwd.x_per_s2 - (-20.0f)) < 1e-5f);
    TEST_ASSERT(std::abs(acc_fwd.y_per_s2 - 40.0f) < 1e-5f);
    TEST_ASSERT(std::abs(acc.x_per_s2 - acc_inv.x_per_s2) < 0.001f);
    TEST_ASSERT(std::abs(acc.y_per_s2 - acc_inv.y_per_s2) < 0.001f);

    // Effective radius scaling
    float r = 24.0f;
    float r_fwd = tf.forward_radius(r);
    float r_inv = tf.inverse_radius(r_fwd);
    TEST_ASSERT(std::abs(r_fwd - 8.0f) < 1e-5f);
    TEST_ASSERT(std::abs(r - r_inv) < 0.001f);

    std::cout << "  -> Kinematics passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 6: Normalized Coordinate Space Converter ([-1, 1])
// -----------------------------------------------------------------------------
void test_normalized_space_conversions() {
    std::cout << "[Test 6] Testing [-1, 1] normalized screen space conversions..." << std::endl;

    const float w = 1920.0f;
    const float h = 1080.0f;
    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);

    // Center -> (0, 0)
    NormalizedPoint norm_c = CoordinateConverter::pixel_to_normalized({960.0f, 540.0f}, w, h);
    TEST_ASSERT(std::abs(norm_c.x) < 1e-6f && std::abs(norm_c.y) < 1e-6f);

    // Top-Left -> (-1, -1)
    NormalizedPoint norm_tl = CoordinateConverter::pixel_to_normalized({0.0f, 0.0f}, w, h);
    TEST_ASSERT(std::abs(norm_tl.x - (-1.0f)) < 1e-6f && std::abs(norm_tl.y - (-1.0f)) < 1e-6f);

    // Bottom-Right -> (+1, +1)
    NormalizedPoint norm_br = CoordinateConverter::pixel_to_normalized({1920.0f, 1080.0f}, w, h);
    TEST_ASSERT(std::abs(norm_br.x - 1.0f) < 1e-6f && std::abs(norm_br.y - 1.0f) < 1e-6f);

    // Model tensor to normalized screen space round trips
    std::vector<PixelPoint> model_pts = {
        {320.0f, 192.0f}, // Model center
        {0.0f, 12.0f},     // Content top-left
        {640.0f, 372.0f},  // Content bottom-right
        {100.5f, 200.75f}, // Arbitrary point
    };

    for (const auto& mpt : model_pts) {
        NormalizedPoint norm_pt = CoordinateConverter::model_to_normalized_screen(mpt, tf, w, h);
        PixelPoint mpt_back = CoordinateConverter::normalized_screen_to_model(norm_pt, tf, w, h);
        TEST_ASSERT(std::abs(mpt.x - mpt_back.x) < 0.001f);
        TEST_ASSERT(std::abs(mpt.y - mpt_back.y) < 0.001f);
    }

    std::cout << "  -> Normalized space conversions passed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 7: Matrix Inverse Identity (M_inv * M_fwd == I)
// -----------------------------------------------------------------------------
void test_matrix_inverses() {
    std::cout << "[Test 7] Testing 3x3 and 2x3 matrix properties..." << std::endl;

    auto tf = AffineTransform2D::create_letterbox(1920, 1080, 640, 384);
    auto m_fwd = tf.forward_matrix_3x3();
    auto m_inv = tf.inverse_matrix_3x3();

    // Multiply M_inv * M_fwd (3x3 row major)
    std::array<float, 9> res{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < 3; ++k) {
                sum += m_inv[r * 3 + k] * m_fwd[k * 3 + c];
            }
            res[r * 3 + c] = sum;
        }
    }

    // Check identity
    TEST_ASSERT(std::abs(res[0] - 1.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[1] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[2] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[3] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[4] - 1.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[5] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[6] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[7] - 0.0f) < 1e-6f);
    TEST_ASSERT(std::abs(res[8] - 1.0f) < 1e-6f);

    std::cout << "  -> Matrix inverses passed." << std::endl;
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running Coordinate Mapping Verification Suite      " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_letterbox_geometries();
    test_subpixel_roundtrips();
    test_bounding_boxes_and_centers();
    test_covariance_scaling();
    test_kinematics();
    test_normalized_space_conversions();
    test_matrix_inverses();

    std::cout << "====================================================" << std::endl;
    std::cout << " All Coordinate Mapping Tests Passed Successfully!  " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
