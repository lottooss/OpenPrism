// tests/cpp/test_yolo_decoder.cpp
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <limits>
#include <new>

#include "aim/bus/types.hpp"
#include "aim/perception/yolo_decoder.hpp"

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace allocation_probe {
inline bool enabled = false;
inline std::size_t count = 0;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t size) {
    if (allocation_probe::enabled) { ++allocation_probe::count; }
    if (void* result = std::malloc(size == 0 ? 1 : size)) { return result; }
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

using namespace aim;
using namespace aim::perception;

void test_yolo_decoder_golden_box() {
    YoloDecoderConfig cfg{};
    cfg.num_anchors = 3;
    cfg.confidence_threshold = 0.50f;
    cfg.nms_iou_threshold = 0.45f;
    // Model resolution 640x384, scale 1/3, pad y = 12.0
    cfg.transform = AffineTransform2D{1.0f / 3.0f, 1.0f / 3.0f, 0.0f, 12.0f};

    YoloDecoder decoder(cfg);

    // 3 anchors:
    // Anchor 0: cx=320, cy=192, w=20, h=20, conf=0.90 -> Maps to Source (960, 540)
    // Anchor 1: duplicate of Anchor 0 with conf=0.80 (should be suppressed by NMS)
    // Anchor 2: cx=100, cy=100, w=10, h=10, conf=0.10 (should be rejected by confidence threshold)
    std::vector<float> tensor_data(5 * 3);
    // cx
    tensor_data[0] = 320.0f; tensor_data[1] = 320.0f; tensor_data[2] = 100.0f;
    // cy
    tensor_data[3] = 192.0f; tensor_data[4] = 192.0f; tensor_data[5] = 100.0f;
    // w
    tensor_data[6] = 20.0f;  tensor_data[7] = 20.0f;  tensor_data[8] = 10.0f;
    // h
    tensor_data[9] = 20.0f;  tensor_data[10] = 20.0f; tensor_data[11] = 10.0f;
    // conf
    tensor_data[12] = 0.90f; tensor_data[13] = 0.80f; tensor_data[14] = 0.10f;

    CorrelationId cid{};
    cid.sequence_id = 100;
    cid.source_timestamp_ns = 1'500'000'000;

    bus::TargetObservationBatch batch{};
    decoder.decode(tensor_data.data(), tensor_data.size(), cid, 1'500'000'000, batch);

    TEST_ASSERT(batch.header.sequence_id == 100);
    TEST_ASSERT(batch.captured_at_ns == 1'500'000'000);
    TEST_ASSERT(batch.target_count == 1); // 1 kept (anchor 1 suppressed, anchor 2 thresholded)

    const auto& target = batch.targets[0];
    TEST_ASSERT(std::abs(target.center_px.x - 960.0f) < 1e-3f);
    TEST_ASSERT(std::abs(target.center_px.y - 540.0f) < 1e-3f);
    TEST_ASSERT(target.confidence == 0.90f);
    TEST_ASSERT(std::abs(target.effective_radius_px - 30.0f) < 1e-3f);

    std::cout << "[PASS] test_yolo_decoder_golden_box\n";
}

void test_yolo_decoder_overflow_bounded() {
    YoloDecoderConfig cfg{};
    cfg.num_anchors = 100;
    cfg.confidence_threshold = 0.10f;
    cfg.nms_iou_threshold = 0.99f; // Allow multiple disjoint boxes

    YoloDecoder decoder(cfg);

    // 100 distinct non-overlapping boxes
    std::vector<float> tensor_data(5 * 100, 0.0f);
    for (std::uint32_t i = 0; i < 100; ++i) {
        tensor_data[i] = static_cast<float>(i * 5);        // cx
        tensor_data[100 + i] = static_cast<float>(i * 3);  // cy
        tensor_data[200 + i] = 2.0f;                       // w
        tensor_data[300 + i] = 2.0f;                       // h
        tensor_data[400 + i] = 0.95f;                      // conf
    }

    CorrelationId cid{1, 1000, 1, 0};
    bus::TargetObservationBatch batch{};
    decoder.decode(tensor_data.data(), tensor_data.size(), cid, 1000, batch);

    TEST_ASSERT(batch.target_count == bus::kMaxObservations);
    TEST_ASSERT(decoder.overflow_count() == 1);

    std::cout << "[PASS] test_yolo_decoder_overflow_bounded\n";
}

void test_dense_first_decode_has_no_allocation() {
    YoloDecoderConfig config{};
    config.max_detections = 2;
    YoloDecoder decoder(config);
    const std::size_t anchors = config.num_anchors;
    std::vector<float> tensor(anchors * 5U);
    for (std::size_t i = 0; i < anchors; ++i) {
        tensor[i] = static_cast<float>(i % 320U) * 2.0F;
        tensor[anchors + i] = static_cast<float>(i % 192U) * 2.0F;
        tensor[2U * anchors + i] = 1.0F;
        tensor[3U * anchors + i] = 1.0F;
        tensor[4U * anchors + i] = 0.95F;
    }
    bus::TargetObservationBatch batch{};
    allocation_probe::count = 0;
    allocation_probe::enabled = true;
    decoder.decode(tensor.data(), tensor.size(), CorrelationId{1, 1000, 1, 0}, 1000, batch);
    allocation_probe::enabled = false;
    TEST_ASSERT(allocation_probe::count == 0);
    TEST_ASSERT(batch.target_count == 2);
}

void test_nonfinite_and_invalid_geometry_are_rejected() {
    YoloDecoderConfig config{};
    config.num_anchors = 4;
    YoloDecoder decoder(config);
    std::vector<float> tensor{
        100, 100, 100, 100,
        100, 100, 100, 100,
        10, -10, 10, 10,
        10, 10, 10, 10,
        std::numeric_limits<float>::quiet_NaN(), 0.9F, 1.1F, 0.9F};
    bus::TargetObservationBatch batch{};
    decoder.decode(tensor.data(), tensor.size(), CorrelationId{1, 1000, 1, 0}, 1000, batch);
    TEST_ASSERT(batch.target_count == 1);
    TEST_ASSERT(std::isfinite(batch.targets[0].confidence));
}

int main() {
    std::cout << "Running YOLO Decoder C++ Unit Tests...\n";
    test_dense_first_decode_has_no_allocation();
    test_nonfinite_and_invalid_geometry_are_rejected();
    test_yolo_decoder_golden_box();
    test_yolo_decoder_overflow_bounded();
    std::cout << "All YOLO Decoder C++ Unit Tests Passed Successfully!\n";
    return 0;
}
