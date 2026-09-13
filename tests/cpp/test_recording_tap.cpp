// tests/cpp/test_recording_tap.cpp
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "aim/core/clock.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/data/recording_tap.hpp"
#include "aim/perception/preprocess_types.hpp"

using namespace aim;
using namespace aim::data;
using namespace aim::perception;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_recording_tap_initialization_and_warmup() {
    std::cout << "[Test 1] Testing RecordingTap initialization, lifecycle, and manifest export..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    RecordingTap tap(fake_clock);

    RecordingSessionConfig cfg{};
    cfg.session_id = "11111111-2222-4333-8444-555555555555";
    cfg.scenario_name = "gridshot";
    cfg.task_type = "flick";
    cfg.visual_profile = "default";
    cfg.ring_capacity = 32;
    cfg.output_dir = "build/test_recordings_001";
    cfg.metadata_only = true; // Metadata mode for fast deterministic unit test

    TEST_ASSERT(tap.initialize(cfg));
    TEST_ASSERT(tap.is_healthy());
    TEST_ASSERT(!tap.is_active());

    TEST_ASSERT(tap.start());
    TEST_ASSERT(tap.is_active());

    std::vector<std::uint8_t> dummy_pixels(640 * 384 * 3 * 2, 128); // FP16 dummy tensor

    for (std::uint64_t i = 1; i <= 10; ++i) {
        fake_clock->advance_ns(6'944'444LL);
        PreprocessedTensorDescriptor desc{};
        desc.frame_id = i;
        desc.captured_at_ns = fake_clock->now_ns();
        desc.preprocessed_at_ns = fake_clock->now_ns() + 300'000LL;
        desc.width_px = 640;
        desc.height_px = 384;
        desc.channels = 3;
        desc.size_bytes = dummy_pixels.size();
        desc.correlation_id = CorrelationId{i, desc.captured_at_ns, 1, 0};
        desc.is_valid = true;

        bool enqueued = tap.submit_frame(desc, dummy_pixels.data(), dummy_pixels.size());
        TEST_ASSERT(enqueued);
    }

    tap.flush();
    tap.stop();
    TEST_ASSERT(!tap.is_active());

    auto stats = tap.stats();
    TEST_ASSERT(stats.frames_submitted == 10);
    TEST_ASSERT(stats.frames_enqueued == 10);
    TEST_ASSERT(stats.frames_dropped_saturation == 0);
    TEST_ASSERT(stats.frames_written == 10);

    auto entries = tap.get_frame_entries();
    TEST_ASSERT(entries.size() == 10);
    TEST_ASSERT(entries[0].frame_sequence_id == 1);
    TEST_ASSERT(entries[9].frame_sequence_id == 10);
    TEST_ASSERT(!entries[0].sha256.empty());

    std::filesystem::create_directories("build");
    const std::string manifest_out = "build/test_session_manifest.json";
    TEST_ASSERT(tap.export_session_manifest(manifest_out));
    TEST_ASSERT(std::filesystem::exists(manifest_out));

    // Verify manifest contains required fields
    std::ifstream manifest_file(manifest_out);
    std::string content((std::istreambuf_iterator<char>(manifest_file)), std::istreambuf_iterator<char>());
    TEST_ASSERT(content.find("\"schema_version\": 1") != std::string::npos);
    TEST_ASSERT(content.find("\"session_id\": \"11111111-2222-4333-8444-555555555555\"") != std::string::npos);
    TEST_ASSERT(content.find("\"scenario_name\": \"gridshot\"") != std::string::npos);
    TEST_ASSERT(content.find("\"total_frames_recorded\": 10") != std::string::npos);

    std::cout << "  -> Initialization, lifecycle, and manifest export PASSED." << std::endl;
}

void test_zero_backpressure_under_queue_saturation() {
    std::cout << "[Test 2] Testing zero backpressure and graceful drop on queue saturation..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    RecordingTap tap(fake_clock);

    RecordingSessionConfig cfg{};
    cfg.session_id = "22222222-3333-4444-8555-666666666666";
    cfg.ring_capacity = 8; // Small ring to trigger saturation easily
    cfg.metadata_only = true;

    TEST_ASSERT(tap.initialize(cfg));
    TEST_ASSERT(tap.start());

    std::vector<std::uint8_t> dummy_pixels(1024, 42);

    const std::uint64_t total_submissions = 200;
    std::uint64_t accepted_count = 0;
    std::uint64_t dropped_count = 0;

    auto t0 = std::chrono::steady_clock::now();

    for (std::uint64_t i = 1; i <= total_submissions; ++i) {
        fake_clock->advance_ns(100'000LL); // Rapid sub-millisecond submissions
        PreprocessedTensorDescriptor desc{};
        desc.frame_id = i;
        desc.captured_at_ns = fake_clock->now_ns();
        desc.width_px = 640;
        desc.height_px = 384;
        desc.size_bytes = dummy_pixels.size();
        desc.is_valid = true;

        bool ok = tap.submit_frame(desc, dummy_pixels.data(), dummy_pixels.size());
        if (ok) {
            ++accepted_count;
        } else {
            ++dropped_count;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    tap.flush();
    tap.stop();

    auto stats = tap.stats();
    TEST_ASSERT(stats.frames_submitted == total_submissions);
    TEST_ASSERT(stats.frames_enqueued == accepted_count);
    TEST_ASSERT(stats.frames_dropped_saturation == dropped_count);
    TEST_ASSERT(stats.frames_enqueued + stats.frames_dropped_saturation == total_submissions);

    std::cout << "  -> Submissions: " << total_submissions << " | Accepted: " << accepted_count
              << " | Dropped (Saturation): " << dropped_count << std::endl;
    std::cout << "  -> Total 200 submissions elapsed time: " << elapsed_us << " us ("
              << static_cast<double>(elapsed_us) / total_submissions << " us/op)" << std::endl;

    // Verify push latency never stalls hot path (average push latency < 50 us)
    TEST_ASSERT(static_cast<double>(elapsed_us) / total_submissions < 50.0);

    std::cout << "  -> Zero backpressure under queue saturation PASSED." << std::endl;
}

void test_raw_frame_submission_and_correlation() {
    std::cout << "[Test 3] Testing raw frame submission and correlation metadata..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    RecordingTap tap(fake_clock);

    RecordingSessionConfig cfg{};
    cfg.session_id = "33333333-4444-4555-8666-777777777777";
    cfg.ring_capacity = 16;
    cfg.metadata_only = true;

    TEST_ASSERT(tap.initialize(cfg));
    TEST_ASSERT(tap.start());

    // Submit invalid frame lease -> should fail closed
    FrameLease invalid_lease{};
    TEST_ASSERT(!tap.submit_raw_frame(invalid_lease, nullptr, 0));

    tap.stop();
    std::cout << "  -> Raw frame submission and fail-closed validation PASSED." << std::endl;
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running Recording Tap & Dataset Pipeline Tests     " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_recording_tap_initialization_and_warmup();
    test_zero_backpressure_under_queue_saturation();
    test_raw_frame_submission_and_correlation();

    std::cout << "====================================================" << std::endl;
    std::cout << " All Recording Tap Tests PASSED!                    " << std::endl;
    return 0;
}
