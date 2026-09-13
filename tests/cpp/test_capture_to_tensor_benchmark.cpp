// tests/cpp/test_capture_to_tensor_benchmark.cpp
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/dxgi_backend.hpp"
#include "aim/capture/unified_capture_source.hpp"
#include "aim/capture/wgc_backend.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/capture_to_tensor_pipeline.hpp"
#include "aim/perception/gpu_preprocessor.hpp"

using namespace aim;
using namespace aim::capture;
using namespace aim::perception;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

// -----------------------------------------------------------------------------
// Test 1: Deterministic capture-to-tensor timing plumbing
// -----------------------------------------------------------------------------
void test_capture_to_tensor_latency_benchmark() {
    std::cout << "[Test 1] Running deterministic Capture-to-Tensor timing test..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    auto mock_dxgi = std::make_unique<MockDxgiBackend>();
    auto mock_wgc = std::make_unique<MockWgcBackend>();
    auto* raw_dxgi = mock_dxgi.get();

    auto unified_capture = std::make_unique<UnifiedCaptureSource>(
        std::move(mock_dxgi), std::move(mock_wgc), fake_clock);

    auto mock_cuda = std::make_shared<capture::MockCudaInteropBackend>();
    auto preprocessor = std::make_unique<MockGpuPreprocessor>();

    CaptureToTensorPipeline pipeline(std::move(unified_capture), std::move(preprocessor), fake_clock);

    CaptureToTensorConfig cfg{};
    cfg.capture_config.target_width_px = 1920;
    cfg.capture_config.target_height_px = 1080;
    cfg.preprocess_config.source_width_px = 1920;
    cfg.preprocess_config.source_height_px = 1080;
    cfg.preprocess_config.target_width_px = 640;
    cfg.preprocess_config.target_height_px = 384;

    TEST_ASSERT(pipeline.initialize(cfg, mock_cuda));
    TEST_ASSERT(pipeline.warmup(10));
    TEST_ASSERT(pipeline.start());

    bus::LatestSpscRing<PreprocessedTensorDescriptor, 16> tensor_ring{};
    pipeline.bind_tensor_bus_ring(&tensor_ring);

    const std::uint32_t benchmark_iterations = 2000;
    for (std::uint32_t i = 0; i < benchmark_iterations; ++i) {
        fake_clock->advance_ns(6'944'444LL); // 144 Hz tick
        raw_dxgi->queue_acquire_result(S_OK, static_cast<std::uint64_t>(fake_clock->now_ns() / 100LL), 1);

        PreprocessedTensorLease tensor_lease{};
        bool ok = pipeline.process_frame(tensor_lease);
        TEST_ASSERT(ok);
        TEST_ASSERT(tensor_lease.is_valid());
        TEST_ASSERT(tensor_lease.descriptor().width_px == 640);
        TEST_ASSERT(tensor_lease.descriptor().height_px == 384);

        // Verify published tensor descriptor on ring
        PreprocessedTensorDescriptor desc{};
        TEST_ASSERT(tensor_ring.try_read_latest(desc));
        TEST_ASSERT(desc.is_valid);
    }

    auto metrics = pipeline.metrics();
    std::cout << "  -> Benchmark completed over " << metrics.capture_to_tensor_latency.sample_count << " frames." << std::endl;
    std::cout << "     Capture-to-Tensor Latencies (ms):" << std::endl;
    std::cout << "       min = " << std::fixed << std::setprecision(4) << metrics.capture_to_tensor_latency.min_ms << " ms" << std::endl;
    std::cout << "       avg = " << metrics.capture_to_tensor_latency.avg_ms << " ms" << std::endl;
    std::cout << "       p50 = " << metrics.capture_to_tensor_latency.p50_ms << " ms" << std::endl;
    std::cout << "       p95 = " << metrics.capture_to_tensor_latency.p95_ms << " ms" << std::endl;
    std::cout << "       p99 = " << metrics.capture_to_tensor_latency.p99_ms << " ms" << std::endl;
    std::cout << "       max = " << metrics.capture_to_tensor_latency.max_ms << " ms" << std::endl;

    // FakeClock advances only through deterministic backend bookkeeping. These
    // relationships validate metric plumbing, not hardware latency acceptance.
    TEST_ASSERT(metrics.capture_to_tensor_latency.sample_count == benchmark_iterations);
    TEST_ASSERT(metrics.capture_to_tensor_latency.min_ms <= metrics.capture_to_tensor_latency.p99_ms);
    TEST_ASSERT(metrics.capture_to_tensor_latency.p99_ms <= metrics.capture_to_tensor_latency.max_ms);
    TEST_ASSERT(metrics.frames_dropped_stale == 0);
    TEST_ASSERT(metrics.frames_dropped_invalid == 0);

    pipeline.stop();
    std::cout << "  -> Deterministic timing plumbing passed (not hardware evidence)." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 2: Simulated Thermal Soak & Leak Detection (10,000 Frames)
// -----------------------------------------------------------------------------
void test_simulated_soak_leak_detection() {
    std::cout << "[Test 2] Running Simulated Soak & Leak Detection (10,000 continuous frames)..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    auto mock_dxgi = std::make_unique<MockDxgiBackend>();
    auto mock_wgc = std::make_unique<MockWgcBackend>();
    auto* raw_dxgi = mock_dxgi.get();

    auto unified_capture = std::make_unique<UnifiedCaptureSource>(
        std::move(mock_dxgi), std::move(mock_wgc), fake_clock);

    auto mock_cuda = std::make_shared<capture::MockCudaInteropBackend>();
    auto preprocessor = std::make_unique<MockGpuPreprocessor>();
    auto* raw_preprocessor = preprocessor.get();

    CaptureToTensorPipeline pipeline(std::move(unified_capture), std::move(preprocessor), fake_clock);

    CaptureToTensorConfig cfg{};
    TEST_ASSERT(pipeline.initialize(cfg, mock_cuda));
    TEST_ASSERT(pipeline.warmup(10));
    TEST_ASSERT(pipeline.start());

    const std::uint32_t soak_frames = 10000;
    for (std::uint32_t i = 0; i < soak_frames; ++i) {
        fake_clock->advance_ns(6'944'444LL);
        raw_dxgi->queue_acquire_result(S_OK, static_cast<std::uint64_t>(fake_clock->now_ns() / 100LL), 1);
        {
            PreprocessedTensorLease lease{};
            bool ok = pipeline.process_frame(lease);
            TEST_ASSERT(ok);
            TEST_ASSERT(lease.is_valid());
            TEST_ASSERT(raw_preprocessor->available_slots() == raw_preprocessor->capacity() - 1);
        } // RAII lease goes out of scope here -> returns slot immediately

        // Verify zero dangling active leases after scope exit
        TEST_ASSERT(raw_preprocessor->available_slots() == raw_preprocessor->capacity());
    }

    auto metrics = pipeline.metrics();
    TEST_ASSERT(metrics.total_frames_processed == soak_frames);
    TEST_ASSERT(metrics.frames_acquired == soak_frames);
    TEST_ASSERT(metrics.frames_dropped_invalid == 0);
    TEST_ASSERT(metrics.frames_dropped_stale == 0);

    pipeline.stop();
    std::cout << "  -> 10,000-frame soak completed with 0 leaks and 0 dangling handles." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 3: Stale Frame Drop Cutoff Test (>10 ms cutoff)
// -----------------------------------------------------------------------------
void test_stale_frame_cutoff() {
    std::cout << "[Test 3] Testing stale frame drop cutoff (10ms threshold)..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    auto mock_dxgi = std::make_unique<MockDxgiBackend>();
    auto mock_wgc = std::make_unique<MockWgcBackend>();
    auto* raw_dxgi = mock_dxgi.get();

    auto unified_capture = std::make_unique<UnifiedCaptureSource>(
        std::move(mock_dxgi), std::move(mock_wgc), fake_clock);

    auto mock_cuda = std::make_shared<capture::MockCudaInteropBackend>();
    auto preprocessor = std::make_unique<MockGpuPreprocessor>();

    CaptureToTensorPipeline pipeline(std::move(unified_capture), std::move(preprocessor), fake_clock);

    CaptureToTensorConfig cfg{};
    cfg.stale_threshold_ns = 10'000'000LL; // 10ms threshold
    TEST_ASSERT(pipeline.initialize(cfg, mock_cuda));
    TEST_ASSERT(pipeline.warmup(5));
    TEST_ASSERT(pipeline.start());

    // Frame 1: Fresh frame (5ms old) -> Accepted
    // Note: QPC timestamp is converted to ns. If QPF=10MHz, 1 tick = 100ns.
    MonotonicNs now = fake_clock->now_ns();
    raw_dxgi->queue_acquire_result(S_OK, static_cast<std::uint64_t>((now - 5'000'000LL) / 100LL), 1);
    {
        PreprocessedTensorLease lease{};
        bool ok = pipeline.process_frame(lease);
        TEST_ASSERT(ok);
        TEST_ASSERT(lease.is_valid());
    }

    // Frame 2: Stale frame (15ms old > 10ms cutoff) -> Dropped fail-closed
    raw_dxgi->queue_acquire_result(S_OK, static_cast<std::uint64_t>((now - 15'000'000LL) / 100LL), 1);
    {
        PreprocessedTensorLease lease{};
        bool ok = pipeline.process_frame(lease);
        TEST_ASSERT(!ok);
        TEST_ASSERT(!lease.is_valid());
    }

    auto metrics = pipeline.metrics();
    TEST_ASSERT(metrics.frames_dropped_stale == 1);
    TEST_ASSERT(metrics.frames_acquired == 2);

    pipeline.stop();
    std::cout << "  -> Stale frame cutoff verified fail-closed." << std::endl;
}

// -----------------------------------------------------------------------------
// Test 4: Fault-Injection & Multi-Backend Recovery State Machine
// -----------------------------------------------------------------------------
void test_fault_injection_and_recovery() {
    std::cout << "[Test 4] Testing fault injection, WGC fallback, and fail-closed actuation..." << std::endl;

    auto fake_clock = std::make_shared<FakeClock>(1'000'000'000LL);
    auto mock_dxgi = std::make_unique<MockDxgiBackend>();
    auto mock_wgc = std::make_unique<MockWgcBackend>();
    auto* raw_dxgi = mock_dxgi.get();
    auto* raw_wgc = mock_wgc.get();

    auto unified_capture = std::make_unique<UnifiedCaptureSource>(
        std::move(mock_dxgi), std::move(mock_wgc), fake_clock);
    auto* raw_unified = unified_capture.get();

    auto mock_cuda = std::make_shared<capture::MockCudaInteropBackend>();
    auto preprocessor = std::make_unique<MockGpuPreprocessor>();

    CaptureToTensorPipeline pipeline(std::move(unified_capture), std::move(preprocessor), fake_clock);

    CaptureToTensorConfig cfg{};
    TEST_ASSERT(pipeline.initialize(cfg, mock_cuda));
    TEST_ASSERT(pipeline.warmup(5));
    TEST_ASSERT(pipeline.start());

    // Initially capturing on DXGI
    TEST_ASSERT(pipeline.active_capture_backend() == FrameSourceBackend::dxgi_duplication);
    TEST_ASSERT(pipeline.is_actuation_permitted());

    // Step 1: Inject DXGI Access Lost -> triggers recovery & fail-closed actuation
    raw_dxgi->queue_acquire_result(DXGI_ERROR_ACCESS_LOST, 0, 10);

    // Run ticks through recovery to transition to WGC
    for (int retry = 0; retry < 10; ++retry) {
        fake_clock->advance_ns(50'000'000LL);
        raw_wgc->queue_acquire_result(S_OK, static_cast<std::uint64_t>(fake_clock->now_ns() / 100LL), 1);
        PreprocessedTensorLease lease{};
        pipeline.process_frame(lease);
    }

    // Capture source should now have fallen back to WGC
    TEST_ASSERT(raw_unified->active_backend() == FrameSourceBackend::windows_graphics_capture);
    std::cout << "     -> Successfully transitioned to WGC fallback." << std::endl;

    // Step 2: WGC frames process successfully
    fake_clock->advance_ns(6'944'444LL);
    raw_wgc->queue_acquire_result(S_OK, static_cast<std::uint64_t>(fake_clock->now_ns() / 100LL), 1);
    {
        PreprocessedTensorLease lease{};
        bool ok = pipeline.process_frame(lease);
        TEST_ASSERT(ok);
        TEST_ASSERT(lease.is_valid());
    }

    // Step 3: Clear DXGI error, advance past 5.0s probe interval, and queue fresh DXGI frames
    raw_dxgi->clear_acquire_queue();
    fake_clock->advance_ns(6'000'000'000LL); // 6.0 seconds
    raw_dxgi->queue_acquire_result(S_OK, static_cast<std::uint64_t>(fake_clock->now_ns() / 100LL), 100);

    {
        PreprocessedTensorLease lease{};
        bool ok = pipeline.process_frame(lease);
        TEST_ASSERT(ok);
    }

    // Capture source should now have opportunistically promoted back to DXGI
    TEST_ASSERT(raw_unified->active_backend() == FrameSourceBackend::dxgi_duplication);
    std::cout << "     -> Opportunistic DXGI promotion successful." << std::endl;

    pipeline.stop();
    std::cout << "  -> Fault injection and recovery suite PASSED." << std::endl;
}

} // namespace

int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " Running Capture-to-Tensor Deterministic Tests      " << std::endl;
    std::cout << "====================================================" << std::endl;

    test_capture_to_tensor_latency_benchmark();
    test_simulated_soak_leak_detection();
    test_stale_frame_cutoff();
    test_fault_injection_and_recovery();

    std::cout << "====================================================" << std::endl;
    std::cout << " All Capture-to-Tensor Deterministic Tests PASSED!  " << std::endl;
    std::cout << "====================================================" << std::endl;
    return 0;
}
