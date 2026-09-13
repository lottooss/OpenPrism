// tests/cpp/test_capture_to_tensor_hardware_validation.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "aim/bus/latest_spsc_ring.hpp"
#include "aim/capture/cuda_interop_backend.hpp"
#include "aim/capture/unified_capture_source.hpp"
#include "aim/core/clock.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/capture_to_tensor_pipeline.hpp"
#include "aim/perception/gpu_preprocessor.hpp"

namespace {

using namespace aim;
using namespace aim::capture;
using namespace aim::perception;

double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) {
        return 0.0;
    }
    const auto index = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(sorted.size())) - 1.0);
    return sorted[std::min(index, sorted.size() - 1U)];
}

bool check_cuda_device(cudaDeviceProp& out_prop) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::cerr << "No CUDA-capable device detected.\n";
        return false;
    }
    if (cudaGetDeviceProperties(&out_prop, 0) != cudaSuccess) {
        std::cerr << "Failed to query CUDA device properties.\n";
        return false;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "Failed to set active CUDA device 0.\n";
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    std::uint32_t target_samples = 1000;
    std::uint32_t warmup_samples = 50;
    std::uint32_t timeout_ms = 15000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--soak") {
            target_samples = 10000;
            timeout_ms = 120000;
        } else if (arg.rfind("--samples=", 0) == 0) {
            target_samples = static_cast<std::uint32_t>(std::stoul(arg.substr(10)));
        } else if (arg.rfind("--warmup=", 0) == 0) {
            warmup_samples = static_cast<std::uint32_t>(std::stoul(arg.substr(9)));
        } else if (arg.rfind("--timeout-ms=", 0) == 0) {
            timeout_ms = static_cast<std::uint32_t>(std::stoul(arg.substr(13)));
        }
    }
    if (target_samples == 0 || target_samples > 10000 || warmup_samples > 1000 ||
        timeout_ms == 0 || timeout_ms > 120000) return 2;

    cudaDeviceProp prop{};
    if (!check_cuda_device(prop)) {
        return 2;
    }

    auto clock = std::make_shared<QpcClock>();
    auto cuda_backend = std::make_shared<RealCudaInteropBackend>();
    auto preprocessor = std::make_unique<GpuPreprocessor>();
    auto unified_capture = std::make_unique<UnifiedCaptureSource>(
        std::unique_ptr<IFrameSource>{nullptr},
        std::unique_ptr<IFrameSource>{nullptr},
        clock);

    CaptureToTensorPipeline pipeline(std::move(unified_capture), std::move(preprocessor), clock);

    CaptureToTensorConfig config{};
    config.capture_config.target_width_px = 1920;
    config.capture_config.target_height_px = 1080;
    config.capture_config.pool_capacity = 4;
    config.capture_config.timeout_ms = 0;
    config.preprocess_config.source_width_px = 1920;
    config.preprocess_config.source_height_px = 1080;
    config.preprocess_config.target_width_px = 640;
    config.preprocess_config.target_height_px = 384;
    config.stale_threshold_ns = 10'000'000LL; // Same hard source cutoff as production.

    if (!pipeline.initialize(config, cuda_backend)) {
        std::cerr << "Failed to initialize CaptureToTensorPipeline on hardware.\n";
        return 1;
    }

    auto* source = dynamic_cast<UnifiedCaptureSource*>(pipeline.frame_source());
    int capture_cuda_device = -1;
    if (!source || !cuda_backend->validate_adapter_match(source->active_adapter_luid(), capture_cuda_device)) {
        std::cerr << "Capture/CUDA adapter identity mismatch.\n";
        return 1;
    }
    std::cout << "active_adapter_luid=0x" << std::hex << source->active_adapter_luid() << std::dec
              << " cuda_device=" << capture_cuda_device << std::endl;
    if (!pipeline.warmup(warmup_samples)) {
        std::cerr << "Pipeline warmup failed.\n";
        return 1;
    }

    if (!pipeline.start()) {
        std::cerr << "Failed to start capture pipeline.\n";
        return 1;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto tensor_ready = [&](PreprocessedTensorLease& lease) {
        const auto event = lease.descriptor().native_ready_event;
        if (!event) return false;
        while (std::chrono::steady_clock::now() < deadline) {
            const auto result = cuda_backend->query_event(event);
            if (result == CudaResult::success) return true;
            if (result != CudaResult::error_not_ready) return false;
            std::this_thread::yield();
        }
        return false;
    };
    // Bound live warmup too: source access loss cannot hang a diagnostic.
    std::uint32_t live_warmup = 0;
    while (live_warmup < warmup_samples && std::chrono::steady_clock::now() < deadline) {
        PreprocessedTensorLease lease{};
        if (pipeline.process_frame(lease)) {
            if (tensor_ready(lease)) ++live_warmup;
        } else {
            std::this_thread::yield();
        }
    }
    if (live_warmup != warmup_samples) {
        std::cerr << "Live warmup timed out after " << live_warmup << " tensors; no acceptance claim.\n";
        return 1;
    }
    pipeline.reset_metrics();

    bus::LatestSpscRing<PreprocessedTensorDescriptor, 16> tensor_ring{};
    pipeline.bind_tensor_bus_ring(&tensor_ring);

    std::vector<double> presentation_age_latencies_ms;
    presentation_age_latencies_ms.reserve(target_samples);
    std::vector<double> poll_to_ready_latencies_ms;
    poll_to_ready_latencies_ms.reserve(target_samples);
    std::uint32_t completion_stale_drops = 0;

    std::uint32_t collected = 0;

    auto start_time = std::chrono::steady_clock::now();

    while (collected < target_samples && std::chrono::steady_clock::now() < deadline) {
        const auto poll_started_ns = clock->now_ns();
        PreprocessedTensorLease lease{};
        if (pipeline.process_frame(lease)) {
            if (lease.is_valid() && tensor_ready(lease)) {
                MonotonicNs t_ready = clock->now_ns();
                MonotonicNs t_presented = lease.descriptor().correlation_id.source_timestamp_ns;
                if (t_presented <= 0 || t_ready < t_presented || t_ready - t_presented > config.stale_threshold_ns) {
                    ++completion_stale_drops;
                    continue;
                }
                presentation_age_latencies_ms.push_back(static_cast<double>(t_ready - t_presented) / 1'000'000.0);
                poll_to_ready_latencies_ms.push_back(static_cast<double>(t_ready - poll_started_ns) / 1'000'000.0);

                // Verify bus ring descriptor
                PreprocessedTensorDescriptor ring_desc{};
                if (tensor_ring.try_read_latest(ring_desc)) {
                    if (!ring_desc.is_valid || ring_desc.width_px != 640 || ring_desc.height_px != 384) {
                        std::cerr << "Invalid ring tensor descriptor received.\n";
                        return 1;
                    }
                }
                ++collected;
            }
        } else {
            // Sleep brief interval to wait for next display refresh
            std::this_thread::yield();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();

    auto metrics = pipeline.metrics();
    const auto capture_health = pipeline.frame_source()->health();
    pipeline.stop();

    if (collected != target_samples) {
        std::cerr << "Insufficient live frames captured (" << collected << " / " << target_samples << ").\n";
        std::cerr << "acquired=" << metrics.frames_acquired << " stale=" << metrics.frames_dropped_stale
                  << " invalid=" << metrics.frames_dropped_invalid << " completion_stale=" << completion_stale_drops
                  << " active=" << capture_health.is_active << " access_lost=" << capture_health.is_access_lost
                  << " backend_timeouts=" << capture_health.total_timeouts << '\n';
        return 1;
    }

    std::sort(presentation_age_latencies_ms.begin(), presentation_age_latencies_ms.end());
    std::sort(poll_to_ready_latencies_ms.begin(), poll_to_ready_latencies_ms.end());

    double age_p50 = percentile(presentation_age_latencies_ms, 0.50);
    double age_p95 = percentile(presentation_age_latencies_ms, 0.95);
    double age_p99 = percentile(presentation_age_latencies_ms, 0.99);

    std::cout << "device=" << prop.name << "\n";
    std::cout << "compute_capability=" << prop.major << "." << prop.minor << "\n";
    std::cout << "capture_backend=" << (pipeline.active_capture_backend() == FrameSourceBackend::dxgi_duplication ? "dxgi_duplication" : "wgc") << "\n";
    std::cout << "source_resolution=1920x1080\n";
    std::cout << "tensor_resolution=640x384_fp16_nchw\n";
    std::cout << "sample_count=" << collected << "\n";
    std::cout << "live_warmup=" << live_warmup << " source_cutoff_ms=10 gpu_completion_verified=true\n";
    std::cout << "elapsed_seconds=" << std::fixed << std::setprecision(2) << elapsed_s << "\n";
    std::cout << "fps=" << std::fixed << std::setprecision(1) << (static_cast<double>(collected) / elapsed_s) << "\n";

    std::cout << "capture_to_tensor_min_ms=" << std::fixed << std::setprecision(4) << metrics.capture_to_tensor_latency.min_ms << "\n";
    std::cout << "capture_to_tensor_avg_ms=" << metrics.capture_to_tensor_latency.avg_ms << "\n";
    std::cout << "capture_to_tensor_p50_ms=" << metrics.capture_to_tensor_latency.p50_ms << "\n";
    std::cout << "capture_to_tensor_p95_ms=" << metrics.capture_to_tensor_latency.p95_ms << "\n";
    std::cout << "capture_to_tensor_p99_ms=" << metrics.capture_to_tensor_latency.p99_ms << "\n";
    std::cout << "capture_to_tensor_max_ms=" << metrics.capture_to_tensor_latency.max_ms << "\n";

    std::cout << "stage_acquire_p50_ms=" << metrics.stage_acquire_latency.p50_ms << "\n";
    std::cout << "stage_acquire_p95_ms=" << metrics.stage_acquire_latency.p95_ms << "\n";
    std::cout << "stage_acquire_p99_ms=" << metrics.stage_acquire_latency.p99_ms << "\n";

    std::cout << "stage_preprocess_p50_ms=" << metrics.stage_preprocess_latency.p50_ms << "\n";
    std::cout << "stage_preprocess_p95_ms=" << metrics.stage_preprocess_latency.p95_ms << "\n";
    std::cout << "stage_preprocess_p99_ms=" << metrics.stage_preprocess_latency.p99_ms << "\n";

    std::cout << "presentation_age_p50_ms=" << age_p50 << "\n";
    std::cout << "presentation_age_p95_ms=" << age_p95 << "\n";
    std::cout << "presentation_age_p99_ms=" << age_p99 << "\n";
    std::cout << "poll_to_gpu_ready_p50_ms=" << percentile(poll_to_ready_latencies_ms, 0.50) << "\n";
    std::cout << "poll_to_gpu_ready_p95_ms=" << percentile(poll_to_ready_latencies_ms, 0.95) << "\n";
    std::cout << "poll_to_gpu_ready_p99_ms=" << percentile(poll_to_ready_latencies_ms, 0.99) << "\n";
    std::cout << "poll_to_gpu_ready_max_ms=" << poll_to_ready_latencies_ms.back() << "\n";
    std::cout << "completion_stale_drops=" << completion_stale_drops << "\n";

    std::cout << "frames_acquired=" << metrics.frames_acquired << "\n";
    std::cout << "frames_dropped_stale=" << metrics.frames_dropped_stale << "\n";
    std::cout << "frames_dropped_invalid=" << metrics.frames_dropped_invalid << "\n";

    // Include event-confirmed GPU completion; CPU enqueue alone cannot certify readiness.
    const double p99_budget_ms = 1.0;
    const double ready_p99_ms = percentile(poll_to_ready_latencies_ms, 0.99);
    if (ready_p99_ms > p99_budget_ms) {
        std::cerr << "Poll-to-GPU-ready p99 latency " << ready_p99_ms
                  << " ms exceeded budget " << p99_budget_ms << " ms.\n";
        return 1;
    }

    if (metrics.frames_dropped_invalid > 0) {
        std::cerr << "Invalid frame drops detected during hardware benchmark: "
                  << metrics.frames_dropped_invalid << "\n";
        return 1;
    }

    std::cout << "M2-06 Hardware Capture-to-Tensor Validation: PASSED\n";
    return 0;
}
