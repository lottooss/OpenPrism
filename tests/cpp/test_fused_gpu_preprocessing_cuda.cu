#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "aim/perception/fused_kernel.hpp"
#include "aim/perception/preprocess_types.hpp"

namespace {

using aim::FrameFormat;
using aim::perception::FusedKernelParams;
using aim::perception::FusedPreprocessKernel;
using aim::perception::HalfFloat;
using aim::perception::PreprocessConfig;

bool cuda_ok(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

double percentile(const std::vector<float>& sorted_ms, double quantile) {
    if (sorted_ms.empty()) {
        return 0.0;
    }
    const auto index = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(sorted_ms.size())) - 1.0);
    return sorted_ms[std::min(index, sorted_ms.size() - 1U)];
}

void fill_bgra_gradient(std::vector<uchar4>& pixels,
                        std::uint32_t width,
                        std::uint32_t height) {
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            pixels[static_cast<std::size_t>(y) * width + x] = make_uchar4(
                static_cast<unsigned char>((x * 13U + y * 3U) & 0xFFU),
                static_cast<unsigned char>((x * 5U + y * 11U) & 0xFFU),
                static_cast<unsigned char>((x * 7U + y * 17U) & 0xFFU),
                255U);
        }
    }
}

} // namespace

int main() {
    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") || device_count < 1) {
        std::cerr << "Hardware validation requires an available CUDA device.\n";
        return 2;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties") ||
        !cuda_ok(cudaSetDevice(0), "cudaSetDevice")) {
        return 2;
    }

    const PreprocessConfig config{};
    std::vector<uchar4> source(
        static_cast<std::size_t>(config.source_width_px) * config.source_height_px);
    fill_bgra_gradient(source, config.source_width_px, config.source_height_px);

    cudaChannelFormatDesc channel = cudaCreateChannelDesc<uchar4>();
    cudaArray_t source_array = nullptr;
    cudaSurfaceObject_t source_surface = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    void* device_output = nullptr;

    const auto cleanup = [&]() {
        FusedPreprocessKernel::free_device_tensor(device_output);
        if (stop_event != nullptr) static_cast<void>(cudaEventDestroy(stop_event));
        if (start_event != nullptr) static_cast<void>(cudaEventDestroy(start_event));
        if (stream != nullptr) static_cast<void>(cudaStreamDestroy(stream));
        if (source_surface != 0) static_cast<void>(cudaDestroySurfaceObject(source_surface));
        if (source_array != nullptr) static_cast<void>(cudaFreeArray(source_array));
    };

    if (!cuda_ok(cudaMallocArray(&source_array, &channel, config.source_width_px,
                                 config.source_height_px, cudaArraySurfaceLoadStore),
                 "cudaMallocArray") ||
        !cuda_ok(cudaMemcpy2DToArray(source_array, 0, 0, source.data(),
                                    static_cast<std::size_t>(config.source_width_px) * sizeof(uchar4),
                                    static_cast<std::size_t>(config.source_width_px) * sizeof(uchar4),
                                    config.source_height_px, cudaMemcpyHostToDevice),
                 "cudaMemcpy2DToArray")) {
        cleanup();
        return 2;
    }

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = source_array;
    if (!cuda_ok(cudaCreateSurfaceObject(&source_surface, &resource),
                 "cudaCreateSurfaceObject") ||
        !cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags") ||
        !cuda_ok(cudaEventCreate(&start_event), "cudaEventCreate(start)") ||
        !cuda_ok(cudaEventCreate(&stop_event), "cudaEventCreate(stop)") ||
        !FusedPreprocessKernel::allocate_device_tensor(&device_output, config.tensor_size_bytes())) {
        cleanup();
        return 2;
    }

    FusedKernelParams gpu_params{};
    gpu_params.src_width = config.source_width_px;
    gpu_params.src_height = config.source_height_px;
    gpu_params.src_stride_bytes = config.source_width_px * 4U;
    gpu_params.src_format = FrameFormat::b8g8r8a8_unorm;
    gpu_params.dst_tensor = device_output;
    gpu_params.dst_width = config.target_width_px;
    gpu_params.dst_height = config.target_height_px;
    gpu_params.scaled_width = config.scaled_width_px;
    gpu_params.scaled_height = config.scaled_height_px;
    gpu_params.pad_x = config.pad_x;
    gpu_params.pad_y = config.pad_y;
    gpu_params.pad_value_normalized = config.pad_value_normalized;
    gpu_params.pad_half_bits = HalfFloat(config.pad_value_normalized).bits;
    gpu_params.scale_x = static_cast<float>(config.source_width_px) /
        static_cast<float>(config.scaled_width_px);
    gpu_params.scale_y = static_cast<float>(config.source_height_px) /
        static_cast<float>(config.scaled_height_px);

    if (!FusedPreprocessKernel::launch_cuda_surface(
            static_cast<aim::capture::CudaSurfaceObjectHandle>(source_surface), gpu_params,
            reinterpret_cast<aim::capture::CudaStreamHandle>(stream)) ||
        !cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize(parity)")) {
        cleanup();
        return 2;
    }

    std::vector<std::uint16_t> gpu_output(config.element_count());
    std::vector<std::uint16_t> cpu_output(config.element_count());
    if (!cuda_ok(cudaMemcpy(gpu_output.data(), device_output, config.tensor_size_bytes(),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(output)")) {
        cleanup();
        return 2;
    }

    FusedKernelParams cpu_params = gpu_params;
    cpu_params.src_pixels = source.data();
    cpu_params.dst_tensor = cpu_output.data();
    if (!FusedPreprocessKernel::execute_software(cpu_params)) {
        std::cerr << "CPU reference execution failed.\n";
        cleanup();
        return 2;
    }

    float max_error = 0.0F;
    double sum_error = 0.0;
    for (std::size_t i = 0; i < gpu_output.size(); ++i) {
        const float error = std::abs(
            HalfFloat(gpu_output[i]).to_float() - HalfFloat(cpu_output[i]).to_float());
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    const double mean_error = sum_error / static_cast<double>(gpu_output.size());
    constexpr float kParityTolerance = 0.002F;
    if (max_error > kParityTolerance) {
        std::cerr << "CPU/GPU parity failed: max_abs_error=" << max_error
                  << " tolerance=" << kParityTolerance << '\n';
        cleanup();
        return 1;
    }

    constexpr std::uint32_t kWarmupIterations = 200;
    constexpr std::uint32_t kSampleCount = 10'000;
    for (std::uint32_t i = 0; i < kWarmupIterations; ++i) {
        if (!FusedPreprocessKernel::launch_cuda_surface(
                static_cast<aim::capture::CudaSurfaceObjectHandle>(source_surface), gpu_params,
                reinterpret_cast<aim::capture::CudaStreamHandle>(stream))) {
            cleanup();
            return 2;
        }
    }
    if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)")) {
        cleanup();
        return 2;
    }

    std::vector<float> samples_ms;
    samples_ms.reserve(kSampleCount);
    for (std::uint32_t i = 0; i < kSampleCount; ++i) {
        if (!cuda_ok(cudaEventRecord(start_event, stream), "cudaEventRecord(start)") ||
            !FusedPreprocessKernel::launch_cuda_surface(
                static_cast<aim::capture::CudaSurfaceObjectHandle>(source_surface), gpu_params,
                reinterpret_cast<aim::capture::CudaStreamHandle>(stream)) ||
            !cuda_ok(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)") ||
            !cuda_ok(cudaEventSynchronize(stop_event), "cudaEventSynchronize")) {
            cleanup();
            return 2;
        }
        float elapsed_ms = 0.0F;
        if (!cuda_ok(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
                     "cudaEventElapsedTime")) {
            cleanup();
            return 2;
        }
        samples_ms.push_back(elapsed_ms);
    }
    std::sort(samples_ms.begin(), samples_ms.end());
    const double average_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0) /
        static_cast<double>(samples_ms.size());
    const double p50_ms = percentile(samples_ms, 0.50);
    const double p95_ms = percentile(samples_ms, 0.95);
    const double p99_ms = percentile(samples_ms, 0.99);
    const double max_ms = samples_ms.back();

    std::cout << "device=" << properties.name << '\n'
              << "compute_capability=" << properties.major << '.' << properties.minor << '\n'
              << "source=1920x1080_bgra8 target=640x384_fp16_nchw\n"
              << "warmup_iterations=" << kWarmupIterations << '\n'
              << "sample_count=" << kSampleCount << '\n'
              << "parity_max_abs_error=" << max_error << '\n'
              << "parity_mean_abs_error=" << mean_error << '\n'
              << "kernel_avg_ms=" << average_ms << '\n'
              << "kernel_p50_ms=" << p50_ms << '\n'
              << "kernel_p95_ms=" << p95_ms << '\n'
              << "kernel_p99_ms=" << p99_ms << '\n'
              << "kernel_max_ms=" << max_ms << '\n';

    cleanup();
    return p99_ms <= 0.65 ? 0 : 1;
}
