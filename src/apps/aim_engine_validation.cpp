// Cold hardware validation tool: stream waits and file I/O here are intentional.
// Its measurements cover GPU inference only; it never creates an actuator.
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "aim/config/sha256.hpp"
#include "aim/perception/engine_artifact.hpp"
#include "aim/perception/yolo_decoder.hpp"

namespace {
constexpr std::size_t kInputElements = 3U * 384U * 640U;
constexpr std::size_t kAnchors = 5040U;
constexpr std::size_t kOutputElements = 5U * kAnchors;
constexpr double kCenterToleranceModelPx = 0.25;
constexpr double kConfidenceTolerance = 0.005;

struct Options {
    std::filesystem::path engine, inputs, reference, actual, report;
    std::string sha256;
    std::uint32_t frames{0}, warmup{50}, iterations{20};
};

std::uint32_t number(std::string_view text) {
    std::uint32_t result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Invalid numeric argument");
    return result;
}

Options parse(int argc, char **argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg.starts_with("--engine="))
            result.engine = arg.substr(9);
        else if (arg.starts_with("--sha256="))
            result.sha256 = arg.substr(9);
        else if (arg.starts_with("--inputs="))
            result.inputs = arg.substr(9);
        else if (arg.starts_with("--reference="))
            result.reference = arg.substr(12);
        else if (arg.starts_with("--actual="))
            result.actual = arg.substr(9);
        else if (arg.starts_with("--report="))
            result.report = arg.substr(9);
        else if (arg.starts_with("--frames="))
            result.frames = number(arg.substr(9));
        else if (arg.starts_with("--warmup="))
            result.warmup = number(arg.substr(9));
        else if (arg.starts_with("--iterations="))
            result.iterations = number(arg.substr(13));
        else
            throw std::runtime_error("Unknown validation argument");
    }
    if (result.engine.empty() || result.inputs.empty() || result.reference.empty() ||
        result.actual.empty() || result.report.empty() || result.frames == 0 ||
        result.frames > 128 || result.warmup == 0 || result.warmup > 1000 ||
        result.iterations == 0 || result.iterations > 1000)
        throw std::runtime_error("Required paths/frames missing or run bounds exceeded");
    return result;
}

void checked(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
std::vector<T> read_exact(const std::filesystem::path &path, std::size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(T))
        throw std::runtime_error("Raw golden file size mismatch");
    std::vector<T> result(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(result.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!file)
        throw std::runtime_error("Cannot read raw golden file");
    return result;
}

std::string hash_bytes(const void *bytes, std::size_t length) {
    aim::config::Sha256 hasher;
    hasher.update(static_cast<const std::uint8_t *>(bytes), length);
    return hasher.finalize_hex();
}

class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kERROR)
            std::cerr << message << '\n';
    }
};

struct Backend {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    cudaEvent_t start{}, end{};
    cudaGraph_t graph{};
    cudaGraphExec_t graph_exec{};
    void *input{};
    void *output{};
    float *host{};
    ~Backend() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (graph_exec)
            cudaGraphExecDestroy(graph_exec);
        if (graph)
            cudaGraphDestroy(graph);
        context.reset();
        engine.reset();
        runtime.reset();
        if (input)
            cudaFree(input);
        if (output)
            cudaFree(output);
        if (host)
            cudaFreeHost(host);
        if (start)
            cudaEventDestroy(start);
        if (end)
            cudaEventDestroy(end);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void initialize(const aim::perception::VerifiedEngineArtifact &artifact) {
        checked(cudaSetDevice(0), "cudaSetDevice");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
            throw std::runtime_error("TensorRT runtime creation failed");
        engine.reset(
            runtime->deserializeCudaEngine(artifact.bytes().data(), artifact.bytes().size()));
        if (!engine || engine->getNbIOTensors() != 2)
            throw std::runtime_error("TensorRT engine deserialization/I/O count failed");
        const auto in = engine->getTensorShape("images"), out = engine->getTensorShape("output0");
        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != 384 || in.d[3] != 640 ||
            out.nbDims != 3 || out.d[0] != 1 || out.d[1] != 5 || out.d[2] != 5040 ||
            engine->getTensorIOMode("images") != nvinfer1::TensorIOMode::kINPUT ||
            engine->getTensorIOMode("output0") != nvinfer1::TensorIOMode::kOUTPUT ||
            engine->getTensorDataType("images") != nvinfer1::DataType::kHALF ||
            engine->getTensorDataType("output0") != nvinfer1::DataType::kFLOAT ||
            engine->getTensorFormat("images") != nvinfer1::TensorFormat::kLINEAR ||
            engine->getTensorFormat("output0") != nvinfer1::TensorFormat::kLINEAR ||
            engine->getTensorLocation("images") != nvinfer1::TensorLocation::kDEVICE ||
            engine->getTensorLocation("output0") != nvinfer1::TensorLocation::kDEVICE)
            throw std::runtime_error("Engine must bind FP16 images[1,3,384,640] and FP32 "
                                     "output0[1,5,5040], linear/device");
        context.reset(engine->createExecutionContext());
        if (!context)
            throw std::runtime_error("Execution context creation failed");
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation");
        checked(cudaEventCreate(&start), "start event creation");
        checked(cudaEventCreate(&end), "end event creation");
        checked(cudaMalloc(&input, kInputElements * sizeof(std::uint16_t)), "input allocation");
        checked(cudaMalloc(&output, kOutputElements * sizeof(float)), "output allocation");
        checked(cudaMallocHost(&host, kOutputElements * sizeof(float)), "pinned output allocation");
        if (!context->setTensorAddress("images", input) ||
            !context->setTensorAddress("output0", output))
            throw std::runtime_error("Tensor address binding failed");
    }
    void warmup(const std::uint16_t *values, std::uint32_t iterations) {
        checked(cudaMemcpyAsync(input, values, kInputElements * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "warmup input");
        for (std::uint32_t i = 0; i < iterations; ++i) {
            if (!context->enqueueV3(stream))
                throw std::runtime_error("Warmup enqueueV3 failed");
            checked(cudaStreamSynchronize(stream), "warmup wait");
        }
        checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "graph begin");
        const bool enqueued = context->enqueueV3(stream);
        const auto captured = cudaStreamEndCapture(stream, &graph);
        if (!enqueued)
            throw std::runtime_error("Graph enqueueV3 failed");
        checked(captured, "graph end");
        checked(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "graph instantiate");
    }
    float infer(const std::uint16_t *values) {
        checked(cudaMemcpyAsync(input, values, kInputElements * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice, stream),
                "input upload");
        checked(cudaEventRecord(start, stream), "inference start");
        checked(cudaGraphLaunch(graph_exec, stream), "inference graph launch");
        checked(cudaEventRecord(end, stream), "inference end");
        checked(cudaMemcpyAsync(host, output, kOutputElements * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
                "output download");
        checked(cudaStreamSynchronize(stream), "validation collection wait");
        float elapsed{};
        checked(cudaEventElapsedTime(&elapsed, start, end), "event elapsed time");
        if (!std::isfinite(elapsed) || elapsed <= 0)
            throw std::runtime_error("Invalid CUDA event timing");
        return elapsed;
    }
};

double percentile(const std::vector<float> &sorted, double quantile) {
    const double index = static_cast<double>(sorted.size() - 1U) * quantile;
    const auto low = static_cast<std::size_t>(index);
    const auto high = std::min(low + 1U, sorted.size() - 1U);
    return sorted[low] + (sorted[high] - sorted[low]) * (index - static_cast<double>(low));
}

bool nms_equal(const aim::bus::TargetObservationBatch &expected,
               const aim::bus::TargetObservationBatch &actual) {
    if (expected.target_count != actual.target_count)
        return false;
    std::array<bool, aim::bus::kMaxObservations> used{};
    for (const auto &target : expected.items()) {
        double nearest = 1.0e30;
        std::size_t choice = used.size();
        for (std::size_t i = 0; i < actual.target_count; ++i) {
            const auto &candidate = actual.targets[i];
            if (used[i] || candidate.semantic_id != target.semantic_id)
                continue;
            const double distance = std::hypot(candidate.center_px.x - target.center_px.x,
                                               candidate.center_px.y - target.center_px.y);
            if (distance < nearest) {
                nearest = distance;
                choice = i;
            }
        }
        if (choice == used.size() || nearest > kCenterToleranceModelPx * 3.0 ||
            std::abs(actual.targets[choice].confidence - target.confidence) > kConfidenceTolerance)
            return false;
        used[choice] = true;
    }
    return true;
}

int run(const Options &options) {
    aim::ModelManifest manifest;
    manifest.model_path = options.engine.string();
    manifest.engine_sha256 = options.sha256;
    aim::perception::VerifiedEngineArtifact artifact;
    if (aim::perception::EngineArtifactVerifier::load_and_verify(manifest, {}, artifact) !=
        aim::PerceptionStatus::ok)
        throw std::runtime_error("Engine hash/integrity verification failed");
    const auto inputs = read_exact<std::uint16_t>(options.inputs, options.frames * kInputElements);
    const auto reference = read_exact<float>(options.reference, options.frames * kOutputElements);
    for (const auto half : inputs)
        if (half > 0x3C00U)
            throw std::runtime_error("Golden FP16 inputs must be finite RGB in [0,1]");
    for (const float value : reference)
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite golden output");
    std::vector<float> actual(reference.size());
    std::vector<float> timings;
    timings.reserve(static_cast<std::size_t>(options.frames) * options.iterations);
    aim::perception::YoloDecoder decoder;
    std::vector<aim::bus::TargetObservationBatch> decoded_reference(options.frames);
    for (std::uint32_t frame = 0; frame < options.frames; ++frame) {
        const aim::CorrelationId id{frame + 1U, 1, 1, 0};
        decoder.decode(reference.data() + frame * kOutputElements, kOutputElements, id, 1,
                       decoded_reference[frame]);
    }
    Backend backend;
    backend.initialize(artifact);
    backend.warmup(inputs.data(), options.warmup);
    double max_center_error = 0, max_confidence_error = 0, max_box_size_error = 0;
    std::uint64_t nms_mismatches = 0;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        for (std::uint32_t frame = 0; frame < options.frames; ++frame) {
            timings.push_back(backend.infer(inputs.data() + frame * kInputElements));
            const auto *expected = reference.data() + frame * kOutputElements;
            for (std::size_t i = 0; i < kOutputElements; ++i)
                if (!std::isfinite(backend.host[i]))
                    throw std::runtime_error("Nonfinite TensorRT output");
            for (std::size_t i = 0; i < kAnchors; ++i) {
                max_center_error = std::max(
                    max_center_error, std::hypot(static_cast<double>(backend.host[i]) - expected[i],
                                                 static_cast<double>(backend.host[kAnchors + i]) -
                                                     expected[kAnchors + i]));
                max_confidence_error =
                    std::max(max_confidence_error,
                             std::abs(static_cast<double>(backend.host[4U * kAnchors + i]) -
                                      expected[4U * kAnchors + i]));
                for (std::size_t channel = 2; channel < 4; ++channel)
                    max_box_size_error = std::max(
                        max_box_size_error,
                        std::abs(static_cast<double>(backend.host[channel * kAnchors + i]) -
                                 expected[channel * kAnchors + i]));
            }
            aim::bus::TargetObservationBatch decoded;
            decoder.decode(backend.host, kOutputElements, aim::CorrelationId{frame + 1U, 1, 1, 0},
                           1, decoded);
            if (!nms_equal(decoded_reference[frame], decoded))
                ++nms_mismatches;
            std::copy_n(backend.host, kOutputElements, actual.data() + frame * kOutputElements);
        }
    }
    if (!options.actual.parent_path().empty())
        std::filesystem::create_directories(options.actual.parent_path());
    {
        std::ofstream file(options.actual, std::ios::binary);
        file.write(reinterpret_cast<const char *>(actual.data()),
                   static_cast<std::streamsize>(actual.size() * sizeof(float)));
        if (!file)
            throw std::runtime_error("Cannot write raw actual outputs");
    }
    cudaDeviceProp properties{};
    checked(cudaGetDeviceProperties(&properties, 0), "device properties");
    int runtime_version{}, driver_version{};
    checked(cudaRuntimeGetVersion(&runtime_version), "CUDA runtime version");
    checked(cudaDriverGetVersion(&driver_version), "CUDA driver version");
    const bool parity = max_center_error <= kCenterToleranceModelPx &&
                        max_confidence_error <= kConfidenceTolerance && nms_mismatches == 0;
    const auto samples = timings;
    std::sort(timings.begin(), timings.end());
    nlohmann::json counts = nlohmann::json::array();
    for (const auto &frame : decoded_reference)
        counts.push_back(frame.target_count);
    const nlohmann::json report{
        {"schema_version", 1},
        {"evidence_kind", "measured"},
        {"validation_scope", "TensorRT with FP16 input versus supplied FP32 golden outputs"},
        {"engine_sha256", hash_bytes(artifact.bytes().data(), artifact.bytes().size())},
        {"golden_inputs_sha256", hash_bytes(inputs.data(), inputs.size() * sizeof(inputs[0]))},
        {"reference_outputs_sha256",
         hash_bytes(reference.data(), reference.size() * sizeof(reference[0]))},
        {"actual_outputs_sha256", hash_bytes(actual.data(), actual.size() * sizeof(actual[0]))},
        {"gpu_device", properties.name},
        {"cuda_runtime_version", runtime_version},
        {"cuda_driver_api_version", driver_version},
        {"tensorrt_version", std::to_string(NV_TENSORRT_MAJOR) + "." +
                                 std::to_string(NV_TENSORRT_MINOR) + "." +
                                 std::to_string(NV_TENSORRT_PATCH)},
        {"frames", options.frames},
        {"warmup_iterations", options.warmup},
        {"iterations_per_frame", options.iterations},
        {"sample_count", timings.size()},
        {"timing_method", "cuda_events"},
        {"cuda_graph", true},
        {"concurrent_workload", nullptr},
        {"workload_scope", "Concurrent workload was not independently verified by this tool"},
        {"center_tolerance_model_px", kCenterToleranceModelPx},
        {"confidence_tolerance", kConfidenceTolerance},
        {"max_center_error_model_px", max_center_error},
        {"max_center_error_source_px", max_center_error * 3.0},
        {"max_confidence_error", max_confidence_error},
        {"max_box_size_error_model_px", max_box_size_error},
        {"nms_mismatch_executions", nms_mismatches},
        {"reference_detection_counts", counts},
        {"tensorrt_parity_passed", parity},
        {"latency_p50_ms", percentile(timings, 0.50)},
        {"latency_p95_ms", percentile(timings, 0.95)},
        {"latency_p99_ms", percentile(timings, 0.99)},
        {"latency_max_ms", timings.back()},
        {"latency_samples_ms", samples},
        {"detector_accuracy_certified", false},
        {"milestone_acceptance_passed", false},
        {"limitations", "FP16 input conversion and inference tolerance are distinct from strict "
                        "FP32 ONNX parity. GPU inference timing excludes transfers, capture, "
                        "control and concurrent-game acceptance."}};
    if (!options.report.parent_path().empty())
        std::filesystem::create_directories(options.report.parent_path());
    std::ofstream file(options.report);
    file << report.dump(2) << '\n';
    if (!file)
        throw std::runtime_error("Cannot write validation report");
    std::cout << "TensorRT parity=" << parity << " center_model_px=" << max_center_error
              << " confidence=" << max_confidence_error << " NMS mismatches=" << nms_mismatches
              << " inference_p99_ms=" << percentile(timings, 0.99) << '\n';
    return parity ? 0 : 3;
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "aim_engine_validation --engine=FILE --sha256=HEX --inputs=FP16_RAW "
                     "--reference=FP32_RAW "
                     "--frames=N --actual=FP32_RAW --report=JSON [--warmup=50] [--iterations=20]\n";
        return 0;
    }
    try {
        return run(parse(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "Engine validation failed: " << error.what() << '\n';
        return 2;
    }
}
