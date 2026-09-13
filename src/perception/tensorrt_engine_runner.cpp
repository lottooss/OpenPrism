#include "aim/perception/tensorrt_engine_runner.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <climits>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
#include <NvInfer.h>
#include <cuda_runtime.h>
#endif

#include "aim/perception/engine_artifact.hpp"
#include "aim/perception/yolo_decoder.hpp"

namespace aim::perception {

#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
namespace {
// TensorRT may log during enqueue: record a bounded error bit, never perform I/O.
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char*) noexcept override {
        if (severity <= Severity::kERROR) {
            error.store(true, std::memory_order_relaxed);
        }
    }
    std::atomic<bool> error{false};
};
}
#endif

struct TensorRtEngineRunner::Impl {
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
    TrtLogger logger_{};
    // Destruction order is context, engine, runtime, then logger.
    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{};
    cudaStream_t stream_{nullptr};
    cudaEvent_t ready_event_{nullptr};
    cudaEvent_t inference_start_{nullptr};
    cudaEvent_t inference_end_{nullptr};
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};
    int device_id_{0};
#endif
    void* d_input_{nullptr};
    void* d_output_{nullptr};
    void* h_output_{nullptr};
    std::size_t input_size_bytes_{0};
    std::size_t output_size_bytes_{0};
    YoloDecoder yolo_decoder_;
    InferenceTicket active_ticket_{};
    PerceptionRequest active_request_{};
    std::uint64_t next_ticket_id_{1};
    bool pending_inference_{false};

    explicit Impl(const TensorRtRunnerConfig& config)
        : yolo_decoder_(YoloDecoderConfig{
              config.confidence_threshold, config.nms_iou_threshold,
              config.num_anchors, config.num_classes, config.input_width,
              config.input_height, 1920, 1080, config.max_detections,
              AffineTransform2D::create_letterbox(1920, 1080, config.input_width, config.input_height)}) {
        input_size_bytes_ = static_cast<std::size_t>(config.input_width) *
            config.input_height * config.input_channels * sizeof(std::uint16_t);
        output_size_bytes_ = static_cast<std::size_t>(config.num_anchors) *
            (static_cast<std::size_t>(config.num_classes) + 4U) * sizeof(float);
    }
    ~Impl() noexcept { cleanup(); }

    void cleanup() noexcept {
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
        // Shutdown/reload are quiescent cold paths: drain before freeing borrowed or owned buffers.
        if (stream_) {
            static_cast<void>(cudaSetDevice(device_id_));
            static_cast<void>(cudaStreamSynchronize(stream_));
        }
        if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
        if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
        context_.reset();
        engine_.reset();
        runtime_.reset();
        if (ready_event_) { cudaEventDestroy(ready_event_); ready_event_ = nullptr; }
        if (inference_start_) { cudaEventDestroy(inference_start_); inference_start_ = nullptr; }
        if (inference_end_) { cudaEventDestroy(inference_end_); inference_end_ = nullptr; }
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
        if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
        if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
        logger_.error.store(false, std::memory_order_relaxed);
#endif
        pending_inference_ = false;
        active_ticket_ = {};
        active_request_ = {};
        // Keep ticket sequence across reloads so old tickets cannot alias new work.
    }

    PerceptionStatus load(const VerifiedEngineArtifact& artifact,
                          const ModelManifest& manifest,
                          const TensorRtRunnerConfig& config) noexcept {
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
        if (manifest.target_gpu_device_id > static_cast<std::uint32_t>(INT_MAX)) {
            return PerceptionStatus::invalid_argument;
        }
        device_id_ = static_cast<int>(manifest.target_gpu_device_id);
        if (cudaSetDevice(device_id_) != cudaSuccess) { return PerceptionStatus::device_error; }
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) { return PerceptionStatus::backend_unavailable; }
        engine_.reset(runtime_->deserializeCudaEngine(artifact.bytes().data(), artifact.bytes().size()));
        if (!engine_ || engine_->getNbIOTensors() != 2) { return PerceptionStatus::unsupported_format; }
        const auto input = engine_->getTensorShape("images");
        const auto output = engine_->getTensorShape("output0");
        if (input.nbDims != 4 || input.d[0] != 1 || input.d[1] != static_cast<std::int64_t>(config.input_channels) ||
            input.d[2] != static_cast<std::int64_t>(config.input_height) || input.d[3] != static_cast<std::int64_t>(config.input_width) ||
            output.nbDims != 3 || output.d[0] != 1 || output.d[1] != static_cast<std::int64_t>(config.num_classes) + 4 ||
            output.d[2] != static_cast<std::int64_t>(config.num_anchors) ||
            engine_->getTensorIOMode("images") != nvinfer1::TensorIOMode::kINPUT ||
            engine_->getTensorIOMode("output0") != nvinfer1::TensorIOMode::kOUTPUT ||
            engine_->getTensorDataType("images") != nvinfer1::DataType::kHALF ||
            engine_->getTensorDataType("output0") != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorFormat("images") != nvinfer1::TensorFormat::kLINEAR ||
            engine_->getTensorFormat("output0") != nvinfer1::TensorFormat::kLINEAR ||
            engine_->getTensorLocation("images") != nvinfer1::TensorLocation::kDEVICE ||
            engine_->getTensorLocation("output0") != nvinfer1::TensorLocation::kDEVICE) {
            return PerceptionStatus::unsupported_format;
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) { return PerceptionStatus::device_error; }
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreate(&inference_start_) != cudaSuccess ||
            cudaEventCreate(&inference_end_) != cudaSuccess ||
            cudaMalloc(&d_input_, input_size_bytes_) != cudaSuccess ||
            cudaMalloc(&d_output_, output_size_bytes_) != cudaSuccess ||
            cudaMallocHost(&h_output_, output_size_bytes_) != cudaSuccess ||
            !context_->setTensorAddress("images", d_input_) ||
            !context_->setTensorAddress("output0", d_output_)) {
            return PerceptionStatus::device_error;
        }
        return PerceptionStatus::ok;
#else
        static_cast<void>(artifact);
        static_cast<void>(manifest);
        static_cast<void>(config);
        return PerceptionStatus::backend_unavailable;
#endif
    }
};

TensorRtEngineRunner::TensorRtEngineRunner(TensorRtRunnerConfig config)
    : impl_(std::make_unique<Impl>(config)),
      config_(std::move(config)) {}

TensorRtEngineRunner::~TensorRtEngineRunner() {
    shutdown();
}

TensorRtEngineRunner::TensorRtEngineRunner(TensorRtEngineRunner&& other) noexcept
    : impl_(std::move(other.impl_)),
      config_(std::move(other.config_)),
      manifest_(std::move(other.manifest_)),
      last_inference_ms_(other.last_inference_ms_),
      is_initialized_(other.is_initialized_),
      is_warmed_up_(other.is_warmed_up_),
      is_graph_captured_(other.is_graph_captured_),
      has_device_fault_(other.has_device_fault_),
      total_inferences_(other.total_inferences_),
      total_faults_(other.total_faults_) {
    other.last_inference_ms_.reset();
    other.is_initialized_ = false;
    other.is_warmed_up_ = false;
    other.is_graph_captured_ = false;
    other.has_device_fault_ = false;
    other.total_inferences_ = 0;
    other.total_faults_ = 0;
}

TensorRtEngineRunner& TensorRtEngineRunner::operator=(TensorRtEngineRunner&& other) noexcept {
    if (this != &other) {
        shutdown();
        impl_ = std::move(other.impl_);
        config_ = std::move(other.config_);
        manifest_ = std::move(other.manifest_);
        last_inference_ms_ = other.last_inference_ms_;
        is_initialized_ = other.is_initialized_;
        is_warmed_up_ = other.is_warmed_up_;
        is_graph_captured_ = other.is_graph_captured_;
        has_device_fault_ = other.has_device_fault_;
        total_inferences_ = other.total_inferences_;
        total_faults_ = other.total_faults_;

        other.last_inference_ms_.reset();
        other.is_initialized_ = false;
        other.is_warmed_up_ = false;
        other.is_graph_captured_ = false;
        other.has_device_fault_ = false;
        other.total_inferences_ = 0;
        other.total_faults_ = 0;
    }
    return *this;
}

ModelContract TensorRtEngineRunner::contract() const noexcept {
    ModelContract result{};
    result.model_name = config_.model_name;
    result.model_version = config_.model_version;
    result.input_width_px = config_.input_width;
    result.input_height_px = config_.input_height;
    result.input_channels = config_.input_channels;
    result.max_detections = config_.max_detections;
    result.confidence_threshold = config_.confidence_threshold;
    return result;
}

bool TensorRtEngineRunner::is_backend_compiled() noexcept {
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
    return true;
#else
    return false;
#endif
}

bool TensorRtEngineRunner::manifest_matches_contract(const ModelManifest& manifest) const noexcept {
    return config_.input_width == 640 && config_.input_height == 384 &&
        config_.input_channels == 3 && config_.num_anchors == 5040 && config_.num_classes == 1 &&
        config_.max_detections > 0 && config_.max_detections <= bus::kMaxObservations &&
        std::isfinite(config_.confidence_threshold) && config_.confidence_threshold > 0.0F &&
        config_.confidence_threshold <= 1.0F && std::isfinite(config_.nms_iou_threshold) &&
        config_.nms_iou_threshold >= 0.0F && config_.nms_iou_threshold <= 1.0F &&
        manifest.runtime_backend == "tensorrt" && manifest.precision == "fp16" &&
        manifest.input_width == config_.input_width &&
        manifest.input_height == config_.input_height &&
        manifest.input_channels == config_.input_channels;
}

PerceptionStatus TensorRtEngineRunner::initialize(const ModelManifest& manifest) noexcept {
    shutdown();
    if (!manifest_matches_contract(manifest)) {
        return PerceptionStatus::unsupported_format;
    }

    VerifiedEngineArtifact artifact{};
    const PerceptionStatus artifact_status = EngineArtifactVerifier::load_and_verify(
        manifest, EngineArtifactLimits{config_.max_engine_size_bytes}, artifact);
    if (artifact_status != PerceptionStatus::ok) {
        return artifact_status;
    }

    try {
        if (!impl_) { impl_ = std::make_unique<Impl>(config_); }
        const auto status = impl_->load(artifact, manifest, config_);
        if (status != PerceptionStatus::ok) { shutdown(); return status; }
        manifest_ = manifest;
        is_initialized_ = true;
        return PerceptionStatus::ok;
    } catch (...) {
        shutdown();
        return PerceptionStatus::device_error;
    }
}

PerceptionStatus TensorRtEngineRunner::warmup(std::uint32_t iterations) noexcept {
    if (!is_initialized_ || !impl_) { return PerceptionStatus::uninitialized; }
    if (has_device_fault_) { return PerceptionStatus::device_error; }
    if (impl_->pending_inference_) { return PerceptionStatus::invalid_argument; }
    if (is_warmed_up_) { return PerceptionStatus::ok; }
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
    const auto fault = [this]() noexcept {
        has_device_fault_ = true; ++total_faults_; return PerceptionStatus::device_error;
    };
    if (cudaSetDevice(impl_->device_id_) != cudaSuccess ||
        cudaMemsetAsync(impl_->d_input_, 0, impl_->input_size_bytes_, impl_->stream_) != cudaSuccess) {
        return fault();
    }
    // Real inference flushes TensorRT's lazy setup before graph capture. Synchronization is cold-only.
    for (std::uint32_t i = 0; i < std::max(iterations, 1U); ++i) {
        if (!impl_->context_->enqueueV3(impl_->stream_) ||
            cudaStreamSynchronize(impl_->stream_) != cudaSuccess) { return fault(); }
    }
    if (config_.enable_cuda_graph) {
        if (cudaStreamBeginCapture(impl_->stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            return fault();
        }
        const bool enqueued = impl_->context_->enqueueV3(impl_->stream_);
        const auto capture_status = cudaStreamEndCapture(impl_->stream_, &impl_->graph_);
        if (!enqueued || capture_status != cudaSuccess || !impl_->graph_ ||
            cudaGraphInstantiate(&impl_->graph_exec_, impl_->graph_, nullptr, nullptr, 0) != cudaSuccess) {
            return fault();
        }
        is_graph_captured_ = true;
    }
    if (impl_->logger_.error.load(std::memory_order_relaxed)) { return fault(); }
    is_warmed_up_ = true;
    return PerceptionStatus::ok;
#else
    static_cast<void>(iterations);
    return PerceptionStatus::backend_unavailable;
#endif
}

PerceptionStatus TensorRtEngineRunner::enqueue(const PerceptionRequest& request,
                                              InferenceTicket& out_ticket) noexcept {
    out_ticket = {};
    if (!is_initialized_ || !is_warmed_up_ || !impl_) { return PerceptionStatus::uninitialized; }
    if (has_device_fault_) { return PerceptionStatus::device_error; }
    if (!request.gpu_tensor_ptr || request.tensor_size_bytes != impl_->input_size_bytes_ ||
        request.frame_id == 0 || request.captured_at_ns <= 0 ||
        request.correlation_id.sequence_id != request.frame_id ||
        request.correlation_id.source_timestamp_ns != request.captured_at_ns) {
        return PerceptionStatus::invalid_argument;
    }
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
    const auto fault = [this]() noexcept {
        has_device_fault_ = true; ++total_faults_; return PerceptionStatus::device_error;
    };
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != impl_->device_id_) { return fault(); }
    // A single slot bounds GPU work. Caller retains its latest frame and retries after collection.
    if (impl_->pending_inference_) { return PerceptionStatus::inference_timeout; }
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, request.gpu_tensor_ptr) != cudaSuccess ||
        attributes.type != cudaMemoryTypeDevice || attributes.device != impl_->device_id_) {
        return PerceptionStatus::invalid_argument;
    }
    if (cudaMemcpyAsync(impl_->d_input_, request.gpu_tensor_ptr, impl_->input_size_bytes_,
                        cudaMemcpyDeviceToDevice, impl_->stream_) != cudaSuccess) { return fault(); }
    last_inference_ms_.reset();
    if (cudaEventRecord(impl_->inference_start_, impl_->stream_) != cudaSuccess) { return fault(); }
    const bool enqueued = is_graph_captured_
        ? cudaGraphLaunch(impl_->graph_exec_, impl_->stream_) == cudaSuccess
        : impl_->context_->enqueueV3(impl_->stream_);
    // The completion event covers inference AND pinned-host transfer; collection only queries it.
    if (!enqueued || cudaEventRecord(impl_->inference_end_, impl_->stream_) != cudaSuccess ||
        cudaMemcpyAsync(impl_->h_output_, impl_->d_output_, impl_->output_size_bytes_,
                                    cudaMemcpyDeviceToHost, impl_->stream_) != cudaSuccess ||
        cudaEventRecord(impl_->ready_event_, impl_->stream_) != cudaSuccess ||
        impl_->logger_.error.load(std::memory_order_relaxed)) { return fault(); }
    out_ticket.ticket_id = impl_->next_ticket_id_++;
    out_ticket.submitted_at_ns = request.captured_at_ns;
    out_ticket.native_gpu_event = impl_->ready_event_;
    impl_->active_ticket_ = out_ticket;
    impl_->active_request_ = request;
    impl_->pending_inference_ = true;
    ++total_inferences_;
    return PerceptionStatus::ok;
#else
    return PerceptionStatus::backend_unavailable;
#endif
}

PollResult TensorRtEngineRunner::try_collect(const InferenceTicket& ticket,
                                             bus::TargetObservationBatch& out_batch) noexcept {
    out_batch = {};
    if (!is_initialized_ || !impl_ || has_device_fault_) { return PollResult::error; }
    if (!impl_->pending_inference_ || ticket.ticket_id != impl_->active_ticket_.ticket_id ||
        ticket.native_gpu_event != impl_->active_ticket_.native_gpu_event ||
        ticket.submitted_at_ns != impl_->active_ticket_.submitted_at_ns) { return PollResult::empty; }
#if defined(AIM_HAS_TENSORRT_BACKEND) && AIM_HAS_TENSORRT_BACKEND == 1
    const auto status = cudaEventQuery(impl_->ready_event_);
    if (status == cudaErrorNotReady) { return PollResult::pending; }
    if (status != cudaSuccess || impl_->logger_.error.load(std::memory_order_relaxed)) {
        has_device_fault_ = true; ++total_faults_; return PollResult::error;
    }
    const auto* values = static_cast<const float*>(impl_->h_output_);
    const auto count = impl_->output_size_bytes_ / sizeof(float);
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            has_device_fault_ = true; ++total_faults_; return PollResult::error;
        }
    }
    float elapsed_ms = 0.0F;
    if (cudaEventElapsedTime(&elapsed_ms, impl_->inference_start_, impl_->inference_end_) != cudaSuccess ||
        !std::isfinite(elapsed_ms) || elapsed_ms < 0.0F) {
        has_device_fault_ = true; ++total_faults_; return PollResult::error;
    }
    last_inference_ms_ = elapsed_ms;
    impl_->yolo_decoder_.decode(values, count, impl_->active_request_.correlation_id,
                               impl_->active_request_.captured_at_ns, out_batch);
    impl_->pending_inference_ = false;
    impl_->active_request_ = {};
    return PollResult::ready;
#else
    return PollResult::error;
#endif
}

void TensorRtEngineRunner::shutdown() noexcept {
    last_inference_ms_.reset();
    is_initialized_ = false;
    is_warmed_up_ = false;
    is_graph_captured_ = false;
    has_device_fault_ = false;
    total_inferences_ = 0;
    manifest_ = ModelManifest{};
    if (impl_) {
        impl_->cleanup();
    }
}

} // namespace aim::perception
