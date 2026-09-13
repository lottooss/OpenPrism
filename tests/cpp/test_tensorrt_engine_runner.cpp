#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "aim/config/sha256.hpp"
#include "aim/interfaces/perception_engine.hpp"
#include "aim/perception/engine_artifact.hpp"
#include "aim/perception/mock_perception_engine.hpp"
#include "aim/perception/tensorrt_engine_runner.hpp"

#define TEST_ASSERT(condition) do { \
    if (!(condition)) { \
        std::cerr << "Assertion failed: (" #condition ") at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(1); \
    } \
} while (false)

namespace {

using namespace aim;
using namespace aim::perception;

class TemporaryEngineFile final {
public:
    explicit TemporaryEngineFile(std::string_view contents) {
        path_ = std::filesystem::temp_directory_path() /
            ("aimlabs-engine-preflight-" + config::Sha256::hash_string(contents) + ".engine");
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        TEST_ASSERT(output.good());
    }

    ~TemporaryEngineFile() {
        std::error_code error{};
        static_cast<void>(std::filesystem::remove(path_, error));
    }

    TemporaryEngineFile(const TemporaryEngineFile&) = delete;
    TemporaryEngineFile& operator=(const TemporaryEngineFile&) = delete;

    [[nodiscard]] std::string path_string() const { return path_.string(); }

private:
    std::filesystem::path path_{};
};

ModelManifest trusted_manifest(const TemporaryEngineFile& artifact, std::string_view contents) {
    ModelManifest manifest{};
    manifest.model_path = artifact.path_string();
    manifest.engine_sha256 = config::Sha256::hash_string(contents);
    return manifest;
}

void test_engine_artifact_preflight() {
    constexpr std::string_view contents = "trusted-engine-fixture";
    TemporaryEngineFile artifact(contents);
    ModelManifest manifest = trusted_manifest(artifact, contents);

    TEST_ASSERT(config::Sha256::hash_string("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    TEST_ASSERT(EngineArtifactVerifier::is_sha256_hex(manifest.engine_sha256));
    TEST_ASSERT(!EngineArtifactVerifier::is_sha256_hex("not-a-hash"));

    VerifiedEngineArtifact verified{};
    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        manifest, EngineArtifactLimits{}, verified) == PerceptionStatus::ok);
    TEST_ASSERT(verified.bytes().size() == contents.size());

    ModelManifest mismatch = manifest;
    mismatch.engine_sha256[0] = mismatch.engine_sha256[0] == '0' ? '1' : '0';
    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        mismatch, EngineArtifactLimits{}, verified) == PerceptionStatus::integrity_error);
    TEST_ASSERT(verified.empty());

    ModelManifest malformed = manifest;
    malformed.engine_sha256 = "short";
    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        malformed, EngineArtifactLimits{}, verified) == PerceptionStatus::invalid_argument);

    ModelManifest missing = manifest;
    missing.model_path += ".missing";
    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        missing, EngineArtifactLimits{}, verified) == PerceptionStatus::artifact_not_found);

    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        manifest, EngineArtifactLimits{contents.size() - 1U}, verified) ==
        PerceptionStatus::invalid_argument);

    std::transform(manifest.engine_sha256.begin(), manifest.engine_sha256.end(),
                   manifest.engine_sha256.begin(), [](char character) {
                       return static_cast<char>(
                           std::toupper(static_cast<unsigned char>(character)));
                   });
    TEST_ASSERT(EngineArtifactVerifier::load_and_verify(
        manifest, EngineArtifactLimits{}, verified) == PerceptionStatus::ok);
}

void test_production_runner_fails_closed_without_backend() {
    constexpr std::string_view contents = "verified-tensorrt-plan-fixture";
    TemporaryEngineFile artifact(contents);
    ModelManifest manifest = trusted_manifest(artifact, contents);
    TensorRtEngineRunner runner{};

    const auto status = runner.initialize(manifest);
    // Hash-valid bytes are not necessarily an executable engine, even when an SDK is installed.
    TEST_ASSERT(status != PerceptionStatus::ok);
    if (!TensorRtEngineRunner::is_backend_compiled()) {
        TEST_ASSERT(status == PerceptionStatus::backend_unavailable);
    } else {
        TEST_ASSERT(status == PerceptionStatus::unsupported_format ||
                    status == PerceptionStatus::device_error ||
                    status == PerceptionStatus::backend_unavailable);
    }
    TEST_ASSERT(!runner.is_initialized());
    TEST_ASSERT(!runner.last_inference_ms().has_value());

    InferenceTicket ticket{.ticket_id = 99};
    TEST_ASSERT(runner.enqueue(PerceptionRequest{}, ticket) == PerceptionStatus::uninitialized);
    TEST_ASSERT(ticket.ticket_id == 0);

    ModelManifest wrong_backend = manifest;
    wrong_backend.runtime_backend = "mock";
    TEST_ASSERT(runner.initialize(wrong_backend) == PerceptionStatus::unsupported_format);

    ModelManifest wrong_shape = manifest;
    wrong_shape.input_width = 1280;
    TEST_ASSERT(runner.initialize(wrong_shape) == PerceptionStatus::unsupported_format);

    ModelManifest tampered = manifest;
    tampered.engine_sha256.assign(64U, '0');
    TEST_ASSERT(runner.initialize(tampered) == PerceptionStatus::integrity_error);
}

void test_explicit_mock_latest_wins_and_decodes() {
    MockTensorRtEngineRunner runner{};
    ModelManifest manifest{};
    TEST_ASSERT(runner.initialize(manifest) == PerceptionStatus::unsupported_format);
    manifest.runtime_backend = "mock";
    TEST_ASSERT(runner.initialize(manifest) == PerceptionStatus::ok);
    TEST_ASSERT(runner.warmup(3) == PerceptionStatus::ok);

    const std::array<DecodedDetection, 3> detections{{
        {960.0F, 540.0F, 60.0F, 60.0F, 30.0F, 0.95F, 0},
        {400.0F, 300.0F, 50.0F, 50.0F, 25.0F, 0.88F, 0},
        {100.0F, 100.0F, 20.0F, 20.0F, 10.0F, 0.10F, 0}
    }};
    TEST_ASSERT(runner.set_detections(detections));

    PerceptionRequest first{};
    first.frame_id = 41;
    first.captured_at_ns = 900'000'000;
    InferenceTicket stale_ticket{};
    TEST_ASSERT(runner.enqueue(first, stale_ticket) == PerceptionStatus::ok);

    PerceptionRequest latest{};
    latest.frame_id = 42;
    latest.captured_at_ns = 1'000'000'000;
    latest.correlation_id.sequence_id = 42;
    latest.correlation_id.source_timestamp_ns = latest.captured_at_ns;
    InferenceTicket latest_ticket{};
    TEST_ASSERT(runner.enqueue(latest, latest_ticket) == PerceptionStatus::ok);

    bus::TargetObservationBatch batch{};
    TEST_ASSERT(runner.try_collect(stale_ticket, batch) == PollResult::empty);
    TEST_ASSERT(runner.try_collect(latest_ticket, batch) == PollResult::ready);
    TEST_ASSERT(batch.frame_id == 42);
    TEST_ASSERT(batch.target_count == 2);
    TEST_ASSERT(batch.targets[0].center_px.x == 960.0F);
    TEST_ASSERT(batch.targets[0].center_norm.x == 0.0F);
    TEST_ASSERT(batch.targets[0].effective_radius_px == 30.0F);
    TEST_ASSERT(batch.targets[1].confidence == 0.88F);
    TEST_ASSERT(runner.try_collect(latest_ticket, batch) == PollResult::empty);

    std::array<DecodedDetection, bus::kMaxObservations + 1U> too_many{};
    TEST_ASSERT(!runner.set_detections(too_many));
}

void test_explicit_mock_sticky_fault() {
    MockTensorRtEngineRunner runner{};
    ModelManifest manifest{};
    manifest.runtime_backend = "mock";
    TEST_ASSERT(runner.initialize(manifest) == PerceptionStatus::ok);
    runner.inject_device_fault();

    InferenceTicket ticket{};
    TEST_ASSERT(runner.enqueue(PerceptionRequest{}, ticket) == PerceptionStatus::device_error);
    TEST_ASSERT(runner.total_faults() == 1);
    bus::TargetObservationBatch batch{};
    TEST_ASSERT(runner.try_collect(ticket, batch) == PollResult::error);

    runner.clear_device_fault();
    TEST_ASSERT(runner.warmup(1) == PerceptionStatus::ok);
    TEST_ASSERT(runner.enqueue(PerceptionRequest{}, ticket) == PerceptionStatus::ok);
}

void test_production_runner_lifecycle_and_moves() {
    TensorRtRunnerConfig config{};
    config.model_name = "yolo11n_test";
    config.model_version = "2.0.0";
    config.input_width = 640;
    config.input_height = 384;
    config.input_channels = 3;
    config.confidence_threshold = 0.30f;
    config.max_detections = 32;

    TensorRtEngineRunner runner(config);
    const ModelContract contract = runner.contract();
    TEST_ASSERT(contract.model_name == "yolo11n_test");
    TEST_ASSERT(contract.model_version == "2.0.0");
    TEST_ASSERT(contract.input_width_px == 640);
    TEST_ASSERT(contract.input_height_px == 384);
    TEST_ASSERT(contract.input_channels == 3);
    TEST_ASSERT(contract.max_detections == 32);
    TEST_ASSERT(contract.confidence_threshold == 0.30f);

    TEST_ASSERT(!runner.is_initialized());
    TEST_ASSERT(!runner.last_inference_ms().has_value());
    TEST_ASSERT(!runner.is_warmed_up());
    TEST_ASSERT(!runner.is_graph_captured());
    TEST_ASSERT(!runner.has_device_fault());
    TEST_ASSERT(runner.total_inferences() == 0);
    TEST_ASSERT(runner.total_faults() == 0);

    // Test move construction
    TensorRtEngineRunner moved_runner(std::move(runner));
    TEST_ASSERT(moved_runner.contract().model_name == "yolo11n_test");
    TEST_ASSERT(!runner.is_initialized());
    TEST_ASSERT(!runner.last_inference_ms().has_value());

    // Test move assignment
    TensorRtEngineRunner assigned_runner{};
    assigned_runner = std::move(moved_runner);
    TEST_ASSERT(assigned_runner.contract().model_name == "yolo11n_test");
}

} // namespace

int main() {
    std::cout << "Running trusted engine preflight and explicit mock tests...\n";
    test_engine_artifact_preflight();
    test_production_runner_fails_closed_without_backend();
    test_production_runner_lifecycle_and_moves();
    test_explicit_mock_latest_wins_and_decodes();
    test_explicit_mock_sticky_fault();
    std::cout << "All trusted engine preflight tests passed.\n";
    return 0;
}
