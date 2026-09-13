// tests/cpp/test_config_manifest.cpp
// Native C++20 deterministic unit tests for OpenPrism Configuration and Run Manifests

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "aim/config/config_loader.hpp"
#include "aim/config/run_manifest.hpp"
#include "aim/config/sha256.hpp"
#include "aim/config/types.hpp"
#include "aim/config/validator.hpp"

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            return 1; \
        } \
    } while (false)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

using namespace aim::config;

// TC-CFG-01: Load and verify default base configuration
int test_default_base_config_load() {
    ResolvedConfig cfg{};
    ASSERT_EQ(cfg.schema_version, 1u);
    ASSERT_EQ(cfg.runtime.internal_deadline_ms, 10.0);
    ASSERT_EQ(cfg.runtime.target_p99_ms, 6.0);
    ASSERT_EQ(cfg.capture.backend, "dxgi");
    ASSERT_EQ(cfg.capture.source_width, 1920u);
    ASSERT_EQ(cfg.capture.source_height, 1080u);
    ASSERT_EQ(cfg.perception.plugin, "yolo11n_aimlabs");
    ASSERT_EQ(cfg.perception.input_width, 640u);
    ASSERT_EQ(cfg.perception.input_height, 384u);
    ASSERT_EQ(cfg.tracking.association, "hungarian");
    ASSERT_EQ(cfg.tracking.models.size(), 3u);
    ASSERT_EQ(cfg.policy.plugin, "deterministic_utility");
    ASSERT_EQ(cfg.policy.horizon_targets, 3u);
    ASSERT_EQ(cfg.trajectory.small_error_mode, "direct_feedforward");
    ASSERT_EQ(cfg.actuator.backend, "sendinput");
    ASSERT_EQ(cfg.actuator.scheduler_hz, 1000u);
    ASSERT_TRUE(cfg.safety.require_foreground_match);
    ASSERT_TRUE(cfg.safety.fail_closed);

    const auto j = ConfigLoader::to_json(cfg);
    ASSERT_TRUE(j.is_object());
    ASSERT_EQ(j["schema_version"], 1);

    std::vector<std::string> errors;
    ASSERT_TRUE(ConfigLoader::validate_json_schema(j, errors));

    return 0;
}

// TC-CFG-02: Deterministic 7-tier layer merge hierarchy
int test_deterministic_layer_merge_hierarchy() {
    const std::string base_yaml = R"(
schema_version: 1
runtime:
  internal_deadline_ms: 10.0
  target_p99_ms: 6.0
  warmup_iterations: 200
  allocation_audit: true
capture:
  backend: "dxgi"
  fallback: "wgc"
  source_width: 1920
  source_height: 1080
  pixel_format: "bgra8_sdr"
  buffers: 2
  latest_only: true
  stale_after_ms: 12.0
perception:
  plugin: "yolo11n_aimlabs"
  precision: "fp16"
  batch: 1
  input_width: 640
  input_height: 384
  max_targets: 64
  cuda_graph: true
  confidence_floor: 0.20
tracking:
  models: ["stationary", "constant_velocity", "constant_acceleration"]
  association: "hungarian"
  max_missed_frames: 3
  immediate_confidence: 0.85
  require_second_observation_below: 0.85
  engage_if_uncertainty_within_radius: true
policy:
  plugin: "deterministic_utility"
  objective: "raw_score"
  horizon_targets: 3
  switch_hysteresis: 0.08
trajectory:
  small_error_mode: "direct_feedforward"
  large_error_mode: "jerk_limited"
  terminal_mode: "critically_damped_pd"
  optional_profile: "minimum_jerk"
actuator:
  backend: "sendinput"
  scheduler_hz: 1000
  relative_counts: true
  cancel_superseded: true
safety:
  require_foreground_match: true
  require_emergency_stop: true
  fail_closed: true
)";

    const std::string hw_yaml = R"(
capture:
  stale_after_ms: 10.0
)";

    const std::string domain_yaml = R"(
policy:
  horizon_targets: 4
)";

    const std::string perception_yaml = R"(
perception:
  confidence_floor: 0.30
)";

    const std::string policy_yaml = R"(
policy:
  switch_hysteresis: 0.12
)";

    const std::string actuator_yaml = R"(
actuator:
  scheduler_hz: 2000
)";

    const std::string override_yaml = R"(
runtime:
  target_p99_ms: 4.5
)";

    std::vector<std::string> layers = {
        base_yaml, hw_yaml, domain_yaml, perception_yaml, policy_yaml, actuator_yaml, override_yaml
    };
    std::vector<std::string> layer_names = {
        "base", "hardware", "domain", "perception", "policy", "actuator", "override"
    };

    const auto res = ConfigLoader::load_from_yaml_strings(layers, layer_names);
    ASSERT_TRUE(res.success);
    ASSERT_EQ(res.config.capture.stale_after_ms, 10.0);
    ASSERT_EQ(res.config.policy.horizon_targets, 4u);
    ASSERT_EQ(res.config.perception.confidence_floor, 0.30);
    ASSERT_EQ(res.config.policy.switch_hysteresis, 0.12);
    ASSERT_EQ(res.config.actuator.scheduler_hz, 2000u);
    ASSERT_EQ(res.config.runtime.target_p99_ms, 4.5);
    ASSERT_EQ(res.source_layers.size(), 7u);

    return 0;
}

// TC-CFG-03: Partial subtree deep merge preserves sibling keys
int test_partial_subtree_deep_merge() {
    nlohmann::json base = {
        {"capture", {
            {"backend", "dxgi"},
            {"source_width", 1920},
            {"source_height", 1080},
            {"stale_after_ms", 12.0}
        }}
    };

    nlohmann::json overlay = {
        {"capture", {
            {"stale_after_ms", 8.0}
        }}
    };

    ConfigLoader::merge_json(base, overlay);
    ASSERT_EQ(base["capture"]["stale_after_ms"], 8.0);
    ASSERT_EQ(base["capture"]["backend"], "dxgi");
    ASSERT_EQ(base["capture"]["source_width"], 1920);
    ASSERT_EQ(base["capture"]["source_height"], 1080);

    return 0;
}

// TC-CFG-04: Array replacement semantics
int test_array_replacement_semantics() {
    nlohmann::json base = {
        {"tracking", {
            {"models", nlohmann::json::array({"stationary", "constant_velocity", "constant_acceleration"})}
        }}
    };

    nlohmann::json overlay = {
        {"tracking", {
            {"models", nlohmann::json::array({"constant_velocity"})}
        }}
    };

    ConfigLoader::merge_json(base, overlay);
    ASSERT_EQ(base["tracking"]["models"].size(), 1u);
    ASSERT_EQ(base["tracking"]["models"][0], "constant_velocity");

    return 0;
}

// TC-CFG-05: Source layers provenance tracking
int test_source_layers_provenance_tracking() {
    std::vector<std::string> layers = {
        "schema_version: 1\nruntime:\n  internal_deadline_ms: 10.0\n  target_p99_ms: 6.0\n  warmup_iterations: 200\n  allocation_audit: true\ncapture:\n  backend: \"dxgi\"\n  fallback: \"wgc\"\n  source_width: 1920\n  source_height: 1080\n  pixel_format: \"bgra8_sdr\"\n  buffers: 2\n  latest_only: true\n  stale_after_ms: 12.0\nperception:\n  plugin: \"yolo11n\"\n  precision: \"fp16\"\n  batch: 1\n  input_width: 640\n  input_height: 384\n  max_targets: 64\n  cuda_graph: true\n  confidence_floor: 0.2\ntracking:\n  models: [\"stationary\"]\n  association: \"hungarian\"\n  max_missed_frames: 3\n  immediate_confidence: 0.85\n  require_second_observation_below: 0.85\n  engage_if_uncertainty_within_radius: true\npolicy:\n  plugin: \"det\"\n  objective: \"raw_score\"\n  horizon_targets: 3\n  switch_hysteresis: 0.08\ntrajectory:\n  small_error_mode: \"direct_feedforward\"\n  large_error_mode: \"jerk_limited\"\n  terminal_mode: \"critically_damped_pd\"\n  optional_profile: \"minimum_jerk\"\nactuator:\n  backend: \"sendinput\"\n  scheduler_hz: 1000\n  relative_counts: true\n  cancel_superseded: true\nsafety:\n  require_foreground_match: true\n  require_emergency_stop: true\n  fail_closed: true\n",
        "runtime:\n  target_p99_ms: 5.0\n"
    };
    std::vector<std::string> names = {"configs/base.yaml", "configs/overrides/fast.yaml"};

    const auto res = ConfigLoader::load_from_yaml_strings(layers, names);
    ASSERT_TRUE(res.success);
    ASSERT_EQ(res.source_layers.size(), 2u);
    ASSERT_EQ(res.source_layers[0], "configs/base.yaml");
    ASSERT_EQ(res.source_layers[1], "configs/overrides/fast.yaml");

    return 0;
}

// TC-CFG-06: Unknown root key rejection
int test_unknown_root_key_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j["unrecognized_setting"] = 123;

    std::vector<std::string> errors;
    const bool valid = ConfigLoader::validate_json_schema(j, errors);
    ASSERT_FALSE(valid);
    ASSERT_TRUE(!errors.empty());

    return 0;
}

// TC-CFG-07: Unknown subdomain key rejection
int test_unknown_subdomain_key_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j["capture"]["invalid_buffer_mode"] = "async";

    std::vector<std::string> errors;
    const bool valid = ConfigLoader::validate_json_schema(j, errors);
    ASSERT_FALSE(valid);
    ASSERT_TRUE(!errors.empty());

    return 0;
}

// TC-CFG-08: Typo key rejection
int test_typo_key_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j["runtime"].erase("target_p99_ms");
    j["runtime"]["target_p99"] = 6.0;

    std::vector<std::string> errors;
    const bool valid = ConfigLoader::validate_json_schema(j, errors);
    ASSERT_FALSE(valid);
    ASSERT_TRUE(!errors.empty());

    return 0;
}

// TC-CFG-09: Structural bounds violation fatal failure
int test_structural_bounds_violation_failure() {
    ResolvedConfig cfg{};
    cfg.capture.source_width = 0;

    std::vector<std::string> warnings, errors;
    const bool ok = Validator::validate_and_clamp(cfg, warnings, errors);
    ASSERT_FALSE(ok);
    ASSERT_TRUE(!errors.empty());

    cfg.capture.source_width = 1920;
    cfg.runtime.internal_deadline_ms = -1.0;
    errors.clear();
    ASSERT_FALSE(Validator::validate_and_clamp(cfg, warnings, errors));

    return 0;
}

// TC-CFG-10: Safe tuning value clamping upper
int test_safe_tuning_value_clamping_upper() {
    ResolvedConfig cfg{};
    cfg.perception.confidence_floor = 1.45;

    std::vector<std::string> warnings, errors;
    const bool ok = Validator::validate_and_clamp(cfg, warnings, errors);
    ASSERT_TRUE(ok);
    ASSERT_EQ(cfg.perception.confidence_floor, 1.0);
    ASSERT_TRUE(!warnings.empty());

    return 0;
}

// TC-CFG-11: Safe tuning value clamping lower
int test_safe_tuning_value_clamping_lower() {
    ResolvedConfig cfg{};
    cfg.perception.confidence_floor = -0.20;
    cfg.tracking.immediate_confidence = -0.50;

    std::vector<std::string> warnings, errors;
    const bool ok = Validator::validate_and_clamp(cfg, warnings, errors);
    ASSERT_TRUE(ok);
    ASSERT_EQ(cfg.perception.confidence_floor, 0.0);
    ASSERT_EQ(cfg.tracking.immediate_confidence, 0.0);
    ASSERT_EQ(warnings.size(), 2u);

    return 0;
}

// TC-CFG-12: Hot reload allowlist enforcement
int test_hot_reload_allowlist_enforcement() {
    ResolvedConfig current{};
    ResolvedConfig safe_update = current;
    safe_update.perception.confidence_floor = 0.35;
    safe_update.policy.switch_hysteresis = 0.15;

    const auto safe_res = Validator::check_hot_reload(current, safe_update);
    ASSERT_TRUE(safe_res.is_allowed());
    ASSERT_EQ(safe_res.status, HotReloadStatus::Allowed);
    ASSERT_EQ(safe_res.modified_allowed_fields.size(), 2u);

    ResolvedConfig struct_update = current;
    struct_update.capture.backend = "wgc";
    const auto struct_res = Validator::check_hot_reload(current, struct_update);
    ASSERT_FALSE(struct_res.is_allowed());
    ASSERT_EQ(struct_res.status, HotReloadStatus::RequiresQuiescentRestart);
    ASSERT_TRUE(!struct_res.conflicting_structural_fields.empty());

    return 0;
}

// TC-CFG-13: Missing schema version rejection
int test_missing_schema_version_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j.erase("schema_version");

    std::vector<std::string> errors;
    ASSERT_FALSE(ConfigLoader::validate_json_schema(j, errors));

    return 0;
}

// TC-CFG-14: Missing domain section rejection
int test_missing_domain_section_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j.erase("safety");

    std::vector<std::string> errors;
    ASSERT_FALSE(ConfigLoader::validate_json_schema(j, errors));

    return 0;
}

// TC-CFG-15: Type mismatch rejection
int test_type_mismatch_rejection() {
    ResolvedConfig cfg{};
    auto j = ConfigLoader::to_json(cfg);
    j["capture"]["source_width"] = "1920";

    std::vector<std::string> errors;
    ASSERT_FALSE(ConfigLoader::validate_json_schema(j, errors));

    return 0;
}

// TC-MAN-01: Run manifest creation and completeness
int test_run_manifest_creation_and_completeness() {
    ResolvedConfig cfg{};
    const auto j_cfg = ConfigLoader::to_json(cfg);
    const std::string cfg_hash = Sha256::hash_string(j_cfg.dump());

    GitRevision git_rev{.commit_hash = "0123456789abcdef", .branch = "main", .is_dirty = false, .describe = "v0.1.0"};
    EnvironmentMetadata env{.os_name = "Windows", .os_version = "10.0.22631", .architecture = "x86_64", .processor = "AMD", .hostname = "TEST-PC"};
    std::map<std::string, std::string> schemas = {{"schemas/config/root.schema.json", "abcdef123456"}};
    std::vector<ArtifactDescriptor> artifacts = {{.name = "yolo11n.onnx", .path = "models/yolo11n.onnx", .sha256 = "1111222233334444", .size_bytes = 1048576, .type = "model_onnx"}};
    ExecutionMetadata exec{.mode = "simulation", .internal_deadline_ms = 10.0, .target_p99_ms = 6.0, .warmup_iterations_completed = 200, .notes = "test"};

    const auto manifest = RunManifest::create(
        cfg,
        cfg_hash,
        {"configs/base.yaml"},
        "12345678-1234-4000-8000-123456789012",
        "2026-08-23T21:00:00Z",
        git_rev,
        env,
        schemas,
        artifacts,
        exec
    );

    const auto j_man = manifest.to_json();
    ASSERT_EQ(j_man["schema_version"], 1);
    ASSERT_EQ(j_man["manifest_id"], "12345678-1234-4000-8000-123456789012");
    ASSERT_EQ(j_man["git_info"]["commit_hash"], "0123456789abcdef");
    ASSERT_EQ(j_man["environment"]["os_name"], "Windows");
    ASSERT_EQ(j_man["config"]["config_sha256"], cfg_hash);
    ASSERT_EQ(j_man["artifacts"].size(), 1u);

    return 0;
}

// TC-MAN-02: Config SHA-256 stability
int test_config_sha256_stability() {
    ResolvedConfig cfg{};
    const auto j1 = ConfigLoader::to_json(cfg);
    const std::string hash1 = Sha256::hash_string(j1.dump());

    const auto j2 = ConfigLoader::to_json(cfg);
    const std::string hash2 = Sha256::hash_string(j2.dump());

    ASSERT_EQ(hash1, hash2);
    ASSERT_EQ(hash1.length(), 64u);

    return 0;
}

// TC-PAR-01: Cross-language config hash parity with Python
int test_cross_language_config_hash_parity() {
    ResolvedConfig cfg{};
    const auto j = ConfigLoader::to_json(cfg);
    const std::string canonical_json_str = j.dump();
    const std::string cpp_hash = Sha256::hash_string(canonical_json_str);

    // Expected golden SHA-256 hash matching Python's canonical JSON dump of base.yaml
    const std::string expected_hash = "1b41a4d4af3dce9db93dd17131b1b521acfbe70268f4ae5f2f7ccd6f2f4f54a7";
    ASSERT_EQ(cpp_hash, expected_hash);

    return 0;
}

// TC-MAN-03: Manifest tamper detection
int test_manifest_tamper_detection() {
    ResolvedConfig cfg{};
    const auto j_cfg = ConfigLoader::to_json(cfg);
    const std::string cfg_hash = Sha256::hash_string(j_cfg.dump());

    const auto manifest = RunManifest::create(cfg, cfg_hash, {"configs/base.yaml"});
    auto j_man = manifest.to_json();

    std::string err;
    ASSERT_TRUE(RunManifest::verify(j_man, err));

    // Tamper with target_p99_ms in resolved_config
    j_man["config"]["resolved_config"]["runtime"]["target_p99_ms"] = 3.0;
    ASSERT_FALSE(RunManifest::verify(j_man, err));
    ASSERT_TRUE(!err.empty());

    return 0;
}

int main() {
    std::cout << "Running Native C++20 Configuration & Run Manifests Test Suite..." << std::endl;

    if (test_default_base_config_load() != 0) return 1;
    if (test_deterministic_layer_merge_hierarchy() != 0) return 1;
    if (test_partial_subtree_deep_merge() != 0) return 1;
    if (test_array_replacement_semantics() != 0) return 1;
    if (test_source_layers_provenance_tracking() != 0) return 1;
    if (test_unknown_root_key_rejection() != 0) return 1;
    if (test_unknown_subdomain_key_rejection() != 0) return 1;
    if (test_typo_key_rejection() != 0) return 1;
    if (test_structural_bounds_violation_failure() != 0) return 1;
    if (test_safe_tuning_value_clamping_upper() != 0) return 1;
    if (test_safe_tuning_value_clamping_lower() != 0) return 1;
    if (test_hot_reload_allowlist_enforcement() != 0) return 1;
    if (test_missing_schema_version_rejection() != 0) return 1;
    if (test_missing_domain_section_rejection() != 0) return 1;
    if (test_type_mismatch_rejection() != 0) return 1;
    if (test_run_manifest_creation_and_completeness() != 0) return 1;
    if (test_config_sha256_stability() != 0) return 1;
    if (test_cross_language_config_hash_parity() != 0) return 1;
    if (test_manifest_tamper_detection() != 0) return 1;

    std::cout << "All C++20 configuration & manifest tests PASSED successfully." << std::endl;
    return 0;
}
