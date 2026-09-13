// include/aim/plugin/plugin_descriptor.hpp
#pragma once

#include <cstdint>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "aim/config/sha256.hpp"
#include "aim/plugin/plugin_abi.h"

namespace aim::plugin {

inline constexpr uint32_t fourcc_to_uint32(std::string_view token) noexcept {
    if (token.size() != 4) return 0;
    return static_cast<uint32_t>(static_cast<uint8_t>(token[0])) |
          (static_cast<uint32_t>(static_cast<uint8_t>(token[1])) << 8) |
          (static_cast<uint32_t>(static_cast<uint8_t>(token[2])) << 16) |
          (static_cast<uint32_t>(static_cast<uint8_t>(token[3])) << 24);
}

inline std::string uint32_to_fourcc(uint32_t code) {
    std::string s(4, '\0');
    s[0] = static_cast<char>(code & 0xFF);
    s[1] = static_cast<char>((code >> 8) & 0xFF);
    s[2] = static_cast<char>((code >> 16) & 0xFF);
    s[3] = static_cast<char>((code >> 24) & 0xFF);
    return s;
}

inline AimPluginKind string_to_plugin_kind(std::string_view str) noexcept {
    if (str == "perception") return AIM_PLUGIN_KIND_PERCEPTION;
    if (str == "tracker" || str == "tracking") return AIM_PLUGIN_KIND_TRACKER;
    if (str == "policy") return AIM_PLUGIN_KIND_POLICY;
    if (str == "trajectory") return AIM_PLUGIN_KIND_TRAJECTORY;
    if (str == "actuator") return AIM_PLUGIN_KIND_ACTUATOR;
    if (str == "scenario") return AIM_PLUGIN_KIND_SCENARIO;
    return AIM_PLUGIN_KIND_UNKNOWN;
}

inline std::string plugin_kind_to_string(AimPluginKind kind) {
    switch (kind) {
        case AIM_PLUGIN_KIND_PERCEPTION: return "perception";
        case AIM_PLUGIN_KIND_TRACKER: return "tracker";
        case AIM_PLUGIN_KIND_POLICY: return "policy";
        case AIM_PLUGIN_KIND_TRAJECTORY: return "trajectory";
        case AIM_PLUGIN_KIND_ACTUATOR: return "actuator";
        case AIM_PLUGIN_KIND_SCENARIO: return "scenario";
        default: return "unknown";
    }
}

struct SchemaContractInfo {
    std::string identifier{"NONE"};
    uint32_t major_min{1};
    uint32_t major_max{1};
};

struct PluginContracts {
    SchemaContractInfo input_schema{};
    SchemaContractInfo output_schema{};
};

struct PluginChecksum {
    std::string sha256{};
    std::size_t size_bytes{0};
};

struct PluginResourceRequirements {
    uint64_t max_latency_budget_ns{6'000'000};
    uint32_t max_output_buffer_bytes{1024 * 1024};
    uint64_t gpu_vram_bytes{0};
    bool requires_cuda{false};
};

struct PluginModelEntry {
    std::string name{};
    std::string path{};
    std::string sha256{};
    std::string framework{"tensorrt"};
    std::string precision{"fp16"};
};

struct PluginManifest {
    uint32_t schema_version{1};
    std::string name{};
    std::string version{"1.0.0"};
    uint32_t abi_version{AIM_PLUGIN_ABI_VERSION_V1};
    AimPluginKind kind{AIM_PLUGIN_KIND_UNKNOWN};
    std::string author{};
    std::string license{"PolyForm-Noncommercial-1.0.0"};
    std::string description{};
    std::string entry_point{};
    PluginChecksum checksum{};
    PluginContracts contracts{};
    std::vector<std::string> capabilities{};
    std::vector<PluginModelEntry> models{};
    PluginResourceRequirements resource_requirements{};

    [[nodiscard]] bool is_abi_compatible(uint32_t host_abi = AIM_PLUGIN_ABI_VERSION_V1) const noexcept {
        return AIM_PLUGIN_ABI_MAJOR(abi_version) == AIM_PLUGIN_ABI_MAJOR(host_abi);
    }

    [[nodiscard]] bool verify_binary_hash(const std::string& binary_path) const {
        if (checksum.sha256.empty() || checksum.size_bytes == 0) return false;
        std::error_code error{};
        const auto actual_size = std::filesystem::file_size(binary_path, error);
        if (error || actual_size != checksum.size_bytes) return false;
        const std::string actual_hash = aim::config::Sha256::hash_file(binary_path);
        return actual_hash == checksum.sha256;
    }
};

class PluginManifestParser {
public:
    static bool parse_json(const nlohmann::json& j, PluginManifest& out_manifest, std::string& out_error) {
        try {
            if (!j.is_object()) {
                out_error = "Manifest root must be an object";
                return false;
            }


            static constexpr std::array<std::string_view, 14> allowed_root_fields = {
                "schema_version", "name", "version", "abi_version", "kind", "author",
                "license", "description", "entry_point", "checksum", "contracts",
                "capabilities", "models", "resource_requirements"
            };
            for (const auto& [key, value] : j.items()) {
                (void)value;
                if (std::find(allowed_root_fields.begin(), allowed_root_fields.end(), key) ==
                    allowed_root_fields.end()) {
                    out_error = "Unknown manifest field: " + key;
                    return false;
                }
            }

            // Check required fields
            const std::vector<std::string> required = {
                "schema_version", "name", "version", "abi_version", "kind",
                "author", "license", "entry_point", "checksum", "contracts", "capabilities"
            };
            for (const auto& req : required) {
                if (!j.contains(req)) {
                    out_error = "Missing required field: " + req;
                    return false;
                }
            }

            out_manifest.schema_version = j.at("schema_version").get<uint32_t>();
            if (out_manifest.schema_version != 1) {
                out_error = "Unsupported schema_version: " + std::to_string(out_manifest.schema_version);
                return false;
            }

            out_manifest.name = j.at("name").get<std::string>();
            out_manifest.version = j.at("version").get<std::string>();
            out_manifest.abi_version = j.at("abi_version").get<uint32_t>();
            out_manifest.kind = string_to_plugin_kind(j.at("kind").get<std::string>());
            if (out_manifest.kind == AIM_PLUGIN_KIND_UNKNOWN) {
                out_error = "Unknown plugin kind: " + j.at("kind").get<std::string>();
                return false;
            }

            out_manifest.author = j.at("author").get<std::string>();
            out_manifest.license = j.at("license").get<std::string>();
            if (j.contains("description") && j["description"].is_string()) {
                out_manifest.description = j["description"].get<std::string>();
            }
            out_manifest.entry_point = j.at("entry_point").get<std::string>();
            if (out_manifest.name.empty() || out_manifest.version.empty() ||
                out_manifest.author.empty() || out_manifest.license.empty() ||
                out_manifest.entry_point.empty() ||
                out_manifest.entry_point.find('/') != std::string::npos ||
                out_manifest.entry_point.find('\\') != std::string::npos) {
                out_error = "Manifest contains an empty field or unsafe entry_point";
                return false;
            }

            const auto& chk = j.at("checksum");
            out_manifest.checksum.sha256 = chk.at("sha256").get<std::string>();
            out_manifest.checksum.size_bytes = chk.at("size_bytes").get<std::size_t>();
            const bool checksum_valid = out_manifest.checksum.sha256.size() == 64 &&
                std::all_of(out_manifest.checksum.sha256.begin(),
                            out_manifest.checksum.sha256.end(),
                            [](unsigned char value) { return std::isxdigit(value) != 0; });
            if (!checksum_valid || out_manifest.checksum.size_bytes == 0) {
                out_error = "Invalid plugin checksum or size";
                return false;
            }
            std::transform(out_manifest.checksum.sha256.begin(),
                           out_manifest.checksum.sha256.end(),
                           out_manifest.checksum.sha256.begin(),
                           [](unsigned char value) {
                               return static_cast<char>(std::tolower(value));
                           });

            const auto& contracts = j.at("contracts");
            const auto& in_sc = contracts.at("input_schema");
            out_manifest.contracts.input_schema.identifier = in_sc.at("identifier").get<std::string>();
            out_manifest.contracts.input_schema.major_min = in_sc.at("major_min").get<uint32_t>();
            out_manifest.contracts.input_schema.major_max = in_sc.at("major_max").get<uint32_t>();

            const auto& out_sc = contracts.at("output_schema");
            out_manifest.contracts.output_schema.identifier = out_sc.at("identifier").get<std::string>();
            out_manifest.contracts.output_schema.major_min = out_sc.at("major_min").get<uint32_t>();
            out_manifest.contracts.output_schema.major_max = out_sc.at("major_max").get<uint32_t>();

            static constexpr std::array<std::string_view, 7> schema_identifiers = {
                "AFR1", "AOB1", "ATT1", "AAI1", "AAC1", "ATE1", "NONE"
            };
            const auto schema_valid = [](const SchemaContractInfo& schema) {
                return std::find(schema_identifiers.begin(), schema_identifiers.end(),
                                 schema.identifier) != schema_identifiers.end() &&
                    schema.major_min >= 1 && schema.major_min <= schema.major_max;
            };
            if (!schema_valid(out_manifest.contracts.input_schema) ||
                !schema_valid(out_manifest.contracts.output_schema)) {
                out_error = "Invalid schema contract range or identifier";
                return false;
            }

            out_manifest.capabilities = j.at("capabilities").get<std::vector<std::string>>();

            out_manifest.models.clear();
            if (j.contains("models")) {
                for (const auto& model_json : j.at("models")) {
                    PluginModelEntry model{};
                    model.name = model_json.at("name").get<std::string>();
                    model.path = model_json.at("path").get<std::string>();
                    model.sha256 = model_json.at("sha256").get<std::string>();
                    model.framework = model_json.at("framework").get<std::string>();
                    model.precision = model_json.at("precision").get<std::string>();
                    out_manifest.models.push_back(std::move(model));
                }
            }

            if (j.contains("resource_requirements") && j["resource_requirements"].is_object()) {
                const auto& req = j["resource_requirements"];
                if (req.contains("max_latency_budget_ns") && req["max_latency_budget_ns"].is_number()) {
                    out_manifest.resource_requirements.max_latency_budget_ns = req["max_latency_budget_ns"].get<uint64_t>();
                }
                if (req.contains("max_output_buffer_bytes") && req["max_output_buffer_bytes"].is_number()) {
                    out_manifest.resource_requirements.max_output_buffer_bytes = req["max_output_buffer_bytes"].get<uint32_t>();
                }
                if (req.contains("gpu_vram_bytes") && req["gpu_vram_bytes"].is_number()) {
                    out_manifest.resource_requirements.gpu_vram_bytes = req["gpu_vram_bytes"].get<uint64_t>();
                }
                if (req.contains("requires_cuda") && req["requires_cuda"].is_boolean()) {
                    out_manifest.resource_requirements.requires_cuda = req["requires_cuda"].get<bool>();
                }
            }

            out_error.clear();
            return true;
        } catch (const std::exception& e) {
            out_error = std::string("Manifest parse error: ") + e.what();
            return false;
        }
    }

    static bool parse_string(std::string_view json_str, PluginManifest& out_manifest, std::string& out_error) {
        try {
            const auto j = nlohmann::json::parse(json_str);
            return parse_json(j, out_manifest, out_error);
        } catch (const std::exception& e) {
            out_error = std::string("JSON parse syntax error: ") + e.what();
            return false;
        }
    }
};

} // namespace aim::plugin
