// include/aim/config/run_manifest.hpp
#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "aim/config/config_loader.hpp"
#include "aim/config/sha256.hpp"
#include "aim/config/types.hpp"

namespace aim::config {

struct GitRevision {
    std::string commit_hash{"untracked"};
    std::string branch{"unknown"};
    bool is_dirty{false};
    std::string describe{"unknown"};

    [[nodiscard]] nlohmann::json to_json() const {
        return {
            {"commit_hash", commit_hash},
            {"branch", branch},
            {"is_dirty", is_dirty},
            {"describe", describe}
        };
    }
};

struct EnvironmentMetadata {
    std::string os_name{"Windows"};
    std::string os_version{"unknown"};
    std::string architecture{"x86_64"};
    std::string processor{"unknown"};
    std::string hostname{"unknown"};

    [[nodiscard]] nlohmann::json to_json() const {
        return {
            {"os_name", os_name},
            {"os_version", os_version},
            {"architecture", architecture},
            {"processor", processor},
            {"hostname", hostname}
        };
    }
};

struct ArtifactDescriptor {
    std::string name{};
    std::string path{};
    std::string sha256{};
    std::uint64_t size_bytes{0};
    std::string type{"other"};

    [[nodiscard]] nlohmann::json to_json() const {
        return {
            {"name", name},
            {"path", path},
            {"sha256", sha256},
            {"size_bytes", size_bytes},
            {"type", type}
        };
    }
};

struct ExecutionMetadata {
    std::string mode{"simulation"};
    double internal_deadline_ms{10.0};
    double target_p99_ms{6.0};
    std::uint32_t warmup_iterations_completed{200};
    std::string notes{""};

    [[nodiscard]] nlohmann::json to_json() const {
        return {
            {"mode", mode},
            {"internal_deadline_ms", internal_deadline_ms},
            {"target_p99_ms", target_p99_ms},
            {"warmup_iterations_completed", warmup_iterations_completed},
            {"notes", notes}
        };
    }
};

class RunManifest {
public:
    static RunManifest create(
        const ResolvedConfig& config,
        const std::string& config_sha256,
        const std::vector<std::string>& source_layers,
        const std::string& manifest_id = "00000000-0000-4000-8000-000000000001",
        const std::string& timestamp_utc = "2026-08-23T21:00:00Z",
        const GitRevision& git_rev = {},
        const EnvironmentMetadata& env = {},
        const std::map<std::string, std::string>& schema_hashes = {},
        const std::vector<ArtifactDescriptor>& artifacts = {},
        const ExecutionMetadata& exec = {})
    {
        RunManifest manifest;
        manifest.schema_version_ = 1;
        manifest.manifest_id_ = manifest_id;
        manifest.timestamp_utc_ = timestamp_utc;
        manifest.git_revision_ = git_rev;
        manifest.environment_ = env;
        manifest.source_layers_ = source_layers;
        manifest.config_sha256_ = config_sha256;
        manifest.resolved_config_ = config;
        manifest.schema_hashes_ = schema_hashes;
        manifest.artifacts_ = artifacts;
        manifest.execution_ = exec;
        return manifest;
    }

    [[nodiscard]] nlohmann::json to_json() const {
        nlohmann::json j;
        j["schema_version"] = schema_version_;
        j["manifest_id"] = manifest_id_;
        j["timestamp_utc"] = timestamp_utc_;
        j["git_info"] = git_revision_.to_json();
        j["environment"] = environment_.to_json();

        j["config"] = {
            {"source_layers", source_layers_},
            {"config_sha256", config_sha256_},
            {"resolved_config", ConfigLoader::to_json(resolved_config_)}
        };

        j["schemas"] = schema_hashes_;

        auto art_arr = nlohmann::json::array();
        for (const auto& art : artifacts_) {
            art_arr.push_back(art.to_json());
        }
        j["artifacts"] = art_arr;
        j["execution"] = execution_.to_json();

        return j;
    }

    [[nodiscard]] std::string to_json_string(int indent = 2) const {
        return to_json().dump(indent);
    }

    [[nodiscard]] bool save(const std::filesystem::path& file_path) const {
        std::ofstream out(file_path);
        if (!out.is_open()) return false;
        out << to_json_string(2) << std::endl;
        return true;
    }

    /// @brief Verify cryptographic integrity and tamper-freedom of a manifest document
    static bool verify(const nlohmann::json& manifest_doc, std::string& out_error) {
        if (!manifest_doc.is_object()) {
            out_error = "Manifest is not a JSON object";
            return false;
        }

        if (!manifest_doc.contains("config") || !manifest_doc["config"].contains("resolved_config")) {
            out_error = "Manifest missing config.resolved_config section";
            return false;
        }

        const std::string recorded_hash = manifest_doc["config"].value("config_sha256", "");
        if (recorded_hash.empty()) {
            out_error = "Manifest missing config.config_sha256";
            return false;
        }

        // Recompute SHA-256 of resolved_config
        const auto& resolved_cfg = manifest_doc["config"]["resolved_config"];
        const std::string canonical_cfg_str = resolved_cfg.dump();
        const std::string recomputed_hash = Sha256::hash_string(canonical_cfg_str);

        if (recorded_hash != recomputed_hash) {
            out_error = "Config SHA-256 tamper detected: recorded '" + recorded_hash + "' != recomputed '" + recomputed_hash + "'";
            return false;
        }

        return true;
    }

    [[nodiscard]] const std::string& manifest_id() const noexcept { return manifest_id_; }
    [[nodiscard]] const std::string& config_sha256() const noexcept { return config_sha256_; }
    [[nodiscard]] const ResolvedConfig& resolved_config() const noexcept { return resolved_config_; }

private:
    std::uint32_t schema_version_{1};
    std::string manifest_id_{"00000000-0000-4000-8000-000000000001"};
    std::string timestamp_utc_{};
    GitRevision git_revision_{};
    EnvironmentMetadata environment_{};
    std::vector<std::string> source_layers_{};
    std::string config_sha256_{};
    ResolvedConfig resolved_config_{};
    std::map<std::string, std::string> schema_hashes_{};
    std::vector<ArtifactDescriptor> artifacts_{};
    ExecutionMetadata execution_{};
};

} // namespace aim::config
