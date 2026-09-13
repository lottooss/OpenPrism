#include "aim/perception/engine_artifact.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

#include "aim/config/sha256.hpp"

namespace aim::perception {

void VerifiedEngineArtifact::reset() noexcept {
    std::vector<std::uint8_t>{}.swap(bytes_);
}

bool EngineArtifactVerifier::is_sha256_hex(std::string_view value) noexcept {
    return value.size() == 64U &&
        std::all_of(value.begin(), value.end(), [](char character) noexcept {
            return (character >= '0' && character <= '9') ||
                (character >= 'a' && character <= 'f') ||
                (character >= 'A' && character <= 'F');
        });
}

PerceptionStatus EngineArtifactVerifier::load_and_verify(
    const ModelManifest& manifest,
    const EngineArtifactLimits& limits,
    VerifiedEngineArtifact& out_artifact) noexcept {
    try {
        out_artifact.reset();
        if (manifest.model_path.empty() || !is_sha256_hex(manifest.engine_sha256) ||
            limits.max_size_bytes == 0) {
            return PerceptionStatus::invalid_argument;
        }

        const std::filesystem::path engine_path(manifest.model_path);
        std::error_code error{};
        if (!std::filesystem::is_regular_file(engine_path, error) || error) {
            return PerceptionStatus::artifact_not_found;
        }

        const std::uintmax_t file_size = std::filesystem::file_size(engine_path, error);
        if (error || file_size == 0 || file_size > limits.max_size_bytes ||
            file_size > std::numeric_limits<std::size_t>::max() ||
            file_size > static_cast<std::uintmax_t>(
                std::numeric_limits<std::streamsize>::max())) {
            return PerceptionStatus::invalid_argument;
        }

        std::ifstream stream(engine_path, std::ios::binary);
        if (!stream.is_open()) {
            return PerceptionStatus::artifact_not_found;
        }

        out_artifact.bytes_.resize(static_cast<std::size_t>(file_size));
        stream.read(reinterpret_cast<char*>(out_artifact.bytes_.data()),
                    static_cast<std::streamsize>(out_artifact.bytes_.size()));
        if (!stream || static_cast<std::size_t>(stream.gcount()) != out_artifact.bytes_.size()) {
            out_artifact.reset();
            return PerceptionStatus::artifact_not_found;
        }

        config::Sha256 hasher{};
        hasher.update(out_artifact.bytes_.data(), out_artifact.bytes_.size());
        std::string actual_hash = hasher.finalize_hex();
        std::string expected_hash = manifest.engine_sha256;
        std::transform(expected_hash.begin(), expected_hash.end(), expected_hash.begin(),
                       [](char character) noexcept {
                           return character >= 'A' && character <= 'F'
                               ? static_cast<char>(character - 'A' + 'a')
                               : character;
                       });
        if (actual_hash != expected_hash) {
            out_artifact.reset();
            return PerceptionStatus::integrity_error;
        }

        return PerceptionStatus::ok;
    } catch (...) {
        out_artifact.reset();
        return PerceptionStatus::device_error;
    }
}

} // namespace aim::perception
