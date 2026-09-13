#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "aim/interfaces/perception_engine.hpp"

namespace aim::perception {

struct EngineArtifactLimits {
    std::size_t max_size_bytes{256U * 1024U * 1024U};
};

class VerifiedEngineArtifact final {
public:
    VerifiedEngineArtifact() = default;
    ~VerifiedEngineArtifact() = default;

    VerifiedEngineArtifact(const VerifiedEngineArtifact&) = delete;
    VerifiedEngineArtifact& operator=(const VerifiedEngineArtifact&) = delete;
    VerifiedEngineArtifact(VerifiedEngineArtifact&&) noexcept = default;
    VerifiedEngineArtifact& operator=(VerifiedEngineArtifact&&) noexcept = default;

    [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool empty() const noexcept { return bytes_.empty(); }
    void reset() noexcept;

private:
    friend class EngineArtifactVerifier;
    std::vector<std::uint8_t> bytes_{};
};

class EngineArtifactVerifier final {
public:
    [[nodiscard]] static bool is_sha256_hex(std::string_view value) noexcept;

    [[nodiscard]] static PerceptionStatus load_and_verify(
        const ModelManifest& manifest,
        const EngineArtifactLimits& limits,
        VerifiedEngineArtifact& out_artifact) noexcept;
};

} // namespace aim::perception
