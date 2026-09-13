#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace aim::bus {

inline constexpr std::uint16_t kSchemaMajorV1 = 1;
inline constexpr std::uint16_t kSchemaMinorV1 = 0;

struct SchemaContract final {
    std::array<std::byte, 4> file_identifier;
    std::uint16_t major;
    std::uint16_t minor;
};

[[nodiscard]] constexpr auto identifier(std::array<char, 4> value) noexcept -> std::array<std::byte, 4> {
    return {
        static_cast<std::byte>(value[0]),
        static_cast<std::byte>(value[1]),
        static_cast<std::byte>(value[2]),
        static_cast<std::byte>(value[3]),
    };
}

inline constexpr SchemaContract kFrameDescriptorContract{identifier({'A', 'F', 'R', '1'}), 1, 0};
inline constexpr SchemaContract kTargetObservationContract{identifier({'A', 'O', 'B', '1'}), 1, 0};
inline constexpr SchemaContract kTrackedTargetContract{identifier({'A', 'T', 'T', '1'}), 1, 0};
inline constexpr SchemaContract kAimIntentContract{identifier({'A', 'A', 'I', '1'}), 1, 0};
inline constexpr SchemaContract kActuationCommandContract{identifier({'A', 'A', 'C', '1'}), 1, 0};
inline constexpr SchemaContract kTelemetryEventContract{identifier({'A', 'T', 'E', '1'}), 1, 0};

[[nodiscard]] constexpr auto is_supported_version(
    std::uint16_t major,
    std::uint16_t minor,
    const SchemaContract& contract) noexcept -> bool {
    return major == contract.major && minor <= contract.minor;
}

[[nodiscard]] constexpr auto has_identifier(
    std::span<const std::byte> buffer,
    const SchemaContract& contract) noexcept -> bool {
    constexpr std::size_t kIdentifierOffset = 4;
    constexpr std::size_t kIdentifierSize = 4;
    if (buffer.size() < kIdentifierOffset + kIdentifierSize) {
        return false;
    }
    for (std::size_t index = 0; index < kIdentifierSize; ++index) {
        if (buffer[kIdentifierOffset + index] != contract.file_identifier[index]) {
            return false;
        }
    }
    return true;
}

}  // namespace aim::bus
