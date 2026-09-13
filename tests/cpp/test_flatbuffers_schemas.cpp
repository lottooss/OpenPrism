#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "actuation_command_generated.h"
#include "aim/bus/schema_version.hpp"
#include "frame_descriptor_generated.h"

namespace {

using aim::bus::v1::ActuationCommand;
using aim::bus::v1::ButtonAction;
using aim::bus::v1::FrameDescriptor;
using aim::bus::v1::MouseButton;

[[nodiscard]] auto as_bytes(std::span<const std::uint8_t> buffer) noexcept -> std::span<const std::byte> {
    return {reinterpret_cast<const std::byte*>(buffer.data()), buffer.size()};
}

[[nodiscard]] auto read_binary(const std::filesystem::path& path) -> std::vector<std::uint8_t> {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const auto end = input.tellg();
    if (end <= 0) {
        return {};
    }
    const auto size = static_cast<std::size_t>(end);
    std::vector<std::uint8_t> bytes(size);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) {
        return {};
    }
    return bytes;
}

[[nodiscard]] auto write_binary(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) -> bool {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

[[nodiscard]] auto build_command() -> flatbuffers::FlatBufferBuilder {
    flatbuffers::FlatBufferBuilder builder(256);
    const auto header = aim::bus::v1::CreateCorrelationHeader(builder, 42, 1'000'000'000, 7, 1);
    const auto transition = aim::bus::v1::CreateButtonTransition(
        builder,
        MouseButton::Left,
        ButtonAction::Press);
    const auto command = aim::bus::v1::CreateActuationCommand(
        builder,
        aim::bus::kSchemaMajorV1,
        aim::bus::kSchemaMinorV1,
        header,
        1'000'500'000,
        1'001'000'000,
        25,
        -14,
        transition,
        false);
    aim::bus::v1::FinishActuationCommandBuffer(builder, command);
    return builder;
}

[[nodiscard]] auto verify_command(std::span<const std::uint8_t> bytes) -> bool {
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!aim::bus::v1::VerifyActuationCommandBuffer(verifier)) {
        return false;
    }
    const ActuationCommand* command = aim::bus::v1::GetActuationCommand(bytes.data());
    if (!command) {
        return false;
    }
    const auto* header = command->header();
    const auto* transition = command->button_transition();
    return aim::bus::is_supported_version(command->schema_major(), command->schema_minor(),
                                           aim::bus::kActuationCommandContract) &&
           header != nullptr && header->sequence_id() == 42 && header->source_timestamp_ns() == 1'000'000'000 &&
           header->pipeline_run_id() == 7 && header->flags() == 1 && command->generated_at_ns() == 1'000'500'000 &&
           command->desired_apply_time_ns() == 1'001'000'000 && command->delta_x_counts() == 25 &&
           command->delta_y_counts() == -14 && transition != nullptr && transition->button() == MouseButton::Left &&
           transition->action() == ButtonAction::Press && !command->cancel_superseded();
}

[[nodiscard]] auto verify_python_frame(std::span<const std::uint8_t> bytes) -> bool {
    if (!aim::bus::has_identifier(
            {reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()}, aim::bus::kFrameDescriptorContract)) {
        return false;
    }
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!aim::bus::v1::VerifyFrameDescriptorBuffer(verifier)) {
        return false;
    }
    const FrameDescriptor* frame = aim::bus::v1::GetFrameDescriptor(bytes.data());
    const auto* header = frame->header();
    return aim::bus::is_supported_version(frame->schema_major(), frame->schema_minor(),
                                           aim::bus::kFrameDescriptorContract) &&
           header != nullptr && header->sequence_id() == 42 && header->source_timestamp_ns() == 1'000'000'000 &&
           header->pipeline_run_id() == 7 && header->flags() == 1 && frame->frame_id() == 101 &&
           frame->captured_at_ns() == 1'000'000'000 && frame->width() == 1920 && frame->height() == 1080 &&
           frame->pool_slot_index() == 3 && frame->shared_nt_handle() == 0x12345678ULL &&
           frame->adapter_luid() == 0x10688ULL && !frame->is_keyframe();
}

[[nodiscard]] auto self_test() -> bool {
    const auto builder = build_command();
    const std::span<const std::uint8_t> bytes(builder.GetBufferPointer(), builder.GetSize());
    if (!aim::bus::has_identifier(as_bytes(bytes), aim::bus::kActuationCommandContract)) {
        return false;
    }
    if (!verify_command(bytes)) {
        return false;
    }
    std::vector<std::uint8_t> corrupted(bytes.begin(), bytes.end());
    corrupted[4] = static_cast<std::uint8_t>('X');
    flatbuffers::Verifier verifier(corrupted.data(), corrupted.size());
    return !aim::bus::v1::VerifyActuationCommandBuffer(verifier) &&
           !aim::bus::has_identifier(as_bytes(std::span<const std::uint8_t>(corrupted)),
                                     aim::bus::kActuationCommandContract);
}

}  // namespace

auto main(int argc, char** argv) -> int {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        if (!self_test()) {
            std::cerr << "native FlatBuffers contract self-test failed\n";
            return 1;
        }
        return 0;
    }

    if (argc != 5 || std::string_view(argv[1]) != "--verify-python-frame" ||
        std::string_view(argv[3]) != "--write-cpp-command") {
        std::cerr << "usage: aim_bus_schema_tests --verify-python-frame <input> --write-cpp-command <output>\n";
        return 2;
    }

    const auto frame_bytes = read_binary(argv[2]);
    if (frame_bytes.empty() || !verify_python_frame(frame_bytes)) {
        std::cerr << "Python-produced FrameDescriptor failed C++ verification\n";
        return 3;
    }

    const auto command_builder = build_command();
    const std::span<const std::uint8_t> command_bytes(
        command_builder.GetBufferPointer(), command_builder.GetSize());
    if (!verify_command(command_bytes) || !write_binary(argv[4], command_bytes)) {
        std::cerr << "C++ ActuationCommand generation failed\n";
        return 4;
    }
    return 0;
}
