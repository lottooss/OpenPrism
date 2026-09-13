// tests/cpp/test_plugin_abi.cpp
// Comprehensive C++20 test suite for OpenPrism Plugin C ABI and Interface Contracts

#include <cassert>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "aim/bus/bus_traits.hpp"
#include "aim/interfaces/interfaces.hpp"
#include "aim/plugin/plugin.hpp"

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << ": " #cond << std::endl; \
            return 1; \
        } \
    } while (false)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))

// Allocation tracking for zero-allocation hot-path test
static bool g_track_allocations = false;
static std::size_t g_allocation_count = 0;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(std::size_t size) {
    if (g_track_allocations) {
        ++g_allocation_count;
    }
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void* operator new[](std::size_t size) {
    if (g_track_allocations) {
        ++g_allocation_count;
    }
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

#if defined(__cpp_sized_deallocation)
void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}
void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static std::filesystem::path get_executable_dir() {
#if defined(_WIN32)
    char buffer[MAX_PATH];
    DWORD len = ::GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

static std::string find_plugin_dll(const std::string& name) {
    const auto exe_dir = get_executable_dir();
    const std::vector<std::filesystem::path> search_paths = {
        exe_dir / (name + ".dll"),
        exe_dir / (name + ".so"),
        exe_dir / ("lib" + name + ".so"),
        exe_dir / "bin" / (name + ".dll"),
        exe_dir / "bin" / (name + ".so"),
        exe_dir / "bin" / ("lib" + name + ".so"),
        exe_dir / "Debug" / (name + ".dll"),
        exe_dir / "Release" / (name + ".dll"),
        exe_dir / ".." / (name + ".dll"),
        exe_dir / ".." / (name + ".so"),
        exe_dir / ".." / ("lib" + name + ".so"),
        std::filesystem::current_path() / (name + ".dll"),
        std::filesystem::current_path() / (name + ".so"),
        std::filesystem::current_path() / ("lib" + name + ".so"),
        std::filesystem::current_path() / "build-wsl" / (name + ".so"),
        std::filesystem::current_path() / "build-wsl" / ("lib" + name + ".so"),
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return path.string();
        }
    }
    return "";
}

// Dummy host services
static std::size_t g_telemetry_emitted_count = 0;
static std::size_t g_error_reported_count = 0;

static void host_emit_telemetry(const uint8_t* buffer, uint32_t size) {
    (void)buffer;
    (void)size;
    ++g_telemetry_emitted_count;
}

static int64_t host_monotonic_time_ns() {
    static int64_t fake_ns = 1'000'000'000LL;
    fake_ns += 6'944'444LL; // ~144 Hz tick
    return fake_ns;
}

static void host_report_error(uint32_t code, const char* message) {
    (void)code;
    (void)message;
    ++g_error_reported_count;
}

static AimHostServicesV1 create_test_host_services() {
    return AimHostServicesV1{
        .abi_version = AIM_PLUGIN_ABI_VERSION_V1,
        .emit_telemetry = host_emit_telemetry,
        .monotonic_time_ns = host_monotonic_time_ns,
        .report_error = host_report_error
    };
}

int test_static_abi_layout() {
    // Verify standard struct sizes and alignment
    ASSERT_EQ(sizeof(AimHostServicesV1), sizeof(void*) * 3 + sizeof(uint32_t) + (sizeof(void*) == 8 ? 4 : 0));
    ASSERT_EQ(sizeof(AimPluginDescriptorV1), sizeof(uint32_t) * 4 + sizeof(const char*) * 4);
    ASSERT_EQ(AIM_PLUGIN_ABI_VERSION_V1, 0x00010000u);
    ASSERT_EQ(AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1), 1u);
    ASSERT_EQ(AIM_PLUGIN_ABI_MINOR(AIM_PLUGIN_ABI_VERSION_V1), 0u);

    // Verify FourCC macros
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("AOB1"), 0x31424F41u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("ATT1"), 0x31545441u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("AAI1"), 0x31494141u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("AAC1"), 0x31434141u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("AFR1"), 0x31524641u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("ATE1"), 0x31455441u);
    ASSERT_EQ(aim::plugin::fourcc_to_uint32("AOB1X"), 0u);

    ASSERT_EQ(aim::plugin::uint32_to_fourcc(0x31424F41u), "AOB1");
    ASSERT_EQ(aim::plugin::uint32_to_fourcc(0x31494141u), "AAI1");

    return 0;
}

int test_version_negotiation() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_incompatible_plugin");
    ASSERT_FALSE(dll_path.empty());

    AimStatusCode status = AIM_STATUS_OK;
    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_UNKNOWN, &status);

    ASSERT_TRUE(plugin == nullptr);
    ASSERT_EQ(status, AIM_STATUS_ERROR_ABI_MISMATCH);

    return 0;
}

int test_malformed_api_rejected_before_start() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_malformed_plugin");
    ASSERT_FALSE(dll_path.empty());

    AimStatusCode status = AIM_STATUS_OK;
    auto plugin = aim::plugin::PluginLoader::load(
        dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION, &status);
    ASSERT_TRUE(plugin == nullptr);
    ASSERT_EQ(status, AIM_STATUS_ERROR_CORRUPTED_DATA);

    status = AIM_STATUS_OK;
    plugin = aim::plugin::PluginLoader::load(
        dll_path, nullptr, AIM_PLUGIN_KIND_PERCEPTION, &status);
    ASSERT_TRUE(plugin == nullptr);
    ASSERT_EQ(status, AIM_STATUS_ERROR_INVALID_ARGUMENT);
    return 0;
}

int test_plugin_lifecycle() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    AimStatusCode status = AIM_STATUS_OK;
    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION, &status);

    ASSERT_TRUE(plugin != nullptr);
    ASSERT_EQ(status, AIM_STATUS_OK);
    ASSERT_TRUE(plugin->is_valid());
    ASSERT_FALSE(plugin->is_active());
    ASSERT_EQ(plugin->kind(), AIM_PLUGIN_KIND_PERCEPTION);

    // Descriptor check
    AimPluginDescriptorV1 desc{};
    ASSERT_EQ(plugin->get_descriptor(&desc), AIM_STATUS_OK);
    ASSERT_EQ(std::string(desc.name), "mock_perception");
    ASSERT_EQ(std::string(desc.version), "1.0.0");
    ASSERT_EQ(desc.output_schema_fourcc, 0x31424F41u); // 'AOB1'

    // Start plugin
    ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_OK);
    ASSERT_TRUE(plugin->is_active());

    // Process 10,000 frames
    std::vector<uint8_t> output_buffer(64);
    uint32_t out_size = 0;

    for (int i = 0; i < 10000; ++i) {
        status = plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size);
        ASSERT_EQ(status, AIM_STATUS_OK);
        ASSERT_EQ(out_size, 24u);
        ASSERT_TRUE(std::memcmp(output_buffer.data() + 4, "AOB1", 4) == 0);
        std::uint64_t sequence = 0;
        std::memcpy(&sequence, output_buffer.data() + 8, sizeof(sequence));
        ASSERT_EQ(sequence, static_cast<uint64_t>(i + 1));
    }

    plugin->stop();
    ASSERT_FALSE(plugin->is_active());

    plugin->unload();
    ASSERT_FALSE(plugin->is_valid());

    return 0;
}

int test_uninitialized_execution_guard() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    AimStatusCode status = AIM_STATUS_OK;
    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION, &status);
    ASSERT_TRUE(plugin != nullptr);

    // Calling process before start must fail with UNINITIALIZED
    std::vector<uint8_t> output_buffer(64);
    uint32_t out_size = 0;
    status = plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size);
    ASSERT_EQ(status, AIM_STATUS_ERROR_UNINITIALIZED);

    // Start once
    ASSERT_EQ(plugin->start(nullptr, 1), AIM_STATUS_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_OK);

    // Calling start a second time must return ALREADY_INITIALIZED
    ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_ERROR_ALREADY_INITIALIZED);

    plugin->stop();
    return 0;
}

int test_buffer_capacity_truncation() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION);
    ASSERT_TRUE(plugin != nullptr);
    ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_OK);

    // Pass buffer that is too small (e.g. 16 bytes)
    std::vector<uint8_t> tiny_buffer(16, 0xFF);
    uint32_t out_size = 0;

    const AimStatusCode status = plugin->process(nullptr, 0, tiny_buffer.data(), 16, &out_size);
    ASSERT_EQ(status, AIM_STATUS_ERROR_BUFFER_TOO_SMALL);
    ASSERT_EQ(out_size, 24u);

    // Ensure tiny_buffer was not corrupted/written past capacity
    for (size_t i = 0; i < tiny_buffer.size(); ++i) {
        ASSERT_EQ(tiny_buffer[i], 0xFF);
    }

    out_size = 123;
    ASSERT_EQ(plugin->process(nullptr, 1, tiny_buffer.data(), 16, &out_size),
              AIM_STATUS_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(out_size, 123u);
    ASSERT_EQ(plugin->process(nullptr, 0, nullptr, 16, &out_size),
              AIM_STATUS_ERROR_INVALID_ARGUMENT);
    ASSERT_EQ(plugin->process(nullptr, 0, tiny_buffer.data(), 16, nullptr),
              AIM_STATUS_ERROR_INVALID_ARGUMENT);

    plugin->stop();
    return 0;
}

int test_exception_containment() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION);
    ASSERT_TRUE(plugin != nullptr);

    // Configure plugin to trigger internal exception
    const std::string bad_config = "{\"trigger_exception\": true}";
    ASSERT_EQ(plugin->start(reinterpret_cast<const uint8_t*>(bad_config.data()), static_cast<uint32_t>(bad_config.size())), AIM_STATUS_OK);

    std::vector<uint8_t> output_buffer(sizeof(aim::bus::TargetObservationBatch));
    uint32_t out_size = 0;

    // Process should catch internal C++ exception and return EXECUTION_FAILED without crashing host
    const AimStatusCode status = plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size);
    ASSERT_EQ(status, AIM_STATUS_ERROR_EXECUTION_FAILED);

    plugin->stop();
    return 0;
}

int test_opaque_serialized_buffer_chain() {
    const auto host_services = create_test_host_services();
    const std::string perc_dll = find_plugin_dll("aim_mock_perception_plugin");
    const std::string pol_dll = find_plugin_dll("aim_mock_policy_plugin");
    ASSERT_FALSE(perc_dll.empty());
    ASSERT_FALSE(pol_dll.empty());

    auto perc_plugin = aim::plugin::PluginLoader::load(perc_dll, &host_services, AIM_PLUGIN_KIND_PERCEPTION);
    auto pol_plugin = aim::plugin::PluginLoader::load(pol_dll, &host_services, AIM_PLUGIN_KIND_POLICY);
    ASSERT_TRUE(perc_plugin != nullptr);
    ASSERT_TRUE(pol_plugin != nullptr);

    ASSERT_EQ(perc_plugin->start(nullptr, 0), AIM_STATUS_OK);
    ASSERT_EQ(pol_plugin->start(nullptr, 0), AIM_STATUS_OK);

    // Stage 1: Perception
    std::vector<uint8_t> obs_buf(64);
    uint32_t obs_size = 0;
    ASSERT_EQ(perc_plugin->process(nullptr, 0, obs_buf.data(), static_cast<uint32_t>(obs_buf.size()), &obs_size), AIM_STATUS_OK);

    ASSERT_EQ(obs_size, 24u);
    ASSERT_TRUE(std::memcmp(obs_buf.data() + 4, "AOB1", 4) == 0);

    // The host transforms observations into a separately serialized tracked
    // target payload; no C++ object or allocator ownership crosses the ABI.
    std::array<std::uint8_t, 16> tracked_payload{};
    tracked_payload[4] = 'A';
    tracked_payload[5] = 'T';
    tracked_payload[6] = 'T';
    tracked_payload[7] = '1';
    std::memcpy(tracked_payload.data() + 8, obs_buf.data() + 8, 8);

    // Stage 2: Policy
    std::vector<uint8_t> intent_buf(32);
    uint32_t intent_size = 0;
    ASSERT_EQ(pol_plugin->process(tracked_payload.data(), static_cast<uint32_t>(tracked_payload.size()),
                                  intent_buf.data(), static_cast<uint32_t>(intent_buf.size()), &intent_size), AIM_STATUS_OK);
    ASSERT_EQ(intent_size, 16u);
    ASSERT_TRUE(std::memcmp(intent_buf.data() + 4, "AAI1", 4) == 0);
    ASSERT_TRUE(std::memcmp(intent_buf.data() + 8, tracked_payload.data() + 8, 8) == 0);

    perc_plugin->stop();
    pol_plugin->stop();
    return 0;
}

int test_zero_allocation_hot_path() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION);
    ASSERT_TRUE(plugin != nullptr);
    ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_OK);

    // Preallocate all buffers before hot path
    std::vector<uint8_t> output_buffer(64);
    uint32_t out_size = 0;

    // Warm up 10 iterations
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size), AIM_STATUS_OK);
    }

    // Begin allocation tracking
    g_allocation_count = 0;
    g_track_allocations = true;

    for (int i = 0; i < 10000; ++i) {
        const AimStatusCode status = plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size);
        ASSERT_EQ(status, AIM_STATUS_OK);
    }

    g_track_allocations = false;
    ASSERT_EQ(g_allocation_count, 0u);

    plugin->stop();
    return 0;
}

int test_repeated_reload_cycles() {
    const auto host_services = create_test_host_services();
    const std::string dll_path = find_plugin_dll("aim_mock_perception_plugin");
    ASSERT_FALSE(dll_path.empty());

    std::vector<uint8_t> output_buffer(64);
    uint32_t out_size = 0;

    for (int cycle = 0; cycle < 50; ++cycle) {
        auto plugin = aim::plugin::PluginLoader::load(dll_path, &host_services, AIM_PLUGIN_KIND_PERCEPTION);
        ASSERT_TRUE(plugin != nullptr);
        ASSERT_EQ(plugin->start(nullptr, 0), AIM_STATUS_OK);

        for (int frame = 0; frame < 5; ++frame) {
            ASSERT_EQ(plugin->process(nullptr, 0, output_buffer.data(), static_cast<uint32_t>(output_buffer.size()), &out_size), AIM_STATUS_OK);
        }

        plugin->stop();
        plugin->unload();
    }

    return 0;
}

int test_manifest_parser() {
    const std::string valid_json = R"({
        "schema_version": 1,
        "name": "yolo11n_perception",
        "version": "1.0.0",
        "abi_version": 65536,
        "kind": "perception",
        "author": "Aim Research",
        "license": "MIT",
        "entry_point": "yolo11n_plugin.dll",
        "checksum": {
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "size_bytes": 1024
        },
        "contracts": {
            "input_schema": {
                "identifier": "AFR1",
                "major_min": 1,
                "major_max": 1
            },
            "output_schema": {
                "identifier": "AOB1",
                "major_min": 1,
                "major_max": 1
            }
        },
        "capabilities": ["bounding_box", "gpu_direct"]
    })";

    aim::plugin::PluginManifest manifest{};
    std::string error{};
    ASSERT_TRUE(aim::plugin::PluginManifestParser::parse_string(valid_json, manifest, error));
    ASSERT_TRUE(error.empty());
    ASSERT_EQ(manifest.name, "yolo11n_perception");
    ASSERT_EQ(manifest.kind, AIM_PLUGIN_KIND_PERCEPTION);
    ASSERT_TRUE(manifest.is_abi_compatible());
    ASSERT_EQ(manifest.contracts.input_schema.identifier, "AFR1");
    ASSERT_EQ(manifest.contracts.output_schema.identifier, "AOB1");

    // Invalid JSON missing required field
    const std::string invalid_json = R"({
        "schema_version": 1,
        "name": "incomplete"
    })";

    aim::plugin::PluginManifest bad_manifest{};
    ASSERT_FALSE(aim::plugin::PluginManifestParser::parse_string(invalid_json, bad_manifest, error));
    ASSERT_FALSE(error.empty());

    const std::string unknown_field_json = valid_json.substr(0, valid_json.size() - 1) +
        R"(,"unexpected":true})";
    ASSERT_FALSE(aim::plugin::PluginManifestParser::parse_string(
        unknown_field_json, bad_manifest, error));

    const std::string invalid_range_json = R"({
        "schema_version":1,"name":"bad_range","version":"1.0.0",
        "abi_version":65536,"kind":"perception","author":"Aim Research",
        "license":"MIT","entry_point":"bad.dll",
        "checksum":{"sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size_bytes":1},
        "contracts":{"input_schema":{"identifier":"AFR1","major_min":2,"major_max":1},
        "output_schema":{"identifier":"AOB1","major_min":1,"major_max":1}},
        "capabilities":[]})";
    ASSERT_FALSE(aim::plugin::PluginManifestParser::parse_string(
        invalid_range_json, bad_manifest, error));

    return 0;
}

int main() {
    std::cout << "Running Aim Dynamic Plugin C ABI & Module Contract Test Suite..." << std::endl;

    if (test_static_abi_layout() != 0) return 1;
    if (test_version_negotiation() != 0) return 1;
    if (test_malformed_api_rejected_before_start() != 0) return 1;
    if (test_plugin_lifecycle() != 0) return 1;
    if (test_uninitialized_execution_guard() != 0) return 1;
    if (test_buffer_capacity_truncation() != 0) return 1;
    if (test_exception_containment() != 0) return 1;
    if (test_opaque_serialized_buffer_chain() != 0) return 1;
    if (test_zero_allocation_hot_path() != 0) return 1;
    if (test_repeated_reload_cycles() != 0) return 1;
    if (test_manifest_parser() != 0) return 1;

    std::cout << "All Plugin C ABI & Module Contract tests PASSED successfully." << std::endl;
    return 0;
}
