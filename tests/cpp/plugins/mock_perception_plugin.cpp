// tests/cpp/plugins/mock_perception_plugin.cpp
// Reference Mock Perception Plugin implementing Aim Dynamic Plugin C ABI v1

#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include "aim/plugin/plugin_abi.h"

namespace {
constexpr std::uint32_t kMockPayloadSize = 24;
constexpr std::uint32_t kFrameFourcc = 0x31524641u;       // AFR1
constexpr std::uint32_t kObservationFourcc = 0x31424F41u; // AOB1
}

struct MockPerceptionState {
    const AimHostServicesV1* host{nullptr};
    bool is_active{false};
    uint64_t processed_frames{0};
    bool trigger_exception{false};
};

extern "C" {

AIM_PLUGIN_EXPORT uint32_t aim_plugin_abi_version(void) {
    return AIM_PLUGIN_ABI_VERSION_V1;
}

static uint32_t mock_start(AimPluginHandle* instance, const uint8_t* config_buffer, uint32_t config_size) {
    if (!instance) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (config_size > 0 && config_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    auto* state = reinterpret_cast<MockPerceptionState*>(instance);

    if (config_buffer && config_size > 0) {
        std::string cfg(reinterpret_cast<const char*>(config_buffer), config_size);
        if (cfg.find("trigger_exception") != std::string::npos) {
            state->trigger_exception = true;
        }
    }

    state->is_active = true;
    return AIM_STATUS_OK;
}

static uint32_t mock_process(
    AimPluginHandle* instance,
    const uint8_t* input_buffer,
    uint32_t input_size,
    uint8_t* output_buffer,
    uint32_t output_capacity,
    uint32_t* out_output_size)
{
    if (!instance) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (!out_output_size) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (input_size > 0 && input_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (output_capacity > 0 && output_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;

    auto* state = reinterpret_cast<MockPerceptionState*>(instance);
    if (!state->is_active) {
        return AIM_STATUS_ERROR_UNINITIALIZED;
    }

    try {
        if (state->trigger_exception) {
            throw std::runtime_error("Simulated internal perception failure");
        }

        const std::uint64_t sequence = ++state->processed_frames;
        const std::int64_t captured_at_ns = state->host && state->host->monotonic_time_ns
            ? state->host->monotonic_time_ns()
            : 1'000'000'000LL;

        // The ABI sees opaque bytes only. This compact fixture mirrors the
        // FlatBuffers file-identifier position without exporting a C++ object.
        std::array<std::uint8_t, kMockPayloadSize> payload{};
        payload[4] = 'A';
        payload[5] = 'O';
        payload[6] = 'B';
        payload[7] = '1';
        std::memcpy(payload.data() + 8, &sequence, sizeof(sequence));
        std::memcpy(payload.data() + 16, &captured_at_ns, sizeof(captured_at_ns));
        *out_output_size = kMockPayloadSize;

        if (output_capacity < kMockPayloadSize) {
            return AIM_STATUS_ERROR_BUFFER_TOO_SMALL;
        }

        std::memcpy(output_buffer, payload.data(), payload.size());

        // Optional telemetry callback
        if (state->host && state->host->emit_telemetry) {
            uint8_t telem[4] = {'T', 'E', 'L', '1'};
            state->host->emit_telemetry(telem, sizeof(telem));
        }

        // Unused parameter suppression
        (void)input_buffer;
        (void)input_size;

        return AIM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        if (state->host && state->host->report_error) {
            state->host->report_error(AIM_STATUS_ERROR_EXECUTION_FAILED, "Out of memory");
        }
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    } catch (const std::exception& e) {
        if (state->host && state->host->report_error) {
            state->host->report_error(AIM_STATUS_ERROR_EXECUTION_FAILED, e.what());
        }
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    } catch (...) {
        if (state->host && state->host->report_error) {
            state->host->report_error(AIM_STATUS_ERROR_EXECUTION_FAILED, "Unknown C++ exception");
        }
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    }
}

static void mock_stop(AimPluginHandle* instance) {
    if (!instance) return;
    auto* state = reinterpret_cast<MockPerceptionState*>(instance);
    state->is_active = false;
}

static void mock_destroy(AimPluginHandle* instance) {
    if (!instance) return;
    auto* state = reinterpret_cast<MockPerceptionState*>(instance);
    delete state;
}

static uint32_t mock_get_descriptor(AimPluginHandle* instance, AimPluginDescriptorV1* out_descriptor) {
    if (!instance || !out_descriptor) return AIM_STATUS_ERROR_INVALID_ARGUMENT;

    out_descriptor->abi_version = AIM_PLUGIN_ABI_VERSION_V1;
    out_descriptor->kind = AIM_PLUGIN_KIND_PERCEPTION;
    out_descriptor->name = "mock_perception";
    out_descriptor->version = "1.0.0";
    out_descriptor->author = "Aim Research Team";
    out_descriptor->description = "Reference mock perception plugin for contract and ABI testing";
    out_descriptor->input_schema_fourcc = kFrameFourcc;
    out_descriptor->output_schema_fourcc = kObservationFourcc;

    return AIM_STATUS_OK;
}

AIM_PLUGIN_EXPORT uint32_t aim_plugin_create_v1(
    const AimHostServicesV1* host_services,
    AimPluginApiV1* out_api)
{
    if (!host_services || !out_api) {
        return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (AIM_PLUGIN_ABI_MAJOR(host_services->abi_version) != AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1)) {
        return AIM_STATUS_ERROR_ABI_MISMATCH;
    }

    auto* state = new (std::nothrow) MockPerceptionState();
    if (!state) {
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    }

    state->host = host_services;
    state->is_active = false;
    state->processed_frames = 0;
    state->trigger_exception = false;

    out_api->abi_version = AIM_PLUGIN_ABI_VERSION_V1;
    out_api->kind = AIM_PLUGIN_KIND_PERCEPTION;
    out_api->instance = reinterpret_cast<AimPluginHandle*>(state);
    out_api->start = mock_start;
    out_api->process = mock_process;
    out_api->stop = mock_stop;
    out_api->destroy = mock_destroy;
    out_api->get_descriptor = mock_get_descriptor;

    return AIM_STATUS_OK;
}

} // extern "C"
