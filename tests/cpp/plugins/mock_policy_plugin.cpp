// tests/cpp/plugins/mock_policy_plugin.cpp
// Reference Mock Policy Plugin implementing Aim Dynamic Plugin C ABI v1

#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include "aim/plugin/plugin_abi.h"

namespace {
constexpr std::uint32_t kMockPayloadSize = 16;
constexpr std::uint32_t kTrackedFourcc = 0x31545441u; // ATT1
constexpr std::uint32_t kIntentFourcc = 0x31494141u;  // AAI1
}

struct MockPolicyState {
    const AimHostServicesV1* host{nullptr};
    bool is_active{false};
    uint64_t processed_frames{0};
};

extern "C" {

AIM_PLUGIN_EXPORT uint32_t aim_plugin_abi_version(void) {
    return AIM_PLUGIN_ABI_VERSION_V1;
}

static uint32_t mock_policy_start(AimPluginHandle* instance, const uint8_t* config_buffer, uint32_t config_size) {
    if (!instance) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (config_size > 0 && config_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    auto* state = reinterpret_cast<MockPolicyState*>(instance);
    state->is_active = true;
    (void)config_buffer;
    (void)config_size;
    return AIM_STATUS_OK;
}

static uint32_t mock_policy_process(
    AimPluginHandle* instance,
    const uint8_t* input_buffer,
    uint32_t input_size,
    uint8_t* output_buffer,
    uint32_t output_capacity,
    uint32_t* out_output_size)
{
    if (!instance || !out_output_size) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (input_size > 0 && input_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    if (output_capacity > 0 && output_buffer == nullptr) return AIM_STATUS_ERROR_INVALID_ARGUMENT;

    auto* state = reinterpret_cast<MockPolicyState*>(instance);
    if (!state->is_active) {
        return AIM_STATUS_ERROR_UNINITIALIZED;
    }

    try {
        if (input_size < 16 || std::memcmp(input_buffer + 4, "ATT1", 4) != 0) {
            return AIM_STATUS_ERROR_CORRUPTED_DATA;
        }

        std::array<std::uint8_t, kMockPayloadSize> payload{};
        payload[4] = 'A';
        payload[5] = 'A';
        payload[6] = 'I';
        payload[7] = '1';
        std::memcpy(payload.data() + 8, input_buffer + 8, 8);
        *out_output_size = kMockPayloadSize;

        if (output_capacity < kMockPayloadSize) {
            return AIM_STATUS_ERROR_BUFFER_TOO_SMALL;
        }

        std::memcpy(output_buffer, payload.data(), payload.size());
        ++state->processed_frames;
        return AIM_STATUS_OK;
    } catch (...) {
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    }
}

static void mock_policy_stop(AimPluginHandle* instance) {
    if (!instance) return;
    auto* state = reinterpret_cast<MockPolicyState*>(instance);
    state->is_active = false;
}

static void mock_policy_destroy(AimPluginHandle* instance) {
    if (!instance) return;
    auto* state = reinterpret_cast<MockPolicyState*>(instance);
    delete state;
}

static uint32_t mock_policy_get_descriptor(AimPluginHandle* instance, AimPluginDescriptorV1* out_descriptor) {
    if (!instance || !out_descriptor) return AIM_STATUS_ERROR_INVALID_ARGUMENT;

    out_descriptor->abi_version = AIM_PLUGIN_ABI_VERSION_V1;
    out_descriptor->kind = AIM_PLUGIN_KIND_POLICY;
    out_descriptor->name = "mock_policy";
    out_descriptor->version = "1.0.0";
    out_descriptor->author = "Aim Research Team";
    out_descriptor->description = "Reference mock aim policy plugin for contract and ABI testing";
    out_descriptor->input_schema_fourcc = kTrackedFourcc;
    out_descriptor->output_schema_fourcc = kIntentFourcc;

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

    auto* state = new (std::nothrow) MockPolicyState();
    if (!state) {
        return AIM_STATUS_ERROR_EXECUTION_FAILED;
    }

    state->host = host_services;
    state->is_active = false;
    state->processed_frames = 0;

    out_api->abi_version = AIM_PLUGIN_ABI_VERSION_V1;
    out_api->kind = AIM_PLUGIN_KIND_POLICY;
    out_api->instance = reinterpret_cast<AimPluginHandle*>(state);
    out_api->start = mock_policy_start;
    out_api->process = mock_policy_process;
    out_api->stop = mock_policy_stop;
    out_api->destroy = mock_policy_destroy;
    out_api->get_descriptor = mock_policy_get_descriptor;

    return AIM_STATUS_OK;
}

} // extern "C"
