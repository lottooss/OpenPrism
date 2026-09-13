// Malformed v1 plugin used to prove that incomplete API tables fail before start().

#include "aim/plugin/plugin_abi.h"

extern "C" {

AIM_PLUGIN_EXPORT uint32_t aim_plugin_abi_version(void) {
    return AIM_PLUGIN_ABI_VERSION_V1;
}

AIM_PLUGIN_EXPORT uint32_t aim_plugin_create_v1(
    const AimHostServicesV1* host_services,
    AimPluginApiV1* out_api) {
    if (host_services == nullptr || out_api == nullptr) {
        return AIM_STATUS_ERROR_INVALID_ARGUMENT;
    }

    out_api->abi_version = AIM_PLUGIN_ABI_VERSION_V1;
    out_api->kind = AIM_PLUGIN_KIND_PERCEPTION;
    out_api->instance = reinterpret_cast<AimPluginHandle*>(static_cast<uintptr_t>(1));
    // All required callbacks are intentionally absent. The host must reject this
    // table without attempting to start or process the plugin.
    return AIM_STATUS_OK;
}

} // extern "C"
