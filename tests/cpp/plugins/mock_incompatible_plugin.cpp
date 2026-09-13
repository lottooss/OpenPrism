// tests/cpp/plugins/mock_incompatible_plugin.cpp
// Incompatible Plugin simulating major ABI mismatch (v2.0)

#include "aim/plugin/plugin_abi.h"

extern "C" {

AIM_PLUGIN_EXPORT uint32_t aim_plugin_abi_version(void) {
    // Return ABI version 2.0 (incompatible with host v1.0)
    return 0x00020000u;
}

AIM_PLUGIN_EXPORT uint32_t aim_plugin_create_v1(
    const AimHostServicesV1* host_services,
    AimPluginApiV1* out_api)
{
    (void)host_services;
    (void)out_api;
    return AIM_STATUS_ERROR_ABI_MISMATCH;
}

} // extern "C"
