// include/aim/plugin/plugin_abi.h
// OpenPrism Stable Dynamic Plugin C ABI v1
#ifndef AIM_PLUGIN_PLUGIN_ABI_H_
#define AIM_PLUGIN_PLUGIN_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(AIM_PLUGIN_EXPORTS)
    #define AIM_PLUGIN_EXPORT __declspec(dllexport)
  #else
    #define AIM_PLUGIN_EXPORT __declspec(dllimport)
  #endif
#else
  #if defined(AIM_PLUGIN_EXPORTS)
    #define AIM_PLUGIN_EXPORT __attribute__((visibility("default")))
  #else
    #define AIM_PLUGIN_EXPORT
  #endif
#endif

/* Major/Minor packed version: 0x00010000u = v1.0 */
#define AIM_PLUGIN_ABI_VERSION_V1 0x00010000u
#define AIM_PLUGIN_ABI_MAJOR(ver) (((ver) >> 16) & 0xFFFFu)
#define AIM_PLUGIN_ABI_MINOR(ver) ((ver) & 0xFFFFu)

/* Opaque handle to plugin instance */
typedef struct AimPluginHandle AimPluginHandle;

/* Plugin Kind Enumeration */
typedef enum AimPluginKind {
    AIM_PLUGIN_KIND_UNKNOWN     = 0,
    AIM_PLUGIN_KIND_PERCEPTION  = 1,
    AIM_PLUGIN_KIND_TRACKER     = 2,
    AIM_PLUGIN_KIND_POLICY      = 3,
    AIM_PLUGIN_KIND_TRAJECTORY  = 4,
    AIM_PLUGIN_KIND_ACTUATOR    = 5,
    AIM_PLUGIN_KIND_SCENARIO    = 6
} AimPluginKind;

/* Strict Integer Status Codes */
typedef enum AimStatusCode {
    AIM_STATUS_OK                          = 0,
    AIM_STATUS_ERROR_INVALID_ARGUMENT      = 1,
    AIM_STATUS_ERROR_BUFFER_TOO_SMALL      = 2,
    AIM_STATUS_ERROR_ABI_MISMATCH          = 3,
    AIM_STATUS_ERROR_UNINITIALIZED         = 4,
    AIM_STATUS_ERROR_ALREADY_INITIALIZED   = 5,
    AIM_STATUS_ERROR_OUT_OF_BOUNDS         = 6,
    AIM_STATUS_ERROR_CORRUPTED_DATA        = 7,
    AIM_STATUS_ERROR_EXECUTION_FAILED      = 8,
    AIM_STATUS_ERROR_HARDWARE_FAULT        = 9,
    AIM_STATUS_ERROR_EMERGENCY_STOP        = 10,
    AIM_STATUS_ERROR_NOT_SUPPORTED         = 11
} AimStatusCode;

/* Host Services Callback Table (provided by the host and valid until destroy()).
 * Callback functions and their pointed-to input bytes are borrowed for the duration
 * of each call. Plugins must not retain or free them. */
typedef struct AimHostServicesV1 {
    uint32_t abi_version;
    void (*emit_telemetry)(const uint8_t* buffer, uint32_t size);
    int64_t (*monotonic_time_ns)(void);
    void (*report_error)(uint32_t code, const char* message);
} AimHostServicesV1;

/* Plugin Metadata Descriptor. String pointers remain plugin-owned and valid until
 * destroy(); the host must copy text it needs after unload and must never free it. */
typedef struct AimPluginDescriptorV1 {
    uint32_t abi_version;
    uint32_t kind;
    const char* name;
    const char* version;
    const char* author;
    const char* description;
    uint32_t input_schema_fourcc;
    uint32_t output_schema_fourcc;
} AimPluginDescriptorV1;

/* Plugin Function API Table (populated by plugin for host). The opaque instance is
 * exclusively created/destroyed by the plugin. Input/output buffers are host-owned,
 * borrowed only for process(), and must never be retained or freed by the plugin. */
typedef struct AimPluginApiV1 {
    uint32_t abi_version;
    uint32_t kind;
    AimPluginHandle* instance;

    /* Lifecycle & Execution function pointers */
    uint32_t (*start)(AimPluginHandle* instance, const uint8_t* config_buffer, uint32_t config_size);
    uint32_t (*process)(AimPluginHandle* instance,
                        const uint8_t* input_buffer, uint32_t input_size,
                        uint8_t* output_buffer, uint32_t output_capacity,
                        uint32_t* out_output_size);
    void (*stop)(AimPluginHandle* instance);
    void (*destroy)(AimPluginHandle* instance);
    uint32_t (*get_descriptor)(AimPluginHandle* instance, AimPluginDescriptorV1* out_descriptor);
} AimPluginApiV1;

/* ========================================================================== */
/* DLL Exported Functions Prototypes                                          */
/* ========================================================================== */

/**
 * @brief Returns the packed ABI version supported by this plugin library.
 * Host calls this FIRST before creating any instance.
 */
AIM_PLUGIN_EXPORT uint32_t aim_plugin_abi_version(void);

/**
 * @brief Factory function creating an instance and binding function pointers.
 * @param host_services Pointer to host service table.
 * @param out_api Pointer to API structure to be populated by the plugin.
 * @return AIM_STATUS_OK on success, or an error status code.
 */
AIM_PLUGIN_EXPORT uint32_t aim_plugin_create_v1(
    const AimHostServicesV1* host_services,
    AimPluginApiV1* out_api
);

static inline const char* aim_status_code_to_string(AimStatusCode code) {
    switch (code) {
        case AIM_STATUS_OK: return "AIM_STATUS_OK";
        case AIM_STATUS_ERROR_INVALID_ARGUMENT: return "AIM_STATUS_ERROR_INVALID_ARGUMENT";
        case AIM_STATUS_ERROR_BUFFER_TOO_SMALL: return "AIM_STATUS_ERROR_BUFFER_TOO_SMALL";
        case AIM_STATUS_ERROR_ABI_MISMATCH: return "AIM_STATUS_ERROR_ABI_MISMATCH";
        case AIM_STATUS_ERROR_UNINITIALIZED: return "AIM_STATUS_ERROR_UNINITIALIZED";
        case AIM_STATUS_ERROR_ALREADY_INITIALIZED: return "AIM_STATUS_ERROR_ALREADY_INITIALIZED";
        case AIM_STATUS_ERROR_OUT_OF_BOUNDS: return "AIM_STATUS_ERROR_OUT_OF_BOUNDS";
        case AIM_STATUS_ERROR_CORRUPTED_DATA: return "AIM_STATUS_ERROR_CORRUPTED_DATA";
        case AIM_STATUS_ERROR_EXECUTION_FAILED: return "AIM_STATUS_ERROR_EXECUTION_FAILED";
        case AIM_STATUS_ERROR_HARDWARE_FAULT: return "AIM_STATUS_ERROR_HARDWARE_FAULT";
        case AIM_STATUS_ERROR_EMERGENCY_STOP: return "AIM_STATUS_ERROR_EMERGENCY_STOP";
        case AIM_STATUS_ERROR_NOT_SUPPORTED: return "AIM_STATUS_ERROR_NOT_SUPPORTED";
        default: return "AIM_STATUS_ERROR_UNKNOWN";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* AIM_PLUGIN_PLUGIN_ABI_H_ */
