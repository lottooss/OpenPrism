// include/aim/plugin/plugin_loader.hpp
#pragma once

#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "aim/plugin/plugin_abi.h"

namespace aim::plugin {

using AbiVersionFn = uint32_t (*)(void);
using PluginCreateFn = uint32_t (*)(const AimHostServicesV1*, AimPluginApiV1*);

namespace detail {

inline constexpr bool is_known_status(uint32_t status) noexcept {
    return status <= static_cast<uint32_t>(AIM_STATUS_ERROR_NOT_SUPPORTED);
}

inline constexpr AimStatusCode checked_status(uint32_t status) noexcept {
    return is_known_status(status)
        ? static_cast<AimStatusCode>(status)
        : AIM_STATUS_ERROR_CORRUPTED_DATA;
}

inline bool invoke_version(AbiVersionFn function, uint32_t* out_version) noexcept {
    if (function == nullptr || out_version == nullptr) return false;
#if defined(_MSC_VER)
    __try {
        *out_version = function();
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#else
    try {
        *out_version = function();
        return true;
    } catch (...) {
        return false;
    }
#endif
}

inline bool invoke_create(PluginCreateFn function, const AimHostServicesV1* host_services,
                          AimPluginApiV1* out_api, uint32_t* out_status) noexcept {
    if (function == nullptr || out_api == nullptr || out_status == nullptr) return false;
#if defined(_MSC_VER)
    __try {
        *out_status = function(host_services, out_api);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#else
    try {
        *out_status = function(host_services, out_api);
        return true;
    } catch (...) {
        return false;
    }
#endif
}

inline bool invoke_descriptor(const AimPluginApiV1& api,
                              AimPluginDescriptorV1* out_descriptor,
                              uint32_t* out_status) noexcept {
    if (api.get_descriptor == nullptr || api.instance == nullptr ||
        out_descriptor == nullptr || out_status == nullptr) return false;
#if defined(_MSC_VER)
    __try {
        *out_status = api.get_descriptor(api.instance, out_descriptor);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
#else
    try {
        *out_status = api.get_descriptor(api.instance, out_descriptor);
        return true;
    } catch (...) {
        return false;
    }
#endif
}

inline void invoke_destroy(const AimPluginApiV1& api) noexcept {
    if (api.destroy == nullptr || api.instance == nullptr) return;
#if defined(_MSC_VER)
    __try {
        api.destroy(api.instance);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    try {
        api.destroy(api.instance);
    } catch (...) {
    }
#endif
}

} // namespace detail

class PluginInstance {
public:
    PluginInstance(void* module_handle, const AimPluginApiV1& api) noexcept
        : module_handle_(module_handle), api_(api), is_active_(false) {}

    ~PluginInstance() noexcept {
        unload();
    }

    PluginInstance(const PluginInstance&) = delete;
    PluginInstance& operator=(const PluginInstance&) = delete;

    PluginInstance(PluginInstance&& other) noexcept
        : module_handle_(other.module_handle_),
          api_(other.api_),
          is_active_(other.is_active_) {
        other.module_handle_ = nullptr;
        other.api_ = {};
        other.is_active_ = false;
    }

    PluginInstance& operator=(PluginInstance&& other) noexcept {
        if (this != &other) {
            unload();
            module_handle_ = other.module_handle_;
            api_ = other.api_;
            is_active_ = other.is_active_;
            other.module_handle_ = nullptr;
            other.api_ = {};
            other.is_active_ = false;
        }
        return *this;
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return module_handle_ != nullptr && api_.instance != nullptr;
    }

    [[nodiscard]] bool is_active() const noexcept {
        return is_active_;
    }

    [[nodiscard]] AimPluginKind kind() const noexcept {
        return static_cast<AimPluginKind>(api_.kind);
    }

    [[nodiscard]] const AimPluginApiV1& api() const noexcept {
        return api_;
    }

    AimStatusCode start(const uint8_t* config_data, uint32_t config_size) noexcept {
        if (!is_valid() || api_.start == nullptr) {
            return AIM_STATUS_ERROR_UNINITIALIZED;
        }
        if (config_size > 0 && config_data == nullptr) {
            return AIM_STATUS_ERROR_INVALID_ARGUMENT;
        }
        if (is_active_) {
            return AIM_STATUS_ERROR_ALREADY_INITIALIZED;
        }

#if defined(_MSC_VER)
        __try {
            const uint32_t res = api_.start(api_.instance, config_data, config_size);
            if (res == AIM_STATUS_OK) {
                is_active_ = true;
            }
            return detail::checked_status(res);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#else
        try {
            const uint32_t res = api_.start(api_.instance, config_data, config_size);
            if (res == AIM_STATUS_OK) {
                is_active_ = true;
            }
            return detail::checked_status(res);
        } catch (...) {
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#endif
    }

    AimStatusCode process(const uint8_t* in_data, uint32_t in_size,
                          uint8_t* out_data, uint32_t out_cap, uint32_t* out_size) noexcept {
        if (!is_valid() || !is_active_ || api_.process == nullptr) {
            return AIM_STATUS_ERROR_UNINITIALIZED;
        }
        if (!out_size || (in_size > 0 && in_data == nullptr) ||
            (out_cap > 0 && out_data == nullptr)) {
            return AIM_STATUS_ERROR_INVALID_ARGUMENT;
        }

        *out_size = 0;

#if defined(_MSC_VER)
        __try {
            const uint32_t res = api_.process(api_.instance, in_data, in_size, out_data, out_cap, out_size);
            if (*out_size > out_cap && res != AIM_STATUS_ERROR_BUFFER_TOO_SMALL &&
                res != AIM_STATUS_ERROR_OUT_OF_BOUNDS) {
                return AIM_STATUS_ERROR_CORRUPTED_DATA;
            }
            return detail::checked_status(res);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            *out_size = 0;
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#else
        try {
            const uint32_t res = api_.process(api_.instance, in_data, in_size, out_data, out_cap, out_size);
            if (*out_size > out_cap && res != AIM_STATUS_ERROR_BUFFER_TOO_SMALL &&
                res != AIM_STATUS_ERROR_OUT_OF_BOUNDS) {
                return AIM_STATUS_ERROR_CORRUPTED_DATA;
            }
            return detail::checked_status(res);
        } catch (...) {
            *out_size = 0;
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#endif
    }

    void stop() noexcept {
        if (is_valid() && is_active_ && api_.stop != nullptr) {
#if defined(_MSC_VER)
            __try {
                api_.stop(api_.instance);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
            }
#else
            try {
                api_.stop(api_.instance);
            } catch (...) {
            }
#endif
            is_active_ = false;
        }
    }

    AimStatusCode get_descriptor(AimPluginDescriptorV1* out_descriptor) noexcept {
        if (!is_valid() || api_.get_descriptor == nullptr || !out_descriptor) {
            return AIM_STATUS_ERROR_INVALID_ARGUMENT;
        }
        *out_descriptor = {};
#if defined(_MSC_VER)
        __try {
            const uint32_t res = api_.get_descriptor(api_.instance, out_descriptor);
            return detail::checked_status(res);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#else
        try {
            const uint32_t res = api_.get_descriptor(api_.instance, out_descriptor);
            return detail::checked_status(res);
        } catch (...) {
            return AIM_STATUS_ERROR_EXECUTION_FAILED;
        }
#endif
    }

    void unload() noexcept {
        stop();
        if (api_.destroy != nullptr && api_.instance != nullptr) {
#if defined(_MSC_VER)
            __try {
                api_.destroy(api_.instance);
            } __except (EXCEPTION_EXECUTE_HANDLER) {
            }
#else
            try {
                api_.destroy(api_.instance);
            } catch (...) {
            }
#endif
            api_.instance = nullptr;
        }
        if (module_handle_ != nullptr) {
#if defined(_WIN32)
            ::FreeLibrary(static_cast<HMODULE>(module_handle_));
#else
            ::dlclose(module_handle_);
#endif
            module_handle_ = nullptr;
        }
    }

private:
    void* module_handle_{nullptr};
    AimPluginApiV1 api_{};
    bool is_active_{false};
};

class PluginLoader {
public:
    static std::unique_ptr<PluginInstance> load(
        std::string_view path,
        const AimHostServicesV1* host_services,
        AimPluginKind expected_kind = AIM_PLUGIN_KIND_UNKNOWN,
        AimStatusCode* out_status = nullptr) noexcept
    {
        auto set_status = [out_status](AimStatusCode code) {
            if (out_status) *out_status = code;
        };

        if (path.empty()) {
            set_status(AIM_STATUS_ERROR_INVALID_ARGUMENT);
            return nullptr;
        }

        if (host_services == nullptr ||
            AIM_PLUGIN_ABI_MAJOR(host_services->abi_version) !=
                AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1) ||
            path.find('\0') != std::string_view::npos) {
            set_status(AIM_STATUS_ERROR_INVALID_ARGUMENT);
            return nullptr;
        }

#if defined(_WIN32)
        HMODULE handle = ::LoadLibraryA(std::string(path).c_str());
        if (!handle) {
            set_status(AIM_STATUS_ERROR_INVALID_ARGUMENT);
            return nullptr;
        }

        auto version_fn = reinterpret_cast<AbiVersionFn>(reinterpret_cast<void*>(::GetProcAddress(handle, "aim_plugin_abi_version")));
        auto create_fn = reinterpret_cast<PluginCreateFn>(reinterpret_cast<void*>(::GetProcAddress(handle, "aim_plugin_create_v1")));
#else
        void* handle = ::dlopen(std::string(path).c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            set_status(AIM_STATUS_ERROR_INVALID_ARGUMENT);
            return nullptr;
        }

        auto version_fn = reinterpret_cast<AbiVersionFn>(::dlsym(handle, "aim_plugin_abi_version"));
        auto create_fn = reinterpret_cast<PluginCreateFn>(::dlsym(handle, "aim_plugin_create_v1"));
#endif

        if (!version_fn || !create_fn) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_CORRUPTED_DATA);
            return nullptr;
        }

        uint32_t plugin_abi = 0;
        if (!detail::invoke_version(version_fn, &plugin_abi)) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_EXECUTION_FAILED);
            return nullptr;
        }
        if (AIM_PLUGIN_ABI_MAJOR(plugin_abi) != AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1)) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_ABI_MISMATCH);
            return nullptr;
        }

        AimPluginApiV1 api{};
        uint32_t create_res = AIM_STATUS_ERROR_EXECUTION_FAILED;
        if (!detail::invoke_create(create_fn, host_services, &api, &create_res)) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_EXECUTION_FAILED);
            return nullptr;
        }
        if (create_res != AIM_STATUS_OK || api.instance == nullptr) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(detail::checked_status(create_res));
            return nullptr;
        }

        const bool api_shape_valid =
            AIM_PLUGIN_ABI_MAJOR(api.abi_version) ==
                AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1) &&
            api.kind >= static_cast<uint32_t>(AIM_PLUGIN_KIND_PERCEPTION) &&
            api.kind <= static_cast<uint32_t>(AIM_PLUGIN_KIND_SCENARIO) &&
            api.start != nullptr && api.process != nullptr && api.stop != nullptr &&
            api.destroy != nullptr && api.get_descriptor != nullptr;
        if (!api_shape_valid) {
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_CORRUPTED_DATA);
            return nullptr;
        }

        if (expected_kind != AIM_PLUGIN_KIND_UNKNOWN &&
            api.kind != static_cast<uint32_t>(expected_kind)) {
            detail::invoke_destroy(api);
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_INVALID_ARGUMENT);
            return nullptr;
        }

        AimPluginDescriptorV1 descriptor{};
        uint32_t descriptor_res = AIM_STATUS_ERROR_EXECUTION_FAILED;
        const bool descriptor_call_ok =
            detail::invoke_descriptor(api, &descriptor, &descriptor_res);
        const bool descriptor_valid = descriptor_call_ok && descriptor_res == AIM_STATUS_OK &&
            AIM_PLUGIN_ABI_MAJOR(descriptor.abi_version) ==
                AIM_PLUGIN_ABI_MAJOR(AIM_PLUGIN_ABI_VERSION_V1) &&
            descriptor.kind == api.kind && descriptor.name != nullptr &&
            descriptor.version != nullptr;
        if (!descriptor_valid) {
            detail::invoke_destroy(api);
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_CORRUPTED_DATA);
            return nullptr;
        }

        set_status(AIM_STATUS_OK);
        auto* instance = new (std::nothrow) PluginInstance(handle, api);
        if (instance == nullptr) {
            detail::invoke_destroy(api);
#if defined(_WIN32)
            ::FreeLibrary(handle);
#else
            ::dlclose(handle);
#endif
            set_status(AIM_STATUS_ERROR_EXECUTION_FAILED);
            return nullptr;
        }
        return std::unique_ptr<PluginInstance>(instance);
    }
};

} // namespace aim::plugin
