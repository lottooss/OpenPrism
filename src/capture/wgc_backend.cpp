// src/capture/wgc_backend.cpp
#include "aim/capture/wgc_backend.hpp"
#include <algorithm>
#include <cstring>

#if defined(_WIN32) || defined(_MSC_VER)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <unknwn.h>
#include <wrl/client.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

namespace aim::capture {

struct MonitorEnumContext {
    std::uint32_t target_index{0};
    std::uint32_t current_index{0};
    HMONITOR found_monitor{nullptr};
};

static BOOL CALLBACK EnumMonProc(HMONITOR hmon, HDC /*hdc*/, LPRECT /*rect*/, LPARAM lparam) {
    auto* ctx = reinterpret_cast<MonitorEnumContext*>(lparam);
    if (ctx->current_index == ctx->target_index) {
        ctx->found_monitor = hmon;
        return FALSE; // Stop enumeration
    }
    ++ctx->current_index;
    return TRUE;
}

struct RealWgcBackend::Impl {
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory{};
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter{};
    Microsoft::WRL::ComPtr<ID3D11Device> device{};
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context{};
    Microsoft::WRL::ComPtr<ID3D11Texture2D> current_texture{};

    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrt_device_{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame current_frame{nullptr};

    HMONITOR monitor_handle{nullptr};
    bool is_session_active{false};
    bool apartment_initialized{false};
    bool frame_held{false};
};

RealWgcBackend::RealWgcBackend() noexcept : impl_(std::make_unique<Impl>()) {}

RealWgcBackend::~RealWgcBackend() noexcept {
    release_all();
}

bool RealWgcBackend::initialize(const FrameSourceConfig& config) noexcept {
    try {
        release_all();

        try {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            impl_->apartment_initialized = true;
        } catch (...) {
            // Already initialized on thread
        }

        if (!is_supported()) {
            return false;
        }

        HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&impl_->factory));
        if (FAILED(hr) || !impl_->factory) {
            return false;
        }

        Microsoft::WRL::ComPtr<IDXGIAdapter1> matched_adapter;
        UINT adapter_index = 0;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> current_adapter;
        while (impl_->factory->EnumAdapters1(adapter_index++, &current_adapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 desc{};
            current_adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                current_adapter.Reset();
                continue;
            }

            UINT output_index = 0;
            Microsoft::WRL::ComPtr<IDXGIOutput> current_output;
            while (current_adapter->EnumOutputs(output_index++, &current_output) != DXGI_ERROR_NOT_FOUND) {
                if (output_index - 1 == config.display_index) {
                    matched_adapter = current_adapter;
                    break;
                }
                current_output.Reset();
            }

            if (matched_adapter) {
                break;
            }
            current_adapter.Reset();
        }

        if (!matched_adapter) {
            adapter_index = 0;
            while (impl_->factory->EnumAdapters1(adapter_index++, &current_adapter) != DXGI_ERROR_NOT_FOUND) {
                DXGI_ADAPTER_DESC1 desc{};
                current_adapter->GetDesc1(&desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    matched_adapter = current_adapter;
                    break;
                }
                current_adapter.Reset();
            }
        }

        if (!matched_adapter) {
            if (FAILED(impl_->factory->EnumAdapters1(0, &matched_adapter)) || !matched_adapter) {
                return false;
            }
        }

        impl_->adapter = matched_adapter;

        DXGI_ADAPTER_DESC1 adapter_desc{};
        if (SUCCEEDED(impl_->adapter->GetDesc1(&adapter_desc))) {
            adapter_luid_ = (static_cast<std::uint64_t>(adapter_desc.AdapterLuid.HighPart) << 32) |
                            static_cast<std::uint64_t>(adapter_desc.AdapterLuid.LowPart);
        }

        D3D_FEATURE_LEVEL feature_levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        D3D_FEATURE_LEVEL feature_level{};

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        hr = D3D11CreateDevice(
            impl_->adapter.Get(),
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            flags,
            feature_levels,
            ARRAYSIZE(feature_levels),
            D3D11_SDK_VERSION,
            &impl_->device,
            &feature_level,
            &impl_->context
        );

        if (FAILED(hr) || !impl_->device || !impl_->context) {
            return false;
        }

        Microsoft::WRL::ComPtr<ID3D11Multithread> multithread;
        if (SUCCEEDED(impl_->device.As(&multithread))) {
            multithread->SetMultithreadProtected(TRUE);
        }

        Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
        hr = impl_->device.As(&dxgi_device);
        if (FAILED(hr) || !dxgi_device) {
            return false;
        }

        winrt::com_ptr<::IInspectable> inspectable_device;
        hr = CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable_device.put());
        if (FAILED(hr) || !inspectable_device) {
            return false;
        }

        impl_->winrt_device_ = inspectable_device.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
        if (!impl_->winrt_device_) {
            return false;
        }

        MonitorEnumContext enum_ctx{};
        enum_ctx.target_index = config.display_index;
        EnumDisplayMonitors(nullptr, nullptr, EnumMonProc, reinterpret_cast<LPARAM>(&enum_ctx));

        if (!enum_ctx.found_monitor) {
            impl_->monitor_handle = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
        } else {
            impl_->monitor_handle = enum_ctx.found_monitor;
        }

        MONITORINFOEXW mon_info{};
        mon_info.cbSize = sizeof(MONITORINFOEXW);
        if (GetMonitorInfoW(impl_->monitor_handle, &mon_info)) {
            output_width_ = static_cast<std::uint32_t>(mon_info.rcMonitor.right - mon_info.rcMonitor.left);
            output_height_ = static_cast<std::uint32_t>(mon_info.rcMonitor.bottom - mon_info.rcMonitor.top);
        } else {
            output_width_ = config.target_width_px;
            output_height_ = config.target_height_px;
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool RealWgcBackend::create_capture_session() noexcept {
    try {
        if (!impl_->winrt_device_ || !impl_->monitor_handle) {
            return false;
        }

        release_capture_session();

        auto interop_factory = winrt::get_activation_factory<
            winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
            IGraphicsCaptureItemInterop>();

        winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item{nullptr};
        HRESULT hr = interop_factory->CreateForMonitor(
            impl_->monitor_handle,
            winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
            winrt::put_abi(capture_item)
        );

        if (FAILED(hr) || !capture_item) {
            return false;
        }

        impl_->capture_item = capture_item;
        auto item_size = capture_item.Size();
        output_width_ = static_cast<std::uint32_t>(item_size.Width);
        output_height_ = static_cast<std::uint32_t>(item_size.Height);

        impl_->frame_pool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            impl_->winrt_device_,
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            item_size
        );

        if (!impl_->frame_pool) {
            return false;
        }

        impl_->session = impl_->frame_pool.CreateCaptureSession(capture_item);
        if (!impl_->session) {
            return false;
        }

        try {
            impl_->session.IsBorderRequired(false);
        } catch (...) {}

        try {
            impl_->session.IsCursorCaptureEnabled(false);
        } catch (...) {}

        impl_->session.StartCapture();
        impl_->is_session_active = true;
        return true;
    } catch (...) {
        return false;
    }
}

void RealWgcBackend::release_capture_session() noexcept {
    try {
        release_frame();
        if (impl_->session) {
            try { impl_->session.Close(); } catch (...) {}
            impl_->session = nullptr;
        }
        if (impl_->frame_pool) {
            try { impl_->frame_pool.Close(); } catch (...) {}
            impl_->frame_pool = nullptr;
        }
        impl_->capture_item = nullptr;
        impl_->is_session_active = false;
    } catch (...) {}
}

void RealWgcBackend::release_all() noexcept {
    try {
        release_capture_session();
        impl_->winrt_device_ = nullptr;
        impl_->current_texture.Reset();
        impl_->context.Reset();
        impl_->device.Reset();
        impl_->adapter.Reset();
        impl_->factory.Reset();
        impl_->monitor_handle = nullptr;
    } catch (...) {}
}

HRESULT RealWgcBackend::try_get_next_frame(CapturedRawFrame& out_frame) noexcept {
    try {
        if (!impl_->frame_pool || !impl_->is_session_active) {
            return DXGI_ERROR_INVALID_CALL;
        }

        auto frame = impl_->frame_pool.TryGetNextFrame();
        if (!frame) {
            return DXGI_ERROR_WAIT_TIMEOUT;
        }

        auto surface = frame.Surface();
        if (!surface) {
            frame.Close();
            return E_FAIL;
        }

        auto dxgi_access = surface.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        if (!dxgi_access) {
            frame.Close();
            return E_FAIL;
        }

        Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d_tex;
        HRESULT hr = dxgi_access->GetInterface(IID_PPV_ARGS(&d3d_tex));
        if (FAILED(hr) || !d3d_tex) {
            frame.Close();
            return hr;
        }

        // Preserve the previous frame on a timeout/error. Its texture remains
        // the drain loop's latest candidate until a valid replacement exists.
        const auto ticks_100ns = frame.SystemRelativeTime().count();

        const auto content_size = frame.ContentSize();
        release_frame();
        out_frame.width = static_cast<std::uint32_t>(content_size.Width);
        out_frame.height = static_cast<std::uint32_t>(content_size.Height);
        out_frame.last_present_time_qpc = static_cast<std::uint64_t>(ticks_100ns);
        out_frame.raw_texture = d3d_tex.Get();
        out_frame.accumulated_frames = 1;
        out_frame.is_rects_coalesced = false;

        impl_->current_frame = frame;
        impl_->current_texture = d3d_tex;
        impl_->frame_held = true;
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT RealWgcBackend::release_frame() noexcept {
    try {
        if (impl_->frame_held) {
            if (impl_->current_frame) {
                try { impl_->current_frame.Close(); } catch (...) {}
                impl_->current_frame = nullptr;
            }
            impl_->current_texture.Reset();
            impl_->frame_held = false;
        }
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

bool RealWgcBackend::allocate_staging_texture(std::uint32_t width,
                                              std::uint32_t height,
                                              FrameFormat format,
                                              void** out_texture,
                                              std::uint64_t* out_shared_handle) noexcept {
    if (!impl_->device || !out_texture || !out_shared_handle || width == 0 || height == 0) {
        return false;
    }

    *out_texture = nullptr;
    *out_shared_handle = 0;

    DXGI_FORMAT dxgi_fmt = (format == FrameFormat::r8g8b8a8_unorm)
                               ? DXGI_FORMAT_R8G8B8A8_UNORM
                               : DXGI_FORMAT_B8G8R8A8_UNORM;

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = dxgi_fmt;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = impl_->device->CreateTexture2D(&desc, nullptr, &texture);
    if (FAILED(hr) || !texture) {
        desc.MiscFlags = 0;
        hr = impl_->device->CreateTexture2D(&desc, nullptr, &texture);
        if (FAILED(hr) || !texture) {
            return false;
        }
    }

    HANDLE shared_h = nullptr;
    if (desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE) {
        Microsoft::WRL::ComPtr<IDXGIResource1> dxgi_res;
        if (SUCCEEDED(texture.As(&dxgi_res))) {
            dxgi_res->CreateSharedHandle(
                nullptr,
                DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                nullptr,
                &shared_h
            );
        }
    } else if (desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED) {
        Microsoft::WRL::ComPtr<IDXGIResource> dxgi_res;
        if (SUCCEEDED(texture.As(&dxgi_res))) {
            dxgi_res->GetSharedHandle(&shared_h);
        }
    }

    *out_texture = texture.Detach();
    *out_shared_handle = reinterpret_cast<std::uint64_t>(shared_h);
    return true;
}

void RealWgcBackend::free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept {
    if (shared_handle != 0) {
        HANDLE h = reinterpret_cast<HANDLE>(shared_handle);
        CloseHandle(h);
    }
    if (texture) {
        auto* tex = static_cast<ID3D11Texture2D*>(texture);
        tex->Release();
    }
}

void RealWgcBackend::copy_texture(void* dst, void* src) noexcept {
    if (impl_->context && dst && src) {
        impl_->context->CopyResource(
            static_cast<ID3D11Resource*>(dst),
            static_cast<ID3D11Resource*>(src)
        );
    }
}

bool RealWgcBackend::is_supported() const noexcept {
    try {
        return winrt::Windows::Graphics::Capture::GraphicsCaptureSession::IsSupported();
    } catch (...) {
        return false;
    }
}

bool RealWgcBackend::is_device_alive() const noexcept {
    if (!impl_->device) return false;
    HRESULT hr = impl_->device->GetDeviceRemovedReason();
    return hr == S_OK;
}

bool RealWgcBackend::is_session_active() const noexcept {
    return impl_->is_session_active;
}

void RealWgcBackend::get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept {
    out_w = output_width_;
    out_h = output_height_;
}

} // namespace aim::capture
#endif

namespace aim::capture {

static std::vector<std::uint8_t> s_mock_wgc_pixels(1920 * 1080 * 4, 128);

MockWgcBackend::~MockWgcBackend() noexcept {
    release_all();
}

bool MockWgcBackend::initialize(const FrameSourceConfig& config) noexcept {
    ++init_count_;
    if (!init_success_) return false;
    output_width_ = config.target_width_px;
    output_height_ = config.target_height_px;
    return true;
}

bool MockWgcBackend::create_capture_session() noexcept {
    ++create_session_count_;
    if (!create_session_success_) return false;
    session_active_ = true;
    return true;
}

void MockWgcBackend::release_capture_session() noexcept {
    ++release_session_count_;
    session_active_ = false;
    release_frame();
}

void MockWgcBackend::release_all() noexcept {
    ++release_all_count_;
    release_capture_session();
}

HRESULT MockWgcBackend::try_get_next_frame(CapturedRawFrame& out_frame) noexcept {
    ++acquire_count_;

    if (!session_active_) {
        return DXGI_ERROR_INVALID_CALL;
    }

    if (!acquire_results_.empty()) {
        auto next = acquire_results_.front();
        acquire_results_.pop();

        if (next.hr == S_OK) {
            if (frame_held_) release_frame();
            out_frame.raw_texture = s_mock_wgc_pixels.data();
            out_frame.last_present_time_qpc = (next.qpc_timestamp != 0) ? next.qpc_timestamp : ++default_synthetic_qpc_;
            out_frame.width = next.width;
            out_frame.height = next.height;
            out_frame.accumulated_frames = 1;
            out_frame.is_rects_coalesced = false;
            frame_held_ = true;
        } else {
            out_frame.raw_texture = nullptr;
        }

        return next.hr;
    }

    // Queue is empty: if auto synthetic frames are enabled, yield 1 frame per release cycle
    if (synthetic_frames_enabled_) {
        if (!synthetic_frame_ready_) {
            return DXGI_ERROR_WAIT_TIMEOUT;
        }
        default_synthetic_qpc_ += 69444; // ~144Hz increment
        out_frame.raw_texture = s_mock_wgc_pixels.data();
        out_frame.last_present_time_qpc = default_synthetic_qpc_;
        out_frame.width = output_width_;
        out_frame.height = output_height_;
        out_frame.accumulated_frames = 1;
        out_frame.is_rects_coalesced = false;
        frame_held_ = true;
        synthetic_frame_ready_ = false;
        return S_OK;
    }

    return DXGI_ERROR_WAIT_TIMEOUT;
}

HRESULT MockWgcBackend::release_frame() noexcept {
    ++release_frame_count_;
    frame_held_ = false;
    synthetic_frame_ready_ = true;
    return S_OK;
}

bool MockWgcBackend::allocate_staging_texture(std::uint32_t /*width*/,
                                              std::uint32_t /*height*/,
                                              FrameFormat /*format*/,
                                              void** out_texture,
                                              std::uint64_t* out_shared_handle) noexcept {
    if (!out_texture || !out_shared_handle) return false;
    const auto handle = ++mock_texture_handle_counter_;
    *out_texture = s_mock_wgc_pixels.data();
    *out_shared_handle = handle * 10;
    ++active_staging_textures_;
    if (external_staging_counter_) {
        external_staging_counter_->fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void MockWgcBackend::free_staging_texture(void* /*texture*/, std::uint64_t /*shared_handle*/) noexcept {
    if (active_staging_textures_ > 0) {
        --active_staging_textures_;
    }
    if (external_staging_counter_) {
        external_staging_counter_->fetch_sub(1, std::memory_order_relaxed);
    }
}

void MockWgcBackend::copy_texture(void* /*dst*/, void* /*src*/) noexcept {
    ++copy_texture_count_;
    if (!frame_held_) ++copy_after_release_count_;
}

} // namespace aim::capture
