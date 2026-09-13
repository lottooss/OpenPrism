// src/capture/dxgi_backend.cpp
#include "aim/capture/dxgi_backend.hpp"

namespace aim::capture {

// =============================================================================
// MockDxgiBackend Implementation
// =============================================================================

MockDxgiBackend::~MockDxgiBackend() noexcept {
    release_all();
}

bool MockDxgiBackend::initialize(const FrameSourceConfig& config) noexcept {
    ++init_count_;
    output_width_ = config.target_width_px;
    output_height_ = config.target_height_px;
    if (!init_success_) {
        return false;
    }
    return create_duplication();
}

bool MockDxgiBackend::create_duplication() noexcept {
    ++create_duplication_count_;
    return create_duplication_success_;
}

void MockDxgiBackend::release_duplication() noexcept {
    ++release_duplication_count_;
    frame_held_ = false;
}

void MockDxgiBackend::release_all() noexcept {
    release_duplication();
    if (external_staging_counter_ && active_staging_textures_ > 0) {
        external_staging_counter_->fetch_sub(active_staging_textures_, std::memory_order_relaxed);
    }
    active_staging_textures_ = 0;
}

static std::vector<std::uint8_t> s_mock_dxgi_pixels(1920 * 1080 * 4, 128);

HRESULT MockDxgiBackend::acquire_next_frame(std::uint32_t /*timeout_ms*/, CapturedRawFrame& out_frame) noexcept {
    ++acquire_count_;
    if (!device_alive_) {
        return DXGI_ERROR_DEVICE_REMOVED;
    }

    if (acquire_results_.empty()) {
        frame_held_ = true;
        out_frame.raw_texture = s_mock_dxgi_pixels.data();
        default_synthetic_qpc_ += 69444ULL; // ~144 Hz at 10 MHz QPF (6.944 ms)
        out_frame.last_present_time_qpc = default_synthetic_qpc_;
        out_frame.width = output_width_;
        out_frame.height = output_height_;
        out_frame.accumulated_frames = 1;
        out_frame.is_rects_coalesced = false;
        return S_OK;
    }

    QueuedAcquire next = acquire_results_.front();
    acquire_results_.pop();

    if (next.hr == S_OK) {
        frame_held_ = true;
        out_frame.raw_texture = s_mock_dxgi_pixels.data();
        if (next.qpc_timestamp != 0) {
            out_frame.last_present_time_qpc = next.qpc_timestamp;
        } else {
            default_synthetic_qpc_ += 69444ULL;
            out_frame.last_present_time_qpc = default_synthetic_qpc_;
        }
        out_frame.width = output_width_;
        out_frame.height = output_height_;
        out_frame.accumulated_frames = 1;
        out_frame.is_rects_coalesced = false;
    }

    return next.hr;
}

HRESULT MockDxgiBackend::release_frame() noexcept {
    ++release_frame_count_;
    frame_held_ = false;
    return S_OK;
}

bool MockDxgiBackend::allocate_staging_texture(std::uint32_t /*width*/,
                                              std::uint32_t /*height*/,
                                              FrameFormat /*format*/,
                                              void** out_texture,
                                              std::uint64_t* out_shared_handle) noexcept {
    if (!out_texture || !out_shared_handle) {
        return false;
    }
    ++active_staging_textures_;
    if (external_staging_counter_) {
        external_staging_counter_->fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint64_t handle = mock_texture_handle_counter_++;
    *out_texture = s_mock_dxgi_pixels.data();
    *out_shared_handle = handle + 10000ULL;
    return true;
}

void MockDxgiBackend::free_staging_texture(void* /*texture*/, std::uint64_t /*shared_handle*/) noexcept {
    if (active_staging_textures_ > 0) {
        --active_staging_textures_;
    }
    if (external_staging_counter_ && external_staging_counter_->load(std::memory_order_relaxed) > 0) {
        external_staging_counter_->fetch_sub(1, std::memory_order_relaxed);
    }
}

void MockDxgiBackend::copy_texture(void* /*dst*/, void* /*src*/) noexcept {
    ++copy_texture_count_;
}

// =============================================================================
// RealDxgiBackend Implementation (Windows Only)
// =============================================================================

#if defined(_WIN32) || defined(_MSC_VER)

RealDxgiBackend::~RealDxgiBackend() noexcept {
    release_all();
}

bool RealDxgiBackend::initialize(const FrameSourceConfig& config) noexcept {
    config_ = config;

    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory_));
    if (FAILED(hr)) {
        return false;
    }

    // Select adapter for display_index
    Microsoft::WRL::ComPtr<IDXGIAdapter1> matched_adapter;
    Microsoft::WRL::ComPtr<IDXGIOutput> matched_output;

    UINT adapter_index = 0;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> current_adapter;
    while (factory_->EnumAdapters1(adapter_index++, &current_adapter) != DXGI_ERROR_NOT_FOUND) {
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
                matched_output = current_output;
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
        // Fallback: pick adapter 0
        if (FAILED(factory_->EnumAdapters1(0, &matched_adapter))) {
            return false;
        }
    }

    adapter_ = matched_adapter;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter_->GetDesc1(&adapter_desc);
    adapter_luid_ = (static_cast<std::uint64_t>(adapter_desc.AdapterLuid.HighPart) << 32) |
                    static_cast<std::uint64_t>(adapter_desc.AdapterLuid.LowPart);

    // Create D3D11 Device
    UINT creation_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    const D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
    D3D_FEATURE_LEVEL selected_feature_level{};

    hr = D3D11CreateDevice(
        adapter_.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        creation_flags,
        feature_levels,
        static_cast<UINT>(std::size(feature_levels)),
        D3D11_SDK_VERSION,
        &device_,
        &selected_feature_level,
        &context_
    );

    if (FAILED(hr)) {
        return false;
    }

    return create_duplication();
}

bool RealDxgiBackend::create_duplication() noexcept {
    if (!adapter_ || !device_) {
        return false;
    }

    Microsoft::WRL::ComPtr<IDXGIOutput> output_base;
    HRESULT hr = adapter_->EnumOutputs(config_.display_index, &output_base);
    if (FAILED(hr)) {
        // Fallback to output 0
        hr = adapter_->EnumOutputs(0, &output_base);
        if (FAILED(hr)) {
            return false;
        }
    }

    hr = output_base.As(&output_);
    if (FAILED(hr)) {
        return false;
    }

    DXGI_OUTPUT_DESC out_desc{};
    if (SUCCEEDED(output_->GetDesc(&out_desc))) {
        output_width_ = static_cast<std::uint32_t>(out_desc.DesktopCoordinates.right - out_desc.DesktopCoordinates.left);
        output_height_ = static_cast<std::uint32_t>(out_desc.DesktopCoordinates.bottom - out_desc.DesktopCoordinates.top);
    }

    hr = output_->DuplicateOutput(device_.Get(), &duplication_);
    return SUCCEEDED(hr);
}

void RealDxgiBackend::release_duplication() noexcept {
    if (frame_held_ && duplication_) {
        duplication_->ReleaseFrame();
        frame_held_ = false;
    }
    current_desktop_resource_.Reset();
    duplication_.Reset();
    output_.Reset();
}

void RealDxgiBackend::release_all() noexcept {
    release_duplication();
    if (context_) {
        context_->ClearState();
        context_->Flush();
        context_.Reset();
    }
    device_.Reset();
    adapter_.Reset();
    factory_.Reset();
}

HRESULT RealDxgiBackend::acquire_next_frame(std::uint32_t timeout_ms, CapturedRawFrame& out_frame) noexcept {
    if (!duplication_) {
        return DXGI_ERROR_ACCESS_LOST;
    }

    DXGI_OUTDUPL_FRAME_INFO frame_info{};
    Microsoft::WRL::ComPtr<IDXGIResource> desktop_resource;

    HRESULT hr = duplication_->AcquireNextFrame(timeout_ms, &frame_info, &desktop_resource);
    if (SUCCEEDED(hr)) {
        frame_held_ = true;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> desktop_tex;
        hr = desktop_resource.As(&desktop_tex);
        if (SUCCEEDED(hr)) {
            out_frame.raw_texture = desktop_tex.Get();
            out_frame.last_present_time_qpc = static_cast<std::uint64_t>(frame_info.LastPresentTime.QuadPart);
            out_frame.width = output_width_;
            out_frame.height = output_height_;
            out_frame.accumulated_frames = frame_info.AccumulatedFrames;
            out_frame.is_rects_coalesced = (frame_info.RectsCoalesced != FALSE);
            current_desktop_resource_ = desktop_resource;
        } else {
            duplication_->ReleaseFrame();
            frame_held_ = false;
        }
    }

    return hr;
}

HRESULT RealDxgiBackend::release_frame() noexcept {
    current_desktop_resource_.Reset();
    if (duplication_ && frame_held_) {
        HRESULT hr = duplication_->ReleaseFrame();
        frame_held_ = false;
        return hr;
    }
    return S_OK;
}

bool RealDxgiBackend::allocate_staging_texture(std::uint32_t width,
                                              std::uint32_t height,
                                              FrameFormat format,
                                              void** out_texture,
                                              std::uint64_t* out_shared_handle) noexcept {
    if (!device_ || !out_texture || !out_shared_handle) {
        return false;
    }

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = static_cast<DXGI_FORMAT>(format);
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
    HRESULT hr = device_->CreateTexture2D(&desc, nullptr, &tex);
    if (FAILED(hr)) {
        return false;
    }

    *out_shared_handle = 0;
    Microsoft::WRL::ComPtr<IDXGIResource> res;
    if (SUCCEEDED(tex.As(&res))) {
        HANDLE h = nullptr;
        if (SUCCEEDED(res->GetSharedHandle(&h))) {
            *out_shared_handle = reinterpret_cast<std::uint64_t>(h);
        }
    }

    *out_texture = tex.Detach();
    return true;
}

void RealDxgiBackend::free_staging_texture(void* texture, std::uint64_t /*shared_handle*/) noexcept {
    if (texture) {
        static_cast<IUnknown*>(texture)->Release();
    }
}

void RealDxgiBackend::copy_texture(void* dst, void* src) noexcept {
    if (context_ && dst && src) {
        context_->CopyResource(static_cast<ID3D11Resource*>(dst), static_cast<ID3D11Resource*>(src));
    }
}

bool RealDxgiBackend::is_device_alive() const noexcept {
    if (!device_) {
        return false;
    }
    return (device_->GetDeviceRemovedReason() == S_OK);
}

void RealDxgiBackend::get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept {
    out_w = output_width_;
    out_h = output_height_;
}

#endif // defined(_WIN32) || defined(_MSC_VER)

} // namespace aim::capture
