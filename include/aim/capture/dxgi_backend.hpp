// include/aim/capture/dxgi_backend.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <queue>
#include <vector>
#include "aim/core/frame_source.hpp"

#if defined(_WIN32) || defined(_MSC_VER)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#else
using HRESULT = std::int32_t;
constexpr HRESULT S_OK = 0;
constexpr HRESULT S_FALSE = 1;
constexpr HRESULT E_FAIL = static_cast<HRESULT>(0x80004005);
constexpr HRESULT E_ACCESSDENIED = static_cast<HRESULT>(0x80070005);
constexpr HRESULT E_INVALIDARG = static_cast<HRESULT>(0x80070057);
constexpr HRESULT DXGI_ERROR_INVALID_CALL = static_cast<HRESULT>(0x887A0001);
constexpr HRESULT DXGI_ERROR_UNSUPPORTED = static_cast<HRESULT>(0x887A0004);
constexpr HRESULT DXGI_ERROR_DEVICE_REMOVED = static_cast<HRESULT>(0x887A0005);
constexpr HRESULT DXGI_ERROR_DEVICE_RESET = static_cast<HRESULT>(0x887A0007);
constexpr HRESULT DXGI_ERROR_NOT_CURRENTLY_AVAILABLE = static_cast<HRESULT>(0x887A0022);
constexpr HRESULT DXGI_ERROR_ACCESS_LOST = static_cast<HRESULT>(0x887A0026);
constexpr HRESULT DXGI_ERROR_WAIT_TIMEOUT = static_cast<HRESULT>(0x887A0027);
constexpr HRESULT DXGI_ERROR_ACCESS_DENIED = static_cast<HRESULT>(0x887A002B);
#endif

namespace aim::capture {

/// @brief Represents raw frame metadata and surface returned by DXGI AcquireNextFrame.
struct CapturedRawFrame {
    void* raw_texture{nullptr};
    std::uint64_t last_present_time_qpc{0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::uint32_t accumulated_frames{1};
    bool is_rects_coalesced{false};
};

/// @brief Texture allocator interface for GPU staging surfaces.
class IGpuTextureAllocator {
public:
    virtual ~IGpuTextureAllocator() = default;

    virtual bool allocate_staging_texture(std::uint32_t width,
                                         std::uint32_t height,
                                         FrameFormat format,
                                         void** out_texture,
                                         std::uint64_t* out_shared_handle) noexcept = 0;
    virtual void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept = 0;
};

/// @brief Abstract driver interface decoupling DXGI and D3D11 OS APIs for deterministic fault testing.
class IDxgiBackend : public IGpuTextureAllocator {
public:
    virtual ~IDxgiBackend() = default;

    // Initialization and duplication lifecycle
    virtual bool initialize(const FrameSourceConfig& config) noexcept = 0;
    virtual bool create_duplication() noexcept = 0;
    virtual void release_duplication() noexcept = 0;
    virtual void release_all() noexcept = 0;

    // Frame acquisition and release
    virtual HRESULT acquire_next_frame(std::uint32_t timeout_ms, CapturedRawFrame& out_frame) noexcept = 0;
    virtual HRESULT release_frame() noexcept = 0;

    // Staging surface management (GPU-only allocations)
    virtual bool allocate_staging_texture(std::uint32_t width,
                                         std::uint32_t height,
                                         FrameFormat format,
                                         void** out_texture,
                                         std::uint64_t* out_shared_handle) noexcept = 0;
    virtual void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept = 0;
    virtual void copy_texture(void* dst, void* src) noexcept = 0;

    // State inspection
    [[nodiscard]] virtual bool is_device_alive() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t adapter_luid() const noexcept = 0;
    virtual void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept = 0;
};

#if defined(_WIN32) || defined(_MSC_VER)
/// @brief Live Direct3D 11 and DXGI Desktop Duplication hardware backend.
class RealDxgiBackend final : public IDxgiBackend {
public:
    RealDxgiBackend() noexcept = default;
    ~RealDxgiBackend() noexcept override;

    RealDxgiBackend(const RealDxgiBackend&) = delete;
    RealDxgiBackend& operator=(const RealDxgiBackend&) = delete;

    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool create_duplication() noexcept override;
    void release_duplication() noexcept override;
    void release_all() noexcept override;

    HRESULT acquire_next_frame(std::uint32_t timeout_ms, CapturedRawFrame& out_frame) noexcept override;
    HRESULT release_frame() noexcept override;

    bool allocate_staging_texture(std::uint32_t width,
                                 std::uint32_t height,
                                 FrameFormat format,
                                 void** out_texture,
                                 std::uint64_t* out_shared_handle) noexcept override;
    void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept override;
    void copy_texture(void* dst, void* src) noexcept override;

    [[nodiscard]] bool is_device_alive() const noexcept override;
    [[nodiscard]] std::uint64_t adapter_luid() const noexcept override { return adapter_luid_; }
    void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept override;

private:
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory_{};
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_{};
    Microsoft::WRL::ComPtr<IDXGIOutput1> output_{};
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_{};
    Microsoft::WRL::ComPtr<ID3D11Device> device_{};
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_{};
    Microsoft::WRL::ComPtr<IDXGIResource> current_desktop_resource_{};

    FrameSourceConfig config_{};
    std::uint64_t adapter_luid_{0};
    std::uint32_t output_width_{1920};
    std::uint32_t output_height_{1080};
    bool frame_held_{false};
};
#endif

/// @brief Deterministic, scriptable mock DXGI backend for CI and fault-injection testing.
class MockDxgiBackend final : public IDxgiBackend {
public:
    explicit MockDxgiBackend(std::shared_ptr<std::atomic<std::size_t>> external_counter = nullptr) noexcept
        : external_staging_counter_(std::move(external_counter)) {}
    ~MockDxgiBackend() noexcept override;

    // Scriptable fault injection
    void set_init_result(bool success) noexcept { init_success_ = success; }
    void set_create_duplication_result(bool success) noexcept { create_duplication_success_ = success; }
    void set_device_alive(bool alive) noexcept { device_alive_ = alive; }
    void set_adapter_luid(std::uint64_t luid) noexcept { adapter_luid_ = luid; }
    void set_output_dimensions(std::uint32_t w, std::uint32_t h) noexcept {
        output_width_ = w;
        output_height_ = h;
    }

    void queue_acquire_result(HRESULT hr, std::uint64_t qpc_timestamp = 0, std::size_t count = 1) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            acquire_results_.push({hr, qpc_timestamp});
        }
    }

    void clear_acquire_queue() noexcept {
        while (!acquire_results_.empty()) {
            acquire_results_.pop();
        }
    }

    // Call metrics and verification
    [[nodiscard]] std::size_t initialize_call_count() const noexcept { return init_count_; }
    [[nodiscard]] std::size_t create_duplication_call_count() const noexcept { return create_duplication_count_; }
    [[nodiscard]] std::size_t release_duplication_call_count() const noexcept { return release_duplication_count_; }
    [[nodiscard]] std::size_t acquire_call_count() const noexcept { return acquire_count_; }
    [[nodiscard]] std::size_t release_frame_call_count() const noexcept { return release_frame_count_; }
    [[nodiscard]] std::size_t copy_texture_call_count() const noexcept { return copy_texture_count_; }
    [[nodiscard]] std::size_t active_staging_texture_count() const noexcept { return active_staging_textures_; }
    [[nodiscard]] bool is_frame_currently_held() const noexcept { return frame_held_; }

    // IDxgiBackend interface
    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool create_duplication() noexcept override;
    void release_duplication() noexcept override;
    void release_all() noexcept override;

    HRESULT acquire_next_frame(std::uint32_t timeout_ms, CapturedRawFrame& out_frame) noexcept override;
    HRESULT release_frame() noexcept override;

    bool allocate_staging_texture(std::uint32_t width,
                                 std::uint32_t height,
                                 FrameFormat format,
                                 void** out_texture,
                                 std::uint64_t* out_shared_handle) noexcept override;
    void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept override;
    void copy_texture(void* dst, void* src) noexcept override;

    [[nodiscard]] bool is_device_alive() const noexcept override { return device_alive_; }
    [[nodiscard]] std::uint64_t adapter_luid() const noexcept override { return adapter_luid_; }
    void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept override {
        out_w = output_width_;
        out_h = output_height_;
    }

private:
    struct QueuedAcquire {
        HRESULT hr{S_OK};
        std::uint64_t qpc_timestamp{0};
    };

    bool init_success_{true};
    bool create_duplication_success_{true};
    bool device_alive_{true};
    std::uint64_t adapter_luid_{0x00010688};
    std::uint32_t output_width_{1920};
    std::uint32_t output_height_{1080};

    std::size_t init_count_{0};
    std::size_t create_duplication_count_{0};
    std::size_t release_duplication_count_{0};
    std::size_t acquire_count_{0};
    std::size_t release_frame_count_{0};
    std::size_t copy_texture_count_{0};
    std::size_t active_staging_textures_{0};
    bool frame_held_{false};
    std::uint64_t mock_texture_handle_counter_{1000};

    std::queue<QueuedAcquire> acquire_results_{};
    std::uint64_t default_synthetic_qpc_{10'000'000};
    std::shared_ptr<std::atomic<std::size_t>> external_staging_counter_{nullptr};
};

} // namespace aim::capture
