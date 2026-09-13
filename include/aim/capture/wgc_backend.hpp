// include/aim/capture/wgc_backend.hpp
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include "aim/capture/dxgi_backend.hpp"
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
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

namespace aim::capture {

/// @brief Abstract driver interface decoupling Windows Graphics Capture OS APIs for deterministic testing.
class IWgcBackend : public IGpuTextureAllocator {
public:
    virtual ~IWgcBackend() = default;

    // Initialization and capture session lifecycle
    virtual bool initialize(const FrameSourceConfig& config) noexcept = 0;
    virtual bool create_capture_session() noexcept = 0;
    virtual void release_capture_session() noexcept = 0;
    virtual void release_all() noexcept = 0;

    // Successful acquisition replaces the backend-held frame. A no-frame
    // result preserves it, so a drain-to-latest caller can copy the last valid
    // texture before release_frame(). The raw pointer is otherwise non-owning.
    virtual HRESULT try_get_next_frame(CapturedRawFrame& out_frame) noexcept = 0;
    virtual HRESULT release_frame() noexcept = 0;

    // Staging surface management (GPU-only allocations)
    virtual bool allocate_staging_texture(std::uint32_t width,
                                         std::uint32_t height,
                                         FrameFormat format,
                                         void** out_texture,
                                         std::uint64_t* out_shared_handle) noexcept override = 0;
    virtual void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept override = 0;
    virtual void copy_texture(void* dst, void* src) noexcept = 0;

    // State inspection
    [[nodiscard]] virtual bool is_supported() const noexcept = 0;
    [[nodiscard]] virtual bool is_device_alive() const noexcept = 0;
    [[nodiscard]] virtual bool is_session_active() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t adapter_luid() const noexcept = 0;
    virtual void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept = 0;
};

#if defined(_WIN32) || defined(_MSC_VER)
/// @brief Live Windows Graphics Capture (WGC) hardware backend.
class RealWgcBackend final : public IWgcBackend {
public:
    RealWgcBackend() noexcept;
    ~RealWgcBackend() noexcept override;

    RealWgcBackend(const RealWgcBackend&) = delete;
    RealWgcBackend& operator=(const RealWgcBackend&) = delete;

    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool create_capture_session() noexcept override;
    void release_capture_session() noexcept override;
    void release_all() noexcept override;

    HRESULT try_get_next_frame(CapturedRawFrame& out_frame) noexcept override;
    HRESULT release_frame() noexcept override;

    bool allocate_staging_texture(std::uint32_t width,
                                 std::uint32_t height,
                                 FrameFormat format,
                                 void** out_texture,
                                 std::uint64_t* out_shared_handle) noexcept override;
    void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept override;
    void copy_texture(void* dst, void* src) noexcept override;

    [[nodiscard]] bool is_supported() const noexcept override;
    [[nodiscard]] bool is_device_alive() const noexcept override;
    [[nodiscard]] bool is_session_active() const noexcept override;
    [[nodiscard]] std::uint64_t adapter_luid() const noexcept override { return adapter_luid_; }
    void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_{nullptr};
    std::uint64_t adapter_luid_{0};
    std::uint32_t output_width_{1920};
    std::uint32_t output_height_{1080};
};
#endif

/// @brief Deterministic, scriptable mock WGC backend for CI and fault-injection testing.
class MockWgcBackend final : public IWgcBackend {
public:
    explicit MockWgcBackend(std::shared_ptr<std::atomic<std::size_t>> external_counter = nullptr) noexcept
        : external_staging_counter_(std::move(external_counter)) {}
    ~MockWgcBackend() noexcept override;

    // Scriptable fault injection
    void set_init_result(bool success) noexcept { init_success_ = success; }
    void set_create_session_result(bool success) noexcept { create_session_success_ = success; }
    void set_supported(bool supported) noexcept { is_supported_ = supported; }
    void set_device_alive(bool alive) noexcept { device_alive_ = alive; }
    void set_session_active(bool active) noexcept { session_active_ = active; }
    void set_adapter_luid(std::uint64_t luid) noexcept { adapter_luid_ = luid; }
    void set_output_dimensions(std::uint32_t w, std::uint32_t h) noexcept {
        output_width_ = w;
        output_height_ = h;
    }

    void queue_acquire_result(HRESULT hr,
                              std::uint64_t qpc_timestamp = 0,
                              std::uint32_t width = 1920,
                              std::uint32_t height = 1080,
                              std::size_t count = 1) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            acquire_results_.push({hr, qpc_timestamp, width, height});
        }
        synthetic_frames_enabled_ = false;
    }

    void clear_acquire_queue() noexcept {
        while (!acquire_results_.empty()) {
            acquire_results_.pop();
        }
        synthetic_frames_enabled_ = true;
        synthetic_frame_ready_ = true;
    }

    void set_auto_synthetic_frames(bool enabled) noexcept {
        synthetic_frames_enabled_ = enabled;
        synthetic_frame_ready_ = enabled;
    }

    // Call metrics and verification
    [[nodiscard]] std::size_t initialize_call_count() const noexcept { return init_count_; }
    [[nodiscard]] std::size_t create_session_call_count() const noexcept { return create_session_count_; }
    [[nodiscard]] std::size_t release_session_call_count() const noexcept { return release_session_count_; }
    [[nodiscard]] std::size_t release_all_call_count() const noexcept { return release_all_count_; }
    [[nodiscard]] std::size_t acquire_call_count() const noexcept { return acquire_count_; }
    [[nodiscard]] std::size_t release_frame_call_count() const noexcept { return release_frame_count_; }
    [[nodiscard]] std::size_t copy_texture_call_count() const noexcept { return copy_texture_count_; }
    [[nodiscard]] std::size_t copy_after_release_count() const noexcept { return copy_after_release_count_; }
    [[nodiscard]] std::size_t active_staging_texture_count() const noexcept { return active_staging_textures_; }
    [[nodiscard]] bool is_frame_currently_held() const noexcept { return frame_held_; }

    // IWgcBackend interface
    bool initialize(const FrameSourceConfig& config) noexcept override;
    bool create_capture_session() noexcept override;
    void release_capture_session() noexcept override;
    void release_all() noexcept override;

    HRESULT try_get_next_frame(CapturedRawFrame& out_frame) noexcept override;
    HRESULT release_frame() noexcept override;

    bool allocate_staging_texture(std::uint32_t width,
                                 std::uint32_t height,
                                 FrameFormat format,
                                 void** out_texture,
                                 std::uint64_t* out_shared_handle) noexcept override;
    void free_staging_texture(void* texture, std::uint64_t shared_handle) noexcept override;
    void copy_texture(void* dst, void* src) noexcept override;

    [[nodiscard]] bool is_supported() const noexcept override { return is_supported_; }
    [[nodiscard]] bool is_device_alive() const noexcept override { return device_alive_; }
    [[nodiscard]] bool is_session_active() const noexcept override { return session_active_; }
    [[nodiscard]] std::uint64_t adapter_luid() const noexcept override { return adapter_luid_; }
    void get_output_dimensions(std::uint32_t& out_w, std::uint32_t& out_h) const noexcept override {
        out_w = output_width_;
        out_h = output_height_;
    }

private:
    struct QueuedAcquire {
        HRESULT hr{S_OK};
        std::uint64_t qpc_timestamp{0};
        std::uint32_t width{1920};
        std::uint32_t height{1080};
    };

    bool init_success_{true};
    bool create_session_success_{true};
    bool is_supported_{true};
    bool device_alive_{true};
    bool session_active_{false};
    bool synthetic_frames_enabled_{true};
    bool synthetic_frame_ready_{true};
    std::uint64_t adapter_luid_{0x00010688};
    std::uint32_t output_width_{1920};
    std::uint32_t output_height_{1080};

    std::size_t init_count_{0};
    std::size_t create_session_count_{0};
    std::size_t release_session_count_{0};
    std::size_t release_all_count_{0};
    std::size_t acquire_count_{0};
    std::size_t release_frame_count_{0};
    std::size_t copy_texture_count_{0};
    std::size_t copy_after_release_count_{0};
    std::size_t active_staging_textures_{0};
    bool frame_held_{false};
    std::uint64_t mock_texture_handle_counter_{2000};

    std::queue<QueuedAcquire> acquire_results_{};
    std::uint64_t default_synthetic_qpc_{10'000'000};
    std::shared_ptr<std::atomic<std::size_t>> external_staging_counter_{nullptr};
};

} // namespace aim::capture
