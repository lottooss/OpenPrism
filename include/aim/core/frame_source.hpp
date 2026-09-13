// include/aim/core/frame_source.hpp
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim {

enum class FrameSourceBackend : std::uint8_t {
    dxgi_duplication = 0,
    windows_graphics_capture = 1,
    synthetic_simulator = 2,
    external_shm = 3
};

enum class FrameFormat : std::uint32_t {
    unknown = 0,
    b8g8r8a8_unorm = 87,     // DXGI_FORMAT_B8G8R8A8_UNORM
    r8g8b8a8_unorm = 28,     // DXGI_FORMAT_R8G8B8A8_UNORM
    nv12 = 103               // DXGI_FORMAT_NV12
};

struct FrameSourceConfig {
    FrameSourceBackend backend{FrameSourceBackend::dxgi_duplication};
    std::uint32_t display_index{0};
    std::uint32_t target_width_px{1920};
    std::uint32_t target_height_px{1080};
    std::uint32_t expected_refresh_hz{144};
    std::uint32_t pool_capacity{4};
    std::uint32_t timeout_ms{16};
    bool allow_cross_adapter_copy{false};
    bool fallback_to_wgc{true};
};

struct FrameSourceHealth {
    bool is_active{false};
    bool is_access_lost{false};
    bool is_cross_adapter{false};
    std::uint64_t total_frames_acquired{0};
    std::uint64_t total_frames_dropped{0};
    std::uint64_t total_access_loss_events{0};
    std::uint64_t total_timeouts{0};
    double current_cadence_fps{0.0};
    MonotonicNs last_frame_timestamp_ns{0};
    std::array<char, 32> active_adapter_luid{};
};

/// @brief Non-allocating callback interface to return a surface to the pool upon lease destruction.
class ISurfacePoolReleaser {
public:
    virtual ~ISurfacePoolReleaser() = default;
    virtual void release_surface(std::uint32_t pool_slot_index) noexcept = 0;
};

/// @brief RAII move-only lease representing an acquired GPU frame.
class FrameLease {
public:
    constexpr FrameLease() noexcept = default;

    FrameLease(SequenceId frame_id,
               MonotonicNs captured_at_ns,
               std::uint32_t width_px,
               std::uint32_t height_px,
               FrameFormat format,
               void* native_texture_ptr,
               std::uint32_t pool_slot_index,
               ISurfacePoolReleaser* releaser = nullptr) noexcept
        : frame_id_(frame_id),
          captured_at_ns_(captured_at_ns),
          width_px_(width_px),
          height_px_(height_px),
          format_(format),
          native_texture_ptr_(native_texture_ptr),
          pool_slot_index_(pool_slot_index),
          releaser_(releaser),
          is_valid_(true) {}

    ~FrameLease() noexcept {
        reset();
    }

    FrameLease(const FrameLease&) = delete;
    FrameLease& operator=(const FrameLease&) = delete;

    FrameLease(FrameLease&& other) noexcept
        : frame_id_(other.frame_id_),
          captured_at_ns_(other.captured_at_ns_),
          width_px_(other.width_px_),
          height_px_(other.height_px_),
          format_(other.format_),
          native_texture_ptr_(other.native_texture_ptr_),
          pool_slot_index_(other.pool_slot_index_),
          releaser_(other.releaser_),
          is_valid_(other.is_valid_) {
        other.is_valid_ = false;
        other.native_texture_ptr_ = nullptr;
        other.releaser_ = nullptr;
    }

    FrameLease& operator=(FrameLease&& other) noexcept {
        if (this != &other) {
            reset();

            frame_id_ = other.frame_id_;
            captured_at_ns_ = other.captured_at_ns_;
            width_px_ = other.width_px_;
            height_px_ = other.height_px_;
            format_ = other.format_;
            native_texture_ptr_ = other.native_texture_ptr_;
            pool_slot_index_ = other.pool_slot_index_;
            releaser_ = other.releaser_;
            is_valid_ = other.is_valid_;

            other.is_valid_ = false;
            other.native_texture_ptr_ = nullptr;
            other.releaser_ = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (is_valid_ && releaser_ != nullptr) {
            releaser_->release_surface(pool_slot_index_);
        }
        is_valid_ = false;
        native_texture_ptr_ = nullptr;
        releaser_ = nullptr;
    }

    [[nodiscard]] constexpr bool is_valid() const noexcept { return is_valid_; }
    [[nodiscard]] constexpr SequenceId frame_id() const noexcept { return frame_id_; }
    [[nodiscard]] constexpr MonotonicNs captured_at_ns() const noexcept { return captured_at_ns_; }
    [[nodiscard]] constexpr std::uint32_t width_px() const noexcept { return width_px_; }
    [[nodiscard]] constexpr std::uint32_t height_px() const noexcept { return height_px_; }
    [[nodiscard]] constexpr FrameFormat format() const noexcept { return format_; }
    [[nodiscard]] void* native_texture_ptr() const noexcept { return native_texture_ptr_; }
    [[nodiscard]] constexpr std::uint32_t pool_slot_index() const noexcept { return pool_slot_index_; }

private:
    SequenceId frame_id_{0};
    MonotonicNs captured_at_ns_{0};
    std::uint32_t width_px_{0};
    std::uint32_t height_px_{0};
    FrameFormat format_{FrameFormat::unknown};
    void* native_texture_ptr_{nullptr};
    std::uint32_t pool_slot_index_{0};
    ISurfacePoolReleaser* releaser_{nullptr};
    bool is_valid_{false};
};

/// @brief Primary abstraction for capture frame acquisition.
class IFrameSource {
public:
    virtual ~IFrameSource() = default;
    virtual bool initialize(const FrameSourceConfig& config) noexcept = 0;
    virtual bool start() noexcept = 0;
    virtual bool try_acquire_latest(FrameLease& out_lease) noexcept = 0;
    virtual void stop() noexcept = 0;
    [[nodiscard]] virtual FrameSourceHealth health() const noexcept = 0;
};

} // namespace aim
