// include/aim/data/recording_tap.hpp
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "aim/core/clock.hpp"
#include "aim/core/frame_source.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"
#include "aim/perception/preprocess_types.hpp"

namespace aim::data {

/// @brief Configuration specifying recording session metadata, storage directories, and ring capacity.
struct RecordingSessionConfig {
    std::string session_id{}; // UUIDv4
    std::string scenario_name{"gridshot"};
    std::string task_type{"flick"}; // "flick", "tracking", "switching", "synthetic", "calibration", "custom"
    std::string visual_profile{"default"};
    std::string notes{""};
    std::string output_dir{"recordings"};
    std::string color_format{"b8g8r8a8_unorm"};
    std::string capture_backend{"dxgi_duplication"};
    std::uint32_t resolution_width{1920};
    std::uint32_t resolution_height{1080};
    double target_fps{144.0};
    std::string adapter_luid{"0x0000000000000000"};
    std::uint32_t data_retention_days{30};

    std::uint32_t ring_capacity{64};
    bool drop_on_full{true};
    bool metadata_only{false};
    std::size_t max_frame_bytes{1920 * 1080 * 4};
};

/// @brief Frame entry indexed during a recording session.
struct RecordingFrameEntry {
    std::string sample_id{};
    std::uint64_t frame_sequence_id{0};
    MonotonicNs timestamp_ns{0};
    std::string relative_path{};
    std::string sha256{};
    std::uint32_t width{0};
    std::uint32_t height{0};
    CorrelationId correlation_id{};
    std::string label_path{""};
    std::string label_sha256{""};
    bool is_annotated{false};
};

/// @brief Telemetry and performance metrics for the asynchronous recording tap.
struct RecordingTapStats {
    std::uint64_t frames_submitted{0};
    std::uint64_t frames_enqueued{0};
    std::uint64_t frames_dropped_saturation{0};
    std::uint64_t frames_written{0};
    std::uint64_t bytes_written{0};
    double avg_push_latency_ns{0.0};
    std::uint64_t max_push_latency_ns{0};
};

/// @brief Asynchronous, zero-backpressure recording tap for offline datasets and telemetry.
class RecordingTap {
public:
    static constexpr std::uint32_t kDefaultCapacity = 64;
    static constexpr std::uint32_t kMaxCapacity = 256;

    explicit RecordingTap(std::shared_ptr<IClock> clock = nullptr) noexcept;
    ~RecordingTap() noexcept;

    RecordingTap(const RecordingTap&) = delete;
    RecordingTap& operator=(const RecordingTap&) = delete;
    RecordingTap(RecordingTap&&) = delete;
    RecordingTap& operator=(RecordingTap&&) = delete;

    /// @brief Initialize the recording session and preallocate buffer slots.
    bool initialize(const RecordingSessionConfig& config) noexcept;

    /// @brief Start the asynchronous background writer thread.
    bool start() noexcept;

    /// @brief Stop the recording tap, drain queue, and write session manifest.
    void stop() noexcept;

    /// @brief Non-blocking frame submission on the hot path. Drops frame if queue is full.
    bool submit_frame(const perception::PreprocessedTensorDescriptor& desc,
                      const void* pixel_data,
                      std::size_t bytes) noexcept;

    /// @brief Non-blocking raw frame submission on the hot path. Drops frame if queue is full.
    bool submit_raw_frame(const aim::FrameLease& frame_lease,
                          const void* pixel_data,
                          std::size_t bytes) noexcept;

    /// @brief Wait until all enqueued frames are written to disk.
    void flush() noexcept;

    /// @brief Export the final JSON session manifest to disk.
    bool export_session_manifest(const std::string& manifest_path) const noexcept;

    /// @brief Return recorded frame index metadata.
    [[nodiscard]] std::vector<RecordingFrameEntry> get_frame_entries() const noexcept;

    /// @brief Query recording statistics and backpressure telemetry.
    [[nodiscard]] RecordingTapStats stats() const noexcept;

    /// @brief Check if the recording tap is currently active.
    [[nodiscard]] bool is_active() const noexcept;

    /// @brief Check if the recording tap is healthy and initialized.
    [[nodiscard]] bool is_healthy() const noexcept;

private:
    struct Slot {
        std::uint64_t sequence_id{0};
        MonotonicNs timestamp_ns{0};
        std::uint32_t width{0};
        std::uint32_t height{0};
        CorrelationId correlation_id{};
        std::vector<std::uint8_t> buffer{};
        std::size_t bytes{0};
        bool is_preprocessed{false};
        std::atomic<bool> is_ready{false};

        Slot() noexcept = default;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&& other) noexcept
            : sequence_id(other.sequence_id),
              timestamp_ns(other.timestamp_ns),
              width(other.width),
              height(other.height),
              correlation_id(other.correlation_id),
              buffer(std::move(other.buffer)),
              bytes(other.bytes),
              is_preprocessed(other.is_preprocessed),
              is_ready(other.is_ready.load(std::memory_order_relaxed)) {}
        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                sequence_id = other.sequence_id;
                timestamp_ns = other.timestamp_ns;
                width = other.width;
                height = other.height;
                correlation_id = other.correlation_id;
                buffer = std::move(other.buffer);
                bytes = other.bytes;
                is_preprocessed = other.is_preprocessed;
                is_ready.store(other.is_ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
            return *this;
        }
    };

    void worker_loop() noexcept;
    void process_slot(Slot& slot) noexcept;

    RecordingSessionConfig config_{};
    std::shared_ptr<IClock> clock_{nullptr};

    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_running_{false};
    std::atomic<bool> stop_requested_{false};

    // Preallocated ring buffer
    std::vector<Slot> ring_slots_{};
    std::uint32_t ring_capacity_{kDefaultCapacity};
    std::atomic<std::uint64_t> write_index_{0};
    std::atomic<std::uint64_t> read_index_{0};

    // Background worker synchronization
    std::thread worker_thread_{};
    std::mutex worker_mutex_{};
    std::condition_variable worker_cv_{};

    // Telemetry counters
    mutable std::atomic<std::uint64_t> frames_submitted_{0};
    mutable std::atomic<std::uint64_t> frames_enqueued_{0};
    mutable std::atomic<std::uint64_t> frames_dropped_saturation_{0};
    mutable std::atomic<std::uint64_t> frames_written_{0};
    mutable std::atomic<std::uint64_t> bytes_written_{0};
    mutable std::atomic<std::uint64_t> total_push_latency_ns_{0};
    mutable std::atomic<std::uint64_t> max_push_latency_ns_{0};

    // Session timestamps
    MonotonicNs start_time_ns_{0};
    MonotonicNs end_time_ns_{0};
    std::string start_timestamp_utc_{};
    std::string end_timestamp_utc_{};

    // Frame index record
    mutable std::mutex entries_mutex_{};
    std::vector<RecordingFrameEntry> frame_entries_{};
};

} // namespace aim::data
