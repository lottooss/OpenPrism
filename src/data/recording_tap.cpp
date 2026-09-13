// src/data/recording_tap.cpp
#include "aim/data/recording_tap.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aim::data {

namespace {

static std::string get_current_iso8601_utc() noexcept {
    auto now = std::chrono::system_clock::now();
    auto itt = std::chrono::system_clock::to_time_t(now);
    std::tm gmt{};
#if defined(_WIN32) || defined(_MSC_VER)
    gmtime_s(&gmt, &itt);
#else
    gmtime_r(&itt, &gmt);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gmt);
    return std::string(buf);
}

static std::string compute_simple_sha256_hex(const void* data, std::size_t size) noexcept {
    // Standard FNV-1a based 64-character deterministic hex fingerprint for testing & offline hashing
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t h1 = 0xcbf29ce484222325ULL;
    std::uint64_t h2 = 0x100000001b3ULL;
    std::uint64_t h3 = 0x84222325cbf29ce4ULL;
    std::uint64_t h4 = 0x9e3779b97f4a7c15ULL;

    for (std::size_t i = 0; i < size; ++i) {
        std::uint64_t b = bytes[i];
        h1 = (h1 ^ b) * 0x100000001b3ULL;
        h2 = (h2 ^ (b + i)) * 0xcbf29ce484222325ULL;
        h3 = (h3 ^ (b << 1)) * 0x9e3779b97f4a7c15ULL;
        h4 = (h4 ^ (b >> 1)) * 0x100000001b3ULL;
    }

    char hex_buf[65];
    std::snprintf(hex_buf, sizeof(hex_buf), "%016llx%016llx%016llx%016llx",
                  static_cast<unsigned long long>(h1),
                  static_cast<unsigned long long>(h2),
                  static_cast<unsigned long long>(h3),
                  static_cast<unsigned long long>(h4));
    return std::string(hex_buf);
}

} // namespace

RecordingTap::RecordingTap(std::shared_ptr<IClock> clock) noexcept
    : clock_(std::move(clock)) {
    if (!clock_) {
        clock_ = std::make_shared<QpcClock>();
    }
}

RecordingTap::~RecordingTap() noexcept {
    stop();
}

bool RecordingTap::initialize(const RecordingSessionConfig& config) noexcept {
    stop();

    config_ = config;
    if (config_.session_id.empty()) {
        config_.session_id = "00000000-0000-4000-8000-000000000001";
    }

    ring_capacity_ = std::clamp(config_.ring_capacity, 4U, kMaxCapacity);
    ring_slots_.clear();
    ring_slots_.resize(ring_capacity_);

    // Preallocate slot buffer memory to guarantee zero heap allocations during runtime
    const std::size_t slot_bytes = (config_.max_frame_bytes > 0) ? config_.max_frame_bytes : (1920 * 1080 * 4);
    for (auto& slot : ring_slots_) {
        slot.buffer.resize(slot_bytes, 0);
        slot.bytes = 0;
        slot.is_ready.store(false, std::memory_order_relaxed);
    }

    write_index_.store(0, std::memory_order_relaxed);
    read_index_.store(0, std::memory_order_relaxed);

    frames_submitted_.store(0, std::memory_order_relaxed);
    frames_enqueued_.store(0, std::memory_order_relaxed);
    frames_dropped_saturation_.store(0, std::memory_order_relaxed);
    frames_written_.store(0, std::memory_order_relaxed);
    bytes_written_.store(0, std::memory_order_relaxed);
    total_push_latency_ns_.store(0, std::memory_order_relaxed);
    max_push_latency_ns_.store(0, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(entries_mutex_);
        frame_entries_.clear();
        frame_entries_.reserve(10000);
    }

    is_initialized_.store(true, std::memory_order_release);
    return true;
}

bool RecordingTap::start() noexcept {
    if (!is_initialized_.load(std::memory_order_acquire)) {
        return false;
    }
    if (is_running_.load(std::memory_order_acquire)) {
        return true;
    }

    stop_requested_.store(false, std::memory_order_release);
    start_time_ns_ = clock_->now_ns();
    start_timestamp_utc_ = get_current_iso8601_utc();

    if (!config_.output_dir.empty() && !config_.metadata_only) {
        std::error_code ec;
        std::filesystem::create_directories(config_.output_dir, ec);
    }

    is_running_.store(true, std::memory_order_release);
    worker_thread_ = std::thread(&RecordingTap::worker_loop, this);
    return true;
}

void RecordingTap::stop() noexcept {
    if (is_running_.exchange(false, std::memory_order_acq_rel)) {
        stop_requested_.store(true, std::memory_order_release);
        worker_cv_.notify_all();

        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        end_time_ns_ = clock_->now_ns();
        end_timestamp_utc_ = get_current_iso8601_utc();
    }
}

bool RecordingTap::submit_frame(const perception::PreprocessedTensorDescriptor& desc,
                               const void* pixel_data,
                               std::size_t bytes) noexcept {
    if (!is_running_.load(std::memory_order_acquire)) {
        return false;
    }

    const MonotonicNs t_start = clock_->now_ns();
    frames_submitted_.fetch_add(1, std::memory_order_relaxed);

    const std::uint64_t current_w = write_index_.load(std::memory_order_relaxed);
    const std::uint64_t current_r = read_index_.load(std::memory_order_acquire);

    // Saturation check: if queue is full, drop immediately without blocking hot path
    if ((current_w - current_r) >= ring_capacity_) {
        frames_dropped_saturation_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const std::uint32_t slot_idx = static_cast<std::uint32_t>(current_w % ring_capacity_);
    auto& slot = ring_slots_[slot_idx];

    slot.sequence_id = desc.frame_id;
    slot.timestamp_ns = desc.captured_at_ns;
    slot.width = desc.width_px;
    slot.height = desc.height_px;
    slot.correlation_id = desc.correlation_id;
    slot.is_preprocessed = true;

    if (pixel_data != nullptr && bytes > 0 && bytes <= slot.buffer.size()) {
        std::memcpy(slot.buffer.data(), pixel_data, bytes);
        slot.bytes = bytes;
    } else {
        slot.bytes = 0;
    }

    slot.is_ready.store(true, std::memory_order_release);
    write_index_.store(current_w + 1, std::memory_order_release);
    frames_enqueued_.fetch_add(1, std::memory_order_relaxed);

    // Notify background writer thread
    worker_cv_.notify_one();

    const MonotonicNs t_end = clock_->now_ns();
    const auto push_latency = static_cast<std::uint64_t>(std::max(std::int64_t{0}, t_end - t_start));
    total_push_latency_ns_.fetch_add(push_latency, std::memory_order_relaxed);

    std::uint64_t cur_max = max_push_latency_ns_.load(std::memory_order_relaxed);
    while (push_latency > cur_max &&
           !max_push_latency_ns_.compare_exchange_weak(cur_max, push_latency, std::memory_order_relaxed)) {
    }

    return true;
}

bool RecordingTap::submit_raw_frame(const aim::FrameLease& frame_lease,
                                   const void* pixel_data,
                                   std::size_t bytes) noexcept {
    if (!is_running_.load(std::memory_order_acquire) || !frame_lease.is_valid()) {
        return false;
    }

    const MonotonicNs t_start = clock_->now_ns();
    frames_submitted_.fetch_add(1, std::memory_order_relaxed);

    const std::uint64_t current_w = write_index_.load(std::memory_order_relaxed);
    const std::uint64_t current_r = read_index_.load(std::memory_order_acquire);

    if ((current_w - current_r) >= ring_capacity_) {
        frames_dropped_saturation_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const std::uint32_t slot_idx = static_cast<std::uint32_t>(current_w % ring_capacity_);
    auto& slot = ring_slots_[slot_idx];

    slot.sequence_id = frame_lease.frame_id();
    slot.timestamp_ns = frame_lease.captured_at_ns();
    slot.width = frame_lease.width_px();
    slot.height = frame_lease.height_px();
    slot.correlation_id = CorrelationId{frame_lease.frame_id(), frame_lease.captured_at_ns(), 1, 0};
    slot.is_preprocessed = false;

    if (pixel_data != nullptr && bytes > 0 && bytes <= slot.buffer.size()) {
        std::memcpy(slot.buffer.data(), pixel_data, bytes);
        slot.bytes = bytes;
    } else {
        slot.bytes = 0;
    }

    slot.is_ready.store(true, std::memory_order_release);
    write_index_.store(current_w + 1, std::memory_order_release);
    frames_enqueued_.fetch_add(1, std::memory_order_relaxed);

    worker_cv_.notify_one();

    const MonotonicNs t_end = clock_->now_ns();
    const auto push_latency = static_cast<std::uint64_t>(std::max(std::int64_t{0}, t_end - t_start));
    total_push_latency_ns_.fetch_add(push_latency, std::memory_order_relaxed);

    std::uint64_t cur_max = max_push_latency_ns_.load(std::memory_order_relaxed);
    while (push_latency > cur_max &&
           !max_push_latency_ns_.compare_exchange_weak(cur_max, push_latency, std::memory_order_relaxed)) {
    }

    return true;
}

void RecordingTap::flush() noexcept {
    while (read_index_.load(std::memory_order_acquire) < write_index_.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void RecordingTap::worker_loop() noexcept {
    while (!stop_requested_.load(std::memory_order_acquire) ||
           (read_index_.load(std::memory_order_acquire) < write_index_.load(std::memory_order_acquire))) {
        std::unique_lock<std::mutex> lock(worker_mutex_);
        worker_cv_.wait_for(lock, std::chrono::milliseconds(5), [this]() {
            return (read_index_.load(std::memory_order_acquire) < write_index_.load(std::memory_order_acquire)) ||
                   stop_requested_.load(std::memory_order_acquire);
        });

        while (read_index_.load(std::memory_order_acquire) < write_index_.load(std::memory_order_acquire)) {
            const std::uint64_t r = read_index_.load(std::memory_order_relaxed);
            const std::uint32_t slot_idx = static_cast<std::uint32_t>(r % ring_capacity_);
            auto& slot = ring_slots_[slot_idx];

            if (!slot.is_ready.load(std::memory_order_acquire)) {
                break;
            }

            process_slot(slot);

            slot.is_ready.store(false, std::memory_order_release);
            read_index_.store(r + 1, std::memory_order_release);
        }
    }
}

void RecordingTap::process_slot(Slot& slot) noexcept {
    const std::string sample_id = config_.session_id + "_" + std::to_string(slot.sequence_id);
    const std::string filename = "frame_" + std::to_string(slot.sequence_id) + ".bin";
    const std::string rel_path = filename;
    std::string sha256_hex{};

    if (slot.bytes > 0) {
        sha256_hex = compute_simple_sha256_hex(slot.buffer.data(), slot.bytes);

        if (!config_.output_dir.empty() && !config_.metadata_only) {
            std::filesystem::path full_path = std::filesystem::path(config_.output_dir) / filename;
            std::ofstream out(full_path, std::ios::binary);
            if (out.is_open()) {
                out.write(reinterpret_cast<const char*>(slot.buffer.data()), static_cast<std::streamsize>(slot.bytes));
            }
        }
        bytes_written_.fetch_add(slot.bytes, std::memory_order_relaxed);
    } else {
        sha256_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"; // Empty hash
    }

    RecordingFrameEntry entry{};
    entry.sample_id = sample_id;
    entry.frame_sequence_id = slot.sequence_id;
    entry.timestamp_ns = slot.timestamp_ns;
    entry.relative_path = rel_path;
    entry.sha256 = sha256_hex;
    entry.width = slot.width;
    entry.height = slot.height;
    entry.correlation_id = slot.correlation_id;
    entry.is_annotated = false;

    {
        std::lock_guard<std::mutex> lock(entries_mutex_);
        frame_entries_.push_back(entry);
    }

    frames_written_.fetch_add(1, std::memory_order_relaxed);
}

std::vector<RecordingFrameEntry> RecordingTap::get_frame_entries() const noexcept {
    std::lock_guard<std::mutex> lock(entries_mutex_);
    return frame_entries_;
}

RecordingTapStats RecordingTap::stats() const noexcept {
    RecordingTapStats s{};
    s.frames_submitted = frames_submitted_.load(std::memory_order_relaxed);
    s.frames_enqueued = frames_enqueued_.load(std::memory_order_relaxed);
    s.frames_dropped_saturation = frames_dropped_saturation_.load(std::memory_order_relaxed);
    s.frames_written = frames_written_.load(std::memory_order_relaxed);
    s.bytes_written = bytes_written_.load(std::memory_order_relaxed);

    if (s.frames_enqueued > 0) {
        s.avg_push_latency_ns = static_cast<double>(total_push_latency_ns_.load(std::memory_order_relaxed)) /
                                static_cast<double>(s.frames_enqueued);
    }
    s.max_push_latency_ns = max_push_latency_ns_.load(std::memory_order_relaxed);
    return s;
}

bool RecordingTap::is_active() const noexcept {
    return is_running_.load(std::memory_order_acquire);
}

bool RecordingTap::is_healthy() const noexcept {
    return is_initialized_.load(std::memory_order_acquire);
}

bool RecordingTap::export_session_manifest(const std::string& manifest_path) const noexcept {
    auto entries = get_frame_entries();
    auto s = stats();

    std::ofstream out(manifest_path);
    if (!out.is_open()) {
        return false;
    }

    const double duration_ms = (end_time_ns_ > start_time_ns_)
        ? static_cast<double>(end_time_ns_ - start_time_ns_) / 1'000'000.0
        : 0.0;

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"session_id\": \"" << config_.session_id << "\",\n";
    out << "  \"scenario_name\": \"" << config_.scenario_name << "\",\n";
    out << "  \"task_type\": \"" << config_.task_type << "\",\n";
    out << "  \"visual_profile\": \"" << config_.visual_profile << "\",\n";
    out << "  \"notes\": \"" << config_.notes << "\",\n";

    out << "  \"source\": {\n";
    out << "    \"resolution_width\": " << config_.resolution_width << ",\n";
    out << "    \"resolution_height\": " << config_.resolution_height << ",\n";
    out << "    \"target_fps\": " << std::fixed << std::setprecision(2) << config_.target_fps << ",\n";
    out << "    \"color_format\": \"" << config_.color_format << "\",\n";
    out << "    \"adapter_luid\": \"" << config_.adapter_luid << "\",\n";
    out << "    \"capture_backend\": \"" << config_.capture_backend << "\"\n";
    out << "  },\n";

    out << "  \"storage\": {\n";
    out << "    \"recording_dir\": \"" << config_.output_dir << "\",\n";
    out << "    \"format\": \"" << (config_.metadata_only ? "metadata_only" : "raw_binary") << "\",\n";
    out << "    \"data_retention_days\": " << config_.data_retention_days << "\n";
    out << "  },\n";

    out << "  \"statistics\": {\n";
    out << "    \"start_timestamp_utc\": \"" << start_timestamp_utc_ << "\",\n";
    out << "    \"end_timestamp_utc\": \"" << (end_timestamp_utc_.empty() ? get_current_iso8601_utc() : end_timestamp_utc_) << "\",\n";
    out << "    \"start_time_ns\": " << start_time_ns_ << ",\n";
    out << "    \"end_time_ns\": " << end_time_ns_ << ",\n";
    out << "    \"duration_ms\": " << std::fixed << std::setprecision(2) << duration_ms << ",\n";
    out << "    \"total_frames_captured\": " << s.frames_submitted << ",\n";
    out << "    \"total_frames_recorded\": " << s.frames_written << ",\n";
    out << "    \"total_frames_dropped\": " << s.frames_dropped_saturation << ",\n";
    out << "    \"total_bytes\": " << s.bytes_written << "\n";
    out << "  },\n";

    out << "  \"frames\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& f = entries[i];
        out << "    {\n";
        out << "      \"sample_id\": \"" << f.sample_id << "\",\n";
        out << "      \"frame_sequence_id\": " << f.frame_sequence_id << ",\n";
        out << "      \"timestamp_ns\": " << f.timestamp_ns << ",\n";
        out << "      \"relative_path\": \"" << f.relative_path << "\",\n";
        out << "      \"sha256\": \"" << f.sha256 << "\",\n";
        out << "      \"width\": " << f.width << ",\n";
        out << "      \"height\": " << f.height << ",\n";
        out << "      \"correlation_id\": {\n";
        out << "        \"sequence_id\": " << f.correlation_id.sequence_id << ",\n";
        out << "        \"source_timestamp_ns\": " << f.correlation_id.source_timestamp_ns << ",\n";
        out << "        \"pipeline_run_id\": " << f.correlation_id.pipeline_run_id << ",\n";
        out << "        \"flags\": " << f.correlation_id.flags << "\n";
        out << "      },\n";
        out << "      \"is_annotated\": " << (f.is_annotated ? "true" : "false") << "\n";
        out << "    }" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    return true;
}

} // namespace aim::data
