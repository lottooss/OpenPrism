// include/aim/bus/types.hpp
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::bus {

constexpr std::size_t kCacheLineBytes = 64;
constexpr std::uint32_t kShmRingMagic = 0x524D5341; // 'ASMR' ("Aim Shared Memory Ring")
constexpr std::uint16_t kShmVersionMajor = 1;
constexpr std::uint16_t kShmVersionMinor = 0;

/// @brief Identifier for canonical bus channels.
enum class ChannelId : std::uint32_t {
    unknown             = 0,
    frame_descriptor    = 1,
    target_observation  = 2,
    tracked_target      = 3,
    aim_intent          = 4,
    actuation_command   = 5,
    telemetry_event     = 6
};

/// @brief Health flags for shared memory communication channel.
enum class ShmHealthFlags : std::uint32_t {
    ok                  = 0,
    writer_active       = 1 << 0,
    writer_stale        = 1 << 1,
    writer_crashed      = 1 << 2,
    buffer_overrun      = 1 << 3,
    schema_mismatch     = 1 << 4
};

/// @brief Status result from ring read operations.
enum class BusReadResult : std::uint8_t {
    ok                  = 0,
    no_new_data         = 1,
    torn_or_overrun     = 2,
    invalid_size        = 3,
    uninitialized       = 4
};

/// @brief Status result from ring write operations.
enum class BusWriteResult : std::uint8_t {
    ok                  = 0,
    payload_too_large   = 1,
    uninitialized       = 2
};

/// @brief Monotonic sequence and generation tracker.
struct MonotonicSequence {
    std::uint64_t sequence{0};
    MonotonicNs timestamp_ns{0};

    [[nodiscard]] constexpr bool is_newer_than(const MonotonicSequence& other) const noexcept {
        return sequence > other.sequence;
    }
};

/// @brief Per-slot synchronization header for lock-free double-sequence validation (SeqLock).
struct alignas(64) BusSlotHeader {
    alignas(64) std::atomic<std::uint64_t> seq_before{0};
    std::uint32_t payload_size{0};
    std::uint32_t magic_identifier{0};
    MonotonicNs published_at_ns{0};
    std::atomic<std::uint64_t> seq_after{0};
    std::uint8_t reserved[32]{0}; // Pad to 64 bytes total
};

static_assert(sizeof(BusSlotHeader) == 64, "BusSlotHeader must be exactly 64 bytes (1 cacheline)");
static_assert(alignof(BusSlotHeader) == 64, "BusSlotHeader must be 64-byte aligned");

/// @brief Shared Memory IPC segment control header.
struct alignas(64) IpcControlHeader {
    // Cacheline 0 (0x00 - 0x3F): Immutable geometry and identity
    std::uint32_t magic{kShmRingMagic};
    std::uint16_t version_major{kShmVersionMajor};
    std::uint16_t version_minor{kShmVersionMinor};
    std::uint32_t header_size_bytes{sizeof(IpcControlHeader)};
    std::uint32_t channel_id{0};
    std::uint32_t pipeline_run_id{0};
    std::uint32_t writer_pid{0};
    std::uint32_t ring_capacity{0};     // Must be power-of-2
    std::uint32_t slot_stride_bytes{0}; // Total slot stride including header
    std::uint32_t max_payload_bytes{0}; // slot_stride - sizeof(BusSlotHeader)
    std::uint8_t reserved0[28]{0};      // Pad to 64 bytes

    // Cacheline 1 (0x40 - 0x7F): Atomic synchronization and telemetry counters
    alignas(64) std::atomic<std::int64_t> writer_heartbeat_ns{0};
    std::atomic<std::uint64_t> producer_head_seq{0};
    std::atomic<std::uint64_t> consumer_tail_seq{0};
    std::atomic<std::uint32_t> health_flags{0};
    std::uint8_t reserved1[36]{0};      // Pad to 128 bytes total
};

static_assert(sizeof(IpcControlHeader) == 128, "IpcControlHeader must be exactly 128 bytes (2 cachelines)");
static_assert(alignof(IpcControlHeader) == 64, "IpcControlHeader must be 64-byte aligned");

/// @brief Statistical summary of bus throughput, drops, and latency.
struct BusStats {
    std::uint64_t total_produced{0};
    std::uint64_t total_consumed{0};
    std::uint64_t total_dropped{0};
    std::uint64_t torn_reads_detected{0};
    std::uint64_t last_published_seq{0};
    std::uint64_t last_consumed_seq{0};
    MonotonicNs last_published_timestamp_ns{0};
    MonotonicNs last_consumed_timestamp_ns{0};
};

} // namespace aim::bus
