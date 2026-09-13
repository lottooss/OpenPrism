// include/aim/bus/shared_memory_ring.hpp
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "aim/bus/bus_traits.hpp"
#include "aim/bus/types.hpp"

namespace aim::bus {

/// @brief Low-level RAII wrapper for operating-system shared memory mappings.
class SharedMemorySegment {
public:
    SharedMemorySegment() noexcept = default;

    ~SharedMemorySegment() noexcept {
        close();
    }

    SharedMemorySegment(const SharedMemorySegment&) = delete;
    SharedMemorySegment& operator=(const SharedMemorySegment&) = delete;

    SharedMemorySegment(SharedMemorySegment&& other) noexcept
        : name_(std::move(other.name_)),
          size_bytes_(other.size_bytes_),
          is_creator_(other.is_creator_),
          is_read_only_(other.is_read_only_),
#if defined(_WIN32) || defined(_WIN64)
          handle_(other.handle_),
#else
          fd_(other.fd_),
#endif
          base_address_(other.base_address_) {
#if defined(_WIN32) || defined(_WIN64)
        other.handle_ = nullptr;
#else
        other.fd_ = -1;
#endif
        other.base_address_ = nullptr;
        other.size_bytes_ = 0;
        other.is_creator_ = false;
    }

    SharedMemorySegment& operator=(SharedMemorySegment&& other) noexcept {
        if (this != &other) {
            close();
            name_ = std::move(other.name_);
            size_bytes_ = other.size_bytes_;
            is_creator_ = other.is_creator_;
            is_read_only_ = other.is_read_only_;
#if defined(_WIN32) || defined(_WIN64)
            handle_ = other.handle_;
            other.handle_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            base_address_ = other.base_address_;
            other.base_address_ = nullptr;
            other.size_bytes_ = 0;
            other.is_creator_ = false;
        }
        return *this;
    }

    /// @brief Creates or attaches to a named shared memory segment.
    static SharedMemorySegment open_or_create(std::string_view name, std::size_t size_bytes, bool read_only = false) noexcept {
        SharedMemorySegment seg;
        seg.name_ = std::string(name);
        seg.size_bytes_ = size_bytes;
        seg.is_read_only_ = read_only;

#if defined(_WIN32) || defined(_WIN64)
        std::wstring wide_name(name.begin(), name.end());
        const DWORD protect = read_only ? PAGE_READONLY : PAGE_READWRITE;
        const DWORD access = read_only ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS;

        HANDLE hMap = nullptr;
        if (!read_only) {
            ULARGE_INTEGER liSize;
            liSize.QuadPart = static_cast<ULONGLONG>(size_bytes);
            hMap = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                nullptr,
                protect,
                liSize.HighPart,
                liSize.LowPart,
                wide_name.c_str()
            );
            if (hMap != nullptr) {
                seg.is_creator_ = (GetLastError() != ERROR_ALREADY_EXISTS);
            }
        } else {
            hMap = OpenFileMappingW(access, FALSE, wide_name.c_str());
            seg.is_creator_ = false;
        }

        if (hMap == nullptr) {
            return seg;
        }

        void* view = MapViewOfFile(hMap, access, 0, 0, size_bytes);
        if (view == nullptr) {
            CloseHandle(hMap);
            return seg;
        }

        seg.handle_ = hMap;
        seg.base_address_ = view;
        return seg;
#else
        const mode_t mode = 0666;
        int fd = -1;
        if (read_only) {
            fd = shm_open(seg.name_.c_str(), O_RDONLY, mode);
        } else {
            // O_EXCL first so that `is_creator_` means "this call created the segment",
            // matching the Windows ERROR_ALREADY_EXISTS check above. Without it every
            // read-write open claimed creation, so attaching re-initialized a live
            // producer's header and unlinked the name on close.
            fd = shm_open(seg.name_.c_str(), O_CREAT | O_EXCL | O_RDWR, mode);
            if (fd != -1) {
                seg.is_creator_ = true;
            } else {
                fd = shm_open(seg.name_.c_str(), O_RDWR, mode);
            }
        }
        if (fd == -1) {
            return seg;
        }

        if (!read_only) {
            struct stat segment_stat{};
            const bool needs_grow = (fstat(fd, &segment_stat) == -1) ||
                (static_cast<std::size_t>(segment_stat.st_size) < size_bytes);
            if (needs_grow && ftruncate(fd, static_cast<off_t>(size_bytes)) == -1) {
                ::close(fd);
                return seg;
            }
        }

        const int prot = read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
        void* addr = mmap(nullptr, size_bytes, prot, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            return seg;
        }

        seg.fd_ = fd;
        seg.base_address_ = addr;
        return seg;
#endif
    }

    void close() noexcept {
        if (base_address_ != nullptr) {
#if defined(_WIN32) || defined(_WIN64)
            UnmapViewOfFile(base_address_);
            base_address_ = nullptr;
#else
            munmap(base_address_, size_bytes_);
            base_address_ = nullptr;
#endif
        }
#if defined(_WIN32) || defined(_WIN64)
        if (handle_ != nullptr) {
            CloseHandle(handle_);
            handle_ = nullptr;
        }
#else
        if (fd_ != -1) {
            ::close(fd_);
            if (is_creator_) {
                shm_unlink(name_.c_str());
            }
            fd_ = -1;
        }
#endif
        size_bytes_ = 0;
        is_creator_ = false;
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return base_address_ != nullptr;
    }

    [[nodiscard]] void* data() noexcept {
        return base_address_;
    }

    [[nodiscard]] const void* data() const noexcept {
        return base_address_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_bytes_;
    }

    [[nodiscard]] bool is_creator() const noexcept {
        return is_creator_;
    }

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

private:
    std::string name_{};
    std::size_t size_bytes_{0};
    bool is_creator_{false};
    bool is_read_only_{false};

#if defined(_WIN32) || defined(_WIN64)
    HANDLE handle_{nullptr};
#else
    int fd_{-1};
#endif
    void* base_address_{nullptr};
};

/// @brief Inter-process / shared-memory bounded latest-wins SPSC ring buffer.
/// Provides zero-allocation, lock-free seqlock publishing with heartbeat and crash recovery.
template <typename T, std::size_t Capacity = 16>
class SharedMemoryRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

public:
    struct alignas(64) Slot {
        BusSlotHeader header{};
        T data{};
    };

    static constexpr std::size_t kHeaderBytes = sizeof(IpcControlHeader);
    static constexpr std::size_t kSlotBytes = sizeof(Slot);
    static constexpr std::size_t kTotalSegmentBytes = kHeaderBytes + Capacity * kSlotBytes;
    static constexpr std::size_t kIndexMask = Capacity - 1;

    SharedMemoryRing() noexcept = default;

    /// @brief Opens or creates a shared memory ring.
    static SharedMemoryRing create(std::string_view name, std::uint32_t pipeline_run_id = 1) noexcept {
        SharedMemoryRing ring;
        ring.segment_ = SharedMemorySegment::open_or_create(name, kTotalSegmentBytes, false);
        if (!ring.segment_.is_valid()) {
            return ring;
        }

        ring.control_header_ = static_cast<IpcControlHeader*>(ring.segment_.data());
        auto* base_slots = reinterpret_cast<std::uint8_t*>(ring.segment_.data()) + kHeaderBytes;
        ring.slots_ = reinterpret_cast<Slot*>(base_slots);

        if (ring.segment_.is_creator()) {
            ring.initialize_header(pipeline_run_id);
        }
        return ring;
    }

    /// @brief Attaches to an existing shared memory ring in read-only mode.
    static SharedMemoryRing open_read_only(std::string_view name) noexcept {
        SharedMemoryRing ring;
        ring.segment_ = SharedMemorySegment::open_or_create(name, kTotalSegmentBytes, true);
        if (!ring.segment_.is_valid()) {
            return ring;
        }

        ring.control_header_ = static_cast<IpcControlHeader*>(ring.segment_.data());
        auto* base_slots = reinterpret_cast<std::uint8_t*>(ring.segment_.data()) + kHeaderBytes;
        ring.slots_ = reinterpret_cast<Slot*>(base_slots);
        return ring;
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return segment_.is_valid() && control_header_ != nullptr && slots_ != nullptr;
    }

    /// @brief Publishes a message to the shared memory ring buffer.
    BusWriteResult write_latest(const T& item, MonotonicNs timestamp_ns = 0) noexcept {
        if (!is_valid()) return BusWriteResult::uninitialized;

        const std::uint64_t seq = local_producer_seq_ + 1;
        const std::size_t idx = (seq - 1) & kIndexMask;
        auto& slot = slots_[idx];

        // 1. Mark slot as writing (odd)
        slot.header.seq_before.store(seq * 2 - 1, std::memory_order_relaxed);

        // 2. Publish the odd marker before any payload byte. A release *store* only orders
        //    prior operations; without this fence the payload write may sink above the
        //    marker and a torn read becomes undetectable by the reader.
        std::atomic_thread_fence(std::memory_order_release);

        // 3. Copy payload
        slot.data = item;
        slot.header.payload_size = sizeof(T);
        slot.header.magic_identifier = BusPayloadTraits<T>::kMagicIdentifier;
        slot.header.published_at_ns = timestamp_ns;

        // 4. Mark slot complete (even)
        slot.header.seq_after.store(seq * 2, std::memory_order_release);
        slot.header.seq_before.store(seq * 2, std::memory_order_release);

        // 5. Update head sequence and heartbeat
        control_header_->producer_head_seq.store(seq, std::memory_order_release);
        if (timestamp_ns != 0) {
            control_header_->writer_heartbeat_ns.store(timestamp_ns, std::memory_order_release);
        }
        local_producer_seq_ = seq;
        ++total_produced_;
        return BusWriteResult::ok;
    }

    /// @brief Reads the newest available message from shared memory.
    BusReadResult try_read_latest(T& out_item, std::uint64_t& last_consumed_seq, std::uint64_t* out_dropped_count = nullptr) const noexcept {
        if (!is_valid()) return BusReadResult::uninitialized;

        const std::uint64_t head = control_header_->producer_head_seq.load(std::memory_order_acquire);
        if (head <= last_consumed_seq || head == 0) {
            return BusReadResult::no_new_data;
        }

        for (std::size_t retry = 0; retry < 3; ++retry) {
            const std::uint64_t current_head = control_header_->producer_head_seq.load(std::memory_order_acquire);
            if (current_head <= last_consumed_seq || current_head == 0) {
                return BusReadResult::no_new_data;
            }

            const std::size_t idx = (current_head - 1) & kIndexMask;
            const auto& slot = slots_[idx];

            const std::uint64_t s1 = slot.header.seq_before.load(std::memory_order_acquire);
            if ((s1 & 1) != 0 || s1 != current_head * 2) {
                continue; // Slot being written or wrapped
            }

            out_item = slot.data;
            std::atomic_thread_fence(std::memory_order_acquire);

            const std::uint64_t s2 = slot.header.seq_after.load(std::memory_order_acquire);
            if (s1 == s2) {
                if (last_consumed_seq > 0 && current_head > last_consumed_seq + 1) {
                    const std::uint64_t drops = current_head - last_consumed_seq - 1;
                    total_dropped_ += drops;
                    if (out_dropped_count != nullptr) *out_dropped_count = drops;
                } else if (last_consumed_seq == 0 && current_head > 1) {
                    const std::uint64_t drops = current_head - 1;
                    total_dropped_ += drops;
                    if (out_dropped_count != nullptr) *out_dropped_count = drops;
                } else if (out_dropped_count != nullptr) {
                    *out_dropped_count = 0;
                }

                last_consumed_seq = current_head;
                ++total_reads_;
                return BusReadResult::ok;
            }
            ++torn_reads_;
        }

        return BusReadResult::torn_or_overrun;
    }

    /// @brief Reads latest item using internal sequence tracking.
    BusReadResult try_read_latest(T& out_item) noexcept {
        return try_read_latest(out_item, last_consumed_seq_);
    }

    /// @brief Checks whether the publishing writer is alive based on heartbeat nanoseconds.
    /// @note Fails closed. A writer that has never recorded a heartbeat is reported dead, so a
    /// producer must either pass a non-zero timestamp to `write_latest` or call
    /// `update_heartbeat` before a consumer will treat its data as fresh.
    [[nodiscard]] bool is_writer_alive(MonotonicNs current_time_ns, MonotonicNs timeout_ns = 50'000'000) const noexcept {
        if (!is_valid()) return false;
        const MonotonicNs last_hb = control_header_->writer_heartbeat_ns.load(std::memory_order_acquire);
        if (last_hb == 0) return false; // Liveness never proven
        return (current_time_ns - last_hb) <= timeout_ns;
    }

    /// @brief Updates writer heartbeat monotonic timestamp.
    void update_heartbeat(MonotonicNs current_ns) noexcept {
        if (control_header_ != nullptr) {
            control_header_->writer_heartbeat_ns.store(current_ns, std::memory_order_release);
        }
    }

    [[nodiscard]] const IpcControlHeader* control_header() const noexcept {
        return control_header_;
    }

    [[nodiscard]] std::uint64_t latest_sequence() const noexcept {
        return is_valid() ? control_header_->producer_head_seq.load(std::memory_order_acquire) : 0;
    }

    [[nodiscard]] std::uint64_t last_consumed_sequence() const noexcept {
        return last_consumed_seq_;
    }

    [[nodiscard]] constexpr std::size_t capacity() const noexcept {
        return Capacity;
    }

    [[nodiscard]] std::uint64_t dropped_count() const noexcept {
        return total_dropped_;
    }

    [[nodiscard]] std::uint64_t total_reads() const noexcept {
        return total_reads_;
    }

    [[nodiscard]] std::uint64_t total_produced() const noexcept {
        return total_produced_;
    }

    [[nodiscard]] std::uint64_t torn_reads_detected() const noexcept {
        return torn_reads_;
    }

    [[nodiscard]] BusStats stats() const noexcept {
        return BusStats{
            .total_produced = total_produced_,
            .total_consumed = total_reads_,
            .total_dropped = total_dropped_,
            .torn_reads_detected = torn_reads_,
            .last_published_seq = latest_sequence(),
            .last_consumed_seq = last_consumed_seq_,
            .last_published_timestamp_ns = 0,
            .last_consumed_timestamp_ns = 0
        };
    }

private:
    void initialize_header(std::uint32_t pipeline_run_id) noexcept {
        control_header_->magic = kShmRingMagic;
        control_header_->version_major = kShmVersionMajor;
        control_header_->version_minor = kShmVersionMinor;
        control_header_->header_size_bytes = static_cast<std::uint32_t>(sizeof(IpcControlHeader));
        control_header_->channel_id = static_cast<std::uint32_t>(BusPayloadTraits<T>::kChannelId);
        control_header_->pipeline_run_id = pipeline_run_id;
#if defined(_WIN32) || defined(_WIN64)
        control_header_->writer_pid = static_cast<std::uint32_t>(GetCurrentProcessId());
#else
        control_header_->writer_pid = static_cast<std::uint32_t>(getpid());
#endif
        control_header_->ring_capacity = static_cast<std::uint32_t>(Capacity);
        control_header_->slot_stride_bytes = static_cast<std::uint32_t>(sizeof(Slot));
        control_header_->max_payload_bytes = static_cast<std::uint32_t>(sizeof(T));

        control_header_->writer_heartbeat_ns.store(0, std::memory_order_release);
        control_header_->producer_head_seq.store(0, std::memory_order_release);
        control_header_->consumer_tail_seq.store(0, std::memory_order_release);
        control_header_->health_flags.store(static_cast<std::uint32_t>(ShmHealthFlags::writer_active), std::memory_order_release);

        for (std::size_t i = 0; i < Capacity; ++i) {
            slots_[i].header.seq_before.store(0, std::memory_order_release);
            slots_[i].header.seq_after.store(0, std::memory_order_release);
            slots_[i].header.payload_size = 0;
            slots_[i].header.magic_identifier = BusPayloadTraits<T>::kMagicIdentifier;
            slots_[i].header.published_at_ns = 0;
            slots_[i].data = T{};
        }
    }

    SharedMemorySegment segment_{};
    IpcControlHeader* control_header_{nullptr};
    Slot* slots_{nullptr};

    std::uint64_t local_producer_seq_{0};
    mutable std::uint64_t last_consumed_seq_{0};
    mutable std::uint64_t total_reads_{0};
    mutable std::uint64_t total_dropped_{0};
    mutable std::uint64_t torn_reads_{0};
    std::uint64_t total_produced_{0};
};

} // namespace aim::bus
