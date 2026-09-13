// include/aim/bus/latest_spsc_ring.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include "aim/bus/types.hpp"

namespace aim::bus {

/// @brief Allocation-free bounded Single-Producer Single-Consumer (SPSC) ring buffer
/// with monotonic latest-message-wins semantics and lock-free seqlock synchronization.
///
/// Guarantees:
/// - Wait-free O(1) non-blocking producer overwrite.
/// - Cache-line isolation (`alignas(64)`) to eliminate false sharing.
/// - Atomic generation counter (SeqLock) detecting and rejecting torn reads.
/// - Zero heap allocations after initialization.
///
/// @note The payload copy is a deliberate seqlock race: the producer may overwrite `slot.data`
/// while the consumer is copying it. That access is not a data race the C++ memory model
/// blesses, so ThreadSanitizer reports it on `slot.data`. Correctness comes from the
/// surrounding generation counter -- the consumer re-reads it after an acquire fence and
/// discards any value whose generation moved. The producer-side release fence below is what
/// makes that detection sound; do not replace it with a plain release store.
template <typename T, std::size_t Capacity = 16>
class LatestSpscRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(Capacity >= 2, "Capacity must be at least 2");
    static_assert(std::is_nothrow_destructible_v<T>, "T must be nothrow destructible");

public:
    using value_type = T;
    using size_type = std::size_t;

    struct alignas(64) Slot {
        alignas(64) std::atomic<std::uint64_t> sequence{0};
        T data{};
    };

    struct alignas(64) ProducerState {
        alignas(64) std::atomic<std::uint64_t> head_sequence{0};
        std::uint64_t local_sequence{0};
        std::uint64_t total_produced{0};
        std::uint8_t padding[64 - sizeof(std::atomic<std::uint64_t>) - sizeof(std::uint64_t) * 2];
    };

    struct alignas(64) ConsumerState {
        std::uint64_t last_consumed_sequence{0};
        std::uint64_t total_reads{0};
        std::uint64_t dropped_messages{0};
        std::uint64_t torn_reads{0};
        std::uint8_t padding[64 - sizeof(std::uint64_t) * 4];
    };

    LatestSpscRing() noexcept = default;
    ~LatestSpscRing() noexcept = default;

    LatestSpscRing(const LatestSpscRing&) = delete;
    LatestSpscRing& operator=(const LatestSpscRing&) = delete;
    LatestSpscRing(LatestSpscRing&&) = delete;
    LatestSpscRing& operator=(LatestSpscRing&&) = delete;

    /// @brief Publishes an item to the ring, overwriting the oldest slot if full.
    /// @param item Data item to copy into the slot.
    /// @return Published monotonic sequence number.
    std::uint64_t push(const T& item) noexcept {
        const std::uint64_t seq = producer_.local_sequence + 1;
        const std::size_t idx = (seq - 1) & kIndexMask;
        auto& slot = slots_[idx];

        // 1. Mark slot as writing (odd generation)
        slot.sequence.store(seq * 2 - 1, std::memory_order_relaxed);

        // 2. Publish the odd marker before any payload byte. A release *store* only orders
        //    prior operations; without this fence the compiler may sink the payload write
        //    above the marker, making a torn read undetectable by the consumer.
        std::atomic_thread_fence(std::memory_order_release);

        // 3. Write payload
        slot.data = item;

        // 4. Mark slot as complete (even generation)
        slot.sequence.store(seq * 2, std::memory_order_release);

        // 5. Publish latest head sequence to consumer
        producer_.head_sequence.store(seq, std::memory_order_release);
        producer_.local_sequence = seq;
        ++producer_.total_produced;
        return seq;
    }

    /// @brief Publishes an item using move semantics.
    std::uint64_t push(T&& item) noexcept {
        const std::uint64_t seq = producer_.local_sequence + 1;
        const std::size_t idx = (seq - 1) & kIndexMask;
        auto& slot = slots_[idx];

        slot.sequence.store(seq * 2 - 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        slot.data = std::move(item);
        slot.sequence.store(seq * 2, std::memory_order_release);

        producer_.head_sequence.store(seq, std::memory_order_release);
        producer_.local_sequence = seq;
        ++producer_.total_produced;
        return seq;
    }

    /// @brief In-place construct an item in the next ring slot.
    template <typename... Args>
    std::uint64_t emplace(Args&&... args) noexcept {
        const std::uint64_t seq = producer_.local_sequence + 1;
        const std::size_t idx = (seq - 1) & kIndexMask;
        auto& slot = slots_[idx];

        slot.sequence.store(seq * 2 - 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        slot.data = T(std::forward<Args>(args)...);
        slot.sequence.store(seq * 2, std::memory_order_release);

        producer_.head_sequence.store(seq, std::memory_order_release);
        producer_.local_sequence = seq;
        ++producer_.total_produced;
        return seq;
    }

    /// @brief Reads the latest available message if newer than last_consumed_seq.
    /// @param out_item Destination buffer for the copied message.
    /// @param last_consumed_seq In/out parameter tracking consumer's last seen sequence.
    /// @param out_dropped_count Optional pointer to receive dropped message count since last read.
    /// @return true if a new untorn message was successfully acquired, false otherwise.
    bool try_read_latest(T& out_item, std::uint64_t& last_consumed_seq, std::uint64_t* out_dropped_count = nullptr) const noexcept {
        const std::uint64_t head = producer_.head_sequence.load(std::memory_order_acquire);
        if (head <= last_consumed_seq || head == 0) {
            return false;
        }

        for (std::size_t retry = 0; retry < kMaxReadRetries; ++retry) {
            const std::uint64_t current_head = producer_.head_sequence.load(std::memory_order_acquire);
            if (current_head <= last_consumed_seq || current_head == 0) {
                return false;
            }

            const std::size_t idx = (current_head - 1) & kIndexMask;
            const auto& slot = slots_[idx];

            const std::uint64_t s1 = slot.sequence.load(std::memory_order_acquire);
            if ((s1 & 1) != 0 || s1 != current_head * 2) {
                continue; // Slot being written or wrapped by producer
            }

            out_item = slot.data;
            std::atomic_thread_fence(std::memory_order_acquire);

            const std::uint64_t s2 = slot.sequence.load(std::memory_order_relaxed);
            if (s1 == s2) {
                // Untorn read verified
                if (last_consumed_seq > 0 && current_head > last_consumed_seq + 1) {
                    const std::uint64_t drops = current_head - last_consumed_seq - 1;
                    consumer_.dropped_messages += drops;
                    if (out_dropped_count != nullptr) {
                        *out_dropped_count = drops;
                    }
                } else if (last_consumed_seq == 0 && current_head > 1) {
                    const std::uint64_t drops = current_head - 1;
                    consumer_.dropped_messages += drops;
                    if (out_dropped_count != nullptr) {
                        *out_dropped_count = drops;
                    }
                } else if (out_dropped_count != nullptr) {
                    *out_dropped_count = 0;
                }

                last_consumed_seq = current_head;
                ++consumer_.total_reads;
                return true;
            }

            ++consumer_.torn_reads;
        }

        return false;
    }

    /// @brief Reads the latest available message using internal consumer state.
    bool try_read_latest(T& out_item) noexcept {
        return try_read_latest(out_item, consumer_.last_consumed_sequence);
    }

    /// @brief Pops the latest available message and reports dropped count.
    bool pop_latest(T& out_item, std::uint64_t& dropped_count) noexcept {
        return try_read_latest(out_item, consumer_.last_consumed_sequence, &dropped_count);
    }

    /// @brief Queries a specific historical sequence if still retained within the ring buffer window.
    bool try_read_sequence(std::uint64_t target_sequence, T& out_item) const noexcept {
        if (target_sequence == 0) return false;
        const std::uint64_t head = producer_.head_sequence.load(std::memory_order_acquire);
        if (target_sequence > head) {
            return false; // Not yet published
        }
        if (head >= Capacity && target_sequence <= head - Capacity) {
            return false; // Overwritten
        }

        const std::size_t idx = (target_sequence - 1) & kIndexMask;
        const auto& slot = slots_[idx];

        const std::uint64_t s1 = slot.sequence.load(std::memory_order_acquire);
        if ((s1 & 1) != 0 || s1 != target_sequence * 2) {
            return false;
        }

        out_item = slot.data;
        std::atomic_thread_fence(std::memory_order_acquire);

        const std::uint64_t s2 = slot.sequence.load(std::memory_order_relaxed);
        return (s1 == s2);
    }

    /// @brief Returns the most recently published sequence number (0 if empty).
    [[nodiscard]] std::uint64_t latest_sequence() const noexcept {
        return producer_.head_sequence.load(std::memory_order_acquire);
    }

    /// @brief Returns the last sequence consumed by this ring instance.
    [[nodiscard]] std::uint64_t last_consumed_sequence() const noexcept {
        return consumer_.last_consumed_sequence;
    }

    /// @brief Returns the fixed ring capacity.
    [[nodiscard]] constexpr std::size_t capacity() const noexcept {
        return Capacity;
    }

    /// @brief Returns true if no messages have been published yet.
    [[nodiscard]] bool empty() const noexcept {
        return latest_sequence() == 0;
    }

    /// @brief Total count of messages dropped due to latest-wins overwrite.
    [[nodiscard]] std::uint64_t dropped_count() const noexcept {
        return consumer_.dropped_messages;
    }

    /// @brief Total count of successful reads.
    [[nodiscard]] std::uint64_t total_reads() const noexcept {
        return consumer_.total_reads;
    }

    /// @brief Total count of produced items.
    [[nodiscard]] std::uint64_t total_produced() const noexcept {
        return producer_.total_produced;
    }

    /// @brief Total torn reads detected and rejected during acquisition.
    [[nodiscard]] std::uint64_t torn_reads_detected() const noexcept {
        return consumer_.torn_reads;
    }

    /// @brief Snapshot of bus statistics.
    [[nodiscard]] BusStats stats() const noexcept {
        return BusStats{
            .total_produced = producer_.total_produced,
            .total_consumed = consumer_.total_reads,
            .total_dropped = consumer_.dropped_messages,
            .torn_reads_detected = consumer_.torn_reads,
            .last_published_seq = producer_.head_sequence.load(std::memory_order_acquire),
            .last_consumed_seq = consumer_.last_consumed_sequence,
            .last_published_timestamp_ns = 0,
            .last_consumed_timestamp_ns = 0
        };
    }

    /// @brief Resets all producer and consumer sequences to clean initial state.
    void reset() noexcept {
        producer_.head_sequence.store(0, std::memory_order_release);
        producer_.local_sequence = 0;
        producer_.total_produced = 0;
        consumer_.last_consumed_sequence = 0;
        consumer_.total_reads = 0;
        consumer_.dropped_messages = 0;
        consumer_.torn_reads = 0;
        for (auto& slot : slots_) {
            slot.sequence.store(0, std::memory_order_release);
            slot.data = T{};
        }
    }

private:
    static constexpr std::size_t kIndexMask = Capacity - 1;
    static constexpr std::size_t kMaxReadRetries = 3;

    alignas(64) ProducerState producer_{};
    alignas(64) mutable ConsumerState consumer_{};
    alignas(64) std::array<Slot, Capacity> slots_{};
};

} // namespace aim::bus
