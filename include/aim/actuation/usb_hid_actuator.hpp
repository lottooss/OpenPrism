// include/aim/actuation/usb_hid_actuator.hpp
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "aim/core/actuator.hpp"
#include "aim/core/safety.hpp"
#include "aim/core/time.hpp"
#include "aim/core/types.hpp"

namespace aim::actuation {

#pragma pack(push, 1)
/// @brief Versioned, checksummed binary wire protocol packet for generic USB HID microcontroller backends.
/// Total packet size: 34 bytes.
struct UsbHidPacket {
    static constexpr std::uint16_t kMagic = 0x5548; // 'U', 'H'
    static constexpr std::uint8_t kProtocolVersion = 1;

    enum MsgType : std::uint8_t {
        command = 0x01,
        heartbeat = 0x02,
        emergency_stop = 0x03,
        ack = 0x04
    };

    std::uint16_t magic{kMagic};
    std::uint8_t version{kProtocolVersion};
    std::uint8_t msg_type{MsgType::command};
    std::uint64_t sequence_id{0};
    std::int64_t timestamp_ns{0};
    std::int32_t delta_x_counts{0};
    std::int32_t delta_y_counts{0};
    std::uint8_t button_action{0};
    std::uint8_t mouse_button{0};
    std::uint8_t held_buttons_mask{0};
    std::uint8_t flags{0};
    std::uint16_t crc16{0};
};
#pragma pack(pop)

static_assert(sizeof(UsbHidPacket) == 34, "UsbHidPacket must be exactly 34 bytes");

/// @brief Compute CCITT-16 CRC (poly 0x1021, init 0xFFFF) across byte buffer
[[nodiscard]] std::uint16_t compute_usb_hid_crc16(const std::uint8_t* data, std::size_t size) noexcept;

/// @brief Verify checksum of a received packet
[[nodiscard]] bool verify_usb_hid_packet_crc(const UsbHidPacket& packet) noexcept;

/// @brief Populate the checksum field of a packet
void populate_usb_hid_packet_crc(UsbHidPacket& packet) noexcept;

struct UsbHidConfig {
    ActuatorConfig base_config{};
    std::string port{"COM3"};
    std::uint32_t baud_rate{115200};
    MonotonicNs max_tolerated_lag_ns{10'000'000LL}; // 10 ms hard stale cutoff
    MonotonicNs heartbeat_timeout_ns{50'000'000LL}; // 50 ms heartbeat loss timeout
    bool enforce_monotonic_sequence{true};
    bool verify_checksum{true};
};

struct UsbHidStats {
    std::uint64_t total_dispatches{0};
    std::uint64_t total_heartbeats{0};
    std::uint64_t stale_dropped{0};
    std::uint64_t out_of_order_dropped{0};
    std::uint64_t transport_errors{0};
    std::uint64_t crc_errors{0};
    std::uint64_t disconnect_failures{0};
    std::uint64_t emergency_stops_sent{0};
    std::uint64_t button_presses{0};
    std::uint64_t button_releases{0};
    std::int64_t cumulative_counts_x{0};
    std::int64_t cumulative_counts_y{0};
    // Microsecond latency tracking
    std::uint64_t last_dispatch_latency_ns{0};
    std::uint64_t min_dispatch_latency_ns{UINT64_MAX};
    std::uint64_t max_dispatch_latency_ns{0};
    std::uint64_t total_dispatch_latency_ns{0};
    std::uint32_t last_transport_error{0};

    [[nodiscard]] double avg_dispatch_latency_us() const noexcept {
        return total_dispatches > 0
                   ? (static_cast<double>(total_dispatch_latency_ns) / static_cast<double>(total_dispatches)) / 1000.0
                   : 0.0;
    }
};

struct TransportResult {
    std::size_t bytes_transferred{0};
    bool success{false};
    std::uint32_t error_code{0};
};

/// @brief Abstract physical/virtual communication transport for USB HID microcontroller.
/// Non-owning transport pointer may be injected for testing or hardware abstraction.
class IUsbHidTransport {
public:
    virtual ~IUsbHidTransport() = default;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    [[nodiscard]] virtual bool is_connected() const noexcept = 0;
    [[nodiscard]] virtual bool open(std::string_view port, std::uint32_t baud_rate) noexcept = 0;
    virtual void close() noexcept = 0;
    [[nodiscard]] virtual TransportResult write(std::span<const std::uint8_t> data) noexcept = 0;
    [[nodiscard]] virtual TransportResult read(std::span<std::uint8_t> buffer) noexcept = 0;
};

/// @brief Production Generic USB HID Actuator (Milestone M9-03).
/// Implements IActuator for authorized lab and robotics hardware experiments.
/// Guarantees fail-closed safety, CRC-16 checksummed binary framing,
/// sequence validation, latency profiling, and paired button release.
class GenericUsbHidActuator final : public IActuator {
public:
    static constexpr std::size_t kMaxRecordedPackets = 1024;

    GenericUsbHidActuator() noexcept;
    explicit GenericUsbHidActuator(IUsbHidTransport* transport) noexcept;
    ~GenericUsbHidActuator() noexcept override {
        shutdown();
    }

    bool initialize(const ActuatorConfig& config) noexcept override;
    bool initialize(const UsbHidConfig& config) noexcept;
    bool start() noexcept override;
    SubmitResult submit_latest(const ActuationCommand& command) noexcept override;
    void cancel_pending() noexcept override;
    void emergency_stop(SafetyReason reason = SafetyReason::emergency_stop_triggered) noexcept override;
    bool reset_emergency_stop(const ResetToken& token) noexcept override;
    void shutdown() noexcept override;
    [[nodiscard]] ActuatorHealth health() const noexcept override;
    [[nodiscard]] UsbHidStats stats() const noexcept;

    /// @brief Send heartbeat packet to maintain link health and measure transport latency
    bool send_heartbeat() noexcept;

    /// @brief Get history of transmitted packets (for deterministic test validation)
    [[nodiscard]] std::vector<UsbHidPacket> recorded_packets() const;
    void clear_recorded_packets() noexcept;

private:
    [[nodiscard]] bool send_packet_unsafe(UsbHidPacket& packet) noexcept;
    [[nodiscard]] bool release_all_held_buttons_unsafe() noexcept;
    void update_latency_stats_unsafe(std::uint64_t latency_ns) noexcept;

    mutable std::mutex mutex_;
    UsbHidConfig config_{};
    EmergencyStopLatch latch_{};
    bool is_initialized_{false};
    bool is_active_{false};
    IUsbHidTransport* transport_{nullptr};

    SequenceId last_sequence_id_{0};
    MonotonicNs last_heartbeat_ns_{0};
    std::array<UsbHidPacket, kMaxRecordedPackets> recorded_packets_{};
    std::size_t recorded_count_{0};

    std::uint64_t total_submitted_{0};
    std::uint64_t total_rejected_{0};
    std::uint64_t total_cancelled_{0};
    MonotonicNs last_dispatch_ns_{0};
    std::uint32_t pressed_buttons_mask_{0};
    UsbHidStats stats_{};
};

} // namespace aim::actuation
