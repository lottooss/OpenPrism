// src/actuation/usb_hid_actuator.cpp
#include "aim/actuation/usb_hid_actuator.hpp"

#include <chrono>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace aim::actuation {
namespace {

constexpr std::uint32_t button_bit(MouseButton button) noexcept {
    const auto value = static_cast<std::uint32_t>(button);
    return button != MouseButton::none && value < 32u ? (1u << value) : 0u;
}

MonotonicNs monotonic_now_ns() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

#if defined(_WIN32)
class PlatformSerialUsbHidTransport final : public IUsbHidTransport {
public:
    PlatformSerialUsbHidTransport() noexcept = default;
    ~PlatformSerialUsbHidTransport() override {
        close();
    }

    [[nodiscard]] bool available() const noexcept override {
        return true;
    }

    [[nodiscard]] bool is_connected() const noexcept override {
        return handle_ != INVALID_HANDLE_VALUE;
    }

    [[nodiscard]] bool open(std::string_view port, std::uint32_t baud_rate) noexcept override {
        close();
        if (port.empty()) {
            return false;
        }

        // Win32 COM port naming: \\.\COMx
        std::string full_path;
        if (port.rfind(R"(\\.\)", 0) != 0) {
            full_path = R"(\\.\)" + std::string(port);
        } else {
            full_path = std::string(port);
        }

        handle_ = ::CreateFileA(
            full_path.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (handle_ == INVALID_HANDLE_VALUE) {
            return false;
        }

        DCB dcb{};
        dcb.DCBlength = sizeof(DCB);
        if (!::GetCommState(handle_, &dcb)) {
            close();
            return false;
        }

        dcb.BaudRate = baud_rate;
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fDtrControl = DTR_CONTROL_ENABLE;
        dcb.fRtsControl = RTS_CONTROL_ENABLE;

        if (!::SetCommState(handle_, &dcb)) {
            close();
            return false;
        }

        COMMTIMEOUTS timeouts{};
        timeouts.ReadIntervalTimeout = MAXDWORD;
        timeouts.ReadTotalTimeoutMultiplier = 0;
        timeouts.ReadTotalTimeoutConstant = 10; // 10 ms read timeout
        timeouts.WriteTotalTimeoutMultiplier = 0;
        timeouts.WriteTotalTimeoutConstant = 10; // 10 ms write timeout

        if (!::SetCommTimeouts(handle_, &timeouts)) {
            close();
            return false;
        }

        return true;
    }

    void close() noexcept override {
        if (handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

    [[nodiscard]] TransportResult write(std::span<const std::uint8_t> data) noexcept override {
        if (handle_ == INVALID_HANDLE_VALUE || data.empty()) {
            return {.bytes_transferred = 0, .success = false, .error_code = ERROR_INVALID_HANDLE};
        }

        DWORD written = 0;
        const BOOL status = ::WriteFile(
            handle_,
            data.data(),
            static_cast<DWORD>(data.size()),
            &written,
            nullptr);

        if (!status || written != data.size()) {
            const DWORD err = ::GetLastError();
            return {.bytes_transferred = written, .success = false, .error_code = static_cast<std::uint32_t>(err)};
        }

        return {.bytes_transferred = written, .success = true, .error_code = 0};
    }

    [[nodiscard]] TransportResult read(std::span<std::uint8_t> buffer) noexcept override {
        if (handle_ == INVALID_HANDLE_VALUE || buffer.empty()) {
            return {.bytes_transferred = 0, .success = false, .error_code = ERROR_INVALID_HANDLE};
        }

        DWORD bytes_read = 0;
        const BOOL status = ::ReadFile(
            handle_,
            buffer.data(),
            static_cast<DWORD>(buffer.size()),
            &bytes_read,
            nullptr);

        if (!status) {
            const DWORD err = ::GetLastError();
            return {.bytes_transferred = bytes_read, .success = false, .error_code = static_cast<std::uint32_t>(err)};
        }

        return {.bytes_transferred = bytes_read, .success = true, .error_code = 0};
    }

private:
    HANDLE handle_{INVALID_HANDLE_VALUE};
};
#else
class PlatformSerialUsbHidTransport final : public IUsbHidTransport {
public:
    [[nodiscard]] bool available() const noexcept override {
        return false;
    }
    [[nodiscard]] bool is_connected() const noexcept override {
        return false;
    }
    [[nodiscard]] bool open(std::string_view, std::uint32_t) noexcept override {
        return false;
    }
    void close() noexcept override {}
    [[nodiscard]] TransportResult write(std::span<const std::uint8_t>) noexcept override {
        return {.bytes_transferred = 0, .success = false, .error_code = 1};
    }
    [[nodiscard]] TransportResult read(std::span<std::uint8_t>) noexcept override {
        return {.bytes_transferred = 0, .success = false, .error_code = 1};
    }
};
#endif

IUsbHidTransport* platform_usb_transport() noexcept {
    static PlatformSerialUsbHidTransport transport{};
    return &transport;
}

} // namespace

std::uint16_t compute_usb_hid_crc16(const std::uint8_t* data, std::size_t size) noexcept {
    if (data == nullptr || size == 0) {
        return 0xFFFF;
    }
    std::uint16_t crc = 0xFFFF;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint16_t>(data[i]) << 8;
        for (int bit = 0; bit < 8; ++bit) {
            if ((crc & 0x8000) != 0) {
                crc = static_cast<std::uint16_t>((crc << 1) ^ 0x1021);
            } else {
                crc = static_cast<std::uint16_t>(crc << 1);
            }
        }
    }
    return crc;
}

bool verify_usb_hid_packet_crc(const UsbHidPacket& packet) noexcept {
    const auto* raw_bytes = reinterpret_cast<const std::uint8_t*>(&packet);
    constexpr std::size_t payload_len = sizeof(UsbHidPacket) - sizeof(std::uint16_t);
    const std::uint16_t expected = compute_usb_hid_crc16(raw_bytes, payload_len);
    return packet.crc16 == expected;
}

void populate_usb_hid_packet_crc(UsbHidPacket& packet) noexcept {
    auto* raw_bytes = reinterpret_cast<const std::uint8_t*>(&packet);
    constexpr std::size_t payload_len = sizeof(UsbHidPacket) - sizeof(std::uint16_t);
    packet.crc16 = compute_usb_hid_crc16(raw_bytes, payload_len);
}

GenericUsbHidActuator::GenericUsbHidActuator() noexcept
    : transport_(platform_usb_transport()) {}

GenericUsbHidActuator::GenericUsbHidActuator(IUsbHidTransport* transport) noexcept
    : transport_(transport) {}

bool GenericUsbHidActuator::initialize(const ActuatorConfig& config) noexcept {
    UsbHidConfig hid_cfg{};
    hid_cfg.base_config = config;
    hid_cfg.port = "COM3";
    hid_cfg.baud_rate = 115200;
    return initialize(hid_cfg);
}

bool GenericUsbHidActuator::initialize(const UsbHidConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    is_initialized_ = false;
    is_active_ = false;

    if (!release_all_held_buttons_unsafe()) {
        latch_.trigger(SafetyReason::driver_error);
        return false;
    }

    if (transport_ == nullptr || !config.base_config.relative_counts ||
        !config.base_config.require_emergency_stop || config.port.empty()) {
        return false;
    }

    config_ = config;
    last_sequence_id_ = 0;
    recorded_count_ = 0;
    total_submitted_ = 0;
    total_rejected_ = 0;
    total_cancelled_ = 0;
    last_dispatch_ns_ = 0;
    last_heartbeat_ns_ = 0;
    pressed_buttons_mask_ = 0;
    stats_ = UsbHidStats{};

    if (!transport_->is_connected()) {
        if (!transport_->open(config_.port, config_.baud_rate)) {
            latch_.trigger(SafetyReason::driver_error);
            return false;
        }
    }

    is_initialized_ = true;
    return true;
}

bool GenericUsbHidActuator::start() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_initialized_ || latch_.is_latched() || transport_ == nullptr || !transport_->is_connected()) {
        return false;
    }

    // Send initial link heartbeat
    UsbHidPacket packet{};
    packet.magic = UsbHidPacket::kMagic;
    packet.version = UsbHidPacket::kProtocolVersion;
    packet.msg_type = UsbHidPacket::heartbeat;
    packet.timestamp_ns = monotonic_now_ns();
    packet.held_buttons_mask = static_cast<std::uint8_t>(pressed_buttons_mask_ & 0xFF);
    populate_usb_hid_packet_crc(packet);

    if (!send_packet_unsafe(packet)) {
        latch_.trigger(SafetyReason::driver_error);
        return false;
    }

    is_active_ = true;
    return true;
}

bool GenericUsbHidActuator::send_packet_unsafe(UsbHidPacket& packet) noexcept {
    if (transport_ == nullptr || !transport_->is_connected()) {
        ++stats_.transport_errors;
        return false;
    }

    if (config_.verify_checksum) {
        populate_usb_hid_packet_crc(packet);
    }

    const auto* raw_bytes = reinterpret_cast<const std::uint8_t*>(&packet);
    const std::span<const std::uint8_t> buffer{raw_bytes, sizeof(UsbHidPacket)};

    const MonotonicNs t0 = monotonic_now_ns();
    const TransportResult result = transport_->write(buffer);
    const MonotonicNs t1 = monotonic_now_ns();

    if (!result.success || result.bytes_transferred != sizeof(UsbHidPacket)) {
        ++stats_.transport_errors;
        stats_.last_transport_error = result.error_code;
        return false;
    }

    const auto latency = static_cast<std::uint64_t>(t1 >= t0 ? (t1 - t0) : 0);
    update_latency_stats_unsafe(latency);

    if (recorded_count_ < kMaxRecordedPackets) {
        recorded_packets_[recorded_count_++] = packet;
    }

    return true;
}

void GenericUsbHidActuator::update_latency_stats_unsafe(std::uint64_t latency_ns) noexcept {
    stats_.last_dispatch_latency_ns = latency_ns;
    if (latency_ns < stats_.min_dispatch_latency_ns) {
        stats_.min_dispatch_latency_ns = latency_ns;
    }
    if (latency_ns > stats_.max_dispatch_latency_ns) {
        stats_.max_dispatch_latency_ns = latency_ns;
    }
    stats_.total_dispatch_latency_ns += latency_ns;
}

bool GenericUsbHidActuator::release_all_held_buttons_unsafe() noexcept {
    if (pressed_buttons_mask_ == 0) {
        return true;
    }

    UsbHidPacket packet{};
    packet.magic = UsbHidPacket::kMagic;
    packet.version = UsbHidPacket::kProtocolVersion;
    packet.msg_type = UsbHidPacket::emergency_stop;
    packet.timestamp_ns = monotonic_now_ns();
    packet.held_buttons_mask = 0;
    populate_usb_hid_packet_crc(packet);

    const bool ok = send_packet_unsafe(packet);
    if (!ok) {
        ++stats_.disconnect_failures;
        return false;
    }

    pressed_buttons_mask_ = 0;
    ++stats_.button_releases;
    return true;
}

SubmitResult GenericUsbHidActuator::submit_latest(const ActuationCommand& command) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_initialized_ || !is_active_) {
        ++total_rejected_;
        return SubmitResult::rejected_uninitialized;
    }
    if (latch_.is_latched()) {
        ++total_rejected_;
        return SubmitResult::rejected_latched;
    }

    if (config_.enforce_monotonic_sequence && command.sequence_id > 0 &&
        command.sequence_id <= last_sequence_id_) {
        ++total_rejected_;
        ++stats_.out_of_order_dropped;
        return SubmitResult::rejected_stale;
    }

    const MonotonicNs now_ns = monotonic_now_ns();
    const MonotonicNs target_time =
        command.desired_apply_time_ns > 0 ? command.desired_apply_time_ns : command.generated_at_ns;

    if (target_time > 0 && now_ns > target_time + config_.max_tolerated_lag_ns) {
        if (command.sequence_id > 0) {
            last_sequence_id_ = command.sequence_id;
        }
        ++total_rejected_;
        ++stats_.stale_dropped;
        (void)release_all_held_buttons_unsafe();
        latch_.trigger(SafetyReason::stale_data);
        return SubmitResult::rejected_stale;
    }

    if (transport_ == nullptr || !transport_->is_connected()) {
        if (command.sequence_id > 0) {
            last_sequence_id_ = command.sequence_id;
        }
        ++total_rejected_;
        ++stats_.disconnect_failures;
        (void)release_all_held_buttons_unsafe();
        latch_.trigger(SafetyReason::driver_error);
        return SubmitResult::rejected_dispatch_failed;
    }

    // Update button tracking state
    const auto& transition = command.button_transition;
    const std::uint32_t bit = button_bit(transition.button);
    if (transition.action == ButtonAction::press) {
        pressed_buttons_mask_ |= bit;
        ++stats_.button_presses;
    } else if (transition.action == ButtonAction::release) {
        pressed_buttons_mask_ &= ~bit;
        ++stats_.button_releases;
    } else if (transition.action == ButtonAction::click) {
        ++stats_.button_presses;
        ++stats_.button_releases;
    }

    UsbHidPacket packet{};
    packet.magic = UsbHidPacket::kMagic;
    packet.version = UsbHidPacket::kProtocolVersion;
    packet.msg_type = UsbHidPacket::command;
    packet.sequence_id = command.sequence_id;
    packet.timestamp_ns = command.generated_at_ns > 0 ? command.generated_at_ns : now_ns;
    packet.delta_x_counts = command.delta_x_counts;
    packet.delta_y_counts = command.delta_y_counts;
    packet.button_action = static_cast<std::uint8_t>(transition.action);
    packet.mouse_button = static_cast<std::uint8_t>(transition.button);
    packet.held_buttons_mask = static_cast<std::uint8_t>(pressed_buttons_mask_ & 0xFF);
    packet.flags = 0;

    if (!send_packet_unsafe(packet)) {
        if (command.sequence_id > 0) {
            last_sequence_id_ = command.sequence_id;
        }
        ++total_rejected_;
        (void)release_all_held_buttons_unsafe();
        latch_.trigger(SafetyReason::driver_error);
        return SubmitResult::rejected_dispatch_failed;
    }

    if (command.sequence_id > 0) {
        last_sequence_id_ = command.sequence_id;
    }
    last_dispatch_ns_ = now_ns;
    ++total_submitted_;
    ++stats_.total_dispatches;
    stats_.cumulative_counts_x += command.delta_x_counts;
    stats_.cumulative_counts_y += command.delta_y_counts;

    return SubmitResult::submitted;
}

bool GenericUsbHidActuator::send_heartbeat() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_initialized_ || latch_.is_latched() || transport_ == nullptr || !transport_->is_connected()) {
        return false;
    }

    const MonotonicNs now_ns = monotonic_now_ns();
    UsbHidPacket packet{};
    packet.magic = UsbHidPacket::kMagic;
    packet.version = UsbHidPacket::kProtocolVersion;
    packet.msg_type = UsbHidPacket::heartbeat;
    packet.sequence_id = 0;
    packet.timestamp_ns = now_ns;
    packet.held_buttons_mask = static_cast<std::uint8_t>(pressed_buttons_mask_ & 0xFF);

    if (!send_packet_unsafe(packet)) {
        (void)release_all_held_buttons_unsafe();
        latch_.trigger(SafetyReason::driver_error);
        return false;
    }

    last_heartbeat_ns_ = now_ns;
    ++stats_.total_heartbeats;
    return true;
}

void GenericUsbHidActuator::cancel_pending() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_cancelled_;
}

void GenericUsbHidActuator::emergency_stop(SafetyReason reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    latch_.trigger(reason);
    is_active_ = false;

    if (transport_ != nullptr && transport_->is_connected()) {
        UsbHidPacket packet{};
        packet.magic = UsbHidPacket::kMagic;
        packet.version = UsbHidPacket::kProtocolVersion;
        packet.msg_type = UsbHidPacket::emergency_stop;
        packet.timestamp_ns = monotonic_now_ns();
        packet.held_buttons_mask = 0;
        (void)send_packet_unsafe(packet);
        ++stats_.emergency_stops_sent;
    }
    pressed_buttons_mask_ = 0;
}

bool GenericUsbHidActuator::reset_emergency_stop(const ResetToken& token) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!token.is_valid || token.token_id.empty()) {
        return false;
    }
    if (transport_ == nullptr || !transport_->is_connected()) {
        return false;
    }

    if (!latch_.try_reset(token)) {
        return false;
    }
    pressed_buttons_mask_ = 0;
    is_active_ = true;
    return true;
}

void GenericUsbHidActuator::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (transport_ != nullptr && transport_->is_connected()) {
        (void)release_all_held_buttons_unsafe();
        transport_->close();
    }
    is_active_ = false;
    is_initialized_ = false;
}

ActuatorHealth GenericUsbHidActuator::health() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return ActuatorHealth{
        .is_active = is_active_,
        .is_latched = latch_.is_latched(),
        .total_commands_submitted = total_submitted_,
        .total_commands_rejected = total_rejected_,
        .total_commands_cancelled = total_cancelled_,
        .last_dispatch_ns = last_dispatch_ns_,
        .pressed_buttons_mask = pressed_buttons_mask_,
    };
}

UsbHidStats GenericUsbHidActuator::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::vector<UsbHidPacket> GenericUsbHidActuator::recorded_packets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<UsbHidPacket>(
        recorded_packets_.begin(),
        recorded_packets_.begin() + static_cast<std::ptrdiff_t>(recorded_count_));
}

void GenericUsbHidActuator::clear_recorded_packets() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    recorded_count_ = 0;
}

} // namespace aim::actuation
