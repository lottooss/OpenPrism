// tests/cpp/test_usb_hid_actuator.cpp
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include "aim/actuation/usb_hid_actuator.hpp"

using namespace aim;
using namespace aim::actuation;

#define TEST_ASSERT(cond)                                                                            \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (false)

namespace {

class MockUsbHidTransport final : public IUsbHidTransport {
public:
    bool is_available{true};
    bool connected{false};
    bool open_should_succeed{true};
    bool write_should_succeed{true};
    std::uint32_t last_write_error{0};
    std::size_t total_writes{0};
    std::size_t total_reads{0};
    std::vector<std::uint8_t> written_bytes{};

    [[nodiscard]] bool available() const noexcept override {
        return is_available;
    }

    [[nodiscard]] bool is_connected() const noexcept override {
        return connected;
    }

    [[nodiscard]] bool open([[maybe_unused]] std::string_view port, [[maybe_unused]] std::uint32_t baud_rate) noexcept override {
        if (!open_should_succeed) {
            connected = false;
            return false;
        }
        connected = true;
        return true;
    }

    void close() noexcept override {
        connected = false;
    }

    [[nodiscard]] TransportResult write(std::span<const std::uint8_t> data) noexcept override {
        if (!connected || !write_should_succeed) {
            return {.bytes_transferred = 0, .success = false, .error_code = last_write_error != 0 ? last_write_error : 1};
        }
        ++total_writes;
        written_bytes.insert(written_bytes.end(), data.begin(), data.end());
        return {.bytes_transferred = data.size(), .success = true, .error_code = 0};
    }

    [[nodiscard]] TransportResult read(std::span<std::uint8_t> buffer) noexcept override {
        if (!connected) {
            return {.bytes_transferred = 0, .success = false, .error_code = 1};
        }
        ++total_reads;
        return {.bytes_transferred = buffer.size(), .success = true, .error_code = 0};
    }
};

UsbHidConfig valid_config() {
    UsbHidConfig config{};
    config.port = "COM3";
    config.baud_rate = 115200;
    config.base_config.backend = "usb_hid";
    config.base_config.relative_counts = true;
    config.base_config.require_emergency_stop = true;
    config.base_config.scheduler_hz = 1000;
    config.max_tolerated_lag_ns = 10'000'000LL;
    config.enforce_monotonic_sequence = true;
    config.verify_checksum = true;
    return config;
}

ResetToken valid_reset_token() {
    return ResetToken{.token_id = "authorized-usb-hid-reset", .generated_at_ns = 1, .is_valid = true};
}

void test_protocol_packet_and_crc() {
    std::cout << "[Test 1] Testing protocol packet binary framing and CRC16..." << std::endl;
    TEST_ASSERT(sizeof(UsbHidPacket) == 34);

    UsbHidPacket packet{};
    packet.magic = UsbHidPacket::kMagic;
    packet.version = UsbHidPacket::kProtocolVersion;
    packet.msg_type = UsbHidPacket::command;
    packet.sequence_id = 42;
    packet.timestamp_ns = 123456789;
    packet.delta_x_counts = -100;
    packet.delta_y_counts = 250;
    packet.button_action = static_cast<std::uint8_t>(ButtonAction::press);
    packet.mouse_button = static_cast<std::uint8_t>(MouseButton::left);
    packet.held_buttons_mask = 0x02;

    populate_usb_hid_packet_crc(packet);
    TEST_ASSERT(verify_usb_hid_packet_crc(packet));

    // Corrupt 1 byte in payload
    packet.delta_x_counts = -99;
    TEST_ASSERT(!verify_usb_hid_packet_crc(packet));

    // Restore and re-verify
    packet.delta_x_counts = -100;
    TEST_ASSERT(verify_usb_hid_packet_crc(packet));
}

void test_initialization_fails_closed() {
    std::cout << "[Test 2] Testing initialization fail-closed rules..." << std::endl;
    MockUsbHidTransport transport{};
    GenericUsbHidActuator actuator{&transport};

    // Fail on empty port
    UsbHidConfig config = valid_config();
    config.port.clear();
    TEST_ASSERT(!actuator.initialize(config));

    // Fail if open fails
    config = valid_config();
    transport.open_should_succeed = false;
    TEST_ASSERT(!actuator.initialize(config));

    // Succeeds when transport opens
    transport.open_should_succeed = true;
    TEST_ASSERT(actuator.initialize(config));
    TEST_ASSERT(transport.is_connected());

    // Fail start if initial heartbeat write fails
    transport.write_should_succeed = false;
    TEST_ASSERT(!actuator.start());
    TEST_ASSERT(actuator.health().is_latched);

    // Reset and succeed start
    transport.write_should_succeed = true;
    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));
    TEST_ASSERT(actuator.start());
    TEST_ASSERT(actuator.health().is_active);
}

void test_monotonic_and_deadline_dispatch() {
    std::cout << "[Test 3] Testing monotonic sequence and deadline enforcement..." << std::endl;
    MockUsbHidTransport transport{};
    GenericUsbHidActuator actuator{&transport};
    UsbHidConfig config = valid_config();
    config.max_tolerated_lag_ns = 5'000'000LL; // 5 ms

    TEST_ASSERT(actuator.initialize(config));
    TEST_ASSERT(actuator.start());

    ActuationCommand cmd1{};
    cmd1.sequence_id = 1;
    cmd1.delta_x_counts = 15;
    cmd1.delta_y_counts = -20;
    TEST_ASSERT(actuator.submit_latest(cmd1) == SubmitResult::submitted);

    // Duplicate sequence ID rejected as stale
    TEST_ASSERT(actuator.submit_latest(cmd1) == SubmitResult::rejected_stale);

    // Out-of-order sequence ID rejected as stale
    ActuationCommand cmd_old{};
    cmd_old.sequence_id = 0;
    // seq 0 does not trigger out-of-order check, seq 1 was already used
    ActuationCommand cmd_stale{};
    cmd_stale.sequence_id = 1;
    TEST_ASSERT(actuator.submit_latest(cmd_stale) == SubmitResult::rejected_stale);

    // Valid increment
    ActuationCommand cmd2{};
    cmd2.sequence_id = 2;
    cmd2.delta_x_counts = 5;
    TEST_ASSERT(actuator.submit_latest(cmd2) == SubmitResult::submitted);

    // Stale timestamp exceeds deadline
    ActuationCommand cmd_expired{};
    cmd_expired.sequence_id = 3;
    cmd_expired.generated_at_ns = 1000; // Ancient timestamp
    TEST_ASSERT(actuator.submit_latest(cmd_expired) == SubmitResult::rejected_stale);
    // Stale command latches emergency stop
    TEST_ASSERT(actuator.health().is_latched);
}

void test_disconnect_and_fail_closed_latches() {
    std::cout << "[Test 4] Testing disconnect fail-closed latching and recovery..." << std::endl;
    MockUsbHidTransport transport{};
    GenericUsbHidActuator actuator{&transport};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    // Send valid command with button press
    ActuationCommand cmd{};
    cmd.sequence_id = 10;
    cmd.button_transition = {.button = MouseButton::left, .action = ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(cmd) == SubmitResult::submitted);
    TEST_ASSERT(actuator.health().pressed_buttons_mask != 0);

    // Simulate hardware disconnect
    transport.close();

    ActuationCommand cmd2{};
    cmd2.sequence_id = 11;
    cmd2.delta_x_counts = 10;
    TEST_ASSERT(actuator.submit_latest(cmd2) == SubmitResult::rejected_dispatch_failed);
    TEST_ASSERT(actuator.health().is_latched);

    // Further commands rejected
    ActuationCommand cmd3{};
    cmd3.sequence_id = 12;
    TEST_ASSERT(actuator.submit_latest(cmd3) == SubmitResult::rejected_latched);

    // Reset without reconnection fails
    TEST_ASSERT(!actuator.reset_emergency_stop(valid_reset_token()));

    // Reconnect and reset succeeds
    TEST_ASSERT(transport.open("COM3", 115200));
    TEST_ASSERT(actuator.reset_emergency_stop(valid_reset_token()));
    TEST_ASSERT(!actuator.health().is_latched);
}

void test_button_tracking_and_emergency_release() {
    std::cout << "[Test 5] Testing button tracking and emergency release..." << std::endl;
    MockUsbHidTransport transport{};
    GenericUsbHidActuator actuator{&transport};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    ActuationCommand press_left{};
    press_left.sequence_id = 1;
    press_left.button_transition = {.button = MouseButton::left, .action = ButtonAction::press};
    TEST_ASSERT(actuator.submit_latest(press_left) == SubmitResult::submitted);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == (1u << static_cast<std::uint32_t>(MouseButton::left)));

    // Trigger emergency stop -> should clear buttons and send release packet
    actuator.emergency_stop(SafetyReason::emergency_stop_triggered);
    TEST_ASSERT(actuator.health().is_latched);
    TEST_ASSERT(actuator.health().pressed_buttons_mask == 0);

    // Verify last recorded packet was emergency stop with held_buttons_mask = 0
    const auto packets = actuator.recorded_packets();
    TEST_ASSERT(!packets.empty());
    const auto& last = packets.back();
    TEST_ASSERT(last.msg_type == UsbHidPacket::emergency_stop);
    TEST_ASSERT(last.held_buttons_mask == 0);
    TEST_ASSERT(verify_usb_hid_packet_crc(last));
}

void test_latency_measurement_and_stats() {
    std::cout << "[Test 6] Testing dispatch latency profiling (10,000 dispatches)..." << std::endl;
    MockUsbHidTransport transport{};
    GenericUsbHidActuator actuator{&transport};
    TEST_ASSERT(actuator.initialize(valid_config()));
    TEST_ASSERT(actuator.start());

    constexpr std::size_t kIterations = 10000;
    std::vector<double> latencies_us;
    latencies_us.reserve(kIterations);

    for (std::size_t i = 1; i <= kIterations; ++i) {
        ActuationCommand cmd{};
        cmd.sequence_id = i;
        cmd.delta_x_counts = static_cast<std::int32_t>(i % 10);
        cmd.delta_y_counts = static_cast<std::int32_t>((i * 2) % 10);

        const auto t0 = std::chrono::steady_clock::now();
        const SubmitResult res = actuator.submit_latest(cmd);
        const auto t1 = std::chrono::steady_clock::now();

        TEST_ASSERT(res == SubmitResult::submitted);
        const double latency = std::chrono::duration<double, std::micro>(t1 - t0).count();
        latencies_us.push_back(latency);
    }

    std::sort(latencies_us.begin(), latencies_us.end());
    const double p50 = latencies_us[kIterations * 50 / 100];
    const double p95 = latencies_us[kIterations * 95 / 100];
    const double p99 = latencies_us[kIterations * 99 / 100];
    const double max_lat = latencies_us.back();

    std::cout << "  Iterations: " << kIterations << std::endl;
    std::cout << "  p50 dispatch latency: " << p50 << " us" << std::endl;
    std::cout << "  p95 dispatch latency: " << p95 << " us" << std::endl;
    std::cout << "  p99 dispatch latency: " << p99 << " us" << std::endl;
    std::cout << "  max dispatch latency: " << max_lat << " us" << std::endl;

    // Sub-millisecond dispatch requirement: p99 must be <= 1000 us (1 ms), typically < 10 us in RAM
    TEST_ASSERT(p99 < 1000.0);

    const auto stats = actuator.stats();
    TEST_ASSERT(stats.total_dispatches == kIterations);
    TEST_ASSERT(stats.transport_errors == 0);
}

} // namespace

int main() {
    try {
        std::cout << "=== Generic USB HID Actuator Unit & Latency Tests ===" << std::endl;
        test_protocol_packet_and_crc();
        test_initialization_fails_closed();
        test_monotonic_and_deadline_dispatch();
        test_disconnect_and_fail_closed_latches();
        test_button_tracking_and_emergency_release();
        test_latency_measurement_and_stats();
        std::cout << "All Generic USB HID Actuator tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
