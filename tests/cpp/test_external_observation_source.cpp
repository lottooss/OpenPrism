// tests/cpp/test_external_observation_source.cpp
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "aim/bus/external_observation_source.hpp"
#include "aim/bus/shared_memory_ring.hpp"

using namespace aim;
using namespace aim::bus;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

void test_nominal_shared_memory_ingestion() {
    std::cout << "[Test 1] Nominal external observation publishing and consumption..." << std::endl;

    const std::string shm_name = "aim_test_nominal_shm";

    // 1. Create Producer Ring
    auto producer = SharedMemoryRing<TargetObservationBatch, 16>::create(shm_name, 101);
    TEST_ASSERT(producer.is_valid());

    // 2. Initialize Consumer Source
    ExternalObservationSourceConfig config{};
    config.shm_name = shm_name;
    config.expected_pipeline_run_id = 101;
    config.max_heartbeat_age_ns = 50'000'000LL; // 50 ms

    ExternalObservationSource source{config};
    TEST_ASSERT(source.start());
    TEST_ASSERT(source.is_running());
    TEST_ASSERT(source.last_status() == ObservationSourceStatus::ok);

    MonotonicNs sim_time_ns = 1'000'000'000LL;

    // 3. Publish 5 batches from producer
    for (std::uint64_t i = 1; i <= 5; ++i) {
        sim_time_ns += 5'000'000LL; // 5 ms intervals
        producer.update_heartbeat(sim_time_ns);

        TargetObservationBatch batch{};
        batch.header.sequence_id = i;
        batch.header.source_timestamp_ns = sim_time_ns;
        batch.header.pipeline_run_id = 101;
        batch.captured_at_ns = sim_time_ns;

        TargetObservation obs{};
        obs.source_id = 1000 + i;
        obs.center_px = PixelPoint{960.0f + static_cast<float>(i), 540.0f};
        obs.center_norm = NormalizedPoint{0.0f, 0.0f};
        obs.effective_radius_px = 15.0f;
        obs.confidence = 0.98f;
        obs.covariance_px2 = Covariance2D{1.0f, 0.0f, 1.0f};
        batch.add_target(obs);

        TEST_ASSERT(producer.write_latest(batch, sim_time_ns) == BusWriteResult::ok);

        TargetObservationBatch out_batch{};
        TEST_ASSERT(source.try_read_latest_at(out_batch, sim_time_ns));
        TEST_ASSERT(out_batch.header.sequence_id == i);
        TEST_ASSERT(out_batch.target_count == 1);
        TEST_ASSERT(out_batch.items()[0].source_id == 1000 + i);
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::ok);
    }

    TEST_ASSERT(source.telemetry().total_batches_read == 5);
    TEST_ASSERT(source.telemetry().total_malformed_rejected == 0);
    TEST_ASSERT(source.telemetry().total_heartbeat_timeouts == 0);

    source.stop();
    TEST_ASSERT(!source.is_running());
    std::cout << "  -> Nominal publishing and consumption passed." << std::endl;
}

void test_producer_heartbeat_timeout_and_death_detection() {
    std::cout << "[Test 2] Producer heartbeat timeout and crash detection..." << std::endl;

    const std::string shm_name = "aim_test_heartbeat_shm";
    auto producer = SharedMemoryRing<TargetObservationBatch, 16>::create(shm_name, 202);
    TEST_ASSERT(producer.is_valid());

    ExternalObservationSourceConfig config{};
    config.shm_name = shm_name;
    config.expected_pipeline_run_id = 202;
    config.max_heartbeat_age_ns = 20'000'000LL; // 20 ms timeout

    ExternalObservationSource source{config};
    TEST_ASSERT(source.start());

    MonotonicNs sim_time_ns = 2'000'000'000LL;
    producer.update_heartbeat(sim_time_ns);

    TargetObservationBatch batch{};
    batch.header.sequence_id = 1;
    batch.captured_at_ns = sim_time_ns;
    TargetObservation obs{};
    obs.center_px = PixelPoint{500.0f, 500.0f};
    obs.effective_radius_px = 12.0f;
    obs.confidence = 0.90f;
    batch.add_target(obs);
    producer.write_latest(batch, sim_time_ns);

    // Read within heartbeat window (10 ms later < 20 ms timeout)
    TargetObservationBatch out_batch{};
    TEST_ASSERT(source.try_read_latest_at(out_batch, sim_time_ns + 10'000'000LL));
    TEST_ASSERT(source.last_status() == ObservationSourceStatus::ok);

    // Advance time past timeout without producer pulsing heartbeat (35 ms later > 20 ms timeout)
    TargetObservationBatch dead_batch{};
    TEST_ASSERT(!source.try_read_latest_at(dead_batch, sim_time_ns + 35'000'000LL));
    TEST_ASSERT(source.last_status() == ObservationSourceStatus::producer_dead);
    TEST_ASSERT(source.telemetry().total_heartbeat_timeouts == 1);

    // Producer recovers and pulses heartbeat
    producer.update_heartbeat(sim_time_ns + 40'000'000LL);
    batch.header.sequence_id = 2;
    producer.write_latest(batch, sim_time_ns + 40'000'000LL);

    TEST_ASSERT(source.try_read_latest_at(out_batch, sim_time_ns + 40'000'000LL));
    TEST_ASSERT(source.last_status() == ObservationSourceStatus::ok);

    source.stop();
    std::cout << "  -> Producer heartbeat timeout and crash detection passed." << std::endl;
}

void test_schema_negotiation_and_mismatch_rejection() {
    std::cout << "[Test 3] Schema negotiation and version mismatch rejection..." << std::endl;

    const std::string shm_name = "aim_test_schema_shm";
    auto producer = SharedMemoryRing<TargetObservationBatch, 16>::create(shm_name, 303);
    TEST_ASSERT(producer.is_valid());

    // Case A: Wrong pipeline run ID
    {
        ExternalObservationSourceConfig config{};
        config.shm_name = shm_name;
        config.expected_pipeline_run_id = 999; // Mismatched ID!

        ExternalObservationSource source{config};
        TEST_ASSERT(!source.start());
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::schema_mismatch);
    }

    // Case B: Non-existent segment
    {
        ExternalObservationSourceConfig config{};
        config.shm_name = "aim_non_existent_shm_segment_12345";

        ExternalObservationSource source{config};
        TEST_ASSERT(!source.start());
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::uninitialized);
    }

    std::cout << "  -> Schema negotiation and version mismatch rejection passed." << std::endl;
}

void test_malformed_payload_fail_closed_rejection() {
    std::cout << "[Test 4] Malformed payload fail-closed rejection..." << std::endl;

    const std::string shm_name = "aim_test_malformed_shm";
    auto producer = SharedMemoryRing<TargetObservationBatch, 16>::create(shm_name, 404);
    TEST_ASSERT(producer.is_valid());

    ExternalObservationSourceConfig config{};
    config.shm_name = shm_name;
    config.expected_pipeline_run_id = 404;

    ExternalObservationSource source{config};
    TEST_ASSERT(source.start());

    MonotonicNs sim_time_ns = 3'000'000'000LL;
    producer.update_heartbeat(sim_time_ns);

    // Case A: NaN Coordinates
    {
        TargetObservationBatch batch{};
        batch.header.sequence_id = 1;
        batch.captured_at_ns = sim_time_ns;
        TargetObservation obs{};
        obs.center_px = PixelPoint{std::numeric_limits<float>::quiet_NaN(), 540.0f};
        obs.effective_radius_px = 15.0f;
        obs.confidence = 0.95f;
        batch.add_target(obs);
        producer.write_latest(batch, sim_time_ns);

        TargetObservationBatch out{};
        TEST_ASSERT(!source.try_read_latest_at(out, sim_time_ns));
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::malformed_payload);
        TEST_ASSERT(source.telemetry().total_malformed_rejected == 1);
    }

    // Case B: Out-of-bounds Confidence (> 1.0)
    {
        TargetObservationBatch batch{};
        batch.header.sequence_id = 2;
        batch.captured_at_ns = sim_time_ns;
        TargetObservation obs{};
        obs.center_px = PixelPoint{960.0f, 540.0f};
        obs.effective_radius_px = 15.0f;
        obs.confidence = 1.50f; // Invalid > 1.0!
        batch.add_target(obs);
        producer.write_latest(batch, sim_time_ns);

        TargetObservationBatch out{};
        TEST_ASSERT(!source.try_read_latest_at(out, sim_time_ns));
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::malformed_payload);
        TEST_ASSERT(source.telemetry().total_malformed_rejected == 2);
    }

    // Case C: Negative Effective Radius
    {
        TargetObservationBatch batch{};
        batch.header.sequence_id = 3;
        batch.captured_at_ns = sim_time_ns;
        TargetObservation obs{};
        obs.center_px = PixelPoint{960.0f, 540.0f};
        obs.effective_radius_px = -5.0f; // Negative radius!
        obs.confidence = 0.95f;
        batch.add_target(obs);
        producer.write_latest(batch, sim_time_ns);

        TargetObservationBatch out{};
        TEST_ASSERT(!source.try_read_latest_at(out, sim_time_ns));
        TEST_ASSERT(source.last_status() == ObservationSourceStatus::malformed_payload);
        TEST_ASSERT(source.telemetry().total_malformed_rejected == 3);
    }

    source.stop();
    std::cout << "  -> Malformed payload fail-closed rejection passed." << std::endl;
}

void test_buffer_overrun_and_drop_accounting() {
    std::cout << "[Test 5] Buffer overrun and drop accounting..." << std::endl;

    const std::string shm_name = "aim_test_overrun_shm";
    auto producer = SharedMemoryRing<TargetObservationBatch, 16>::create(shm_name, 505);
    TEST_ASSERT(producer.is_valid());

    ExternalObservationSourceConfig config{};
    config.shm_name = shm_name;
    config.expected_pipeline_run_id = 505;

    ExternalObservationSource source{config};
    TEST_ASSERT(source.start());

    MonotonicNs sim_time_ns = 4'000'000'000LL;

    // Produce 30 batches without consumer reading
    for (std::uint64_t i = 1; i <= 30; ++i) {
        sim_time_ns += 1'000'000LL;
        producer.update_heartbeat(sim_time_ns);

        TargetObservationBatch batch{};
        batch.header.sequence_id = i;
        batch.captured_at_ns = sim_time_ns;
        TargetObservation obs{};
        obs.center_px = PixelPoint{100.0f, 100.0f};
        obs.effective_radius_px = 10.0f;
        obs.confidence = 0.90f;
        batch.add_target(obs);
        producer.write_latest(batch, sim_time_ns);
    }

    // Now consumer reads newest
    TargetObservationBatch out{};
    TEST_ASSERT(source.try_read_latest_at(out, sim_time_ns));
    TEST_ASSERT(out.header.sequence_id == 30);
    // Overrun should account for dropped items (30 produced - 1 read = 29 dropped)
    TEST_ASSERT(source.telemetry().total_dropped_overrun == 29);
    TEST_ASSERT(source.telemetry().total_batches_read == 1);

    source.stop();
    std::cout << "  -> Buffer overrun and drop accounting passed." << std::endl;
}

int verify_external_shm(const std::string& shm_name, std::uint64_t expected_seq) {
    ExternalObservationSourceConfig config{};
    config.shm_name = shm_name;
    config.require_heartbeat = true;
    config.max_heartbeat_age_ns = 5'000'000'000LL; // 5s tolerance for IPC tests
    config.validate_schema = true;

    ExternalObservationSource source{config};
    if (!source.start()) {
        std::cerr << "Failed to attach to shared memory segment: " << shm_name << std::endl;
        return 1;
    }

    TargetObservationBatch batch{};
    if (!source.try_read_latest(batch)) {
        std::cerr << "Failed to read latest batch from " << shm_name
                  << " (status: " << to_string(source.last_status()) << ")" << std::endl;
        return 1;
    }

    if (batch.header.sequence_id != expected_seq) {
        std::cerr << "Sequence mismatch: expected " << expected_seq
                  << ", got " << batch.header.sequence_id << std::endl;
        return 1;
    }

    if (batch.target_count == 0) {
        std::cerr << "Target count is zero!" << std::endl;
        return 1;
    }

    std::cout << "Verified IPC segment " << shm_name << " at sequence " << batch.header.sequence_id
              << " with " << batch.target_count << " targets." << std::endl;
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc >= 4 && std::string(argv[1]) == "--verify-shm") {
        const std::string name = argv[2];
        const std::uint64_t seq = std::stoull(argv[3]);
        return verify_external_shm(name, seq);
    }

    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M8-01 External Observation Source Tests      " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_nominal_shared_memory_ingestion();
    test_producer_heartbeat_timeout_and_death_detection();
    test_schema_negotiation_and_mismatch_rejection();
    test_malformed_payload_fail_closed_rejection();
    test_buffer_overrun_and_drop_accounting();

    std::cout << "================================================================" << std::endl;
    std::cout << " All Milestone M8-01 External Observation Source Tests Passed!   " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
