// tests/cpp/test_scenario_adapter.cpp
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include "aim/scenario/scenario_adapter.hpp"
#include "aim/scenario/scenario_profile_loader.hpp"

using namespace aim;
using namespace aim::scenario;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

namespace {

/// @brief Build an observation batch whose target_value is a recognisable sentinel,
/// so a test can prove whether enrich() did or did not touch it.
bus::TargetObservationBatch make_batch(std::uint32_t count, float sentinel) noexcept {
    bus::TargetObservationBatch batch{};
    for (std::uint32_t i = 0; i < count; ++i) {
        bus::TargetObservation obs{};
        obs.center_px = {100.0f + static_cast<float>(i), 200.0f};
        obs.effective_radius_px = 12.0f;
        obs.confidence = 0.9f;
        obs.target_value = sentinel;
        (void)batch.add_target(obs);
    }
    return batch;
}

ObservableScenarioState aimlabs_state(float elapsed_seconds) noexcept {
    ObservableScenarioState state{};
    state.observed_at_ns = 1'000'000'000LL;
    state.active_process_name = "Aimlab_tb.exe";
    state.active_window_title = "Aimlabs";
    state.foreground_confirmed = true;
    state.elapsed_session_seconds = elapsed_seconds;
    state.visible_target_count = 3;
    return state;
}

// -----------------------------------------------------------------------------

void test_unloaded_adapter_fails_closed() {
    std::cout << "[Test 1] Unloaded adapter advertises nothing and fails closed..." << std::endl;

    ScenarioAdapter adapter{};
    TEST_ASSERT(!adapter.has_profile());
    TEST_ASSERT(adapter.capabilities() == ScenarioCapabilities::none());

    // Every capability-gated accessor returns its unavailable default.
    TEST_ASSERT(!adapter.calibration_seed().valid);
    TEST_ASSERT(!adapter.crosshair_context().valid);
    const ForegroundIdentity identity = adapter.foreground_identity();
    TEST_ASSERT(identity.require_match);
    TEST_ASSERT(identity.process_name.empty());
    TEST_ASSERT(identity.window_title_substring.empty());

    // An unconfirmable identity can never authorize actuation, even when the host
    // reports that our own window is focused.
    TEST_ASSERT(!adapter.foreground_authorized(aimlabs_state(1.0f)));
    ObservableScenarioState unfocused = aimlabs_state(1.0f);
    unfocused.foreground_confirmed = false;
    TEST_ASSERT(!adapter.foreground_authorized(unfocused));

    // Completion is never observable without the capability.
    TEST_ASSERT(!adapter.scenario_complete(aimlabs_state(9'999.0f)));

    // enrich() must not touch observations it was never configured to weight.
    bus::TargetObservationBatch batch = make_batch(4, 0.25f);
    adapter.enrich(batch, aimlabs_state(1.0f));
    for (std::uint32_t i = 0; i < batch.target_count; ++i) {
        TEST_ASSERT(batch.targets[i].target_value == 0.25f);
    }
    TEST_ASSERT(adapter.stamped_targets() == 0);
    TEST_ASSERT(adapter.enriched_batches() == 0);

    std::cout << "  -> Unloaded adapter is fully fail-closed." << std::endl;
}

void test_generic_profile_is_domain_neutral() {
    std::cout << "[Test 2] Generic profile is domain-neutral and replaceable..." << std::endl;

    ScenarioAdapter adapter{};
    TEST_ASSERT(adapter.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);
    TEST_ASSERT(adapter.has_profile());
    TEST_ASSERT(adapter.profile().profile_id == "generic_targets");

    // Only a generic target value and an explicitly unconstrained identity.
    const ScenarioCapabilities caps = adapter.capabilities();
    TEST_ASSERT(caps.provides_target_value);
    TEST_ASSERT(caps.provides_foreground_identity);
    TEST_ASSERT(!caps.provides_crosshair_context);
    TEST_ASSERT(!caps.provides_calibration_seed);
    TEST_ASSERT(!caps.reports_scenario_completion);

    // Un-advertised capabilities still resolve to their safe defaults.
    TEST_ASSERT(!adapter.calibration_seed().valid);
    TEST_ASSERT(!adapter.crosshair_context().valid);
    TEST_ASSERT(!adapter.scenario_complete(aimlabs_state(9'999.0f)));

    // The unconstrained identity authorizes any *focused* window, which is what makes
    // this profile usable for offline replay and portability runs.
    TEST_ASSERT(!adapter.foreground_identity().require_match);
    TEST_ASSERT(adapter.foreground_authorized(aimlabs_state(1.0f)));
    ObservableScenarioState other_app = aimlabs_state(1.0f);
    other_app.active_process_name = "notepad.exe";
    other_app.active_window_title = "Untitled - Notepad";
    TEST_ASSERT(adapter.foreground_authorized(other_app));

    // Being unconstrained relaxes WHICH application may be targeted. It must never
    // relax whether our window is actually focused: the host's own focus result is
    // checked before, and independently of, require_match.
    ObservableScenarioState unfocused = aimlabs_state(1.0f);
    unfocused.foreground_confirmed = false;
    TEST_ASSERT(!adapter.foreground_authorized(unfocused));
    ObservableScenarioState unfocused_elsewhere = other_app;
    unfocused_elsewhere.foreground_confirmed = false;
    TEST_ASSERT(!adapter.foreground_authorized(unfocused_elsewhere));

    // The generic value is applied uniformly.
    bus::TargetObservationBatch batch = make_batch(5, -1.0f);
    adapter.enrich(batch, aimlabs_state(1.0f));
    for (std::uint32_t i = 0; i < batch.target_count; ++i) {
        TEST_ASSERT(batch.targets[i].target_value == 1.0f);
    }
    TEST_ASSERT(adapter.stamped_targets() == 5);

    std::cout << "  -> Generic profile carries no domain semantics." << std::endl;
}

void test_enrich_touches_only_target_value() {
    std::cout << "[Test 3] enrich() never mutates geometry, confidence, or timing..." << std::endl;

    ScenarioAdapter adapter{make_aimlabs_profile()};
    bus::TargetObservationBatch batch = make_batch(3, 0.0f);

    // Snapshot every field enrich() must leave alone.
    const bus::TargetObservationBatch before = batch;
    adapter.enrich(batch, aimlabs_state(1.0f));

    TEST_ASSERT(batch.target_count == before.target_count);
    for (std::uint32_t i = 0; i < batch.target_count; ++i) {
        TEST_ASSERT(batch.targets[i].center_px.x == before.targets[i].center_px.x);
        TEST_ASSERT(batch.targets[i].center_px.y == before.targets[i].center_px.y);
        TEST_ASSERT(batch.targets[i].effective_radius_px == before.targets[i].effective_radius_px);
        TEST_ASSERT(batch.targets[i].confidence == before.targets[i].confidence);
        TEST_ASSERT(batch.targets[i].captured_at_ns == before.targets[i].captured_at_ns);
        // Only the generic value changed.
        TEST_ASSERT(batch.targets[i].target_value == 1.0f);
    }

    // An empty batch is handled without stamping anything.
    bus::TargetObservationBatch empty{};
    adapter.enrich(empty, aimlabs_state(1.0f));
    TEST_ASSERT(empty.target_count == 0);
    TEST_ASSERT(adapter.stamped_targets() == 3);

    std::cout << "  -> enrich() stamps target_value and nothing else." << std::endl;
}

void test_foreground_authorization_is_injection_free() {
    std::cout << "[Test 4] Foreground authorization fails closed on every mismatch..." << std::endl;

    ScenarioAdapter adapter{make_aimlabs_profile()};
    TEST_ASSERT(adapter.capabilities().provides_foreground_identity);

    // Exact identity, host-confirmed focus.
    TEST_ASSERT(adapter.foreground_authorized(aimlabs_state(1.0f)));

    // Case-insensitive on both process name and title substring.
    ObservableScenarioState mixed_case = aimlabs_state(1.0f);
    mixed_case.active_process_name = "AIMLAB_TB.EXE";
    mixed_case.active_window_title = "aimlabs - gridshot";
    TEST_ASSERT(adapter.foreground_authorized(mixed_case));

    // Host says our window is not focused -> refuse regardless of the strings.
    ObservableScenarioState unfocused = aimlabs_state(1.0f);
    unfocused.foreground_confirmed = false;
    TEST_ASSERT(!adapter.foreground_authorized(unfocused));

    // Wrong process -> refuse.
    ObservableScenarioState wrong_process = aimlabs_state(1.0f);
    wrong_process.active_process_name = "notepad.exe";
    TEST_ASSERT(!adapter.foreground_authorized(wrong_process));

    // Right process, wrong title -> refuse (every non-empty pattern must match).
    ObservableScenarioState wrong_title = aimlabs_state(1.0f);
    wrong_title.active_window_title = "Untitled - Notepad";
    TEST_ASSERT(!adapter.foreground_authorized(wrong_title));

    // Empty observations -> refuse.
    ObservableScenarioState empty_strings = aimlabs_state(1.0f);
    empty_strings.active_process_name = "";
    empty_strings.active_window_title = "";
    TEST_ASSERT(!adapter.foreground_authorized(empty_strings));

    // A near-miss process name must not match: equality, not substring.
    ObservableScenarioState near_miss = aimlabs_state(1.0f);
    near_miss.active_process_name = "Aimlab_tb.exe.bak";
    TEST_ASSERT(!adapter.foreground_authorized(near_miss));

    // The underlying ASCII helpers behave as documented.
    TEST_ASSERT(ascii_iequals("AbC", "aBc"));
    TEST_ASSERT(!ascii_iequals("abc", "abcd"));
    TEST_ASSERT(ascii_iequals("", ""));
    TEST_ASSERT(ascii_icontains("Aimlabs - Gridshot", "gridshot"));
    TEST_ASSERT(!ascii_icontains("short", "much longer needle"));
    // Fail-closed inversion: an empty needle is never "found".
    TEST_ASSERT(!ascii_icontains("anything", ""));

    std::cout << "  -> Authorization is exact, case-insensitive, and fail-closed." << std::endl;
}

void test_scenario_completion_is_wall_clock_only() {
    std::cout << "[Test 5] Completion is observable from wall-clock alone..." << std::endl;

    ScenarioAdapter adapter{make_aimlabs_profile()};
    TEST_ASSERT(adapter.capabilities().reports_scenario_completion);
    TEST_ASSERT(adapter.profile().max_engagement_seconds == 60.0f);

    TEST_ASSERT(!adapter.scenario_complete(aimlabs_state(0.0f)));
    TEST_ASSERT(!adapter.scenario_complete(aimlabs_state(59.9f)));
    TEST_ASSERT(adapter.scenario_complete(aimlabs_state(60.0f)));
    TEST_ASSERT(adapter.scenario_complete(aimlabs_state(120.0f)));

    std::cout << "  -> Completion follows the configured wall-clock window." << std::endl;
}

void test_invalid_profiles_are_rejected() {
    std::cout << "[Test 6] Invalid profiles are rejected and leave the adapter closed..." << std::endl;

    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float inf_value = std::numeric_limits<float>::infinity();

    struct Case {
        const char* name;
        ScenarioProfile profile;
        ScenarioProfileStatus expected;
    };

    auto mutate = [](auto fn) {
        ScenarioProfile p = make_aimlabs_profile();
        fn(p);
        return p;
    };

    const Case cases[] = {
        {"empty profile_id", mutate([](ScenarioProfile& p) { p.profile_id.clear(); }),
         ScenarioProfileStatus::empty_profile_id},
        {"profile_id with uppercase",
         mutate([](ScenarioProfile& p) { p.profile_id = "Aimlabs"; }),
         ScenarioProfileStatus::invalid_profile_id},
        {"profile_id with a space",
         mutate([](ScenarioProfile& p) { p.profile_id = "aim labs"; }),
         ScenarioProfileStatus::invalid_profile_id},
        {"profile_id starting with underscore",
         mutate([](ScenarioProfile& p) { p.profile_id = "_leading"; }),
         ScenarioProfileStatus::invalid_profile_id},
        {"profile_id too long",
         mutate([](ScenarioProfile& p) {
             p.profile_id = std::string(scenario_limits::kMaxProfileIdLength + 1, 'a');
         }),
         ScenarioProfileStatus::invalid_profile_id},
        {"negative target value", mutate([](ScenarioProfile& p) { p.default_target_value = -1.0f; }),
         ScenarioProfileStatus::invalid_target_value},
        {"NaN target value", mutate([nan_value](ScenarioProfile& p) { p.default_target_value = nan_value; }),
         ScenarioProfileStatus::invalid_target_value},
        {"target value above the schema maximum",
         mutate([](ScenarioProfile& p) { p.default_target_value = scenario_limits::kMaxTargetValue + 1.0f; }),
         ScenarioProfileStatus::invalid_target_value},
        {"advertises a target value of zero",
         mutate([](ScenarioProfile& p) { p.default_target_value = 0.0f; }),
         ScenarioProfileStatus::capability_without_data},
        {"NaN engagement window", mutate([nan_value](ScenarioProfile& p) { p.max_engagement_seconds = nan_value; }),
         ScenarioProfileStatus::invalid_engagement_window},
        {"infinite engagement window",
         mutate([inf_value](ScenarioProfile& p) { p.max_engagement_seconds = inf_value; }),
         ScenarioProfileStatus::invalid_engagement_window},
        {"engagement window above the schema maximum",
         mutate([](ScenarioProfile& p) {
             p.max_engagement_seconds = scenario_limits::kMaxEngagementSeconds + 1.0f;
         }),
         ScenarioProfileStatus::invalid_engagement_window},
        {"seed claims valid but has zero gain",
         mutate([](ScenarioProfile& p) { p.calibration_seed.counts_per_pixel_x = 0.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"seed claims valid but has negative gain",
         mutate([](ScenarioProfile& p) { p.calibration_seed.counts_per_pixel_y = -2.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"seed gain above the schema maximum",
         mutate([](ScenarioProfile& p) {
             p.calibration_seed.counts_per_pixel_x = scenario_limits::kMaxCountsPerPixel + 1.0f;
         }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"seed FOV at the exclusive maximum",
         mutate([](ScenarioProfile& p) { p.calibration_seed.fov_horizontal_deg = 180.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"seed sensitivity non-positive",
         mutate([](ScenarioProfile& p) { p.calibration_seed.in_game_sensitivity = 0.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"advertises a seed it does not have",
         mutate([](ScenarioProfile& p) { p.calibration_seed.valid = false; }),
         ScenarioProfileStatus::capability_without_data},
        {"advertises an unconfirmable identity", mutate([](ScenarioProfile& p) {
             p.foreground.process_name.clear();
             p.foreground.window_title_substring.clear();
         }),
         ScenarioProfileStatus::unconfirmable_identity},
        {"identity string too long", mutate([](ScenarioProfile& p) {
             p.foreground.process_name = std::string(scenario_limits::kMaxIdentityLength + 1, 'a');
         }),
         ScenarioProfileStatus::unconfirmable_identity},
        {"advertises completion with no window",
         mutate([](ScenarioProfile& p) { p.max_engagement_seconds = 0.0f; }),
         ScenarioProfileStatus::capability_without_data},
        {"normalized crosshair out of range",
         mutate([](ScenarioProfile& p) { p.crosshair.center_norm.x = 1.5f; }),
         ScenarioProfileStatus::invalid_crosshair},
        {"NaN crosshair", mutate([nan_value](ScenarioProfile& p) { p.crosshair.center_px.y = nan_value; }),
         ScenarioProfileStatus::invalid_crosshair},
        {"negative crosshair pixel",
         mutate([](ScenarioProfile& p) { p.crosshair.center_px.x = -1.0f; }),
         ScenarioProfileStatus::invalid_crosshair},
        {"crosshair pixel beyond the schema maximum",
         mutate([](ScenarioProfile& p) { p.crosshair.center_px.x = scenario_limits::kMaxCrosshairPxX + 1.0f; }),
         ScenarioProfileStatus::invalid_crosshair},
        {"advertises a crosshair it marked invalid",
         mutate([](ScenarioProfile& p) { p.crosshair.valid = false; }),
         ScenarioProfileStatus::capability_without_data},
        // --- branches reachable only once the seed's own consistency block runs ---
        {"valid seed with zero y gain",
         mutate([](ScenarioProfile& p) { p.calibration_seed.counts_per_pixel_y = 0.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"valid seed with zero FOV",
         mutate([](ScenarioProfile& p) { p.calibration_seed.fov_horizontal_deg = 0.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"negative engagement window",
         mutate([](ScenarioProfile& p) { p.max_engagement_seconds = -1.0f; }),
         ScenarioProfileStatus::invalid_engagement_window},
        {"sensitivity above the schema maximum",
         mutate([](ScenarioProfile& p) {
             p.calibration_seed.in_game_sensitivity = scenario_limits::kMaxInGameSensitivity + 1.0f;
         }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"NaN FOV",
         mutate([nan_value](ScenarioProfile& p) { p.calibration_seed.fov_horizontal_deg = nan_value; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"negative FOV",
         mutate([](ScenarioProfile& p) { p.calibration_seed.fov_horizontal_deg = -1.0f; }),
         ScenarioProfileStatus::invalid_calibration_seed},
        {"window title too long", mutate([](ScenarioProfile& p) {
             p.foreground.window_title_substring =
                 std::string(scenario_limits::kMaxIdentityLength + 1, 'a');
         }),
         ScenarioProfileStatus::unconfirmable_identity},
        {"title-only identity is not an authorization boundary",
         mutate([](ScenarioProfile& p) { p.foreground.process_name.clear(); }),
         ScenarioProfileStatus::unconfirmable_identity},
        {"normalized crosshair y below range",
         mutate([](ScenarioProfile& p) { p.crosshair.center_norm.y = -1.5f; }),
         ScenarioProfileStatus::invalid_crosshair},
        {"crosshair y beyond the schema maximum",
         mutate([](ScenarioProfile& p) {
             p.crosshair.center_px.y = scenario_limits::kMaxCrosshairPxY + 1.0f;
         }),
         ScenarioProfileStatus::invalid_crosshair},
    };

    for (const Case& c : cases) {
        const ScenarioProfileStatus status = ScenarioAdapter::validate(c.profile);
        if (status != c.expected) {
            std::cerr << "  !! '" << c.name << "' gave status " << to_string(status)
                      << ", expected " << to_string(c.expected) << std::endl;
        }
        TEST_ASSERT(status == c.expected);
        TEST_ASSERT(!ScenarioAdapter::profile_is_valid(c.profile));

        // A rejected load must leave the adapter advertising nothing.
        ScenarioAdapter fresh{};
        TEST_ASSERT(fresh.load_profile(c.profile) == c.expected);
        TEST_ASSERT(!fresh.has_profile());
        TEST_ASSERT(fresh.last_status() == c.expected);
        TEST_ASSERT(fresh.capabilities() == ScenarioCapabilities::none());
        TEST_ASSERT(!fresh.foreground_authorized(aimlabs_state(1.0f)));

        // A rejected load must not silently downgrade an adapter that was already
        // good, and must not leave the previous domain's data readable.
        ScenarioAdapter loaded{make_aimlabs_profile()};
        TEST_ASSERT(loaded.has_profile());
        TEST_ASSERT(loaded.load_profile(c.profile) == c.expected);
        TEST_ASSERT(!loaded.has_profile());
        TEST_ASSERT(loaded.capabilities() == ScenarioCapabilities::none());
        TEST_ASSERT(loaded.profile().foreground.process_name.empty());
        TEST_ASSERT(!loaded.foreground_authorized(aimlabs_state(1.0f)));
    }

    // The constructor taking an invalid profile must also fail closed, and must
    // still report why rather than silently degrading.
    ScenarioProfile bad = make_aimlabs_profile();
    bad.profile_id.clear();
    ScenarioAdapter constructed{bad};
    TEST_ASSERT(!constructed.has_profile());
    TEST_ASSERT(constructed.last_status() == ScenarioProfileStatus::empty_profile_id);
    TEST_ASSERT(constructed.capabilities() == ScenarioCapabilities::none());

    // Both built-in profiles must of course validate.
    TEST_ASSERT(ScenarioAdapter::validate(make_generic_profile()) == ScenarioProfileStatus::ok);
    TEST_ASSERT(ScenarioAdapter::validate(make_aimlabs_profile()) == ScenarioProfileStatus::ok);

    std::cout << "  -> All " << (sizeof(cases) / sizeof(cases[0]))
              << " malformed profiles were rejected with the expected status." << std::endl;
}

void test_profile_is_hot_swappable_through_the_interface() {
    std::cout << "[Test 7] A profile is replaceable at runtime through IScenarioAdapter..." << std::endl;

    ScenarioAdapter concrete{make_aimlabs_profile()};
    IScenarioAdapter& adapter = concrete; // The core only ever sees the interface.

    TEST_ASSERT(adapter.capabilities().reports_scenario_completion);
    TEST_ASSERT(adapter.crosshair_context().valid);
    TEST_ASSERT(adapter.calibration_seed().valid);
    TEST_ASSERT(adapter.foreground_authorized(aimlabs_state(1.0f)));
    TEST_ASSERT(adapter.scenario_complete(aimlabs_state(60.0f)));

    // Swap in the domain-neutral profile: the advertised surface must shrink.
    TEST_ASSERT(adapter.load_profile(make_generic_profile()) == ScenarioProfileStatus::ok);
    TEST_ASSERT(!adapter.capabilities().reports_scenario_completion);
    TEST_ASSERT(!adapter.crosshair_context().valid);
    TEST_ASSERT(!adapter.calibration_seed().valid);
    TEST_ASSERT(!adapter.scenario_complete(aimlabs_state(600.0f)));

    // Counters restart with the new profile.
    TEST_ASSERT(concrete.stamped_targets() == 0);
    bus::TargetObservationBatch batch = make_batch(2, 0.0f);
    adapter.enrich(batch, aimlabs_state(1.0f));
    TEST_ASSERT(concrete.stamped_targets() == 2);
    TEST_ASSERT(concrete.enriched_batches() == 1);

    adapter.reset();
    TEST_ASSERT(concrete.stamped_targets() == 0);
    TEST_ASSERT(concrete.enriched_batches() == 0);
    TEST_ASSERT(concrete.has_profile()); // reset() keeps the loaded profile

    std::cout << "  -> Profiles swap cleanly behind the interface." << std::endl;
}

void test_yaml_loader_accepts_canonical_profiles() {
    std::cout << "[Test 8] Strict YAML loader accepts the canonical profiles..." << std::endl;

    const std::string aimlabs_yaml = R"(
schema_version: 1
profile_id: "aimlabs_gridshot"
default_target_value: 1.0
max_engagement_seconds: 60.0
capabilities:
  provides_target_value: true
  provides_crosshair_context: true
  provides_calibration_seed: true
  provides_foreground_identity: true
  reports_scenario_completion: true
foreground:
  process_name: "Aimlab_tb.exe"
  window_title_substring: "aimlab"
  require_match: true
calibration_seed:
  counts_per_pixel_x: 1.25
  counts_per_pixel_y: 1.25
  fov_horizontal_deg: 103.0
  in_game_sensitivity: 1.0
  valid: true
crosshair:
  center_norm_x: 0.0
  center_norm_y: 0.0
  center_px_x: 960.0
  center_px_y: 540.0
  valid: true
)";

    const ScenarioProfileLoadResult result =
        ScenarioProfileLoader::load_from_yaml_string(aimlabs_yaml, "aimlabs.yaml");
    TEST_ASSERT(result.success);
    TEST_ASSERT(result.status == ScenarioProfileStatus::ok);
    TEST_ASSERT(result.errors.empty());
    TEST_ASSERT(result.source == "aimlabs.yaml");
    TEST_ASSERT(result.sha256_hash.size() == 64);

    // The parsed profile must equal the built-in Aimlabs profile exactly.
    TEST_ASSERT(result.profile == make_aimlabs_profile());

    // Hashing is deterministic and profile-sensitive.
    const ScenarioProfileLoadResult again =
        ScenarioProfileLoader::load_from_yaml_string(aimlabs_yaml, "aimlabs.yaml");
    TEST_ASSERT(again.sha256_hash == result.sha256_hash);

    const std::string generic_yaml = R"(
schema_version: 1
profile_id: "generic_targets"
default_target_value: 1.0
max_engagement_seconds: 0.0
capabilities:
  provides_target_value: true
  provides_foreground_identity: true
foreground:
  require_match: false
crosshair:
  center_px_x: 960.0
  center_px_y: 540.0
  valid: false
)";
    const ScenarioProfileLoadResult generic =
        ScenarioProfileLoader::load_from_yaml_string(generic_yaml, "generic.yaml");
    TEST_ASSERT(generic.success);
    TEST_ASSERT(generic.profile == make_generic_profile());
    TEST_ASSERT(generic.sha256_hash != result.sha256_hash);

    std::cout << "  -> Canonical profiles round-trip with stable hashes." << std::endl;
}

void test_yaml_loader_rejects_malformed_documents() {
    std::cout << "[Test 9] Strict YAML loader rejects malformed documents..." << std::endl;

    // `expected` is the structural verdict. Documents rejected before structural
    // validation (parse errors, unknown keys, type errors) never reach validate(),
    // so they keep the default `ok` status and are distinguished by `errors`.
    struct Case {
        const char* name;
        const char* yaml;
        ScenarioProfileStatus expected;
    };

    constexpr ScenarioProfileStatus kShapeError = ScenarioProfileStatus::ok;

    // Every well-formed case carries `schema_version: 1`, so a case is never rejected
    // for a missing version when it is meant to exercise some other rule.
    const Case cases[] = {
        {"unknown root key", "schema_version: 1\nprofile_id: \"x\"\nunexpected_key: 1\n", kShapeError},
        {"unknown nested key",
         "schema_version: 1\nprofile_id: \"x\"\ncapabilities:\n  provides_score: true\n", kShapeError},
        {"missing profile_id", "schema_version: 1\ndefault_target_value: 1.0\n", kShapeError},
        {"wrong scalar type",
         "schema_version: 1\nprofile_id: \"x\"\ndefault_target_value: \"not-a-number\"\n", kShapeError},
        {"wrong boolean type",
         "schema_version: 1\nprofile_id: \"x\"\ncapabilities:\n  provides_target_value: \"yes-ish\"\n",
         kShapeError},
        {"root is a sequence", "- profile_id: \"x\"\n", kShapeError},
        {"root is a scalar", "just-a-string\n", kShapeError},
        {"section is not a mapping", "schema_version: 1\nprofile_id: \"x\"\ncapabilities: 5\n", kShapeError},
        {"unterminated flow sequence", "schema_version: 1\nprofile_id: [unclosed\n", kShapeError},
        {"tab indentation",
         "schema_version: 1\nprofile_id: \"x\"\ncapabilities:\n\tprovides_target_value: true\n", kShapeError},
        // A versioned document family must reject anything it cannot migrate.
        {"missing schema_version", "profile_id: \"x\"\n", kShapeError},
        {"future schema_version", "schema_version: 2\nprofile_id: \"x\"\n", kShapeError},
        {"fractional schema_version", "schema_version: 1.5\nprofile_id: \"x\"\n", kShapeError},
        {"textual schema_version", "schema_version: one\nprofile_id: \"x\"\n", kShapeError},
        // Note: `schema_version: "1"` is NOT tested as a failure. ConfigLoader::yaml_to_json
        // type-guesses from the raw scalar and discards YAML quoting, so a quoted "1"
        // arrives as a JSON integer. That coercion is shared by every config family in
        // this repository; it is not a scenario-profile behaviour to override here.
        // These parse cleanly, so validate() must be what rejects them.
        {"fails range validation",
         "schema_version: 1\nprofile_id: \"x\"\ncrosshair:\n  center_norm_x: 4.0\n",
         ScenarioProfileStatus::invalid_crosshair},
        {"profile_id violates the schema pattern",
         "schema_version: 1\nprofile_id: \"Not Allowed\"\n",
         ScenarioProfileStatus::invalid_profile_id},
        {"exceeds the schema maximum",
         "schema_version: 1\nprofile_id: \"x\"\nmax_engagement_seconds: 100000.0\n",
         ScenarioProfileStatus::invalid_engagement_window},
        {"advertises an unconfirmable identity",
         "schema_version: 1\nprofile_id: \"x\"\ncapabilities:\n  provides_foreground_identity: true\n"
         "foreground:\n  require_match: true\n",
         ScenarioProfileStatus::unconfirmable_identity},
        {"advertises a target value of zero",
         "schema_version: 1\nprofile_id: \"x\"\ndefault_target_value: 0.0\n"
         "capabilities:\n  provides_target_value: true\n",
         ScenarioProfileStatus::capability_without_data},
    };

    for (const Case& c : cases) {
        const ScenarioProfileLoadResult result =
            ScenarioProfileLoader::load_from_yaml_string(c.yaml, c.name);
        if (result.success) {
            std::cerr << "  !! document '" << c.name
                      << "' was accepted but must be rejected" << std::endl;
        }
        TEST_ASSERT(!result.success);
        TEST_ASSERT(!result.errors.empty());
        // The structural verdict must reach the caller, not just a message.
        if (result.status != c.expected) {
            std::cerr << "  !! document '" << c.name << "' reported status "
                      << to_string(result.status) << ", expected "
                      << to_string(c.expected) << std::endl;
        }
        TEST_ASSERT(result.status == c.expected);
        // A rejected document must yield no hash and a profile that advertises
        // nothing, so a caller that ignores `success` still gets a fail-closed adapter.
        TEST_ASSERT(result.sha256_hash.empty());
        TEST_ASSERT(result.profile.capabilities == ScenarioCapabilities::none());
        TEST_ASSERT(!ScenarioAdapter{result.profile}.foreground_authorized(aimlabs_state(1.0f)));
    }

    // A missing file is an error, not a crash.
    const ScenarioProfileLoadResult missing =
        ScenarioProfileLoader::load_from_file("configs/scenario/does_not_exist.yaml");
    TEST_ASSERT(!missing.success);
    TEST_ASSERT(!missing.errors.empty());

    std::cout << "  -> All " << (sizeof(cases) / sizeof(cases[0]))
              << " malformed documents were rejected." << std::endl;
}

void test_enrich_is_bounded_and_repeatable() {
    std::cout << "[Test 10] enrich() stays bounded over a sustained loop..." << std::endl;

    ScenarioAdapter adapter{make_aimlabs_profile()};

    // A full batch must be stamped completely and never past capacity.
    bus::TargetObservationBatch full =
        make_batch(static_cast<std::uint32_t>(bus::kMaxObservations), 0.0f);
    TEST_ASSERT(full.target_count == bus::kMaxObservations);
    adapter.enrich(full, aimlabs_state(1.0f));
    TEST_ASSERT(adapter.stamped_targets() == bus::kMaxObservations);
    for (std::uint32_t i = 0; i < full.target_count; ++i) {
        TEST_ASSERT(full.targets[i].target_value == 1.0f);
    }

    // A batch whose count was corrupted beyond capacity must be clamped, never
    // read or written out of bounds.
    bus::TargetObservationBatch overflowing = make_batch(4, 0.0f);
    overflowing.target_count = static_cast<std::uint32_t>(bus::kMaxObservations) + 32;
    adapter.reset();
    adapter.enrich(overflowing, aimlabs_state(1.0f));
    TEST_ASSERT(adapter.stamped_targets() == bus::kMaxObservations);

    adapter.reset();

    // Sustained repetition must not drift or accumulate state.
    constexpr std::uint64_t kIterations = 10'000;
    for (std::uint64_t i = 0; i < kIterations; ++i) {
        bus::TargetObservationBatch batch = make_batch(4, 0.0f);
        adapter.enrich(batch, aimlabs_state(1.0f));
        TEST_ASSERT(batch.targets[0].target_value == 1.0f);
        TEST_ASSERT(batch.targets[3].target_value == 1.0f);
    }
    TEST_ASSERT(adapter.enriched_batches() == kIterations);
    TEST_ASSERT(adapter.stamped_targets() == kIterations * 4);

    std::cout << "  -> " << kIterations << " enrich() calls remained bounded and exact." << std::endl;
}

void test_shipped_profiles_load_from_disk() {
    std::cout << "[Test 11] The shipped configs/scenario/*.yaml profiles load and match..." << std::endl;

    const std::filesystem::path repo_root{AIM_REPO_ROOT};
    const std::filesystem::path scenario_dir = repo_root / "configs" / "scenario";

    // The real Aimlabs profile on disk must parse and equal the built-in helper.
    // Without this, editing aimlabs.yaml would break nothing.
    const ScenarioProfileLoadResult aimlabs =
        ScenarioProfileLoader::load_from_file(scenario_dir / "aimlabs.yaml");
    if (!aimlabs.success) {
        for (const std::string& e : aimlabs.errors) {
            std::cerr << "  !! configs/scenario/aimlabs.yaml: " << e << std::endl;
        }
    }
    TEST_ASSERT(aimlabs.success);
    TEST_ASSERT(aimlabs.status == ScenarioProfileStatus::ok);
    TEST_ASSERT(aimlabs.profile == make_aimlabs_profile());
    TEST_ASSERT(aimlabs.sha256_hash.size() == 64);

    const ScenarioProfileLoadResult generic =
        ScenarioProfileLoader::load_from_file(scenario_dir / "generic.yaml");
    if (!generic.success) {
        for (const std::string& e : generic.errors) {
            std::cerr << "  !! configs/scenario/generic.yaml: " << e << std::endl;
        }
    }
    TEST_ASSERT(generic.success);
    TEST_ASSERT(generic.profile == make_generic_profile());
    TEST_ASSERT(generic.sha256_hash != aimlabs.sha256_hash);

    // Both must drive a working adapter straight off disk.
    ScenarioAdapter from_disk{};
    TEST_ASSERT(from_disk.load_profile(aimlabs.profile) == ScenarioProfileStatus::ok);
    TEST_ASSERT(from_disk.foreground_authorized(aimlabs_state(1.0f)));
    auto actual_title = aimlabs_state(1.0f);
    actual_title.active_window_title = "aimlab_tb";
    TEST_ASSERT(from_disk.foreground_authorized(actual_title));
    actual_title.active_process_name = "unrelated.exe";
    TEST_ASSERT(!from_disk.foreground_authorized(actual_title));
    TEST_ASSERT(from_disk.scenario_complete(aimlabs_state(60.0f)));

    // Loading the same file twice yields the same hash: no ordering or float drift.
    const ScenarioProfileLoadResult again =
        ScenarioProfileLoader::load_from_file(scenario_dir / "aimlabs.yaml");
    TEST_ASSERT(again.sha256_hash == aimlabs.sha256_hash);

    std::cout << "  -> Both shipped profiles match their built-in equivalents." << std::endl;
}

} // namespace

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " Running OpenPrism M7-01 Scenario Adapter Unit Tests            " << std::endl;
    std::cout << "================================================================" << std::endl;

    test_unloaded_adapter_fails_closed();
    test_generic_profile_is_domain_neutral();
    test_enrich_touches_only_target_value();
    test_foreground_authorization_is_injection_free();
    test_scenario_completion_is_wall_clock_only();
    test_invalid_profiles_are_rejected();
    test_profile_is_hot_swappable_through_the_interface();
    test_yaml_loader_accepts_canonical_profiles();
    test_yaml_loader_rejects_malformed_documents();
    test_enrich_is_bounded_and_repeatable();
    test_shipped_profiles_load_from_disk();

    std::cout << "================================================================" << std::endl;
    std::cout << " All M7-01 Scenario Adapter Tests Passed Successfully!          " << std::endl;
    std::cout << "================================================================" << std::endl;
    return 0;
}
