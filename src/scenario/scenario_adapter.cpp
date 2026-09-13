// src/scenario/scenario_adapter.cpp
#include "aim/scenario/scenario_adapter.hpp"

#include <cmath>

namespace aim::scenario {

namespace {

using namespace aim::scenario_limits;

[[nodiscard]] bool in_range(float v, float lo, float hi) noexcept {
    return std::isfinite(v) && v >= lo && v <= hi;
}

/// @brief Mirrors the JSON Schema's `^[a-z0-9][a-z0-9_]*$` with `maxLength: 64`.
[[nodiscard]] bool profile_id_is_well_formed(const std::string& id) noexcept {
    // Callers reject the empty id first, but front() must never be reached on an
    // empty string even if this helper is reused.
    if (id.empty() || id.size() > kMaxProfileIdLength) {
        return false;
    }
    const char first = id.front();
    const bool first_ok = (first >= 'a' && first <= 'z') || (first >= '0' && first <= '9');
    if (!first_ok) {
        return false;
    }
    for (const char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) {
            return false;
        }
    }
    return true;
}

} // namespace

ScenarioProfileStatus ScenarioAdapter::validate(const ScenarioProfile& profile) noexcept {
    if (profile.profile_id.empty()) {
        return ScenarioProfileStatus::empty_profile_id;
    }
    if (!profile_id_is_well_formed(profile.profile_id)) {
        return ScenarioProfileStatus::invalid_profile_id;
    }

    if (!in_range(profile.default_target_value, 0.0f, kMaxTargetValue)) {
        return ScenarioProfileStatus::invalid_target_value;
    }
    // A profile that advertises a target value must supply a usable one; stamping 0
    // would silently erase the bus default of 1.0 for every observation.
    if (profile.capabilities.provides_target_value && !(profile.default_target_value > 0.0f)) {
        return ScenarioProfileStatus::capability_without_data;
    }

    if (!in_range(profile.max_engagement_seconds, 0.0f, kMaxEngagementSeconds)) {
        return ScenarioProfileStatus::invalid_engagement_window;
    }

    const CalibrationSeed& seed = profile.calibration_seed;
    if (!in_range(seed.counts_per_pixel_x, 0.0f, kMaxCountsPerPixel) ||
        !in_range(seed.counts_per_pixel_y, 0.0f, kMaxCountsPerPixel) ||
        !in_range(seed.in_game_sensitivity, 0.0f, kMaxInGameSensitivity) ||
        !std::isfinite(seed.fov_horizontal_deg) || seed.fov_horizontal_deg < 0.0f ||
        seed.fov_horizontal_deg >= kMaxFovDegExclusive) {
        return ScenarioProfileStatus::invalid_calibration_seed;
    }
    if (seed.valid) {
        // A seed that claims to be usable must be usable.
        if (seed.counts_per_pixel_x <= 0.0f || seed.counts_per_pixel_y <= 0.0f ||
            seed.fov_horizontal_deg <= 0.0f || seed.in_game_sensitivity <= 0.0f) {
            return ScenarioProfileStatus::invalid_calibration_seed;
        }
    }
    if (profile.capabilities.provides_calibration_seed && !seed.valid) {
        return ScenarioProfileStatus::capability_without_data;
    }

    const ForegroundIdentity& fg = profile.foreground;
    if (fg.process_name.size() > kMaxIdentityLength ||
        fg.window_title_substring.size() > kMaxIdentityLength) {
        return ScenarioProfileStatus::unconfirmable_identity;
    }
    if (profile.capabilities.provides_foreground_identity && fg.require_match &&
        fg.process_name.empty()) {
        // A constrained identity must name a process. A window title alone is not an
        // authorization boundary: any application can set its own title to contain an
        // arbitrary substring, so a title-only rule authorizes an arbitrary process.
        return ScenarioProfileStatus::unconfirmable_identity;
    }

    if (profile.capabilities.reports_scenario_completion && profile.max_engagement_seconds <= 0.0f) {
        return ScenarioProfileStatus::capability_without_data;
    }

    const CrosshairContext& xh = profile.crosshair;
    if (!in_range(xh.center_norm.x, -1.0f, 1.0f) || !in_range(xh.center_norm.y, -1.0f, 1.0f) ||
        !in_range(xh.center_px.x, 0.0f, kMaxCrosshairPxX) ||
        !in_range(xh.center_px.y, 0.0f, kMaxCrosshairPxY)) {
        return ScenarioProfileStatus::invalid_crosshair;
    }
    if (profile.capabilities.provides_crosshair_context && !xh.valid) {
        return ScenarioProfileStatus::capability_without_data;
    }

    return ScenarioProfileStatus::ok;
}

ScenarioAdapter::ScenarioAdapter(const ScenarioProfile& profile) noexcept {
    (void)load_profile(profile);
}

ScenarioCapabilities ScenarioAdapter::capabilities() const noexcept {
    return loaded_ ? profile_.capabilities : ScenarioCapabilities::none();
}

ScenarioProfileStatus ScenarioAdapter::load_profile(const ScenarioProfile& profile) noexcept {
    const ScenarioProfileStatus status = validate(profile);
    last_status_ = status;
    if (status != ScenarioProfileStatus::ok) {
        // Fail closed: advertise nothing and retain no stale domain data.
        profile_ = ScenarioProfile{};
        loaded_ = false;
        enriched_batches_ = 0;
        stamped_targets_ = 0;
        return status;
    }
    profile_ = profile;
    loaded_ = true;
    enriched_batches_ = 0;
    stamped_targets_ = 0;
    return ScenarioProfileStatus::ok;
}

void ScenarioAdapter::enrich(bus::TargetObservationBatch& observations,
                             const ObservableScenarioState& state) noexcept {
    (void)state; // The thin profile does not weight by observable state.

    if (!loaded_ || !profile_.capabilities.provides_target_value) {
        return;
    }

    const float value = profile_.default_target_value;
    const std::uint32_t count =
        (observations.target_count < bus::kMaxObservations)
            ? observations.target_count
            : static_cast<std::uint32_t>(bus::kMaxObservations);

    for (std::uint32_t i = 0; i < count; ++i) {
        observations.targets[i].target_value = value;
    }

    ++enriched_batches_;
    stamped_targets_ += count;
}

CalibrationSeed ScenarioAdapter::calibration_seed() const noexcept {
    if (loaded_ && profile_.capabilities.provides_calibration_seed) {
        return profile_.calibration_seed;
    }
    return CalibrationSeed{};
}

CrosshairContext ScenarioAdapter::crosshair_context() const noexcept {
    if (loaded_ && profile_.capabilities.provides_crosshair_context) {
        return profile_.crosshair;
    }
    // Fail-closed default: an explicitly invalid context the consumer must not use.
    return CrosshairContext{.center_norm = {0.0f, 0.0f}, .center_px = {0.0f, 0.0f}, .valid = false};
}

bool ScenarioAdapter::identity_advertised() const noexcept {
    return loaded_ && profile_.capabilities.provides_foreground_identity;
}

ForegroundIdentity ScenarioAdapter::foreground_identity() const noexcept {
    if (identity_advertised()) {
        return profile_.foreground;
    }
    // Fail-closed default: require_match == true with no confirmable pattern.
    return ForegroundIdentity{};
}

bool ScenarioAdapter::foreground_authorized(const ObservableScenarioState& state) const noexcept {
    // No advertised identity means no authority to authorize. Checked first so the
    // per-frame path touches no std::string at all in the common refusal case.
    if (!identity_advertised()) {
        return false;
    }

    // The host's own OS-level focus result is non-negotiable and is checked BEFORE
    // require_match. An unconstrained identity relaxes *which application* may be
    // targeted; it must never relax *whether our window is actually focused*, or an
    // unfocused desktop would authorize actuation.
    if (!state.foreground_confirmed) {
        return false;
    }

    // Bound by reference: matching never copies the profile's strings.
    const ForegroundIdentity& identity = profile_.foreground;
    if (!identity.require_match) {
        return true;
    }
    return identity.matches(state.active_process_name, state.active_window_title);
}

bool ScenarioAdapter::scenario_complete(const ObservableScenarioState& state) const noexcept {
    if (!loaded_ || !profile_.capabilities.reports_scenario_completion) {
        return false;
    }
    if (profile_.max_engagement_seconds <= 0.0f) {
        return false;
    }
    return state.elapsed_session_seconds >= profile_.max_engagement_seconds;
}

void ScenarioAdapter::reset() noexcept {
    enriched_batches_ = 0;
    stamped_targets_ = 0;
}

ScenarioProfile make_generic_profile() noexcept {
    ScenarioProfile profile{};
    profile.profile_id = "generic_targets";
    profile.default_target_value = 1.0f;
    profile.max_engagement_seconds = 0.0f;
    profile.capabilities = ScenarioCapabilities{
        .provides_target_value = true,
        .provides_crosshair_context = false,
        .provides_calibration_seed = false,
        // Advertised so that the "intentionally unconstrained" choice below is
        // visible to consumers rather than being silently replaced by the
        // fail-closed default. Offline replay and portability only.
        .provides_foreground_identity = true,
        .reports_scenario_completion = false,
    };
    profile.foreground = ForegroundIdentity{.process_name = {}, .window_title_substring = {}, .require_match = false};
    profile.calibration_seed = CalibrationSeed{};
    profile.crosshair = CrosshairContext{.center_norm = {0.0f, 0.0f}, .center_px = {960.0f, 540.0f}, .valid = false};
    return profile;
}

ScenarioProfile make_aimlabs_profile() noexcept {
    ScenarioProfile profile{};
    profile.profile_id = "aimlabs_gridshot";
    profile.default_target_value = 1.0f;   // gridshot: every target is worth the same
    profile.max_engagement_seconds = 60.0f; // canonical Aimlabs task length (wall-clock only)
    profile.capabilities = ScenarioCapabilities{
        .provides_target_value = true,
        .provides_crosshair_context = true,
        .provides_calibration_seed = true,
        .provides_foreground_identity = true,
        .reports_scenario_completion = true,
    };
    profile.foreground = ForegroundIdentity{
        .process_name = "Aimlab_tb.exe",
        .window_title_substring = "aimlab",
        .require_match = true,
    };
    profile.calibration_seed = CalibrationSeed{
        .counts_per_pixel_x = 1.25f,
        .counts_per_pixel_y = 1.25f,
        .fov_horizontal_deg = 103.0f,
        .in_game_sensitivity = 1.0f,
        .valid = true,
    };
    profile.crosshair = CrosshairContext{
        .center_norm = {0.0f, 0.0f},
        .center_px = {960.0f, 540.0f},
        .valid = true,
    };
    return profile;
}

} // namespace aim::scenario
