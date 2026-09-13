// include/aim/scenario/scenario_adapter.hpp
#pragma once

#include <cstdint>

#include "aim/interfaces/scenario_adapter.hpp"

namespace aim::scenario {

/// @brief Single, fully profile-driven implementation of @ref IScenarioAdapter.
///
/// The same class serves every domain; a domain is only a @ref ScenarioProfile.
/// Before a profile validates, and for any capability the profile does not
/// advertise, every accessor returns the unavailable / fail-closed default.
///
/// @par Thread safety
/// Not internally synchronized, and deliberately so: the per-frame accessors must
/// stay lock-free. @ref load_profile() and @ref reset() mutate state that
/// @ref foreground_authorized() and @ref enrich() read, so a host must either own
/// this object on a single thread or quiesce the frame loop before swapping a
/// profile. Structural configuration changes require a quiescent restart in any
/// case; only bounded tuning values are hot-reloadable in this system.
class ScenarioAdapter final : public IScenarioAdapter {
public:
    ScenarioAdapter() noexcept = default;

    /// @brief Construct and attempt to load @p profile.
    ///
    /// A rejected profile leaves the adapter fail-closed; the reason is never lost,
    /// query it with @ref last_status(). Prefer default construction followed by a
    /// checked @ref load_profile() where the caller can act on the failure.
    explicit ScenarioAdapter(const ScenarioProfile& profile) noexcept;

    [[nodiscard]] ScenarioCapabilities capabilities() const noexcept override;
    ScenarioProfileStatus load_profile(const ScenarioProfile& profile) noexcept override;
    void enrich(bus::TargetObservationBatch& observations,
                const ObservableScenarioState& state) noexcept override;
    [[nodiscard]] CalibrationSeed calibration_seed() const noexcept override;
    [[nodiscard]] CrosshairContext crosshair_context() const noexcept override;
    [[nodiscard]] ForegroundIdentity foreground_identity() const noexcept override;
    [[nodiscard]] bool foreground_authorized(const ObservableScenarioState& state) const noexcept override;
    [[nodiscard]] bool scenario_complete(const ObservableScenarioState& state) const noexcept override;
    void reset() noexcept override;

    /// @brief The currently loaded profile.
    /// @warning Only meaningful when @ref has_profile() is true; a rejected load
    /// resets this to a default-constructed profile so no stale domain data leaks.
    [[nodiscard]] const ScenarioProfile& profile() const noexcept { return profile_; }
    [[nodiscard]] bool has_profile() const noexcept { return loaded_; }

    /// @brief Result of the most recent @ref load_profile() attempt.
    [[nodiscard]] ScenarioProfileStatus last_status() const noexcept { return last_status_; }

    /// @brief Number of batches actually enriched since construction or the last
    /// reset(). Calls that stamped nothing (no profile, or the capability is not
    /// advertised) are not counted. A successful load_profile() also restarts the
    /// counters, so they always describe the profile currently loaded.
    [[nodiscard]] std::uint64_t enriched_batches() const noexcept { return enriched_batches_; }
    /// @brief Number of individual observations stamped by enrich().
    [[nodiscard]] std::uint64_t stamped_targets() const noexcept { return stamped_targets_; }

    /// @brief Validate a profile without mutating adapter state.
    ///
    /// Enforces exactly the ranges declared by
    /// `schemas/config/scenario_profile.schema.json`, so the C++ contract and the
    /// JSON Schema cannot drift apart.
    [[nodiscard]] static ScenarioProfileStatus validate(const ScenarioProfile& profile) noexcept;

    /// @brief Convenience predicate over @ref validate().
    [[nodiscard]] static bool profile_is_valid(const ScenarioProfile& profile) noexcept {
        return validate(profile) == ScenarioProfileStatus::ok;
    }

private:
    /// @brief Whether a validated profile actually advertises a foreground identity.
    [[nodiscard]] bool identity_advertised() const noexcept;

    ScenarioProfile profile_{};
    bool loaded_{false};
    ScenarioProfileStatus last_status_{ScenarioProfileStatus::ok};
    std::uint64_t enriched_batches_{0};
    std::uint64_t stamped_targets_{0};
};

/// @brief Built-in generic (domain-neutral) profile — matches configs/scenario/generic.yaml.
///
/// Advertises a uniform target value and an explicitly UNCONSTRAINED foreground
/// identity (`require_match = false`), which is what makes it usable for offline
/// replay and portability runs. It must never drive live actuation against a real
/// application; use a profile with a real identity for that.
[[nodiscard]] ScenarioProfile make_generic_profile() noexcept;

/// @brief Built-in thin Aimlabs profile — matches configs/scenario/aimlabs.yaml.
///
/// Supplies a uniform target value, a crosshair/calibration seed, an authorized
/// foreground identity, and a 60 s wall-clock engagement window. It advertises no
/// score, recoil, movement, or weapon capability and reads nothing from the game.
[[nodiscard]] ScenarioProfile make_aimlabs_profile() noexcept;

} // namespace aim::scenario
