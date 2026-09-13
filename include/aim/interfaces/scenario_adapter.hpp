// include/aim/interfaces/scenario_adapter.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "aim/bus/bus_traits.hpp"
#include "aim/core/types.hpp"

namespace aim {

/// @file
/// @brief Contract for the optional benchmark scenario / game adapter (blueprint 6.7).
///
/// The adapter supplies ONLY generic, domain-neutral context to the reusable core:
/// a generic per-target value, crosshair/calibration seeds, capability flags, and an
/// authorized foreground identity. It never introduces game-specific types, and it
/// never inspects another process' memory or injects code. Tracking, aim policy,
/// trajectory planning, calibration, actuation, and safety compile and run without
/// any knowledge of this header.
///
/// @par Freshness
/// This adapter deliberately implements no staleness gate of its own.
/// @ref aim::safety::SafetySupervisor owns the single freshness policy for the whole
/// system (M6-05: a hard 10 ms cutoff on every actuation command). A second, private
/// deadline here would be a competing source of truth for a safety-critical rule.
/// @ref ObservableScenarioState::observed_at_ns is therefore carried for correlation
/// and telemetry, and the supervisor remains the authority on stale work.

/// @brief Why a scenario profile was accepted or rejected.
///
/// Mirrors the blueprint's `Status load_profile(...)` and the repository convention of
/// a per-domain status enum (compare @ref PerceptionStatus, @ref SafetyReason).
/// A rejected profile always leaves the adapter fully fail-closed.
enum class ScenarioProfileStatus : std::uint32_t {
    ok = 0,
    empty_profile_id = 1,
    invalid_profile_id = 2,     ///< too long, or characters outside [a-z0-9_]
    invalid_target_value = 3,   ///< not finite, negative, or out of range
    invalid_engagement_window = 4,
    invalid_calibration_seed = 5,
    invalid_crosshair = 6,
    unconfirmable_identity = 7, ///< advertises an identity that can never be confirmed
    capability_without_data = 8 ///< advertises a capability the profile does not back
};

/// @brief Human-readable status name, for diagnostics and test failure messages.
[[nodiscard]] constexpr std::string_view to_string(ScenarioProfileStatus status) noexcept {
    switch (status) {
        case ScenarioProfileStatus::ok:                        return "ok";
        case ScenarioProfileStatus::empty_profile_id:          return "empty_profile_id";
        case ScenarioProfileStatus::invalid_profile_id:        return "invalid_profile_id";
        case ScenarioProfileStatus::invalid_target_value:      return "invalid_target_value";
        case ScenarioProfileStatus::invalid_engagement_window: return "invalid_engagement_window";
        case ScenarioProfileStatus::invalid_calibration_seed:  return "invalid_calibration_seed";
        case ScenarioProfileStatus::invalid_crosshair:         return "invalid_crosshair";
        case ScenarioProfileStatus::unconfirmable_identity:    return "unconfirmable_identity";
        case ScenarioProfileStatus::capability_without_data:   return "capability_without_data";
    }
    return "unknown";
}

/// @brief Accepted ranges for every scenario profile field.
///
/// These are the single C++ source of truth and are mirrored exactly by
/// `schemas/config/scenario_profile.schema.json`. A parity test keeps the two in step,
/// so YAML that the schema rejects can never be accepted by the loader.
namespace scenario_limits {

inline constexpr std::size_t kMaxProfileIdLength = 64;
inline constexpr std::size_t kMaxIdentityLength = 260; ///< Windows MAX_PATH
inline constexpr float kMaxTargetValue = 100.0f;
inline constexpr float kMaxEngagementSeconds = 600.0f;
inline constexpr float kMaxCountsPerPixel = 1000.0f;
inline constexpr float kMaxInGameSensitivity = 100.0f;
inline constexpr float kMaxFovDegExclusive = 180.0f;
inline constexpr float kMaxCrosshairPxX = 7680.0f;
inline constexpr float kMaxCrosshairPxY = 4320.0f;

} // namespace scenario_limits

/// @brief Optional capabilities an adapter may advertise.
///
/// Every flag defaults to @c false. A consumer must treat an un-advertised
/// capability as unavailable and fall back to its own safe default; it must not
/// assume the adapter fills in a value.
struct ScenarioCapabilities {
    bool provides_target_value{false};        ///< enrich() stamps a generic target value
    bool provides_crosshair_context{false};   ///< a crosshair seed is meaningful
    bool provides_calibration_seed{false};    ///< a starting calibration hint is available
    bool provides_foreground_identity{false}; ///< an authorized foreground identity is defined
    bool reports_scenario_completion{false};  ///< scenario end can be observed (wall-clock only)

    [[nodiscard]] static constexpr ScenarioCapabilities none() noexcept { return {}; }
    [[nodiscard]] bool operator==(const ScenarioCapabilities&) const = default;
};

/// @brief ASCII, allocation-free, case-insensitive equality.
[[nodiscard]] inline bool ascii_iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') { ca = static_cast<char>(ca - 'A' + 'a'); }
        if (cb >= 'A' && cb <= 'Z') { cb = static_cast<char>(cb - 'A' + 'a'); }
        if (ca != cb) {
            return false;
        }
    }
    return true;
}

/// @brief ASCII, allocation-free, case-insensitive substring test.
///
/// @note Deliberately fail-closed and therefore NOT equivalent to
/// `std::string_view::contains`: an EMPTY needle is never considered found. An empty
/// pattern means "unspecified", and an unspecified pattern must never authorize a
/// match on its own. Callers relying on standard contains-semantics must special-case
/// the empty needle themselves.
[[nodiscard]] inline bool ascii_icontains(std::string_view haystack, std::string_view needle) noexcept {
    if (needle.empty()) {
        return false;
    }
    if (needle.size() > haystack.size()) {
        return false;
    }
    const std::size_t last = haystack.size() - needle.size();
    for (std::size_t start = 0; start <= last; ++start) {
        bool hit = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            char ch = haystack[start + j];
            char cn = needle[j];
            if (ch >= 'A' && ch <= 'Z') { ch = static_cast<char>(ch - 'A' + 'a'); }
            if (cn >= 'A' && cn <= 'Z') { cn = static_cast<char>(cn - 'A' + 'a'); }
            if (ch != cn) {
                hit = false;
                break;
            }
        }
        if (hit) {
            return true;
        }
    }
    return false;
}

/// @brief Authorized foreground identity the adapter is allowed to act against.
///
/// This is pure configuration data. The adapter never enumerates processes or
/// reads another process' memory; a host performs the OS foreground query and
/// passes the observed strings back in through @ref ObservableScenarioState.
struct ForegroundIdentity {
    std::string process_name{};           ///< exact executable name, e.g. "Aimlab_tb.exe" ("" = unspecified)
    std::string window_title_substring{}; ///< case-insensitive title substring ("" = unspecified)
    bool require_match{true};             ///< when true, an unconfirmed identity fails closed

    /// @brief Injection-free authorization test against an already-observed window.
    ///
    /// Fail-closed semantics: every non-empty pattern must match, and at least one
    /// pattern must be specified. With @c require_match disabled the identity is
    /// treated as intentionally unconstrained (offline replay / portability runs);
    /// such a profile must never be used to drive live actuation against a real
    /// application, and a validating loader accepts it only as an explicit choice.
    ///
    /// @note This tests IDENTITY only. It says nothing about whether our window is
    /// focused; @ref IScenarioAdapter::foreground_authorized() checks the host's
    /// focus result separately and unconditionally.
    ///
    /// @warning A window title alone is not an authorization boundary - any
    /// application can set its own title. @ref ScenarioAdapter::validate() therefore
    /// requires a process name whenever @c require_match is set.
    [[nodiscard]] bool matches(std::string_view active_process_name,
                               std::string_view active_window_title) const noexcept {
        if (!require_match) {
            return true;
        }
        bool any_pattern = false;
        if (!process_name.empty()) {
            any_pattern = true;
            if (!ascii_iequals(active_process_name, process_name)) {
                return false;
            }
        }
        if (!window_title_substring.empty()) {
            any_pattern = true;
            if (!ascii_icontains(active_window_title, window_title_substring)) {
                return false;
            }
        }
        return any_pattern;
    }

    [[nodiscard]] bool operator==(const ForegroundIdentity&) const = default;
};

/// @brief Generic calibration starting point.
///
/// A seed is only a hint for the online calibrator (M6-04); it never replaces a
/// measured calibration. @c valid defaults to @c false so an adapter that has no
/// seed contributes nothing.
struct CalibrationSeed {
    float counts_per_pixel_x{0.0f};
    float counts_per_pixel_y{0.0f};
    float fov_horizontal_deg{0.0f};
    float in_game_sensitivity{0.0f};
    bool valid{false};

    [[nodiscard]] bool operator==(const CalibrationSeed&) const = default;
};

/// @brief Screen-space crosshair seed handed to the aim policy.
struct CrosshairContext {
    NormalizedPoint center_norm{0.0f, 0.0f};
    PixelPoint center_px{960.0f, 540.0f};
    bool valid{true};

    /// @note Written out rather than defaulted: the shared @ref NormalizedPoint and
    /// @ref PixelPoint core types carry no equality operator, so a defaulted
    /// comparison here would be silently deleted and would take
    /// @ref ScenarioProfile's comparison down with it.
    [[nodiscard]] bool operator==(const CrosshairContext& other) const noexcept {
        return center_norm.x == other.center_norm.x &&
               center_norm.y == other.center_norm.y &&
               center_px.x == other.center_px.x &&
               center_px.y == other.center_px.y &&
               valid == other.valid;
    }
};

/// @brief The resolved, fully generic scenario profile.
///
/// Contains no game-specific types or semantics. "Aimlabs" is expressed only as a
/// set of values in this struct (or the equivalent YAML), never as a C++ type.
struct ScenarioProfile {
    std::string profile_id{"generic"};
    float default_target_value{1.0f};   ///< uniform generic reward weight applied by enrich()
    float max_engagement_seconds{0.0f}; ///< 0 = no adapter-imposed engagement cap
    ScenarioCapabilities capabilities{};
    ForegroundIdentity foreground{};
    CalibrationSeed calibration_seed{};
    CrosshairContext crosshair{};

    [[nodiscard]] bool operator==(const ScenarioProfile&) const = default;
};

/// @brief Runtime state that a host can observe WITHOUT process injection or
/// memory inspection: OS foreground-window query results, elapsed wall-clock
/// time, and the perception target count. Strings are non-owning views into
/// host-owned buffers so that population stays allocation-free on the hot path.
struct ObservableScenarioState {
    MonotonicNs observed_at_ns{0};         ///< carried for correlation; see the freshness note above
    std::string_view active_process_name{};
    std::string_view active_window_title{};
    bool foreground_confirmed{false};      ///< host's own OS-level "our window is focused" result
    float elapsed_session_seconds{0.0f};   ///< monotonic wall-clock since scenario start
    std::uint32_t visible_target_count{0}; ///< purely observational; from perception output
};

/// @brief Optional benchmark scenario / game adapter (blueprint 6.7).
class IScenarioAdapter {
public:
    virtual ~IScenarioAdapter() = default;

    /// @brief Capabilities currently advertised (all-false until a profile validates).
    [[nodiscard]] virtual ScenarioCapabilities capabilities() const noexcept = 0;

    /// @brief Load and validate a generic profile.
    ///
    /// On any status other than @ref ScenarioProfileStatus::ok the adapter retains no
    /// profile and advertises no capabilities.
    virtual ScenarioProfileStatus load_profile(const ScenarioProfile& profile) noexcept = 0;

    /// @brief Annotate a canonical observation batch with generic context.
    ///
    /// Must remain allocation-free and bounded. The thin implementation only
    /// stamps @c target_value and never touches positions, radii, confidence,
    /// covariance, or timestamps.
    virtual void enrich(bus::TargetObservationBatch& observations,
                        const ObservableScenarioState& state) noexcept = 0;

    /// @brief Current calibration seed, or an invalid seed when unavailable.
    [[nodiscard]] virtual CalibrationSeed calibration_seed() const noexcept = 0;

    /// @brief Current crosshair seed, or an invalid context when unavailable.
    [[nodiscard]] virtual CrosshairContext crosshair_context() const noexcept = 0;

    /// @brief Current authorized foreground identity, or a fail-closed default.
    [[nodiscard]] virtual ForegroundIdentity foreground_identity() const noexcept = 0;

    /// @brief Injection-free foreground authorization decision for @p state.
    ///
    /// Returns true only when ALL of the following hold, in this order:
    ///   1. a validated profile advertises @c provides_foreground_identity;
    ///   2. the host reports @c foreground_confirmed - checked before, and
    ///      independently of, @c require_match, so an unconstrained identity can
    ///      never authorize an unfocused window; and
    ///   3. the identity matches, or is explicitly unconstrained.
    ///
    /// This is an INPUT to the safety gate, never a replacement for it: it can only
    /// withhold authorization, never grant what @c SafetySupervisor refuses.
    ///
    /// @note Consulted per frame; allocation-free.
    [[nodiscard]] virtual bool foreground_authorized(const ObservableScenarioState& state) const noexcept = 0;

    /// @brief Whether the scenario is observably complete (wall-clock only).
    /// Always false unless @c reports_scenario_completion is advertised.
    [[nodiscard]] virtual bool scenario_complete(const ObservableScenarioState& state) const noexcept = 0;

    /// @brief Clear transient counters; keeps the loaded profile.
    virtual void reset() noexcept = 0;
};

/// @brief Compatibility alias. Game semantics remain optional and capability-gated.
using GameAdapter = IScenarioAdapter;

} // namespace aim
