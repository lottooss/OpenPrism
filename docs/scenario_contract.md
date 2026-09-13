# Scenario adapter contract v1

## Purpose

The scenario adapter is the **only** place a benchmark domain may influence the reusable core, and it may
do so through exactly four channels: a generic per-target value, a crosshair seed, a calibration seed,
and an authorized foreground identity. Everything else about a domain is invisible to the system.

Tracking, aim policy, trajectory planning, calibration, actuation, and safety compile and run without
including `aim/interfaces/scenario_adapter.hpp`. The adapter is optional; a system with no adapter at all
behaves exactly like one holding a profile that advertises nothing.

Defined by `include/aim/interfaces/scenario_adapter.hpp`. See the [architecture overview](ARCHITECTURE.md) for pipeline context.

## A domain is a profile, not a class

There is one implementation, `aim::scenario::ScenarioAdapter`. A domain is a `ScenarioProfile` value,
authored as YAML under `configs/scenario/` and validated by
`schemas/config/scenario_profile.schema.json`. No C++ type names a game.

| Profile | Advertises | Intended use |
|---|---|---|
| `configs/scenario/generic.yaml` | uniform target value; explicitly unconstrained identity | offline replay, portability runs (M8) |
| `configs/scenario/aimlabs.yaml` | target value, crosshair, calibration seed, identity, 60 s window | the M7 supervised live benchmark |

## Capabilities gate everything

Every flag in `ScenarioCapabilities` defaults to `false`. A consumer must treat an un-advertised
capability as unavailable and fall back to its own default — never assume the adapter filled a value in.

| Capability | Accessor | Value when not advertised |
|---|---|---|
| `provides_target_value` | `enrich()` | observations are left untouched |
| `provides_crosshair_context` | `crosshair_context()` | `valid == false` |
| `provides_calibration_seed` | `calibration_seed()` | `valid == false` |
| `provides_foreground_identity` | `foreground_identity()`, `foreground_authorized()` | unconfirmable identity; authorization always `false` |
| `reports_scenario_completion` | `scenario_complete()` | always `false` |

A profile may not advertise a capability it cannot honour; `load_profile()` returns
`ScenarioProfileStatus::capability_without_data` and the adapter retains nothing.

## Authorization is injection-free

The adapter never enumerates processes, opens process handles, or reads another process' memory. The
**host** performs the ordinary OS foreground-window query and passes what it observed back in:

```cpp
aim::ObservableScenarioState state{};
state.observed_at_ns        = clock.now_ns();
state.active_process_name   = host.foreground_process_name(); // host-owned buffer
state.active_window_title   = host.foreground_window_title(); // host-owned buffer
state.foreground_confirmed  = host.our_window_has_focus();
state.elapsed_session_seconds = host.elapsed_since_scenario_start();
state.visible_target_count  = observations.target_count;
```

`foreground_authorized(state)` then requires **all** of the following, in this order:

1. a validated profile advertises `provides_foreground_identity`;
2. the host reports `foreground_confirmed`; and
3. the identity matches, or is explicitly unconstrained.

Step 2 is checked **before** and **independently of** `require_match`, so an unconstrained identity can
never authorize an unfocused window. Any doubt returns `false`.

`require_match: false` is the one explicit escape hatch, for offline replay and portability. It relaxes
*which application* may be targeted — never *whether our window is focused*. It must never be used to
drive live actuation against a real application.

A window title alone is **not** an authorization boundary: any application can set its own title to
contain an arbitrary substring. `validate()` therefore rejects a profile that sets `require_match: true`
without a `process_name`.

## Freshness and safety are not the adapter's job

The adapter enforces **no** staleness deadline. `aim::safety::SafetySupervisor` (M6-05) owns the single
freshness policy for the system — a hard 10 ms cutoff on every actuation command — and duplicating it here
would create a competing source of truth. `observed_at_ns` is carried for correlation and telemetry.

Likewise, `foreground_authorized()` is an **input** to the safety gate, never a replacement for it. The
adapter can only withhold authorization; it can never grant what `SafetySupervisor` refuses. Where
`configs/domain/aimlabs.yaml` and a scenario profile both describe the foreground identity, the safety
configuration wins; a test asserts the two agree.

Note that the supervisor's focus gate is itself conditional — `src/safety/safety_supervisor.cpp` gates on
`config_.require_foreground && focus_check_`, so a host that installs no focus callback gets no focus
enforcement from it. Do not treat the supervisor as an automatic backstop for a permissive profile.

## Example producer and consumer

```cpp
#include "aim/scenario/scenario_adapter.hpp"
#include "aim/scenario/scenario_profile_loader.hpp"

using namespace aim;

// --- Producer: resolve a profile once, at startup (cold path; may allocate). ---
const auto loaded = scenario::ScenarioProfileLoader::load_from_file("configs/scenario/aimlabs.yaml");
if (!loaded.success) {
    for (const std::string& e : loaded.errors) { log_error(e); }
    return false;  // fail closed: run with no adapter rather than a half-valid one
}
record_in_run_manifest("scenario_profile_sha256", loaded.sha256_hash);

scenario::ScenarioAdapter adapter{};
if (adapter.load_profile(loaded.profile) != ScenarioProfileStatus::ok) {
    log_error(to_string(adapter.last_status()));
    return false;
}

// Seed the calibrator only if the profile actually offers a seed.
if (const CalibrationSeed seed = adapter.calibration_seed(); seed.valid) {
    calibration.seed_from(seed.counts_per_pixel_x, seed.counts_per_pixel_y);
}

// --- Consumer: per frame (hot path; allocation-free). ---
IScenarioAdapter& scenario_ctx = adapter;   // the core only ever sees the interface

scenario_ctx.enrich(observations, state);   // stamps target_value, nothing else

if (!scenario_ctx.foreground_authorized(state)) {
    supervisor.trigger_emergency_stop(SafetyReason::lost_focus);
    return;                                  // no command is produced
}
if (scenario_ctx.scenario_complete(state)) {
    session.stop();
    return;
}
// ... tracking -> policy -> trajectory -> supervisor.check_actuation_safety() -> actuator
```

Swapping domains is a profile load, not a code change:

```cpp
adapter.load_profile(scenario::make_generic_profile());  // portability / replay
```

## Failure behavior

- An invalid profile is rejected whole. The adapter resets to a default-constructed state, advertises no
  capabilities, and authorizes nothing. The reason survives in `last_status()`.
- A rejected profile never leaves the previous domain's data readable through `profile()`.
- The YAML loader rejects unknown keys, wrong types, non-mapping roots, and out-of-range values, and
  returns a fail-closed profile plus an empty hash on any failure.
- `enrich()` clamps to `bus::kMaxObservations` even if a caller supplies a corrupted `target_count`.

## Verification

```powershell
cmake --build --preset windows-msvc-debug --target aim_scenario_adapter_tests
ctest --preset windows-msvc-debug -R aim_scenario_adapter_test --output-on-failure
uv run --project python pytest tests/test_scenario_adapter.py
```

`ScenarioAdapter::validate()` and `schemas/config/scenario_profile.schema.json` are two independent
validators over the same data. The accepted ranges live once, in `aim::scenario_limits`; the schema
mirrors them, and `tests/test_scenario_adapter.py` parses the header and fails if they ever drift.
