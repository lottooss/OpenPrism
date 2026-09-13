# OpenPrism architecture

## Data flow and timing

The Windows x64 C++20 runtime captures a full 1920×1080 field of view through DXGI, with WGC fallback. GPU preprocessing produces a fixed 640×384 model tensor. A replaceable perception backend converts detections into canonical observations rather than exposing detector-specific tensors to the controller.

Each hot-path message carries monotonic time and sequence identity. Coordinates have explicit conventions; pixels support local actuation while normalized coordinates support portability. Letterbox transformations must round-trip consistently. Bounded latest-message-wins transport discards superseded work instead of building a queue.

The internal capture-arrival-to-dispatch target is 6 ms p99. Source age above 10 ms is rejected; the native runner uses an 8 ms pre-dispatch budget to reserve time for operating-system checks. Prediction to the later measured command-effect time does not extend the source freshness deadline.

## Independent modules

- Capture supplies image surfaces, or an external producer supplies canonical observations.
- `IPerceptionEngine` owns perception; TensorRT is a deployment backend, not a tracking dependency.
- `ITrackingEngine` maintains adaptive Kalman state with gated Hungarian association and uncertainty-aware lifecycle.
- Prediction extrapolates tracked motion to command-effect time.
- `IAimPolicy` ranks candidate targets. The default utility policy is separate from perception and optional learned-policy experiments.
- `ITrajectoryPlanner` combines direct small-error motion with bounded trajectories for larger errors and terminal correction.
- Calibration estimates the relationship between visual error and actuator counts.
- The latest-plan scheduler dispatches at a nominal 1,000 Hz through guarded actuation. SendInput is the default; NullActuator supports deterministic tests.

Native convenience interfaces remain C++. Dynamic plugin boundaries use opaque state, host-owned buffers and a versioned C ABI; exceptions and STL ownership must not cross them. Existing `aim` namespaces and executable identifiers remain stable despite the OpenPrism project name.

The Python distribution name `aim-agent-tooling`, existing `AimAgent` schema titles and `aimlabs.research` schema identifiers are also retained for compatibility. Schema identifiers are contract names, not hosted service dependencies.

## Safety and resource ownership

Freshness, valid calibration, focus, command bounds, target uncertainty, capture health and emergency-stop checks gate physical input. Losing the authorized foreground window prevents input. Shutdown releases held buttons. Allocation, disk, network, recording, training and UI work stay outside the warmed hot path.

Implementations must preserve message timestamps, coordinate contracts, bounded capacity and resource lifetime. A replacement model requires input/output compatibility, manifest verification and parity evaluation. A new domain also needs suitable observations, calibration and scenario constraints; portability is not automatic.

## Source map

| Location | Purpose |
|---|---|
| `include/aim/` | Native module interfaces and shared types |
| `src/` | C++ implementations and runtime applications |
| `schemas/` | Bus, configuration and manifest contracts |
| `tools/` | Offline data, training, evaluation and benchmark tooling |
| `tests/` | C++ and Python validation |
| `configs/` | Scenario and runtime configuration |
| `scripts/` | Build helpers, packaging and guarded launcher |

See [bus](bus_contract.md), [plugin](plugin_contract.md) and [scenario](scenario_contract.md) contracts for integration details.
