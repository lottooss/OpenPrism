# Native plugin contract v1

## Purpose

The native plugin boundary allows perception, tracking, policy, trajectory, actuator, and scenario modules to be replaced without exposing C++ ownership or compiler-specific types. `include/aim/plugin/plugin_abi.h` is the only dynamic-library ABI. The C++ interfaces under `include/aim/interfaces/` are internal conveniences and are not binary-stable plugin contracts.

## Load sequence

1. Load and strictly validate the plugin manifest against `schemas/manifest/plugin_manifest.schema.json`.
2. Verify the binary byte size and SHA-256 before loading the library. A mismatch fails closed; the binary must not be executed.
3. Resolve `aim_plugin_abi_version` and reject an incompatible major version.
4. Resolve `aim_plugin_create_v1` and request the v1 API table.
5. Validate the returned ABI version, plugin kind, opaque instance, every required callback, and descriptor before `start()`.
6. Compare manifest kind and input/output FourCC contracts with the descriptor.
7. Only then call `start()` and `process()`.

`PluginContractValidator` implements this sequence for Python tooling. The native loader validates the dynamic API and descriptor; production native integration must perform manifest hash/size verification before calling it.

## Ownership and lifetime

- `AimPluginHandle` is opaque. Only the plugin creates and destroys it.
- Host service callbacks and their byte inputs are borrowed for the duration of a call. Plugins must not retain or free them.
- `process()` input and output buffers are host-owned and borrowed only for that invocation.
- A plugin writes no more than `output_capacity`. It reports the required size through `out_output_size` when returning `AIM_STATUS_ERROR_BUFFER_TOO_SMALL`.
- Descriptor strings remain plugin-owned until `destroy()`. The host copies any text needed after unload and never frees those pointers.
- C++ objects, STL containers, exceptions, allocators, and GPU-resource ownership never cross the C ABI.

## Serialized message contract

`process()` buffers are opaque serialized messages identified by the manifest and descriptor FourCC. Production modules use the versioned FlatBuffers schemas in `schemas/bus/`; they do not reinterpret buffers as native C++ structs. Every consumer validates the message file identifier, schema range, length, and FlatBuffers verifier result before field access.

The mock DLLs intentionally use small opaque byte fixtures with identifiers at the FlatBuffers file-identifier offset. They exercise ABI ownership, lifecycle, negotiation, and bounds without pretending to replace the C++/Python golden FlatBuffers tests.

## Failure behavior

- Missing exports, malformed API tables, ABI mismatches, descriptor mismatches, wrong plugin kinds, invalid pointers, and corrupted sizes fail before processing.
- Unknown status codes are treated as corrupted plugin behavior.
- The host resets output size before each call and rejects a successful response claiming more bytes than capacity.
- Native plugins must catch their own language exceptions. Host wrappers additionally contain C++ exceptions and Windows structured exceptions where supported; no plugin fault is allowed to unwind through application code.
- Stop/destroy are idempotent from the host wrapper’s perspective, and library unload happens only after plugin destruction.

## Verification

The native harness builds reference perception and policy plugins plus incompatible and malformed fixtures. It checks negotiation, descriptor validation, lifecycle, wrong-state calls, buffer truncation, invalid pointers, opaque message chaining, zero warmed allocations, and repeated reloads. Python tests cover manifest validation, hash/size fail-closed behavior, callback-table bounds, exception containment, and 1,000 lifecycle cycles.
