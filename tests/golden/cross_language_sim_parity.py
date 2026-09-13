"""Cross-language parity between the C++ and Python deterministic target simulators.

`include/aim/sim/` and `tools/sim/` are two independent implementations of the same
simulator. Nothing previously compared them, and they had silently diverged: the Python
side evaluated in binary64 where the C++ side evaluates in binary32, so every float field
differed from the first frame onward while both suites stayed green against themselves.

This test drives both implementations over the same presets and seeds and compares raw
IEEE-754 bit patterns. Bit patterns rather than printed decimals are the contract -- a
Python float that merely formats the same is not the same value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import sys
from typing import Dict, List, Tuple

# `tools` is a source tree, not an installed distribution: pytest puts it on the path via
# `pythonpath = .` in pytest.ini, but ctest invokes this script directly, where only the
# script's own directory is on sys.path. Anchor to the repository root so the script works
# from any working directory and under any runner. `python/` goes on the end so that the
# environment's installed `aim` package wins when there is one, and the in-tree copy only
# serves a bare interpreter.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_REPO_ROOT / "python"))

from tools.sim.scenario import ScenarioBuilder  # noqa: E402
from tools.sim.target_simulator import TargetSimulator  # noqa: E402

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")

# (preset, seed, frames). Seeds differ per preset so a seed-plumbing bug cannot hide.
CASES: Tuple[Tuple[str, int, int], ...] = (
    ("gridshot", 0xA11CE, 48),
    ("strafe_track", 0xA11CE, 48),
    ("occlusion", 0xBADF00D, 64),
    ("density", 0x5EED, 32),
)

# Field order must match emit_float() in tests/cpp/test_target_simulator.cpp.
FLOAT_FIELDS: Tuple[str, ...] = (
    "cx", "cy", "nx", "ny",
    "bl", "bt", "br", "bb",
    "r", "conf",
    "cxx", "cxy", "cyy",
    "vx", "vy", "vc", "val",
)


def bits(value: float) -> str:
    """Renders a Python float as its binary32 bit pattern, matching the native emitter."""
    return format(_U32.unpack(_F32.pack(value))[0], "08x")


def build_python_scenario(preset: str, seed: int):
    if preset == "gridshot":
        return ScenarioBuilder.make_gridshot_preset(seed)
    if preset == "strafe_track":
        return ScenarioBuilder.make_strafe_track_preset(seed)
    if preset == "occlusion":
        return ScenarioBuilder.make_occlusion_preset(seed)
    return ScenarioBuilder.make_density_preset(seed)


def run_native(native: Path, preset: str, seed: int, frames: int) -> List[str]:
    result = subprocess.run(
        [str(native), "--dump-trace", preset, hex(seed), str(frames)],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def python_trace(preset: str, seed: int, frames: int) -> List[str]:
    simulator = TargetSimulator(build_python_scenario(preset, seed))
    lines: List[str] = []

    for frame_index in range(frames):
        batch = simulator.step()
        lines.append(
            "batch frame={0} seq={1} frame_id={2} captured={3} published={4} count={5}".format(
                frame_index,
                batch.header.sequence_id,
                batch.frame_id,
                batch.captured_at_ns,
                batch.published_at_ns,
                len(batch.targets),
            )
        )
        for obs in batch.targets:
            values: Dict[str, float] = {
                "cx": obs.center_px.x,
                "cy": obs.center_px.y,
                "nx": obs.center_norm.x,
                "ny": obs.center_norm.y,
                "bl": obs.bbox_px.left,
                "bt": obs.bbox_px.top,
                "br": obs.bbox_px.right,
                "bb": obs.bbox_px.bottom,
                "r": obs.effective_radius_px,
                "conf": obs.confidence,
                "cxx": obs.covariance_px2.xx,
                "cxy": obs.covariance_px2.xy,
                "cyy": obs.covariance_px2.yy,
                "vx": obs.velocity_px_per_s.x,
                "vy": obs.velocity_px_per_s.y,
                "vc": obs.velocity_confidence,
                "val": obs.target_value,
            }
            rendered = " ".join("{0}={1}".format(name, bits(values[name])) for name in FLOAT_FIELDS)
            lines.append(
                "  obs id={0} vis={1} semantic={2} {3}".format(
                    obs.source_id, int(obs.visibility), obs.semantic_id, rendered
                )
            )
    return lines


def compare(preset: str, native_lines: List[str], py_lines: List[str]) -> List[str]:
    failures: List[str] = []
    if len(native_lines) != len(py_lines):
        failures.append(
            "{0}: trace length differs -- native {1} lines, python {2} lines".format(
                preset, len(native_lines), len(py_lines)
            )
        )
    for index, (native, python) in enumerate(zip(native_lines, py_lines)):
        if native != python:
            failures.append(
                "{0}: line {1} differs\n    native: {2}\n    python: {3}".format(
                    preset, index, native, python
                )
            )
            if len(failures) >= 6:
                failures.append("{0}: further differences suppressed".format(preset))
                break
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path, help="aim_sim_tests executable")
    args = parser.parse_args()

    if not args.native.is_file():
        print("native simulator binary not found: {0}".format(args.native), file=sys.stderr)
        return 2

    failures: List[str] = []
    for preset, seed, frames in CASES:
        native_lines = run_native(args.native, preset, seed, frames)
        py_lines = python_trace(preset, seed, frames)
        case_failures = compare(preset, native_lines, py_lines)
        status = "FAIL" if case_failures else "ok"
        print("  [{0}] {1} seed={2} frames={3} ({4} trace lines)".format(
            status, preset, hex(seed), frames, len(native_lines)))
        failures.extend(case_failures)

    if failures:
        print("\nC++/Python simulator parity broken:", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1

    print("  [+] C++ and Python simulators are bit-identical across all presets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
