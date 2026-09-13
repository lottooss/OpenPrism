"""tools/sim/prng.py
Deterministic 64-bit pseudo-random number generator (SplitMix64 + Box-Muller).
Guarantees cross-platform bit-exact simulation replay.

Every method suffixed ``_f32`` mirrors a C++ member returning ``float``, so it rounds to
IEEE-754 binary32 at each step the C++ expression does. Python floats are binary64; without
that rounding this module silently diverged from ``include/aim/sim/prng.hpp`` from the first
uniform sample onward. ``tests/golden/cross_language_sim_parity.py`` pins the two together.
"""

from __future__ import annotations

import math
import struct
from typing import Tuple

_F32 = struct.Struct("<f")


def f32(value: float) -> float:
    """Rounds a Python binary64 float to the nearest IEEE-754 binary32 value."""
    return float(_F32.unpack(_F32.pack(value))[0])


class DeterministicRng:
    """SplitMix64 PRNG matching C++ implementation with Box-Muller Gaussian sampling."""

    def __init__(self, seed: int = 0x853C49E6748FEA9B) -> None:
        self._state: int = 0x853C49E6748FEA9B if seed == 0 else (seed & 0xFFFFFFFFFFFFFFFF)
        self._cached_gaussian: float = 0.0
        self._has_cached_gaussian: bool = False

    def reseed(self, seed: int) -> None:
        """Reseeds the PRNG state."""
        self._state = 0x853C49E6748FEA9B if seed == 0 else (seed & 0xFFFFFFFFFFFFFFFF)
        self._has_cached_gaussian = False
        self._cached_gaussian = 0.0

    def next_u64(self) -> int:
        """Generates next pseudo-random 64-bit unsigned integer using SplitMix64."""
        self._state = (self._state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def next_uniform_f64(self) -> float:
        """Generates uniform float64 in [0.0, 1.0)."""
        return float(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def next_uniform_f32(self) -> float:
        """Generates uniform float32 in [0.0, 1.0)."""
        return f32(self.next_uniform_f64())

    def next_range_f32(self, min_val: float, max_val: float) -> float:
        """Generates uniform float in [min_val, max_val)."""
        low = f32(min_val)
        span = f32(f32(max_val) - low)
        return f32(low + f32(self.next_uniform_f32() * span))

    def next_gaussian_pair(self, mean: float = 0.0, stddev: float = 1.0) -> Tuple[float, float]:
        """Generates a pair of independent Gaussian samples N(mean, stddev^2) using Box-Muller."""
        u1 = self.next_uniform_f64()
        if u1 < 1e-15:
            u1 = 1e-15
        u2 = self.next_uniform_f64()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * 3.14159265358979323846 * u2
        z0 = f32(r * math.cos(theta))
        z1 = f32(r * math.sin(theta))
        centre, spread = f32(mean), f32(stddev)
        return (f32(centre + f32(z0 * spread)), f32(centre + f32(z1 * spread)))

    def next_gaussian(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        """Generates a single Gaussian sample, caching the secondary sample."""
        centre, spread = f32(mean), f32(stddev)
        if self._has_cached_gaussian:
            self._has_cached_gaussian = False
            return f32(centre + f32(self._cached_gaussian * spread))
        g0, g1 = self.next_gaussian_pair(0.0, 1.0)
        self._cached_gaussian = g1
        self._has_cached_gaussian = True
        return f32(centre + f32(g0 * spread))
