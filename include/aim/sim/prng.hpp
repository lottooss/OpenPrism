// include/aim/sim/prng.hpp
// Deterministic 64-bit pseudo-random number generator for 100% bit-exact simulation replay
#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

namespace aim::sim {

/// @brief Deterministic 64-bit SplitMix64 PRNG with exact Box-Muller Gaussian sampling.
///
/// The integer stream is pure 64-bit arithmetic and is bit-exact everywhere. The float
/// stream is bit-exact across MSVC, GCC, and Clang because Box-Muller is evaluated in
/// binary64 and narrowed to binary32 exactly once: the single-precision libm entry points
/// are not correctly rounded and disagree between C runtimes, so calling them directly
/// would make the sequence depend on which libm the binary links against. See
/// aim::sim::sin_f32 in "aim/sim/types.hpp" for the same reasoning applied to kinematics.
///
/// tools/sim/prng.py mirrors this class and is held to it by
/// tests/golden/cross_language_sim_parity.py.
class DeterministicRng {
public:
    constexpr explicit DeterministicRng(std::uint64_t seed = 0x853c49e6748fea9bULL) noexcept
        : state_(seed == 0 ? 0x853c49e6748fea9bULL : seed) {}

    /// @brief Resets the PRNG state with a new seed.
    constexpr void reseed(std::uint64_t seed) noexcept {
        state_ = (seed == 0 ? 0x853c49e6748fea9bULL : seed);
        has_cached_gaussian_ = false;
        cached_gaussian_ = 0.0f;
    }

    /// @brief Generates next pseudo-random 64-bit integer using SplitMix64.
    constexpr std::uint64_t next_u64() noexcept {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    /// @brief Generates uniform floating point in [0.0, 1.0).
    constexpr double next_uniform_f64() noexcept {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    /// @brief Generates uniform float in [0.0f, 1.0f).
    constexpr float next_uniform_f32() noexcept {
        return static_cast<float>(next_uniform_f64());
    }

    /// @brief Generates uniform float in specified range [min_val, max_val).
    constexpr float next_range_f32(float min_val, float max_val) noexcept {
        return min_val + next_uniform_f32() * (max_val - min_val);
    }

    /// @brief Generates a pair of independent Gaussian samples N(mean, stddev^2) using Box-Muller.
    std::pair<float, float> next_gaussian_pair(float mean = 0.0f, float stddev = 1.0f) noexcept {
        double u1 = next_uniform_f64();
        if (u1 < 1e-15) u1 = 1e-15; // Prevent log(0)
        const double u2 = next_uniform_f64();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * 3.14159265358979323846 * u2;
        const float z0 = static_cast<float>(r * std::cos(theta));
        const float z1 = static_cast<float>(r * std::sin(theta));
        return { mean + z0 * stddev, mean + z1 * stddev };
    }

    /// @brief Generates a single Gaussian sample N(mean, stddev^2), caching the secondary sample.
    float next_gaussian(float mean = 0.0f, float stddev = 1.0f) noexcept {
        if (has_cached_gaussian_) {
            has_cached_gaussian_ = false;
            return mean + cached_gaussian_ * stddev;
        }
        const auto [g0, g1] = next_gaussian_pair(0.0f, 1.0f);
        cached_gaussian_ = g1;
        has_cached_gaussian_ = true;
        return mean + g0 * stddev;
    }

    [[nodiscard]] constexpr std::uint64_t state() const noexcept {
        return state_;
    }

private:
    std::uint64_t state_{0x853c49e6748fea9bULL};
    float cached_gaussian_{0.0f};
    bool has_cached_gaussian_{false};
};

} // namespace aim::sim
