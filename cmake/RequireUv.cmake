# Executed as a ctest command when `uv` is unavailable at configure time.
#
# The cross-language golden tests need uv to resolve the locked Python environment. Skipping
# them silently made ctest report "100% tests passed" over a smaller suite than intended, so
# a missing toolchain looked identical to a passing one. Failing loudly keeps the count honest.
message(FATAL_ERROR
    "uv was not found at configure time, so this cross-language golden test could not run. "
    "Install uv 0.12.5 and re-configure, or exclude this test deliberately with "
    "`ctest -E cross_language`.")
