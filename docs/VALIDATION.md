# Publication validation — 13 September 2026

The source snapshot was built locally with MSVC 14.44, CUDA 12.6 and TensorRT 10.8 on Windows x64. The local portable toolchain reused an existing dependency installation; the configure command supplied `VCPKG_MANIFEST_INSTALL=OFF`, `VCPKG_INSTALLED_DIR`, `CMAKE_MAKE_PROGRAM` and `AIM_TENSORRT_ROOT` explicitly. Normal registered Visual Studio installations use the README presets.

| Check | Result |
|---|---|
| `cmake --build --preset windows-msvc-release` | Passed |
| `ctest --preset windows-msvc-release --output-on-failure` | 57/57 passed, 109.86 seconds |
| `uv lock --check --project python` | Passed |
| `uv sync --frozen --project python` | Passed in a separate base environment |
| `uv run --frozen --project python pytest` | 324/324 passed, 25.45 seconds |
| `uv run --frozen --project python ruff check python tools tests` | Passed |
| `uv run --frozen --project python mypy python/aim tools` | Passed, 60 source files |
| `uv run --frozen --project python python -m compileall -q python/aim tools tests` | Passed |
| Local Markdown paths and staged whitespace | Passed |

Canonical bus schemas retain their prior definitions; only project-name comments changed. Generated bindings are unchanged. The model-manifest license enumeration additively accepts PolyForm Noncommercial.

After replacing example host identifiers and unmeasured environment defaults, the affected 32 Python tests and native configuration test passed again, and the Release build remained successful.

These results are local validation, not a claim that hosted CI has completed. Deterministic test benchmarks are not substitutes for concurrent end-to-end measurements. See the [recorded test](../media/README.md) for separate physical-run evidence and the README for remaining performance and model-validation limitations.
