# Third-Party Runtime and Generated-Code Inputs

| Component | Pinned version | License | Use |
|---|---:|---|---|
| [Google FlatBuffers](https://github.com/google/flatbuffers) | 25.12.19 (`7e163021e59cca4f8e1e35a7c828b5c6b7915953`) | [Apache-2.0](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE) | Canonical bus runtime, compiler, and generated C++/Python bindings |

Dependency resolution is pinned by `vcpkg.json` and `python/uv.lock`. Generated files identify themselves as compiler output and are reproducible with `scripts/generate-bus-bindings.ps1`.

The full [Apache 2.0 license](Apache-2.0.txt) accompanies the generated bindings. Copyright and generated-code notices must be preserved. The root noncommercial license applies to project-owned code and does not override third-party rights.

## Dependencies obtained separately

| Component | License / upstream terms | Role |
|---|---|---|
| [nlohmann/json](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT) | MIT | Native JSON handling |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp/blob/master/LICENSE) | MIT | Native YAML handling |
| [GoogleTest](https://github.com/google/googletest/blob/main/LICENSE) | BSD-3-Clause | C++ testing |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause and bundled component notices | Offline arrays |
| [PyYAML](https://github.com/yaml/pyyaml/blob/main/LICENSE) | MIT | Python YAML handling |
| [jsonschema](https://github.com/python-jsonschema/jsonschema/blob/main/COPYING) | MIT | Schema validation |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style and third-party notices | Optional training |
| [TorchVision](https://github.com/pytorch/vision/blob/main/LICENSE) | BSD-3-Clause | Optional vision tooling |
| [ONNX](https://github.com/onnx/onnx/blob/main/LICENSE) | Apache-2.0 | Optional model interchange |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) | MIT | Optional parity evaluation |
| [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | AGPL-3.0 or separately obtained commercial terms | Optional reference detector tooling |

These packages are resolved during setup; their source distributions, binaries, weights and SDKs are not bundled in this repository. Installed wheels may include further components and notices; preserve those when redistributing any package or binary build. Development tools and their transitive packages are recorded in `python/uv.lock` and retain their upstream terms.

CUDA and TensorRT are separately installed NVIDIA SDKs under NVIDIA's applicable agreements. A locally built engine and its source model need their own verified license and provenance. Ultralytics-derived model artifacts must not be relabeled with OpenPrism's license. Dataset and external plugin licenses must likewise reflect their actual source rights.

The demo depicts Aimlabs, whose game content and trademarks belong to their respective owners. The project source license does not grant rights to that third-party content.
