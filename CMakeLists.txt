cmake_minimum_required(VERSION 3.25)
project(OpenPrism VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

include(FetchContent)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# ------------------------------------------------------------------------------
# Dependencies (vcpkg / FetchContent fallback)
# ------------------------------------------------------------------------------
find_package(nlohmann_json CONFIG QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json AND NOT TARGET nlohmann_json)
    FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(json)
endif()

find_package(yaml-cpp CONFIG QUIET)
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
    FetchContent_Declare(
        yaml_cpp
        URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml_cpp)
endif()

find_package(flatbuffers CONFIG QUIET)
if(NOT TARGET flatbuffers::flatbuffers AND NOT TARGET FlatBuffers::FlatBuffers AND NOT TARGET flatbuffers)
    set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATLIB OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
    set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        flatbuffers
        URL https://github.com/google/flatbuffers/archive/7e163021e59cca4f8e1e35a7c828b5c6b7915953.tar.gz
        URL_HASH SHA256=4236c5d22309abeac2384d3800febcc8d60423e8cb2f573abfa8f798352c4536
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(flatbuffers)
endif()

if(TARGET flatbuffers::flatbuffers)
    set(AIM_FLATBUFFERS_TARGET flatbuffers::flatbuffers)
elseif(TARGET FlatBuffers::FlatBuffers)
    set(AIM_FLATBUFFERS_TARGET FlatBuffers::FlatBuffers)
elseif(TARGET flatbuffers)
    set(AIM_FLATBUFFERS_TARGET flatbuffers)
else()
    message(FATAL_ERROR "FlatBuffers headers target is unavailable")
endif()

# ------------------------------------------------------------------------------
# Core Interface Library (Header-Only Core)
# ------------------------------------------------------------------------------
add_library(aim_core INTERFACE)
target_include_directories(aim_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(aim_core INTERFACE nlohmann_json::nlohmann_json)
endif()

if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(aim_core INTERFACE yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    target_link_libraries(aim_core INTERFACE yaml-cpp)
endif()

target_compile_features(aim_core INTERFACE cxx_std_20)

add_library(aim_bus_schema INTERFACE)
target_include_directories(aim_bus_schema SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/generated/flatbuffers/cpp>
)
target_link_libraries(aim_bus_schema INTERFACE aim_core ${AIM_FLATBUFFERS_TARGET})
target_compile_features(aim_bus_schema INTERFACE cxx_std_20)

# ------------------------------------------------------------------------------
# C++ Unit Tests
# ------------------------------------------------------------------------------
enable_testing()

add_executable(aim_tests
    tests/cpp/test_core_contracts.cpp
)

target_link_libraries(aim_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_tests)

add_test(NAME aim_core_contracts_test COMMAND aim_tests)

add_executable(aim_config_tests
    tests/cpp/test_config_manifest.cpp
)

target_link_libraries(aim_config_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_config_tests)

add_test(NAME aim_config_manifest_test COMMAND aim_config_tests)

add_executable(aim_bus_tests
    tests/cpp/test_bus_ring.cpp
)

target_link_libraries(aim_bus_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_bus_tests)

add_test(NAME aim_bus_ring_test COMMAND aim_bus_tests)

add_executable(aim_bus_schema_tests
    tests/cpp/test_flatbuffers_schemas.cpp
)

target_link_libraries(aim_bus_schema_tests PRIVATE aim_bus_schema)
aim_set_compiler_warnings(aim_bus_schema_tests)

add_test(NAME aim_bus_schema_native_test COMMAND aim_bus_schema_tests --self-test)

find_program(AIM_UV_EXECUTABLE NAMES uv)
if(NOT AIM_UV_EXECUTABLE)
    message(WARNING
        "uv was not found; cross-language golden tests will fail loudly rather than vanish "
        "from the ctest suite. Install uv 0.12.5 to run them.")
endif()

# Registers a golden test that shells out to the locked Python environment. When uv is
# missing the test is still registered, but fails with an explanatory message -- a missing
# prerequisite must not be indistinguishable from a green run.
function(aim_add_cross_language_test test_name)
    if(AIM_UV_EXECUTABLE)
        add_test(
            NAME ${test_name}
            COMMAND
                ${AIM_UV_EXECUTABLE} run --frozen --project ${CMAKE_CURRENT_SOURCE_DIR}/python
                python ${ARGN}
        )
    else()
        add_test(
            NAME ${test_name}
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RequireUv.cmake
        )
    endif()
    set_tests_properties(${test_name} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

aim_add_cross_language_test(aim_bus_schema_cross_language_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/cross_language_flatbuffers.py
    --native $<TARGET_FILE:aim_bus_schema_tests>
    --workspace ${CMAKE_CURRENT_BINARY_DIR}/golden-flatbuffers
)

add_executable(aim_sim_tests
    tests/cpp/test_target_simulator.cpp
)

target_link_libraries(aim_sim_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_sim_tests)

add_test(NAME aim_target_simulator_test COMMAND aim_sim_tests)

# include/aim/sim/ and tools/sim/ are two independent implementations of the same simulator.
# Each suite previously only tested its own side, which let them diverge silently.
aim_add_cross_language_test(aim_sim_cross_language_parity_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/cross_language_sim_parity.py
    --native $<TARGET_FILE:aim_sim_tests>
)

add_executable(aim_m1_pipeline_tests
    tests/cpp/test_m1_image_free_pipeline.cpp
)

target_link_libraries(aim_m1_pipeline_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_m1_pipeline_tests)

add_test(NAME aim_m1_image_free_pipeline_test COMMAND aim_m1_pipeline_tests)

# ------------------------------------------------------------------------------
# Mock Dynamic Plugin DLLs
# ------------------------------------------------------------------------------
add_library(aim_mock_perception_plugin MODULE
    tests/cpp/plugins/mock_perception_plugin.cpp
)
target_link_libraries(aim_mock_perception_plugin PRIVATE aim_core)
aim_set_compiler_warnings(aim_mock_perception_plugin)
target_compile_definitions(aim_mock_perception_plugin PRIVATE AIM_PLUGIN_EXPORTS)

add_library(aim_mock_policy_plugin MODULE
    tests/cpp/plugins/mock_policy_plugin.cpp
)
target_link_libraries(aim_mock_policy_plugin PRIVATE aim_core)
aim_set_compiler_warnings(aim_mock_policy_plugin)
target_compile_definitions(aim_mock_policy_plugin PRIVATE AIM_PLUGIN_EXPORTS)

add_library(aim_mock_incompatible_plugin MODULE
    tests/cpp/plugins/mock_incompatible_plugin.cpp
)
target_link_libraries(aim_mock_incompatible_plugin PRIVATE aim_core)
aim_set_compiler_warnings(aim_mock_incompatible_plugin)
target_compile_definitions(aim_mock_incompatible_plugin PRIVATE AIM_PLUGIN_EXPORTS)

add_library(aim_mock_malformed_plugin MODULE
    tests/cpp/plugins/mock_malformed_plugin.cpp
)
target_link_libraries(aim_mock_malformed_plugin PRIVATE aim_core)
aim_set_compiler_warnings(aim_mock_malformed_plugin)
target_compile_definitions(aim_mock_malformed_plugin PRIVATE AIM_PLUGIN_EXPORTS)

# ------------------------------------------------------------------------------
# Plugin ABI Test Harness Executable
# ------------------------------------------------------------------------------
add_executable(aim_plugin_abi_tests
    tests/cpp/test_plugin_abi.cpp
)
target_link_libraries(aim_plugin_abi_tests PRIVATE aim_core)
aim_set_compiler_warnings(aim_plugin_abi_tests)
add_dependencies(aim_plugin_abi_tests
    aim_mock_perception_plugin
    aim_mock_policy_plugin
    aim_mock_incompatible_plugin
    aim_mock_malformed_plugin
)

add_test(NAME aim_plugin_abi_test COMMAND aim_plugin_abi_tests)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Capture Subsystem Library & Tests (Milestone M2-01 & M2-02)
# ------------------------------------------------------------------------------
add_library(aim_capture STATIC
    src/capture/dxgi_frame_source.cpp
    src/capture/gpu_surface_pool.cpp
    src/capture/dxgi_backend.cpp
    src/capture/wgc_backend.cpp
    src/capture/wgc_frame_source.cpp
    src/capture/capture_recovery_state_machine.cpp
    src/capture/unified_capture_source.cpp
    src/capture/cuda_interop_backend.cpp
    src/capture/cuda_surface_pool.cpp
)

target_include_directories(aim_capture PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aim_capture PUBLIC
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_capture PUBLIC
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
    target_compile_definitions(aim_capture PUBLIC _SILENCE_EXPERIMENTAL_COROUTINE_WARNING)
endif()

aim_set_compiler_warnings(aim_capture)
target_compile_features(aim_capture PUBLIC cxx_std_20)

add_executable(aim_capture_tests
    tests/cpp/test_dxgi_frame_source.cpp
)

target_link_libraries(aim_capture_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_capture_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_capture_tests)

add_test(NAME aim_dxgi_capture_test COMMAND aim_capture_tests)

add_executable(aim_surface_pool_tests
    tests/cpp/test_gpu_surface_pool.cpp
)

target_link_libraries(aim_surface_pool_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_surface_pool_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_surface_pool_tests)

add_test(NAME aim_gpu_surface_pool_test COMMAND aim_surface_pool_tests)

add_executable(aim_capture_adversarial_tests
    tests/cpp/test_dxgi_adversarial_stress.cpp
)

target_link_libraries(aim_capture_adversarial_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_capture_adversarial_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_capture_adversarial_tests)

add_test(NAME aim_capture_adversarial_test COMMAND aim_capture_adversarial_tests)

add_executable(aim_challenger1_stress_tests
    tests/cpp/test_challenger1_stress.cpp
)

target_link_libraries(aim_challenger1_stress_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_challenger1_stress_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_challenger1_stress_tests)

add_test(NAME aim_challenger1_stress_test COMMAND aim_challenger1_stress_tests)

add_executable(aim_wgc_capture_tests
    tests/cpp/test_wgc_frame_source.cpp
)

target_link_libraries(aim_wgc_capture_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_wgc_capture_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_wgc_capture_tests)

add_test(NAME aim_wgc_capture_test COMMAND aim_wgc_capture_tests)

add_executable(aim_capture_recovery_tests
    tests/cpp/test_capture_recovery.cpp
)

target_link_libraries(aim_capture_recovery_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_capture_recovery_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_capture_recovery_tests)

add_test(NAME aim_capture_recovery_test COMMAND aim_capture_recovery_tests)

add_executable(aim_wgc_recovery_adversarial_tests
    tests/cpp/test_wgc_recovery_adversarial_stress.cpp
)

target_link_libraries(aim_wgc_recovery_adversarial_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_wgc_recovery_adversarial_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_wgc_recovery_adversarial_tests)

add_test(NAME aim_wgc_recovery_adversarial_test COMMAND aim_wgc_recovery_adversarial_tests)

add_executable(aim_challenger2_stress_tests
    tests/cpp/test_challenger2_stress.cpp
)

target_link_libraries(aim_challenger2_stress_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_challenger2_stress_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_challenger2_stress_tests)

add_test(NAME aim_challenger2_stress_test COMMAND aim_challenger2_stress_tests)

add_executable(aim_cuda_interop_tests
    tests/cpp/test_cuda_interop.cpp
)

target_link_libraries(aim_cuda_interop_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_cuda_interop_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_cuda_interop_tests)

add_test(NAME aim_cuda_interop_test COMMAND aim_cuda_interop_tests)

add_executable(aim_cuda_surface_pool_tests
    tests/cpp/test_cuda_surface_pool.cpp
)

target_link_libraries(aim_cuda_surface_pool_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_cuda_surface_pool_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_cuda_surface_pool_tests)

add_test(NAME aim_cuda_surface_pool_test COMMAND aim_cuda_surface_pool_tests)

add_executable(aim_cuda_interop_challenger_stress_tests
    tests/cpp/test_cuda_interop_challenger_stress.cpp
)

target_link_libraries(aim_cuda_interop_challenger_stress_tests PRIVATE
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_cuda_interop_challenger_stress_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_cuda_interop_challenger_stress_tests)

add_test(NAME aim_cuda_interop_challenger_stress_test COMMAND aim_cuda_interop_challenger_stress_tests)

# ------------------------------------------------------------------------------
# Perception Preprocessing Subsystem (Milestone M2-04)
# ------------------------------------------------------------------------------
option(AIM_ENABLE_CUDA_PREPROCESS "Build the real CUDA fused preprocessing kernel when nvcc is available" ON)
set(AIM_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures for the fused preprocessing kernel")
set(AIM_HAS_CUDA_PREPROCESS OFF)

if(AIM_ENABLE_CUDA_PREPROCESS)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(AIM_HAS_CUDA_PREPROCESS ON)
            add_library(aim_cuda_preprocess_kernel OBJECT
                src/perception/fused_kernel_cuda.cu
            )
            target_include_directories(aim_cuda_preprocess_kernel PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
            )
            target_link_libraries(aim_cuda_preprocess_kernel PRIVATE CUDA::cudart)
            set_target_properties(aim_cuda_preprocess_kernel PROPERTIES
                CUDA_STANDARD 20
                CUDA_STANDARD_REQUIRED ON
                CUDA_EXTENSIONS OFF
                CUDA_ARCHITECTURES ${AIM_CUDA_ARCHITECTURES}
                POSITION_INDEPENDENT_CODE ON
            )
        endif()
    endif()
endif()

add_library(aim_perception STATIC
    src/perception/capture_to_tensor_pipeline.cpp
    src/perception/cpu_preprocessor.cpp
    src/perception/engine_artifact.cpp
    src/perception/fused_kernel.cpp
    src/perception/gpu_preprocessor.cpp
    src/perception/mock_perception_engine.cpp
    src/perception/tensorrt_engine_runner.cpp
    src/perception/yolo_decoder.cpp
)

if(AIM_HAS_CUDA_PREPROCESS)
    target_sources(aim_perception PRIVATE $<TARGET_OBJECTS:aim_cuda_preprocess_kernel>)
    target_link_libraries(aim_perception PRIVATE CUDA::cudart)
    target_compile_definitions(aim_perception PRIVATE AIM_HAS_CUDA_PREPROCESS=1)
else()
    target_compile_definitions(aim_perception PRIVATE AIM_HAS_CUDA_PREPROCESS=0)
    message(STATUS "CUDA compiler/toolkit unavailable: production GPU preprocessing will fail closed")
endif()

target_include_directories(aim_perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aim_perception PUBLIC
    aim_core
    aim_capture
    aim_bus_schema
)

aim_set_compiler_warnings(aim_perception)
target_compile_features(aim_perception PUBLIC cxx_std_20)
set(AIM_TENSORRT_ROOT "" CACHE PATH "Trusted TensorRT 10 SDK installation")
set(AIM_HAS_TENSORRT_BACKEND OFF)
if(AIM_TENSORRT_ROOT)
    find_path(AIM_TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS "${AIM_TENSORRT_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(AIM_TENSORRT_LIBRARY NAMES nvinfer_10 nvinfer
        PATHS "${AIM_TENSORRT_ROOT}/lib" "${AIM_TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(AIM_HAS_TENSORRT_BACKEND ON)
    target_include_directories(aim_perception SYSTEM PRIVATE ${AIM_TENSORRT_INCLUDE_DIR})
    target_link_libraries(aim_perception PRIVATE ${AIM_TENSORRT_LIBRARY} CUDA::cudart)
endif()
if(AIM_HAS_TENSORRT_BACKEND)
    add_executable(aim_engine_validation src/apps/aim_engine_validation.cpp)
    target_include_directories(aim_engine_validation SYSTEM PRIVATE ${AIM_TENSORRT_INCLUDE_DIR})
    target_link_libraries(aim_engine_validation PRIVATE
        aim_perception ${AIM_TENSORRT_LIBRARY} CUDA::cudart nlohmann_json::nlohmann_json)
    aim_set_compiler_warnings(aim_engine_validation)
endif()

target_compile_definitions(aim_perception PRIVATE
    AIM_HAS_TENSORRT_BACKEND=$<BOOL:${AIM_HAS_TENSORRT_BACKEND}>)
if(NOT AIM_HAS_TENSORRT_BACKEND)
    message(STATUS "TensorRT SDK not configured: inference will fail closed (set AIM_TENSORRT_ROOT)")
endif()

if(AIM_HAS_CUDA_PREPROCESS)
    add_executable(aim_cuda_preprocess_hardware_validation
        tests/cpp/test_fused_gpu_preprocessing_cuda.cu
    )
    target_link_libraries(aim_cuda_preprocess_hardware_validation PRIVATE
        aim_perception
        CUDA::cudart
    )
    set_target_properties(aim_cuda_preprocess_hardware_validation PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_ARCHITECTURES ${AIM_CUDA_ARCHITECTURES}
    )

    if(WIN32)
        add_executable(aim_capture_to_tensor_hardware_validation
            tests/cpp/test_capture_to_tensor_hardware_validation.cpp
        )
        target_link_libraries(aim_capture_to_tensor_hardware_validation PRIVATE
            aim_perception
            aim_capture
            aim_core
            aim_bus_schema
            CUDA::cudart
            d3d11
            dxgi
            dxguid
            windowsapp
            dwmapi
        )
        aim_set_compiler_warnings(aim_capture_to_tensor_hardware_validation)
    endif()
endif()

add_executable(aim_preprocess_tests
    tests/cpp/test_fused_gpu_preprocessing.cpp
)

target_link_libraries(aim_preprocess_tests PRIVATE
    aim_perception
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_preprocess_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_preprocess_tests)

add_test(NAME aim_fused_gpu_preprocessing_test COMMAND aim_preprocess_tests)

# ------------------------------------------------------------------------------
# Coordinate Mapping & Subpixel Round-Trip Test Suite (Milestone M2-05)
# ------------------------------------------------------------------------------
add_executable(aim_coordinate_mapping_tests
    tests/cpp/test_coordinate_mapping.cpp
)

target_link_libraries(aim_coordinate_mapping_tests PRIVATE
    aim_perception
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_coordinate_mapping_tests)

add_test(NAME aim_coordinate_mapping_test COMMAND aim_coordinate_mapping_tests)

# ------------------------------------------------------------------------------
# Capture-to-Tensor Benchmark & Acceptance Suite (Milestone M2-06 Exit Gate)
# ------------------------------------------------------------------------------
add_executable(aim_capture_to_tensor_benchmark_tests
    tests/cpp/test_capture_to_tensor_benchmark.cpp
)

target_link_libraries(aim_capture_to_tensor_benchmark_tests PRIVATE
    aim_perception
    aim_capture
    aim_core
    aim_bus_schema
)

if(WIN32)
    target_link_libraries(aim_capture_to_tensor_benchmark_tests PRIVATE
        d3d11
        dxgi
        dxguid
        windowsapp
        dwmapi
    )
endif()

aim_set_compiler_warnings(aim_capture_to_tensor_benchmark_tests)

add_test(NAME aim_capture_to_tensor_benchmark_test COMMAND aim_capture_to_tensor_benchmark_tests)

# ------------------------------------------------------------------------------
# Data Subsystem Library & Recording Tap (Milestone M3-01)
# ------------------------------------------------------------------------------
add_library(aim_data STATIC
    src/data/recording_tap.cpp
)

target_include_directories(aim_data PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aim_data PUBLIC
    aim_core
    aim_perception
)

aim_set_compiler_warnings(aim_data)

# ------------------------------------------------------------------------------
# Recording Tap & Dataset Pipeline Tests (Milestone M3-01)
# ------------------------------------------------------------------------------
add_executable(aim_recording_tap_tests
    tests/cpp/test_recording_tap.cpp
)

target_link_libraries(aim_recording_tap_tests PRIVATE
    aim_data
    aim_perception
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_recording_tap_tests)

add_test(NAME aim_recording_tap_test COMMAND aim_recording_tap_tests)

# ------------------------------------------------------------------------------
# TensorRT Perception Runner Tests (Milestone M3-06)
# ------------------------------------------------------------------------------
add_executable(aim_tensorrt_runner_tests
    tests/cpp/test_tensorrt_engine_runner.cpp
)

target_link_libraries(aim_tensorrt_runner_tests PRIVATE
    aim_perception
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_tensorrt_runner_tests)

add_test(NAME aim_tensorrt_runner_test COMMAND aim_tensorrt_runner_tests)

# ------------------------------------------------------------------------------
# YOLO Decoder Tests (Milestone M3-07)
# ------------------------------------------------------------------------------
add_executable(aim_yolo_decoder_tests
    tests/cpp/test_yolo_decoder.cpp
)

target_link_libraries(aim_yolo_decoder_tests PRIVATE
    aim_perception
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_yolo_decoder_tests)

add_test(NAME aim_yolo_decoder_test COMMAND aim_yolo_decoder_tests)

# ------------------------------------------------------------------------------
# Tracking Subsystem Library & Kalman Filter Tests (Milestone M4-01 & M4-02 & M4-03)
# ------------------------------------------------------------------------------
add_library(aim_tracking STATIC
    src/tracking/kalman_models.cpp
    src/tracking/gated_hungarian.cpp
    src/tracking/track_lifecycle.cpp
)

target_include_directories(aim_tracking PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(aim_tracking PUBLIC
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_tracking)

add_executable(aim_kalman_models_tests
    tests/cpp/test_kalman_models.cpp
)

target_link_libraries(aim_kalman_models_tests PRIVATE
    aim_tracking
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_kalman_models_tests)

add_test(NAME aim_kalman_models_test COMMAND aim_kalman_models_tests)

# ------------------------------------------------------------------------------
# Gated Hungarian Multi-Target Association Tests (Milestone M4-02)
# ------------------------------------------------------------------------------
add_executable(aim_gated_hungarian_tests
    tests/cpp/test_gated_hungarian.cpp
)

target_link_libraries(aim_gated_hungarian_tests PRIVATE
    aim_tracking
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_gated_hungarian_tests)

add_test(NAME aim_gated_hungarian_test COMMAND aim_gated_hungarian_tests)

# ------------------------------------------------------------------------------
# Track Lifecycle & Management Tests (Milestone M4-03)
# ------------------------------------------------------------------------------
add_executable(aim_track_lifecycle_tests
    tests/cpp/test_track_lifecycle.cpp
)

target_link_libraries(aim_track_lifecycle_tests PRIVATE
    aim_tracking
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_track_lifecycle_tests)

add_test(NAME aim_track_lifecycle_test COMMAND aim_track_lifecycle_tests)

# ------------------------------------------------------------------------------
# Latency Estimator & Prediction Extrapolation Tests (Milestone M4-04)
# ------------------------------------------------------------------------------
add_executable(aim_prediction_extrapolation_tests
    tests/cpp/test_latency_estimator_and_extrapolation.cpp
)

target_link_libraries(aim_prediction_extrapolation_tests PRIVATE
    aim_tracking
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_prediction_extrapolation_tests)

add_test(NAME aim_prediction_extrapolation_test COMMAND aim_prediction_extrapolation_tests)

# ------------------------------------------------------------------------------
# Tracker & Prediction Replay Acceptance Benchmark (Milestone M4-05)
# ------------------------------------------------------------------------------
add_executable(aim_tracker_prediction_benchmark_tests
    tests/cpp/test_tracker_prediction_benchmark.cpp
)

target_link_libraries(aim_tracker_prediction_benchmark_tests PRIVATE
    aim_tracking
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_tracker_prediction_benchmark_tests)

add_test(NAME aim_tracker_prediction_benchmark_test COMMAND aim_tracker_prediction_benchmark_tests)

# ------------------------------------------------------------------------------
# Policy Subsystem Library & Tests (Milestone M5-01)
# ------------------------------------------------------------------------------
add_library(aim_policy STATIC
    src/policy/utility_policy.cpp
)

target_link_libraries(aim_policy PUBLIC
    aim_core
    aim_bus_schema
)

target_include_directories(aim_policy PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_policy)

add_executable(aim_utility_policy_tests
    tests/cpp/test_utility_aim_policy.cpp
)

target_link_libraries(aim_utility_policy_tests PRIVATE
    aim_policy
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_utility_policy_tests)

add_test(NAME aim_utility_policy_test COMMAND aim_utility_policy_tests)

# ------------------------------------------------------------------------------
# Trajectory Planner Subsystem Library & Tests (Milestone M5-02)
# ------------------------------------------------------------------------------
add_library(aim_trajectory STATIC
    src/trajectory/trajectory_planner.cpp
)

target_link_libraries(aim_trajectory PUBLIC
    aim_core
    aim_bus_schema
)

target_include_directories(aim_trajectory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_trajectory)

add_executable(aim_trajectory_planner_tests
    tests/cpp/test_trajectory_planner.cpp
)

target_link_libraries(aim_trajectory_planner_tests PRIVATE
    aim_trajectory
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_trajectory_planner_tests)

add_test(NAME aim_trajectory_planner_test COMMAND aim_trajectory_planner_tests)

# ------------------------------------------------------------------------------
# Policy & Trajectory Benchmark (Milestone M5-04)
# ------------------------------------------------------------------------------
add_executable(aim_policy_trajectory_benchmark_tests
    tests/cpp/test_policy_trajectory_benchmark.cpp
)

target_link_libraries(aim_policy_trajectory_benchmark_tests PRIVATE
    aim_policy
    aim_trajectory
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_policy_trajectory_benchmark_tests)

add_test(NAME aim_policy_trajectory_benchmark_test COMMAND aim_policy_trajectory_benchmark_tests)

# ------------------------------------------------------------------------------
# SendInput Actuation & Scheduler Subsystem (Milestones M6-01, M6-02)
# ------------------------------------------------------------------------------
add_library(aim_actuation STATIC
    src/actuation/sendinput_actuator.cpp
    src/actuation/usb_hid_actuator.cpp
    src/actuation/actuator_scheduler.cpp
    src/safety/safety_supervisor.cpp
)

target_link_libraries(aim_actuation PUBLIC
    aim_core
    aim_bus_schema
    aim_trajectory
)
if(WIN32)
    target_link_libraries(aim_actuation PRIVATE winmm)
endif()

target_include_directories(aim_actuation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_actuation)

add_executable(aim_sendinput_actuator_tests
    tests/cpp/test_sendinput_actuator.cpp
)

target_link_libraries(aim_sendinput_actuator_tests PRIVATE
    aim_actuation
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_sendinput_actuator_tests)

add_test(NAME aim_sendinput_actuator_test COMMAND aim_sendinput_actuator_tests)

add_executable(aim_usb_hid_actuator_tests
    tests/cpp/test_usb_hid_actuator.cpp
)

target_link_libraries(aim_usb_hid_actuator_tests PRIVATE
    aim_actuation
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_usb_hid_actuator_tests)

add_test(NAME aim_usb_hid_actuator_test COMMAND aim_usb_hid_actuator_tests)

add_executable(aim_actuator_scheduler_tests
    tests/cpp/test_actuator_scheduler.cpp
)

target_link_libraries(aim_actuator_scheduler_tests PRIVATE
    aim_actuation
    aim_trajectory
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_actuator_scheduler_tests)

add_test(NAME aim_actuator_scheduler_test COMMAND aim_actuator_scheduler_tests)

add_executable(aim_safety_supervisor_tests
    tests/cpp/test_safety_supervisor.cpp
)

target_link_libraries(aim_safety_supervisor_tests PRIVATE
    aim_actuation
    aim_core
)

aim_set_compiler_warnings(aim_safety_supervisor_tests)

add_test(NAME aim_safety_supervisor_test COMMAND aim_safety_supervisor_tests)

# ------------------------------------------------------------------------------
# Calibration Subsystem (Milestone M6-03)
# ------------------------------------------------------------------------------
add_library(aim_calibration STATIC
    src/calibration/calibration_model.cpp
    src/calibration/control_calibration.cpp
    src/runtime/visual_calibration.cpp
    src/runtime/visual_calibration_session.cpp
)

target_link_libraries(aim_calibration PUBLIC
    aim_core
)

target_include_directories(aim_calibration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_calibration)

add_executable(aim_calibration_model_tests
    tests/cpp/test_calibration_model.cpp
)

target_link_libraries(aim_calibration_model_tests PRIVATE
    aim_calibration
    aim_core
)

aim_set_compiler_warnings(aim_calibration_model_tests)

add_test(NAME aim_calibration_model_test COMMAND aim_calibration_model_tests)

add_executable(aim_visual_calibration_tests tests/cpp/test_visual_calibration.cpp)
target_link_libraries(aim_visual_calibration_tests PRIVATE aim_calibration)
aim_set_compiler_warnings(aim_visual_calibration_tests)
add_test(NAME aim_visual_calibration_test COMMAND aim_visual_calibration_tests)

add_executable(aim_visual_calibration_session_tests tests/cpp/test_visual_calibration_session.cpp)
target_link_libraries(aim_visual_calibration_session_tests PRIVATE aim_calibration aim_bus_schema)
aim_set_compiler_warnings(aim_visual_calibration_session_tests)
add_test(NAME aim_visual_calibration_session_test COMMAND aim_visual_calibration_session_tests)

add_executable(aim_control_calibration_tests
    tests/cpp/test_control_calibration.cpp
)

target_link_libraries(aim_control_calibration_tests PRIVATE
    aim_calibration
    aim_core
)

aim_set_compiler_warnings(aim_control_calibration_tests)

add_test(NAME aim_control_calibration_test COMMAND aim_control_calibration_tests)

# ------------------------------------------------------------------------------
# Actuation & Safety Soak Benchmark (Milestone M6-04)
# ------------------------------------------------------------------------------
add_executable(aim_actuation_safety_soak_tests
    tests/cpp/test_actuation_safety_soak.cpp
)

target_link_libraries(aim_actuation_safety_soak_tests PRIVATE
    aim_actuation
    aim_calibration
    aim_trajectory
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_actuation_safety_soak_tests)

add_test(NAME aim_actuation_safety_soak_test COMMAND aim_actuation_safety_soak_tests)

# ------------------------------------------------------------------------------
# Scenario Adapter Subsystem (Milestone M7-01)
# ------------------------------------------------------------------------------
add_library(aim_scenario STATIC
    src/scenario/scenario_adapter.cpp
)

target_link_libraries(aim_scenario PUBLIC
    aim_core
)

target_include_directories(aim_scenario PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_scenario)

add_executable(aim_scenario_adapter_tests
    tests/cpp/test_scenario_adapter.cpp
)

target_link_libraries(aim_scenario_adapter_tests PRIVATE
    aim_scenario
    aim_core
)

# Lets the suite load the shipped configs/scenario/*.yaml profiles regardless of the
# working directory ctest happens to run from.
target_compile_definitions(aim_scenario_adapter_tests PRIVATE
    AIM_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

aim_set_compiler_warnings(aim_scenario_adapter_tests)

add_test(NAME aim_scenario_adapter_test COMMAND aim_scenario_adapter_tests)

# ------------------------------------------------------------------------------
# Closed-Loop Integration & Shot Gating (Milestones M7-02, M7-03, M7-04)
# ------------------------------------------------------------------------------
add_library(aim_pipeline STATIC
    src/policy/shot_gate.cpp
    src/pipeline/closed_loop_pipeline.cpp
)

target_link_libraries(aim_pipeline PUBLIC
    aim_tracking
    aim_policy
    aim_trajectory
    aim_actuation
    aim_calibration
    aim_scenario
    aim_core
    aim_bus_schema
)

target_include_directories(aim_pipeline PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_pipeline)

# M7-02 Closed-Loop Pipeline Tests
add_executable(aim_closed_loop_pipeline_tests
    tests/cpp/test_closed_loop_pipeline.cpp
)

target_link_libraries(aim_closed_loop_pipeline_tests PRIVATE
    aim_pipeline
    aim_core
)

aim_set_compiler_warnings(aim_closed_loop_pipeline_tests)

add_test(NAME aim_closed_loop_pipeline_test COMMAND aim_closed_loop_pipeline_tests)

# M7-03 End-to-End Latency & Score Benchmark Tests
add_executable(aim_end_to_end_benchmark_tests
    tests/cpp/test_end_to_end_benchmark.cpp
)

target_link_libraries(aim_end_to_end_benchmark_tests PRIVATE
    aim_pipeline
    aim_core
)

aim_set_compiler_warnings(aim_end_to_end_benchmark_tests)

add_test(NAME aim_end_to_end_benchmark_test COMMAND aim_end_to_end_benchmark_tests)

# M7-04 30-Minute Thermal Soak & Hardening Tests
add_executable(aim_thermal_soak_hardening_tests
    tests/cpp/test_thermal_soak_hardening.cpp
)

target_link_libraries(aim_thermal_soak_hardening_tests PRIVATE
    aim_pipeline
    aim_core
)

aim_set_compiler_warnings(aim_thermal_soak_hardening_tests)


add_test(NAME aim_thermal_soak_hardening_test COMMAND aim_thermal_soak_hardening_tests)

add_executable(aim_firing_regression_tests tests/cpp/test_firing_regressions.cpp)
target_link_libraries(aim_firing_regression_tests PRIVATE aim_pipeline)
aim_set_compiler_warnings(aim_firing_regression_tests)
add_test(NAME aim_firing_regression_test COMMAND aim_firing_regression_tests)

add_executable(aim_runtime_assets_tests tests/cpp/test_runtime_assets.cpp)
target_link_libraries(aim_runtime_assets_tests PRIVATE aim_calibration aim_core)
aim_set_compiler_warnings(aim_runtime_assets_tests)
add_test(NAME aim_runtime_assets_test COMMAND aim_runtime_assets_tests)

# Experimental observation-only diagnostic. Never linked to a physical input path.
if(WIN32)
add_executable(aim_runtime_candidate src/apps/aim_runtime_candidate.cpp)
target_link_libraries(aim_runtime_candidate PRIVATE aim_pipeline aim_perception)
aim_set_compiler_warnings(aim_runtime_candidate)
add_test(NAME aim_runtime_candidate_help COMMAND aim_runtime_candidate --help)
add_test(NAME aim_runtime_candidate_missing_assets COMMAND aim_runtime_candidate --actuate)
set_tests_properties(aim_runtime_candidate_missing_assets PROPERTIES WILL_FAIL TRUE)
add_test(NAME aim_runtime_candidate_preflight
    COMMAND ${CMAKE_COMMAND}
        "-DCANDIDATE=$<TARGET_FILE:aim_runtime_candidate>"
        "-DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DFIXTURE_ROOT=${CMAKE_CURRENT_BINARY_DIR}/runtime_candidate_fixtures"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/runtime_candidate_preflight.cmake")

add_executable(aim_live_runner
    src/apps/aim_live_runner.cpp
)

target_link_libraries(aim_live_runner PRIVATE
    aim_pipeline
    aim_core
)

aim_set_compiler_warnings(aim_live_runner)
add_test(NAME aim_live_runner_help COMMAND aim_live_runner --help)
add_test(NAME aim_live_runner_reject_actuation COMMAND aim_live_runner --actuate)
set_tests_properties(aim_live_runner_reject_actuation PROPERTIES WILL_FAIL TRUE)
endif()

# ------------------------------------------------------------------------------
# External Observation Source Subsystem (Milestone M8-01)
# ------------------------------------------------------------------------------
add_library(aim_bus_shm STATIC
    src/bus/external_observation_source.cpp
)

target_link_libraries(aim_bus_shm PUBLIC
    aim_core
    aim_bus_schema
)

target_include_directories(aim_bus_shm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_bus_shm)

add_executable(aim_external_observation_source_tests
    tests/cpp/test_external_observation_source.cpp
)

target_link_libraries(aim_external_observation_source_tests PRIVATE
    aim_bus_shm
    aim_core
    aim_bus_schema
)

aim_set_compiler_warnings(aim_external_observation_source_tests)

add_test(NAME aim_external_observation_source_test COMMAND aim_external_observation_source_tests)

# ------------------------------------------------------------------------------
# Second-Domain Perception Subsystem (Milestone M8-02)
# ------------------------------------------------------------------------------
add_library(aim_second_domain_perception STATIC
    src/perception/second_domain_perception.cpp
)

target_link_libraries(aim_second_domain_perception PUBLIC
    aim_core
    aim_bus_schema
)

target_include_directories(aim_second_domain_perception PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

aim_set_compiler_warnings(aim_second_domain_perception)

add_executable(aim_second_domain_integration_tests
    tests/cpp/test_second_domain_integration.cpp
)

target_link_libraries(aim_second_domain_integration_tests PRIVATE
    aim_second_domain_perception
    aim_bus_shm
    aim_pipeline
    aim_scenario
    aim_core
    aim_bus_schema
)

target_compile_definitions(aim_second_domain_integration_tests PRIVATE
    AIM_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

aim_set_compiler_warnings(aim_second_domain_integration_tests)

add_test(NAME aim_second_domain_integration_test COMMAND aim_second_domain_integration_tests)

# ------------------------------------------------------------------------------
# Portability Conformance Suite (Milestone M8-03)
# ------------------------------------------------------------------------------
add_executable(aim_portability_conformance_tests
    tests/cpp/test_portability_conformance.cpp
)

target_link_libraries(aim_portability_conformance_tests PRIVATE
    aim_second_domain_perception
    aim_bus_shm
    aim_pipeline
    aim_scenario
    aim_core
    aim_bus_schema
)

target_compile_definitions(aim_portability_conformance_tests PRIVATE
    AIM_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

aim_set_compiler_warnings(aim_portability_conformance_tests)

add_test(NAME aim_portability_conformance_test COMMAND aim_portability_conformance_tests)
