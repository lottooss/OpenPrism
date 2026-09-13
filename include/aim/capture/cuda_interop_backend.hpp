// include/aim/capture/cuda_interop_backend.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

namespace aim::capture {

using CudaGraphicsResourceHandle = void*;
using CudaArrayHandle = void*;
using CudaSurfaceObjectHandle = std::uint64_t;
using CudaStreamHandle = void*;
using CudaEventHandle = void*;

enum class CudaResult : std::int32_t {
    success = 0,
    error_invalid_value = 1,
    error_out_of_memory = 2,
    error_not_initialized = 3,
    error_deinitialized = 4,
    error_no_device = 100,
    error_invalid_device = 101,
    error_invalid_image = 200,
    error_invalid_context = 201,
    error_map_failed = 205,
    error_unmap_failed = 206,
    error_array_is_mapped = 207,
    error_already_mapped = 208,
    error_no_binary_for_gpu = 209,
    error_already_acquired = 210,
    error_not_mapped = 211,
    error_not_mapped_as_array = 212,
    error_not_mapped_as_pointer = 213,
    error_ecc_uncorrectable = 214,
    error_unsupported_limit = 215,
    error_context_already_in_use = 216,
    error_peer_access_unsupported = 217,
    error_invalid_graphics_context = 219,
    error_adapter_mismatch = 300,
    error_not_ready = 600,
    error_unknown = 999
};

enum class CudaGraphicsRegisterFlags : std::uint32_t {
    none = 0,
    read_only = 1,
    write_discard = 2,
    surface_load_store = 4,
    texture_gather = 8
};

enum class CudaGraphicsMapFlags : std::uint32_t {
    none = 0,
    read_only = 1,
    write_discard = 2
};

enum class CudaStreamFlags : std::uint32_t {
    default_stream = 0,
    non_blocking = 1
};

enum class CudaEventFlags : std::uint32_t {
    default_event = 0,
    blocking_sync = 1,
    disable_timing = 2,
    interprocess = 4
};

/// @brief Abstract interface for D3D11-CUDA interop operations.
class ICudaInteropBackend {
public:
    virtual ~ICudaInteropBackend() noexcept = default;

    /// @brief Checks if CUDA runtime/driver is available on the host system.
    [[nodiscard]] virtual bool is_cuda_available() const noexcept = 0;

    /// @brief Validates if the specified DXGI adapter LUID matches an available CUDA device.
    /// @param dxgi_adapter_luid 64-bit LUID from DXGI adapter.
    /// @param out_cuda_device_id Output CUDA device ID if matched.
    /// @return true if matching CUDA device found, false otherwise.
    [[nodiscard]] virtual bool validate_adapter_match(std::uint64_t dxgi_adapter_luid,
                                                     int& out_cuda_device_id) noexcept = 0;

    /// @brief Sets the active CUDA device.
    virtual CudaResult set_device(int device_id) noexcept = 0;

    /// @brief Registers a D3D11 texture resource for CUDA interoperability.
    /// @param d3d11_texture Pointer to ID3D11Resource / ID3D11Texture2D.
    /// @param flags Registration flags.
    /// @param out_resource Output registered graphics resource handle.
    virtual CudaResult register_d3d11_texture(void* d3d11_texture,
                                             CudaGraphicsRegisterFlags flags,
                                             CudaGraphicsResourceHandle& out_resource) noexcept = 0;

    /// @brief Unregisters a previously registered graphics resource.
    virtual CudaResult unregister_resource(CudaGraphicsResourceHandle resource) noexcept = 0;

    /// @brief Maps graphics resources for access by CUDA on the given stream.
    virtual CudaResult map_resources(CudaGraphicsResourceHandle* resources,
                                    std::size_t count,
                                    CudaStreamHandle stream) noexcept = 0;

    /// @brief Unmaps graphics resources from CUDA access.
    virtual CudaResult unmap_resources(CudaGraphicsResourceHandle* resources,
                                      std::size_t count,
                                      CudaStreamHandle stream) noexcept = 0;

    /// @brief Gets a mapped cudaArray_t from a mapped graphics resource.
    virtual CudaResult get_mapped_array(CudaGraphicsResourceHandle resource,
                                       unsigned int array_index,
                                       unsigned int mip_level,
                                       CudaArrayHandle& out_array) noexcept = 0;

    /// @brief Creates a CUDA surface object from a mapped cudaArray.
    virtual CudaResult create_surface_object(CudaArrayHandle array,
                                            CudaSurfaceObjectHandle& out_surface_object) noexcept = 0;

    /// @brief Destroys a previously created CUDA surface object.
    virtual CudaResult destroy_surface_object(CudaSurfaceObjectHandle surface_object) noexcept = 0;

    // Stream and event management
    virtual CudaResult create_stream(CudaStreamHandle& out_stream,
                                    CudaStreamFlags flags = CudaStreamFlags::non_blocking) noexcept = 0;
    virtual CudaResult destroy_stream(CudaStreamHandle stream) noexcept = 0;

    virtual CudaResult create_event(CudaEventHandle& out_event,
                                   CudaEventFlags flags = CudaEventFlags::disable_timing) noexcept = 0;
    virtual CudaResult destroy_event(CudaEventHandle event) noexcept = 0;

    virtual CudaResult record_event(CudaEventHandle event, CudaStreamHandle stream) noexcept = 0;
    virtual CudaResult stream_wait_event(CudaStreamHandle stream,
                                        CudaEventHandle event,
                                        unsigned int flags = 0) noexcept = 0;
    virtual CudaResult query_event(CudaEventHandle event) noexcept = 0;

    /// @brief Synchronizes a stream (teardown / testing only; prohibited on frame hot path).
    virtual CudaResult synchronize_stream(CudaStreamHandle stream) noexcept = 0;

    /// @brief Returns a descriptive string for a CudaResult.
    [[nodiscard]] virtual const char* get_error_string(CudaResult result) const noexcept = 0;
};

/// @brief Real hardware implementation of ICudaInteropBackend wrapping CUDA Runtime and D3D11 Interop.
class RealCudaInteropBackend final : public ICudaInteropBackend {
public:
    RealCudaInteropBackend() noexcept;
    ~RealCudaInteropBackend() noexcept override;

    RealCudaInteropBackend(const RealCudaInteropBackend&) = delete;
    RealCudaInteropBackend& operator=(const RealCudaInteropBackend&) = delete;

    [[nodiscard]] bool is_cuda_available() const noexcept override;
    [[nodiscard]] bool validate_adapter_match(std::uint64_t dxgi_adapter_luid,
                                             int& out_cuda_device_id) noexcept override;
    CudaResult set_device(int device_id) noexcept override;

    CudaResult register_d3d11_texture(void* d3d11_texture,
                                     CudaGraphicsRegisterFlags flags,
                                     CudaGraphicsResourceHandle& out_resource) noexcept override;
    CudaResult unregister_resource(CudaGraphicsResourceHandle resource) noexcept override;

    CudaResult map_resources(CudaGraphicsResourceHandle* resources,
                            std::size_t count,
                            CudaStreamHandle stream) noexcept override;
    CudaResult unmap_resources(CudaGraphicsResourceHandle* resources,
                              std::size_t count,
                              CudaStreamHandle stream) noexcept override;

    CudaResult get_mapped_array(CudaGraphicsResourceHandle resource,
                               unsigned int array_index,
                               unsigned int mip_level,
                               CudaArrayHandle& out_array) noexcept override;

    CudaResult create_surface_object(CudaArrayHandle array,
                                    CudaSurfaceObjectHandle& out_surface_object) noexcept override;
    CudaResult destroy_surface_object(CudaSurfaceObjectHandle surface_object) noexcept override;

    CudaResult create_stream(CudaStreamHandle& out_stream,
                            CudaStreamFlags flags = CudaStreamFlags::non_blocking) noexcept override;
    CudaResult destroy_stream(CudaStreamHandle stream) noexcept override;

    CudaResult create_event(CudaEventHandle& out_event,
                           CudaEventFlags flags = CudaEventFlags::disable_timing) noexcept override;
    CudaResult destroy_event(CudaEventHandle event) noexcept override;

    CudaResult record_event(CudaEventHandle event, CudaStreamHandle stream) noexcept override;
    CudaResult stream_wait_event(CudaStreamHandle stream,
                                CudaEventHandle event,
                                unsigned int flags = 0) noexcept override;
    CudaResult query_event(CudaEventHandle event) noexcept override;

    CudaResult synchronize_stream(CudaStreamHandle stream) noexcept override;

    [[nodiscard]] const char* get_error_string(CudaResult result) const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Deterministic, scriptable mock CUDA interop backend for CI and fault-injection testing.
class MockCudaInteropBackend final : public ICudaInteropBackend {
public:
    struct MockResourceInfo {
        void* d3d11_texture{nullptr};
        CudaGraphicsRegisterFlags register_flags{CudaGraphicsRegisterFlags::none};
        CudaArrayHandle mapped_array{nullptr};
        CudaSurfaceObjectHandle surface_object{0};
        bool is_mapped{false};
        CudaStreamHandle mapped_stream{nullptr};
    };

    MockCudaInteropBackend() noexcept = default;
    ~MockCudaInteropBackend() noexcept override;

    MockCudaInteropBackend(const MockCudaInteropBackend&) = delete;
    MockCudaInteropBackend& operator=(const MockCudaInteropBackend&) = delete;

    // Scriptable fault injection
    void set_cuda_available(bool available) noexcept { cuda_available_ = available; }
    void set_adapter_match(bool match, int device_id = 0) noexcept {
        adapter_match_ = match;
        mock_device_id_ = device_id;
    }
    void set_expected_adapter_luid(std::uint64_t expected_luid) noexcept {
        expected_adapter_luid_ = expected_luid;
    }
    void set_register_result(CudaResult res) noexcept { register_result_ = res; }
    void set_map_result(CudaResult res) noexcept { map_result_ = res; }
    void set_unmap_result(CudaResult res) noexcept { unmap_result_ = res; }
    void set_get_array_result(CudaResult res) noexcept { get_array_result_ = res; }
    void set_create_surface_result(CudaResult res) noexcept { create_surface_result_ = res; }
    void set_destroy_surface_result(CudaResult res) noexcept { destroy_surface_result_ = res; }
    void queue_event_query_result(CudaResult res) noexcept { queued_event_query_results_.push(res); }

    void queue_map_result(CudaResult res) noexcept { queued_map_results_.push(res); }

    // Metrics & Inspection
    [[nodiscard]] std::size_t register_count() const noexcept { return register_count_; }
    [[nodiscard]] std::size_t unregister_count() const noexcept { return unregister_count_; }
    [[nodiscard]] std::size_t map_count() const noexcept { return map_count_; }
    [[nodiscard]] std::size_t unmap_count() const noexcept { return unmap_count_; }
    [[nodiscard]] std::size_t get_array_count() const noexcept { return get_array_count_; }
    [[nodiscard]] std::size_t create_surface_count() const noexcept { return create_surface_count_; }
    [[nodiscard]] std::size_t destroy_surface_count() const noexcept { return destroy_surface_count_; }
    [[nodiscard]] std::size_t stream_create_count() const noexcept { return stream_create_count_; }
    [[nodiscard]] std::size_t stream_destroy_count() const noexcept { return stream_destroy_count_; }
    [[nodiscard]] std::size_t event_create_count() const noexcept { return event_create_count_; }
    [[nodiscard]] std::size_t event_destroy_count() const noexcept { return event_destroy_count_; }
    [[nodiscard]] std::size_t event_record_count() const noexcept { return event_record_count_; }
    [[nodiscard]] std::size_t stream_wait_event_count() const noexcept { return stream_wait_event_count_; }
    [[nodiscard]] std::size_t active_resources_count() const noexcept { return resources_.size(); }
    [[nodiscard]] std::size_t active_surfaces_count() const noexcept { return active_surfaces_; }
    [[nodiscard]] std::size_t active_streams_count() const noexcept { return active_streams_; }
    [[nodiscard]] std::size_t active_events_count() const noexcept { return active_events_; }
    [[nodiscard]] bool is_resource_mapped(CudaGraphicsResourceHandle res) const noexcept;

    // ICudaInteropBackend interface
    [[nodiscard]] bool is_cuda_available() const noexcept override { return cuda_available_; }
    [[nodiscard]] bool validate_adapter_match(std::uint64_t dxgi_adapter_luid,
                                             int& out_cuda_device_id) noexcept override;
    CudaResult set_device(int device_id) noexcept override;

    CudaResult register_d3d11_texture(void* d3d11_texture,
                                     CudaGraphicsRegisterFlags flags,
                                     CudaGraphicsResourceHandle& out_resource) noexcept override;
    CudaResult unregister_resource(CudaGraphicsResourceHandle resource) noexcept override;

    CudaResult map_resources(CudaGraphicsResourceHandle* resources,
                            std::size_t count,
                            CudaStreamHandle stream) noexcept override;
    CudaResult unmap_resources(CudaGraphicsResourceHandle* resources,
                              std::size_t count,
                              CudaStreamHandle stream) noexcept override;

    CudaResult get_mapped_array(CudaGraphicsResourceHandle resource,
                               unsigned int array_index,
                               unsigned int mip_level,
                               CudaArrayHandle& out_array) noexcept override;

    CudaResult create_surface_object(CudaArrayHandle array,
                                    CudaSurfaceObjectHandle& out_surface_object) noexcept override;
    CudaResult destroy_surface_object(CudaSurfaceObjectHandle surface_object) noexcept override;

    CudaResult create_stream(CudaStreamHandle& out_stream,
                            CudaStreamFlags flags = CudaStreamFlags::non_blocking) noexcept override;
    CudaResult destroy_stream(CudaStreamHandle stream) noexcept override;

    CudaResult create_event(CudaEventHandle& out_event,
                           CudaEventFlags flags = CudaEventFlags::disable_timing) noexcept override;
    CudaResult destroy_event(CudaEventHandle event) noexcept override;

    CudaResult record_event(CudaEventHandle event, CudaStreamHandle stream) noexcept override;
    CudaResult stream_wait_event(CudaStreamHandle stream,
                                CudaEventHandle event,
                                unsigned int flags = 0) noexcept override;
    CudaResult query_event(CudaEventHandle event) noexcept override;

    CudaResult synchronize_stream(CudaStreamHandle stream) noexcept override;

    [[nodiscard]] const char* get_error_string(CudaResult result) const noexcept override;

private:
    bool cuda_available_{true};
    bool adapter_match_{true};
    std::uint64_t expected_adapter_luid_{0};
    int mock_device_id_{0};
    int current_device_id_{0};

    CudaResult register_result_{CudaResult::success};
    CudaResult map_result_{CudaResult::success};
    CudaResult unmap_result_{CudaResult::success};
    CudaResult get_array_result_{CudaResult::success};
    CudaResult create_surface_result_{CudaResult::success};
    CudaResult destroy_surface_result_{CudaResult::success};

    std::queue<CudaResult> queued_map_results_{};
    std::queue<CudaResult> queued_event_query_results_{};

    std::size_t register_count_{0};
    std::size_t unregister_count_{0};
    std::size_t map_count_{0};
    std::size_t unmap_count_{0};
    std::size_t get_array_count_{0};
    std::size_t create_surface_count_{0};
    std::size_t destroy_surface_count_{0};
    std::size_t stream_create_count_{0};
    std::size_t stream_destroy_count_{0};
    std::size_t event_create_count_{0};
    std::size_t event_destroy_count_{0};
    std::size_t event_record_count_{0};
    std::size_t stream_wait_event_count_{0};

    std::size_t active_surfaces_{0};
    std::size_t active_streams_{0};
    std::size_t active_events_{0};

    std::uint64_t resource_handle_seq_{0xCAFE0000ULL};
    std::uint64_t array_handle_seq_{0xBA5E0000ULL};
    std::uint64_t surface_handle_seq_{0x50FACE00ULL};
    std::uint64_t stream_handle_seq_{0x578EA000ULL};
    std::uint64_t event_handle_seq_{0xEFE00000ULL};

    std::unordered_map<CudaGraphicsResourceHandle, MockResourceInfo> resources_{};
};

} // namespace aim::capture
