// tests/cpp/test_cuda_interop.cpp
// Unit and Fault-Injection Test Suite for ICudaInteropBackend & MockCudaInteropBackend

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#if defined(_WIN32)
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

#include "aim/capture/cuda_interop_backend.hpp"

namespace aim::capture {
inline std::ostream& operator<<(std::ostream& os, CudaResult r) {
    return os << static_cast<std::int32_t>(r);
}
} // namespace aim::capture

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion FAILED: " #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        std::cerr << "Assertion FAILED: !" #cond " at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::cerr << "Assertion FAILED: " #a " == " #b " (" << (a) << " != " << (b) << ") at " __FILE__ ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

namespace {

// =============================================================================
// Test 1: Adapter Matching and Device Selection
// =============================================================================
int test_adapter_matching_and_device_selection() {
    std::cout << "  [Test 1] Adapter Matching & Device Selection..." << std::endl;

    aim::capture::MockCudaInteropBackend backend;
    ASSERT_TRUE(backend.is_cuda_available());

    int dev_id = -1;
    ASSERT_TRUE(backend.validate_adapter_match(0x00010688ULL, dev_id));
    ASSERT_EQ(dev_id, 0);

    ASSERT_EQ(backend.set_device(dev_id), aim::capture::CudaResult::success);

    // Test adapter mismatch rejection
    backend.set_adapter_match(false);
    int mismatch_dev = -1;
    ASSERT_FALSE(backend.validate_adapter_match(0x00010688ULL, mismatch_dev));
    ASSERT_EQ(mismatch_dev, -1);

    // Test CUDA unavailable
    backend.set_cuda_available(false);
    ASSERT_FALSE(backend.is_cuda_available());
    ASSERT_FALSE(backend.validate_adapter_match(0x00010688ULL, dev_id));
    ASSERT_EQ(backend.set_device(0), aim::capture::CudaResult::error_no_device);

    return 0;
}

// =============================================================================
// Test 2: Texture Registration and Unregistration
// =============================================================================
int test_texture_registration_and_unregistration() {
    std::cout << "  [Test 2] Texture Registration & Unregistration..." << std::endl;

    aim::capture::MockCudaInteropBackend backend;

    void* mock_tex1 = reinterpret_cast<void*>(0x11110000ULL);
    void* mock_tex2 = reinterpret_cast<void*>(0x22220000ULL);

    aim::capture::CudaGraphicsResourceHandle res1 = nullptr;
    aim::capture::CudaGraphicsResourceHandle res2 = nullptr;

    ASSERT_EQ(backend.register_d3d11_texture(mock_tex1, aim::capture::CudaGraphicsRegisterFlags::none, res1),
              aim::capture::CudaResult::success);
    ASSERT_TRUE(res1 != nullptr);
    ASSERT_EQ(backend.active_resources_count(), 1u);

    ASSERT_EQ(backend.register_d3d11_texture(mock_tex2, aim::capture::CudaGraphicsRegisterFlags::surface_load_store, res2),
              aim::capture::CudaResult::success);
    ASSERT_TRUE(res2 != nullptr);
    ASSERT_TRUE(res1 != res2);
    ASSERT_EQ(backend.active_resources_count(), 2u);

    // Unregister resource 1
    ASSERT_EQ(backend.unregister_resource(res1), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.active_resources_count(), 1u);

    // Double unregister should fail
    ASSERT_EQ(backend.unregister_resource(res1), aim::capture::CudaResult::error_invalid_value);

    // Unregister resource 2
    ASSERT_EQ(backend.unregister_resource(res2), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.active_resources_count(), 0u);

    ASSERT_EQ(backend.register_count(), 2u);
    ASSERT_EQ(backend.unregister_count(), 3u);

    return 0;
}

// =============================================================================
// Test 3: Resource Mapping and Surface Object Lifecycle
// =============================================================================
int test_mapping_and_surface_object_lifecycle() {
    std::cout << "  [Test 3] Resource Mapping & Surface Object Lifecycle..." << std::endl;

    aim::capture::MockCudaInteropBackend backend;

    void* mock_tex = reinterpret_cast<void*>(0x33330000ULL);
    aim::capture::CudaGraphicsResourceHandle res = nullptr;
    ASSERT_EQ(backend.register_d3d11_texture(mock_tex, aim::capture::CudaGraphicsRegisterFlags::none, res),
              aim::capture::CudaResult::success);

    aim::capture::CudaStreamHandle stream = nullptr;
    ASSERT_EQ(backend.create_stream(stream, aim::capture::CudaStreamFlags::non_blocking),
              aim::capture::CudaResult::success);
    ASSERT_TRUE(stream != nullptr);

    // 1. Map resource
    ASSERT_EQ(backend.map_resources(&res, 1, stream), aim::capture::CudaResult::success);
    ASSERT_TRUE(backend.is_resource_mapped(res));

    // Double map should fail
    ASSERT_EQ(backend.map_resources(&res, 1, stream), aim::capture::CudaResult::error_already_mapped);

    // 2. Get mapped array
    aim::capture::CudaArrayHandle array = nullptr;
    ASSERT_EQ(backend.get_mapped_array(res, 0, 0, array), aim::capture::CudaResult::success);
    ASSERT_TRUE(array != nullptr);

    // 3. Create surface object
    aim::capture::CudaSurfaceObjectHandle surf = 0;
    ASSERT_EQ(backend.create_surface_object(array, surf), aim::capture::CudaResult::success);
    ASSERT_TRUE(surf != 0);
    ASSERT_EQ(backend.active_surfaces_count(), 1u);

    // 4. Destroy surface object
    ASSERT_EQ(backend.destroy_surface_object(surf), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.active_surfaces_count(), 0u);

    // 5. Unmap resource
    ASSERT_EQ(backend.unmap_resources(&res, 1, stream), aim::capture::CudaResult::success);
    ASSERT_FALSE(backend.is_resource_mapped(res));

    // Getting array while unmapped should fail
    aim::capture::CudaArrayHandle unmapped_array = nullptr;
    ASSERT_EQ(backend.get_mapped_array(res, 0, 0, unmapped_array), aim::capture::CudaResult::error_not_mapped);

    // Double unmap should fail
    ASSERT_EQ(backend.unmap_resources(&res, 1, stream), aim::capture::CudaResult::error_not_mapped);

    // Cleanup
    ASSERT_EQ(backend.unregister_resource(res), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.destroy_stream(stream), aim::capture::CudaResult::success);

    return 0;
}

// =============================================================================
// Test 4: Streams and Events Management
// =============================================================================
int test_streams_and_events_management() {
    std::cout << "  [Test 4] Streams & Events Management..." << std::endl;

    aim::capture::MockCudaInteropBackend backend;

    aim::capture::CudaStreamHandle stream1 = nullptr;
    aim::capture::CudaStreamHandle stream2 = nullptr;
    ASSERT_EQ(backend.create_stream(stream1, aim::capture::CudaStreamFlags::non_blocking),
              aim::capture::CudaResult::success);
    ASSERT_EQ(backend.create_stream(stream2, aim::capture::CudaStreamFlags::non_blocking),
              aim::capture::CudaResult::success);
    ASSERT_TRUE(stream1 != nullptr);
    ASSERT_TRUE(stream2 != nullptr);
    ASSERT_TRUE(stream1 != stream2);
    ASSERT_EQ(backend.active_streams_count(), 2u);

    aim::capture::CudaEventHandle event1 = nullptr;
    aim::capture::CudaEventHandle event2 = nullptr;
    ASSERT_EQ(backend.create_event(event1, aim::capture::CudaEventFlags::disable_timing),
              aim::capture::CudaResult::success);
    ASSERT_EQ(backend.create_event(event2, aim::capture::CudaEventFlags::disable_timing),
              aim::capture::CudaResult::success);
    ASSERT_TRUE(event1 != nullptr);
    ASSERT_TRUE(event2 != nullptr);
    ASSERT_TRUE(event1 != event2);
    ASSERT_EQ(backend.active_events_count(), 2u);

    // Record and synchronize events across streams
    ASSERT_EQ(backend.record_event(event1, stream1), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.stream_wait_event(stream2, event1, 0), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.query_event(event1), aim::capture::CudaResult::success);

    // Destroy
    ASSERT_EQ(backend.destroy_event(event1), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.destroy_event(event2), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.destroy_stream(stream1), aim::capture::CudaResult::success);
    ASSERT_EQ(backend.destroy_stream(stream2), aim::capture::CudaResult::success);

    ASSERT_EQ(backend.active_events_count(), 0u);
    ASSERT_EQ(backend.active_streams_count(), 0u);

    return 0;
}

// =============================================================================
// Test 5: Scriptable Fault Injection
// =============================================================================
int test_scriptable_fault_injection() {
    std::cout << "  [Test 5] Scriptable Fault Injection..." << std::endl;

    aim::capture::MockCudaInteropBackend backend;
    void* mock_tex = reinterpret_cast<void*>(0x44440000ULL);

    // 1. Injected registration failure
    backend.set_register_result(aim::capture::CudaResult::error_out_of_memory);
    aim::capture::CudaGraphicsResourceHandle res = nullptr;
    ASSERT_EQ(backend.register_d3d11_texture(mock_tex, aim::capture::CudaGraphicsRegisterFlags::none, res),
              aim::capture::CudaResult::error_out_of_memory);
    ASSERT_TRUE(res == nullptr);

    // Restore registration
    backend.set_register_result(aim::capture::CudaResult::success);
    ASSERT_EQ(backend.register_d3d11_texture(mock_tex, aim::capture::CudaGraphicsRegisterFlags::none, res),
              aim::capture::CudaResult::success);

    aim::capture::CudaStreamHandle stream = nullptr;
    backend.create_stream(stream);

    // 2. Injected queued map failure
    backend.queue_map_result(aim::capture::CudaResult::error_map_failed);
    ASSERT_EQ(backend.map_resources(&res, 1, stream), aim::capture::CudaResult::error_map_failed);
    ASSERT_FALSE(backend.is_resource_mapped(res));

    // Next map should succeed
    ASSERT_EQ(backend.map_resources(&res, 1, stream), aim::capture::CudaResult::success);
    ASSERT_TRUE(backend.is_resource_mapped(res));

    // 3. Injected get_mapped_array failure
    backend.set_get_array_result(aim::capture::CudaResult::error_invalid_value);
    aim::capture::CudaArrayHandle arr = nullptr;
    ASSERT_EQ(backend.get_mapped_array(res, 0, 0, arr), aim::capture::CudaResult::error_invalid_value);
    ASSERT_TRUE(arr == nullptr);

    backend.set_get_array_result(aim::capture::CudaResult::success);
    ASSERT_EQ(backend.get_mapped_array(res, 0, 0, arr), aim::capture::CudaResult::success);
    ASSERT_TRUE(arr != nullptr);

    // 4. Injected surface creation failure
    backend.set_create_surface_result(aim::capture::CudaResult::error_out_of_memory);
    aim::capture::CudaSurfaceObjectHandle surf = 0;
    ASSERT_EQ(backend.create_surface_object(arr, surf), aim::capture::CudaResult::error_out_of_memory);
    ASSERT_EQ(surf, 0u);

    backend.set_create_surface_result(aim::capture::CudaResult::success);
    ASSERT_EQ(backend.create_surface_object(arr, surf), aim::capture::CudaResult::success);
    ASSERT_TRUE(surf != 0);

    // Cleanup
    backend.destroy_surface_object(surf);
    backend.unmap_resources(&res, 1, stream);
    backend.unregister_resource(res);
    backend.destroy_stream(stream);

    return 0;
}

// =============================================================================
// Test 6: RealCudaInteropBackend Hardware & Safe Fallback Verification
// =============================================================================
int test_real_backend_safe_fallback() {
    std::cout << "  [Test 6] RealCudaInteropBackend Hardware & Fail-Closed Fallback..." << std::endl;

    aim::capture::RealCudaInteropBackend real_backend;
    const bool is_avail = real_backend.is_cuda_available();
    std::cout << "    [Real Backend Status] CUDA Driver Available: " << (is_avail ? "YES" : "NO") << std::endl;

    if (is_avail) {
        // LUIDs are assigned by Windows and change after a reboot. Discover
        // the current adapters instead of coupling this test to one boot.
        int dev_matched = -1;
#if defined(_WIN32)
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        ASSERT_TRUE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()))));
        for (UINT index = 0;; ++index) {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            const HRESULT result = factory->EnumAdapters1(index, adapter.GetAddressOf());
            if (result == DXGI_ERROR_NOT_FOUND) break;
            ASSERT_TRUE(SUCCEEDED(result));
            DXGI_ADAPTER_DESC1 description{};
            ASSERT_TRUE(SUCCEEDED(adapter->GetDesc1(&description)));
            const auto luid = (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(description.AdapterLuid.HighPart)) << 32U) |
                static_cast<std::uint64_t>(description.AdapterLuid.LowPart);
            int device = -1;
            const bool matched = real_backend.validate_adapter_match(luid, device);
            std::cout << "    [Current adapter] vendor=" << description.VendorId
                      << " device=" << description.DeviceId << " cuda=" << device << '\n';
            if (matched) {
                ASSERT_TRUE(device >= 0);
                if (dev_matched < 0) dev_matched = device;
            } else {
                ASSERT_EQ(device, -1);
            }
            if (description.VendorId != 0x10DEU) ASSERT_FALSE(matched);
        }
#endif
        // CUDA-capable hardware must exercise a real match; no silent skip.
        ASSERT_TRUE(dev_matched >= 0);

        // Fail-closed verification on arbitrary non-existent LUID
        int dev_dummy = -1;
        ASSERT_FALSE(real_backend.validate_adapter_match(0xDEADBEEFCAFE0001ULL, dev_dummy));
        ASSERT_EQ(dev_dummy, -1);

        // 2. Real device selection
        ASSERT_EQ(real_backend.set_device(dev_matched), aim::capture::CudaResult::success);
        ASSERT_EQ(real_backend.set_device(-1), aim::capture::CudaResult::error_invalid_device);
        ASSERT_EQ(real_backend.set_device(9999), aim::capture::CudaResult::error_invalid_device);

        // 3. Real Stream & Event lifecycle on physical GPU
        aim::capture::CudaStreamHandle stream = nullptr;
        ASSERT_EQ(real_backend.create_stream(stream, aim::capture::CudaStreamFlags::non_blocking),
                  aim::capture::CudaResult::success);
        ASSERT_TRUE(stream != nullptr);

        aim::capture::CudaEventHandle event = nullptr;
        ASSERT_EQ(real_backend.create_event(event, aim::capture::CudaEventFlags::disable_timing),
                  aim::capture::CudaResult::success);
        ASSERT_TRUE(event != nullptr);

        ASSERT_EQ(real_backend.record_event(event, stream), aim::capture::CudaResult::success);
        ASSERT_EQ(real_backend.stream_wait_event(stream, event, 0), aim::capture::CudaResult::success);
        ASSERT_EQ(real_backend.synchronize_stream(stream), aim::capture::CudaResult::success);
        ASSERT_EQ(real_backend.query_event(event), aim::capture::CudaResult::success);

        ASSERT_EQ(real_backend.destroy_event(event), aim::capture::CudaResult::success);
        ASSERT_EQ(real_backend.destroy_stream(stream), aim::capture::CudaResult::success);

        // 4. Invalid handle boundary safety
        aim::capture::CudaGraphicsResourceHandle res = nullptr;
        ASSERT_EQ(real_backend.register_d3d11_texture(nullptr, aim::capture::CudaGraphicsRegisterFlags::none, res),
                  aim::capture::CudaResult::error_invalid_value);
        ASSERT_EQ(real_backend.unregister_resource(nullptr), aim::capture::CudaResult::error_invalid_value);
        ASSERT_EQ(real_backend.map_resources(nullptr, 0, nullptr), aim::capture::CudaResult::error_invalid_value);
        ASSERT_EQ(real_backend.unmap_resources(nullptr, 0, nullptr), aim::capture::CudaResult::error_invalid_value);

        aim::capture::CudaArrayHandle arr = nullptr;
        ASSERT_EQ(real_backend.get_mapped_array(nullptr, 0, 0, arr), aim::capture::CudaResult::error_invalid_value);

        aim::capture::CudaSurfaceObjectHandle surf = 0;
        ASSERT_EQ(real_backend.create_surface_object(nullptr, surf), aim::capture::CudaResult::error_invalid_value);
        ASSERT_EQ(real_backend.destroy_surface_object(0), aim::capture::CudaResult::error_invalid_value);
    } else {
        // Headless / CI verification
        int dev = -1;
        ASSERT_FALSE(real_backend.validate_adapter_match(0x00010688ULL, dev));
        ASSERT_EQ(dev, -1);
        ASSERT_EQ(real_backend.set_device(0), aim::capture::CudaResult::error_no_device);

        aim::capture::CudaGraphicsResourceHandle res = nullptr;
        ASSERT_EQ(real_backend.register_d3d11_texture(nullptr, aim::capture::CudaGraphicsRegisterFlags::none, res),
                  aim::capture::CudaResult::error_not_initialized);
    }

    return 0;
}

} // namespace

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "Test Suite: ICudaInteropBackend & MockCudaInteropBackend" << std::endl;
    std::cout << "=================================================================" << std::endl;

    int res = 0;
    res = test_adapter_matching_and_device_selection();
    if (res != 0) return res;

    res = test_texture_registration_and_unregistration();
    if (res != 0) return res;

    res = test_mapping_and_surface_object_lifecycle();
    if (res != 0) return res;

    res = test_streams_and_events_management();
    if (res != 0) return res;

    res = test_scriptable_fault_injection();
    if (res != 0) return res;

    res = test_real_backend_safe_fallback();
    if (res != 0) return res;

    std::cout << "=================================================================" << std::endl;
    std::cout << "ALL 6 CUDA INTEROP TESTS PASSED CLEANLY!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
