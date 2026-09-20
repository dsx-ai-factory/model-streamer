#include "device/cuda/cuda_device.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "utils/temp/env/env.h"

namespace runai::llm::streamer::device::cuda
{

namespace
{

// Driver entry points are plain function pointers and cannot capture, so what the stubs saw is
// recorded here and read back by the test.
struct Recorded
{
    static inline CUcontext current_context = nullptr;
    static inline unsigned  retain_calls = 0;
    static inline unsigned  release_calls = 0;
    static inline size_t    requested_bytes = 0;
    static inline void *    registered_ptr = nullptr;
    static inline size_t    registered_bytes = 0;
    static inline unsigned  registered_flags = 0;
    static inline unsigned  unregister_calls = 0;
    static inline unsigned  free_host_calls = 0;
    static inline unsigned  stream_flags = 0;
    static inline unsigned  event_flags = 0;

    // What the next stubbed operation returns, so a failure path can be reached without a device.
    static inline CUresult register_result = CUDA_SUCCESS;
    static inline CUresult query_result = CUDA_SUCCESS;
    static inline CUresult copy_result = CUDA_SUCCESS;
    static inline CUresult retain_result = CUDA_SUCCESS;

    static void reset()
    {
        current_context = nullptr;
        retain_calls = release_calls = unregister_calls = free_host_calls = 0;
        requested_bytes = registered_bytes = 0;
        registered_ptr = nullptr;
        registered_flags = stream_flags = event_flags = 0;
        register_result = query_result = copy_result = retain_result = CUDA_SUCCESS;
    }
};

CUresult stub_set_current(CUcontext context) { Recorded::current_context = context; return CUDA_SUCCESS; }
CUresult stub_get_device(CUdevice * device, int ordinal) { *device = ordinal; return CUDA_SUCCESS; }
CUresult stub_device_count(int * count) { *count = 4; return CUDA_SUCCESS; }

CUresult stub_retain(CUcontext * context, CUdevice)
{
    ++Recorded::retain_calls;
    if (Recorded::retain_result != CUDA_SUCCESS)
    {
        return Recorded::retain_result;
    }
    *context = reinterpret_cast<CUcontext>(0xC0FFEE);
    return CUDA_SUCCESS;
}

CUresult stub_release(CUdevice) { ++Recorded::release_calls; return CUDA_SUCCESS; }

CUresult stub_host_alloc(void ** ptr, size_t bytesize, unsigned int)
{
    Recorded::requested_bytes = bytesize;
    *ptr = std::malloc(bytesize);
    return CUDA_SUCCESS;
}

CUresult stub_free_host(void * ptr) { ++Recorded::free_host_calls; std::free(ptr); return CUDA_SUCCESS; }

CUresult stub_register(void * ptr, size_t bytesize, unsigned int flags)
{
    Recorded::registered_ptr = ptr;
    Recorded::registered_bytes = bytesize;
    Recorded::registered_flags = flags;
    return Recorded::register_result;
}

CUresult stub_unregister(void *) { ++Recorded::unregister_calls; return CUDA_SUCCESS; }

CUresult stub_stream_create(CUstream * stream, unsigned int flags)
{
    Recorded::stream_flags = flags;
    *stream = reinterpret_cast<CUstream>(0x5EA);
    return CUDA_SUCCESS;
}

CUresult stub_event_create(CUevent * event, unsigned int flags)
{
    Recorded::event_flags = flags;
    *event = reinterpret_cast<CUevent>(0xE7E);
    return CUDA_SUCCESS;
}

CUresult stub_event_query(CUevent) { return Recorded::query_result; }
CUresult stub_stream_query(CUstream) { return Recorded::query_result; }
CUresult stub_copy(CUdeviceptr, const void *, size_t, CUstream) { return Recorded::copy_result; }

// Every pointer the code under test may reach must be set: a half-filled table would crash rather
// than fail, and the destructor alone calls cuDevicePrimaryCtxRelease.
CudaLib stub_lib()
{
    CudaLib lib = {};
    lib.cuCtxSetCurrent = &stub_set_current;
    lib.cuDeviceGet = &stub_get_device;
    lib.cuDeviceGetCount = &stub_device_count;
    lib.cuDevicePrimaryCtxRetain = &stub_retain;
    lib.cuDevicePrimaryCtxRelease = &stub_release;
    lib.cuMemHostAlloc = &stub_host_alloc;
    lib.cuMemFreeHost = &stub_free_host;
    lib.cuMemHostRegister = &stub_register;
    lib.cuMemHostUnregister = &stub_unregister;
    lib.cuStreamCreate = &stub_stream_create;
    lib.cuStreamQuery = &stub_stream_query;
    lib.cuEventCreate = &stub_event_create;
    lib.cuEventQuery = &stub_event_query;
    lib.cuMemcpyHtoDAsync = &stub_copy;
    return lib;
}

CUcontext the_context = reinterpret_cast<CUcontext>(0xC0FFEE);

class CudaDeviceTest : public ::testing::Test
{
 protected:
    void SetUp() override { Recorded::reset(); }

    CudaLib _lib = stub_lib();
};

} // namespace

TEST_F(CudaDeviceTest, BindThreadMakesThisDevicesContextCurrent)
{
    CudaDevice device(_lib, 3, the_context, PinnedMode::Allocate);

    ASSERT_EQ(device.bind_thread(), common::ResponseCode::Success);
    EXPECT_EQ(Recorded::current_context, the_context);
}

TEST_F(CudaDeviceTest, AllocateModeAsksTheDriverForPinnedMemory)
{
    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    void * ptr = nullptr;
    ASSERT_EQ(device.host_alloc(1000, &ptr), common::ResponseCode::Success);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(Recorded::requested_bytes, 1000u);
    EXPECT_EQ(Recorded::registered_ptr, nullptr);   // no registration in this mode

    ASSERT_EQ(device.host_free(ptr), common::ResponseCode::Success);
    EXPECT_EQ(Recorded::free_host_calls, 1u);
}

// Page-locking works on whole pages, and the PORTABLE flag is what makes registered memory pinned
// for every context - cuda.h promises that for allocated memory but not for registered memory.
TEST_F(CudaDeviceTest, RegisterModePinsWholePagesAndIsPortable)
{
    CudaDevice device(_lib, 0, the_context, PinnedMode::Register);

    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));

    void * ptr = nullptr;
    ASSERT_EQ(device.host_alloc(page + 1, &ptr), common::ResponseCode::Success);
    ASSERT_NE(ptr, nullptr);

    EXPECT_EQ(Recorded::registered_ptr, ptr);
    EXPECT_EQ(Recorded::registered_bytes, 2 * page);
    EXPECT_EQ(Recorded::registered_flags & CU_MEMHOSTREGISTER_PORTABLE, CU_MEMHOSTREGISTER_PORTABLE);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % page, 0u);

    ASSERT_EQ(device.host_free(ptr), common::ResponseCode::Success);
    EXPECT_EQ(Recorded::unregister_calls, 1u);
    EXPECT_EQ(Recorded::free_host_calls, 0u);       // freed by us, not by the driver
}

// The host memory is ours until registration succeeds; a failure must not hand back a pointer the
// caller would then try to free through the driver.
TEST_F(CudaDeviceTest, RegisterFailureReturnsNoPointer)
{
    Recorded::register_result = CUDA_ERROR_OUT_OF_MEMORY;

    CudaDevice device(_lib, 0, the_context, PinnedMode::Register);

    void * ptr = reinterpret_cast<void *>(0xBAD);
    EXPECT_EQ(device.host_alloc(4096, &ptr), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_EQ(ptr, nullptr);
    EXPECT_EQ(Recorded::unregister_calls, 0u);
}

TEST_F(CudaDeviceTest, StreamsDoNotSynchroniseWithTheCallersDefaultStream)
{
    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    StreamHandle stream = nullptr;
    ASSERT_EQ(device.stream_create(stream), common::ResponseCode::Success);
    EXPECT_EQ(Recorded::stream_flags, static_cast<unsigned>(CU_STREAM_NON_BLOCKING));
}

TEST_F(CudaDeviceTest, EventsCarryNoTiming)
{
    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    EventHandle event = nullptr;
    ASSERT_EQ(device.event_create(event), common::ResponseCode::Success);
    EXPECT_EQ(Recorded::event_flags & CU_EVENT_DISABLE_TIMING, static_cast<unsigned>(CU_EVENT_DISABLE_TIMING));
}

// Work still running is not an error: this is how a reader asks whether a staging buffer is free.
TEST_F(CudaDeviceTest, NotReadyIsSuccess)
{
    Recorded::query_result = CUDA_ERROR_NOT_READY;

    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    Status status = Status::Ready;
    ASSERT_EQ(device.event_query(nullptr, status), common::ResponseCode::Success);
    EXPECT_EQ(status, Status::NotReady);

    status = Status::Ready;
    ASSERT_EQ(device.stream_query(nullptr, status), common::ResponseCode::Success);
    EXPECT_EQ(status, Status::NotReady);
}

// A failed query must not report Ready. Whoever reads the status first would otherwise reuse a
// staging buffer whose copy never landed.
TEST_F(CudaDeviceTest, FailedQueryNeverReportsReady)
{
    Recorded::query_result = CUDA_ERROR_INVALID_CONTEXT;

    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    Status status = Status::NotReady;
    EXPECT_EQ(device.event_query(nullptr, status), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(status, Status::NotReady);

    status = Status::NotReady;
    EXPECT_EQ(device.stream_query(nullptr, status), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(status, Status::NotReady);
}

TEST_F(CudaDeviceTest, AFailedCopyIsATransferError)
{
    Recorded::copy_result = CUDA_ERROR_LAUNCH_FAILED;

    CudaDevice device(_lib, 0, the_context, PinnedMode::Allocate);

    EXPECT_EQ(device.memcpy_h2d_async(nullptr, nullptr, 16, nullptr), common::ResponseCode::DeviceTransferError);
}

TEST_F(CudaDeviceTest, ReleasesTheRetainedContextOnce)
{
    {
        CudaDevice device(_lib, 7, the_context, PinnedMode::Allocate);
    }
    EXPECT_EQ(Recorded::release_calls, 1u);
}

TEST_F(CudaDeviceTest, DeviceCountIsReported)
{
    CudaBackend backend(_lib);

    unsigned count = 0;
    ASSERT_EQ(backend.device_count(count), common::ResponseCode::Success);
    EXPECT_EQ(count, 4u);
}

// The context is retained once per ordinal and the device is shared. Retaining again per caller
// would leave each one with its own table and its own pinned buffers.
TEST_F(CudaDeviceTest, OneDevicePerOrdinal)
{
    CudaBackend backend(_lib);

    std::shared_ptr<Device> first;
    std::shared_ptr<Device> again;
    std::shared_ptr<Device> other;

    ASSERT_EQ(backend.open_device(1, first), common::ResponseCode::Success);
    ASSERT_EQ(backend.open_device(1, again), common::ResponseCode::Success);
    ASSERT_EQ(backend.open_device(2, other), common::ResponseCode::Success);

    EXPECT_EQ(first.get(), again.get());
    EXPECT_NE(first.get(), other.get());
    EXPECT_EQ(Recorded::retain_calls, 2u);
}

TEST_F(CudaDeviceTest, OpenDeviceReportsADriverFailure)
{
    Recorded::retain_result = CUDA_ERROR_INVALID_CONTEXT;

    CudaBackend backend(_lib);

    std::shared_ptr<Device> device;
    EXPECT_EQ(backend.open_device(0, device), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(device, nullptr);
}

// One backend per process. A second one would keep its own device table and retain the same
// primary contexts again, so two callers would not share a device at all.
//
// Passes with or without a driver present: with none, backend() returns nullptr both times, which
// is the other thing worth asserting - a machine with no CUDA gets an answer, not a crash.
TEST(Backend, IsTheSameInstanceEveryTime)
{
    EXPECT_EQ(backend().get(), backend().get());
}

TEST(PinnedMode, ReadFromTheEnvironment)
{
    const std::string variable = "RUNAI_STREAMER_PINNED_MEMORY_MODE";

    {
        utils::temp::UnsetEnv unset(variable);
        EXPECT_EQ(pinned_mode(), PinnedMode::Allocate);
    }
    {
        utils::temp::Env env(variable, "register");
        EXPECT_EQ(pinned_mode(), PinnedMode::Register);
    }
    {
        utils::temp::Env env(variable, "allocate");
        EXPECT_EQ(pinned_mode(), PinnedMode::Allocate);
    }
    // An unreadable value must not silently change how memory is obtained.
    {
        utils::temp::Env env(variable, "nonsense");
        EXPECT_EQ(pinned_mode(), PinnedMode::Allocate);
    }
}

} // namespace runai::llm::streamer::device::cuda
