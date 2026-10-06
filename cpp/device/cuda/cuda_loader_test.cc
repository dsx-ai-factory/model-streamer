/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "device/cuda/cuda_lib.h"

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstring>

#include "device/cuda/cuda_device.h"

namespace runai::llm::streamer::device::cuda
{

// THE LOADER, against a fake libcuda.so.1 found through this binary's DT_RPATH.
//
// Everything else about CudaDevice and CudaBackend is reachable by filling CudaLib's function pointers
// by hand. What that cannot reach is loading: finding the library, resolving all 31 names, falling
// back from a _v2 form, and cuInit. This is the only test of that, and it FAILS rather than skips when
// the mock is not found - a broken DT_RPATH must not quietly reduce CUDA coverage to nothing.
class CudaLoader : public ::testing::Test
{
 protected:
    // EVERY test here, not just one: without this, a broken DT_RPATH makes these tests load the REAL
    // driver on a GPU machine and pass, while testing nothing they claim to. They must fail instead.
    void SetUp() override
    {
        ASSERT_NE(CudaLib::get(), nullptr) << "no CUDA driver library could be loaded at all";

        ASSERT_NE(marker(), nullptr)
            << "the loaded libcuda.so.1 is not the mock - check this target's DT_RPATH. These tests"
            << " would otherwise pass against the real driver and prove nothing about the loader";

        // The mock's device count is process-wide, so a test that changes it would otherwise decide
        // what the next one sees - including when it fails before putting it back. Reset here rather
        // than restored there, so no test depends on another having tidied up.
        ASSERT_TRUE(set_device_count(1));
    }

    // RETURNS rather than asserting: ASSERT_* only leaves the function it sits in, so a failure inside
    // a helper would let the caller carry on with a count it did not set.
    static bool set_device_count(unsigned count)
    {
        const auto setter = reinterpret_cast<void (*)(unsigned)>(symbol("runai_mock_cuda_set_device_count"));
        if (setter == nullptr)
        {
            return false;
        }

        setter(count);
        return true;
    }

    // The mock's own symbol, resolved from the library the loader ALREADY opened. NOLOAD, because a
    // second dlopen could find a different libcuda and say nothing about what CudaLib is using.
    static void * symbol(const char * name)
    {
        void * const handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (handle == nullptr)
        {
            return nullptr;
        }

        void * const found = ::dlsym(handle, name);

        // NOLOAD still took a reference. The library stays loaded either way - the loader holds its
        // own - so this only gives back what this call took.
        ::dlclose(handle);
        return found;
    }

    static void * marker() { return symbol("runai_mock_cuda_init_calls"); }
};

TEST_F(CudaLoader, Loads_The_Mock_And_Resolves_Every_Symbol)
{
    const auto * lib = CudaLib::get();
    ASSERT_NE(lib, nullptr);

    // One assertion per pointer would say the same thing thirty-one times. What matters is that
    // load_symbols() accepted the library, which it only does when every name resolved - a single
    // missing symbol returns an empty CudaLib and get() answers null, which the assertion above
    // catches.
    EXPECT_NE(lib->cuMemcpyHtoDAsync, nullptr) << "resolved through the _v2 name";
    EXPECT_NE(lib->cuEventDestroy, nullptr) << "resolved through the _v2 name";
}

// cuInit is called exactly once, by the loader, however many times a caller asks for the library:
// CudaLib::get() holds it in a function-local static.
TEST_F(CudaLoader, Initialises_The_Driver_Once)
{
    ASSERT_NE(CudaLib::get(), nullptr);
    ASSERT_NE(CudaLib::get(), nullptr);

    const auto calls = reinterpret_cast<unsigned (*)()>(marker());
    EXPECT_EQ(calls(), 1u);
}

// A DRIVER WITH NO DEVICES. The library loads, every symbol resolves, cuInit succeeds - and there is
// nothing to stream to. That is not hypothetical: the CUDA toolkit ships a stub libcuda.so which does
// exactly this, and a host with the toolkit but no GPU loads it.
//
// The backend must answer honestly rather than hand out a device that cannot be opened.
TEST_F(CudaLoader, A_Driver_With_No_Devices_Is_Reported)
{
    ASSERT_TRUE(set_device_count(0));

    auto backend = cuda::backend();
    ASSERT_NE(backend, nullptr) << "the library loaded, so there is a backend - it just has nothing";

    unsigned count = 1;
    EXPECT_EQ(backend->device_count(count), common::ResponseCode::Success);
    EXPECT_EQ(count, 0u);

    // AN ORDINAL NOTHING HAS OPENED, not 0. The backend caches the devices it opens and is shared
    // across this suite, so ordinal 0 may already be in that cache - and handing a cached device back
    // is right: a real driver's device count does not fall to zero under a running process.
    std::shared_ptr<Device> device;
    EXPECT_NE(backend->open_device(7, device), common::ResponseCode::Success)
        << "a device was opened on a driver that reports none";

}

// The backend on top of the loaded library: a device can be opened and its memory reached, with no
// driver anywhere. This is the shape a real host takes, exercised where there is no GPU.
TEST_F(CudaLoader, Opens_A_Device_Through_The_Loaded_Library)
{
    auto backend = cuda::backend();
    ASSERT_NE(backend, nullptr) << "no backend was built from the mock";

    unsigned count = 0;
    ASSERT_EQ(backend->device_count(count), common::ResponseCode::Success);
    ASSERT_GT(count, 0u);

    std::shared_ptr<Device> device;
    ASSERT_EQ(backend->open_device(0, device), common::ResponseCode::Success);
    ASSERT_NE(device, nullptr);

    ASSERT_EQ(device->bind_thread(), common::ResponseCode::Success);

    void * pinned = nullptr;
    ASSERT_EQ(device->host_alloc(1024, &pinned), common::ResponseCode::Success);
    ASSERT_NE(pinned, nullptr);

    void * target = nullptr;
    ASSERT_EQ(device->device_alloc(1024, &target), common::ResponseCode::Success);

    StreamHandle stream = nullptr;
    ASSERT_EQ(device->stream_create(stream), common::ResponseCode::Success);

    // The mock's copy is a memcpy, so the bytes are checkable - which is what makes this a test of the
    // path rather than of the return codes.
    std::memset(pinned, 0x5a, 1024);
    ASSERT_EQ(device->memcpy_h2d_async(target, pinned, 1024, stream), common::ResponseCode::Success);
    ASSERT_EQ(device->stream_synchronize(stream), common::ResponseCode::Success);
    EXPECT_EQ(std::memcmp(target, pinned, 1024), 0);

    EXPECT_EQ(device->stream_destroy(stream), common::ResponseCode::Success);
    EXPECT_EQ(device->device_free(target), common::ResponseCode::Success);
    EXPECT_EQ(device->host_free(pinned), common::ResponseCode::Success);
}

} // namespace runai::llm::streamer::device::cuda
