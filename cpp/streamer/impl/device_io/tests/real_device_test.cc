// The only tests in this repository that talk to a real driver.
//
// Everything else on the device path runs against MockDevice or against stubbed function pointers,
// and neither can stand in for what matters most here: MockDevice's copy is a synchronous memcpy, so
// a buffer is never genuinely held across a DMA, and a host destination never punishes a stray write
// the way a device pointer does.
//
// SKIPPED where there is no driver or no device, so CI stays green on a CPU host. A skip is reported
// with its reason - a silently empty run would look the same as a passing one.
//
// Its own package because it needs cuda.h, which is otherwise visible only under device/cuda. The
// readback needs the driver directly - the Device interface has no device-to-host copy, and adding
// one for a test would be the wrong reason to widen it. Widening cuda.h to THIS package alone keeps
// the dependency pointing downward, which a test under device/cuda could not.

#include "device/cuda/cuda_device.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <iostream>
#include <vector>

#include "device/cuda/cuda_lib.h"
#include "streamer/impl/device_io/device_writer/device_writer.h"
#include "streamer/impl/device_io/device_writer_client/device_writer_client.h"
#include "streamer/impl/streamer/streamer.h"
#include "streamer/streamer.h"
#include "utils/random/random.h"
#include "utils/temp/env/env.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "utils/dylib/dylib.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 8ul << 20;   // a realistic chunk, and large enough for a copy to take time

class RealDevice : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _lib = device::cuda::CudaLib::get();
        if (_lib == nullptr)
        {
            GTEST_SKIP() << "no CUDA driver on this host (libcuda.so.1 could not be loaded)";
        }

        _backend = device::cuda::backend();
        ASSERT_NE(_backend, nullptr) << "the driver loaded but no backend was built";

        // SKIPPED rather than failed from here on. A host with a driver we cannot get a usable device
        // out of - no GPU assigned to the container, every device in exclusive mode and taken, a MIG
        // layout we were not given a slice of - is a host this test can say nothing about. Failing
        // there would turn somebody else's machine configuration into a red build.
        unsigned count = 0;
        if (_backend->device_count(count) != common::ResponseCode::Success || count == 0)
        {
            GTEST_SKIP() << "a CUDA driver, but no device this process can use";
        }

        if (_backend->open_device(0, _device) != common::ResponseCode::Success ||
            _device->bind_thread() != common::ResponseCode::Success)
        {
            GTEST_SKIP() << "device 0 could not be opened here";
        }
    }

    // Read device memory back.
    //
    // Resolved HERE rather than added to CudaLib: the streamer only ever writes to a device, and a
    // symbol carried in the product for a test's benefit is a symbol someone later assumes is used.
    // RTLD_NOLOAD because the loader has already opened it - this only takes a handle to it.
    std::vector<char> read_back(void * device_ptr, size_t bytesize)
    {
        using CopyDtoH = CUresult (*)(void *, CUdeviceptr, size_t);

        void * const handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        EXPECT_NE(handle, nullptr) << "the driver is loaded, so this cannot fail";

        const auto copy = reinterpret_cast<CopyDtoH>(::dlsym(handle, "cuMemcpyDtoH_v2"));
        EXPECT_NE(copy, nullptr);

        std::vector<char> out(bytesize);
        if (copy != nullptr)
        {
            EXPECT_EQ(copy(out.data(), reinterpret_cast<CUdeviceptr>(device_ptr), bytesize), CUDA_SUCCESS);
        }

        ::dlclose(handle);
        return out;
    }

    const device::cuda::CudaLib * _lib = nullptr;
    std::shared_ptr<device::Backend> _backend;
    std::shared_ptr<device::Device> _device;
};

} // namespace

// Both pinned modes, end to end through the device interface: pin, allocate, copy, wait, verify.
TEST_F(RealDevice, Pinned_Memory_Copies_To_The_Device)
{
    for (const char * mode : { "allocate", "register" })
    {
        utils::temp::Env pinned(std::string("RUNAI_STREAMER_PINNED_MEMORY_MODE"), std::string(mode));

        // A fresh backend so the mode is read again. The context is retained per device, so this is
        // the same device either way.
        std::shared_ptr<device::Device> device;
        ASSERT_EQ(device::cuda::backend()->open_device(0, device), common::ResponseCode::Success) << mode;
        ASSERT_EQ(device->bind_thread(), common::ResponseCode::Success) << mode;

        void * host = nullptr;
        ASSERT_EQ(device->host_alloc(Buffer, &host), common::ResponseCode::Success) << mode;
        ASSERT_NE(host, nullptr) << mode;

        void * target = nullptr;
        ASSERT_EQ(device->device_alloc(Buffer, &target), common::ResponseCode::Success) << mode;

        const auto data = utils::random::buffer(Buffer);
        std::memcpy(host, data.data(), Buffer);

        device::StreamHandle stream = nullptr;
        device::EventHandle event = nullptr;
        ASSERT_EQ(device->stream_create(stream), common::ResponseCode::Success) << mode;
        ASSERT_EQ(device->event_create(event), common::ResponseCode::Success) << mode;

        ASSERT_EQ(device->memcpy_h2d_async(target, host, Buffer, stream), common::ResponseCode::Success) << mode;
        ASSERT_EQ(device->event_record(event, stream), common::ResponseCode::Success) << mode;
        ASSERT_EQ(device->event_synchronize(event), common::ResponseCode::Success) << mode;

        const auto landed = read_back(target, Buffer);
        EXPECT_EQ(std::memcmp(landed.data(), data.data(), Buffer), 0) << mode;

        EXPECT_EQ(device->event_destroy(event), common::ResponseCode::Success) << mode;
        EXPECT_EQ(device->stream_destroy(stream), common::ResponseCode::Success) << mode;
        EXPECT_EQ(device->device_free(target), common::ResponseCode::Success) << mode;
        EXPECT_EQ(device->host_free(host), common::ResponseCode::Success) << mode;
    }
}

// THE property the mock cannot show: the copy is still running when memcpy_h2d_async returns, so a
// staging buffer really is held across it and a response really must wait for the event.
//
// Asserted as a ratio with a wide margin rather than an absolute time, so a slow or busy machine does
// not fail it: enqueuing 16 copies must be far cheaper than performing them.
TEST_F(RealDevice, A_Copy_Is_Still_Running_When_Enqueue_Returns)
{
    constexpr unsigned Copies = 16;

    void * host = nullptr;
    ASSERT_EQ(_device->host_alloc(Buffer, &host), common::ResponseCode::Success);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Buffer, &target), common::ResponseCode::Success);

    device::StreamHandle stream = nullptr;
    device::EventHandle event = nullptr;
    ASSERT_EQ(_device->stream_create(stream), common::ResponseCode::Success);
    ASSERT_EQ(_device->event_create(event), common::ResponseCode::Success);

    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < Copies; ++i)
    {
        ASSERT_EQ(_device->memcpy_h2d_async(target, host, Buffer, stream), common::ResponseCode::Success);
    }
    ASSERT_EQ(_device->event_record(event, stream), common::ResponseCode::Success);
    const auto enqueued = std::chrono::steady_clock::now();

    // Not yet done, almost certainly - but a query that says ready is not a failure, only a very fast
    // machine. What follows is the real assertion.
    device::Status status = device::Status::Ready;
    EXPECT_EQ(_device->event_query(event, status), common::ResponseCode::Success);

    ASSERT_EQ(_device->event_synchronize(event), common::ResponseCode::Success);
    const auto done = std::chrono::steady_clock::now();

    const auto enqueue_us = std::chrono::duration_cast<std::chrono::microseconds>(enqueued - start).count();
    const auto wait_us = std::chrono::duration_cast<std::chrono::microseconds>(done - enqueued).count();

    std::cerr << Copies << " copies of " << Buffer << " bytes: enqueue " << enqueue_us
              << " us, wait " << wait_us << " us\n";

    EXPECT_GT(wait_us, enqueue_us)
        << "the copies finished before the last one was even enqueued - this is not asynchronous, and"
        << " every buffer-lifetime rule in the staging pool rests on it being so";

    EXPECT_EQ(_device->event_destroy(event), common::ResponseCode::Success);
    EXPECT_EQ(_device->stream_destroy(stream), common::ResponseCode::Success);
    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
    EXPECT_EQ(_device->host_free(host), common::ResponseCode::Success);
}

// The whole copy path as a worker uses it: a client takes buffers from its pool, writes them to a
// real device, and every one comes back. The completion arrives on the StreamWaiter's thread, after a
// real DMA rather than after a memcpy.
TEST_F(RealDevice, The_Writer_Client_Copies_To_A_Real_Device)
{
    constexpr unsigned Window = 4;
    constexpr unsigned Rounds = 12;

    auto writer = std::make_shared<DeviceWriter>(device::cuda::backend);

    DeviceWriterClient::Buffers geometry;
    geometry.buffer_bytesize = Buffer;
    geometry.slab_bytesize = 2 * Buffer;

    DeviceWriterClient client(writer, geometry, Window);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Rounds * Buffer, &target), common::ResponseCode::Success);

    std::vector<std::vector<uint8_t>> written;
    std::atomic<unsigned> done{0};

    for (unsigned round = 0; round < Rounds; ++round)
    {
        StagingBuffer buffer;
        // The pool's ceiling is the window, so this can only be empty while copies are in flight.
        for (unsigned attempt = 0; attempt < 10000 && !buffer.valid(); ++attempt)
        {
            ASSERT_EQ(client.take(0, buffer), common::ResponseCode::Success);
        }
        ASSERT_TRUE(buffer.valid()) << "round " << round;

        written.push_back(utils::random::buffer(Buffer));
        std::memcpy(buffer.data, written.back().data(), Buffer);

        ASSERT_EQ(client.write(0, buffer, Buffer, static_cast<char *>(target) + round * Buffer,
                               [&](common::ResponseCode ret)
                               {
                                   EXPECT_EQ(ret, common::ResponseCode::Success);
                                   ++done;
                               }),
                  common::ResponseCode::Success) << "round " << round;
    }

    for (unsigned i = 0; i < 10000 && done.load() < Rounds; ++i)
    {
        ::usleep(1000);
    }
    ASSERT_EQ(done.load(), Rounds) << "a copy never completed";

    // Every round landed at its own offset, in full. A buffer reused before its copy retired would
    // show up here as one round's bytes appearing under another's offset.
    for (unsigned round = 0; round < Rounds; ++round)
    {
        const auto landed = read_back(static_cast<char *>(target) + round * Buffer, Buffer);
        EXPECT_EQ(std::memcmp(landed.data(), written[round].data(), Buffer), 0) << "round " << round;
    }

    EXPECT_LE(client.buffers(), Window) << "the pool stayed inside the window";
    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

// Pinned memory is allocated through ONE context and used from every other. The whole per-worker
// pool design rests on it: a worker names a second device and keeps the buffers it already has.
//
// Needs two devices, so it skips on a single-GPU host.
TEST_F(RealDevice, One_Pool_Serves_A_Second_Device)
{
    unsigned count = 0;
    ASSERT_EQ(_backend->device_count(count), common::ResponseCode::Success);
    if (count < 2)
    {
        GTEST_SKIP() << "one device on this host; this needs two";
    }

    auto writer = std::make_shared<DeviceWriter>(device::cuda::backend);

    DeviceWriterClient::Buffers geometry;
    geometry.buffer_bytesize = Buffer;
    geometry.slab_bytesize = Buffer;

    // Two buffers, because this holds one for device 0 while opening device 1.
    DeviceWriterClient client(writer, geometry, 2 /* window */);

    std::shared_ptr<device::Device> second;
    ASSERT_EQ(_backend->open_device(1, second), common::ResponseCode::Success);
    ASSERT_EQ(second->bind_thread(), common::ResponseCode::Success);

    void * target = nullptr;
    ASSERT_EQ(second->device_alloc(Buffer, &target), common::ResponseCode::Success);

    // Taken for device 0, which is what builds the pool - so the memory is pinned through device 0's
    // context and then written to device 1.
    StagingBuffer buffer;
    ASSERT_EQ(client.take(0, buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    // Opens device 1 on this client, as a worker does by reading for it. write() refuses an ordinal
    // take() never opened, so this is the call that makes the next one legal.
    StagingBuffer opened;
    ASSERT_EQ(client.take(1, opened), common::ResponseCode::Success);
    ASSERT_TRUE(opened.valid());

    const auto data = utils::random::buffer(Buffer);
    std::memcpy(buffer.data, data.data(), Buffer);

    std::atomic<int> reported{-1};
    ASSERT_EQ(client.write(1, buffer, Buffer, target,
                           [&](common::ResponseCode ret) { reported.store(static_cast<int>(ret)); }),
              common::ResponseCode::Success);

    for (unsigned i = 0; i < 10000 && reported.load() < 0; ++i)
    {
        ::usleep(1000);
    }
    ASSERT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);

    ASSERT_EQ(_device->bind_thread(), common::ResponseCode::Success);
    const auto landed = read_back(target, Buffer);
    EXPECT_EQ(std::memcmp(landed.data(), data.data(), Buffer), 0)
        << "memory pinned through device 0 did not reach device 1";

    EXPECT_EQ(client.devices(), 2u);

    // The split this test exists for. Buffers: ONE pool, holding the two that were taken - pinned
    // memory reaches every context, and the copy above proved it by going to device 1 from a buffer
    // taken for device 0. Events: one pool PER DEVICE, and only device 1 was copied to, so only its
    // pool made one.
    EXPECT_EQ(client.buffers(), 2u) << "one pool of buffers, shared";
    EXPECT_EQ(client.events(1), 1u) << "the event came from device 1's own pool";
    EXPECT_EQ(client.events(0), 0u) << "device 0 was never copied to, so it made no event";

    EXPECT_EQ(second->device_free(target), common::ResponseCode::Success);
}

// The whole of step 8a against a real driver: a file read by the async io worker into pinned memory,
// copied to a device buffer, and the caller answered only once it is there.
//
// Skipped where the file system resolves to the synchronous reader, which cannot serve a device
// destination yet and answers UnsupportedDeviceType by design.
TEST_F(RealDevice, A_Submission_Reads_Into_Device_Memory)
{
    constexpr size_t Total = 4 * Buffer;

    const auto data = utils::random::buffer(Total);
    utils::temp::File file(data);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Total, &target), common::ResponseCode::Success);

    Streamer streamer;

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    constexpr unsigned Ranges = 8;
    for (unsigned i = 0; i < Ranges; ++i)
    {
        request[0].ranges.push_back(
            ReadRange{ i * (Total / Ranges), Total / Ranges,
                       static_cast<char *>(target) + i * (Total / Ranges) });
    }

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, common::Device::cuda(0), &submission_id),
              common::ResponseCode::Success);

    unsigned answered = 0;
    auto worst = common::ResponseCode::Success;
    for (unsigned i = 0; i < Ranges; ++i)
    {
        bool done = false;
        const auto response = streamer.response(30000, done);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "only " << i << " ranges answered";
        if (response.ret != common::ResponseCode::Success)
        {
            worst = response.ret;
        }
        ++answered;
    }
    EXPECT_EQ(answered, Ranges);

    if (worst == common::ResponseCode::UnsupportedDeviceType)
    {
        GTEST_SKIP() << "this file system resolved to the synchronous reader, which cannot serve a"
                     << " device destination yet (strategy " << streamer.fs_strategy() << ")";
    }

    ASSERT_EQ(worst, common::ResponseCode::Success) << "strategy " << streamer.fs_strategy();

    const auto landed = read_back(target, Total);
    EXPECT_EQ(std::memcmp(landed.data(), data.data(), Total), 0)
        << "the bytes on the device are not the bytes in the file";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

// THE PUBLIC API, to a real device. Every other test here drives impl::Streamer, which never sees the
// boundary check - so this is the only one that shows a caller of the shipped C API reaching device
// memory at all.
//
// The API ADMITS a device submission and lets the readers answer per range: whether there is a driver,
// a device, or memory to pin is not known at admission, and by then every range owes a response.
TEST_F(RealDevice, The_Public_Api_Reads_Into_Device_Memory)
{
    constexpr size_t Total = 4 * Buffer;
    constexpr unsigned Ranges = 4;

    const auto data = utils::random::buffer(Total);
    utils::temp::File file(data);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Total, &target), common::ResponseCode::Success);

    void * streamer = nullptr;
    ASSERT_EQ(runai_file_streamer_start(&streamer), static_cast<int>(common::ResponseCode::Success));

    const char * path = file.path.c_str();
    unsigned num_ranges = Ranges;

    std::vector<size_t> offsets(Ranges);
    std::vector<size_t> sizes(Ranges, Total / Ranges);
    std::vector<void *> dsts(Ranges);
    for (unsigned i = 0; i < Ranges; ++i)
    {
        offsets[i] = i * (Total / Ranges);
        dsts[i] = static_cast<char *>(target) + offsets[i];
    }

    RunaiFileStreamerDevice device;
    device.type = RUNAI_FILE_STREAMER_DEVICE_CUDA;
    device.id = 0;

    RunaiFileStreamerSubmissionId submission_id = 0;
    ASSERT_EQ(runai_file_streamer_request(streamer, &submission_id, 1, &path, &num_ranges,
                                          offsets.data(), sizes.data(), dsts.data(), device),
              static_cast<int>(common::ResponseCode::Success))
        << "the public API refused a device submission";

    for (unsigned i = 0; i < Ranges; ++i)
    {
        RunaiFileStreamerSubmissionId answered = 0;
        unsigned file_index = 0;
        unsigned index = 0;
        int submission_done = 0;
        EXPECT_EQ(runai_file_streamer_response(streamer, &answered, &file_index, &index,
                                               &submission_done, 30000 /* ms */),
                  static_cast<int>(common::ResponseCode::Success)) << "range " << i;
        EXPECT_EQ(answered, submission_id);
    }

    runai_file_streamer_end(streamer);

    const auto landed = read_back(target, Total);
    EXPECT_EQ(std::memcmp(landed.data(), data.data(), Total), 0)
        << "the bytes on the device are not the bytes in the file";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

// OBJECT STORAGE to a real device, through the s3 MOCK plugin: a real driver, real pinned buffers and
// a real copy, without a bucket.
//
// This is the arrangement that has hidden bugs twice before. The mock device did not model that an
// event belongs to the context that made it, nor that a thread needs one bound - both only showed on
// hardware. The object path pins buffers sized by a PLUGIN's window, which no other test does, and
// hands that pinned memory to a plugin to write into.
//
// NOT a byte check: the s3 mock records reads and never writes into the destination, so nothing here
// can tell correct bytes from zeroed ones. What it proves is that the pinning, the context, the events
// and the copy all work against the driver for object storage as they do for files.
TEST_F(RealDevice, An_Object_Storage_Submission_Reads_Into_Device_Memory)
{
    constexpr size_t Total = 4 * Buffer;
    constexpr unsigned Ranges = 8;

    utils::Dylib dylib("libstreamers3.so");
    dylib.dlsym<void(*)(unsigned)>("runai_mock_s3_set_response_time_ms")(0);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Total, &target), common::ResponseCode::Success);

    Streamer streamer;

    std::vector<FileRanges> request(1);
    request[0].path = "s3://a-bucket/an-object";
    for (unsigned i = 0; i < Ranges; ++i)
    {
        request[0].ranges.push_back(
            ReadRange{ i * (Total / Ranges), Total / Ranges,
                       static_cast<char *>(target) + i * (Total / Ranges) });
    }

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, common::Device::cuda(0), &submission_id),
              common::ResponseCode::Success);

    auto worst = common::ResponseCode::Success;
    for (unsigned i = 0; i < Ranges; ++i)
    {
        bool done = false;
        const auto response = streamer.response(30000, done);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "only " << i << " ranges answered";
        if (response.ret != common::ResponseCode::Success)
        {
            worst = response.ret;
        }
    }

    EXPECT_EQ(worst, common::ResponseCode::Success)
        << "an object storage range bound for a device was not read";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);

    dylib.dlsym<void(*)()>("runai_mock_s3_cleanup")();
    common::s3::S3ClientWrapper::shutdown();
}

// A FULL WINDOW against a real device, which is the state every invariant here is written for and
// the one a mock never reaches: copies genuinely in flight, buffers genuinely not back yet.
//
// Two things are asserted. The pool never grows past its window - if it did, a worker would pin more
// than the queue depth it was sized for.
//
// And the assumption every caller of take() makes: a copy REPORTED AS DONE has given its buffer back.
// A worker frees its window slot on that report and immediately submits the next chunk, so a refusal
// while fewer than `window` copies are outstanding means a buffer was accounted before it returned.
// That is not a pool invariant - the pool is entitled to be empty when everything is out - which is
// why asserting on the pool alone misses it.
TEST_F(RealDevice, The_Pool_Never_Exceeds_Its_Window)
{
    constexpr unsigned Window = 8;
    constexpr unsigned Rounds = 400;

    auto writer = std::make_shared<DeviceWriter>(device::cuda::backend);

    DeviceWriterClient::Buffers geometry;
    geometry.buffer_bytesize = Buffer;
    geometry.slab_bytesize = 2 * Buffer;

    DeviceWriterClient client(writer, geometry, Window);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Buffer, &target), common::ResponseCode::Success);

    std::atomic<unsigned> done{0};
    unsigned issued = 0;
    unsigned refusals = 0;

    for (unsigned round = 0; round < Rounds; ++round)
    {
        StagingBuffer buffer;
        ASSERT_EQ(client.take(0, buffer), common::ResponseCode::Success) << "round " << round;

        if (!buffer.valid())
        {
            // Legitimate only when every buffer this caller took is still in flight. `done` counts
            // the copies REPORTED complete, so issued - done is what a worker believes is outstanding
            // - and what it sizes its window against.
            ++refusals;
            ASSERT_EQ(issued - done.load(), Window)
                << "refused with only " << (issued - done.load()) << " of " << Window
                << " copies outstanding: a buffer was reported done before it came back, round " << round;

            // Let the copies retire and try again, as a worker does by waiting for completions.
            for (unsigned i = 0; i < 10000 && done.load() < issued; ++i)
            {
                ::usleep(100);
            }
            ASSERT_EQ(client.take(0, buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid()) << "still empty after every copy retired, round " << round;
        }

        ASSERT_LE(client.buffers(), Window) << "the pool grew past its window at round " << round;

        ++issued;
        ASSERT_EQ(client.write(0, buffer, Buffer, target,
                               [&](common::ResponseCode ret)
                               {
                                   EXPECT_EQ(ret, common::ResponseCode::Success);
                                   ++done;
                               }),
                  common::ResponseCode::Success) << "round " << round;
    }

    for (unsigned i = 0; i < 100000 && done.load() < issued; ++i)
    {
        ::usleep(100);
    }
    EXPECT_EQ(done.load(), issued);

    EXPECT_LE(client.buffers(), Window);
    EXPECT_LE(client.events(0), Window) << "one event per in-flight copy, so the same ceiling";

    std::cerr << Rounds << " copies through a window of " << Window
              << ": pool " << client.buffers() << " buffers, " << client.events(0) << " events, "
              << refusals << " waits for a free buffer\n";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

// tmpfs, which is the one filesystem that never reaches the asynchronous reader: it is memory
// backed, so FsAsyncRouter leaves it ungrouped and its workloads go to the SYNCHRONOUS pool. That
// pool reads with pread, which cannot target device memory - so these bytes have to go through
// pinned buffers and a copy, and this is the only test that proves they do.
//
// A /dev/shm model cache is a common deployment, and before this path existed a device submission
// from one failed outright.
TEST_F(RealDevice, A_Tmpfs_Submission_Reads_Into_Device_Memory)
{
    // Tested rather than assumed: /dev/shm is tmpfs nearly everywhere, but a container can mount
    // anything there, and reading from the wrong filesystem would quietly test the async path again.
    if (::system("test \"$(stat -f -c %T /dev/shm)\" = tmpfs") != 0)
    {
        GTEST_SKIP() << "/dev/shm is not tmpfs here, so there is no memory-backed mount to read from";
    }

    constexpr size_t Total = 4 * Buffer;
    constexpr unsigned Ranges = 8;

    const auto data = utils::random::buffer(Total);
    utils::temp::File file("/dev/shm", utils::random::string(), data);

    void * target = nullptr;
    ASSERT_EQ(_device->device_alloc(Total, &target), common::ResponseCode::Success);

    Streamer streamer;

    std::vector<FileRanges> request(1);
    request[0].path = file.path;
    for (unsigned i = 0; i < Ranges; ++i)
    {
        request[0].ranges.push_back(
            ReadRange{ i * (Total / Ranges), Total / Ranges,
                       static_cast<char *>(target) + i * (Total / Ranges) });
    }

    SubmissionId submission_id = 0;
    ASSERT_EQ(streamer.async_request(request, common::Device::cuda(0), &submission_id),
              common::ResponseCode::Success);

    for (unsigned i = 0; i < Ranges; ++i)
    {
        bool done = false;
        const auto response = streamer.response(30000, done);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "only " << i << " ranges answered";
        EXPECT_EQ(response.ret, common::ResponseCode::Success) << "range " << i;
    }

    // What makes this test about the synchronous reader rather than a second copy of the async one.
    EXPECT_FALSE(streamer.async_pool_used())
        << "a memory-backed mount must not reach the asynchronous reader";

    const auto landed = read_back(target, Total);
    EXPECT_EQ(std::memcmp(landed.data(), data.data(), Total), 0)
        << "the bytes on the device are not the bytes in the file";

    EXPECT_EQ(_device->device_free(target), common::ResponseCode::Success);
}

} // namespace runai::llm::streamer::impl
