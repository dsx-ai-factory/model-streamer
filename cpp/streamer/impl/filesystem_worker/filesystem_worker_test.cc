/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

#include "common/device/device.h"
#include "device/mock/mock_device.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Block = 4096;

class FileSystemWorkerTest : public ::testing::Test
{
 protected:
    // One file, one contiguous transfer, `ranges` ranges - the layout a model read produces.
    Workload workload_of(const std::string & path, common::Device device, void * destination,
                         unsigned ranges, size_t range_bytesize)
    {
        _responder->increment(ranges);

        Tasks tasks;
        for (unsigned i = 0; i < ranges; ++i)
        {
            auto request = std::make_shared<Request>(i * range_bytesize, 0 /* file */, i /* index */,
                                                     1 /* tasks */, range_bytesize,
                                                     static_cast<char *>(destination) + i * range_bytesize);
            _requests.push_back(request);
            tasks.emplace_back(request, i * range_bytesize, range_bytesize, 0);
        }

        common::s3::S3ClientWrapper::Params params;
        Workload workload;
        EXPECT_EQ(workload.add_batch(Batch(1 /* submission */, 0, 0, path, params, std::move(tasks),
                                           _responder, _config, Block, device)),
                  common::ResponseCode::Success);
        return workload;
    }

    std::shared_ptr<Config> _config = std::make_shared<Config>(false /* do not force minimum */);
    std::shared_ptr<common::Responder> _responder =
        std::make_shared<common::Responder>(0, common::QueueMode::PERSISTENT);
    std::vector<std::shared_ptr<Request>> _requests;

    std::shared_ptr<device::MockBackend> _backend = std::make_shared<device::MockBackend>();
};

} // namespace

TEST_F(FileSystemWorkerTest, Executes_And_Is_Always_Idle)
{
    FileSystemWorker worker;
    std::atomic<bool> stopped{false};

    EXPECT_TRUE(worker.idle());

    worker.execute(Workload{}, stopped);
    EXPECT_TRUE(worker.idle()) << "a synchronous read is done when execute returns";

    worker.drain(stopped);
    EXPECT_TRUE(worker.idle());
}

// A device destination would be written with pread - a segmentation fault, not an error - so a
// reader with no copy path refuses it rather than trying. The ranges are still answered.
TEST_F(FileSystemWorkerTest, Refuses_A_Device_Workload_Without_A_Copy_Path)
{
    std::vector<char> destination(Block);
    auto workload = workload_of("/tmp/does-not-matter", common::Device::cuda(0), destination.data(), 1, Block);

    FileSystemWorker worker;   // no writer, no issuer
    std::atomic<bool> stopped{false};
    worker.execute(std::move(workload), stopped);

    const auto response = _responder->pop(5000);
    EXPECT_EQ(response.ret, common::ResponseCode::DeviceUnavailable)
        << "the range must be answered, not left to hang";
}

// A buffer of no bytes reads no bytes, so the batch would never advance: it would take a buffer per
// turn and wait for the fourth forever. Refused instead, while the ranges can still be answered.
TEST_F(FileSystemWorkerTest, Refuses_A_Device_Workload_With_No_Block_Size)
{
    const auto data = utils::random::buffer(Block);
    utils::temp::File file(data);

    std::vector<char> destination(Block);
    auto workload = workload_of(file.path, common::Device::cuda(0), destination.data(), 1, Block);

    auto writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { return _backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    FileSystemWorker worker(writer, issuer, 0 /* block */);
    std::atomic<bool> stopped{false};
    worker.execute(std::move(workload), stopped);

    const auto response = _responder->pop(5000);
    EXPECT_EQ(response.ret, common::ResponseCode::InvalidParameterError)
        << "the reader hung instead of refusing";
}

// THE step: a device workload is read into this thread's pinned buffers, copied from there, and its
// ranges answered only once the bytes have landed.
TEST_F(FileSystemWorkerTest, Reads_A_Device_Workload_Through_Pinned_Buffers)
{
    constexpr unsigned Ranges = 8;

    const auto data = utils::random::buffer(Ranges * Block);
    utils::temp::File file(data);

    // The mock's "device memory" is ordinary host memory, so the bytes are checkable.
    std::vector<char> destination(Ranges * Block, 0);

    auto workload = workload_of(file.path, common::Device::cuda(0), destination.data(), Ranges, Block);

    auto writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { return _backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    FileSystemWorker worker(writer, issuer, Block);
    std::atomic<bool> stopped{false};
    worker.execute(std::move(workload), stopped);

    for (unsigned i = 0; i < Ranges; ++i)
    {
        const auto response = _responder->pop(5000);
        EXPECT_EQ(response.ret, common::ResponseCode::Success) << "range " << i;
    }

    EXPECT_EQ(std::memcmp(destination.data(), data.data(), data.size()), 0)
        << "the bytes went through the staging buffers and arrived in order";

    const auto device = _backend->opened(0);
    ASSERT_NE(device, nullptr);
    EXPECT_EQ(device->copies, Ranges) << "one block, one buffer, one copy";

    // AT MOST three buffers, whatever the batch size. A pool that grew with the batch would pin the
    // whole file. Not exactly three: the pool grows only when no buffer is free, so a copy that
    // lands before the next block is read means the third is never needed.
    EXPECT_LE(device->host_allocs, FileSystemWorker::BuffersPerThread)
        << "the pool grew past the pipeline depth";
    EXPECT_GE(device->host_allocs, 1u);
}

// Every reading thread has its own pool, so a second worker pins its own three and shares nothing
// but the issuer and the stream.
TEST_F(FileSystemWorkerTest, Each_Worker_Has_Its_Own_Buffers)
{
    const auto data = utils::random::buffer(4 * Block);
    utils::temp::File file(data);

    auto writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { return _backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    std::vector<std::vector<char>> destinations(2, std::vector<char>(4 * Block, 0));
    std::atomic<bool> stopped{false};

    // Counted per worker, because how MANY buffers a worker pins depends on how fast the copies come
    // back. What must be true is that the second worker pins its OWN: sharing the first one's pool
    // would let it reuse buffers that are all free by now, and register nothing at all.
    std::vector<unsigned> pinned_after(2, 0);

    for (unsigned w = 0; w < 2; ++w)
    {
        auto workload = workload_of(file.path, common::Device::cuda(0), destinations[w].data(), 4, Block);
        FileSystemWorker worker(writer, issuer, Block);
        worker.execute(std::move(workload), stopped);

        for (unsigned i = 0; i < 4; ++i)
        {
            EXPECT_EQ(_responder->pop(5000).ret, common::ResponseCode::Success) << "worker " << w;
        }

        pinned_after[w] = _backend->opened(0)->host_allocs;
    }

    const auto device = _backend->opened(0);
    EXPECT_EQ(device->streams_created, 1u) << "one device, one stream, one issuer";

    EXPECT_GT(pinned_after[1], pinned_after[0]) << "the second worker read out of the first one's pool";
    EXPECT_LE(pinned_after[0], FileSystemWorker::BuffersPerThread);
    EXPECT_LE(pinned_after[1] - pinned_after[0], FileSystemWorker::BuffersPerThread);

    for (unsigned w = 0; w < 2; ++w)
    {
        EXPECT_EQ(std::memcmp(destinations[w].data(), data.data(), data.size()), 0) << "worker " << w;
    }
}

// A copy that never lands must not be reported as read. The progress counter the reader answers
// ranges from advances per COMPLETION, so counting a failed one would answer those ranges Success
// with nothing on the device - silent data loss, and the failure mode a byte check never sees
// because it is the response codes that lie.
TEST_F(FileSystemWorkerTest, A_Failed_Copy_Is_Never_Answered_As_Read)
{
    constexpr unsigned Ranges = 8;

    const auto data = utils::random::buffer(Ranges * Block);
    utils::temp::File file(data);
    std::vector<char> destination(Ranges * Block, 0);

    auto workload = workload_of(file.path, common::Device::cuda(0), destination.data(), Ranges, Block);

    auto writer = std::make_shared<DeviceWriter>([this](common::DeviceType) -> DeviceWriter::BackendFactory { return [this]() { return _backend; }; });
    auto issuer = std::make_shared<DeviceIssuer>(writer);

    // Every copy fails at the event, which is where a real transfer error surfaces.
    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(common::Device::cuda(0), channel), common::ResponseCode::Success);
    _backend->opened(0)->fail_event_synchronize = true;

    FileSystemWorker worker(writer, issuer, Block);
    std::atomic<bool> stopped{false};
    worker.execute(std::move(workload), stopped);

    for (unsigned i = 0; i < Ranges; ++i)
    {
        const auto response = _responder->pop(5000);
        EXPECT_NE(response.ret, common::ResponseCode::TimedOut) << "range " << i << " was never answered";
        EXPECT_NE(response.ret, common::ResponseCode::Success)
            << "range " << i << " was answered as read, but its copy never landed";
    }
}

} // namespace runai::llm::streamer::impl
