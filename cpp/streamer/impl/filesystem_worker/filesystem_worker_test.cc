#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <vector>

#include "common/device/device.h"

namespace runai::llm::streamer::impl
{

// An empty workload reads nothing, which is enough to check that the pool's per-worker routine drives
// this the same way its stateless handler used to: execute once, never wait for a drain.
TEST(FileSystemWorker, Executes_And_Is_Always_Idle)
{
    FileSystemWorker worker;
    std::atomic<bool> stopped{false};

    EXPECT_TRUE(worker.idle());

    worker.execute(Workload{}, stopped);
    EXPECT_TRUE(worker.idle()) << "a synchronous read is done when execute returns";

    worker.drain(stopped);
    EXPECT_TRUE(worker.idle());
}

// A device destination would be written with pread - a segmentation fault, not an error - so it is
// refused here until this reader stages through pinned memory.
TEST(FileSystemWorker, Refuses_A_Device_Workload)
{
    auto config = std::make_shared<Config>(false /* do not force minimum */);
    auto responder = std::make_shared<common::Responder>(0, common::QueueMode::PERSISTENT);
    common::s3::S3ClientWrapper::Params params;

    std::vector<char> destination(64);
    auto request = std::make_shared<Request>(0 /* file offset */, 0 /* file index */, 0 /* index */,
                                             1 /* tasks */, destination.size(), destination.data());
    responder->increment(1);

    Tasks tasks;
    tasks.emplace_back(request, 0, destination.size(), 0);

    Workload workload;
    ASSERT_EQ(workload.add_batch(Batch(1, 0, 0, "/tmp/file", params, std::move(tasks), responder,
                                       config, 4096, common::Device::cuda(0))),
              common::ResponseCode::Success);

    FileSystemWorker worker;
    std::atomic<bool> stopped{false};
    worker.execute(std::move(workload), stopped);

    const auto response = responder->pop(5000);
    EXPECT_EQ(response.ret, common::ResponseCode::UnsupportedDeviceType)
        << "the range must be answered, not left to hang";
}

} // namespace runai::llm::streamer::impl
