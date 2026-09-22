#include "streamer/impl/filesystem_worker/filesystem_worker.h"

#include <gtest/gtest.h>

#include <atomic>

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

} // namespace runai::llm::streamer::impl
