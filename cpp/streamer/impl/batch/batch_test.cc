#include "streamer/impl/batch/batch.h"
#include "common/device/device.h"

#include <gtest/gtest.h>
#include <utility>
#include <cstring>
#include <memory>
#include <chrono>
#include <set>

#include "device/mock/mock_device.h"
#include "utils/logging/logging.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"
#include "utils/thread/thread.h"
#include "utils/dylib/dylib.h"
#include "utils/scope_guard/scope_guard.h"

#include "common/s3_wrapper/s3_wrapper.h"

#include "streamer/impl/file/file.h"
#include "streamer/impl/workload/workload.h"
namespace runai::llm::streamer::impl
{

TEST(Batch, Finished_Until)
{
    unsigned num_tasks = utils::random::number(1, 10);
    const auto path = utils::random::string();
    common::s3::S3ClientWrapper::Params params;

    // File range to read
    auto start = utils::random::number<size_t>(0, 1024);
    auto size = utils::random::number<size_t>(num_tasks, 1024 * 1024);
    EXPECT_LT(num_tasks, size);

    // divide range into chunks - a chunk per task
    auto chunks = utils::random::chunks(size, num_tasks);

    auto responder = std::make_shared<common::Responder>(1);
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), num_tasks, size, nullptr);

    // create tasks

    size_t offset = start;
    Tasks tasks;

    for (unsigned i = 0; i < num_tasks; ++i)
    {
        auto task = Task(request, offset, chunks[i], utils::random::number<size_t>());
        offset += chunks[i];
        tasks.push_back(std::move(task));
    }

    // create batch
    const auto config = std::make_shared<Config>();

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    // execute part of the tasks

    auto mid_point = utils::random::number<size_t>(start, start + size);
    unsigned expected = 0;
    size_t total = start;
    while (total < mid_point)
    {
        total += chunks[expected];
        ++expected;
    }
    expected = (total > mid_point ? expected - 1 : expected);

    batch.finished_until(mid_point);

    EXPECT_EQ(batch.finished_until(), expected);

    EXPECT_FALSE(batch.responder->finished());

    // execute rest of the tasks

    batch.finished_until(start + size);

    EXPECT_EQ(batch.finished_until(), num_tasks);

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::Success);
}

TEST(Read, Sanity)
{
    unsigned num_tasks = utils::random::number(1, 10);

    // File range to read
    const auto start = utils::random::number<size_t>(0, 1024);
    const auto size = utils::random::number<size_t>(num_tasks, 1024 * 1024);
    EXPECT_LT(num_tasks, size);

    const auto data = utils::random::buffer(start + size);
    utils::temp::File file(data);
    const auto path = file.path;
    common::s3::S3ClientWrapper::Params params;

    // divide range into chunks - a chunk per task
    auto chunks = utils::random::chunks(size, num_tasks);

    auto responder = std::make_shared<common::Responder>(1);

    const auto chunk_bytesize = utils::random::number<size_t>(1, size);
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), 1 /* s3 concurrency */,
                                                 chunk_bytesize /* s3 block */,
                                                 utils::random::number<size_t>(1, chunk_bytesize) /* fs read block */,
                                                 false /* do not force the minimum */);

    std::vector<char> dst(size);
    auto dst_ptr = dst.data();

    // create tasks
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), num_tasks, size, dst_ptr);

    size_t offset = start;
    size_t relative_offset = 0;

    Tasks tasks;
    for (unsigned i = 0; i < num_tasks; ++i)
    {
        // task offset is relative to the beginning of the request offset
        auto task = Task(request, offset, chunks[i], relative_offset);
        offset += chunks[i];
        relative_offset += chunks[i];
        tasks.push_back(std::move(task));
    }

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped));

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::Success);

    // verify read data
    bool mismatch = false;
    for (size_t i = 0; i < size && !mismatch; ++i)
    {
        mismatch = dst[i] != static_cast<char>(data[start + i]);
    }
    EXPECT_FALSE(mismatch);
}

// A batch whose range is empty (start == end) reads nothing: neither the block loop nor the tail read
// in Batch::read runs, because num_chunks is 0 and file_offset == range.end. Its tasks must still be
// notified - a zero sized range owes exactly one response, like any other range - otherwise the
// submission waits forever for a response that never comes.
TEST(Read, Empty_Range)
{
    const auto start = utils::random::number<size_t>(0, 1024);

    // the file must exist and be seekable to start; its contents are never read
    const auto data = utils::random::buffer(start + 1);
    utils::temp::File file(data);
    const auto path = file.path;
    common::s3::S3ClientWrapper::Params params;

    auto responder = std::make_shared<common::Responder>(1);
    const auto config = std::make_shared<Config>();

    const auto file_index = utils::random::number();
    const auto range_index = utils::random::number();

    // a single zero sized range: one task, no bytes
    auto request = std::make_shared<Request>(start, file_index, range_index, 1 /* tasks */, 0 /* bytesize */, nullptr);

    Tasks tasks;
    tasks.push_back(Task(request, start, 0 /* size */, 0 /* destination offset */));

    Batch batch(utils::random::number(), utils::random::number(), file_index, path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    EXPECT_EQ(batch.total_bytes(), 0);

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped));

    // The response must arrive even though nothing was read. Timed rather than blocking so that a
    // regression fails the test instead of hanging it.
    auto r = batch.responder->pop(5000);
    EXPECT_EQ(r.ret, common::ResponseCode::Success);
    EXPECT_EQ(r.file_index, file_index);
    EXPECT_EQ(r.index, range_index);
}

TEST(Read, Error)
{
    std::string path;
    unsigned num_tasks = utils::random::number(2, 10); // need at least two tasks in this test

    // File range to read
    auto start = utils::random::number<size_t>(0, 1024);
    auto size = utils::random::number<size_t>(num_tasks, 1024 * 1024);
    EXPECT_LT(num_tasks, size);

    const auto data = utils::random::buffer(start + size - utils::random::number<size_t>(1, size));
    utils::temp::File file(data);
    path = file.path;
    common::s3::S3ClientWrapper::Params params;

    // divide range into chunks - a chunk per task
    auto chunks = utils::random::chunks(size, num_tasks);

    auto responder = std::make_shared<common::Responder>(1);

    const auto config = std::make_shared<Config>();

    std::vector<char> dst(size);
    auto dst_ptr = dst.data();

    // create tasks
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), num_tasks, size, dst_ptr);

    size_t offset = start;

    Tasks tasks;
    size_t relative_offset = 0;

    for (unsigned i = 0; i < num_tasks; ++i)
    {
        // task offset is relative to the beginning of the request offset
        auto task = Task(request, offset, chunks[i], relative_offset);
        offset += chunks[i];
        relative_offset += chunks[i];
        tasks.push_back(std::move(task));
    }

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped));

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::EofError);
}

// A read that throws must give its staging buffer back. Nothing else will: the buffer was never
// submitted, so no completion returns it, and the pool is three deep and lives as long as the
// reading thread. Three failed batches and the fourth waits for a buffer that never comes.
TEST(Read, A_Failed_Read_Returns_Its_Staging_Buffer)
{
    constexpr size_t Block = 4096;
    constexpr unsigned Blocks = 4;

    // Short of the range: the first two blocks read, the third hits the end of the file.
    const auto data = utils::random::buffer(2 * Block + Block / 2);
    utils::temp::File file(data);
    common::s3::S3ClientWrapper::Params params;

    auto responder = std::make_shared<common::Responder>(1);
    const auto config = std::make_shared<Config>(false /* do not force minimum */);

    std::vector<char> dst(Blocks * Block, 0);
    auto request = std::make_shared<Request>(0 /* file offset */, 0 /* file */, 0 /* index */,
                                             Blocks /* tasks */, dst.size(), dst.data());

    Tasks tasks;
    for (unsigned i = 0; i < Blocks; ++i)
    {
        tasks.emplace_back(request, i * Block, Block, i * Block);
    }

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend]() { return backend; });
    auto issuer = std::make_shared<DeviceIssuer>(writer, 3 /* copies in flight */);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(0, channel), common::ResponseCode::Success);

    StagingPool::Params pool_params;
    pool_params.buffer_bytesize = Block;
    pool_params.slab_bytesize = Block;
    pool_params.max_buffers = 3;

    DeviceStaging staging;
    staging.pool = std::make_shared<StagingPool>(writer->device(channel), pool_params);
    staging.issuer = issuer.get();

    Batch batch(1 /* submission */, 0, 0, file.path, params, std::move(tasks), responder, config,
                Block, common::Device::cuda(0));

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped, &staging));

    EXPECT_EQ(responder->pop().ret, common::ResponseCode::EofError);

    // Every buffer is back: the two that were copied, and the one the failed read was holding.
    for (unsigned i = 0; i < pool_params.max_buffers; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(staging.pool->try_acquire(buffer), common::ResponseCode::Success);
        EXPECT_TRUE(buffer.valid()) << "buffer " << i << " of the pool was lost by the failed read";
    }
}

// One copy failing PART WAY through is not the same case as all of them failing. The blocks that
// landed before it really are on the device and were already answered; the failing block and
// everything after it must not be. So the Success answers are a prefix, and it stops at the copy
// that failed - which is what checking `failure` before advancing the progress counter buys.
TEST(Read, A_Copy_That_Fails_Part_Way_Stops_The_Answers_There)
{
    constexpr size_t Block = 4096;
    constexpr unsigned Blocks = 8;
    constexpr unsigned FailFrom = 5;   // the fifth copy onwards, counting from 1

    const auto data = utils::random::buffer(Blocks * Block);
    utils::temp::File file(data);
    common::s3::S3ClientWrapper::Params params;

    // One response per range, all of them real: a responder that expects fewer answers the rest with
    // its own end-of-stream marker, which reads exactly like a failed range and hides both.
    auto responder = std::make_shared<common::Responder>(0, common::QueueMode::PERSISTENT);
    responder->increment(Blocks);

    const auto config = std::make_shared<Config>(false /* do not force minimum */);

    std::vector<char> dst(Blocks * Block, 0);

    // One range per block, so a response answers exactly one copy.
    Tasks tasks;
    std::vector<std::shared_ptr<Request>> requests;
    for (unsigned i = 0; i < Blocks; ++i)
    {
        auto request = std::make_shared<Request>(i * Block, 0 /* file */, i /* index */,
                                                 1 /* tasks */, Block, dst.data() + i * Block);
        requests.push_back(request);
        tasks.emplace_back(request, i * Block, Block, 0);
    }

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend]() { return backend; });
    auto issuer = std::make_shared<DeviceIssuer>(writer, 3 /* copies in flight */);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(0, channel), common::ResponseCode::Success);
    backend->opened(0)->fail_event_synchronize_from = FailFrom;

    StagingPool::Params pool_params;
    pool_params.buffer_bytesize = Block;
    pool_params.slab_bytesize = Block;
    pool_params.max_buffers = 3;

    DeviceStaging staging;
    staging.pool = std::make_shared<StagingPool>(writer->device(channel), pool_params);
    staging.issuer = issuer.get();

    Batch batch(1 /* submission */, 0, 0, file.path, params, std::move(tasks), responder, config,
                Block, common::Device::cuda(0));

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped, &staging));

    std::vector<common::ResponseCode> answers;
    for (unsigned i = 0; i < Blocks; ++i)
    {
        const auto response = responder->pop(5000);
        ASSERT_NE(response.ret, common::ResponseCode::TimedOut) << "range " << i << " was never answered";
        answers.push_back(response.ret);
    }

    // Ranges are answered in file order, so the Success answers run from the front and stop.
    unsigned succeeded = 0;
    while (succeeded < Blocks && answers[succeeded] == common::ResponseCode::Success)
    {
        ++succeeded;
    }

    for (unsigned i = succeeded; i < Blocks; ++i)
    {
        EXPECT_NE(answers[i], common::ResponseCode::Success)
            << "range " << i << " was answered as read after an earlier copy had already failed";
    }

    EXPECT_LE(succeeded, FailFrom - 1)
        << "the copy that failed, or one after it, was answered as read";

    // Whatever was answered Success really did land. The reader is free to have answered fewer,
    // since it only answers what has already completed.
    EXPECT_EQ(std::memcmp(dst.data(), data.data(), succeeded * Block), 0)
        << "a range answered as read does not hold the file's bytes";
}

// A stopped pool hands out nothing, so the rest of the range is never read. Answering it Success
// would report bytes that never left the file, let alone reached the device.
TEST(Read, A_Stopped_Pool_Is_Not_Answered_As_Read)
{
    constexpr size_t Block = 4096;
    constexpr unsigned Blocks = 4;

    const auto data = utils::random::buffer(Blocks * Block);
    utils::temp::File file(data);
    common::s3::S3ClientWrapper::Params params;

    auto responder = std::make_shared<common::Responder>(1);
    const auto config = std::make_shared<Config>(false /* do not force minimum */);

    std::vector<char> dst(Blocks * Block, 0);
    auto request = std::make_shared<Request>(0 /* file offset */, 0 /* file */, 0 /* index */,
                                             Blocks /* tasks */, dst.size(), dst.data());

    Tasks tasks;
    for (unsigned i = 0; i < Blocks; ++i)
    {
        tasks.emplace_back(request, i * Block, Block, i * Block);
    }

    auto backend = std::make_shared<device::MockBackend>();
    auto writer = std::make_shared<DeviceWriter>([backend]() { return backend; });
    auto issuer = std::make_shared<DeviceIssuer>(writer, 3 /* copies in flight */);

    DeviceWriter::Channel channel = nullptr;
    ASSERT_EQ(writer->open(0, channel), common::ResponseCode::Success);

    StagingPool::Params pool_params;
    pool_params.buffer_bytesize = Block;
    pool_params.slab_bytesize = Block;
    pool_params.max_buffers = 3;

    DeviceStaging staging;
    staging.pool = std::make_shared<StagingPool>(writer->device(channel), pool_params);
    staging.issuer = issuer.get();

    // Stopped before the first acquire, which is the state a teardown leaves the pool in.
    staging.pool->stop();

    Batch batch(1 /* submission */, 0, 0, file.path, params, std::move(tasks), responder, config,
                Block, common::Device::cuda(0));

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped, &staging));

    EXPECT_EQ(responder->pop().ret, common::ResponseCode::FinishedError)
        << "the range was answered as read, but nothing was ever read into it";

    EXPECT_EQ(backend->opened(0)->copies, 0u);
}

TEST(Read, Already_Stopped)
{
    unsigned num_tasks = utils::random::number(1, 10);

    // File range to read
    auto start = utils::random::number<size_t>(0, 1024);
    auto size = utils::random::number<size_t>(num_tasks, 1024 * 1024);
    EXPECT_LT(num_tasks, size);

    const auto data = utils::random::buffer(start + size);
    utils::temp::File file(data);
    const auto path = file.path;
    common::s3::S3ClientWrapper::Params params;

    // divide range into chunks - a chunk per task
    auto chunks = utils::random::chunks(size, num_tasks);

    auto responder = std::make_shared<common::Responder>(1);

    const auto chunk_bytesize = utils::random::number<size_t>(1, size);
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), 1 /* s3 concurrency */,
                                                 chunk_bytesize /* s3 block */,
                                                 utils::random::number<size_t>(1, chunk_bytesize) /* fs read block */,
                                                 false /* do not force the minimum */);

    std::vector<char> dst(size);
    auto dst_ptr = dst.data();

    // create tasks
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), num_tasks, size, dst_ptr);

    size_t offset = start;
    size_t relative_offset = 0;

    Tasks tasks;
    for (unsigned i = 0; i < num_tasks; ++i)
    {
        // task offset is relative to the beginning of the request offset
        auto task = Task(request, offset, chunks[i], relative_offset);
        offset += chunks[i];
        relative_offset += chunks[i];
        tasks.push_back(std::move(task));
    }

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    std::atomic<bool> stopped(true);
    EXPECT_NO_THROW(batch.execute(stopped));

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);

    // verify data not read
    bool mismatch = false;
    for (size_t i = 0; i < size && !mismatch; ++i)
    {
        mismatch = dst[i] != static_cast<char>(data[start + i]);
    }
    EXPECT_TRUE(mismatch);
}

TEST(Read, Stopped_During_Read)
{
    unsigned num_requests = utils::random::number(1, 10);

    // File range to read
    const auto start = utils::random::number<size_t>(0, 1024);
    const auto size = utils::random::number<size_t>(512 * 1024, 1024 * 1024);
    EXPECT_LT(num_requests, size);

    const auto data = utils::random::buffer(start + size);
    utils::temp::File file(data);
    const auto path = file.path;
    common::s3::S3ClientWrapper::Params params;

    // divide range into chunks - a chunk per request

    const auto chunks = utils::random::chunks(size, num_requests);

    auto responder = std::make_shared<common::Responder>(num_requests);

    const auto chunk_bytesize = utils::random::number<size_t>(1, size);
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), 1 /* s3 concurrency */,
                                                 chunk_bytesize /* s3 block */,
                                                 utils::random::number<size_t>(1, chunk_bytesize) /* fs read block */,
                                                 false /* do not force the minimum */);

    std::vector<char> dst(size);
    auto dst_ptr = dst.data();

    // create task for each request
    Tasks tasks;
    std::vector<std::shared_ptr<Request>> requests(num_requests);
    std::vector<size_t> offsets;
    auto offset = start;
    auto request_offset = dst_ptr;
    for (unsigned i = 0; i < num_requests; ++i)
    {
        requests[i] = std::make_shared<Request>(offset, utils::random::number(), i, 1, chunks[i], request_offset);
        request_offset += chunks[i];
        EXPECT_EQ(requests[i]->bytesize, chunks[i]);
        EXPECT_EQ(requests[i]->offset, offset);

        auto task = Task(requests[i], offset, chunks[i], 0);
        tasks.push_back(std::move(task));

        offsets.push_back(offset);
        offset += chunks[i];
    }

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    std::atomic<bool> stopped(false);

    auto thread = utils::Thread([&]()
    {
        EXPECT_NO_THROW(batch.execute(stopped));
    });

    ::usleep(utils::random::number(300));
    stopped = true;

    // collect responses
    std::vector<common::Response> responses;
    std::set<unsigned> responded_requests;
    for (unsigned i = 0; i < num_requests; ++i)
    {
        auto r = batch.responder->pop();
        responded_requests.insert(r.index);
        responses.push_back(r);
    }

    EXPECT_EQ(responded_requests.size(), num_requests);

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);

    // verify that all responses were sent
    for (const auto & r : responses)
    {
        EXPECT_LT(r.index, num_requests);

        bool mismatch = false;
        const auto j_start = offsets[r.index]; // request offset is the file offset
        const auto j_end = j_start + chunks[r.index];
        for (size_t j = j_start; j < j_end; ++j)
        {
            char dst_ = dst[j - start];
            char data_ =  static_cast<char>(data[j]);
            mismatch = (data_ != dst_);
            if (mismatch)
            {
                break;
            }
        }

        if (r.ret == common::ResponseCode::Success)
        {
            // verify read data
            EXPECT_FALSE(mismatch);
        }
        else
        {
            EXPECT_EQ(r.ret, common::ResponseCode::FinishedError);
            // verify unread data
            EXPECT_TRUE(mismatch);
        }
    }
}

// handle_error on a batch that ALREADY completed must produce no further response.
//
// Reached in practice: ObjectStorageWorker::report_workload records errors per FILE index, and one file now
// contributes several batches (one per contiguous transfer) - so when one transfer fails and another
// succeeds, handle_error is called on the successful one too. A second response for a range that already
// answered would overrun the submission's expected count.
//
// Nothing in Batch prevents it. Task::_finished and Request::finished's exact-count check do, independently,
// so this fails only when both are lost. It pins the contract - one response per range, ever - not either
// mechanism.
TEST(Batch, Handle_Error_After_Completion_Is_Silent)
{
    const auto start = utils::random::number<size_t>(0, 1024);
    const auto size = utils::random::number<size_t>(1, 1024);
    const auto data = utils::random::buffer(start + size);
    utils::temp::File file(data);
    common::s3::S3ClientWrapper::Params params;

    // ONE range, so exactly one response is owed - the whole subject of the test. PERSISTENT for the reason
    // the streamer uses it: a drained FINISH_ON_DRAIN responder answers FinishedError, hiding whether
    // anything was pushed; a drained persistent one has nothing to give, so the second pop times out.
    auto responder = std::make_shared<common::Responder>(1, common::QueueMode::PERSISTENT);

    const auto chunk_bytesize = utils::random::number<size_t>(1, size);
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), 1 /* s3 concurrency */,
                                                 chunk_bytesize /* s3 block */,
                                                 utils::random::number<size_t>(1, chunk_bytesize) /* fs read block */,
                                                 false /* do not force the minimum */);

    std::vector<char> dst(size);
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), 1, size, dst.data());

    Tasks tasks;
    tasks.push_back(Task(request, start, size, 0));

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), file.path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize, common::Device::host());

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped));

    EXPECT_EQ(responder->pop().ret, common::ResponseCode::Success);

    // the successful batch is failed anyway, as report_workload would do
    EXPECT_NO_THROW(batch.handle_error(common::ResponseCode::FileAccessError));

    // TimedOut, not a second response: nothing more was pushed
    EXPECT_EQ(responder->pop(200).ret, common::ResponseCode::TimedOut);

    // and no push beyond the one expected: with the count at zero an extra push is REJECTED rather than
    // queued, so the timeout above would not see it - only this flag does
    EXPECT_EQ(responder->valid(), common::ResponseCode::Success);
}

}; // namespace runai::llm::streamer::impl
