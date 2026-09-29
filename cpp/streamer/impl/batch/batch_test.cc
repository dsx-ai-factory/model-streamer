/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/batch/batch.h"

#include <gtest/gtest.h>
#include <utility>
#include <memory>
#include <chrono>
#include <set>

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), chunk_bytesize, utils::random::number<size_t>(1, chunk_bytesize), false /* do not force minimum chunk size */);

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

    std::atomic<bool> stopped(false);
    EXPECT_NO_THROW(batch.execute(stopped));

    auto r = batch.responder->pop();
    EXPECT_EQ(r.ret, common::ResponseCode::EofError);
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
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), chunk_bytesize, utils::random::number<size_t>(1, chunk_bytesize), false /* do not force minimum chunk size */);

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), chunk_bytesize, utils::random::number<size_t>(1, chunk_bytesize), false /* do not force minimum chunk size */);

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
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
    const auto config = std::make_shared<Config>(utils::random::number(1, 4), chunk_bytesize, utils::random::number<size_t>(1, chunk_bytesize), false /* do not force minimum chunk size */);

    std::vector<char> dst(size);
    auto request = std::make_shared<Request>(start, utils::random::number(), utils::random::number(), 1, size, dst.data());

    Tasks tasks;
    tasks.push_back(Task(request, start, size, 0));

    Batch batch(utils::random::number(), utils::random::number(), utils::random::number(), file.path, params, std::move(tasks), responder, config,
                params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize);

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
