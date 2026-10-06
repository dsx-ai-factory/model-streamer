/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/shared_queue/shared_queue.h"

#include <gtest/gtest.h>
#include <set>
#include <utility>
#include <atomic>
#include <chrono>

#include <unistd.h>

#include "utils/random/random.h"
#include "utils/threadpool/threadpool.h"
#include "utils/thread/thread.h"
#include "utils/semaphore/semaphore.h"

#include "common/response/response.h"

namespace runai::llm::streamer::common
{

TEST(Creation, Empty)
{
    auto responder = SharedQueue<Response>(0);

    auto times = utils::random::number(1, 10);
    for (unsigned i = 0; i < times; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::FinishedError);
    }
}

TEST(Destruction, Sanity)
{
    auto size = utils::random::number(1, 100);
    auto responder = SharedQueue<Response>(size);

    for (unsigned i = 0; i < size; ++i)
    {
        responder.push(i);
    }

    // D'tor waits until all tasks are finished
}

TEST(Pop, Sanity)
{
    auto size = utils::random::number(1, 100);
    auto responder = SharedQueue<Response>(size);

    for (unsigned i = 0; i < size; ++i)
    {
        responder.push(i);
    }

    for (unsigned i = 0; i < size; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.index, i);
        EXPECT_EQ(r.ret, ResponseCode::Success);
    }

    auto times = utils::random::number(1, 10);
    for (unsigned i = 0; i < times; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::FinishedError);
    }
}

TEST(Pop, Wait)
{
    auto size = utils::random::number(1, 100);
    auto responder = SharedQueue<Response>(size);

    // create threadpool to push
    auto pool = utils::ThreadPool<unsigned>([&](unsigned i, std::atomic<bool> &)
    {
        responder.push(i);
    }, size);

    std::set<unsigned> expected;
    for (unsigned i = 0; i < size; ++i)
    {
        expected.insert(i);
        unsigned value = i;
        pool.push(std::move(value));
    }

    // wait for all the responses

    for (unsigned i = 0; i < size; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::Success);
        EXPECT_EQ(expected.count(r.index), 1);
        expected.erase(r.index);
    }

    EXPECT_EQ(expected.size(), 0);

    auto times = utils::random::number(1, 10);
    for (unsigned i = 0; i < times; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::FinishedError);
    }
}

TEST(Pop, Error)
{
    for (auto rc : {ResponseCode::FileAccessError, ResponseCode::EofError, ResponseCode::UnknownError} )
    {
        auto size = utils::random::number(1, 3);

        // create a responder for the expected responses

        auto responder = SharedQueue<Response>(size);

        // create threadpool to push
        auto pool = utils::ThreadPool<int>([&](int i, std::atomic<bool> &)
        {
            auto r = Response(rc);
            responder.push(std::move(r));
        }, size);

        for (unsigned i = 0; i < size; ++i)
        {
            unsigned value = i;
            pool.push(std::move(value));
        }

        // wait for all the responses

        for (unsigned i = 0; i < size; ++i)
        {
            auto r = responder.pop();
            EXPECT_EQ(r.ret, rc);
        }

        auto times = utils::random::number(1, 10);
        for (unsigned i = 0; i < times; ++i)
        {
            auto r = responder.pop();
            EXPECT_EQ(r.ret, ResponseCode::FinishedError);
        }
    }
}

TEST(Pop, Unexpected_Responses)
{
    auto size = utils::random::number(2, 100);
    const auto expected = utils::random::number(1, size-1);
    auto responder = SharedQueue<Response>(expected);

    // create threadpool to push

    std::atomic<unsigned> completed = 0;
    auto finished = utils::Semaphore(0);
    auto pool = utils::ThreadPool<unsigned>([&](unsigned i, std::atomic<bool> &)
    {
        responder.push(i);
        completed++;
        if (completed == size)
        {
            finished.post();
        }
    }, size);

    for (unsigned i = 0; i < size; ++i)
    {
        auto value = i;
        pool.push(std::move(value));
    }

    // wait until all the responses are pushed
    finished.wait();

    unsigned error_responses = 0;
    unsigned success_responses = 0;

    // wait for all the responses

    for (unsigned i = 0; i < size; ++i)
    {
        auto r = responder.pop();
        if (r.ret == ResponseCode::Success)
        {
            ++success_responses;
        }
        else
        {
            EXPECT_EQ(r.ret, ResponseCode::FinishedError);
            ++error_responses;
        }
    }

    EXPECT_EQ(success_responses, expected);
    EXPECT_EQ(error_responses, size - expected);
    EXPECT_EQ(responder.valid(), ResponseCode::UnknownError);
    auto times = utils::random::number(1, 10);
    for (unsigned i = 0; i < times; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::FinishedError);
    }
}

TEST(Stop, Sanity)
{
    auto size = utils::random::number(1, 100);
    auto responder = SharedQueue<Response>(size);

    // create a thread to wait
    auto waiting = utils::Thread([&]()
    {
        for (unsigned i = 0; i < size; ++i)
        {
            auto r = responder.pop();
            if (r == ResponseCode::FinishedError)
            {
                break;
            }
            EXPECT_EQ(r.ret, ResponseCode::Success);
        }
    });

    // create threadpool to push
    auto pool = utils::ThreadPool<unsigned>([&](unsigned i, std::atomic<bool> &)
    {
        responder.push(i);
    }, size);

    for (unsigned i = 0; i < size; ++i)
    {
        unsigned value = i;
        usleep(utils::random::number(100));
        pool.push(std::move(value));
    }

    // stop the responder
    usleep(utils::random::number(100 * 1000));
    responder.stop();

    auto times = utils::random::number(1, 10);
    for (unsigned i = 0; i < times; ++i)
    {
        auto r = responder.pop();
        EXPECT_EQ(r.ret, ResponseCode::FinishedError);
    }
}

// --- Behaviors that must hold in BOTH modes ---
// Each behavior is a helper run from one TEST per mode (kept as plain TEST/helper rather than
// TEST_P to avoid depending on the parameterized-test macro version), so both modes are always
// exercised and named explicitly.

namespace
{

void run_delivery(QueueMode mode)
{
    // push N then pop N: every response is delivered exactly once.
    const auto size = utils::random::number(1, 100);
    auto responder = SharedQueue<Response>(size, mode);

    for (unsigned i = 0; i < size; ++i)
    {
        responder.push(i);
    }

    std::set<unsigned> expected;
    for (unsigned i = 0; i < size; ++i) expected.insert(i);

    for (unsigned i = 0; i < size; ++i)
    {
        auto r = responder.pop(5000);
        EXPECT_EQ(r.ret, ResponseCode::Success);
        EXPECT_EQ(expected.erase(r.index), 1);
    }
    EXPECT_EQ(expected.size(), 0);
}

void run_multi_producer_single_consumer(QueueMode mode)
{
    // Multiple producers enqueue concurrently; a single consumer dequeues with a per-pop
    // timeout. All responses received, no loss, no deadlock.
    const unsigned total = utils::random::number(50, 200);
    const unsigned producers = utils::random::number(2, 8);
    auto responder = SharedQueue<Response>(total, mode);

    auto pool = utils::ThreadPool<unsigned>([&](unsigned i, std::atomic<bool> &)
    {
        usleep(utils::random::number(200)); // jitter the producers
        responder.push(i);
    }, producers);

    std::set<unsigned> expected;
    for (unsigned i = 0; i < total; ++i)
    {
        expected.insert(i);
        unsigned v = i;
        pool.push(std::move(v));
    }

    for (unsigned i = 0; i < total; ++i)
    {
        auto r = responder.pop(5000);
        EXPECT_EQ(r.ret, ResponseCode::Success);
        EXPECT_EQ(expected.erase(r.index), 1);
    }
    EXPECT_EQ(expected.size(), 0);
}

void run_timeout_while_running(QueueMode mode)
{
    // Outstanding request but nothing pushed yet: pop(timeout) returns TimedOut (does not block
    // forever) - FINISH_ON_DRAIN's drained shortcut only applies once _running == 0.
    auto responder = SharedQueue<Response>(1, mode);

    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(responder.pop(100).ret, ResponseCode::TimedOut);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_GE(ms, 90); // actually waited ~the timeout
}

void run_blocked_pop_woken_by_push(QueueMode mode)
{
    // An indefinite pop(0) blocked on an empty queue is woken by a later push and delivers it.
    auto responder = SharedQueue<Response>(1, mode);

    std::atomic<int> ret{ -2 };
    std::atomic<unsigned> idx{ 0 };
    auto consumer = utils::Thread([&]()
    {
        auto r = responder.pop(0); // blocks until pushed
        ret.store(static_cast<int>(r.ret));
        idx.store(r.index);
    });

    usleep(100 * 1000);
    responder.push(7u);
    consumer.join(); // must not deadlock

    EXPECT_EQ(ret.load(), static_cast<int>(ResponseCode::Success));
    EXPECT_EQ(idx.load(), 7u);
}

void run_stop_unblocks_blocked_consumer(QueueMode mode)
{
    // A consumer blocked in pop(0) must be woken by stop() with FinishedError.
    auto responder = SharedQueue<Response>(1, mode);

    std::atomic<int> result{ -1 };
    auto consumer = utils::Thread([&]()
    {
        result.store(static_cast<int>(responder.pop(0).ret));
    });

    usleep(100 * 1000);
    responder.stop();
    consumer.join(); // must not deadlock

    EXPECT_EQ(result.load(), static_cast<int>(ResponseCode::FinishedError));
}

} // namespace

TEST(FinishOnDrain, Delivery)                     { run_delivery(QueueMode::FINISH_ON_DRAIN); }
TEST(Persistent,    Delivery)                     { run_delivery(QueueMode::PERSISTENT); }

TEST(FinishOnDrain, MultiProducerSingleConsumer)  { run_multi_producer_single_consumer(QueueMode::FINISH_ON_DRAIN); }
TEST(Persistent,    MultiProducerSingleConsumer)  { run_multi_producer_single_consumer(QueueMode::PERSISTENT); }

TEST(FinishOnDrain, TimeoutWhileRunning)          { run_timeout_while_running(QueueMode::FINISH_ON_DRAIN); }
TEST(Persistent,    TimeoutWhileRunning)          { run_timeout_while_running(QueueMode::PERSISTENT); }

TEST(FinishOnDrain, BlockedPopWokenByPush)        { run_blocked_pop_woken_by_push(QueueMode::FINISH_ON_DRAIN); }
TEST(Persistent,    BlockedPopWokenByPush)        { run_blocked_pop_woken_by_push(QueueMode::PERSISTENT); }

TEST(FinishOnDrain, StopUnblocksBlockedConsumer)  { run_stop_unblocks_blocked_consumer(QueueMode::FINISH_ON_DRAIN); }
TEST(Persistent,    StopUnblocksBlockedConsumer)  { run_stop_unblocks_blocked_consumer(QueueMode::PERSISTENT); }

// --- Mode-specific: the drained-state semantics that differ between the two modes ---

TEST(Persistent, DrainedIsNotTerminal)
{
    // PERSISTENT: a drained queue (_running == 0 && empty) must NOT return FinishedError - it
    // times out - and the responder stays usable for the next submission.
    auto size = utils::random::number(1, 20);
    auto responder = SharedQueue<Response>(size, QueueMode::PERSISTENT);

    for (unsigned i = 0; i < size; ++i) responder.push(i);
    for (unsigned i = 0; i < size; ++i) EXPECT_EQ(responder.pop(2000).ret, ResponseCode::Success);

    // drained, but persistent -> TimedOut (not FinishedError)
    EXPECT_EQ(responder.pop(50).ret, ResponseCode::TimedOut);

    // still usable: grow the expected count and push more
    responder.increment(1);
    responder.push(size);
    auto again = responder.pop(2000);
    EXPECT_EQ(again.ret, ResponseCode::Success);
    EXPECT_EQ(again.index, size);
}

TEST(FinishOnDrain, DrainedIsTerminal)
{
    // FINISH_ON_DRAIN: once drained (_running == 0 && empty), pop() is terminal (FinishedError),
    // returned immediately at entry without waiting.
    auto responder = SharedQueue<Response>(1, QueueMode::FINISH_ON_DRAIN);
    responder.push(0u);
    EXPECT_EQ(responder.pop(2000).ret, ResponseCode::Success);

    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(responder.pop(60000).ret, ResponseCode::FinishedError); // long timeout, but returns at once
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(ms, 1000); // did not wait out the timeout
}

}; // namespace runai::llm::streamer::common
