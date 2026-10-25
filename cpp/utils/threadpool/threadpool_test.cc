/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/threadpool/threadpool.h"

#include <unistd.h>

#include <gtest/gtest.h>
#include <atomic>
#include <memory>

#include "utils/logging/logging.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::utils
{

namespace
{

// A minimal Worker with no capacity window: execute is synchronous and the worker is always idle. This
// exercises the pool's per-worker routine itself (try_pop / blocking-pop dispatch and clean shutdown),
// independent of the CapacityWorker window machinery (which is covered in capacity_worker_test).
class SyncWorker : public Worker<unsigned>
{
 public:
    explicit SyncWorker(std::atomic<unsigned> & executed) : _executed(executed) {}

    void execute(unsigned && value, std::atomic<bool> &) override { _executed += value; }
    void drain(std::atomic<bool> &) override {}   // nothing is ever in flight
    bool idle() const override { return true; }

 private:
    std::atomic<unsigned> & _executed;
};

// A Worker whose execute() holds its request until the pool stops, then leaves it in flight for drain() to
// finish - the shape of an async backend worker at shutdown.
class StoppableWorker : public Worker<unsigned>
{
 public:
    StoppableWorker(std::atomic<unsigned> & executed, std::atomic<unsigned> & drained) :
        _executed(executed),
        _drained(drained)
    {}

    void execute(unsigned &&, std::atomic<bool> & stopped) override
    {
        _executed++;
        while (!stopped)
        {
            ::usleep(1000);
        }
        _in_flight = true;
    }

    void drain(std::atomic<bool> &) override
    {
        if (_in_flight)
        {
            _in_flight = false;
            _drained++;
        }
    }

    bool idle() const override { return !_in_flight; }

 private:
    std::atomic<unsigned> & _executed;
    std::atomic<unsigned> & _drained;
    bool _in_flight = false;   // touched only by this worker's own thread
};

} // namespace

TEST(PerWorker, StoppedDropsLateRequestsAndDrainsInFlight)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        std::atomic<unsigned> executed{0};
        std::atomic<unsigned> drained{0};

        {
            ThreadPool<unsigned> pool(
                [&]() -> std::unique_ptr<Worker<unsigned>>
                {
                    return std::make_unique<StoppableWorker>(executed, drained);
                },
                size);

            const auto count = utils::random::number(5000, 10000);
            for (unsigned i = 0; i < count; ++i)
            {
                pool.push(utils::random::number(1, 100));
            }

            // every worker is inside its first request, so none can miss the stop by starting late
            for (int i = 0; i < 10000 && executed < size; ++i)
            {
                ::usleep(1000);
            }
            ASSERT_EQ(executed, size);

            ::usleep(utils::random::number(10000, 100000));
        }

        // each worker ran only the request it held when the pool stopped, and drained that one
        EXPECT_EQ(executed, size);
        EXPECT_EQ(drained, size);
    }
}

TEST(PerWorker, DispatchesEveryRequest)
{
    std::atomic<unsigned> executed{0};

    const unsigned num_workers = utils::random::number(2, 8);

    unsigned total = 0;
    {
        ThreadPool<unsigned> pool(
            [&]() -> std::unique_ptr<Worker<unsigned>>
            {
                return std::make_unique<SyncWorker>(executed);
            },
            num_workers);

        const auto num_requests = utils::random::number(1000, 5000);
        for (unsigned i = 0; i < num_requests; ++i)
        {
            unsigned value = utils::random::number(1, 100);
            total += value;
            pool.push(std::move(value));
        }

        for (int i = 0; i < 5000 && executed.load() < total; ++i)
        {
            ::usleep(1000);
        }
    }   // ~pool stops the deque and joins all workers cleanly

    EXPECT_EQ(executed.load(), total);
}

TEST(Creation, Sanity)
{
    for (auto size : { 1U, 0U, utils::random::number(2, 100), 100U })
    {
        ThreadPool<unsigned> pool([](unsigned, std::atomic<bool> &){}, size);
        (void)pool;
    }
}

TEST(Handle, Sanity)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        constexpr auto Options = 1000;

        std::array<std::atomic<int>, Options> counters{};
        std::atomic<unsigned> total = 0;

        ThreadPool<unsigned> pool([&](unsigned index, std::atomic<bool> & stopped)
            {
                total++;
                counters[index]--;
            }, size);

        const auto count = utils::random::number(50000, 100000);

        for (unsigned i = 0; i < count; ++i)
        {
            auto index = utils::random::number(Options);
            counters[index]++;
            pool.push(std::move(index));
        }

        sleep(1);

        EXPECT_EQ(total, count);

        for (const auto & counter : counters)
        {
            EXPECT_EQ(counter, 0);
        }
    }
}

TEST(Handle, Exception_Thrown_From_Handler)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        std::atomic<unsigned> counter = 0;

        ThreadPool<bool> pool([&](auto _, std::atomic<bool> & stopped)
            {
                counter++;
                throw std::exception();
            }, size);

        const auto count = utils::random::number(50, 100);

        for (unsigned i = 0; i < count; ++i)
        {
            pool.push(utils::random::boolean());
        }

        sleep(1);

        EXPECT_EQ(count, counter);
    }
}

TEST(Handle, Stopped)
{
    for (auto size : { 1U, 2U, 10U, utils::random::number(10, 100), 100U })
    {
        std::atomic<unsigned> started = 0;
        std::atomic<unsigned> released = 0;
        std::atomic<unsigned> total = 0;
        const auto count = utils::random::number(5000, 10000);

        {
            ThreadPool<unsigned> pool([&](auto _, std::atomic<bool> & stopped)
                {
                    // Until the pool stops, every thread is held in its first request, so the first `size`
                    // entries are those requests and any later one is a request taken after the stop.
                    const auto index = started++;
                    while (!stopped)
                    {
                        ::usleep(utils::random::number(10000));
                    }
                    if (index < size)
                    {
                        released++;
                    }
                    total++;
                }, size);

            for (unsigned i = 0; i < count; ++i)
            {
                pool.push(utils::random::boolean());
            }

            // wait for every thread to be inside its request, so none can miss the stop by starting late
            for (int i = 0; i < 10000 && started < size; ++i)
            {
                ::usleep(1000);
            }
            ASSERT_EQ(started, size);

            ::usleep(utils::random::number(10000, 100000));

            EXPECT_EQ(total, 0);
        }

        // verify all threads stopped, and that no request was handled after the stop

        EXPECT_EQ(released, size);
        EXPECT_EQ(total, size);
    }
}

} // namespace runai::llm::streamer::utils
