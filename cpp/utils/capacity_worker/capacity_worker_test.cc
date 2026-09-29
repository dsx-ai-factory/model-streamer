/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/capacity_worker/capacity_worker.h"

#include <unistd.h>

#include <gtest/gtest.h>
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "utils/threadpool/threadpool.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::utils
{

namespace
{

// A concrete CapacityWorker whose "backend" completes instantly: each request is a count of unit chunks.
// The three hooks model async I/O - submit "sends" a chunk (counts it), drain_batch "completes" one ready
// chunk. This exercises the base's submit/drain/execute pattern and the pool's per-worker routine end to
// end (self-draining execute, one-batch drain, idle, and the tail).
class CountingWorker : public CapacityWorker<unsigned, unsigned>
{
 public:
    CountingWorker(std::size_t capacity, std::atomic<unsigned> & submitted, std::atomic<unsigned> & completed) :
        _capacity(capacity),
        _submitted(submitted),
        _completed(completed)
    {}

 protected:
    std::size_t capacity(const unsigned &) override { return _capacity; }

    // capacity() above cannot throw, so discard() is never reached - an empty body satisfies the contract.
    void discard(unsigned && /*count*/) override {}

    void enqueue(unsigned && count) override
    {
        for (unsigned i = 0; i < count; ++i)
        {
            _queue->enqueue(0u, 1);   // one chunk, cost 1
        }
    }

    void submit(const unsigned &) override
    {
        _submitted += 1;             // "sent" a chunk to the backend
    }

    void drain_batch(std::atomic<bool> &) override
    {
        if (_queue->inflight() > 0)   // a ready completion
        {
            _queue->complete(1);
            _completed += 1;
        }
    }

 private:
    std::size_t _capacity;
    std::atomic<unsigned> & _submitted;
    std::atomic<unsigned> & _completed;
};

// Like CountingWorker but its completions are slow, and on shutdown it aborts (completes) all in-flight
// chunks at once from drain_batch(stopped) - modelling how the real worker fails in-flight reads on stop.
// This keeps in-flight work present when the pool is torn down, so the stop path is actually exercised.
class StopAwareWorker : public CapacityWorker<unsigned, unsigned>
{
 public:
    StopAwareWorker(std::size_t capacity, std::atomic<unsigned> & submitted, std::atomic<unsigned> & completed) :
        _capacity(capacity),
        _submitted(submitted),
        _completed(completed)
    {}

 protected:
    std::size_t capacity(const unsigned &) override { return _capacity; }

    // capacity() above cannot throw, so discard() is never reached - an empty body satisfies the contract.
    void discard(unsigned && /*count*/) override {}

    void enqueue(unsigned && count) override
    {
        for (unsigned i = 0; i < count; ++i) { _queue->enqueue(0u, 1); }
    }

    void submit(const unsigned &) override { _submitted += 1; }

    void drain_batch(std::atomic<bool> & stopped) override
    {
        if (stopped)   // abort: fail every in-flight chunk at once, no waiting
        {
            const auto n = _queue->inflight();
            for (std::size_t i = 0; i < n; ++i) { _queue->complete(1); _completed += 1; }
            return;
        }

        ::usleep(50);   // model async latency so in-flight persists until the pool is stopped
        if (_queue->inflight() > 0)
        {
            _queue->complete(1);
            _completed += 1;
        }
    }

 private:
    std::size_t _capacity;
    std::atomic<unsigned> & _submitted;
    std::atomic<unsigned> & _completed;
};

// Models a worker with one deferred item that becomes ready immediately before the next pump. The hook
// inserts it at the front so it is submitted before chunks from the newly admitted request.
class PrePumpWorker : public CapacityWorker<unsigned, unsigned>
{
 public:
    const std::vector<unsigned> & submitted() const { return _submitted; }

 protected:
    std::size_t capacity(const unsigned &) override { return 10; }
    void discard(unsigned && /*count*/) override {}

    void enqueue(unsigned && count) override
    {
        for (unsigned i = 0; i < count; ++i)
        {
            _queue->enqueue(i, 1);
        }
    }

    void pre_pump() override
    {
        if (!_deferred_promoted)
        {
            _queue->enqueue_front(99, 1);
            _deferred_promoted = true;
        }
    }

    void submit(const unsigned & chunk) override { _submitted.push_back(chunk); }

    void drain_batch(std::atomic<bool> &) override
    {
        if (_queue->inflight() > 0)
        {
            _queue->complete(1);
        }
    }

 private:
    bool _deferred_promoted = false;
    std::vector<unsigned> _submitted;
};

} // namespace

// --- base pattern, driven synchronously (no pool) ---

TEST(CapacityWorker, ExecuteLeavesInflightThenDrainCompletes)
{
    std::atomic<unsigned> submitted{0};
    std::atomic<unsigned> completed{0};
    std::atomic<bool> stopped{false};

    CountingWorker w(/*capacity=*/2, submitted, completed);

    EXPECT_TRUE(w.idle());

    unsigned one = 1;
    w.execute(std::move(one), stopped);

    EXPECT_EQ(submitted.load(), 1u);
    EXPECT_EQ(completed.load(), 0u);   // submitted, not yet completed
    EXPECT_FALSE(w.idle());            // one chunk in flight

    while (!w.idle()) { w.drain(stopped); }

    EXPECT_EQ(completed.load(), 1u);
    EXPECT_TRUE(w.idle());
}

TEST(CapacityWorker, ExecuteSelfDrainsRequestLargerThanWindow)
{
    std::atomic<unsigned> submitted{0};
    std::atomic<unsigned> completed{0};
    std::atomic<bool> stopped{false};

    CountingWorker w(/*capacity=*/2, submitted, completed);

    unsigned five = 5;
    w.execute(std::move(five), stopped);   // 5 chunks through a window of 2

    EXPECT_EQ(submitted.load(), 5u);       // all submitted despite the small window
    EXPECT_GE(completed.load(), 3u);       // execute drained to make room (+ freed a slot on return)
    EXPECT_FALSE(w.idle());                // some still in flight

    while (!w.idle()) { w.drain(stopped); }
    EXPECT_EQ(completed.load(), 5u);
}

TEST(CapacityWorker, PrePumpCanPrioritizeDeferredWork)
{
    std::atomic<bool> stopped{false};
    PrePumpWorker worker;

    unsigned two = 2;
    worker.execute(std::move(two), stopped);

    ASSERT_EQ(worker.submitted().size(), 3u);
    EXPECT_EQ(worker.submitted()[0], 99u);
    EXPECT_EQ(worker.submitted()[1], 0u);
    EXPECT_EQ(worker.submitted()[2], 1u);
}

// --- a ThreadPool of CapacityWorkers, stopped mid-flight ---

TEST(CapacityWorker, PoolStopsCleanlyWithInflight)
{
    std::atomic<unsigned> submitted{0};
    std::atomic<unsigned> completed{0};

    const unsigned num_workers = utils::random::number(2, 6);
    const std::size_t capacity = utils::random::number<std::size_t>(1, 4);

    {
        ThreadPool<unsigned> pool(
            [&]() -> std::unique_ptr<Worker<unsigned>>
            {
                return std::make_unique<StopAwareWorker>(capacity, submitted, completed);
            },
            num_workers);

        for (unsigned i = 0; i < 2000; ++i)
        {
            pool.push(utils::random::number(5, 20));   // many-chunk requests keep workers busy
        }

        ::usleep(50000);   // let workers accumulate in-flight work
    }   // ~pool stops the deque and sets stopped WHILE work is in flight; must join without hanging

    // Reaching here means shutdown didn't hang. Every dispatched chunk was submitted and then
    // completed - finished normally or aborted on stop - so the two counts match; undispatched
    // requests were dropped by Deque::stop and never counted.
    EXPECT_EQ(submitted.load(), completed.load());
    EXPECT_GT(submitted.load(), 0u);   // some work actually happened before the stop
}

TEST(CapacityWorker, PatternProcessesAllViaPool)
{
    std::atomic<unsigned> submitted{0};
    std::atomic<unsigned> completed{0};

    const unsigned num_workers = utils::random::number(2, 6);
    const std::size_t capacity = utils::random::number<std::size_t>(1, 4);

    unsigned total = 0;
    {
        ThreadPool<unsigned> pool(
            [&]() -> std::unique_ptr<Worker<unsigned>>
            {
                return std::make_unique<CountingWorker>(capacity, submitted, completed);
            },
            num_workers);

        const auto num_requests = utils::random::number(50, 200);
        for (unsigned i = 0; i < num_requests; ++i)
        {
            unsigned count = utils::random::number(1, 10);
            total += count;
            pool.push(std::move(count));
        }

        // every chunk is eventually submitted and then completed (incl. the tail drain)
        for (int i = 0; i < 10000 && completed.load() < total; ++i)
        {
            ::usleep(1000);
        }
    }   // ~pool stops the deque and joins all workers cleanly

    EXPECT_EQ(submitted.load(), total);
    EXPECT_EQ(completed.load(), total);
}

} // namespace runai::llm::streamer::utils
