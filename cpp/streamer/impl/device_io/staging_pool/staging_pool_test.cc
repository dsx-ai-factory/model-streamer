#include "streamer/impl/device_io/staging_pool/staging_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 4096;
constexpr size_t Slab = 4 * Buffer;   // four buffers per slab

StagingPool::Params params(unsigned max_buffers, size_t slab = Slab)
{
    StagingPool::Params p;
    p.buffer_bytesize = Buffer;
    p.slab_bytesize = slab;
    p.max_buffers = max_buffers;
    return p;
}

class StagingPoolTest : public ::testing::Test
{
 protected:
    std::shared_ptr<device::MockDevice> _mock = std::make_shared<device::MockDevice>();
};

} // namespace

// Nothing is pinned until a buffer is asked for, so a load with no device destination costs
// nothing at all.
TEST_F(StagingPoolTest, AllocatesNothingUntilAsked)
{
    StagingPool pool(_mock, params(16));

    EXPECT_EQ(_mock->host_allocs, 0u);
    EXPECT_EQ(_mock->events_created, 0u);
    EXPECT_EQ(pool.created(), 0u);
}

// One registration covers a whole slab, not one per buffer - the fixed per-call cost is the reason
// slabs exist.
TEST_F(StagingPoolTest, OneRegistrationPerSlab)
{
    StagingPool pool(_mock, params(16));

    StagingBuffer first;
    ASSERT_EQ(pool.try_acquire(first), common::ResponseCode::Success);
    ASSERT_TRUE(first.valid());

    EXPECT_EQ(_mock->host_allocs, 1u);
    EXPECT_EQ(_mock->host_alloc_sizes.front(), Slab);
    EXPECT_EQ(pool.created(), 4u);        // the whole slab is carved at once
    EXPECT_EQ(_mock->events_created, 4u); // one event per buffer, created with it
}

TEST_F(StagingPoolTest, BuffersAreCarvedFromTheSlabInOrder)
{
    StagingPool pool(_mock, params(4));

    std::vector<StagingBuffer> taken;
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        taken.push_back(buffer);
    }

    EXPECT_EQ(_mock->host_allocs, 1u);
    for (unsigned i = 1; i < 4; ++i)
    {
        EXPECT_EQ(taken[i].data, taken[i - 1].data + Buffer);
        EXPECT_NE(taken[i].event, taken[i - 1].event);
    }
}

// The pool grows only when a buffer is actually needed, so a small load never reaches the ceiling.
TEST_F(StagingPoolTest, GrowsOnDemandAndStopsAtTheCeiling)
{
    StagingPool pool(_mock, params(6));   // ceiling is not a multiple of the slab

    std::vector<StagingBuffer> taken;
    for (unsigned i = 0; i < 6; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid()) << "buffer " << i;
        taken.push_back(buffer);
    }

    EXPECT_EQ(pool.created(), 6u);
    EXPECT_EQ(pool.slabs(), 2u);
    // The second slab is cut to the room left, not to the full slab size.
    EXPECT_EQ(_mock->host_alloc_sizes[1], 2 * Buffer);

    StagingBuffer none;
    ASSERT_EQ(pool.try_acquire(none), common::ResponseCode::Success);
    EXPECT_FALSE(none.valid()) << "at the ceiling with everything in flight, acquire yields nothing";
    EXPECT_EQ(_mock->host_allocs, 2u) << "a refused acquire must not register another slab";
}

// Buffers are interchangeable, so returning them in an order unrelated to how they were taken is
// not a case the pool has to handle - which is what makes out-of-order read completion free.
TEST_F(StagingPoolTest, ReleaseOrderDoesNotMatter)
{
    StagingPool pool(_mock, params(4));

    std::vector<StagingBuffer> taken;
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        taken.push_back(buffer);
    }

    pool.release(taken[2]);
    pool.release(taken[0]);
    pool.release(taken[3]);
    pool.release(taken[1]);

    std::vector<char *> again;
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        again.push_back(buffer.data);
    }

    std::vector<char *> expected;
    for (const auto & buffer : taken)
    {
        expected.push_back(buffer.data);
    }
    std::sort(again.begin(), again.end());
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(again, expected) << "every buffer came back, whatever order they were returned in";
}

TEST_F(StagingPoolTest, AReleasedBufferIsHandedOutAgain)
{
    StagingPool pool(_mock, params(1, Buffer));

    StagingBuffer first;
    ASSERT_EQ(pool.try_acquire(first), common::ResponseCode::Success);
    ASSERT_TRUE(first.valid());

    StagingBuffer none;
    ASSERT_EQ(pool.try_acquire(none), common::ResponseCode::Success);
    ASSERT_FALSE(none.valid());

    pool.release(first);

    StagingBuffer again;
    ASSERT_EQ(pool.try_acquire(again), common::ResponseCode::Success);
    ASSERT_TRUE(again.valid());
    EXPECT_EQ(again.data, first.data);
    EXPECT_EQ(_mock->host_allocs, 1u) << "reuse, not another registration";
}

// A returned buffer must be usable immediately. A pool that handed back a buffer whose copy had
// not landed would corrupt the next read into it.
TEST_F(StagingPoolTest, TheBufferCarriesItsOwnEvent)
{
    StagingPool pool(_mock, params(4));

    StagingBuffer buffer;
    ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_NE(buffer.event, nullptr);

    device::Status status = device::Status::NotReady;
    ASSERT_EQ(_mock->event_query(buffer.event, status), common::ResponseCode::Success);
    EXPECT_EQ(status, device::Status::Ready) << "a never-recorded event is empty work, so a fresh buffer is free";

    ASSERT_EQ(_mock->event_record(buffer.event, nullptr), common::ResponseCode::Success);
    ASSERT_EQ(_mock->event_query(buffer.event, status), common::ResponseCode::Success);
    EXPECT_EQ(status, device::Status::NotReady);
}

TEST_F(StagingPoolTest, AFailedRegistrationIsReported)
{
    _mock->fail_host_alloc_at = 1;

    StagingPool pool(_mock, params(4));

    StagingBuffer buffer;
    EXPECT_EQ(pool.try_acquire(buffer), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(pool.created(), 0u);
}

// A slab whose first event fails yields no usable buffer. Keeping it and reporting success makes
// acquire() ask again, and grow() allocate another slab that fails the same way - pinned memory
// growing without bound under exactly the condition that made the event fail.
TEST_F(StagingPoolTest, ASlabThatYieldsNoBufferIsFreedAndReported)
{
    _mock->fail_event_create_from = 1;

    StagingPool pool(_mock, params(8, 4 * Buffer));

    for (unsigned attempt = 0; attempt < 5; ++attempt)
    {
        StagingBuffer buffer;
        EXPECT_EQ(pool.try_acquire(buffer), common::ResponseCode::DeviceOutOfMemory) << "attempt " << attempt;
        EXPECT_FALSE(buffer.valid());
    }

    EXPECT_EQ(pool.created(), 0u);
    EXPECT_EQ(_mock->host_allocs, 5u) << "one slab per attempt, and no more";
    EXPECT_EQ(_mock->host_frees, 5u) << "each unusable slab was freed at once, not held to teardown";
    EXPECT_TRUE(_mock->live_host.empty());
}

// A slab whose LATER events fail still yields the buffers it managed to build, so it is kept.
TEST_F(StagingPoolTest, ASlabThatYieldsSomeBuffersIsKept)
{
    _mock->fail_event_create_from = 3;      // two succeed, the rest fail

    StagingPool pool(_mock, params(8, 4 * Buffer));

    StagingBuffer first;
    ASSERT_EQ(pool.try_acquire(first), common::ResponseCode::Success);
    EXPECT_TRUE(first.valid());

    StagingBuffer second;
    ASSERT_EQ(pool.try_acquire(second), common::ResponseCode::Success);
    EXPECT_TRUE(second.valid());

    EXPECT_EQ(pool.created(), 2u) << "the two buffers whose events were made";
    EXPECT_EQ(_mock->host_allocs, 1u);
    EXPECT_EQ(_mock->host_frees, 0u) << "the slab is in use, so it stays";
}

// Events first, then the memory they refer to. The other order frees pinned pages the driver may
// still be writing into.
TEST_F(StagingPoolTest, TeardownDestroysEveryEventAndFreesEverySlab)
{
    {
        StagingPool pool(_mock, params(6));
        for (unsigned i = 0; i < 6; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        }
        EXPECT_EQ(_mock->events_created, 6u);
        EXPECT_EQ(_mock->host_allocs, 2u);
    }

    EXPECT_EQ(_mock->events_destroyed, 6u);
    EXPECT_EQ(_mock->event_syncs, 6u) << "drained before anything is freed";
    EXPECT_EQ(_mock->host_frees, 2u);
    EXPECT_TRUE(_mock->live_host.empty()) << "no slab left pinned";
}

// The reaper returns buffers from its own thread while the worker takes them. Two atomics and no
// lock, so this is the test that would catch a broken memory ordering.
TEST_F(StagingPoolTest, ConsumerAndProducerRunOnDifferentThreads)
{
    StagingPool pool(_mock, params(4));

    constexpr unsigned rounds = 20000;

    // A real handoff: the worker takes a buffer and passes it on, the reaper gives it back. Only
    // the worker touches the mock, which is not thread safe.
    std::mutex handoff_lock;
    std::deque<StagingBuffer> handoff;
    std::atomic<unsigned> returned{0};

    std::thread reaper([&]()
        {
            unsigned seen = 0;
            while (seen < rounds)
            {
                StagingBuffer buffer;
                {
                    const std::lock_guard<std::mutex> guard(handoff_lock);
                    if (handoff.empty())
                    {
                        continue;
                    }
                    buffer = handoff.front();
                    handoff.pop_front();
                }
                pool.release(buffer);
                ++seen;
                returned.fetch_add(1, std::memory_order_release);
            }
        });

    unsigned taken = 0;
    while (taken < rounds)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        if (!buffer.valid())
        {
            continue;   // everything is in flight; the reaper will hand one back
        }
        ++taken;
        const std::lock_guard<std::mutex> guard(handoff_lock);
        handoff.push_back(buffer);
    }

    reaper.join();
    EXPECT_EQ(taken, rounds);
    EXPECT_EQ(returned.load(), rounds);
    EXPECT_EQ(_mock->host_allocs, 1u) << "recycled, never grown past the first slab";
    EXPECT_EQ(pool.created(), 4u);
}

// The synchronous threadpool's threads have nothing else to do, so they wait rather than spin.
TEST_F(StagingPoolTest, SharedPoolBlocksUntilABufferComesBack)
{
    SharedStagingPool pool(_mock, params(1, Buffer));

    StagingBuffer held;
    ASSERT_EQ(pool.acquire(held), common::ResponseCode::Success);
    ASSERT_TRUE(held.valid());

    std::atomic<bool> got{false};
    std::thread waiter([&]()
        {
            StagingBuffer buffer;
            EXPECT_EQ(pool.acquire(buffer), common::ResponseCode::Success);
            EXPECT_TRUE(buffer.valid());
            got.store(true, std::memory_order_release);
        });

    // Still blocked: the only buffer is out.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(got.load(std::memory_order_acquire));

    pool.release(held);
    waiter.join();
    EXPECT_TRUE(got.load(std::memory_order_acquire));
}

// Without this, a waiter sleeps for a reaper that has already stopped.
TEST_F(StagingPoolTest, StopWakesAWaiter)
{
    SharedStagingPool pool(_mock, params(1, Buffer));

    StagingBuffer held;
    ASSERT_EQ(pool.acquire(held), common::ResponseCode::Success);

    std::thread waiter([&]()
        {
            StagingBuffer buffer;
            EXPECT_EQ(pool.acquire(buffer), common::ResponseCode::Success);
            EXPECT_FALSE(buffer.valid()) << "stopped, so nothing is handed out";
        });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pool.stop();
    waiter.join();

    pool.release(held);
}

} // namespace runai::llm::streamer::impl
