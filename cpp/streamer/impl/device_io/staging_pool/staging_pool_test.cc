#include "streamer/impl/device_io/staging_pool/staging_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <memory>
#include <thread>
#include <vector>

#include "common/exception/exception.h"
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
TEST_F(StagingPoolTest, Allocates_Nothing_Until_Asked)
{
    StagingPool pool(_mock, params(16));

    EXPECT_EQ(_mock->host_allocs, 0u);
    EXPECT_EQ(pool.created(), 0u);
}

// One registration covers a whole slab, not one per buffer - the fixed per-call cost is the reason
// slabs exist.
TEST_F(StagingPoolTest, One_Registration_Per_Slab)
{
    StagingPool pool(_mock, params(16));

    StagingBuffer first;
    ASSERT_EQ(pool.try_acquire(first), common::ResponseCode::Success);
    ASSERT_TRUE(first.valid());

    EXPECT_EQ(_mock->host_allocs, 1u);
    EXPECT_EQ(_mock->host_alloc_sizes.front(), Slab);
    EXPECT_EQ(pool.created(), 4u);        // the whole slab is carved at once
}

TEST_F(StagingPoolTest, Buffers_Are_Carved_From_The_Slab_In_Order)
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
    }
}

// The pool grows only when a buffer is actually needed, so a small load never reaches the ceiling.
TEST_F(StagingPoolTest, Grows_On_Demand_And_Stops_At_The_Ceiling)
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

// Buffers of no bytes divide by zero when a slab is planned, which kills the process rather than
// telling the caller.
//
// And REPORTED, not answered with an empty hand: this pool holds nothing and never will, so
// acquire() would wait for a buffer to come back when none was ever handed out. That wait can only
// end when the process does.
TEST_F(StagingPoolTest, A_Pool_That_Can_Hold_Nothing_Is_Refused)
{
    StagingPool::Params zero;
    zero.max_buffers = 4;   // a ceiling, but no size to cut buffers to

    StagingPool pool(_mock, zero);

    StagingBuffer buffer;
    EXPECT_EQ(pool.try_acquire(buffer), common::ResponseCode::UnknownError);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(pool.acquire(buffer), common::ResponseCode::UnknownError) << "acquire waited";
    EXPECT_FALSE(buffer.valid());

    EXPECT_EQ(pool.created(), 0u);
    EXPECT_EQ(_mock->host_allocs, 0u) << "nothing was registered for buffers that hold nothing";
}

// The same for a window of no buffers at all, which is the other way to ask for a pool that cannot
// hand anything out.
TEST_F(StagingPoolTest, A_Window_Of_No_Buffers_Is_Refused)
{
    StagingPool::Params none;
    none.buffer_bytesize = Buffer;
    none.slab_bytesize = Buffer;

    StagingPool pool(_mock, none);

    StagingBuffer buffer;
    EXPECT_EQ(pool.try_acquire(buffer), common::ResponseCode::UnknownError);
    EXPECT_EQ(pool.acquire(buffer), common::ResponseCode::UnknownError) << "acquire waited";
}

// Buffers are interchangeable, so returning them in an order unrelated to how they were taken is
// not a case the pool has to handle - which is what makes out-of-order read completion free.
TEST_F(StagingPoolTest, Release_Order_Does_Not_Matter)
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

TEST_F(StagingPoolTest, A_Released_Buffer_Is_Handed_Out_Again)
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

TEST_F(StagingPoolTest, A_Failed_Registration_Is_Reported)
{
    _mock->fail_host_alloc_at = 1;

    StagingPool pool(_mock, params(4));

    StagingBuffer buffer;
    EXPECT_EQ(pool.try_acquire(buffer), common::ResponseCode::DeviceOutOfMemory);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(pool.created(), 0u);
}

// Every slab is freed. The pool waits for nothing: a buffer comes back only after its copy's event
// was synchronised, so every buffer being back already means no DMA is reading out of this memory.
TEST_F(StagingPoolTest, Teardown_Frees_Every_Slab)
{
    {
        StagingPool pool(_mock, params(6));
        for (unsigned i = 0; i < 6; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        }
        EXPECT_EQ(_mock->host_allocs, 2u);
        EXPECT_EQ(_mock->events_created, 0u) << "the pool owns no events";
    }

    EXPECT_EQ(_mock->host_frees, 2u);
    EXPECT_TRUE(_mock->live_host.empty()) << "no slab left pinned";
}

// The reaper returns buffers from its own thread while the worker takes them - the shape every user
// has. This is the test that would catch a free list left unprotected.
TEST_F(StagingPoolTest, Consumer_And_Producer_Run_On_Different_Threads)
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
TEST_F(StagingPoolTest, Acquire_Blocks_Until_A_Buffer_Comes_Back)
{
    StagingPool pool(_mock, params(1, Buffer));

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

// Pinning happens with the lock released. A driver call that THROWS must leave the pool exactly as
// it was, so the next consumer can register a slab itself - and must not leave the lock held, which
// would stop every thread that touches this pool.
TEST_F(StagingPoolTest, A_Throwing_Registration_Leaves_The_Pool_Usable)
{
    StagingPool pool(_mock, params(4));

    _mock->throw_host_alloc_at = 1;

    StagingBuffer buffer;
    EXPECT_THROW(pool.try_acquire(buffer), common::Exception);
    EXPECT_FALSE(buffer.valid());
    EXPECT_EQ(pool.created(), 0u);

    // The next consumer grows the pool itself, rather than being told one is already on its way.
    _mock->throw_host_alloc_at = 0;

    StagingBuffer again;
    ASSERT_EQ(pool.try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the throw left the pool unable to grow";
    EXPECT_EQ(pool.created(), 4u);
}

// Without this, a waiter sleeps for a reaper that has already stopped.
TEST_F(StagingPoolTest, Stop_Wakes_A_Waiter)
{
    StagingPool pool(_mock, params(1, Buffer));

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
