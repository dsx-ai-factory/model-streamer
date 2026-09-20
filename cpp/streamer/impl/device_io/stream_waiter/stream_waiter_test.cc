#include "streamer/impl/device_io/stream_waiter/stream_waiter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include "device/mock/mock_device.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr size_t Buffer = 4096;

StagingPool::Params params(unsigned max_buffers)
{
    StagingPool::Params p;
    p.buffer_bytesize = Buffer;
    p.slab_bytesize = Buffer;
    p.max_buffers = max_buffers;
    return p;
}

class StreamWaiterTest : public ::testing::Test
{
 protected:
    std::shared_ptr<device::MockDevice> _mock = std::make_shared<device::MockDevice>();
};

// Waits until a condition holds, so a test never depends on a thread being scheduled within some
// fixed sleep.
bool eventually(const std::function<bool()> & holds)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (holds())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

} // namespace

// A load whose destinations are all host memory never issues a copy, so it never enqueues, so no
// thread exists and no driver call is made.
TEST_F(StreamWaiterTest, StartsNoThreadUntilTheFirstCopy)
{
    StagingPool pool(_mock, params(4));
    StreamWaiter waiter(_mock, pool);

    EXPECT_FALSE(waiter.running());
    EXPECT_EQ(_mock->bind_calls, 0u);
}

TEST_F(StreamWaiterTest, TheThreadStartsOnTheFirstEnqueue)
{
    StagingPool pool(_mock, params(4));
    StreamWaiter waiter(_mock, pool);

    StagingBuffer buffer;
    ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    waiter.enqueue(buffer, nullptr);

    EXPECT_TRUE(eventually([&]() { return waiter.completed() == 1u; }));
    EXPECT_TRUE(waiter.running());

    // The thread binds a context before any driver call, because a new thread inherits none.
    EXPECT_GE(_mock->bind_calls, 1u);
}

// What the waiter is for: the buffer must be back in the pool once its copy has landed.
TEST_F(StreamWaiterTest, ABufferComesBackAfterItsCopy)
{
    StagingPool pool(_mock, params(1));
    StreamWaiter waiter(_mock, pool);

    StagingBuffer buffer;
    ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);

    StagingBuffer none;
    ASSERT_EQ(pool.try_acquire(none), common::ResponseCode::Success);
    ASSERT_FALSE(none.valid()) << "the only buffer is out";

    std::atomic<int> reported{-1};
    waiter.enqueue(buffer, [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);

    StagingBuffer again;
    ASSERT_EQ(pool.try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the buffer was returned, not lost";
    EXPECT_EQ(again.data, buffer.data);
}

// The event is what is waited on, not the stream: waiting on the stream would drain everything
// queued on it rather than this one buffer's copy, which is what keeps the pipeline at depth.
TEST_F(StreamWaiterTest, ItWaitsOnTheEventNotTheStream)
{
    StagingPool pool(_mock, params(2));
    StreamWaiter waiter(_mock, pool);

    StagingBuffer buffer;
    ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
    waiter.enqueue(buffer, nullptr);

    ASSERT_TRUE(eventually([&]() { return waiter.completed() == 1u; }));
    EXPECT_EQ(_mock->event_syncs, 1u);
}

// Order is kept, because one stream executes in issue order: waiting on the head is exact, and
// nothing behind it can have landed first.
TEST_F(StreamWaiterTest, CompletionsAreReportedInIssueOrder)
{
    StagingPool pool(_mock, params(8));
    StreamWaiter waiter(_mock, pool);

    std::mutex order_lock;
    std::vector<unsigned> order;

    for (unsigned i = 0; i < 8; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        waiter.enqueue(buffer, [&, i](common::ResponseCode)
            {
                const std::lock_guard<std::mutex> guard(order_lock);
                order.push_back(i);
            });
    }

    ASSERT_TRUE(eventually([&]() { return waiter.completed() == 8u; }));

    const std::lock_guard<std::mutex> guard(order_lock);
    ASSERT_EQ(order.size(), 8u);
    for (unsigned i = 0; i < 8; ++i)
    {
        EXPECT_EQ(order[i], i);
    }
}

// A buffer lost on an error path is a deadlock that arrives later, so it comes back either way.
TEST_F(StreamWaiterTest, AFailedCopyStillReturnsItsBuffer)
{
    StagingPool pool(_mock, params(1));
    StreamWaiter waiter(_mock, pool);

    StagingBuffer buffer;
    ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);

    _mock->fail_event_synchronize = true;

    std::atomic<int> reported{-1};
    waiter.enqueue(buffer, [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::DeviceTransferError);

    StagingBuffer again;
    ASSERT_EQ(pool.try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "returned despite the failure";
}

// The pool's teardown assumes every buffer is back, so stopping must drain rather than abandon.
TEST_F(StreamWaiterTest, StopWaitsForWhatIsAlreadyQueued)
{
    StagingPool pool(_mock, params(4));

    {
        StreamWaiter waiter(_mock, pool);
        for (unsigned i = 0; i < 4; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());
            waiter.enqueue(buffer, nullptr);
        }
        waiter.stop();
        EXPECT_EQ(waiter.completed(), 4u) << "drained, not abandoned";
    }

    // Every buffer is back, so the pool hands out four again without growing.
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        EXPECT_TRUE(buffer.valid()) << "buffer " << i;
    }
}

// The contract that replaced the refusal path: stopping drains, so the pool gets every buffer back
// and the caller of each enqueue was told. A violation - enqueueing after stop - is a caller bug
// that ASSERT reports, not a case this class handles.
TEST_F(StreamWaiterTest, EveryEnqueuedBufferIsReturnedByTheTimeStopReturns)
{
    StagingPool pool(_mock, params(4));
    StreamWaiter waiter(_mock, pool);

    std::atomic<unsigned> reported{0};
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        waiter.enqueue(buffer, [&](common::ResponseCode) { ++reported; });
    }

    waiter.stop();

    EXPECT_EQ(reported.load(), 4u) << "every caller was told before stop returned";
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool.try_acquire(buffer), common::ResponseCode::Success);
        EXPECT_TRUE(buffer.valid()) << "buffer " << i << " came back";
    }
}

TEST_F(StreamWaiterTest, DestructionStopsAThreadThatWasNeverUsed)
{
    StagingPool pool(_mock, params(4));
    {
        StreamWaiter waiter(_mock, pool);
    }
    SUCCEED() << "no thread was started, so none had to be joined";
}

} // namespace runai::llm::streamer::impl
