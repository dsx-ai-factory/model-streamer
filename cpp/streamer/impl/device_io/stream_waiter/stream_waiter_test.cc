/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/device_io/stream_waiter/stream_waiter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include "device/mock/mock_device.h"
#include "streamer/impl/device_io/event_pool/event_pool.h"

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
    // One copy's worth: a buffer from `pool` and an event from this device's own pool. The waiter
    // gives both back, so both have to be real.
    StreamWaiter::Copy copy_of(const std::shared_ptr<StagingPool> & pool, const StagingBuffer & buffer)
    {
        if (_events == nullptr)
        {
            _events = std::make_shared<EventPool>(_mock, 64);
        }

        StreamWaiter::Copy copy;
        copy.pool = pool;
        copy.buffer = buffer;
        copy.events = _events;
        EXPECT_EQ(_events->acquire(copy.event), common::ResponseCode::Success);
        return copy;
    }

    std::shared_ptr<device::MockDevice> _mock = std::make_shared<device::MockDevice>();
    std::shared_ptr<EventPool> _events;
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
TEST_F(StreamWaiterTest, Starts_No_Thread_Until_The_First_Copy)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(4));
    StreamWaiter waiter(_mock);

    EXPECT_FALSE(waiter.running());
    EXPECT_EQ(_mock->bind_calls, 0u);
}

TEST_F(StreamWaiterTest, The_Thread_Starts_On_The_First_Enqueue)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(4));
    StreamWaiter waiter(_mock);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    ASSERT_TRUE(buffer.valid());

    waiter.enqueue(copy_of(pool, buffer), nullptr);

    EXPECT_TRUE(eventually([&]() { return waiter.completed() == 1u; }));
    EXPECT_TRUE(waiter.running());

    // The thread binds a context before any driver call, because a new thread inherits none.
    EXPECT_GE(_mock->bind_calls, 1u);
}

// What the waiter is for: the buffer must be back in the pool once its copy has landed.
TEST_F(StreamWaiterTest, A_Buffer_Comes_Back_After_Its_Copy)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(1));
    StreamWaiter waiter(_mock);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    StagingBuffer none;
    ASSERT_EQ(pool->try_acquire(none), common::ResponseCode::Success);
    ASSERT_FALSE(none.valid()) << "the only buffer is out";

    std::atomic<int> reported{-1};
    waiter.enqueue(copy_of(pool, buffer), [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);

    StagingBuffer again;
    ASSERT_EQ(pool->try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the buffer was returned, not lost";
    EXPECT_EQ(again.data, buffer.data);
}

// The event is what is waited on, not the stream: waiting on the stream would drain everything
// queued on it rather than this one buffer's copy, which is what keeps the pipeline at depth.
TEST_F(StreamWaiterTest, It_Waits_On_The_Event_Not_The_Stream)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(2));
    StreamWaiter waiter(_mock);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
    waiter.enqueue(copy_of(pool, buffer), nullptr);

    ASSERT_TRUE(eventually([&]() { return waiter.completed() == 1u; }));
    EXPECT_EQ(_mock->event_syncs, 1u);
}

// Order is kept, because one stream executes in issue order: waiting on the head is exact, and
// nothing behind it can have landed first.
TEST_F(StreamWaiterTest, Completions_Are_Reported_In_Issue_Order)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(8));
    StreamWaiter waiter(_mock);

    std::mutex order_lock;
    std::vector<unsigned> order;

    for (unsigned i = 0; i < 8; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        waiter.enqueue(copy_of(pool, buffer), [&, i](common::ResponseCode)
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

// A COPY THAT CANNOT BE WAITED FOR DOES NOT GIVE ITS BUFFER BACK.
//
// The wait is what would prove the copy stopped reading, and the wait is what failed - so the DMA may
// still be reading this buffer and writing the caller's destination. Handing it to the next chunk
// would corrupt that chunk's read, silently. The pool loses one buffer instead; the same decision
// DeviceWriter::write makes when it cannot drain the stream.
//
// This test used to assert the opposite ("returned despite the failure"), on the grounds that a lost
// buffer is a later deadlock. It is not: the pool reports a retired buffer rather than blocking, so a
// caller is told why instead of waiting for one that can never come back.
TEST_F(StreamWaiterTest, A_Copy_That_Cannot_Be_Waited_For_Keeps_Its_Buffer)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(1));
    StreamWaiter waiter(_mock);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    _mock->fail_event_synchronize = true;

    std::atomic<int> reported{-1};
    waiter.enqueue(copy_of(pool, buffer), [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));

    // NOT the driver's DeviceTransferError, which tells a caller the destination is free. The copy may
    // still be landing in it.
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::DeviceDriverError);

    StagingBuffer again;
    EXPECT_EQ(pool->try_acquire(again), common::ResponseCode::DeviceDriverError)
        << "the buffer went back to the pool although nothing can say when the copy stops reading it";
    EXPECT_FALSE(again.valid());
    EXPECT_EQ(pool->retired(), 1u);

    _mock->fail_event_synchronize = false;
}

// A CONTEXT THAT COULD NOT BE BOUND IS RETRIED, and its copies keep their buffers.
//
// bind_thread used to be marked done whether or not it succeeded, so one failure left every later copy
// running without a context - each failing in event_synchronize, naming the wait rather than the bind
// that never happened. The flag is now set only on success.
TEST_F(StreamWaiterTest, A_Failed_Bind_Is_Retried_And_Keeps_The_Buffer)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(2));
    StreamWaiter waiter(_mock);

    StagingBuffer first;
    ASSERT_EQ(pool->try_acquire(first), common::ResponseCode::Success);

    _mock->fail_bind_thread = true;

    std::atomic<int> reported{-1};
    waiter.enqueue(copy_of(pool, first), [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));

    // The copy was enqueued on the writer's thread and is beyond reach from here, so its buffer is
    // retained exactly as a failed wait would retain it.
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::DeviceDriverError);
    EXPECT_EQ(pool->retired(), 1u);

    // THE POINT: the next copy tries to bind again rather than running with a context we never got.
    const auto binds = _mock->bind_calls.load();
    _mock->fail_bind_thread = false;

    StagingBuffer second;
    ASSERT_EQ(pool->try_acquire(second), common::ResponseCode::Success);

    reported.store(-1);
    waiter.enqueue(copy_of(pool, second), [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_GT(_mock->bind_calls.load(), binds) << "the waiter never tried to bind again";
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success)
        << "the bind succeeded this time, so the copy completes normally";
}

// And a copy that CAN be waited for still gives it back - the ordinary path, unchanged.
TEST_F(StreamWaiterTest, A_Failed_Copy_Returns_Its_Buffer_When_The_Wait_Succeeded)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(1));
    StreamWaiter waiter(_mock);

    StagingBuffer buffer;
    ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);

    std::atomic<int> reported{-1};
    waiter.enqueue(copy_of(pool, buffer), [&](common::ResponseCode code) { reported.store(static_cast<int>(code)); });

    ASSERT_TRUE(eventually([&]() { return reported.load() >= 0; }));
    EXPECT_EQ(static_cast<common::ResponseCode>(reported.load()), common::ResponseCode::Success);

    StagingBuffer again;
    ASSERT_EQ(pool->try_acquire(again), common::ResponseCode::Success);
    EXPECT_TRUE(again.valid()) << "the wait succeeded, so the buffer is free";
    EXPECT_EQ(pool->retired(), 0u);
}

// The pool's teardown assumes every buffer is back, so stopping must drain rather than abandon.
TEST_F(StreamWaiterTest, Stop_Waits_For_What_Is_Already_Queued)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(4));

    {
        StreamWaiter waiter(_mock);
        for (unsigned i = 0; i < 4; ++i)
        {
            StagingBuffer buffer;
            ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
            ASSERT_TRUE(buffer.valid());
            waiter.enqueue(copy_of(pool, buffer), nullptr);
        }
        waiter.stop();
        EXPECT_EQ(waiter.completed(), 4u) << "drained, not abandoned";
    }

    // Every buffer is back, so the pool hands out four again without growing.
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
        EXPECT_TRUE(buffer.valid()) << "buffer " << i;
    }
}

// The contract that replaced the refusal path: stopping drains, so the pool gets every buffer back
// and the caller of each enqueue was told. A violation - enqueueing after stop - is a caller bug
// that ASSERT reports, not a case this class handles.
TEST_F(StreamWaiterTest, Every_Enqueued_Buffer_Is_Returned_By_The_Time_Stop_Returns)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(4));
    StreamWaiter waiter(_mock);

    std::atomic<unsigned> reported{0};
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
        ASSERT_TRUE(buffer.valid());
        waiter.enqueue(copy_of(pool, buffer), [&](common::ResponseCode) { ++reported; });
    }

    waiter.stop();

    EXPECT_EQ(reported.load(), 4u) << "every caller was told before stop returned";
    for (unsigned i = 0; i < 4; ++i)
    {
        StagingBuffer buffer;
        ASSERT_EQ(pool->try_acquire(buffer), common::ResponseCode::Success);
        EXPECT_TRUE(buffer.valid()) << "buffer " << i << " came back";
    }
}

TEST_F(StreamWaiterTest, Destruction_Stops_A_Thread_That_Was_Never_Used)
{
    auto pool = std::make_shared<StagingPool>(_mock, params(4));
    {
        StreamWaiter waiter(_mock);
    }
    SUCCEED() << "no thread was started, so none had to be joined";
}

} // namespace runai::llm::streamer::impl
