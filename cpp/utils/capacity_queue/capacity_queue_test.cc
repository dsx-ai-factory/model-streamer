/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/capacity_queue/capacity_queue.h"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace runai::llm::streamer::utils
{

TEST(CapacityQueue, EmptyIsIdleAndTakesNothing)
{
    CapacityQueue<int> q(100);

    EXPECT_TRUE(q.empty());
    EXPECT_TRUE(q.idle());
    EXPECT_EQ(q.pending(), 0U);
    EXPECT_EQ(q.inflight(), 0U);
    EXPECT_FALSE(q.try_take().has_value());
}

TEST(CapacityQueue, TakesUntilWindowFull)
{
    CapacityQueue<int> q(100);
    q.enqueue(1, 40);
    q.enqueue(2, 40);
    q.enqueue(3, 40);

    EXPECT_EQ(q.pending(), 3U);

    auto a = q.try_take();          // inflight 0 -> 40
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, 1);
    EXPECT_EQ(q.inflight(), 40U);

    auto b = q.try_take();          // 40 + 40 = 80 <= 100
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, 2);
    EXPECT_EQ(q.inflight(), 80U);

    auto c = q.try_take();          // 80 + 40 = 120 > 100 -> blocked
    EXPECT_FALSE(c.has_value());
    EXPECT_EQ(q.inflight(), 80U);
    EXPECT_EQ(q.pending(), 1U);
}

TEST(CapacityQueue, CompleteFreesCredit)
{
    CapacityQueue<int> q(100);
    q.enqueue(1, 40);
    q.enqueue(2, 40);
    q.enqueue(3, 40);

    ASSERT_TRUE(q.try_take().has_value());   // inflight 40
    ASSERT_TRUE(q.try_take().has_value());   // inflight 80
    ASSERT_FALSE(q.try_take().has_value());  // full

    q.complete(40);                          // inflight 40
    EXPECT_EQ(q.inflight(), 40U);

    auto c = q.try_take();                   // 40 + 40 = 80 <= 100
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(*c, 3);
    EXPECT_EQ(q.inflight(), 80U);
}

TEST(CapacityQueue, AlwaysTakesOneWhenIdleEvenIfOverCapacity)
{
    CapacityQueue<int> q(10);
    q.enqueue(1, 1000);   // larger than the whole window
    q.enqueue(2, 5);

    auto a = q.try_take();   // inflight 0 -> allowed despite cost > capacity
    ASSERT_TRUE(a.has_value());
    EXPECT_EQ(*a, 1);
    EXPECT_EQ(q.inflight(), 1000U);

    // now something is in flight, so the next item must wait
    EXPECT_FALSE(q.try_take().has_value());

    q.complete(1000);
    auto b = q.try_take();
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(*b, 2);
}

TEST(CapacityQueue, PreservesFifoOrder)
{
    CapacityQueue<int> q(1000);
    for (int i = 0; i < 5; ++i)
    {
        q.enqueue(i, 1);
    }
    for (int i = 0; i < 5; ++i)
    {
        auto v = q.try_take();
        ASSERT_TRUE(v.has_value());
        EXPECT_EQ(*v, i);
    }
    EXPECT_TRUE(q.empty());
}

TEST(CapacityQueue, EnqueueFrontTakesPriorityOverPendingItems)
{
    CapacityQueue<int> q(1000);
    q.enqueue(1, 1);
    q.enqueue(2, 1);
    q.enqueue_front(9, 1);

    auto priority = q.try_take();
    ASSERT_TRUE(priority.has_value());
    EXPECT_EQ(*priority, 9);

    auto first = q.try_take();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, 1);

    auto second = q.try_take();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*second, 2);
}

TEST(CapacityQueue, CompleteClampsAtZero)
{
    CapacityQueue<int> q(100);
    q.enqueue(1, 30);
    ASSERT_TRUE(q.try_take().has_value());   // inflight 30
    q.complete(1000);                        // over-complete
    EXPECT_EQ(q.inflight(), 0U);
    EXPECT_TRUE(q.idle());
}

TEST(CapacityQueue, CompletionDrivenDrainNeverExceedsCapacity)
{
    constexpr size_t capacity = 100;
    constexpr size_t cost = 30;
    constexpr int total = 10;

    CapacityQueue<int> q(capacity);
    for (int i = 0; i < total; ++i)
    {
        q.enqueue(i, cost);
    }

    std::vector<int> taken;
    int completed = 0;

    auto pump = [&]
    {
        while (auto v = q.try_take())
        {
            taken.push_back(*v);
            EXPECT_LE(q.inflight(), capacity);   // costs are <= capacity, so never exceeded
        }
    };

    pump();
    while (completed < total)
    {
        ASSERT_GT(q.inflight(), 0U);   // something must be in flight to complete
        q.complete(cost);
        ++completed;
        pump();
    }

    // everything drained, in FIFO order, and the queue is idle
    ASSERT_EQ(static_cast<int>(taken.size()), total);
    for (int i = 0; i < total; ++i)
    {
        EXPECT_EQ(taken[i], i);
    }
    EXPECT_TRUE(q.idle());
}

TEST(CapacityQueue, ClearDropsPendingAndInflightInOneStep)
{
    // pending far exceeds the window, and some items are in flight: a try_take()/complete() drain would stop
    // at the full-window boundary, but clear() empties both in one step and leaves the queue idle().
    constexpr size_t capacity = 10;
    CapacityQueue<int> q(capacity);
    for (int i = 0; i < 100; ++i)   // 100 chunks, cost 1 -> pending >> capacity
    {
        q.enqueue(i, 1);
    }

    for (int i = 0; i < static_cast<int>(capacity); ++i)   // fill the window
    {
        ASSERT_TRUE(q.try_take().has_value());
    }
    ASSERT_FALSE(q.try_take().has_value());   // full: try_take() blocks with items still pending
    EXPECT_EQ(q.inflight(), capacity);
    EXPECT_EQ(q.pending(), 90U);

    q.clear();

    EXPECT_TRUE(q.idle());
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.pending(), 0U);
    EXPECT_EQ(q.inflight(), 0U);
    EXPECT_FALSE(q.try_take().has_value());

    // usable after clear(): still bounded by the same capacity
    q.enqueue(7, 1);
    auto v = q.try_take();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 7);
    EXPECT_EQ(q.capacity(), capacity);
}


// abort_pending drops what never reached the backend and keeps what did. The in-flight credit is the
// only record that something out there may still be writing into our memory, so an abort that wants to
// wait for it must not fabricate zero.
TEST(CapacityQueue, Abort_Pending_Keeps_Inflight_Credit)
{
    CapacityQueue<int> queue(4);

    queue.enqueue(1, 2);
    queue.enqueue(2, 2);
    queue.enqueue(3, 2);

    ASSERT_TRUE(queue.try_take().has_value());   // 2 of 4 in flight, two still pending

    queue.abort_pending();

    EXPECT_TRUE(queue.empty()) << "pending items survived the abort";
    EXPECT_EQ(queue.inflight(), 2u) << "in-flight credit was released without a completion";
    EXPECT_FALSE(queue.idle()) << "the queue called itself idle with work still at the backend";

    queue.complete(2);
    EXPECT_TRUE(queue.idle());
}

} // namespace runai::llm::streamer::utils
