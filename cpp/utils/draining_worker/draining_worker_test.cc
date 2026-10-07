/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/draining_worker/draining_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace runai::llm::streamer::utils
{

namespace
{

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

// An object that is never used costs nothing: no thread, and no handler call.
TEST(DrainingWorker, Starts_Nothing_Until_The_First_Push)
{
    std::atomic<unsigned> handled{0};
    DrainingWorker<unsigned> worker([&](unsigned &&) { ++handled; });

    EXPECT_FALSE(worker.running());
    EXPECT_EQ(worker.handled(), 0u);
    EXPECT_EQ(handled.load(), 0u);
}

TEST(DrainingWorker, The_Thread_Starts_On_The_First_Push)
{
    std::atomic<unsigned> handled{0};
    DrainingWorker<unsigned> worker([&](unsigned &&) { ++handled; });

    worker.push(1);
    EXPECT_TRUE(eventually([&]() { return handled.load() == 1u; }));
    EXPECT_TRUE(worker.running());
}

// One thread over a FIFO, so the handler sees what was pushed, in the order it was pushed.
TEST(DrainingWorker, Messages_Are_Handled_In_Order)
{
    std::mutex lock;
    std::vector<unsigned> seen;

    DrainingWorker<unsigned> worker([&](unsigned && value)
        {
            const std::lock_guard<std::mutex> guard(lock);
            seen.push_back(value);
        });

    for (unsigned i = 0; i < 100; ++i)
    {
        worker.push(std::move(i));
    }
    worker.stop();

    const std::lock_guard<std::mutex> guard(lock);
    ASSERT_EQ(seen.size(), 100u);
    for (unsigned i = 0; i < 100; ++i)
    {
        EXPECT_EQ(seen[i], i);
    }
}

// The whole reason this type exists rather than ThreadPool: a queued message is a promise to
// somebody, so stopping must finish it rather than drop it.
TEST(DrainingWorker, Stop_Handles_What_Is_Already_Queued)
{
    std::atomic<unsigned> handled{0};
    std::atomic<bool> release{false};

    DrainingWorker<unsigned> worker([&](unsigned &&)
        {
            // Hold the first message so the rest are still queued when stop() is called.
            while (!release.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ++handled;
        });

    for (unsigned i = 0; i < 16; ++i)
    {
        worker.push(std::move(i));
    }

    ASSERT_TRUE(eventually([&]() { return worker.running(); }));
    release.store(true, std::memory_order_release);

    worker.stop();
    EXPECT_EQ(handled.load(), 16u) << "queued messages were handled, not dropped";
    EXPECT_EQ(worker.handled(), 16u);
}

TEST(DrainingWorker, The_Destructor_Drains_Too)
{
    std::atomic<unsigned> handled{0};
    {
        DrainingWorker<unsigned> worker([&](unsigned &&) { ++handled; });
        for (unsigned i = 0; i < 32; ++i)
        {
            worker.push(std::move(i));
        }
    }
    EXPECT_EQ(handled.load(), 32u);
}

// Stopping twice must not push a second sentinel or join twice. That is the only job the stopped
// flag still has - it no longer guards ordering against push.
TEST(DrainingWorker, Stopping_Twice_Is_Harmless)
{
    std::atomic<unsigned> handled{0};
    DrainingWorker<unsigned> worker([&](unsigned &&) { ++handled; });

    worker.push(1);
    worker.stop();
    worker.stop();

    EXPECT_EQ(handled.load(), 1u);
    EXPECT_FALSE(worker.running());
}

TEST(DrainingWorker, Stop_Is_Idempotent_And_Safe_When_Unused)
{
    DrainingWorker<unsigned> worker([](unsigned &&) {});

    worker.stop();
    worker.stop();
    SUCCEED() << "stopping an unused worker starts nothing and joins nothing";
}

// The sentinel is a queue entry, not a message, and must not be handed to the handler.
TEST(DrainingWorker, The_Sentinel_Never_Reaches_The_Handler)
{
    std::atomic<unsigned> handled{0};
    DrainingWorker<std::string> worker([&](std::string && value)
        {
            EXPECT_FALSE(value.empty()) << "a default-constructed message reached the handler";
            ++handled;
        });

    worker.push(std::string("one"));
    worker.push(std::string("two"));
    worker.stop();

    EXPECT_EQ(handled.load(), 2u);
    EXPECT_EQ(worker.handled(), 2u);
}

} // namespace runai::llm::streamer::utils
