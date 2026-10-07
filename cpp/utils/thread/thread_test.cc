/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/thread/thread.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <utility>
#include <chrono>

#include "utils/random/random.h"
#include "utils/logging/logging.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::utils
{

TEST(Creation, Empty)
{
    Thread t;
    (void)t;

    EXPECT_NO_THROW(t.join());
}

TEST(Creation, Function)
{
    Thread([](){});
}

TEST(Creation, Bind)
{
    const auto value = random::number();
    unsigned var;

    ASSERT_NE(var, value);

    {
        Thread([&var, value](){ var = value; });
    }

    EXPECT_EQ(var, value);
}

TEST(Join, Sanity)
{
    const auto start = std::chrono::steady_clock::now();

    auto thread = Thread([]()
    {
        ::sleep(1);
    });

    thread.join();

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GT(duration.count(), 900);
}

TEST(Join, Many_Times)
{
    auto thread = Thread([](){});

    for (auto count = random::number(2, 10); count > 0; --count)
    {
        thread.join();
    }
}

TEST(Dtor, Joins)
{
    const auto start = std::chrono::steady_clock::now();

    {
        auto thread = Thread([]()
        {
            ::sleep(1);
        });

        (void)thread;
        // d'tor is supposed to join the thread
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GT(duration.count(), 900);
}

TEST(Move_Operator, Empty)
{
    Thread a;
    Thread b;

    b = std::move(a);
}

TEST(Move_Operator, Running)
{
    const auto start = std::chrono::steady_clock::now();

    {
        Thread a([](){ sleep(2); });
        Thread b([](){ sleep(1); });

        b = std::move(a); // joins `b` for <1 seconds

        const auto end = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GT(duration.count(), 900);

        // joins `a`
    }

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GT(duration.count(), 1900);
}

TEST(Identifier, Sanity)
{
    temp::Env enable_log("RUNAI_STREAMER_LOG_TO_STDERR", "1");
    temp::Env set_log_level("RUNAI_STREAMER_LOG_LEVEL", "WARNING");

    Thread t([](){
        LOG(WARNING) << utils::random::string();
    });
    EXPECT_NO_THROW(t.join());
}

TEST(Exception, Dont_Crash)
{
    auto counter{0};

    Thread t([&](){
        ++counter;
        throw std::exception{};
    });

    t.join();
}

} // namespace runai::llm::streamer::utils
