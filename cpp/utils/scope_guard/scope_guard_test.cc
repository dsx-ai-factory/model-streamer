/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scope_guard/scope_guard.h"

#include <gtest/gtest.h>

#include <utility>

namespace runai::llm::streamer::utils
{

TEST(Creation, Lambda)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter](){ counter++; });
        (void) guard;

        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 1);
}

TEST(Creation, Moved)
{
    int counter = 0;

    {
        auto a = ScopeGuard([&counter](){ counter++; });

        {
            auto b = ScopeGuard(std::move(a));
            EXPECT_EQ(counter, 0);
        }

        EXPECT_EQ(counter, 1);
    }

    EXPECT_EQ(counter, 1);
}

TEST(Creation, Moved__Canceled)
{
    int counter = 0;

    {
        auto a = ScopeGuard([&counter](){ counter++; });
        a.cancel();

        {
            auto b = ScopeGuard(std::move(a));
            EXPECT_EQ(counter, 0);
        }

        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 0);
}

TEST(Canceled, Lambda_Not_Called)
{
    int counter = 0;

    {
        auto a = ScopeGuard([&counter](){ counter++; });

        a.cancel();
    }

    EXPECT_EQ(counter, 0);
}

TEST(Moveable, Creation)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter](){ counter++; });
        ScopeGuard guard2(std::move(guard));

        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 1);
}

TEST(Moveable, Assignment)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter](){ counter++; });
        ScopeGuard guard2([](){});
        guard2 = std::move(guard);

        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 1);
}

TEST(Moveable, Assignment__Old_Is_Executed)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter]() {
            counter++;
        });
        auto guard2 = ScopeGuard([&counter]() {
            counter+=42;
        });

        guard2 = std::move(guard);
        // The move should invoke the old function if not cancelled
        EXPECT_EQ(counter, 42);
    }

    // The movee scope guard callback should be abandoned
    EXPECT_EQ(counter, 43);
}

TEST(Moveable, Assignment__Old_Cancelled_Not_Executed)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter]() {
            counter++;
        });
        auto guard2 = ScopeGuard([&counter]() {
            counter += 42;
        });

        guard2.cancel();
        guard2 = std::move(guard);
        // The move should not invoke the old function if cancelled
        EXPECT_EQ(counter, 0);
    }

    // The movee scope guard callback should be abandoned
    EXPECT_EQ(counter, 1);
}

TEST(Moveable, ExceptionSafe__Move)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter](){
            counter++;
            throw std::exception{};
        });

        ScopeGuard guard2(std::move(guard));
        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 1);
}

TEST(Moveable, ExceptionSafe__Move_Assign)
{
    int counter = 0;

    {
        auto guard = ScopeGuard([&counter](){
            counter++;
            throw std::exception{};
        });

        ScopeGuard guard2([](){});
        guard2 = std::move(guard);
        EXPECT_EQ(counter, 0);
    }

    EXPECT_EQ(counter, 1);
}

} // namespace runai::llm::streamer::utils
