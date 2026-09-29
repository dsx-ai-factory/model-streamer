/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/temp/env/env.h"

#include <gtest/gtest.h>

#include <utility>

#include "utils/env/env.h"

namespace runai::llm::streamer::utils::temp
{

TEST(Creation, Sanity)
{
    std::string name;

    {
        auto env = temp::Env();
        name = env.name;

        EXPECT_NE(::getenv(name.c_str()), nullptr);
    }

    EXPECT_EQ(::getenv(name.c_str()), nullptr);
}

TEST(Temp, Sanity)
{
    std::string name;

    {
        const temp::Env env;
        name = env.name;

        EXPECT_NE(::getenv(name.c_str()), nullptr);
    }

    EXPECT_EQ(::getenv(name.c_str()), nullptr);
}

TEST(Value, String)
{
    const std::string value = random::string();

    const temp::Env env(value);

    EXPECT_EQ(utils::getenv<std::string>(env.name), value);
}

TEST(Value, Int)
{
    const auto value = random::number<int>();

    const temp::Env env(random::string(), value);
    EXPECT_EQ(utils::getenv<int>(env.name), value);
}

TEST(Value, Unsigned_long)
{
    const auto value = random::number<unsigned long>();

    const temp::Env env(random::string(), value);
    EXPECT_EQ(utils::getenv<unsigned long>(env.name), value);
}

TEST(Value, Bool)
{
    for (auto value : { true, false })
    {
        const temp::Env env(random::string(), value);
        EXPECT_EQ(utils::getenv<bool>(env.name), value);
    }
}

TEST(Name, Sanity)
{
    const std::string name = random::string();
    const std::string value = random::string();

    const temp::Env env(name, value);
    EXPECT_EQ(utils::getenv(name), value);
}

} // namespace runai::llm::streamer::utils::temp
