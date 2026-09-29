/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/temp/file/file.h"

#include <gtest/gtest.h>

#include <fcntl.h>

#include <utility>

#include "utils/fd/fd.h"

namespace runai::llm::streamer::utils::temp
{

TEST(Creation, NonExistingDirectory)
{
    EXPECT_THROW(temp::File f(random::string()), std::exception);
}

TEST(Creation, Name)
{
    const auto name = random::string();

    temp::File f(".", name);
    EXPECT_TRUE(Path::exists("./" + name));
}

TEST(Creation, RValue)
{
    temp::File a;

    const auto name = a.name;
    const auto path = a.path;

    {
        temp::File b(std::move(a));

        // verify it was moved
        EXPECT_EQ(b.name, name);
        EXPECT_EQ(b.path, path);

        // verify it still exists
        EXPECT_TRUE(temp::Path::exists(path));
    }

    // verify it was deleted
    EXPECT_FALSE(temp::Path::exists(path));
}

TEST(Creation, Operator_Assignment_RValue)
{
    std::string a_name, a_path, b_name, b_path;

    {
        temp::File a;

        a_name = a.name;
        a_path = a.path;

        {
            temp::File b;

            b_name = b.name;
            b_path = b.path;

            a = std::move(b);

            // verify it was moved
            EXPECT_EQ(a.name, b_name);
            EXPECT_EQ(a.path, b_path);

            // verify `a` was deleted
            EXPECT_FALSE(temp::Path::exists(a_path));

            // verify `b` still exists
            EXPECT_TRUE(temp::Path::exists(b_path));
        }

        // verify `b` still exists
        EXPECT_TRUE(temp::Path::exists(b_path));
    }

    // verify deleted
    EXPECT_FALSE(temp::Path::exists(b_path));
}

TEST(Sanity, Sanity)
{
    std::string path;

    {
        temp::File f;
        path = f.path;

        EXPECT_TRUE(temp::Path::exists(path));
    }

    EXPECT_FALSE(temp::Path::exists(path));
}

TEST(Sanity, Data)
{
    const auto dir  = ".";
    const auto name = random::string();
    auto size = random::number(100, 1000);
    const auto data = random::buffer(size);

    {
        temp::File f(dir, name, data);
        (void)f;

        EXPECT_TRUE(temp::Path::exists("./" + name));

        const auto expected = Fd::read(f.path);
        EXPECT_EQ(expected.size(), data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            EXPECT_EQ(data[i], expected[i]);
        }
    }

    EXPECT_FALSE(temp::Path::exists("./" + name));
}

TEST(Deletion, NonExisting)
{
    temp::Path f;
}

} // namespace runai::llm::streamer::utils::temp
