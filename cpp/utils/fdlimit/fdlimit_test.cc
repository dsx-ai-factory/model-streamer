/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fdlimit/fdlimit.h"

#include <fcntl.h>
#include <unistd.h>

#include <vector>

#include <gtest/gtest.h>

#include "utils/random/random.h"

namespace runai::llm::streamer::utils
{

TEST(GetFdLimit, Sanity)
{
    EXPECT_NO_THROW(get_cur_file_descriptors());
    EXPECT_NO_THROW(get_max_file_descriptors());

    auto res = get_cur_file_descriptors();
    auto res_max = get_max_file_descriptors();
    EXPECT_NE(res, 0);
    EXPECT_NE(res_max, 0);
    EXPECT_GE(res_max, res);
}

TEST(SetFdLimit, Sanity)
{
    auto current = get_cur_file_descriptors();
    auto hard_limit = get_max_file_descriptors();
    auto expected = utils::random::number<rlim_t>(1, hard_limit);

    {
        FdLimitSetter temp(expected);
        EXPECT_EQ(get_cur_file_descriptors(), expected);
        EXPECT_EQ(get_max_file_descriptors(), hard_limit);
    }
    auto res = get_cur_file_descriptors();
    EXPECT_EQ(res, current);
    EXPECT_EQ(get_max_file_descriptors(), hard_limit);
}

TEST(SetFdLimit, Hard_Limit)
{
    auto current = get_cur_file_descriptors();
    auto hard_limit = get_max_file_descriptors();
    auto expected = utils::random::number<rlim_t>(hard_limit +1 , 2 * hard_limit);
    {
        FdLimitSetter temp(expected);
        EXPECT_EQ(get_cur_file_descriptors(), hard_limit);
        EXPECT_EQ(get_max_file_descriptors(), hard_limit);
    }
    auto res = get_max_file_descriptors();
    EXPECT_EQ(res, current);
    EXPECT_EQ(get_max_file_descriptors(), hard_limit);
}

TEST(CountOpenFds, Counts_Opened_And_Closed)
{
    const auto before = count_open_fds();
    ASSERT_NE(before, 0);

    std::vector<int> fds;
    for (int i = 0; i < 5; ++i)
    {
        fds.push_back(open("/dev/null", O_RDONLY));
        ASSERT_GE(fds.back(), 0);
    }
    EXPECT_EQ(count_open_fds(), before + 5);

    for (int fd : fds)
    {
        close(fd);
    }
    EXPECT_EQ(count_open_fds(), before);
}

TEST(RaiseFdLimit, Raises_And_Restores_The_Original)
{
    const auto current = get_cur_file_descriptors();
    const auto hard_limit = get_max_file_descriptors();
    if (current + 2 > hard_limit)
    {
        GTEST_SKIP() << "no room to raise the soft limit";
    }

    {
        FdLimitSetter setter(current);
        EXPECT_TRUE(setter.raise_to(current + 1));
        EXPECT_EQ(get_cur_file_descriptors(), current + 1);

        // raised again, never lowered
        EXPECT_TRUE(setter.raise_to(current + 2));
        EXPECT_EQ(get_cur_file_descriptors(), current + 2);
        EXPECT_TRUE(setter.raise_to(current));
        EXPECT_EQ(get_cur_file_descriptors(), current + 2);
    }

    EXPECT_EQ(get_cur_file_descriptors(), current);
}

TEST(RaiseFdLimit, Capped_At_The_Hard_Limit)
{
    const auto current = get_cur_file_descriptors();
    const auto hard_limit = get_max_file_descriptors();
    if (hard_limit == RLIM_INFINITY)
    {
        GTEST_SKIP() << "no hard limit to exceed";
    }

    {
        FdLimitSetter setter(current);
        EXPECT_FALSE(setter.raise_to(hard_limit + 1));
        EXPECT_EQ(get_cur_file_descriptors(), hard_limit);
    }

    EXPECT_EQ(get_cur_file_descriptors(), current);
}

} // namespace runai::llm::streamer::utils
