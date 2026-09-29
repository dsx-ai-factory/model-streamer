/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/fdlimit/fdlimit.h"

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

} // namespace runai::llm::streamer::utils
