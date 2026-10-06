/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/request/request.h"

#include <gtest/gtest.h>
#include <atomic>
#include <memory>

#include "utils/threadpool/threadpool.h"
#include "utils/random/random.h"

namespace runai::llm::streamer::impl
{

TEST(Creation, Sanity)
{
    unsigned num_tasks = utils::random::number(1, 10);
    Request request(utils::random::number<size_t>(), utils::random::number(), utils::random::number(), num_tasks, utils::random::number<size_t>(), nullptr);
    std::atomic<unsigned int> finished = 0;
    unsigned retries = 5;

    auto pool = utils::ThreadPool<int>([&](int i, std::atomic<bool> &)
    {
        finished += request.finished();
    }, utils::random::number(1, 2 * num_tasks));

    // send task
    for (unsigned i = 0; i < num_tasks; ++i)
    {
        pool.push(i);
    }

    while (finished == 0 && retries > 0)
    {
        usleep(1000000);
        retries--;
    }

    EXPECT_EQ(finished, 1);
    EXPECT_EQ(request.ret(), common::ResponseCode::Success);
}

}; // namespace runai::llm::streamer::impl
