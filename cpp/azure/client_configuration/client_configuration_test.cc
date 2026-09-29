/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "azure/client_configuration/client_configuration.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <thread>

#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl::azure
{

namespace
{

// The formula the configuration implements, restated so the tests assert against the machine they run
// on rather than a literal thread count.
unsigned expected(unsigned readers)
{
    const unsigned nprocs = std::thread::hardware_concurrency();
    const unsigned floor = nprocs == 0 ? 8U : 1U;
    return std::max(floor, nprocs * 2 / readers);
}

} // namespace

// The variable the formula reads is cleared before each test, so a test states only what it SETS.
// Without this, a run that inherits RUNAI_STREAMER_PROCESS_GROUP_SIZE fails these against correct
// code - `bazel test` scrubs the environment, but `--test_env` and a direct binary run do not.
class AzureConcurrency : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        _group = std::make_unique<utils::temp::UnsetEnv>(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"));
    }

    void TearDown() override
    {
        _group.reset();
    }

    std::unique_ptr<utils::temp::UnsetEnv> _group;
};

// The threads are sized by the number of CLIENTS the caller will build, which reaches the plugin as a
// client parameter - no environment variable is read for it here.
TEST_F(AzureConcurrency, Threads_Follow_The_Reader_Count)
{
    for (const unsigned readers : { 2U, 4U, 8U, 16U })
    {
        EXPECT_EQ(ClientConfiguration(readers).max_concurrency, expected(readers)) << readers;
    }
}

// The property the divisor exists for: N clients with nprocs*2/N threads each is the same total
// whatever N is.
//
// Only reader counts that DIVIDE the total are checked. The division truncates otherwise, and that
// loss belongs to the formula rather than to the count it was given.
TEST_F(AzureConcurrency, The_Total_Thread_Count_Does_Not_Move)
{
    const unsigned nprocs = std::thread::hardware_concurrency();
    ASSERT_GT(nprocs, 0u) << "the formula falls back to a fixed 8 with no cores to divide";

    const unsigned total = nprocs * 2;
    unsigned checked = 0;

    for (unsigned readers = 2; readers <= total; ++readers)
    {
        if (total % readers != 0)
        {
            continue;
        }

        EXPECT_EQ(readers * ClientConfiguration(readers).max_concurrency, total) << readers;
        ++checked;
    }

    EXPECT_GT(checked, 0u) << "no reader count divided " << total << ", so nothing was asserted";
}

// Only a caller outside the streamer can send zero, and dividing by it would be a crash rather than a
// misconfiguration. Floored to one - the streamer itself always states a count.
TEST_F(AzureConcurrency, Zero_Is_Floored_Rather_Than_Dividing_By_Zero)
{
    EXPECT_EQ(ClientConfiguration(0).max_concurrency, expected(1));
}

// Azure divides by the process group size too, so one setting describes the node at any tensor
// parallel size.
TEST_F(AzureConcurrency, The_Process_Group_Divides_As_Well)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 2UL);

    const unsigned nprocs = std::thread::hardware_concurrency();
    const unsigned floor = nprocs == 0 ? 8U : 1U;

    EXPECT_EQ(ClientConfiguration(4).max_concurrency, std::max(floor, nprocs * 2 / (4 * 2)));
}

}; // namespace runai::llm::streamer::impl::azure
