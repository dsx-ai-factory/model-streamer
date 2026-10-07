/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/submissions/submissions_mgr.h"

#include <gtest/gtest.h>

#include <chrono>
#include <set>

#include "utils/random/random.h"

namespace runai::llm::streamer::impl
{

using Outcome = SubmissionsMgr::Result::Outcome;

TEST(SubmissionsMgr, GenerateUniqueAndNonZero)
{
    SubmissionsMgr mgr;

    const auto count = utils::random::number(100, 1000);
    std::set<unsigned> ids;
    for (unsigned i = 0; i < count; ++i)
    {
        const auto id = mgr.generate();
        EXPECT_NE(id, 0u);              // 0 is reserved
        EXPECT_EQ(ids.count(id), 0);    // distinct
        ids.insert(id);
    }
    EXPECT_EQ(ids.size(), count);
}

TEST(SubmissionsMgr, PendingUntilLastResponse)
{
    SubmissionsMgr mgr;

    const auto expected = utils::random::number(1, 50);
    const auto id = mgr.generate();
    mgr.add(id, expected, /*total_bytes=*/1024);
    EXPECT_EQ(mgr.size(), 1u);

    for (unsigned i = 0; i + 1 < expected; ++i)
    {
        EXPECT_EQ(mgr.consume(id).outcome, Outcome::Pending);
    }

    EXPECT_EQ(mgr.consume(id).outcome, Outcome::Completed); // last response
    EXPECT_EQ(mgr.size(), 0u);                              // forgotten on completion
}

TEST(SubmissionsMgr, ThroughputComputedFromClock)
{
    // Injected clock makes the throughput math deterministic.
    std::chrono::steady_clock::time_point clock = std::chrono::steady_clock::now();
    SubmissionsMgr mgr([&]() { return clock; });

    const size_t total_bytes = 1000;
    const auto id = mgr.generate();
    mgr.add(id, /*expected=*/1, total_bytes); // submit_time = clock (now)

    clock += std::chrono::milliseconds(100);  // 100 ms elapse before the (only) response
    const auto r = mgr.consume(id);

    EXPECT_EQ(r.outcome, Outcome::Completed);
    EXPECT_EQ(r.total_bytes, total_bytes);
    EXPECT_EQ(r.elapsed_ms, 100);
    EXPECT_EQ(r.throughput_bps, total_bytes * 1000 / 100); // 1000 bytes in 0.1s -> 10000 B/s
}

TEST(SubmissionsMgr, ZeroElapsedThroughputIsZero)
{
    // same clock at add and consume -> elapsed 0 -> no division by zero
    std::chrono::steady_clock::time_point clock = std::chrono::steady_clock::now();
    SubmissionsMgr mgr([&]() { return clock; });

    const auto id = mgr.generate();
    mgr.add(id, 1, 4096);
    const auto r = mgr.consume(id);

    EXPECT_EQ(r.outcome, Outcome::Completed);
    EXPECT_EQ(r.elapsed_ms, 0);
    EXPECT_EQ(r.throughput_bps, 0u);
}

TEST(SubmissionsMgr, ConsumeUnknownAsserts)
{
    SubmissionsMgr mgr;
    // consuming a response for an id that was never registered is an accounting bug -> ASSERT throws
    EXPECT_THROW(mgr.consume(12345), std::exception);
}

TEST(SubmissionsMgr, DuplicateRegisterAsserts)
{
    SubmissionsMgr mgr;
    const auto id = mgr.generate();
    mgr.add(id, 1, 10);
    EXPECT_THROW(mgr.add(id, 1, 10), std::exception); // re-registering a live id is a bug
}

TEST(SubmissionsMgr, SizeTracksLiveSubmissions)
{
    SubmissionsMgr mgr;

    const auto a = mgr.generate();
    const auto b = mgr.generate();
    mgr.add(a, 2, 100);
    mgr.add(b, 1, 200);
    EXPECT_EQ(mgr.size(), 2u);

    EXPECT_EQ(mgr.consume(b).outcome, Outcome::Completed); // b done
    EXPECT_EQ(mgr.size(), 1u);

    EXPECT_EQ(mgr.consume(a).outcome, Outcome::Pending);   // a still has one left
    EXPECT_EQ(mgr.size(), 1u);
    EXPECT_EQ(mgr.consume(a).outcome, Outcome::Completed); // a done
    EXPECT_EQ(mgr.size(), 0u);
}

} // namespace runai::llm::streamer::impl
