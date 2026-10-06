/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/async_io_stats/async_io_stats.h"

#include <gtest/gtest.h>

#include <sstream>
#include <thread>
#include <vector>

namespace runai::llm::streamer::impl
{

namespace
{

using posix_io::Strategy;

SubmissionStats made(SubmissionId id, unsigned shared_mounts = 0)
{
    SubmissionStats stats;
    stats.submission_id = id;
    stats.shared_engine_mounts = shared_mounts;
    return stats;
}

} // namespace

// The reason this class exists: a test must be able to read the numbers, not grep a log line.
TEST(AsyncIoStats, Finds_A_Submission_By_Id)
{
    AsyncIoStats stats;

    stats.record(made(7, 2));

    SubmissionStats found;
    ASSERT_TRUE(stats.find(7, found));
    EXPECT_EQ(found.shared_engine_mounts, 2u);

    EXPECT_FALSE(stats.find(8, found)) << "an unknown id must not report someone else's numbers";
}

// A submission served half by the synchronous pool reports zero for engine counters. That is only
// readable if each submission is kept apart.
TEST(AsyncIoStats, Submissions_Are_Kept_Apart)
{
    AsyncIoStats stats;

    SubmissionStats sync_only = made(1);
    sync_only.files.push_back({ "/dev/shm/a", Strategy::SyncBuffered });
    stats.record(sync_only);

    SubmissionStats engine_served = made(2, 3);
    engine_served.files.push_back({ "/mnt/a", Strategy::IoUringBuffered });
    stats.record(engine_served);

    SubmissionStats first;
    SubmissionStats second;
    ASSERT_TRUE(stats.find(1, first));
    ASSERT_TRUE(stats.find(2, second));

    EXPECT_EQ(first.shared_engine_mounts, 0u);
    EXPECT_EQ(second.shared_engine_mounts, 3u);
    EXPECT_EQ(first.files[0].strategy, Strategy::SyncBuffered);
    EXPECT_EQ(second.files[0].strategy, Strategy::IoUringBuffered);
}

// A streamer can run for hours. The list must not grow without limit, and a reader must be able to
// tell that it is no longer complete.
TEST(AsyncIoStats, Old_Submissions_Are_Dropped)
{
    AsyncIoStats stats;

    for (size_t i = 0; i < AsyncIoStats::MaxSubmissions + 10; ++i)
    {
        stats.record(made(i + 1, 1));
    }

    EXPECT_EQ(stats.submissions().size(), AsyncIoStats::MaxSubmissions);
    EXPECT_EQ(stats.dropped(), 10u);

    SubmissionStats found;
    EXPECT_FALSE(stats.find(1, found)) << "the oldest are dropped first";
    EXPECT_TRUE(stats.find(AsyncIoStats::MaxSubmissions + 10, found)) << "the newest is kept";

    // Dropping must not lose the totals - otherwise a long run would look like it did less work.
    EXPECT_EQ(stats.total().shared_engine_mounts, 1u);
}

// Which reader served which file. A split submission is the case this answers.
TEST(AsyncIoStats, Records_The_Strategy_Per_File)
{
    AsyncIoStats stats;

    SubmissionStats submission = made(1);
    submission.files.push_back({ "/mnt/disk/model.safetensors", Strategy::IoUringBuffered });
    submission.files.push_back({ "/dev/shm/cache.safetensors", Strategy::SyncBuffered });
    stats.record(submission);

    SubmissionStats found;
    ASSERT_TRUE(stats.find(1, found));
    ASSERT_EQ(found.files.size(), 2u);
    EXPECT_EQ(found.files[0].strategy, Strategy::IoUringBuffered);
    EXPECT_EQ(found.files[1].strategy, Strategy::SyncBuffered);

    // The printed line counts the engine files rather than listing every file: a model has hundreds.
    std::ostringstream os;
    os << found;
    EXPECT_NE(os.str().find("2 (1 through an engine)"), std::string::npos) << os.str();
}

// Submissions run at the same time, so their workers report at the same time.
TEST(AsyncIoStats, Concurrent_Recording)
{
    AsyncIoStats stats;

    constexpr unsigned threads = 8;
    constexpr unsigned each = 20;

    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t)
    {
        workers.emplace_back([&stats, t]()
        {
            for (unsigned i = 0; i < each; ++i)
            {
                stats.record(made(t * each + i + 1, 1));
            }
        });
    }
    for (auto & worker : workers)
    {
        worker.join();
    }

    // Every record counted once, whichever order they arrived in.
    EXPECT_EQ(stats.submissions().size() + stats.dropped(), threads * each);
}


// The peak and the average answer different questions, and only the average says whether the window
// STAYED full. A run that touches 64 outstanding once and then averages three reports the same
// achieved_depth as one that sits at 64 throughout.
TEST(AsyncIoCounters, Average_Inflight_Is_Time_Weighted)
{
    AsyncIoCounters counters;

    // Two levels: 8 outstanding for 1 ms, then 2 outstanding for 3 ms.
    // (8*1 + 2*3) / 4 = 3.5
    counters.inflight_nanos = 8 * 1000000ULL + 2 * 3000000ULL;
    counters.observed_nanos = 4000000ULL;

    EXPECT_DOUBLE_EQ(counters.average_inflight(), 3.5)
        << "a long spell at a low level must outweigh a brief spike";
}

TEST(AsyncIoCounters, Average_Inflight_Is_Zero_Before_Anything_Is_Observed)
{
    AsyncIoCounters counters;
    EXPECT_EQ(counters.average_inflight(), 0.0) << "no division by zero, and no invented number";
}

// Summed rather than maxed, unlike achieved_depth - which is why the two parts are carried separately:
// means of different durations cannot be added, numerators and denominators can.
TEST(AsyncIoCounters, Average_Inflight_Combines_Across_Workers)
{
    AsyncIoCounters a;
    a.inflight_nanos = 10 * 1000000ULL;   // 10 outstanding for 1 ms
    a.observed_nanos = 1000000ULL;

    AsyncIoCounters b;
    b.inflight_nanos = 2 * 3000000ULL;    // 2 outstanding for 3 ms
    b.observed_nanos = 3000000ULL;

    a += b;

    // (10 + 6) / 4 = 4, not the (10 + 2) / 2 = 6 that averaging the averages would give.
    EXPECT_DOUBLE_EQ(a.average_inflight(), 4.0);
}

}; // namespace runai::llm::streamer::impl
