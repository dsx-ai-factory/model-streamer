/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/object_storage_worker/object_storage_retry.h"

#include <gtest/gtest.h>

#include <chrono>

namespace runai::llm::streamer::impl
{

namespace
{

std::chrono::milliseconds fixed_delay(unsigned)
{
    return std::chrono::milliseconds(50);
}

} // namespace

TEST(ObjectStorageRetryTest, DisabledAllowsInitialAttemptButDoesNotScheduleRetry)
{
    ObjectStorageRetry retry(ObjectStorageRetry::Duration::zero(), fixed_delay);
    ObjectStorageRetry::State state;
    const ObjectStorageRetry::TimePoint now;

    EXPECT_FALSE(retry.enabled());
    EXPECT_TRUE(retry.begin_attempt(state, now));
    EXPECT_FALSE(retry.schedule(state, 1, now).has_value());
    EXPECT_EQ(retry.retry_count(state), 0u);
    EXPECT_FALSE(retry.has_pending());
}

TEST(ObjectStorageRetryTest, SchedulesAndPromotesRetryWhenItBecomesDue)
{
    using namespace std::chrono_literals;

    ObjectStorageRetry retry(1s, fixed_delay);
    ObjectStorageRetry::State state;
    const ObjectStorageRetry::TimePoint started;
    EXPECT_TRUE(retry.enabled());
    ASSERT_TRUE(retry.begin_attempt(state, started));

    const auto failed_at = started + 10ms;
    const auto scheduled = retry.schedule(state, 7, failed_at);
    ASSERT_TRUE(scheduled.has_value());
    EXPECT_EQ(scheduled->delay, 50ms);
    EXPECT_EQ(scheduled->retry_count, 1u);
    EXPECT_EQ(retry.retry_count(state), 1u);
    EXPECT_TRUE(retry.has_pending());

    const auto due = retry.next_due();
    ASSERT_TRUE(due.has_value());
    EXPECT_EQ(due.value(), failed_at + 50ms);

    EXPECT_FALSE(retry.pop_due(failed_at + 49ms).has_value());

    const auto promoted = retry.pop_due(failed_at + 50ms);
    ASSERT_TRUE(promoted.has_value());
    EXPECT_EQ(promoted.value(), 7u);
    EXPECT_FALSE(retry.has_pending());
    EXPECT_TRUE(retry.begin_attempt(state, failed_at + 50ms));
}

TEST(ObjectStorageRetryTest, DoesNotScheduleBackoffThatReachesDeadline)
{
    using namespace std::chrono_literals;

    ObjectStorageRetry retry(100ms, fixed_delay);
    ObjectStorageRetry::State state;
    const ObjectStorageRetry::TimePoint started;
    ASSERT_TRUE(retry.begin_attempt(state, started));

    // 50ms of backoff from 50ms reaches the 100ms deadline exactly; submit() would reject that attempt.
    EXPECT_FALSE(retry.schedule(state, 1, started + 50ms).has_value());
    EXPECT_EQ(retry.retry_count(state), 0u);
    EXPECT_FALSE(retry.has_pending());
}

TEST(ObjectStorageRetryTest, RejectsRetryPromotedAfterDeadline)
{
    using namespace std::chrono_literals;

    ObjectStorageRetry retry(100ms, fixed_delay);
    ObjectStorageRetry::State state;
    const ObjectStorageRetry::TimePoint started;
    ASSERT_TRUE(retry.begin_attempt(state, started));
    ASSERT_TRUE(retry.schedule(state, 1, started).has_value());

    ASSERT_TRUE(retry.pop_due(started + 100ms).has_value());
    EXPECT_FALSE(retry.begin_attempt(state, started + 100ms));
}

TEST(ObjectStorageRetryTest, ClearDropsAllDelayedRetries)
{
    using namespace std::chrono_literals;

    ObjectStorageRetry retry(1s, fixed_delay);
    ObjectStorageRetry::State first;
    ObjectStorageRetry::State second;
    const ObjectStorageRetry::TimePoint started;
    ASSERT_TRUE(retry.begin_attempt(first, started));
    ASSERT_TRUE(retry.begin_attempt(second, started));
    ASSERT_TRUE(retry.schedule(first, 1, started).has_value());
    ASSERT_TRUE(retry.schedule(second, 2, started).has_value());

    retry.clear();

    EXPECT_FALSE(retry.has_pending());
    EXPECT_FALSE(retry.next_due().has_value());
}

}; // namespace runai::llm::streamer::impl
