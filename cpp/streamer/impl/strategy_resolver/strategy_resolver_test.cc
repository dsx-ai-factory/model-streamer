/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/strategy_resolver/strategy_resolver.h"

#include <gtest/gtest.h>

#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#include <string>

namespace runai::llm::streamer::impl
{

namespace
{

using posix_io::Strategy;

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif

// The kernel directly, not IoUringProbe - the resolver consults the probe, so asking the probe here
// would compare the resolver against its own input rather than against the truth.
bool ring_works()
{
    struct params_stub { char opaque[512]; } params;
    std::memset(&params, 0, sizeof(params));

    const int fd = ::syscall(__NR_io_uring_setup, 8, &params);
    if (fd < 0)
    {
        return false;
    }

    ::close(fd);
    return true;
}


// Say that one strategy cannot be served here, and let the probes answer for the rest.
//
// Needed because every strategy is available on a normal host: sync_buffered always, libaio nearly
// always, io_uring wherever seccomp permits. Without this there is no candidate that reliably fails,
// so a rejected head and an exhausted list cannot be reached.
//
// These tests used to name `libaio_direct` for that, which held only while libaio had no engine.
StrategyResolver::Availability all_but(Strategy unavailable)
{
    return [unavailable](Strategy strategy)
    {
        return strategy == unavailable ? common::ResponseCode::FsStrategyUnavailable
                                       : common::ResponseCode::Success;
    };
}

} // namespace

// The synchronous reader needs nothing probed, so it always wins if it is reachable. This is the
// default, and it is what keeps io_uring off until the measurement says otherwise.
TEST(StrategyResolver, Sync_Buffered_Always_Resolves)
{
    StrategyResolver resolver("sync_buffered");

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
    EXPECT_EQ(resolver.resolved(), Strategy::SyncBuffered);
}

// First the host can provide wins, in list order - so an unavailable head falls through to the tail
// rather than failing the whole list.
TEST(StrategyResolver, Takes_The_First_Available_Candidate)
{
    // The head cannot be served here, so the walk must go on rather than fail the whole list.
    StrategyResolver resolver("libaio_direct,io_uring_buffered,sync_buffered",
                              all_but(Strategy::LibaioDirect));

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);

    EXPECT_EQ(resolver.resolved(), Strategy::IoUringBuffered)
        << "the second candidate is available, so the walk stops there";
}

// Both io_uring strategies use the same ring, so both are available wherever the ring is. Whether a
// FILE is really opened with O_DIRECT is decided later, per file, from the mount and from congruence.
TEST(StrategyResolver, Direct_And_Buffered_Are_Equally_Available)
{
    StrategyResolver direct("io_uring_direct,sync_buffered");
    StrategyResolver buffered("io_uring_buffered,sync_buffered");

    ASSERT_EQ(direct.resolve(), common::ResponseCode::Success);
    ASSERT_EQ(buffered.resolve(), common::ResponseCode::Success);

    if (ring_works())
    {
        EXPECT_EQ(direct.resolved(), Strategy::IoUringDirect);
        EXPECT_EQ(buffered.resolved(), Strategy::IoUringBuffered);
    }
    else
    {
        EXPECT_EQ(direct.resolved(), Strategy::SyncBuffered);
        EXPECT_EQ(buffered.resolved(), Strategy::SyncBuffered);
    }
}

// Exhaustion is an error, never a silent fall-through. An operator who asked for io_uring and got
// the synchronous reader without being told has no way to discover it.
TEST(StrategyResolver, Exhausted_List_Is_An_Error)
{
    // Nothing follows the one candidate, and it cannot be served here.
    StrategyResolver resolver("libaio_direct", all_but(Strategy::LibaioDirect));

    // The SPECIFIC code, not just "not Success". These used to report UnsupportedBackendMix, whose
    // message is about mixing S3, GCS and Azure - so an operator was sent to object storage for a
    // filesystem problem. A test asserting only "not Success" cannot notice that.
    EXPECT_EQ(resolver.resolve(), common::ResponseCode::FsStrategyUnavailable);
    EXPECT_FALSE(resolver.is_resolved());
}

// A typo must not become a fallback nobody asked for.
TEST(StrategyResolver, Unknown_Name_Is_Rejected)
{
    StrategyResolver resolver("io_uring_bufferd");   // sic

    EXPECT_EQ(resolver.resolve(), common::ResponseCode::InvalidParameterError);
    EXPECT_FALSE(resolver.is_resolved());

    StrategyResolver settable("sync_buffered");
    EXPECT_EQ(settable.set_candidates("nonsense"), common::ResponseCode::InvalidParameterError);
}

// Idempotent: every submission after the first calls this, and must get the same answer without
// re-deciding.
TEST(StrategyResolver, Resolve_Is_Idempotent)
{
    StrategyResolver resolver("sync_buffered");

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
    const auto first = resolver.resolved();

    for (int i = 0; i < 3; ++i)
    {
        ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
        EXPECT_EQ(resolver.resolved(), first);
    }
}

// Set before use: the caller's list replaces the default.
TEST(StrategyResolver, Set_Candidates_Overrides_The_Default)
{
    StrategyResolver resolver("libaio_direct", all_but(Strategy::LibaioDirect));   // cannot be served

    ASSERT_EQ(resolver.set_candidates("sync_buffered"), common::ResponseCode::Success);
    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
    EXPECT_EQ(resolver.resolved(), Strategy::SyncBuffered);
}

// First set wins, as for credentials: the same value again is a no-op, a different one is refused.
// Silently replacing a value the caller may already have acted on is worse than refusing.
TEST(StrategyResolver, First_Set_Wins)
{
    StrategyResolver resolver("sync_buffered");

    ASSERT_EQ(resolver.set_candidates("sync_buffered"), common::ResponseCode::Success);
    EXPECT_EQ(resolver.set_candidates("sync_buffered"), common::ResponseCode::Success)
        << "the same value again must be a no-op, not a conflict";
    // Conflict, so the message tells the operator to set it once - not that they mixed S3 with GCS.
    EXPECT_EQ(resolver.set_candidates("io_uring_buffered,sync_buffered"),
              common::ResponseCode::FsStrategyConflict);
}

// THE HOLE that first-set-wins alone leaves open.
//
// Nobody sets anything; the first request resolves from the default; only then does the caller set a
// strategy. Under first-set-wins alone there is no earlier value to conflict with, so this would
// return Success and change nothing at all - a setter that reports success and does nothing.
TEST(StrategyResolver, Set_After_Resolution_Is_Rejected)
{
    StrategyResolver resolver("sync_buffered");

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
    ASSERT_EQ(resolver.resolved(), Strategy::SyncBuffered);

    EXPECT_EQ(resolver.set_candidates("io_uring_buffered,sync_buffered"),
              common::ResponseCode::FsStrategyConflict)
        << "accepting this would report success and have no effect";

    EXPECT_EQ(resolver.resolved(), Strategy::SyncBuffered) << "and it must not have changed anything";
}

// The comparison is against what was actually WALKED, which may have come from the environment. So
// re-setting the value already in force is still a no-op rather than a conflict.
TEST(StrategyResolver, Set_After_Resolution_With_The_Same_List_Is_Accepted)
{
    StrategyResolver resolver("sync_buffered");

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::Success);
    EXPECT_EQ(resolver.set_candidates("sync_buffered"), common::ResponseCode::Success);
}

// Even a failed resolution fixes the list, so a later set cannot silently change the answer for a
// streamer that has already reported an error to its caller.
TEST(StrategyResolver, Set_After_Failed_Resolution_Is_Rejected)
{
    StrategyResolver resolver("libaio_direct", all_but(Strategy::LibaioDirect));

    ASSERT_EQ(resolver.resolve(), common::ResponseCode::FsStrategyUnavailable);

    EXPECT_EQ(resolver.set_candidates("sync_buffered"), common::ResponseCode::FsStrategyConflict);
    EXPECT_EQ(resolver.resolved_from(), "libaio_direct");
}

}; // namespace runai::llm::streamer::impl
