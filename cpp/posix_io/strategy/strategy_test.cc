/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/strategy/strategy.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "common/exception/exception.h"

namespace runai::llm::streamer::posix_io
{

// Every enumerator, so a value added later without a name fails here rather than logging "unknown".
const std::vector<Strategy> all = {
    Strategy::IoUringDirect,
    Strategy::IoUringBuffered,
    Strategy::LibaioDirect,
    Strategy::SyncBuffered,
};

TEST(Strategy, Name_Roundtrip)
{
    for (const auto strategy : all)
    {
        const std::string text = name(strategy);
        EXPECT_NE(text, "unknown");

        Strategy parsed;
        EXPECT_TRUE(strategy_from_name(text, parsed));
        EXPECT_EQ(parsed, strategy);
    }
}

TEST(Strategy, Name_Unknown)
{
    Strategy parsed;
    EXPECT_FALSE(strategy_from_name("uring", parsed));
    EXPECT_FALSE(strategy_from_name("", parsed));
    EXPECT_FALSE(strategy_from_name("IO_URING_DIRECT", parsed));   // case sensitive
    EXPECT_FALSE(strategy_from_name("threadpool", parsed));        // the name we deliberately dropped
}

TEST(Strategy, IsAsync)
{
    EXPECT_TRUE(is_async(Strategy::IoUringDirect));
    EXPECT_TRUE(is_async(Strategy::IoUringBuffered));
    EXPECT_TRUE(is_async(Strategy::LibaioDirect));

    // The sync reader is deliberately NOT an IoEngine - it is dispatched to its own pool.
    EXPECT_FALSE(is_async(Strategy::SyncBuffered));
}

TEST(Strategy, IsDirect)
{
    EXPECT_TRUE(is_direct(Strategy::IoUringDirect));
    EXPECT_TRUE(is_direct(Strategy::LibaioDirect));

    EXPECT_FALSE(is_direct(Strategy::IoUringBuffered));
    EXPECT_FALSE(is_direct(Strategy::SyncBuffered));
}

// libaio is async only with O_DIRECT. The flat enum exists so the buffered pair cannot be spelled.
TEST(Strategy, NoBufferedLibaio)
{
    Strategy parsed;
    EXPECT_FALSE(strategy_from_name("libaio_buffered", parsed));

    for (const auto strategy : all)
    {
        if (strategy == Strategy::LibaioDirect)
        {
            EXPECT_TRUE(is_direct(strategy));
        }
    }
}

TEST(Candidates, Single)
{
    const auto candidates = parse_candidates("sync_buffered");
    EXPECT_EQ(candidates.size(), 1);
    EXPECT_EQ(candidates[0], Strategy::SyncBuffered);
}

TEST(Candidates, OrderIsPreserved)
{
    const auto candidates = parse_candidates("io_uring_direct,io_uring_buffered,sync_buffered");
    ASSERT_EQ(candidates.size(), 3);
    EXPECT_EQ(candidates[0], Strategy::IoUringDirect);
    EXPECT_EQ(candidates[1], Strategy::IoUringBuffered);
    EXPECT_EQ(candidates[2], Strategy::SyncBuffered);

    // A preference order, so a permutation is a different configuration - not a set.
    const auto reversed = parse_candidates("sync_buffered,io_uring_direct");
    ASSERT_EQ(reversed.size(), 2);
    EXPECT_EQ(reversed[0], Strategy::SyncBuffered);
    EXPECT_EQ(reversed[1], Strategy::IoUringDirect);
}

TEST(Candidates, Whitespace)
{
    const auto candidates = parse_candidates("  io_uring_direct ,\tsync_buffered  ");
    ASSERT_EQ(candidates.size(), 2);
    EXPECT_EQ(candidates[0], Strategy::IoUringDirect);
    EXPECT_EQ(candidates[1], Strategy::SyncBuffered);
}

// A typo must not resolve to a fallback nobody asked for.
TEST(Candidates, UnknownNameThrows)
{
    EXPECT_THROW(parse_candidates("io_uring"), common::Exception);
    EXPECT_THROW(parse_candidates("io_uring_direct,threadpool"), common::Exception);
    EXPECT_THROW(parse_candidates("libaio_buffered"), common::Exception);
}

TEST(Candidates, EmptyThrows)
{
    EXPECT_THROW(parse_candidates(""), common::Exception);
    EXPECT_THROW(parse_candidates("   "), common::Exception);
}

TEST(Candidates, StrayCommaThrows)
{
    EXPECT_THROW(parse_candidates("io_uring_direct,"), common::Exception);
    EXPECT_THROW(parse_candidates(",sync_buffered"), common::Exception);
    EXPECT_THROW(parse_candidates("io_uring_direct,,sync_buffered"), common::Exception);
}

// A repeat is unreachable - the first occurrence decides - so it is a typo.
TEST(Candidates, DuplicateThrows)
{
    EXPECT_THROW(parse_candidates("sync_buffered,sync_buffered"), common::Exception);
    EXPECT_THROW(parse_candidates("io_uring_direct,sync_buffered,io_uring_direct"), common::Exception);
}

TEST(Strategy, Stream)
{
    std::stringstream ss;
    ss << Strategy::IoUringBuffered;
    EXPECT_EQ(ss.str(), "io_uring_buffered");
}

}; // namespace runai::llm::streamer::posix_io
