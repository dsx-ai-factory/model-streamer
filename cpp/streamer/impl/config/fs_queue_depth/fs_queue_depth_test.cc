/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/config/fs_queue_depth/fs_queue_depth.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "common/exception/exception.h"

namespace runai::llm::streamer::impl
{

// A plain number is the whole value, and stays the whole meaning of the variable for anyone who never
// needs per-type entries.
TEST(FsQueueDepth, A_Plain_Number_Applies_Everywhere)
{
    const auto depth = FsQueueDepth::parse("512");

    EXPECT_EQ(depth.default_value(), 512u);
    EXPECT_TRUE(depth.entries().empty());

    for (const auto * type : { "ext4", "nfs", "nfs4", "virtiofs", "overlay", "" })
    {
        EXPECT_EQ(depth.for_type(type), 512u) << type;
    }
}

TEST(FsQueueDepth, Type_Entries_Override_The_Default)
{
    const auto depth = FsQueueDepth::parse("512,nfs=64,virtiofs=256");

    EXPECT_EQ(depth.default_value(), 512u);
    ASSERT_EQ(depth.entries().size(), 2u);

    EXPECT_EQ(depth.for_type("nfs"), 64u);
    EXPECT_EQ(depth.for_type("virtiofs"), 256u);
    EXPECT_EQ(depth.for_type("ext4"), 512u) << "a type with no entry falls back to the default";
}

// The reason keys are prefixes: the kernel reports `nfs` for a v3 mount and `nfs4` for a v4 one, and a
// user should not have to know which they have.
TEST(FsQueueDepth, A_Key_Matches_As_A_Prefix)
{
    const auto depth = FsQueueDepth::parse("512,nfs=64,fuse=128");

    EXPECT_EQ(depth.for_type("nfs"), 64u);
    EXPECT_EQ(depth.for_type("nfs4"), 64u);
    EXPECT_EQ(depth.for_type("fuse.gvfsd-fuse"), 128u);

    EXPECT_EQ(depth.for_type("nf"), 512u) << "a shorter type is not matched by a longer key";
}

// First match, not longest match. Written down because it is the one surprising case: `nfs4` mounts
// take the `nfs` entry when it is listed first, and writing the longer key first gives the other
// answer.
TEST(FsQueueDepth, The_First_Matching_Entry_Wins)
{
    const auto first_shorter = FsQueueDepth::parse("512,nfs=64,nfs4=32");
    EXPECT_EQ(first_shorter.for_type("nfs4"), 64u);

    const auto first_longer = FsQueueDepth::parse("512,nfs4=32,nfs=64");
    EXPECT_EQ(first_longer.for_type("nfs4"), 32u);
    EXPECT_EQ(first_longer.for_type("nfs"), 64u);
}

TEST(FsQueueDepth, Whitespace_And_Case_Are_Ignored)
{
    const auto depth = FsQueueDepth::parse("  512 , NFS = 64 , VirtioFS=256 ");

    EXPECT_EQ(depth.default_value(), 512u);
    EXPECT_EQ(depth.for_type("nfs4"), 64u);
    EXPECT_EQ(depth.for_type("VIRTIOFS"), 256u);
}

// Every malformed value is refused at parse time, with the variable named: a value the user meant to
// set and mistyped must never read as a working one.
TEST(FsQueueDepth, Malformed_Values_Are_Refused)
{
    for (const auto * bad : {
            "",                     // nothing at all
            "   ",                  // nothing but spaces
            "nfs=64",               // no default, so a mount matching nothing has no answer
            "abc",                  // not a number
            "512,nfs",              // an entry with no value
            "512,=64",              // an entry with no type
            "512,nfs=",             // an entry with an empty value
            "512,nfs=abc",          // a non-numeric value
            "0",                    // a reader that admits nothing
            "512,nfs=0",            // the same, for one type
            "-1",                   // negative, which stoul would otherwise wrap
            "512,nfs=-4",
            "512,nfs=64,nfs=32",    // a repeat: both numbers were written on purpose
            "512,NFS=64,nfs=32",    // the same repeat, differently cased
            "99999999999999999999", // beyond unsigned
         })
    {
        EXPECT_THROW(FsQueueDepth::parse(bad), common::Exception) << "accepted: '" << bad << "'";
    }
}

TEST(FsQueueDepth, A_Constructed_Value_Behaves_Like_A_Plain_Number)
{
    const FsQueueDepth depth(64);

    EXPECT_EQ(depth.default_value(), 64u);
    EXPECT_TRUE(depth.entries().empty());
    EXPECT_EQ(depth.for_type("nfs"), 64u);
}

// The log line an operator reads back. It has to show the entries, or a value that parsed differently
// from what was intended is invisible.
TEST(FsQueueDepth, It_Prints_What_It_Parsed)
{
    std::ostringstream stream;
    stream << FsQueueDepth::parse("512,nfs=64,virtiofs=256");

    EXPECT_EQ(stream.str(), "512, nfs=64, virtiofs=256");
}


}; // namespace runai::llm::streamer::impl
