/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/io_engine/io_engine.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <climits>

#include "posix_io/strategy/strategy.h"

namespace runai::llm::streamer::posix_io
{

// The page sizes that matter, as constants - possible only because the page size is an argument.
//
// Every machine we run on has 4 KiB pages, so testing sysconf() alone passes whether the value is
// computed or hard-coded: measured, `return 0x7FFFF000;` passed the whole suite. The 64 KiB row is
// what the computation exists for, and nothing on this host reaches it otherwise.
TEST(MaxReadBytesize, KnownPageSizes)
{
    EXPECT_EQ(max_read_bytesize(4096), 0x7FFFF000ul);
    EXPECT_EQ(max_read_bytesize(16384), 0x7FFFC000ul);
    EXPECT_EQ(max_read_bytesize(65536), 0x7FFF0000ul);

    // 60 KiB below the 4 KiB cap - the gap where a hard-coded constant admits too-large requests.
    EXPECT_EQ(max_read_bytesize(4096) - max_read_bytesize(65536), 60ul * 1024);
}

// The mask must hold for every power of two, not just the three above.
TEST(MaxReadBytesize, PageAlignedForEveryPageSize)
{
    for (size_t page = 4096; page <= (1ul << 20); page <<= 1)
    {
        const auto cap = max_read_bytesize(page);

        EXPECT_EQ(cap % page, 0) << "cap " << cap << " is not a whole number of " << page << "-byte pages";
        EXPECT_LT(cap, static_cast<size_t>(INT_MAX));

        // The largest page-aligned value below INT_MAX, so one more page overshoots.
        EXPECT_GT(cap + page, static_cast<size_t>(INT_MAX));
    }
}

// The no-argument form must agree with the arithmetic for this host's page.
TEST(MaxReadBytesize, HostFormUsesTheHostPageSize)
{
    const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
    ASSERT_GT(page, 0);

    EXPECT_EQ(max_read_bytesize(), max_read_bytesize(page));
}

TEST(MaxReadBytesize, IsALargeCeilingNotALimit)
{
    // Any plausible chunk size is far below this, so the clamp should be unreachable in practice.
    EXPECT_GT(max_read_bytesize(), static_cast<size_t>(1) << 30);
}

// A half-filled config should fail loudly: depth 0 admits nothing and chunk 0 reads nothing.
TEST(AsyncIoConfig, DefaultsAreInert)
{
    AsyncIoConfig config;

    EXPECT_EQ(config.depth, 0);
    EXPECT_EQ(config.chunk_bytesize, 0);
}

// The default must be the safe one. Defaulting to direct would give a caller who forgot to set it
// O_DIRECT alignment rules on a file that may not support them, and -EINVAL would then look like our
// own bug.
TEST(FileRef, DefaultsToBuffered)
{
    FileRef file;

    EXPECT_EQ(file.fd, -1);
    EXPECT_FALSE(file.direct);
}

// 1 means "no constraint", not 0 - zero would make an alignment check divide by zero or be trivially
// true, depending on how it was written.
TEST(Limits, DefaultAlignmentsAreOne)
{
    Limits limits;

    EXPECT_EQ(limits.offset_alignment, 1);
    EXPECT_EQ(limits.buffer_alignment, 1);
}

}; // namespace runai::llm::streamer::posix_io
