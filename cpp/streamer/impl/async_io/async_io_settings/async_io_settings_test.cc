/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/async_io_settings/async_io_settings.h"

#include <gtest/gtest.h>

#include "posix_io/io_engine/io_engine.h"

namespace runai::llm::streamer::impl
{

namespace
{

// enforce_minimum = false so the block sizes below do not get floored, which would obscure what is
// being asserted.
Config config_with(size_t chunk)
{
    return Config(16, 8, 5 * 1024 * 1024, 2 * 1024 * 1024, false, chunk);
}

// The depth and the process count reach the settings directly: the streamer resolves them, so they
// are not read from the config here.
AsyncIoSettings settings_for(size_t chunk, unsigned depth, unsigned process_group_size = 1)
{
    return AsyncIoSettings(config_with(chunk), depth, process_group_size);
}

AsyncIoSettings settings_for(size_t chunk, unsigned depth, unsigned process_group_size, size_t max_read_bytesize)
{
    return AsyncIoSettings(config_with(chunk), depth, process_group_size, max_read_bytesize);
}

} // namespace

TEST(AsyncIoSettings, Single_Process_Uses_The_Whole_Depth)
{
    const auto settings = settings_for(8 << 20, 512);

    EXPECT_EQ(settings.process_group_size(), 1);
    EXPECT_EQ(settings.depth(), 512);
    EXPECT_EQ(settings.chunk_bytesize(), 8u << 20);
}

// The reason depth is configured node-wide: one number that means the same thing at TP=1 and TP=8.
TEST(AsyncIoSettings, Depth_Is_Divided_By_The_Process_Group)
{
    const auto settings = settings_for(8 << 20, 512, 8);

    EXPECT_EQ(settings.process_group_size(), 8);
    EXPECT_EQ(settings.depth(), 64) << "the device sees 8 x this, which is the configured 512";
}

// A count of zero is a divisor nobody meant; it is floored rather than dividing by zero.
TEST(AsyncIoSettings, Zero_Processes_Count_As_One)
{
    const auto settings = settings_for(8 << 20, 512, 0);

    EXPECT_EQ(settings.process_group_size(), 1);
    EXPECT_EQ(settings.depth(), 512);
}

// The division can round to zero, and a depth of 1 or 2 is a serial reader paying for the whole
// asynchronous apparatus. Floored rather than left to degenerate.
TEST(AsyncIoSettings, Depth_Is_Floored)
{
    const auto settings = settings_for(8 << 20, 8, 64);   // 8 / 64 rounds to 0

    EXPECT_EQ(settings.depth(), AsyncIoSettings::MinDepth);
}

// A mis-set variable must not produce a ring far larger than anything that can help. The cap is a
// safety bound, not a target: bytes in flight is what saturates a device, and this is already well
// past it.
TEST(AsyncIoSettings, Depth_Is_Capped)
{
    const auto settings = settings_for(8 << 20, 65536, 1);

    EXPECT_EQ(settings.depth(), AsyncIoSettings::MaxDepth);
}

// Between the bounds the configured value survives division untouched - the case that actually runs.
TEST(AsyncIoSettings, Depth_Is_Divided_Within_The_Bounds)
{
    const auto settings = settings_for(8 << 20, 512, 8);

    EXPECT_EQ(settings.depth(), 64u) << "512 node-wide over 8 processes";
}

// Above the kernel's per-read ceiling the bytes in flight become depth x the cap, not depth x the
// configured chunk - a window smaller than the one reserved against the memory limit, with no error.
TEST(AsyncIoSettings, Chunk_Is_Clamped_To_The_Kernel_Ceiling)
{
    const size_t cap = 1 << 20;
    const auto settings = settings_for(64 << 20, 512, 1, cap);

    EXPECT_EQ(settings.chunk_bytesize(), cap);
}

TEST(AsyncIoSettings, Chunk_Below_The_Ceiling_Is_Untouched)
{
    const auto settings = settings_for(4 << 20, 512, 1, 1 << 30);

    EXPECT_EQ(settings.chunk_bytesize(), 4u << 20);
}

}; // namespace runai::llm::streamer::impl
