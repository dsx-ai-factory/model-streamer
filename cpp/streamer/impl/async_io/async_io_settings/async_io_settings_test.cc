#include "streamer/impl/async_io/async_io_settings/async_io_settings.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "posix_io/io_engine/io_engine.h"
#include "utils/temp/env/env.h"

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

// The depth reaches the settings directly: the streamer resolves it per mount, so it is not read from
// the config here.
AsyncIoSettings settings_for(size_t chunk, unsigned depth)
{
    return AsyncIoSettings(config_with(chunk), depth);
}

AsyncIoSettings settings_for(size_t chunk, unsigned depth, size_t max_read_bytesize)
{
    return AsyncIoSettings(config_with(chunk), depth, max_read_bytesize);
}

} // namespace

TEST(AsyncIoSettings, Single_Process_Uses_The_Whole_Depth)
{
    utils::temp::UnsetEnv unset(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"));

    const auto settings = settings_for(8 << 20, 512);

    EXPECT_EQ(settings.process_group_size(), 1);
    EXPECT_EQ(settings.depth(), 512);
    EXPECT_EQ(settings.chunk_bytesize(), 8u << 20);
}

// The reason depth is configured node-wide: one number that means the same thing at TP=1 and TP=8.
TEST(AsyncIoSettings, Depth_Is_Divided_By_The_Process_Group)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 8UL);

    const auto settings = settings_for(8 << 20, 512);

    EXPECT_EQ(settings.process_group_size(), 8);
    EXPECT_EQ(settings.depth(), 64) << "the device sees 8 x this, which is the configured 512";
}

// A negative process group size is refused at startup, not treated as a huge one.
//
// std::stoul used to wrap "-1" to ULONG_MAX, and getenv_positive capped that to UINT_MAX. As a
// divisor that is silent and expensive: the depth falls to its minimum, so the engine keeps every
// cost of asynchronous reading and none of the benefit. The reader would look configured and be
// serial.
TEST(AsyncIoSettings, Negative_Process_Group_Is_Refused)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), std::string("-1"));

    EXPECT_THROW(settings_for(8 << 20, 512), std::exception);
}

// A depth of zero admits nothing, so a small node-wide value over many processes must floor at one
// rather than silently disabling the engine.
// The division can round to zero, and a depth of 1 or 2 is a serial reader paying for the whole
// asynchronous apparatus. Floored rather than left to degenerate - InstantTensor floors the same
// formula at the same value.
TEST(AsyncIoSettings, Depth_Is_Floored)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 64UL);

    const auto settings = settings_for(8 << 20, 8);   // 8 / 64 rounds to 0

    EXPECT_EQ(settings.depth(), AsyncIoSettings::MinDepth);
}

// A mis-set variable must not produce a ring far larger than anything that can help. The cap is a
// safety bound, not a target: bytes in flight is what saturates a device, and this is already well
// past it.
TEST(AsyncIoSettings, Depth_Is_Capped)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 1UL);

    const auto settings = settings_for(8 << 20, 65536);

    EXPECT_EQ(settings.depth(), AsyncIoSettings::MaxDepth);
}

// Between the bounds the configured value survives division untouched - the case that actually runs.
TEST(AsyncIoSettings, Depth_Is_Divided_Within_The_Bounds)
{
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 8UL);

    const auto settings = settings_for(8 << 20, 512);

    EXPECT_EQ(settings.depth(), 64u) << "512 node-wide over 8 processes";
}

// Above the kernel's per-read ceiling the bytes in flight become depth x the cap, not depth x the
// configured chunk - a window smaller than the one reserved against the memory limit, with no error.
TEST(AsyncIoSettings, Chunk_Is_Clamped_To_The_Kernel_Ceiling)
{
    utils::temp::UnsetEnv unset(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"));

    const size_t cap = 1 << 20;
    const auto settings = settings_for(64 << 20, 512, cap);

    EXPECT_EQ(settings.chunk_bytesize(), cap);
}

TEST(AsyncIoSettings, Chunk_Below_The_Ceiling_Is_Untouched)
{
    utils::temp::UnsetEnv unset(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"));

    const auto settings = settings_for(4 << 20, 512, 1 << 30);

    EXPECT_EQ(settings.chunk_bytesize(), 4u << 20);
}

// The ordering trap this class exists for. Python writes PROCESS_GROUP_SIZE inside stream_files(),
// long after runai_file_streamer_start() built the Config - so a settings object built with the Config would read
// the unset default of 1 and skip the division entirely.
TEST(AsyncIoSettings, Reads_The_Group_Size_When_Constructed_Not_When_Config_Was)
{
    std::unique_ptr<Config> config;
    {
        utils::temp::UnsetEnv unset(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"));
        config = std::make_unique<Config>(config_with(8 << 20));
    }

    // ... later, on the first workload, once the Python layer has published it.
    utils::temp::Env group(std::string("RUNAI_STREAMER_PROCESS_GROUP_SIZE"), 4UL);
    const AsyncIoSettings settings(*config, 512);

    EXPECT_EQ(settings.process_group_size(), 4);
    EXPECT_EQ(settings.depth(), 128) << "built too early, this would be 512 and the device would see 4x";
}

}; // namespace runai::llm::streamer::impl
