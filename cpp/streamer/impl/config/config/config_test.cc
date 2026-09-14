#include "streamer/impl/config/config/config.h"

#include <gtest/gtest.h>

#include "common/exception/exception.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "common/s3_wrapper/s3_wrapper.h"

#include "utils/random/random.h"
#include "utils/temp/env/env.h"

namespace runai::llm::streamer::impl
{

// Every variable Config reads is cleared before each test, so a test states only what it SETS and the
// defaults are the defaults. Guarding per test was how Default and Chunk_Size came to assert the
// concurrency defaults with nothing cleared: `--test_env` or running the binary directly then failed
// them against correct code. (A plain `bazel test` scrubs the environment, so it did not.)
class Creation : public ::testing::Test
{
 protected:
    void SetUp() override
    {
        for (const auto * variable : { "RUNAI_STREAMER_CONCURRENCY",
                                       "RUNAI_STREAMER_OBJ_CONCURRENCY",
                                       "RUNAI_STREAMER_FS_QUEUE_DEPTH",
                                       "RUNAI_STREAMER_CHUNK_BYTESIZE",
                                       "RUNAI_STREAMER_FS_CHUNK_BYTESIZE",
                                       "RUNAI_STREAMER_FS_STRATEGY",
                                       "RUNAI_STREAMER_S3_TIMEOUT",
                                       "RUNAI_STREAMER_DIRECT_BLOCK" })
        {
            _cleared.push_back(std::make_unique<utils::temp::UnsetEnv>(std::string(variable)));
        }
    }

    void TearDown() override
    {
        _cleared.clear();
    }

 private:
    std::vector<std::unique_ptr<utils::temp::UnsetEnv>> _cleared;
};

TEST_F(Creation, Default)
{
    Config config;
    EXPECT_EQ(config.concurrency, 16UL);
    EXPECT_EQ(config.s3_concurrency, 8UL);
    EXPECT_EQ(config.s3_block_bytesize, 8 * 1024 * 1024);
    EXPECT_EQ(config.fs_sync_read_block_bytesize, 2 * 1024 * 1024);
    EXPECT_EQ(config.object_storage_retry_timeout, std::chrono::seconds(0));
}

TEST_F(Creation, ObjectStorageRetryTimeout)
{
    const auto expected = utils::random::number<unsigned long>(1, 3600);
    utils::temp::Env timeout("RUNAI_STREAMER_S3_TIMEOUT", expected);

    Config config;
    EXPECT_EQ(config.object_storage_retry_timeout, std::chrono::seconds(expected));
}

// The legacy variable alone still configures both backends.
TEST_F(Creation, Concurrency)
{
    const auto expected = utils::random::number<int>(1, 1000);
    utils::temp::Env size_("RUNAI_STREAMER_CONCURRENCY", expected);

    Config config;
    EXPECT_EQ(config.concurrency, expected);
    EXPECT_EQ(config.s3_concurrency, expected);
    EXPECT_EQ(config.s3_block_bytesize, 8 * 1024 * 1024);
    EXPECT_EQ(config.fs_sync_read_block_bytesize, 2 * 1024 * 1024);
}

TEST_F(Creation, Chunk_Size)
{
    const size_t min_ = common::s3::S3ClientWrapper::min_chunk_bytesize;
    for (size_t expected : { 1UL, utils::random::number<size_t>(1UL, min_ - 1UL), utils::random::number<size_t>(min_, 10UL * min_)})
    {
        utils::temp::Env size_("RUNAI_STREAMER_CHUNK_BYTESIZE", expected);
        Config config;

        EXPECT_EQ(config.concurrency, 16UL);
        EXPECT_EQ(config.s3_concurrency, 8UL);
        EXPECT_EQ(config.s3_block_bytesize, std::max(expected, min_));
        EXPECT_EQ(config.fs_sync_read_block_bytesize, std::max(expected, Config::min_fs_sync_read_block_bytesize));
    }
}

TEST_F(Creation, Zero_Chunk_Size)
{
    utils::temp::Env size_("RUNAI_STREAMER_CHUNK_BYTESIZE", 0);
    EXPECT_THROW(Config(), std::exception);
}

TEST_F(Creation, Zero_Concurrency)
{
    utils::temp::Env size_("RUNAI_STREAMER_CONCURRENCY", 0);
    EXPECT_THROW(Config(), std::exception);
}

// The queue depth is PARSED, not read as a number, so a per-type value survives into Config.
TEST_F(Creation, Queue_Depth_Is_Parsed)
{
    {
        utils::temp::Env depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string("512,nfs=64"));

        const Config config;

        EXPECT_EQ(config.fs_async_queue_depth.default_value(), 512u);
        EXPECT_EQ(config.fs_async_queue_depth.for_type("nfs4"), 64u);
        EXPECT_EQ(config.fs_async_queue_depth.for_type("ext4"), 512u);
    }

    const Config config;

    EXPECT_EQ(config.fs_async_queue_depth.default_value(), Config::default_fs_async_queue_depth);
    EXPECT_TRUE(config.fs_async_queue_depth.entries().empty());
}

// A malformed value fails where every other malformed variable fails - building the Config, which
// runai_file_streamer_start turns into InvalidParameterError. Silently falling back would leave the typo undetected.
TEST_F(Creation, Malformed_Queue_Depth_Is_Rejected)
{
    for (const auto * bad : { "nfs=64", "abc", "0", "512,nfs=0", "512,nfs=64,nfs=32", "-1" })
    {
        utils::temp::Env depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string(bad));

        EXPECT_THROW(Config(), common::Exception) << "accepted: '" << bad << "'";
    }
}

// Both file system readers take the queue depth, so a host that resolves the synchronous reader is
// configured by the same variable. The per-type entries are for the mounts, so the pool takes the
// leading default.
TEST_F(Creation, Queue_Depth_Serves_Both_File_System_Readers)
{
    utils::temp::Env depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string("256,nfs=64"));

    const Config config;

    EXPECT_EQ(config.concurrency, 256u);
    EXPECT_EQ(config.fs_async_queue_depth.for_type("nfs4"), 64u);
    EXPECT_EQ(config.fs_async_queue_depth.for_type("ext4"), 256u);
}

// The legacy variable still configures the file system when the specific one is unset - otherwise an
// existing setting would stop working on upgrade, silently.
TEST_F(Creation, Concurrency_Serves_The_File_System_When_Queue_Depth_Is_Unset)
{
    utils::temp::Env legacy(std::string("RUNAI_STREAMER_CONCURRENCY"), 32UL);

    const Config config;

    EXPECT_EQ(config.concurrency, 32u);
    EXPECT_EQ(config.fs_async_queue_depth.default_value(), 32u);
    EXPECT_TRUE(config.fs_async_queue_depth.entries().empty());
}

TEST_F(Creation, Queue_Depth_Wins_Over_Concurrency)
{
    utils::temp::Env legacy(std::string("RUNAI_STREAMER_CONCURRENCY"), 32UL);
    utils::temp::Env depth(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string("256"));

    const Config config;

    EXPECT_EQ(config.concurrency, 256u);
    EXPECT_EQ(config.fs_async_queue_depth.default_value(), 256u);
}

// Unset is not "set to the default": with nothing set the two readers differ, because a read costs a
// thread in one and a queue slot in the other.
TEST_F(Creation, The_Two_Readers_Default_Apart)
{
    const Config config;

    EXPECT_EQ(config.concurrency, Config::default_concurrency);
    EXPECT_EQ(config.fs_async_queue_depth.default_value(), Config::default_fs_async_queue_depth);
}

// Object storage takes the specific variable, and nothing about the file system moves with it.
TEST_F(Creation, Obj_Concurrency_Serves_Object_Storage_Only)
{
    utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 24UL);

    const Config config;

    EXPECT_EQ(config.s3_concurrency, 24u);
    EXPECT_EQ(config.concurrency, Config::default_concurrency);
    EXPECT_EQ(config.fs_async_queue_depth.default_value(), Config::default_fs_async_queue_depth);
}

TEST_F(Creation, Obj_Concurrency_Wins_Over_Concurrency)
{
    utils::temp::Env legacy(std::string("RUNAI_STREAMER_CONCURRENCY"), 4UL);
    utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 24UL);

    const Config config;

    EXPECT_EQ(config.s3_concurrency, 24u);
    EXPECT_EQ(config.concurrency, 4u) << "the legacy variable still serves the file system";
}

TEST_F(Creation, Zero_Obj_Concurrency)
{
    utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 0UL);
    EXPECT_THROW(Config(), std::exception);
}

// A negative value is already rejected when the variable is parsed. A large positive one is not, and
// each unit costs a thread - and a client with its own connections for object storage. Capped rather
// than refused, so the load still runs.
TEST_F(Creation, Concurrency_Is_Capped)
{
    utils::temp::Env fs(std::string("RUNAI_STREAMER_FS_QUEUE_DEPTH"), std::string("100000"));
    utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 100000UL);

    const Config config;

    EXPECT_EQ(config.concurrency, Config::max_concurrency);
    EXPECT_EQ(config.s3_concurrency, Config::max_concurrency);

    EXPECT_EQ(config.fs_async_queue_depth.default_value(), 100000u)
        << "the cap is on worker counts, not on the queue depth, which costs a slot rather than a thread";
}

// The cap has to be applied to the parsed 64-bit value, not to the narrowed one. A cast alone wraps:
// 4294967301 becomes 5, and 4294967296 becomes 0, which the zero assertion then reads as a deliberate
// zero and rejects. Both are silently wrong rather than merely too large.
TEST_F(Creation, Concurrency_Above_The_Word_Size_Does_Not_Wrap)
{
    {
        utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 4294967301UL);
        EXPECT_EQ(Config().s3_concurrency, Config::max_concurrency) << "must not narrow to 5";
    }

    {
        utils::temp::Env obj(std::string("RUNAI_STREAMER_OBJ_CONCURRENCY"), 4294967296UL);
        EXPECT_EQ(Config().s3_concurrency, Config::max_concurrency) << "must not narrow to 0 and throw";
    }

    {
        utils::temp::Env legacy(std::string("RUNAI_STREAMER_CONCURRENCY"), 4294967301UL);

        const Config config;
        EXPECT_EQ(config.concurrency, Config::max_concurrency);
        EXPECT_EQ(config.s3_concurrency, Config::max_concurrency);
    }
}

// The cap must not move a value that is already below it.
TEST_F(Creation, The_Cap_Leaves_A_Normal_Value_Alone)
{
    utils::temp::Env legacy(std::string("RUNAI_STREAMER_CONCURRENCY"), 32UL);

    const Config config;

    EXPECT_EQ(config.concurrency, 32u);
    EXPECT_EQ(config.s3_concurrency, 32u);
}

// A plain number is a complete value: it applies to every mount.
TEST_F(Creation, A_Plain_Number_Applies_Everywhere)
{
    const Config config(16, 8, 5 * 1024 * 1024, 2 * 1024 * 1024, false, 8 * 1024 * 1024, FsQueueDepth(64));

    EXPECT_EQ(config.fs_async_queue_depth.default_value(), 64u);
    EXPECT_EQ(config.fs_async_queue_depth.for_type("nfs"), 64u);
}

}; // namespace runai::llm::streamer::impl
