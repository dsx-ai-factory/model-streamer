/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/fs_async_router/fs_async_router.h"

#include <gtest/gtest.h>

#include <sys/sysmacros.h>   // makedev

#include <string>
#include <utility>
#include <vector>

#include "common/exception/exception.h"

namespace runai::llm::streamer::impl
{

namespace
{

constexpr const char * async_candidates = "io_uring_buffered,sync_buffered";
constexpr const char * direct_candidates = "io_uring_direct,sync_buffered";

// Every strategy is available on a normal host, so a test cannot otherwise reach a failed resolution.
FsAsyncRouter::Environment with_mounts(FsAsyncRouter::MountProbe mount)
{
    FsAsyncRouter::Environment environment;
    environment.mount = std::move(mount);
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    return environment;
}

// One range each, so no file is skipped as empty.
std::vector<FileRanges> request_for(const std::vector<std::string> & paths)
{
    static char destination[4096];

    std::vector<FileRanges> request(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
    {
        request[i].path = paths[i];
        request[i].ranges.push_back(ReadRange{ 0, sizeof(destination), destination });
    }
    return request;
}

} // namespace

// The depth a mount reads at comes from its file system type, and this is the only place the type is
// known.
TEST(FsAsyncRouter, Queue_Depth_Is_Resolved_Per_Mount)
{
    FsAsyncRouter router(async_candidates, with_mounts([](const std::string & directory)
    {
        const bool is_nfs = directory == "/nfs";
        return posix_io::MountCapability{ is_nfs ? makedev(8, 1) : makedev(8, 2), false,
                                          is_nfs ? "nfs4" : "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/nfs/a.st", "/local/b.st" }),
                                      FsQueueDepth::parse("512,nfs=64"));

    ASSERT_EQ(groups.devices.size(), 2u) << "one group per mount";
    EXPECT_EQ(groups.by_file, (std::vector<int>{ 0, 1 }));
    EXPECT_EQ(groups.depths[0], 64u) << "the nfs4 mount takes the nfs entry";
    EXPECT_EQ(groups.depths[1], 512u) << "the ext4 mount takes the default";
}

// Directories on one mount share a group, so they share an engine.
TEST(FsAsyncRouter, One_Group_Per_Mount_Not_Per_Directory)
{
    FsAsyncRouter router(async_candidates, with_mounts([](const std::string &)
    {
        return posix_io::MountCapability{ makedev(8, 1), false, "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/one/a.st", "/two/b.st", "/one/c.st" }),
                                      FsQueueDepth(512));

    EXPECT_EQ(groups.devices.size(), 1u);
    EXPECT_EQ(groups.by_file, (std::vector<int>{ 0, 0, 0 }));
}

// tmpfs and ramfs are pure memcpy with no device to overlap, so the synchronous pool serves them
// however the strategy resolved.
TEST(FsAsyncRouter, A_Memory_Backed_Mount_Goes_To_The_Synchronous_Reader)
{
    FsAsyncRouter router(async_candidates, with_mounts([](const std::string & directory)
    {
        const bool is_tmpfs = directory == "/dev/shm";
        return posix_io::MountCapability{ is_tmpfs ? makedev(0, 20) : makedev(8, 2), is_tmpfs,
                                          is_tmpfs ? "tmpfs" : "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/dev/shm/a.st", "/local/b.st" }), FsQueueDepth(512));

    EXPECT_EQ(groups.by_file[0], -1);
    EXPECT_GE(groups.by_file[1], 0);
    EXPECT_EQ(groups.devices.size(), 1u) << "the memory-backed mount opens no group";
}

// A directory that cannot be probed sends its file to the synchronous reader rather than failing the
// submission.
TEST(FsAsyncRouter, An_Unprobeable_Directory_Is_Not_Fatal)
{
    FsAsyncRouter router(async_candidates,
                         with_mounts([](const std::string & directory) -> posix_io::MountCapability
    {
        if (directory == "/gone")
        {
            throw common::Exception(common::ResponseCode::FileAccessError);
        }
        return posix_io::MountCapability{ makedev(8, 2), false, "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/gone/a.st", "/local/b.st" }), FsQueueDepth(512));

    EXPECT_EQ(groups.by_file[0], -1);
    EXPECT_GE(groups.by_file[1], 0);
}

// A file with no ranges reaches no storage, so probing its mount would fail on a path that was never
// going to be read.
TEST(FsAsyncRouter, A_File_With_No_Ranges_Is_Never_Probed)
{
    bool probed = false;
    FsAsyncRouter router(async_candidates, with_mounts([&probed](const std::string &)
    {
        probed = true;
        return posix_io::MountCapability{ makedev(8, 2), false, "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    std::vector<FileRanges> request(1);
    request[0].path = "/local/a.st";

    const auto groups = router.groups(request, FsQueueDepth(512));

    EXPECT_FALSE(probed);
    EXPECT_EQ(groups.by_file, (std::vector<int>{ -1 }));
}

// The synchronous reader serves everything, so no mount is probed at all.
TEST(FsAsyncRouter, A_Synchronous_Strategy_Groups_Nothing)
{
    bool probed = false;
    FsAsyncRouter router("sync_buffered", with_mounts([&probed](const std::string &)
    {
        probed = true;
        return posix_io::MountCapability{ makedev(8, 2), false, "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/local/a.st" }), FsQueueDepth(512));

    EXPECT_FALSE(probed);
    EXPECT_TRUE(groups.devices.empty());
    EXPECT_EQ(groups.by_file, (std::vector<int>{ -1 }));
}

// The caller must satisfy every mount it reads, so the largest block wins.
TEST(FsAsyncRouter, Direct_Block_Is_The_Largest_Any_Mount_Requires)
{
    auto environment = with_mounts([](const std::string & directory)
    {
        return posix_io::MountCapability{ directory == "/big" ? makedev(8, 1) : makedev(8, 2),
                                          false, "ext4" };
    });
    environment.direct_block = [](dev_t device, const std::string &) -> size_t
    {
        return device == makedev(8, 1) ? 4096 : 512;
    };

    FsAsyncRouter router(direct_candidates, std::move(environment));

    size_t block = 0;
    EXPECT_EQ(router.direct_block_for({ "/small/a.st", "/big/b.st" }, block), common::ResponseCode::Success);
    EXPECT_EQ(block, 4096u);
}

// Nothing measurable is not Success: the caller can still lay out at the page size, but must ask again
// rather than treat this as the mount's answer.
TEST(FsAsyncRouter, Direct_Block_Reports_Unknown_When_Nothing_Answers)
{
    auto environment = with_mounts([](const std::string &)
    {
        return posix_io::MountCapability{ makedev(8, 2), false, "ext4" };
    });
    environment.direct_block = [](dev_t, const std::string &) -> size_t { return 0; };

    FsAsyncRouter router(direct_candidates, std::move(environment));

    size_t block = 0;
    EXPECT_EQ(router.direct_block_for({ "/local/a.st" }, block), common::ResponseCode::UnknownError);
    EXPECT_GT(block, 0u) << "a layout value is still reported";
}

// A memory-backed mount has no device to bypass, so it imposes no alignment.
TEST(FsAsyncRouter, Direct_Block_Ignores_A_Memory_Backed_Mount)
{
    auto environment = with_mounts([](const std::string & directory)
    {
        const bool is_tmpfs = directory == "/dev/shm";
        return posix_io::MountCapability{ is_tmpfs ? makedev(0, 20) : makedev(8, 2), is_tmpfs,
                                          is_tmpfs ? "tmpfs" : "ext4" };
    });
    environment.direct_block = [](dev_t, const std::string &) -> size_t { return 512; };

    FsAsyncRouter router(direct_candidates, std::move(environment));

    size_t block = 0;
    EXPECT_EQ(router.direct_block_for({ "/dev/shm/a.st" }, block), common::ResponseCode::UnknownError);
}

}; // namespace runai::llm::streamer::impl
