/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// reads_directly() is consulted for THIS strategy ONLY - libaio has no asynchronous buffered mode, so
// a file it cannot read directly has to be routed away before dispatch. Without a way to ask for it,
// that whole function is unreachable from a test.
constexpr const char * libaio_candidates = "libaio_direct,sync_buffered";

// The block every test here reasons about. Named rather than repeated so a reader can see that the
// offsets below are chosen against it.
constexpr size_t BlockForTests = 4096;

// Every strategy is available on a normal host, so a test cannot otherwise reach a failed resolution.
FsAsyncRouter::Environment with_mounts(FsAsyncRouter::MountProbe mount)
{
    FsAsyncRouter::Environment environment;
    environment.mount = std::move(mount);
    environment.availability = [](posix_io::Strategy) { return common::ResponseCode::Success; };
    return environment;
}

// A block-aligned destination, so a test says what it means about congruence instead of depending on
// where the linker happened to put a static array.
//
// Static storage: a FileRanges holds a raw pointer, and the request outlives the call that built it.
char * aligned_destination()
{
    static char storage[3 * BlockForTests];

    const auto base = reinterpret_cast<uintptr_t>(storage);
    return storage + ((-base) % BlockForTests);
}

// One range each, so no file is skipped as empty.
//
// The destination is always block aligned, so `offset` alone decides congruence -
// `(address % block) == (offset % block)`. Offset 0 is congruent; anything that is not a block
// multiple is not, which is what the routing tests turn on.
std::vector<FileRanges> request_for(const std::vector<std::string> & paths, size_t offset = 0)
{
    char * const destination = aligned_destination();

    std::vector<FileRanges> request(paths.size());
    for (size_t i = 0; i < paths.size(); ++i)
    {
        request[i].path = paths[i];
        request[i].ranges.push_back(ReadRange{ offset, BlockForTests, destination });
    }
    return request;
}

// A router whose mount, direct support AND block are all answered by the test.
//
// The BLOCK matters more than it looks. Without this hook the block falls back to DirectBlockSize,
// which is 65536 - so a test that aligned its destination to 4096 and said nothing would be having
// its congruence judged against a different number than the one it named, and would pass or fail for
// a reason it never stated.
FsAsyncRouter::Environment with_direct(FsAsyncRouter::MountProbe mount,
                                       posix_io::DirectSupport support = posix_io::DirectSupport::Yes,
                                       size_t block = BlockForTests)
{
    auto environment = with_mounts(std::move(mount));
    environment.direct = [support](dev_t, const std::string &) { return support; };
    environment.direct_block = [block](dev_t, const std::string &) { return block; };
    return environment;
}

// Every mount here is an ordinary one, which is what these tests are about.
posix_io::MountCapability ext4(const std::string &)
{
    return posix_io::MountCapability{ makedev(8, 1), false, "ext4" };
}

} // namespace

// --- reads_directly: consulted for libaio_direct only -------------------------------------------

// THE SEAM TEST. reads_directly() is reached only under libaio_direct, and no test could ask for that
// strategy until now - so every decision the function makes was unreachable, and the first one anyone
// noticed being wrong was a bug in the field.
//
// A congruent destination keeps its file on the async engine.
TEST(FsAsyncRouter, Libaio_Keeps_A_Congruent_File)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);
    ASSERT_EQ(router.strategy(), posix_io::Strategy::LibaioDirect) << "the rest of this is vacuous"
                                                                      " unless libaio resolved";

    // Offset 0 into a block-aligned destination: congruent by construction.
    const auto groups = router.groups(request_for({ "/one/a.st" }), FsQueueDepth(512), common::Device::host());

    EXPECT_GE(groups.by_file[0], 0) << "a congruent file was sent to the synchronous reader";
}

// A destination that cannot be congruent with the offset goes to the synchronous reader, because
// libaio has no asynchronous buffered mode to fall back to.
TEST(FsAsyncRouter, Libaio_Sends_A_Non_Congruent_File_To_The_Synchronous_Reader)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    // A block-aligned address against an offset that is not a block multiple: no aligned read lands
    // where the caller's bytes have to go.
    const auto groups = router.groups(request_for({ "/one/a.st" }, 1000 /* offset */), FsQueueDepth(512), common::Device::host());

    EXPECT_EQ(groups.by_file[0], -1);
}

// The same file, the same offset, under a strategy that HAS an asynchronous buffered mode: it keeps
// its engine, because the worker can open it buffered per file rather than losing it to the pool.
//
// This is what says the routing above belongs to libaio and is not a property of the offset.
TEST(FsAsyncRouter, Io_Uring_Keeps_A_Non_Congruent_File)
{
    FsAsyncRouter router(direct_candidates, with_direct(ext4));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/one/a.st" }, 1000 /* offset */), FsQueueDepth(512), common::Device::host());

    EXPECT_GE(groups.by_file[0], 0);
}

// A DEVICE DESTINATION IS NOT THE PLACE THE READ LANDS, so its address says nothing about congruence.
//
// A device read goes into a pinned staging buffer that the library allocates and places; the pointer
// in the request is where the bytes end up AFTERWARDS, on the GPU. Testing that pointer against a
// file offset asks a question about the wrong memory, and the answer is no for almost every offset -
// so the file is handed to the synchronous reader and the configured strategy is silently unused.
//
// The mount check still has to run. A mount that cannot serve O_DIRECT cannot serve one for a device
// destination either.
TEST(FsAsyncRouter, Libaio_Keeps_A_Device_Destination_At_An_Unaligned_Offset)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/one/a.st" }, 1000 /* offset */),
                                      FsQueueDepth(512), common::Device::cuda(0));

    EXPECT_GE(groups.by_file[0], 0)
        << "a device read was routed to the synchronous reader because the GPU pointer it names is"
           " not congruent with the file offset - but the read never lands there";
}

// THE COUNTERPART, so the rule above cannot be read as "device destinations skip every check".
TEST(FsAsyncRouter, Libaio_Still_Refuses_A_Device_Destination_On_A_Mount_Without_Direct_Support)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4, posix_io::DirectSupport::No));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    // Offset 0, so congruence would PASS for a host destination too. The mount check is then the only
    // thing that can reject this file, which is what the test is about.
    const auto groups = router.groups(request_for({ "/one/a.st" }), FsQueueDepth(512),
                                      common::Device::cuda(0));

    EXPECT_EQ(groups.by_file[0], -1) << "the mount serves no direct read at all, for any destination";
}

// A mount that cannot serve O_DIRECT at all loses its files under libaio, whatever the addresses are.
TEST(FsAsyncRouter, Libaio_Sends_A_Mount_Without_Direct_Support_To_The_Synchronous_Reader)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4, posix_io::DirectSupport::No));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/one/a.st" }), FsQueueDepth(512), common::Device::host());

    EXPECT_EQ(groups.by_file[0], -1) << "congruent, but the mount serves no direct read at all";
}

// Unknown is not No. It means the probe could not open the file, which the read itself will report
// attributably - and refusing the mount would take every file beside it down too.
TEST(FsAsyncRouter, Libaio_Keeps_A_Mount_Whose_Direct_Support_Is_Unknown)
{
    FsAsyncRouter router(libaio_candidates, with_direct(ext4, posix_io::DirectSupport::Unknown));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/one/a.st" }), FsQueueDepth(512), common::Device::host());

    EXPECT_GE(groups.by_file[0], 0);
}

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
                                      FsQueueDepth::parse("512,nfs=64"), common::Device::host());

    ASSERT_EQ(groups.devices.size(), 2u) << "one group per mount";
    EXPECT_EQ(groups.by_file, (std::vector<int>{ 0, 1 }));
    EXPECT_EQ(groups.depths[0], 64u) << "the nfs4 mount takes the nfs entry";
    EXPECT_EQ(groups.depths[1], 512u) << "the ext4 mount takes the default";
}

// Registered buffers are decided per mount, like the depth, and from the same file system type.
//
// The policy itself lives in Config with its own tests; what this checks is that the router ASKS and
// carries the answer per group - without it an NFS mount would read through registered buffers, which
// costs there rather than paying.
TEST(FsAsyncRouter, Registered_Buffers_Are_Decided_Per_Mount)
{
    FsAsyncRouter router(async_candidates, with_mounts([](const std::string & directory)
    {
        const bool is_nfs = directory == "/nfs";
        return posix_io::MountCapability{ is_nfs ? makedev(8, 1) : makedev(8, 2), false,
                                          is_nfs ? "nfs4" : "ext4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/nfs/a.st", "/local/b.st" }), FsQueueDepth(512),
                                      common::Device::host(),
                                      [](const std::string & fs_type) { return fs_type != "nfs4"; });

    ASSERT_EQ(groups.registers.size(), 2u) << "one answer per group, like the depth";
    EXPECT_FALSE(groups.registers[0]) << "the nfs4 mount was denied";
    EXPECT_TRUE(groups.registers[1]) << "the ext4 mount was allowed";
}

// No policy means register. A caller that does not care gets what it had before the deny list existed.
TEST(FsAsyncRouter, Without_A_Policy_Every_Mount_Registers)
{
    FsAsyncRouter router(async_candidates, with_mounts([](const std::string &)
    {
        return posix_io::MountCapability{ makedev(8, 1), false, "nfs4" };
    }));
    ASSERT_EQ(router.resolve(), common::ResponseCode::Success);

    const auto groups = router.groups(request_for({ "/nfs/a.st" }), FsQueueDepth(512), common::Device::host());

    ASSERT_EQ(groups.registers.size(), 1u);
    EXPECT_TRUE(groups.registers[0]);
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
                                      FsQueueDepth(512), common::Device::host());

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

    const auto groups = router.groups(request_for({ "/dev/shm/a.st", "/local/b.st" }), FsQueueDepth(512), common::Device::host());

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

    const auto groups = router.groups(request_for({ "/gone/a.st", "/local/b.st" }), FsQueueDepth(512), common::Device::host());

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

    const auto groups = router.groups(request, FsQueueDepth(512), common::Device::host());

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

    const auto groups = router.groups(request_for({ "/local/a.st" }), FsQueueDepth(512), common::Device::host());

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
