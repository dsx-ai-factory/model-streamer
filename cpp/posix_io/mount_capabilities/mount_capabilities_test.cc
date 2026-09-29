/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/mount_capabilities/mount_capabilities.h"

#include "posix_io/alignment/alignment.h"   // direct_block_size - what the probe reads at

#include <gtest/gtest.h>
#include <sys/sysmacros.h>

#include <fstream>
#include <sstream>
#include <vector>

#include <fcntl.h>
#include <linux/magic.h>
#include <cerrno>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <sys/syscall.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unistd.h>

#include <string>

#include "common/exception/exception.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// Answered from the FILESYSTEM, never from the code under test. A skip must depend on the
// environment; deciding it from MountCapability would let a broken probe skip instead of fail.
bool is_tmpfs(const std::string & path)
{
    struct statfs fs;
    return ::statfs(path.c_str(), &fs) == 0 && fs.f_type == TMPFS_MAGIC;
}

// The same question MountCapabilities answers - tmpfs OR ramfs (mount_capabilities.cc) - asked of
// the filesystem for the same reason as above. is_tmpfs above is deliberately narrower: its caller
// wants tmpfs specifically, because it is checking /dev/shm.
bool raw_memory_backed(const std::string & path)
{
    struct statfs fs;
    return ::statfs(path.c_str(), &fs) == 0 &&
           (fs.f_type == TMPFS_MAGIC || fs.f_type == RAMFS_MAGIC);
}

// STATX_DIOALIGN asked of the kernel directly, so the test knows which answer the implementation
// could have had. Declared here rather than shared with mount_capabilities.cc for the usual reason:
// an oracle that imports the code it checks agrees with a wrong implementation.
#ifndef STATX_DIOALIGN
#define STATX_DIOALIGN 0x00002000U
#endif

struct RawDioAlign
{
    bool answered = false;
    size_t mem = 0;
    size_t offset = 0;
};

RawDioAlign raw_statx_dio_align(const std::string & path)
{
    struct StatxCompat
    {
        uint32_t mask, blksize;
        uint64_t attributes;
        uint32_t nlink, uid, gid;
        uint16_t mode, spare0[1];
        uint64_t ino, size, blocks, attributes_mask;
        struct { int64_t tv_sec; uint32_t tv_nsec; int32_t pad; } atime, btime, ctime, mtime;
        uint32_t rdev_major, rdev_minor, dev_major, dev_minor;
        uint64_t mnt_id;
        uint32_t dio_mem_align, dio_offset_align;
        uint64_t spare[12];
    } buf;

    std::memset(&buf, 0, sizeof(buf));

    if (::syscall(__NR_statx, AT_FDCWD, path.c_str(), 0, STATX_DIOALIGN, &buf) != 0 ||
        (buf.mask & STATX_DIOALIGN) == 0 ||
        buf.dio_mem_align == 0 || buf.dio_offset_align == 0)
    {
        return {};
    }

    return { true, buf.dio_mem_align, buf.dio_offset_align };
}

dev_t device_of(const std::string & path)
{
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 ? st.st_dev : 0;
}

} // namespace

TEST(MountCapabilities, Regular_File_Agrees_With_The_Filesystem)
{
    utils::temp::File file(utils::random::buffer(64));
    MountCapabilities mounts;

    const auto capability = mounts.of_path(file.path);

    EXPECT_NE(capability.dev, 0);

    // NOT hardcoded to false, which is what this used to assert. temp::File defaults to "."
    // (file.h), and under bazel that is the sandbox beneath the output base - so --output_user_root
    // pointing at a tmpfs, a normal thing to do for speed, made a CORRECT answer fail the test.
    //
    // Checked against the filesystem in both directions instead of skipping on a memory-backed
    // mount. A skip would give up the case entirely on such a host, and a skip reads exactly like a
    // pass; this still asserts something everywhere, and on an ordinary filesystem it asserts what
    // the old test did.
    EXPECT_EQ(capability.memory_backed, raw_memory_backed(file.path));
}

// The probe that keeps tmpfs out of the O_DIRECT cell. Nothing else detects it: an O_DIRECT open
// SUCCEEDS on tmpfs, and so does every read through it - see
// Tmpfs_Can_Never_Be_Probed_For_A_Block below. So statfs magic is the only working test.
TEST(MountCapabilities, Tmpfs_Is_Memory_Backed)
{
    if (!is_tmpfs("/dev/shm"))
    {
        GTEST_SKIP() << "/dev/shm is not tmpfs here";
    }

    const std::string path = "/dev/shm/runai_mount_test_" + utils::random::string();
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ::close(fd);

    MountCapabilities mounts;
    const auto capability = mounts.of_path(path);
    ::unlink(path.c_str());

    EXPECT_TRUE(capability.memory_backed);
}

// The point of the class: a model's shards share a mount, so only the first costs a probe.
TEST(MountCapabilities, Probes_Once_Per_Mount)
{
    utils::temp::File a(utils::random::buffer(64));
    utils::temp::File b(utils::random::buffer(64));

    MountCapabilities mounts;
    EXPECT_EQ(mounts.size(), 0);

    const auto first = mounts.of_path(a.path);
    EXPECT_EQ(mounts.size(), 1);

    const auto second = mounts.of_path(b.path);
    EXPECT_EQ(mounts.size(), 1) << "two files on one mount must not be two entries";

    EXPECT_EQ(first.dev, second.dev);
    EXPECT_EQ(first.memory_backed, second.memory_backed);
}

// of_fd is verification on an fd the caller was opening anyway - it must agree with of_path and must
// not add an entry for a mount already known.
TEST(MountCapabilities, Of_Fd_Agrees_With_Of_Path)
{
    utils::temp::File file(utils::random::buffer(64));
    MountCapabilities mounts;

    const auto by_path = mounts.of_path(file.path);
    EXPECT_EQ(mounts.size(), 1);

    const int fd = ::open(file.path.c_str(), O_RDONLY);
    ASSERT_GE(fd, 0);
    const auto by_fd = mounts.of_fd(fd);
    ::close(fd);

    EXPECT_EQ(mounts.size(), 1);
    EXPECT_EQ(by_fd.dev, by_path.dev);
    EXPECT_EQ(by_fd.memory_backed, by_path.memory_backed);
}

// Two mounts must be two entries, or one would answer for the other - which is what selects the
// engine, so a collapse here would put both mounts on one worker and lose the isolation.
TEST(MountCapabilities, Distinct_Mounts_Are_Distinct_Entries)
{
    utils::temp::File regular(utils::random::buffer(64));

    const std::string shm = "/dev/shm/runai_mount_test_" + utils::random::string();
    const int fd = ::open(shm.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ::close(fd);

    // Whether these are two mounts is a fact about the machine, so ask the machine.
    if (device_of(regular.path) == device_of(shm))
    {
        ::unlink(shm.c_str());
        GTEST_SKIP() << "/dev/shm and the temp directory are the same mount here";
    }

    MountCapabilities mounts;
    const auto a = mounts.of_path(regular.path);
    const auto b = mounts.of_path(shm);
    ::unlink(shm.c_str());

    EXPECT_EQ(mounts.size(), 2);
    EXPECT_NE(a.dev, b.dev);
}

TEST(MountCapabilities, Missing_Path_Throws)
{
    MountCapabilities mounts;
    EXPECT_THROW(mounts.of_path("/no/such/path/" + utils::random::string()), common::Exception);
    EXPECT_EQ(mounts.size(), 0) << "a failed probe must not leave an entry";
}

TEST(MountCapabilities, Bad_Fd_Throws)
{
    MountCapabilities mounts;
    EXPECT_THROW(mounts.of_fd(-1), common::Exception);
}

// The answer must come from the FILESYSTEM, not from the code under test, or a broken probe would
// agree with itself. So this opens the file directly and compares.
//
// The oracle OPENS AND READS, because that is what direct_support does now. An open alone answers a
// narrower question: a mount whose real alignment is above DirectBlockSize opens without complaint
// and then fails every read with EINVAL. Comparing against the open alone would call that mount
// supported and disagree with the code under test on exactly the case the read was added for.
TEST(MountCapabilities, Direct_Support_Matches_A_Real_Aligned_Read)
{
    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    bool really_supported = false;
    int fd = ::open(file.path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0)
    {
        // ANY rung the mount accepts proves direct reads work here, because that is the question
        // direct_support answers: not "does this one block work" but "can this mount serve direct
        // reads at all".
        //
        // Reading at one fixed block was wrong twice over. Pinned to DirectBlockSize it drifted the
        // moment the block became configurable; pinned to direct_block_size() it still asks about the
        // PROCESS-WIDE block while direct_support now measures PER MOUNT. Set
        // RUNAI_STREAMER_DIRECT_BLOCK=512 on a mount that needs 4096 and the two disagree: the
        // override makes direct_block answer Yes, while a 512-byte read returns EINVAL.
        //
        // The ladder is restated here rather than shared with the implementation on purpose. An
        // oracle that imported the values it is checking would agree with a wrong implementation.
        for (const size_t block : { size_t(512), size_t(4096), size_t(16384), size_t(65536) })
        {
            void * buffer = nullptr;
            if (::posix_memalign(&buffer, block, block) != 0)
            {
                continue;
            }

            const ssize_t got = ::pread(fd, buffer, block, 0);
            const bool worked = got >= 0 || errno != EINVAL;
            ::free(buffer);

            if (worked)
            {
                really_supported = true;
                break;
            }
        }

        ::close(fd);
    }

    MountCapabilities mounts;
    const auto support = mounts.direct_support(device_of(file.path), file.path);

    EXPECT_EQ(support, really_supported ? DirectSupport::Yes : DirectSupport::No);
}

// The measured block must be usable, and must be the SMALLEST the mount accepts.
//
// Smallest is the whole point of measuring. A block larger than the mount needs pads more between
// ranges and bounces more of every chunk's head and tail - measured on NFS under the chunks policy,
// a 64 KiB block against a 4 KiB mount cost 2.4x the load time.
TEST(MountCapabilities, Direct_Block_Is_The_Smallest_The_Mount_Accepts)
{
    const auto data = utils::random::buffer(64 * 1024);
    utils::temp::File file(data);

    MountCapabilities mounts;
    size_t block = 0;
    mounts.direct_block(device_of(file.path), file.path, block);

    if (block == 0)
    {
        // No direct I/O here. That is an answer, and direct_support must agree with it.
        EXPECT_EQ(mounts.direct_support(device_of(file.path), file.path), DirectSupport::No);
        return;
    }

    // An override replaces the measurement, and is deliberately NOT the smallest workable block - so
    // the minimality check below would be asserting the wrong property. Assert what an override
    // promises instead: that it is obeyed exactly.
    if (const size_t forced = direct_block_override(); forced != 0)
    {
        EXPECT_EQ(block, forced) << "RUNAI_STREAMER_DIRECT_BLOCK must replace the measurement";
        return;
    }

    EXPECT_EQ(block & (block - 1), 0u) << "must be a power of two, or posix_memalign refuses it";
    EXPECT_GE(block, 512u) << "no device reports a logical block below 512";
    EXPECT_LE(block, 65536u) << "larger than any rung on the ladder, so it cannot have been measured";

    // Asked of the kernel, not of the code under test: the reported block must actually work...
    int fd = ::open(file.path.c_str(), O_RDONLY | O_DIRECT);
    ASSERT_GE(fd, 0);

    void * buffer = nullptr;
    ASSERT_EQ(::posix_memalign(&buffer, block, block), 0);
    const ssize_t got = ::pread(fd, buffer, block, 0);
    EXPECT_TRUE(got >= 0 || errno != EINVAL) << "the reported block " << block << " is refused";
    ::free(buffer);

    // Which property holds depends on WHERE the number came from, so ask the kernel the same
    // question the implementation asks.
    const auto reported = raw_statx_dio_align(file.path);

    if (reported.answered)
    {
        // statx answered, so assert the EXACT value rather than minimality. This is the stronger
        // check: the implementation must take max(mem, offset), because one block has to satisfy
        // both the buffer and the offset/length rules.
        //
        // Minimality would be the WRONG property here. statx reports what the filesystem REQUIRES,
        // and a lenient filesystem may still accept a smaller read - by falling back to buffered I/O
        // rather than by being aligned - which would fail a minimality assertion on a correct answer.
        EXPECT_EQ(block, std::max<size_t>(reported.mem, reported.offset))
            << "statx reported mem=" << reported.mem << " offset=" << reported.offset;
    }
    else
    {
        // No statx answer, so the ladder produced this - and the ladder returns the FIRST rung that
        // works, so by construction nothing smaller may work.
        //
        // Only meaningful above the ladder's floor. At 512 this loop has no iterations at all, and an
        // earlier version of this test asserted minimality in exactly that vacuous way on every host
        // where the floor is the answer - claiming to check something it never ran.
        if (block == 512)
        {
            GTEST_SKIP() << "the ladder returned its floor, so there is no smaller rung to rule out";
        }

        for (size_t smaller = 512; smaller < block; smaller <<= 1)
        {
            void * small = nullptr;
            ASSERT_EQ(::posix_memalign(&small, smaller, smaller), 0);
            const ssize_t r = ::pread(fd, small, smaller, 0);
            const bool worked = r >= 0 || errno != EINVAL;
            ::free(small);

            EXPECT_FALSE(worked) << smaller << " also works, so " << block << " was not the smallest";
        }
    }

    ::close(fd);
}

// A probe cannot recognise tmpfs, on any kernel - so statfs magic has to.
//
// This test was written to reach the ladder. tmpfs is the one mount here where statx reports no
// DIOALIGN, so a probe would fall through to it. Measured on kernel 6.8, tmpfs then accepts EVERY
// direct read - unaligned buffer, odd length, odd offset. It enforces nothing, because there is no
// device to bypass. The ladder's first rung succeeds and the answer is 512 for a filesystem with no
// direct I/O at all.
//
// That behaviour is KERNEL-DEPENDENT and this test does not assert it. Older kernels refuse the
// O_DIRECT open itself, because the VFS rejected an open on a mapping with no direct_IO operation.
// Both outcomes are fine for us and neither is something we control.
//
// What IS asserted is the property both outcomes share, and the one the routing depends on: a read
// can never report that tmpfs does direct I/O at some block. Either the open fails, or a badly
// aligned read is accepted. The remaining case - tmpfs opens AND enforces alignment - is the one that
// would make the ladder's answer meaningful there, and if a kernel ever does that, this fails and the
// comments around it need rewriting.
//
// The always-true half is the last check: memory_backed. FsAsyncRouter::groups and
// FsAsyncRouter::direct_block_for both test it and skip such a mount, which is what keeps any of this
// off the direct path.
//
// One consequence for the ladder: it stays UNEXERCISED here. statx answers on every real mount, and
// tmpfs is skipped upstream.
TEST(MountCapabilities, Tmpfs_Can_Never_Be_Probed_For_A_Block)
{
    if (!is_tmpfs("/dev/shm"))
    {
        GTEST_SKIP() << "/dev/shm is not tmpfs here";
    }

    const std::string path = "/dev/shm/runai_tmpfs_direct_" + utils::random::string();
    int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    const auto data = utils::random::buffer(256 * 1024);
    ASSERT_EQ(::write(fd, data.data(), data.size()), static_cast<ssize_t>(data.size()));
    ::close(fd);

    // Nothing upstream of the ladder answers here, which is what would leave the ladder deciding.
    EXPECT_FALSE(raw_statx_dio_align(path).answered)
        << "statx now answers on tmpfs, so this reasoning needs revisiting";

    fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);

    // ONE tail for both branches. Written as if/else rather than an early return because everything
    // below applies to both, and duplicating it is how the two got out of step: the refused branch
    // unlinked the file before asking of_path about it, and of_path stats the path first, so it threw
    // instead of asserting.
    if (fd < 0)
    {
        // This kernel refuses at open, so a probe never gets to ask. Recorded, not asserted - the
        // other branch is equally acceptable.
        RecordProperty("tmpfs_o_direct_open", "refused");
    }
    else
    {
        RecordProperty("tmpfs_o_direct_open", "accepted");

        // Aligned by construction, the way the other tests here do it, so nothing can fail between
        // creating the file and removing it.
        std::vector<char> raw(3 * 4096);
        char * const buffer = raw.data() + ((-reinterpret_cast<uintptr_t>(raw.data())) % 4096);

        // A filesystem doing real direct I/O at 4096 rejects each of these with EINVAL. tmpfs, having
        // opened the fd, must accept at least one - otherwise it is enforcing an alignment it cannot
        // honour, and the ladder would be entitled to believe it.
        const bool tolerated = ::pread(fd, buffer + 1, 4096, 0) >= 0   // unaligned buffer
                            || ::pread(fd, buffer, 4095, 0) >= 0       // odd length
                            || ::pread(fd, buffer, 4096, 1) >= 0;      // odd offset

        ::close(fd);

        EXPECT_TRUE(tolerated)
            << "tmpfs both accepted an O_DIRECT open and enforced alignment on this kernel;"
            << " a probe would now return a meaningful block for it";
    }

    // The half that holds on every kernel, and the one the routing actually uses. Asked while the
    // file still exists.
    MountCapabilities mounts;
    EXPECT_TRUE(mounts.of_path(path).memory_backed);

    ::unlink(path.c_str());
}

// The two answers come from ONE probe, so they cannot disagree: a mount with no usable block serves
// no direct reads, and a mount with one does.
TEST(MountCapabilities, Direct_Block_And_Direct_Support_Agree)
{
    utils::temp::File file(utils::random::buffer(4096));
    MountCapabilities mounts;

    size_t block = 0;
    mounts.direct_block(device_of(file.path), file.path, block);
    const auto support = mounts.direct_support(device_of(file.path), file.path);

    EXPECT_EQ(block != 0, support == DirectSupport::Yes);
}

// One open per mount, however many files sit on it. A model's shards share a mount, so without the
// cache a 200-shard load would open and close 200 extra times.
TEST(MountCapabilities, Direct_Support_Is_Probed_Once_Per_Mount)
{
    const auto data = utils::random::buffer(4096);
    utils::temp::File first(data);
    utils::temp::File second(data);

    const dev_t dev = device_of(first.path);
    ASSERT_EQ(dev, device_of(second.path)) << "both temp files should be on one mount";

    MountCapabilities mounts;
    const auto answer = mounts.direct_support(dev, first.path);

    // The second call names a path that does not exist. If it probed, it would report Unknown - so
    // getting the first answer back proves it did not.
    EXPECT_EQ(mounts.direct_support(dev, "/no/such/path/" + utils::random::string()), answer);
}

// A file we cannot open says nothing about its mount. Reporting No would send every other file on
// that mount to the synchronous reader because of one bad path.
TEST(MountCapabilities, A_Missing_File_Leaves_Direct_Support_Unknown)
{
    MountCapabilities mounts;
    const auto path = "/no/such/path/" + utils::random::string();

    EXPECT_EQ(mounts.direct_support(device_of("/tmp"), path), DirectSupport::Unknown);

    // And nothing was remembered, so a readable file on the same mount can still answer.
    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);
    EXPECT_NE(mounts.direct_support(device_of(file.path), file.path), DirectSupport::Unknown);
}

// A directory can never be opened with O_DIRECT - measured, EINVAL - so probing with one would report
// every mount as incapable. This pins that the probe is given a file.
TEST(MountCapabilities, A_Directory_Is_Not_A_Valid_Probe_Target)
{
    const auto data = utils::random::buffer(4096);
    utils::temp::File file(data);

    MountCapabilities mounts;
    const auto by_file = mounts.direct_support(device_of(file.path), file.path);

    // A fresh instance, so the cache cannot answer for it.
    MountCapabilities by_directory_mounts;
    const auto by_directory = by_directory_mounts.direct_support(device_of("/tmp"), "/tmp");

    EXPECT_EQ(by_directory, DirectSupport::No)
        << "a directory refuses O_DIRECT whatever the mount can do";

    if (by_file == DirectSupport::Yes)
    {
        EXPECT_NE(by_file, by_directory) << "which is exactly why the probe must be given a file";
    }
}

// The filesystem type comes from the kernel, and is checked against the kernel - not against a name
// hard-coded here, which would only hold on the machine it was written on.
TEST(MountCapabilities, Fs_Type_Matches_What_Mountinfo_Says)
{
    utils::temp::File file(utils::random::buffer(64));

    MountCapabilities mounts;
    const auto capability = mounts.of_path(file.path);

    // The same lookup, done independently: find this device in mountinfo and take the field after the
    // separator.
    struct stat st;
    ASSERT_EQ(::stat(file.path.c_str(), &st), 0);

    std::string expected;
    std::ifstream mountinfo("/proc/self/mountinfo");
    ASSERT_TRUE(mountinfo.is_open()) << "this host does not present mountinfo; the test cannot judge";

    std::string line;
    while (std::getline(mountinfo, line))
    {
        std::istringstream fields(line);
        std::string field, device;
        std::vector<std::string> before;
        bool separated = false;

        while (fields >> field)
        {
            if (field == "-") { separated = true; break; }
            before.push_back(field);
        }
        if (!separated || before.size() < 3) continue;

        const auto colon = before[2].find(':');
        if (colon == std::string::npos) continue;

        if (::makedev(std::stoul(before[2].substr(0, colon)),
                      std::stoul(before[2].substr(colon + 1))) == st.st_dev)
        {
            std::string type;
            if (fields >> type && expected.empty())
            {
                expected = type;   // first entry for the device wins, as the implementation does
            }
        }
    }

    // Both outcomes are checked. A listed device must agree with mountinfo; an unlisted one must
    // answer empty, which is the documented fallback and is otherwise unreachable - fs_type is only
    // ever filled by of_path, keyed on the st_dev of a real path.
    EXPECT_EQ(capability.fs_type, expected);
}

// The type cache is a process-wide static, so instances must not disagree about a device.
//
// The unknown-device case - an empty type rather than a guess or a throw - is NOT tested here and is
// not reachable: fs_type is only ever filled by of_path, keyed on the st_dev of a real path, so a
// device absent from mountinfo cannot be asked for.
TEST(MountCapabilities, Every_Instance_Reports_The_Same_Fs_Type)
{
    utils::temp::File file(utils::random::buffer(64));

    MountCapabilities one;
    MountCapabilities two;

    EXPECT_EQ(one.of_path(file.path).fs_type, two.of_path(file.path).fs_type);
}

}; // namespace runai::llm::streamer::posix_io
