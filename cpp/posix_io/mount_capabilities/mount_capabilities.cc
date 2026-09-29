/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// O_DIRECT is a GNU extension, so this must come before any libc header.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "posix_io/mount_capabilities/mount_capabilities.h"

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysmacros.h>   // major/minor - moved out of sys/types.h in glibc 2.28
#include <sys/vfs.h>
#include <sys/syscall.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "common/exception/exception.h"
#include "posix_io/alignment/alignment.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// tmpfs and ramfs hold their pages in the page cache, so there is no device to bypass and O_DIRECT
// cannot work - which is what routes them to the synchronous pool.
//
// This is the ONLY probe that detects the case. An O_DIRECT open does not: measured, opening a tmpfs
// file with O_RDONLY | O_DIRECT SUCCEEDS, and the failure comes later at read time.
bool is_memory_backed(const struct statfs & fs)
{
    return fs.f_type == TMPFS_MAGIC || fs.f_type == RAMFS_MAGIC;
}


// STATX_DIOALIGN, asked of the kernel directly.
//
// glibc's struct statx here ends at stx_mnt_id - the direct-I/O fields are missing from the build
// headers even on a 6.8 kernel - so this declares the layout it needs and calls the syscall. Same
// approach io_uring_probe_test uses to ask the kernel without going through a wrapper.
//
// Returns 0 when the kernel cannot answer: below 6.1, on a filesystem that does not implement it, or
// when it reports direct I/O as unsupported (both alignments zero). The caller then measures instead.
#ifndef STATX_DIOALIGN
#define STATX_DIOALIGN 0x00002000U
#endif

struct DioAlign
{
    size_t mem = 0;
    size_t offset = 0;
};

DioAlign statx_dio_align(const std::string & path)
{
    // Only the two fields matter, but the struct must be the size the kernel expects to write.
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

    if (::syscall(__NR_statx, AT_FDCWD, path.c_str(), 0, STATX_DIOALIGN, &buf) != 0)
    {
        return {};
    }

    if ((buf.mask & STATX_DIOALIGN) == 0)
    {
        return {};   // the kernel did not fill these in
    }

    return { buf.dio_mem_align, buf.dio_offset_align };
}

// MaxProbeBlock comes from alignment.h: it is the ladder's top rung AND what a worker runs at while
// it has no measurement, so the two must be the same number.

// Can this fd serve a direct read of exactly `block` bytes, at that alignment?
//
// A real read, because that is the only answer that cannot be wrong. A file shorter than a block
// returns a short read rather than EINVAL, which still proves the alignment was accepted.
//
// The buffer is placed at `base + block`, NOT at whatever posix_memalign(block) returns. That matters:
// posix_memalign only promises AT LEAST the requested alignment, so asking for 512 can hand back a
// page-aligned address - and then a mount that really needs 4096 would accept the read and the ladder
// would report 512. Measured on this container: posix_memalign(512) returned an address that happened
// to be 512 mod 4096, so the answer was right by luck rather than by construction.
//
// base is aligned to MaxProbeBlock, so base + block is aligned to `block` and, because block divides
// MaxProbeBlock, is NOT aligned to 2 * block. That is the strictest test of this candidate.
bool direct_read_at(int fd, size_t block)
{
    void * base = nullptr;
    if (::posix_memalign(&base, MaxProbeBlock, 2 * MaxProbeBlock) != 0)
    {
        return false;
    }

    char * const buffer = static_cast<char *>(base) + block;

    const ssize_t got = ::pread(fd, buffer, block, 0);
    const int error = errno;

    ::free(base);

    // EINVAL is the alignment answer. Anything else belongs to this file or this moment.
    return got >= 0 || error != EINVAL;
}

// What direct_read_at above assumes of EVERY rung it is given, checked at build time.
//
// Four properties, and each one is silent if it breaks:
//
//   - a power of two, or posix_memalign rejects it and the rung reports "no" for every mount;
//   - a divisor of MaxProbeBlock, or base + block is not aligned to block, and the probe then asks a
//     weaker question than the rung it claims to be testing;
//   - not larger than MaxProbeBlock, or the read writes [block, 2 * block) past the end of a
//     2 * MaxProbeBlock allocation - a heap overflow;
//   - strictly larger than the rung before it, because the caller takes the first acceptance as the
//     smallest usable block, which is only true of a sorted ladder.
//
// A rung of 131072 breaks the third, and is the plausible addition: larger logical blocks are what
// this whole path exists for. Inserting 2048 in the wrong place breaks the fourth, and would make the
// answer wrong rather than crash - the worse failure of the two.
//
// The last rung must also BE MaxProbeBlock, not merely fit under it. A worker runs at MaxProbeBlock
// while it waits for a measurement, so a ladder that stopped at 16384 could never confirm the value
// already in use, and every mount needing more would be reported as serving no direct reads at all.
template <size_t N>
constexpr bool ladder_is_usable(const std::array<size_t, N> & rungs)
{
    for (size_t i = 0; i < N; ++i)
    {
        const size_t rung = rungs[i];

        if (rung == 0 || (rung & (rung - 1)) != 0)
        {
            return false;
        }

        if (rung > MaxProbeBlock || MaxProbeBlock % rung != 0)
        {
            return false;
        }

        if (i != 0 && rung <= rungs[i - 1])
        {
            return false;
        }
    }

    return N != 0 && rungs[N - 1] == MaxProbeBlock;
}

// Field 3 is "major:minor", which is st_dev, and the type is the field after the " - " separator:
//
//     36 25 252:1 / /mnt rw,relatime - ext4 /dev/sda1 rw
//     ^^ ^^ ^^^^^                      ^^^^
//     id parent DEVICE                 TYPE
//
// Read once and cached: mountinfo does not change under a running streamer in any way that matters,
// and the alternative is one open and parse per mount probe. An empty map is normal - a container may
// not present mountinfo, and the caller then keeps its global default.
const std::map<dev_t, std::string> & fs_types_by_device()
{
    static const std::map<dev_t, std::string> types = []
    {
        std::map<dev_t, std::string> out;

        std::ifstream mountinfo("/proc/self/mountinfo");
        if (!mountinfo.is_open())
        {
            LOG(DEBUG) << "/proc/self/mountinfo is not readable; filesystem types are unknown here";
            return out;
        }

        std::string line;
        while (std::getline(mountinfo, line))
        {
            std::istringstream fields(line);
            std::string field;
            std::vector<std::string> before;

            // Everything up to the separator. The optional fields between field 6 and the separator
            // are why the type cannot be taken by index.
            bool separated = false;
            while (fields >> field)
            {
                if (field == "-")
                {
                    separated = true;
                    break;
                }
                before.push_back(field);
            }

            std::string type;
            if (!separated || before.size() < 3 || !(fields >> type))
            {
                continue;
            }

            const auto colon = before[2].find(':');
            if (colon == std::string::npos)
            {
                continue;
            }

            try
            {
                const auto major_number = std::stoul(before[2].substr(0, colon));
                const auto minor_number = std::stoul(before[2].substr(colon + 1));

                // The FIRST entry for a device wins. A device can be mounted more than once, and the
                // type is the same every time - so this only decides which duplicate we keep.
                out.try_emplace(makedev(major_number, minor_number), type);
            }
            catch (const std::exception &)
            {
                continue;   // a line we cannot read tells us nothing about the ones after it
            }
        }

        return out;
    }();

    return types;
}

} // namespace

std::ostream & operator<<(std::ostream & os, const MountCapability & capability)
{
    os << "mount " << major(capability.dev) << ":" << minor(capability.dev);

    if (!capability.fs_type.empty())
    {
        os << " (" << capability.fs_type << ")";
    }

    return os << (capability.memory_backed ? " memory-backed" : "");
}

std::ostream & operator<<(std::ostream & os, DirectSupport support)
{
    switch (support)
    {
    case DirectSupport::Yes:     return os << "O_DIRECT supported";
    case DirectSupport::No:      return os << "O_DIRECT unsupported";
    case DirectSupport::Unknown: return os << "O_DIRECT support unknown";
    }
    return os << "O_DIRECT support " << static_cast<int>(support);
}

MountCapability MountCapabilities::remember(dev_t dev, bool memory_backed)
{
    const auto & types = fs_types_by_device();
    const auto type = types.find(dev);

    const MountCapability capability{ dev, memory_backed,
                                      type == types.end() ? std::string() : type->second };

    std::unique_lock<std::mutex> lock(_mutex);

    // try_emplace, not assignment: two threads can probe the same mount at once, and the results are
    // identical, so the first one wins and the second is a no-op rather than a race worth avoiding.
    const auto [it, inserted] = _mounts.try_emplace(dev, capability);
    if (inserted)
    {
        LOG(DEBUG) << "Probed " << capability;
    }
    return it->second;
}


MountCapability MountCapabilities::of_path(const std::string & path)
{
    // stat before statfs so a missing file is reported as such rather than as a mount failure.
    struct stat st;
    if (::stat(path.c_str(), &st) != 0)
    {
        LOG(ERROR) << "Failed to stat " << path << " : " << std::strerror(errno);
        throw common::Exception(common::ResponseCode::FileAccessError);
    }

    {
        // Known mount: answer from the cache and issue no statfs at all. This is the whole point - a
        // model's shards share a mount, so only the first one costs anything.
        std::unique_lock<std::mutex> lock(_mutex);
        const auto it = _mounts.find(st.st_dev);
        if (it != _mounts.end())
        {
            return it->second;
        }
    }

    struct statfs fs;
    if (::statfs(path.c_str(), &fs) != 0)
    {
        LOG(ERROR) << "Failed to statfs " << path << " : " << std::strerror(errno);
        throw common::Exception(common::ResponseCode::FileAccessError);
    }

    return remember(st.st_dev, is_memory_backed(fs));
}

MountCapability MountCapabilities::of_fd(int fd)
{
    struct stat st;
    if (::fstat(fd, &st) != 0)
    {
        LOG(ERROR) << "Failed to fstat " << fd << " : " << std::strerror(errno);
        throw common::Exception(common::ResponseCode::FileAccessError);
    }

    {
        std::unique_lock<std::mutex> lock(_mutex);
        const auto it = _mounts.find(st.st_dev);
        if (it != _mounts.end())
        {
            return it->second;
        }
    }

    struct statfs fs;
    if (::fstatfs(fd, &fs) != 0)
    {
        LOG(ERROR) << "Failed to fstatfs " << fd << " : " << std::strerror(errno);
        throw common::Exception(common::ResponseCode::FileAccessError);
    }

    return remember(st.st_dev, is_memory_backed(fs));
}

namespace
{

} // namespace

size_t MountCapabilities::measure_direct_block(const std::string & file_path)
{
    // The kernel's own answer first: exact, and no I/O.
    //
    // Two numbers, and they govern different things - stx_dio_mem_align the buffer address,
    // stx_dio_offset_align the file offset AND the transfer length. We test congruence with a single
    // block, so take the larger; congruence at a power of two implies congruence at every smaller one.
    const auto reported = statx_dio_align(file_path);
    if (reported.mem != 0 && reported.offset != 0)
    {
        const size_t block = std::max(reported.mem, reported.offset);

        LOG(DEBUG) << "statx reports direct-I/O alignment for " << file_path << ": memory "
                   << reported.mem << ", offset and length " << reported.offset;
        return block;
    }

    // No statx answer - below 6.1, or a filesystem that does not implement it. Measure by reading.
    const int fd = ::open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0)
    {
        return 0;   // the caller already distinguishes "mount refuses" from "cannot tell"
    }

    // ASCENDING, and the first success wins. The smallest workable block is the one we want: a larger
    // one pads more between ranges and bounces more of every chunk's head and tail. 512 is the
    // smallest logical block any device reports; 65536 covers every size announced so far.
    //
    // A GUESS, and weaker than statx above. It assumes the filesystem REFUSES a badly aligned read.
    // Most do - measured on this container's overlay, a wrong offset, a wrong length and a wrong
    // buffer address are each rejected with EINVAL. But a lenient filesystem accepts the read and
    // quietly serves it buffered instead, with no error.
    //
    // On such a filesystem the first rung always "works", so this returns 512 for a mount that really
    // needs 4096. Nothing fails: we would believe reads are direct while the kernel copies through
    // its cache, and the log would agree with us. Slower than we think, and silent about it.
    //
    // Reachable only when BOTH hold: statx cannot answer (below kernel 6.1, or a filesystem that does
    // not implement STATX_DIOALIGN), AND that filesystem is lenient about alignment. Phase 1's EINVAL
    // fallback does not help here either - a lenient filesystem never returns EINVAL, which is the
    // whole problem.
    //
    // The top rung IS MaxProbeBlock, not a literal that happens to equal it, so lowering that constant
    // cannot leave a rung behind that overruns the probe buffer. ladder_is_usable checks that and the
    // rest of what direct_read_at needs, on every rung.
    static constexpr std::array<size_t, 4> ladder{ 512, 4096, 16384, MaxProbeBlock };
    static_assert(ladder_is_usable(ladder),
                  "a rung must be a power of two, divide MaxProbeBlock, and come after the rung below"
                  " it; the last rung must be MaxProbeBlock");

    for (const size_t candidate : ladder)
    {
        if (direct_read_at(fd, candidate))
        {
            ::close(fd);
            return candidate;
        }
    }

    ::close(fd);

    // Opens but reads at no size we would use. Treated as no direct I/O at all, which is what the
    // caller already handles - rather than inventing a block and failing every read with EINVAL.
    return 0;
}

DirectSupport MountCapabilities::direct_support(dev_t dev, const std::string & file_path)
{
    size_t block = 0;
    switch (direct_block(dev, file_path, block))
    {
    case common::ResponseCode::Success:         return DirectSupport::Yes;
    case common::ResponseCode::FileAccessError: return DirectSupport::No;
    default:                                    return DirectSupport::Unknown;
    }
}

common::ResponseCode MountCapabilities::direct_block(dev_t dev, const std::string & file_path,
                                                     size_t & out_block)
{
    out_block = 0;

    {
        const auto guard = std::unique_lock<std::mutex>(_mutex);
        const auto it = _direct.find(dev);
        if (it != _direct.end())
        {
            out_block = it->second;
            return out_block != 0 ? common::ResponseCode::Success : common::ResponseCode::FileAccessError;
        }
    }

    // A memory-backed mount must never reach here, and this function cannot notice if one does.
    //
    // MEASURED on tmpfs, kernel 6.8: an O_DIRECT open succeeds AND every read is accepted - an
    // unaligned buffer, an odd length, an odd offset, all of them. tmpfs enforces nothing, because
    // there is no device to bypass; the pages are the page cache. So the ladder's first rung
    // "succeeds" and this reports 512 for a filesystem that cannot do direct I/O at all.
    //
    // The callers keep that out: FsAsyncRouter::groups and FsAsyncRouter::direct_block_for both test
    // memory_backed and skip such a mount before asking. The statfs magic check is the only thing that
    // works here, because a probe by definition cannot detect a filesystem that accepts everything.
    const int fd = ::open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0)
    {
        const int error = errno;

        if (error == EINVAL || error == EOPNOTSUPP)
        {
            // The mount itself refuses O_DIRECT. That belongs to the mount, so it is remembered.
            LOG(INFO) << "Mount " << major(dev) << ":" << minor(dev) << " does not support O_DIRECT ("
                      << std::strerror(error) << ")";
            const auto guard = std::unique_lock<std::mutex>(_mutex);
            _direct.try_emplace(dev, 0);
            return common::ResponseCode::FileAccessError;
        }

        // Anything else is about the FILE - it is missing, or we may not read it. Nothing is
        // remembered, so the next file answers instead of one bad path condemning the mount.
        // Deliberately NOT remembered, so the next file - or the next submission - measures again.
        LOG(DEBUG) << "Cannot tell whether " << file_path << " supports O_DIRECT: "
                   << std::strerror(error);
        return common::ResponseCode::UnknownError;
    }
    ::close(fd);

    // An override replaces the MEASUREMENT, not the capability check above. It must not turn a
    // missing file or a directory into "this mount serves direct reads" - the open decides that, and
    // it has already run. See direct_block_override().
    const size_t forced = direct_block_override();
    const size_t block = forced != 0 ? forced : measure_direct_block(file_path);

    const auto guard = std::unique_lock<std::mutex>(_mutex);
    _direct.try_emplace(dev, block);

    if (block != 0)
    {
        LOG(INFO) << "Mount " << major(dev) << ":" << minor(dev) << " serves direct reads at a block"
                  << " of " << block << " bytes";
    }
    else
    {
        LOG(INFO) << "Mount " << major(dev) << ":" << minor(dev) << " accepts an O_DIRECT open but"
                  << " serves no aligned read we would use";
    }

    out_block = _direct.at(dev);
    return out_block != 0 ? common::ResponseCode::Success : common::ResponseCode::FileAccessError;
}

size_t MountCapabilities::size() const
{
    std::unique_lock<std::mutex> lock(_mutex);
    return _mounts.size();
}

}; // namespace runai::llm::streamer::posix_io
