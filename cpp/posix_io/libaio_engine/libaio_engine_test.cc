/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/libaio_engine/libaio_engine.h"

#include <gtest/gtest.h>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <set>
#include <vector>

#include "posix_io/alignment/alignment.h"
#include "utils/random/random.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::posix_io
{

namespace
{

// The routing code decides congruence before any engine exists, so it carries its own copy of the
// block size. If the two ever differ, routing and the worker disagree about which files can be read
// directly, and nothing fails - the reads just take a path nobody intended. Tying the tests to the
// constant is what turns that into a build failure.
// Follows the engine rather than the default, so the test still lines up if
// RUNAI_STREAMER_DIRECT_BLOCK overrides it. const, not constexpr: it is read at runtime now.
const size_t Block = direct_block_size();

// These tests have no availability gate, unlike the io_uring ones.
//
// io_uring needs a gate because container runtimes block io_uring_setup in their default seccomp
// profile. libaio is not in that set - measured, io_setup returns 0 both with and without seccomp -
// so these tests run everywhere, including CI. That is the whole point of shipping this engine.

AsyncIoConfig config_with(unsigned depth)
{
    AsyncIoConfig config;
    config.depth = depth;
    config.chunk_bytesize = 1 << 20;
    return config;
}

// A file of known bytes, and the fd to read it through.
struct Fixture
{
    explicit Fixture(size_t bytesize) :
        data(utils::random::buffer(bytesize)),
        file(data),
        fd(::open(file.path.c_str(), O_RDONLY))
    {
        EXPECT_GE(fd, 0) << "open " << file.path << ": " << std::strerror(errno);
    }

    ~Fixture()
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    FileRef ref() const { return FileRef{ fd, false }; }

    std::vector<char> expected_at(size_t offset, size_t bytesize) const
    {
        return std::vector<char>(data.begin() + offset, data.begin() + offset + bytesize);
    }

    std::vector<uint8_t> data;
    utils::temp::File file;
    int fd = -1;
};

// The same file opened with O_DIRECT, which some mounts refuse.
struct DirectFixture
{
    explicit DirectFixture(size_t bytesize) :
        data(utils::random::buffer(bytesize)),
        file(data),
        fd(::open(file.path.c_str(), O_RDONLY | O_DIRECT))
    {
        // Kept rather than read from errno later: errno belongs to the last failed call anywhere, so
        // by the time a test reports why it stopped, something else may have overwritten it.
        if (fd < 0)
        {
            error = errno;
        }
    }

    ~DirectFixture()
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    bool supported() const { return fd >= 0; }

    FileRef ref() const { return FileRef{ fd, true }; }

    std::vector<char> expected_at(size_t offset, size_t bytesize) const
    {
        return std::vector<char>(data.begin() + offset, data.begin() + offset + bytesize);
    }

    std::vector<uint8_t> data;
    utils::temp::File file;
    int fd = -1;
    int error = 0;
};

// A buffer whose address is a multiple of Block.
struct AlignedBuffer
{
    explicit AlignedBuffer(size_t bytesize)
    {
        void * raw = nullptr;
        if (::posix_memalign(&raw, Block, bytesize) != 0)
        {
            ADD_FAILURE() << "posix_memalign(" << bytesize << ") failed";
            return;
        }
        base = static_cast<char *>(raw);
        size = bytesize;
    }

    ~AlignedBuffer() { ::free(base); }

    char * base = nullptr;
    size_t size = 0;
};

// Reap until `count` completions have arrived, or too many rounds return nothing.
std::vector<Completion> reap(LibaioEngine & engine, unsigned count)
{
    std::vector<Completion> out;
    std::vector<Completion> batch(count);

    for (unsigned round = 0; round < count + 8 && out.size() < count; ++round)
    {
        unsigned got = 0;
        EXPECT_EQ(engine.wait_for_completions(batch.data(), batch.size(), got, WaitMode::Block, 2000),
                  common::ResponseCode::Success);
        out.insert(out.end(), batch.begin(), batch.begin() + got);
    }

    EXPECT_EQ(out.size(), count) << "only " << out.size() << " of " << count << " completions arrived";
    return out;
}

const Completion * completion_for(const std::vector<Completion> & completions, RequestId id)
{
    const auto it = std::find_if(completions.begin(), completions.end(),
                                 [id](const Completion & c) { return c.id == id; });
    return it == completions.end() ? nullptr : &*it;
}

} // namespace

TEST(LibaioEngine, Reads_A_Range)
{
    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(8));

    std::vector<char> got(1024);
    ASSERT_EQ(engine.stage(7, fixture.ref(), 2048, got.size(), got.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_EQ(completions[0].id, 7);
    EXPECT_EQ(completions[0].bytes_transferred(), got.size());
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(got, fixture.expected_at(2048, got.size()));
}

// One read per io_submit. Fixed, not configurable - see SubmitBatch in the .cc.
//
// io_submit BLOCKS on a filesystem that cannot queue a direct read without work in the submission
// path, and the cost per read grows with the batch. Measured on NFS: 687 us per read at ~44 per call
// against fio's 559 us submitting one at a time, and the throughput gap matched that ratio. So
// stage-then-flush, which exists to amortise a syscall, amortises nothing here and makes each read
// dearer.
//
// Pinned because nothing else would notice it changing. A larger batch keeps every other test green -
// the same reads go out, against the same destinations, and the same bytes arrive - and shows up only
// as a slower run on a network filesystem.
//
// io_uring is deliberately NOT capped this way: measured at 1 us per read on the same mount, and it
// already averages 1.1 reads per call without being asked.
TEST(LibaioEngine, Submits_One_Read_Per_Call)
{
    constexpr unsigned Reads = 6;

    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(16));

    std::vector<std::vector<char>> buffers(Reads, std::vector<char>(512));
    for (unsigned i = 0; i < Reads; ++i)
    {
        ASSERT_EQ(engine.stage(300 + i, fixture.ref(), i * 512, 512, buffers[i].data()),
                  common::ResponseCode::Success);
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    EXPECT_EQ(issued, Reads);
    EXPECT_EQ(engine.submit_stats().calls, Reads) << "one io_submit per read, always";
    EXPECT_EQ(engine.submit_stats().requests, Reads);

    const auto completions = reap(engine, Reads);
    for (unsigned i = 0; i < Reads; ++i)
    {
        const auto * completion = completion_for(completions, 300 + i);
        ASSERT_NE(completion, nullptr) << "no completion for id " << 300 + i;
        EXPECT_EQ(buffers[i], fixture.expected_at(i * 512, 512));
    }
}

// min_nr and nr answer different questions. min_nr is how many completions to wait for; nr is how
// many to return. So min_nr = 1 means "do not wait for a second one", and the call still gives back
// everything that is ready. Measured: with five reads finished, io_getevents(min_nr = 1, nr = 16)
// returns all five.
//
// Pinned because nothing else would notice it breaking. Sizing `nr` or the event buffer to 1 would
// cost one syscall per read instead of one per batch, and the only symptom would be lower throughput
// - no failure, no error, nothing in a log.
TEST(LibaioEngine, One_Wait_Returns_Every_Ready_Completion)
{
    constexpr unsigned Reads = 5;

    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(16));

    std::vector<std::vector<char>> buffers(Reads, std::vector<char>(512));
    for (unsigned i = 0; i < Reads; ++i)
    {
        ASSERT_EQ(engine.stage(i, fixture.ref(), i * 512, 512, buffers[i].data()), common::ResponseCode::Success);
    }

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    ASSERT_EQ(issued, Reads);

    // Let every read finish, so that several really are ready at the same moment - which is the state
    // this test is about. The file is small and warm, so these take microseconds; the wait is four
    // orders of magnitude longer than needed.
    ::usleep(200000);

    // Non-blocking on purpose. With a blocking wait, someone could argue it had waited for the rest,
    // and then the test would prove nothing about how much one call collects.
    std::vector<Completion> completions(Reads * 2);
    unsigned count = 0;
    ASSERT_EQ(engine.wait_for_completions(completions.data(), completions.size(), count,
                                          WaitMode::NonBlocking),
              common::ResponseCode::Success);

    EXPECT_EQ(count, Reads) << "one call must collect every completion that is ready, not just the first";
}

// The iocbs are a fixed pool, so a leak here shows up as stage() refusing work after `depth` reads
// rather than as anything visible in one pass.
TEST(LibaioEngine, Iocbs_Are_Reused_Across_Many_Rounds)
{
    constexpr unsigned Depth = 2;
    constexpr unsigned Rounds = 20;

    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(Depth));

    for (unsigned round = 0; round < Rounds; ++round)
    {
        std::vector<char> got(256);
        ASSERT_EQ(engine.stage(round, fixture.ref(), round * 256, got.size(), got.data()),
                  common::ResponseCode::Success)
            << "stage refused at round " << round << " - an iocb was not returned to the free list";

        unsigned issued = 0;
        ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
        ASSERT_EQ(issued, 1);

        const auto completions = reap(engine, 1);
        ASSERT_EQ(completions.size(), 1);
        EXPECT_EQ(got, fixture.expected_at(round * 256, got.size()));
    }
}

TEST(LibaioEngine, Staging_Past_The_Depth_Is_Refused_Rather_Than_Fatal)
{
    constexpr unsigned Depth = 2;

    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(Depth));

    std::vector<std::vector<char>> buffers(Depth + 1, std::vector<char>(256));
    for (unsigned i = 0; i < Depth; ++i)
    {
        ASSERT_EQ(engine.stage(i, fixture.ref(), 0, 256, buffers[i].data()), common::ResponseCode::Success);
    }

    // The caller's window makes this unreachable. It is reported rather than asserted because killing
    // the host process over a broken invariant is worse than failing one read.
    EXPECT_EQ(engine.stage(Depth, fixture.ref(), 0, 256, buffers[Depth].data()),
              common::ResponseCode::UnknownError);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    reap(engine, Depth);
}

// io_submit accepts a prefix and stops at the first read it will not take. Measured: with a bad fd at
// index 1 of three, it returns 1.
//
// Design 5.9.2 asks for a test that fills the context, then checks that flush reports a partial issue
// and the next drain finishes the backlog without spinning. The context cannot be filled here: the
// free list caps staging at exactly the number of events the context holds, so staged plus
// outstanding can never pass it. This reaches the same partial path the other way, with a read the
// kernel will not take, and checks the same two things.
TEST(LibaioEngine, A_Refused_Read_Does_Not_Block_The_Ones_Behind_It)
{
    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(8));

    std::vector<char> first(256), second(256), third(256);

    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, first.size(), first.data()), common::ResponseCode::Success);
    ASSERT_EQ(engine.stage(2, FileRef{ -1, false }, 0, second.size(), second.data()),
              common::ResponseCode::Success);
    ASSERT_EQ(engine.stage(3, fixture.ref(), 1024, third.size(), third.data()), common::ResponseCode::Success);

    // Pass one: the good head goes out, io_submit refuses the bad fd, and flush answers it as a
    // synthetic completion inside this same call. Success, because the flush itself did not fail -
    // a failure would make the worker abort every read on this engine.
    //
    // TWO, not one. The refused read is counted even though the kernel took nothing, because a
    // completion is coming for it. This assertion used to read `EXPECT_EQ(issued, 1)` with the note
    // "only the good head should have been issued" - true of the kernel, and wrong for the caller,
    // which uses this number to know how many completions to expect. See flush() in io_engine.h.
    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    unsigned total_issued = issued;
    EXPECT_EQ(issued, 2) << "the good head, plus the refused read that will be answered anyway";

    // Pass two: the read that was behind the bad one goes out normally - it was never stuck.
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    total_issued += issued;
    EXPECT_EQ(issued, 1) << "the read behind the refused one must still go out";

    const auto completions = reap(engine, 3);

    const auto * good = completion_for(completions, 1);
    ASSERT_NE(good, nullptr);
    EXPECT_EQ(good->bytes_transferred(), first.size());

    // The refused read is reported the same way io_uring would report it: one completion, minus the
    // errno. The caller cannot tell the two engines apart, which is what lets one mapper serve both.
    const auto * refused = completion_for(completions, 2);
    ASSERT_NE(refused, nullptr);
    EXPECT_TRUE(refused->failed());
    EXPECT_EQ(refused->res, -EBADF);
    EXPECT_EQ(refused->bytes_transferred(), 0);

    const auto * behind = completion_for(completions, 3);
    ASSERT_NE(behind, nullptr);
    EXPECT_EQ(behind->bytes_transferred(), third.size());
    EXPECT_EQ(third, fixture.expected_at(1024, third.size()));

    // THE invariant, asserted rather than left implied: one completion for every read counted as
    // issued. The caller adds the first number to its in-flight count and subtracts the second, one
    // completion at a time, so any gap between them either trips its assertion or - worse, and
    // silently - leaves the count low, and quiesce() reports while the kernel still holds a buffer.
    EXPECT_EQ(total_issued, completions.size())
        << "every completion must have been counted by a flush, or the caller's in-flight count drifts";
}

TEST(LibaioEngine, A_Short_Read_Is_Not_An_Error)
{
    constexpr size_t Bytesize = 1000;

    Fixture fixture(Bytesize);
    LibaioEngine engine(config_with(4));

    // Ask for more than the file holds. The kernel returns what there was, as a small positive
    // number - which is why a caller may never read "no error" as "all the bytes arrived".
    std::vector<char> got(Bytesize * 2);
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, got.size(), got.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(completions[0].bytes_transferred(), Bytesize);
}

TEST(LibaioEngine, Flushing_Nothing_Is_Success)
{
    LibaioEngine engine(config_with(4));

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 0);
    EXPECT_EQ(engine.submit_stats().calls, 0) << "an empty flush should not call io_submit at all";
}

TEST(LibaioEngine, A_Non_Blocking_Wait_With_Nothing_In_Flight_Returns_At_Once)
{
    LibaioEngine engine(config_with(4));

    Completion completions[4];
    unsigned count = 7;

    EXPECT_EQ(engine.wait_for_completions(completions, 4, count, WaitMode::NonBlocking),
              common::ResponseCode::Success);
    EXPECT_EQ(count, 0);
}

TEST(LibaioEngine, An_Expired_Wait_Is_Success_With_Nothing)
{
    LibaioEngine engine(config_with(4));

    Completion completions[4];
    unsigned count = 7;

    // io_getevents reports a timeout by returning zero events rather than by failing, so this must
    // not look like an error. It is also the teardown wake-up: nothing else may touch the engine, so
    // this returning is the only way a waiting worker learns it should stop.
    EXPECT_EQ(engine.wait_for_completions(completions, 4, count, WaitMode::Block, 50),
              common::ResponseCode::Success);
    EXPECT_EQ(count, 0);
}

TEST(LibaioEngine, Limits_Describe_A_Direct_Read)
{
    LibaioEngine engine(config_with(4));

    const auto limits = engine.limits();
    EXPECT_EQ(limits.max_read_bytesize, max_read_bytesize());

    // Not 1. The caller tests congruence against these, everything is congruent modulo 1, and every
    // file would then be opened with O_DIRECT and every unaligned read would fail with EINVAL.
    EXPECT_EQ(limits.offset_alignment, Block);
    EXPECT_EQ(limits.buffer_alignment, Block);
}

TEST(LibaioEngine, Depth_Is_What_The_Context_Granted)
{
    LibaioEngine engine(config_with(8));

    // Never larger than asked for. It can be smaller when the node's aio limit is reached, and the
    // caller sizes its window from this rather than from the configured depth.
    EXPECT_GE(engine.depth(), 1);
    EXPECT_LE(engine.depth(), 8);
}

TEST(LibaioEngine, Submit_Time_Is_Measured)
{
    Fixture fixture(64 * 1024);
    LibaioEngine engine(config_with(4));

    std::vector<char> got(256);
    ASSERT_EQ(engine.stage(1, fixture.ref(), 0, got.size(), got.data()), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    reap(engine, 1);

    // These numbers decide whether one thread was the right choice, or whether a submit thread has to
    // be designed. A counter that is never filled would answer that question with a silent zero.
    const auto stats = engine.submit_stats();
    EXPECT_EQ(stats.calls, 1);
    EXPECT_EQ(stats.requests, 1);
    EXPECT_GT(stats.nanos, 0);
    EXPECT_GT(stats.max_nanos, 0);
    EXPECT_LE(stats.max_nanos, stats.nanos);
}

TEST(LibaioEngine, Reads_A_Direct_Range)
{
    // Sized FROM the block, not to a fixed 64 KB. It reads one block starting one block in, so a file
    // of exactly one block would put that read at EOF - which is what a hardcoded size did once the
    // block was raised to 64 KB.
    const size_t Bytesize = 4 * Block;

    DirectFixture fixture(Bytesize);
    if (!fixture.supported())
    {
        GTEST_SKIP() << "this mount refuses O_DIRECT (" << std::strerror(fixture.error) << ")";
    }

    LibaioEngine engine(config_with(4));
    AlignedBuffer buffer(Block);
    ASSERT_NE(buffer.base, nullptr);

    ASSERT_EQ(engine.stage(1, fixture.ref(), Block, Block, buffer.base), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_FALSE(completions[0].failed()) << "res " << completions[0].res;
    EXPECT_EQ(completions[0].bytes_transferred(), Block);
    EXPECT_EQ(std::vector<char>(buffer.base, buffer.base + Block), fixture.expected_at(Block, Block));
}

// Alignment is checked by the kernel at COMPLETION time, not at submit time - measured on both a
// misaligned offset, a misaligned buffer and a misaligned length. That is the same place io_uring
// reports it, which is what lets one completion mapper serve both engines.
//
// What this test does not cover: io_event.res is declared unsigned long, and the engine casts it back
// to signed. That cast is only for the reader - the implicit conversion keeps the same bits, so -22
// arrives as -22 with or without it. Removing the cast was tried, and no test noticed.
TEST(LibaioEngine, A_Misaligned_Direct_Read_Fails_As_A_Completion)
{
    constexpr size_t Bytesize = 64 * 1024;

    DirectFixture fixture(Bytesize);
    if (!fixture.supported())
    {
        GTEST_SKIP() << "this mount refuses O_DIRECT (" << std::strerror(fixture.error) << ")";
    }

    LibaioEngine engine(config_with(4));
    AlignedBuffer buffer(Block * 2);
    ASSERT_NE(buffer.base, nullptr);

    // An offset inside a block, which O_DIRECT does not allow.
    ASSERT_EQ(engine.stage(1, fixture.ref(), 100, Block, buffer.base), common::ResponseCode::Success);

    unsigned issued = 0;
    ASSERT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1) << "the kernel accepts the submission and reports the problem later";

    const auto completions = reap(engine, 1);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_TRUE(completions[0].failed());
    EXPECT_EQ(completions[0].res, -EINVAL);
}


}; // namespace runai::llm::streamer::posix_io
