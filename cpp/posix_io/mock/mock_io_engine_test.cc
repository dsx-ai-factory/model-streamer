/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/mock/mock_io_engine.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <vector>

namespace runai::llm::streamer::posix_io
{

namespace
{

constexpr unsigned Depth = 8;
constexpr size_t   Bytes = 64;

// Sentinel byte on each side: a write one byte too long, or one byte early, is invisible if only the
// requested region is compared.
struct Destination
{
    explicit Destination(size_t bytesize = Bytes) :
        storage(bytesize + 2, Guard)
    {}

    char * buffer() { return storage.data() + 1; }

    bool guards_intact() const { return storage.front() == Guard && storage.back() == Guard; }

    // What a full completion of [offset, offset + bytesize) should have written.
    static std::vector<char> expected(size_t offset, size_t bytesize)
    {
        std::vector<char> out(bytesize);
        MockIoEngine::fill_expected(out.data(), offset, bytesize);
        return out;
    }

    std::vector<char> contents(size_t bytesize) const
    {
        return std::vector<char>(storage.begin() + 1, storage.begin() + 1 + bytesize);
    }

    static constexpr char Guard = '\x7E';

    std::vector<char> storage;
};

std::vector<Completion> drain(MockIoEngine & engine, WaitMode mode = WaitMode::NonBlocking, unsigned max = Depth)
{
    std::vector<Completion> out(max);
    unsigned count = 0;

    EXPECT_EQ(engine.wait_for_completions(out.data(), max, count, mode), common::ResponseCode::Success);

    out.resize(count);
    return out;
}

} // namespace

TEST(MockIoEngine, StagedIsNotIssued)
{
    MockIoEngine engine(Depth);
    Destination dst;

    EXPECT_EQ(engine.stage(0, FileRef{ 7, false }, 1024, Bytes, dst.buffer()), common::ResponseCode::Success);

    // stage() need not issue a syscall - that is why flush() exists - so nothing is in flight yet.
    EXPECT_EQ(engine.staged_count(), 1);
    EXPECT_EQ(engine.in_flight_count(), 0);
    EXPECT_TRUE(drain(engine).empty());

    // A chunk staged at the wrong offset or destination still completes cleanly, so assert the
    // request itself.
    const auto & request = engine.request(0);
    EXPECT_EQ(request.file.fd, 7);
    EXPECT_FALSE(request.file.direct);
    EXPECT_EQ(request.offset, 1024);
    EXPECT_EQ(request.bytesize, Bytes);
    EXPECT_EQ(request.buffer, dst.buffer());
}

TEST(MockIoEngine, FlushIssuesEverythingByDefault)
{
    MockIoEngine engine(Depth);
    Destination a, b;

    engine.stage(0, FileRef{ 1, false }, 0, Bytes, a.buffer());
    engine.stage(1, FileRef{ 1, false }, Bytes, Bytes, b.buffer());

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 2);

    EXPECT_EQ(engine.staged_count(), 0);
    EXPECT_EQ(engine.in_flight(), (std::vector<RequestId>{ 0, 1 }));
}

// Both real APIs issue a prefix, so the unissued set is always the tail. Assert WHICH were issued,
// not how many - issuing the wrong two passes a count check.
TEST(MockIoEngine, PartialFlushIssuesAPrefixAndKeepsTheTail)
{
    MockIoEngine engine(Depth);
    std::vector<Destination> dsts(4);

    for (unsigned i = 0; i < 4; ++i)
    {
        engine.stage(i, FileRef{ 1, false }, i * Bytes, Bytes, dsts[i].buffer());
    }

    engine.set_flush_limit(2);

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 2);
    EXPECT_EQ(engine.in_flight(), (std::vector<RequestId>{ 0, 1 }));
    EXPECT_EQ(engine.staged(), (std::vector<RequestId>{ 2, 3 }));

    // The rest is retried on the next flush, in order.
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 2);
    EXPECT_EQ(engine.in_flight(), (std::vector<RequestId>{ 0, 1, 2, 3 }));
    EXPECT_TRUE(engine.staged().empty());
}

// Zero progress is backpressure, not an error - mistaking it for one deadlocks a single worker.
TEST(MockIoEngine, StalledFlushIssuesNothingAndIsNotAnError)
{
    MockIoEngine engine(Depth);
    Destination dst;

    engine.stage(0, FileRef{ 1, false }, 0, Bytes, dst.buffer());
    engine.set_flush_stalled(true);

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 0);
    EXPECT_EQ(engine.staged_count(), 1);
    EXPECT_EQ(engine.in_flight_count(), 0);

    // Nothing is in flight, so no completion can relieve it - the RWF_NOWAIT EAGAIN case, which
    // "reaping frees capacity" does not cover.
    EXPECT_TRUE(drain(engine).empty());

    engine.set_flush_stalled(false);
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::Success);
    EXPECT_EQ(issued, 1);
}

// The reason this double exists: the test picks the order, so a later request landing first is
// produced deliberately rather than waited for.
TEST(MockIoEngine, CompletionsArriveInTheOrderTheTestChose)
{
    MockIoEngine engine(Depth);
    std::vector<Destination> dsts(3);

    for (unsigned i = 0; i < 3; ++i)
    {
        engine.stage(i, FileRef{ 1, false }, i * Bytes, Bytes, dsts[i].buffer());
    }
    unsigned issued = 0;
    engine.flush(issued);

    engine.complete(2);
    engine.complete(0);
    engine.complete(1);

    const auto completions = drain(engine);
    ASSERT_EQ(completions.size(), 3);
    EXPECT_EQ(completions[0].id, 2);
    EXPECT_EQ(completions[1].id, 0);
    EXPECT_EQ(completions[2].id, 1);
}

// An expired timeout is Success with zero completions, not a distinct code, so the caller's loop has
// one success shape. With nothing ready both wait modes are that case.
TEST(MockIoEngine, NothingReadyIsSuccessWithZeroForBothWaitModes)
{
    MockIoEngine engine(Depth);

    EXPECT_TRUE(drain(engine, WaitMode::NonBlocking).empty());
    EXPECT_TRUE(drain(engine, WaitMode::Block).empty());
}

TEST(MockIoEngine, WaitForCompletionsRespectsMax)
{
    MockIoEngine engine(Depth);
    std::vector<Destination> dsts(3);

    for (unsigned i = 0; i < 3; ++i)
    {
        engine.stage(i, FileRef{ 1, false }, i * Bytes, Bytes, dsts[i].buffer());
    }
    unsigned issued = 0;
    engine.flush(issued);
    engine.complete_all();

    EXPECT_EQ(drain(engine, WaitMode::NonBlocking, 2).size(), 2);
    EXPECT_EQ(drain(engine, WaitMode::NonBlocking, 2).size(), 1);   // the remainder
    EXPECT_TRUE(drain(engine).empty());
}

TEST(MockIoEngine, FullCompletionWritesThePatternAndNothingElse)
{
    MockIoEngine engine(Depth);
    Destination dst;

    const size_t offset = 4096;
    engine.stage(0, FileRef{ 1, false }, offset, Bytes, dst.buffer());

    unsigned issued = 0;
    engine.flush(issued);
    engine.complete(0);

    const auto completions = drain(engine);
    ASSERT_EQ(completions.size(), 1);
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(completions[0].bytes_transferred(), Bytes);

    EXPECT_EQ(dst.contents(Bytes), Destination::expected(offset, Bytes));
    EXPECT_TRUE(dst.guards_intact());
}

// Only what it reported. Filling the whole buffer would mask a caller that treats "no error" as
// "all bytes arrived".
TEST(MockIoEngine, ShortCompletionWritesOnlyTheReportedBytes)
{
    MockIoEngine engine(Depth);
    Destination dst;

    const size_t offset = 8192;
    const size_t landed = 20;

    engine.stage(0, FileRef{ 1, false }, offset, Bytes, dst.buffer());
    unsigned issued = 0;
    engine.flush(issued);
    engine.complete_short(0, landed);

    const auto completions = drain(engine);
    ASSERT_EQ(completions.size(), 1);

    // Still not a failure - a short count is visible only by comparing it against what was asked for.
    EXPECT_FALSE(completions[0].failed());
    EXPECT_EQ(completions[0].bytes_transferred(), landed);

    EXPECT_EQ(dst.contents(landed), Destination::expected(offset, landed));

    // The tail is untouched, so a caller that assumed the whole range arrived sees its own bytes.
    for (size_t i = landed; i < Bytes; ++i)
    {
        EXPECT_EQ(dst.buffer()[i], Destination::Guard) << "byte " << i << " should not have been written";
    }
    EXPECT_TRUE(dst.guards_intact());
}

TEST(MockIoEngine, FailureTransfersNothing)
{
    MockIoEngine engine(Depth);
    Destination dst;

    engine.stage(0, FileRef{ 1, false }, 0, Bytes, dst.buffer());
    unsigned issued = 0;
    engine.flush(issued);
    engine.fail(0, EIO);

    const auto completions = drain(engine);
    ASSERT_EQ(completions.size(), 1);
    // The raw errno, as a real engine reports it. Turning it into a common::ResponseCode is the caller's
    // job, so the mock must not do it here.
    EXPECT_EQ(completions[0].res, -EIO);
    EXPECT_EQ(completions[0].bytes_transferred(), 0u);

    for (size_t i = 0; i < Bytes; ++i)
    {
        EXPECT_EQ(dst.buffer()[i], Destination::Guard);
    }
}

// A refused stage records nothing - otherwise a test could not tell it from a staged request.
TEST(MockIoEngine, RefusedStageIsNotRecorded)
{
    MockIoEngine engine(Depth);
    Destination dst;

    engine.set_stage_result(common::ResponseCode::UnknownError);

    EXPECT_EQ(engine.stage(0, FileRef{ 1, false }, 0, Bytes, dst.buffer()), common::ResponseCode::UnknownError);
    EXPECT_EQ(engine.staged_count(), 0);
    EXPECT_TRUE(engine.history().empty());
}

TEST(MockIoEngine, FlushCanFail)
{
    MockIoEngine engine(Depth);
    Destination dst;

    engine.stage(0, FileRef{ 1, false }, 0, Bytes, dst.buffer());
    engine.set_flush_result(common::ResponseCode::UnknownError);

    unsigned issued = 0;
    EXPECT_EQ(engine.flush(issued), common::ResponseCode::UnknownError);
    EXPECT_EQ(issued, 0);
    EXPECT_EQ(engine.staged_count(), 1);
}

// The live views empty out as completions arrive, so the full set needs a record that outlives them.
TEST(MockIoEngine, HistoryOutlivesCompletion)
{
    MockIoEngine engine(Depth);
    std::vector<Destination> dsts(3);

    for (unsigned i = 0; i < 3; ++i)
    {
        engine.stage(i, FileRef{ 1, false }, i * Bytes, Bytes, dsts[i].buffer());
    }
    unsigned issued = 0;
    engine.flush(issued);
    engine.complete_all();
    drain(engine);

    EXPECT_EQ(engine.staged_count(), 0);
    EXPECT_EQ(engine.in_flight_count(), 0);

    ASSERT_EQ(engine.history().size(), 3);
    for (unsigned i = 0; i < 3; ++i)
    {
        EXPECT_EQ(engine.history()[i].id, i);
        EXPECT_EQ(engine.history()[i].offset, i * Bytes);
    }
}

// Page- and block-sized strides are the likely destination mistakes, and a plain (offset & 0xFF)
// would compare equal for all of them.
TEST(MockIoEngine, PatternDistinguishesPageAndBlockStrides)
{
    for (const size_t stride : { 256ul, 512ul, 4096ul, 65536ul, 2ul << 20 })
    {
        unsigned differing = 0;
        for (size_t i = 0; i < 64; ++i)
        {
            if (MockIoEngine::pattern(i) != MockIoEngine::pattern(i + stride))
            {
                ++differing;
            }
        }

        // Not every byte need differ, but a run of 64 must not be identical.
        EXPECT_GT(differing, 0u) << "pattern repeats exactly at stride " << stride;
    }
}

TEST(MockIoEngine, LimitsDefaultToTheRealKernelCap)
{
    MockIoEngine engine(Depth);

    EXPECT_EQ(engine.depth(), Depth);
    EXPECT_EQ(engine.limits().max_read_bytesize, max_read_bytesize());

    Limits limits;
    limits.max_read_bytesize = 4096;
    limits.offset_alignment = 512;
    MockIoEngine constrained(Depth, limits);

    EXPECT_EQ(constrained.limits().max_read_bytesize, 4096);
    EXPECT_EQ(constrained.limits().offset_alignment, 512);
}

// ---- the O_DIRECT alignment rule ----

namespace
{

constexpr size_t Block = 4096;

Limits direct_limits()
{
    Limits limits;
    limits.max_read_bytesize = max_read_bytesize();
    limits.offset_alignment = Block;
    limits.buffer_alignment = Block;
    return limits;
}

// A buffer whose address is a multiple of Block, so the address is never the reason a read is
// refused. Tests that want to break the address shift a pointer inside it.
struct AlignedDestination
{
    explicit AlignedDestination(size_t bytesize) :
        storage(bytesize + Block)
    {
        const auto raw = reinterpret_cast<uintptr_t>(storage.data());
        _base = storage.data() + ((Block - raw % Block) % Block);
    }

    char * buffer() { return _base; }

 private:
    std::vector<char> storage;
    char * _base = nullptr;
};

} // namespace

// A direct read that follows all three rules is staged like any other.
TEST(MockIoEngine, DirectStageIsAcceptedWhenAligned)
{
    MockIoEngine engine(Depth, direct_limits());
    AlignedDestination dst(Block);

    EXPECT_EQ(engine.stage(1, FileRef{ 3, true }, Block, Block, dst.buffer()), common::ResponseCode::Success);

    EXPECT_EQ(engine.misaligned_direct_stages(), 0u);
    EXPECT_EQ(engine.staged_count(), 1u);
}

// Each of the three values is checked on its own, so a test that breaks one is not passing because
// another was already wrong.
TEST(MockIoEngine, DirectStageIsRefusedWhenAnyValueIsMisaligned)
{
    AlignedDestination dst(Block * 2);

    // the file offset
    {
        MockIoEngine engine(Depth, direct_limits());

        EXPECT_NE(engine.stage(1, FileRef{ 3, true }, Block + 1, Block, dst.buffer()),
                  common::ResponseCode::Success);

        EXPECT_EQ(engine.misaligned_direct_stages(), 1u);

        // Refused means NOTHING was staged and no completion will arrive, which is what the kernel
        // does with EINVAL. The caller has to resolve the request itself.
        EXPECT_EQ(engine.staged_count(), 0u);
    }

    // the length
    {
        MockIoEngine engine(Depth, direct_limits());

        EXPECT_NE(engine.stage(1, FileRef{ 3, true }, Block, Block - 8, dst.buffer()),
                  common::ResponseCode::Success);

        EXPECT_EQ(engine.misaligned_direct_stages(), 1u);
        EXPECT_EQ(engine.staged_count(), 0u);
    }

    // the buffer address
    {
        MockIoEngine engine(Depth, direct_limits());

        EXPECT_NE(engine.stage(1, FileRef{ 3, true }, Block, Block, dst.buffer() + 1),
                  common::ResponseCode::Success);

        EXPECT_EQ(engine.misaligned_direct_stages(), 1u);
        EXPECT_EQ(engine.staged_count(), 0u);
    }
}

// The rule belongs to O_DIRECT alone. A buffered read at the same offset, length and address is fine,
// and checking it would refuse reads the kernel accepts.
TEST(MockIoEngine, BufferedStageIgnoresAlignment)
{
    MockIoEngine engine(Depth, direct_limits());
    AlignedDestination dst(Block);

    EXPECT_EQ(engine.stage(1, FileRef{ 3, false }, Block + 1, Block - 8, dst.buffer() + 1),
              common::ResponseCode::Success);

    EXPECT_EQ(engine.misaligned_direct_stages(), 0u);
    EXPECT_EQ(engine.staged_count(), 1u);
}

// The default Limits have an alignment of 1, where every value is a multiple. Tests written before
// this check existed pass unchanged, and only a test that asks for a real block size gets checked.
TEST(MockIoEngine, DefaultLimitsImposeNoAlignment)
{
    MockIoEngine engine(Depth);
    Destination dst;

    EXPECT_EQ(engine.stage(1, FileRef{ 3, true }, 7, 13, dst.buffer() + 1), common::ResponseCode::Success);

    EXPECT_EQ(engine.misaligned_direct_stages(), 0u);
}

}; // namespace runai::llm::streamer::posix_io
