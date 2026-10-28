/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/inflight_chunks/inflight_chunks.h"

#include <gtest/gtest.h>

#include <vector>

#include "common/exception/exception.h"

namespace runai::llm::streamer::impl
{

namespace
{

Chunk chunk_at(size_t offset, size_t bytesize, char * buffer)
{
    Chunk chunk;
    chunk.offset = offset;
    chunk.bytesize = bytesize;
    chunk.buffer = buffer;
    chunk.first_task = 0;
    chunk.task_count = 1;
    return chunk;
}

} // namespace

TEST(InflightChunks, Add_And_Release)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    EXPECT_EQ(chunks.size(), 0);

    const auto id = chunks.add(chunk_at(4096, 512, buffer.data()), 1 /* workload */, 0 /* batch */);
    EXPECT_EQ(chunks.size(), 1);

    const auto * entry = chunks.find(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->chunk.offset, 4096);
    EXPECT_EQ(entry->cursor, 4096) << "the cursor starts at the chunk's offset";
    EXPECT_EQ(entry->remaining, 512) << "and everything is still owed";

    const auto released = chunks.release(id);
    EXPECT_EQ(released.offset, 4096);
    EXPECT_EQ(chunks.size(), 0);
    EXPECT_EQ(chunks.find(id), nullptr);
}

// Ids are never reused, so a completion for a released request finds nothing - which is how a late
// completion from an abandoned request is recognised instead of landing on whatever took its place.
TEST(InflightChunks, Ids_Are_Never_Reused)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto first = chunks.add(chunk_at(0, 100, buffer.data()), 1 /* workload */, 0 /* batch */);
    chunks.release(first);

    const auto second = chunks.add(chunk_at(100, 100, buffer.data()), 1 /* workload */, 0 /* batch */);

    EXPECT_NE(first, second);
    EXPECT_EQ(chunks.find(first), nullptr) << "a late completion for the released id must find nothing";
    EXPECT_NE(chunks.find(second), nullptr);
}

TEST(InflightChunks, Zero_Is_Never_A_Live_Id)
{
    std::vector<char> buffer(16);
    InflightChunks chunks;

    EXPECT_EQ(chunks.find(0), nullptr);
    EXPECT_NE(chunks.add(chunk_at(0, 16, buffer.data()), 1 /* workload */, 0 /* batch */), 0u);
}

TEST(InflightChunks, Full_Completion_Is_Complete)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(0, 512, buffer.data()), 1 /* workload */, 0 /* batch */);
    EXPECT_EQ(chunks.record(id, 512), Progress::Complete);
}

// The case the three lengths exist for. With one field the second pass would be compared against the
// original 512, judged short again, and re-issued forever.
TEST(InflightChunks, Short_Read_Resumes_Where_It_Stopped)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(4096, 512, buffer.data()), 1 /* workload */, 0 /* batch */);

    ASSERT_EQ(chunks.record(id, 200), Progress::Partial);

    const auto rest = chunks.pending(id);
    EXPECT_EQ(rest.offset, 4096 + 200) << "resume where the first pass stopped";
    EXPECT_EQ(rest.bytesize, 312) << "and ask only for what is left";
    EXPECT_EQ(rest.buffer, buffer.data() + 200) << "the destination advances with the cursor";

    // The next completion is measured against THIS pass, not the original extent.
    EXPECT_EQ(chunks.record(id, 312), Progress::Complete);
}

TEST(InflightChunks, Several_Short_Reads_In_A_Row)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(0, 1000, buffer.data()), 1 /* workload */, 0 /* batch */);

    ASSERT_EQ(chunks.record(id, 400), Progress::Partial);
    ASSERT_EQ(chunks.record(id, 300), Progress::Partial);
    EXPECT_EQ(chunks.pending(id).offset, 700);
    EXPECT_EQ(chunks.pending(id).bytesize, 300);

    EXPECT_EQ(chunks.record(id, 300), Progress::Complete);

    // The extent is untouched by re-staging - it is what the tasks are accounted against.
    EXPECT_EQ(chunks.release(id).bytesize, 1000);
}

// Zero further bytes while bytes are still owed means the file is shorter than the caller asked for.
// That is EofError, not success - the contract the synchronous reader already has.
TEST(InflightChunks, Zero_Further_Bytes_Is_Eof)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(0, 512, buffer.data()), 1 /* workload */, 0 /* batch */);
    EXPECT_EQ(chunks.record(id, 0), Progress::Eof);

    const auto after_partial = chunks.add(chunk_at(512, 512, buffer.data()), 1 /* workload */, 0 /* batch */);
    ASSERT_EQ(chunks.record(after_partial, 100), Progress::Partial);
    EXPECT_EQ(chunks.record(after_partial, 0), Progress::Eof) << "even after progress, zero more is EOF";
}

// More than asked for cannot happen, but if it did, treating it as complete is safer than letting
// remaining underflow into an enormous number.
TEST(InflightChunks, Over_Long_Completion_Is_Complete)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(0, 512, buffer.data()), 1 /* workload */, 0 /* batch */);
    EXPECT_EQ(chunks.record(id, 4096), Progress::Complete);
}

// Routing travels with the entry, so a completion can find the tasks its chunk covered without the
// caller having to remember anything about it.
TEST(InflightChunks, Carries_Its_Routing)
{
    std::vector<char> buffer(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(0, 100, buffer.data()), 7 /* workload */, 3 /* batch */);

    const auto * entry = chunks.find(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->workload_id, 7u);
    EXPECT_EQ(entry->batch_index, 3u);

    // Indices, not pointers: a chunk can outlive its workload on an abort path, and resolving through
    // the worker's map means a late completion for an erased workload simply fails to resolve.
    ASSERT_EQ(chunks.record(id, 40), Progress::Partial);
    EXPECT_EQ(chunks.find(id)->workload_id, 7u) << "routing survives a re-stage";
}

TEST(InflightChunks, Unknown_Id_Asserts)
{
    InflightChunks chunks;

    EXPECT_THROW(chunks.record(7, 100), std::exception);
    EXPECT_THROW(chunks.release(7), std::exception);
    EXPECT_THROW(chunks.pending(7), std::exception);
}

// --- the transient retry budget --------------------------------------------------------------

// A chunk may retry many times and then must stop, so an interrupted read cannot loop for ever.
TEST(InflightChunks, A_Chunk_Spends_Its_Transient_Budget_And_Stops)
{
    std::vector<char> buffer(4096);
    InflightChunks chunks;
    const auto id = chunks.add(chunk_at(0, 100, buffer.data()), 1 /* workload */, 0 /* batch */);

    unsigned allowed = 0;
    while (chunks.may_retry_transient(id))
    {
        ASSERT_LT(++allowed, 100000u) << "the budget is not bounded at all";
    }

    // The NUMBER, not a relation between the counter and itself: comparing the two would hold for any
    // bound and for either comparison, which is no assertion at all.
    EXPECT_EQ(allowed, 1024u);
    EXPECT_EQ(chunks.find(id)->transient_retries, 1024u) << "the count stops at the budget";

    EXPECT_FALSE(chunks.may_retry_transient(id)) << "exhausted stays exhausted";
    EXPECT_EQ(chunks.find(id)->transient_retries, 1024u) << "a refused call still moved the count";
}

// The budget covers the chunk's whole life. Resetting it on progress would let a chunk that advances
// between interruptions retry without limit, which is the loop the bound exists to stop.
TEST(InflightChunks, Progress_Does_Not_Refund_The_Transient_Budget)
{
    std::vector<char> buffer(4096);
    InflightChunks chunks;
    const auto id = chunks.add(chunk_at(0, 100, buffer.data()), 1 /* workload */, 0 /* batch */);

    ASSERT_TRUE(chunks.may_retry_transient(id));
    ASSERT_TRUE(chunks.may_retry_transient(id));
    ASSERT_EQ(chunks.record(id, 40), Progress::Partial);

    EXPECT_EQ(chunks.find(id)->transient_retries, 2u) << "a short read refunded the budget";
}

// Budgets belong to a chunk, not to the worker: two chunks interrupted once each are not a pattern.
TEST(InflightChunks, Transient_Budgets_Are_Per_Chunk)
{
    std::vector<char> buffer(4096);
    InflightChunks chunks;
    const auto a = chunks.add(chunk_at(0, 100, buffer.data()), 1 /* workload */, 0 /* batch */);
    const auto b = chunks.add(chunk_at(100, 200, buffer.data() + 100), 1 /* workload */, 0 /* batch */);

    ASSERT_TRUE(chunks.may_retry_transient(a));
    ASSERT_TRUE(chunks.may_retry_transient(a));

    EXPECT_EQ(chunks.find(b)->transient_retries, 0u) << "b paid for a's interruptions";
}

// Unlike record() and release(), this does NOT assert on an unknown id: it is asked on a failure
// path, where a chunk may already have been released, and refusing is the safe answer.
TEST(InflightChunks, An_Unknown_Id_May_Not_Retry)
{
    InflightChunks chunks;

    EXPECT_FALSE(chunks.may_retry_transient(7));
}

TEST(InflightChunks, Several_Chunks_Are_Independent)
{
    std::vector<char> buffer(4096);
    InflightChunks chunks;

    const auto a = chunks.add(chunk_at(0, 100, buffer.data()), 1 /* workload */, 0 /* batch */);
    const auto b = chunks.add(chunk_at(100, 200, buffer.data() + 100), 1 /* workload */, 0 /* batch */);

    ASSERT_EQ(chunks.record(a, 50), Progress::Partial);

    EXPECT_EQ(chunks.find(b)->remaining, 200) << "b is untouched by a's short read";
    EXPECT_EQ(chunks.record(b, 200), Progress::Complete);
    EXPECT_EQ(chunks.pending(a).bytesize, 50);
}

// --- staging ---------------------------------------------------------------------------------------

namespace
{

// A staging buffer over memory the test owns, so a destination can be compared against an address.
StagingBuffer staging_over(std::vector<char> & memory)
{
    StagingBuffer buffer;
    buffer.data = memory.data();
    buffer.bytesize = memory.size();
    return buffer;
}

} // namespace

// A staged chunk reads into the staging buffer rather than into the caller's destination. Without a
// skip it starts at the front of it, which is what a buffered read does.
TEST(InflightChunks, A_Staged_Chunk_Reads_Into_Its_Buffer)
{
    std::vector<char> destination(1024);
    std::vector<char> staging(4096);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(1000, 600, destination.data()), 1 /* workload */, 0 /* batch */);
    chunks.set_staging(id, staging_over(staging));

    EXPECT_EQ(chunks.pending(id).buffer, staging.data());
}

// THE SKIP: a rounded direct read puts the bytes before the chunk at the front of the buffer, so the
// chunk's own bytes start after them.
//
// 1000 is `offset % block` for a chunk at 1000 on a 4096 mount - the read starts at 0 and the first
// 1000 bytes it brings back belong to nobody.
TEST(InflightChunks, A_Skip_Moves_The_Chunk_Down_Its_Staging_Buffer)
{
    std::vector<char> destination(1024);
    std::vector<char> staging(8192);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(1000, 600, destination.data()), 1 /* workload */, 0 /* batch */);
    chunks.set_staging(id, staging_over(staging));
    chunks.set_staging_skip(id, 1000);

    EXPECT_EQ(chunks.pending(id).buffer, staging.data() + 1000);
    EXPECT_EQ(chunks.pending(id).offset, 1000u) << "the FILE offset is the chunk's own, not rounded";
}

// A resumed pass has to land where the first one left off, or the chunk is assembled out of order.
TEST(InflightChunks, A_Skip_Holds_Across_A_Short_Read)
{
    std::vector<char> destination(1024);
    std::vector<char> staging(8192);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(1000, 600, destination.data()), 1 /* workload */, 0 /* batch */);
    chunks.set_staging(id, staging_over(staging));
    chunks.set_staging_skip(id, 1000);

    ASSERT_EQ(chunks.record(id, 250), Progress::Partial);

    const auto rest = chunks.pending(id);
    EXPECT_EQ(rest.buffer, staging.data() + 1000 + 250) << "the skip still applies after the cursor";
    EXPECT_EQ(rest.offset, 1250u);
    EXPECT_EQ(rest.bytesize, 350u);
}

// THE HOST PATH MUST NOT MOVE. A chunk with no staging buffer writes into the caller's memory, where
// there is nothing in front of it and nothing we may put there.
TEST(InflightChunks, A_Host_Chunk_Has_No_Skip)
{
    std::vector<char> destination(1024);
    InflightChunks chunks;

    const auto id = chunks.add(chunk_at(1000, 600, destination.data()), 1 /* workload */, 0 /* batch */);

    EXPECT_EQ(chunks.pending(id).buffer, destination.data());

    // And a skip cannot be given to one: it describes a position inside a staging buffer.
    EXPECT_THROW(chunks.set_staging_skip(id, 1000), std::exception);
}

}; // namespace runai::llm::streamer::impl
