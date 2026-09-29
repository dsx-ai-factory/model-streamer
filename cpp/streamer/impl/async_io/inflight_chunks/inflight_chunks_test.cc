/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

}; // namespace runai::llm::streamer::impl
