/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/chunk_splitter/chunk_splitter.h"
#include "common/device/device.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "streamer/impl/assigner/assigner.h"
#include "streamer/impl/batches/batches.h"
#include "streamer/impl/config/config/config.h"

#include "common/exception/exception.h"
#include "utils/random/random.h"
#include "utils/temp/env/env.h"
#include "utils/temp/file/file.h"

namespace runai::llm::streamer::impl
{

namespace
{

// Tasks from (offset, bytesize) pairs, one Request each, laid out consecutively from `base` so that
// Task::destination() is meaningful.
std::vector<Task> make_tasks(const std::vector<std::pair<size_t, size_t>> & spans, char * base)
{
    std::vector<Task> tasks;
    tasks.reserve(spans.size());

    const size_t start = spans.empty() ? 0 : spans.front().first;
    for (unsigned i = 0; i < spans.size(); ++i)
    {
        const auto [offset, bytesize] = spans[i];
        char * const destination = base + (offset - start);

        auto request = std::make_shared<Request>(offset, 0 /* file_index */, i, 1 /* tasks */, bytesize, destination);
        tasks.emplace_back(request, Task::Info(offset, bytesize, 0 /* relative_offset */));
    }
    return tasks;
}

std::vector<std::pair<size_t, size_t>> extents(const std::vector<Chunk> & chunks)
{
    std::vector<std::pair<size_t, size_t>> out;
    out.reserve(chunks.size());
    for (const auto & chunk : chunks)
    {
        out.emplace_back(chunk.offset, chunk.bytesize);
    }
    return out;
}

// Every task belongs to exactly one chunk, and the spans tile [0, tasks.size()).
void expect_spans_tile(const std::vector<Chunk> & chunks, size_t num_tasks)
{
    unsigned next = 0;
    for (const auto & chunk : chunks)
    {
        EXPECT_EQ(chunk.first_task, next) << "chunk spans must be contiguous over the task list";
        EXPECT_GT(chunk.task_count, 0);
        next = chunk.first_task + chunk.task_count;
    }
    EXPECT_EQ(next, num_tasks) << "every task must belong to exactly one chunk";
}

} // namespace

// Many small ranges inside one chunk become ONE read covering all of them - which is the case the
// grouping exists for.
TEST(ChunkSplitter, Small_Tasks_Share_A_Chunk)
{
    std::vector<char> buffer(4096);

    const auto tasks = make_tasks({ { 0, 100 }, { 100, 100 }, { 200, 100 }, { 300, 100 } }, buffer.data());
    const auto chunks = split_into_chunks(tasks, 4096);

    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].offset, 0);
    EXPECT_EQ(chunks[0].bytesize, 400);
    EXPECT_EQ(chunks[0].buffer, buffer.data());
    EXPECT_EQ(chunks[0].first_task, 0);
    EXPECT_EQ(chunks[0].task_count, 4);

    expect_spans_tile(chunks, tasks.size());
}

// A range already cut into chunk-sized tasks becomes one chunk each.
TEST(ChunkSplitter, One_Chunk_Per_Task_When_Tasks_Are_Chunk_Sized)
{
    std::vector<char> buffer(3 * 4096);

    const auto tasks = make_tasks({ { 0, 4096 }, { 4096, 4096 }, { 8192, 4096 } }, buffer.data());
    const auto chunks = split_into_chunks(tasks, 4096);

    ASSERT_EQ(chunks.size(), 3);
    EXPECT_EQ(extents(chunks), (std::vector<std::pair<size_t, size_t>>{ { 0, 4096 }, { 4096, 4096 }, { 8192, 4096 } }));

    for (unsigned i = 0; i < chunks.size(); ++i)
    {
        EXPECT_EQ(chunks[i].buffer, buffer.data() + i * 4096);
    }
    expect_spans_tile(chunks, tasks.size());
}

// Boundaries are absolute file offsets, so a batch starting mid-chunk has a short first chunk and
// whole ones after it.
TEST(ChunkSplitter, Batch_Starting_Mid_Chunk)
{
    std::vector<char> buffer(20000);

    const auto tasks = make_tasks({ { 1000, 3096 }, { 4096, 4096 }, { 8192, 4096 }, { 12288, 100 } }, buffer.data());
    const auto chunks = split_into_chunks(tasks, 4096);

    EXPECT_EQ(extents(chunks),
              (std::vector<std::pair<size_t, size_t>>{ { 1000, 3096 }, { 4096, 4096 }, { 8192, 4096 }, { 12288, 100 } }));
    expect_spans_tile(chunks, tasks.size());
}

// A zero-sized range owes a response but reads nothing, so it opens no chunk of its own.
TEST(ChunkSplitter, Zero_Sized_Task_Is_Absorbed)
{
    std::vector<char> buffer(4096);

    const auto tasks = make_tasks({ { 0, 100 }, { 100, 0 }, { 100, 100 } }, buffer.data());
    const auto chunks = split_into_chunks(tasks, 4096);

    ASSERT_EQ(chunks.size(), 1);
    EXPECT_EQ(chunks[0].bytesize, 200);
    EXPECT_EQ(chunks[0].task_count, 3);   // including the empty one

    expect_spans_tile(chunks, tasks.size());
}

// A batch with no bytes at all reads nothing. It must report NO chunks - its tasks still owe a
// response each, and the caller completes them directly rather than waiting for I/O never issued.
TEST(ChunkSplitter, All_Zero_Sized_Yields_No_Chunks)
{
    std::vector<char> buffer(1);

    const auto tasks = make_tasks({ { 0, 0 }, { 0, 0 } }, buffer.data());
    EXPECT_TRUE(split_into_chunks(tasks, 4096).empty());

    EXPECT_TRUE(split_into_chunks(std::vector<Task>{}, 4096).empty());
}

// The tasks of a batch are contiguous and ascending, and three things now rely on it. A gap means the
// invariant broke upstream, and a chunk covering it would read bytes nobody asked for.
TEST(ChunkSplitter, Gap_Between_Tasks_Is_Rejected)
{
    std::vector<char> buffer(4096);

    const auto gapped = make_tasks({ { 0, 1000 }, { 1200, 800 } }, buffer.data());
    EXPECT_THROW(split_into_chunks(gapped, 4096), std::exception);

    const auto descending = make_tasks({ { 1000, 100 }, { 0, 100 } }, buffer.data());
    EXPECT_THROW(split_into_chunks(descending, 4096), std::exception);
}

// A task crossing a chunk boundary means this and Batches::handle_request have drifted apart.
TEST(ChunkSplitter, Task_Crossing_A_Boundary_Is_Rejected)
{
    std::vector<char> buffer(8192);

    const auto uncut = make_tasks({ { 0, 100 }, { 100, 8000 } }, buffer.data());
    EXPECT_THROW(split_into_chunks(uncut, 4096), std::exception);
}

// End to end against the real Assigner and Batches, because the invariant the asserts rely on is
// THEIRS. A hole in the file must end the transfer, so the two sides never land in one batch - if
// that is wrong, this fails here rather than in production.
TEST(ChunkSplitter, Ranges_With_A_Hole_Become_Separate_Batches)
{
    constexpr size_t chunk = 4096;

    utils::temp::Env concurrency(std::string("RUNAI_STREAMER_CONCURRENCY"), 1UL);
    utils::temp::Env chunk_bytesize(std::string("RUNAI_STREAMER_FS_CHUNK_BYTESIZE"), static_cast<unsigned long>(chunk));

    auto config = std::make_shared<Config>(false /* do not force minimum */);
    auto responder = std::make_shared<common::Responder>(0);
    common::s3::S3ClientWrapper::Params s3_params;

    auto data = utils::random::buffer(20000);
    utils::temp::File file(data);
    std::vector<char> buffer(20000);

    // [0, 1000) and [1200, 2000) - the bytes between are not requested.
    FileRanges ranges;
    ranges.path = file.path;
    ranges.ranges.push_back(ReadRange{ 0, 1000, buffer.data() });
    ranges.ranges.push_back(ReadRange{ 1200, 800, buffer.data() + 1000 });

    Assigner assigner(std::vector<FileRanges>{ ranges }, config);

    // The hole ends the transfer, so this is two transfers rather than one spanning it.
    ASSERT_EQ(assigner.transfers().size(), 2);

    const std::vector<std::pair<size_t, size_t>> expected = { { 0, 1000 }, { 1200, 800 } };

    for (unsigned t = 0; t < assigner.transfers().size(); ++t)
    {
        const auto & transfer = assigner.transfers()[t];
        Batches batches(utils::random::number(), transfer.file_index, transfer.tasks, config, responder,
                        ranges.path, s3_params, transfer.range_sizes, transfer.first_range_index, common::Device::host());

        ASSERT_EQ(batches.size(), 1);

        const auto chunks = split_into_chunks(batches[0].tasks, chunk);
        EXPECT_EQ(extents(chunks), (std::vector<std::pair<size_t, size_t>>{ expected[t] }));
        expect_spans_tile(chunks, batches[0].tasks.size());
    }
}

}; // namespace runai::llm::streamer::impl
