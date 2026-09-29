/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/chunk_splitter/chunk_splitter.h"

#include <algorithm>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

std::ostream & operator<<(std::ostream & os, const Chunk & chunk)
{
    return os << "chunk [" << chunk.offset << ", " << chunk.offset + chunk.bytesize << ")"
              << " tasks [" << chunk.first_task << ", " << chunk.first_task + chunk.task_count << ")";
}

std::vector<Chunk> split_into_chunks(const std::vector<Task> & tasks, size_t chunk_bytesize)
{
    ASSERT(chunk_bytesize) << "chunk bytesize must be positive";

    std::vector<Chunk> chunks;
    if (tasks.empty())
    {
        return chunks;
    }

    // At most one chunk per task, and at most one per chunk-sized span plus one for a batch that
    // starts mid-chunk.
    const auto & first = tasks.front().info;
    const auto & last = tasks.back().info;
    const size_t span = last.end - first.offset;
    chunks.reserve(std::min<size_t>(tasks.size(), span / chunk_bytesize + 2));

    // The end of the chunk the current task belongs to. Compared against rather than dividing per
    // task: chunk_bytesize is a runtime value, so a division would be a real one.
    size_t chunk_end = 0;
    size_t expected_offset = first.offset;

    for (unsigned i = 0; i < tasks.size(); ++i)
    {
        const auto & info = tasks[i].info;

        // A batch's tasks are contiguous and ascending. Assigner::coalesce only extends a transfer
        // when the next range is adjacent in the file, so a gap ends the transfer and never reaches a
        // batch; within a transfer the worker split only cuts, never reorders.
        //
        // Asserted because this is now the third thing relying on it - Batch::read seeks once and
        // reads on, and finished_until walks forward and stops at the first task it has not reached.
        // If it ever stops holding, failing here says so; the alternative is a chunk quietly covering
        // bytes nobody asked for.
        ASSERT(info.offset == expected_offset) << "task " << i << " starts at " << info.offset
                                               << ", expected " << expected_offset
                                               << " - a batch's tasks must be contiguous and ascending";
        expected_offset = info.end;

        // Zero-sized tasks carry no bytes, so they open no chunk of their own - they are absorbed into
        // the neighbouring one and accounted when it lands. A batch that is entirely zero-sized
        // therefore produces no chunks at all, and its caller must complete those tasks itself.
        const bool starts_a_chunk = chunks.empty() || (info.bytesize > 0 && info.offset >= chunk_end);

        if (starts_a_chunk)
        {
            chunk_end = (info.offset / chunk_bytesize + 1) * chunk_bytesize;
            chunks.push_back(Chunk{ info.offset, info.bytesize, tasks[i].destination(), i, 1 });
            continue;
        }

        auto & chunk = chunks.back();
        chunk.bytesize += info.bytesize;
        ++chunk.task_count;

        // S3a cuts tasks on chunk boundaries, so a task can only ever extend the chunk it started in.
        // If this fires, the two rules have drifted and a completion would report bytes that had not
        // arrived.
        ASSERT(chunk.offset + chunk.bytesize <= chunk_end)
            << "task " << i << " [" << info.offset << ", " << info.end << ") crosses the boundary at "
            << chunk_end << " - tasks must be cut on chunk boundaries";
    }

    // A batch of nothing but zero-sized tasks opened one chunk of zero bytes above; it reads nothing,
    // so report no chunks and let the caller finish those tasks directly.
    if (chunks.size() == 1 && chunks.front().bytesize == 0)
    {
        chunks.clear();
    }

    return chunks;
}

}; // namespace runai::llm::streamer::impl
