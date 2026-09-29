/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "streamer/impl/task/task.h"

namespace runai::llm::streamer::impl
{

// One read request to a backend, and the tasks it covers.
//
// Backend-agnostic on purpose, and shared: Batch builds its chunks with the splitter below, and BOTH
// readers consume them - the filesystem worker turns each into an IoEngine read, and the
// object-storage worker into a ranged GET with its own handle. Only the chunk SIZE differs between
// them, and that is a parameter.
//
// A batch is read as a sequence of these: contiguous in the file and in the destination, since a
// batch covers one contiguous transfer.
struct Chunk
{
    size_t offset = 0;       // absolute file offset
    size_t bytesize = 0;
    char * buffer = nullptr;

    // Tasks [first_task, first_task + task_count) of the batch. When this chunk lands, every one of
    // them has all of its bytes, so each can be accounted at once.
    unsigned first_task = 0;
    unsigned task_count = 0;
};

std::ostream & operator<<(std::ostream &, const Chunk &);

// Group a batch's tasks into chunks.
//
// This does NOT re-apply the boundary rule. Batches::handle_request already cuts tasks so that none
// crosses a multiple of chunk_bytesize, so the chunks are simply the tasks grouped by which chunk
// they fall in. Deriving them instead of computing them again is what stops the two from drifting
// apart - and if they did drift, a chunk would cover part of a task and its completion would report
// bytes that had not arrived.
//
// Chunks cover the NON-EMPTY tasks. Every task with bytes belongs to exactly one chunk, and reading
// that chunk is what accounts it.
//
// Zero-sized tasks - which a zero-sized range produces - are NOT covered. They read nothing, so no
// completion will ever arrive for them, and the caller must complete them AT ENQUEUE. That rule is
// unconditional: a zero-sized range owes a response like any other, and waiting for I/O that is never
// issued strands the caller's buffers forever. Object storage already does exactly this.
//
// Spans stay contiguous for simplicity, so a zero-sized task sitting between two non-empty ones falls
// inside a span. Accounting it a second time is harmless - Task::finished_request is idempotent behind
// its _finished guard - but a worker should skip zero-sized tasks when accounting a span rather than
// rely on that.
//
// A batch with no bytes at all therefore yields NO chunks, and all of its tasks are completed at
// enqueue.
//
// Takes std::vector<Task> rather than the Tasks alias, which lives in batch.h - the splitter has no
// business pulling in Batch, the responder and the object-storage wrapper just to name a vector.
std::vector<Chunk> split_into_chunks(const std::vector<Task> & tasks, size_t chunk_bytesize);

}; // namespace runai::llm::streamer::impl
