/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <functional>
#include <map>

#include "posix_io/io_engine/io_engine.h"
#include "streamer/impl/chunk_splitter/chunk_splitter.h"

namespace runai::llm::streamer::impl
{

// What one staged chunk is worth remembering while it is in flight.
//
// THREE lengths, not one. "The requested bytesize" is ambiguous once a short read can be re-issued,
// and comparing a completion against the wrong one mis-accounts the second pass with no error
// anywhere:
//
//   chunk.bytesize  the original extent - what the tasks are accounted against once ALL of it lands
//   cursor          where the next re-stage resumes
//   remaining       what THIS pass's completion is compared against
//
// Ask for 8 MiB, get 5: cursor advances by 5 and remaining becomes 3. With a single field the next
// completion of 3 would be compared against 8, judged short again, and re-issued forever.
struct InflightChunk
{
    Chunk  chunk;
    size_t cursor = 0;
    size_t remaining = 0;

    // Where to route the completion. Indices rather than pointers on purpose: a chunk can still be in
    // flight when its workload is erased (abort_all on stop, a drained responder, an OOM), and a
    // Batch * would then dangle. Resolving through the worker's workload map means the lookup that
    // finds the batch is also the check that the workload is still there - a late completion for an
    // erased workload simply fails to resolve and is dropped.
    uint64_t workload_id = 0;
    unsigned batch_index = 0;   // into the workload's batches; chunk.first_task is batch-local

    // Set only while a BOUNCED pass is in flight, and cleared when it lands.
    //
    // A direct read needs its offset, length and address to be block multiples. The first and last
    // block of a region are rarely whole, so those passes read one block into a scratch buffer and the
    // wanted bytes are copied out. Everything between is read straight into the destination and none
    // of these are set.
    //
    // `scratch` is borrowed from ScratchPool and must be given back when the pass lands, on every path
    // including failure - otherwise the pool drains and later reads fall back to buffered.
    char * scratch = nullptr;
    size_t scratch_skip = 0;    // where the wanted bytes start inside the scratch buffer
    size_t scratch_wanted = 0;  // how many of them this pass yields
};

// What a completion meant for the chunk it belongs to.
enum class Progress
{
    Complete,   // the whole extent has landed - its tasks can be accounted
    Partial,    // some bytes arrived; re-stage the rest, on the same entry
    Eof,        // zero further bytes while bytes were still owed - the file is shorter than asked
};

// Chunks issued and not yet finished, by request id.
//
// Ids are MONOTONIC AND NEVER REUSED, deliberately - not indices into a table sized `depth`. Reuse
// would let a completion that arrives after its request was abandoned land on whatever now holds that
// id, routing it to the wrong batch and the wrong destination with no error. Never reusing makes such
// a completion simply absent from the table, which is how it is recognised and dropped. Object
// storage does the same, for the same reason (object_storage_worker.cc:325-328).
//
// The in-flight bound is NOT here - CapacityQueue already enforces it. This is only a way to find a
// request's context again, which is why a map costs nothing worth counting: ~128 us/s at 1,280
// requests/s, against 5.4 rejecting fixed files for saving 64 us/s as unmeasurable.
//
// NOT THREAD SAFE. One worker owns it, like everything else on this path.
class InflightChunks
{
 public:
    // Record a chunk about to be staged, and return the id to stage it under. workload_id and
    // batch_index are how its completion finds the tasks it covers.
    posix_io::RequestId add(const Chunk & chunk, uint64_t workload_id, unsigned batch_index);

    // The entry for `id`, or nullptr if there is none - which is exactly how a late completion from an
    // abandoned request is recognised. Callers must check rather than assume.
    const InflightChunk * find(posix_io::RequestId id) const;

    // Account one completion of `bytes_transferred` against `id`.
    //
    // Complete and Eof leave the entry in place; the caller then reads it and calls release(). Partial
    // advances cursor and remaining, so the entry is ready to be re-staged with pending() below.
    //
    // ASSERTs if `id` is not live: only the caller can know whether an unknown id is a stale
    // completion to drop or its own bookkeeping gone wrong, so it must check find() first.
    Progress record(posix_io::RequestId id, size_t bytes_transferred);

    // Note the three scratch fields describe the CURRENT PASS, not the chunk. A chunk with a partial
    // head and a partial tail bounces twice, with a direct pass in between, and each sets them afresh.

    // What still has to be read for `id`: where to resume, how much is left, and where it goes. Only
    // meaningful after a Partial.
    Chunk pending(posix_io::RequestId id) const;

    // Remove `id` and return the chunk it was reading, so the caller can account its tasks.
    Chunk release(posix_io::RequestId id);

    // Forget everything, for teardown. Ids are not rewound, so a completion for a dropped request
    // still finds nothing rather than colliding with a later one.
    // Record that this pass reads into `scratch`, and which part of it is wanted. Cleared by
    // clear_bounce() when the pass lands.
    void set_bounce(posix_io::RequestId id, char * scratch, size_t skip, size_t wanted);

    // Forget the current pass's scratch, and return what it was so the caller can give it back. Null
    // when the pass was not bounced. Safe to call on every path, including failure - which is the
    // point, since a leaked buffer drains the pool.
    char * clear_bounce(posix_io::RequestId id);

    // Mutable access for the worker, which needs the scratch fields to copy out of. Const would mean
    // a second lookup to clear them.
    InflightChunk * find_mutable(posix_io::RequestId id);

    // Hand back every scratch buffer still held, then forget them. For teardown: clear() drops the
    // chunks, so without this the buffers would be lost and the pool empty for good.
    void release_all_scratch(const std::function<void(char *)> & give);

    void clear();

    size_t size() const;

 private:
    // Starts at 1 so that 0 is never a live id, leaving it usable as an unmistakable "not a real
    // request" value - the same reason object storage starts its counter at 1.
    posix_io::RequestId _next_id = 1;

    std::map<posix_io::RequestId, InflightChunk> _chunks;
};

}; // namespace runai::llm::streamer::impl
