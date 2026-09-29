/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "streamer/impl/assigner/file_read_task/file_read_task.h"
#include "streamer/impl/config/config/config.h"
#include "streamer/impl/request/request.h"

// Turn a multi-file read request into read assignments:
//   1. coalesce each file's ranges into ContiguousTransfers
//   2. group the transfers by the pool that will serve them, and divide each group between that
//      pool's workers, each slice represented as a FileReadTask
// A FileReadTask is later transformed into a Batch object

namespace runai::llm::streamer::impl
{

// ContiguousTransfer - a maximal group of the caller's ranges that are adjacent in BOTH the file and
// the destination buffer, so the whole group can be served by one sequential read. A transfer never
// spans two files.
//
// Coalescing happens BEFORE any work is divided between workers, so a single-worker streamer still
// reads a coalesced group with one read rather than one read per range.
//
// Ranges are coalesced in the order the caller supplied them: a caller that lists a file's ranges by
// ascending offset gets maximal coalescing (the .cc logs at DEBUG when they arrive unordered, so the
// lost coalescing is visible rather than silent).
//
// The transfer is also the unit that O_DIRECT alignment is priced per - each one needs its own
// block-aligned destination region - so coalescing bounds alignment overhead as well as seeks.

struct ContiguousTransfer
{
    unsigned file_index = 0;

    size_t offset = 0;             // start offset in the file
    size_t size = 0;               // total bytes of all the ranges it covers
    char * destination = nullptr;  // destination of its first range

    // Index of its first range within the file. The ranges it covers are
    // [first_range_index, first_range_index + range_sizes.size()), which is what makes the response
    // index computable: the j-th range of this transfer is request index first_range_index + j.
    unsigned first_range_index = 0;

    // Sizes of the ranges it covers, in order. Batches turns these into Request objects - one
    // response each, whatever the size.
    std::vector<size_t> range_sizes;

    // This transfer's slices, one per worker it was divided between (usually one)
    std::vector<FileReadTask> tasks;
};

// Holds all tasks assigned to a single worker

struct WorkerTasks
{
    std::vector<FileReadTask> tasks;
    size_t total_bytes = 0; // Total bytes assigned to this worker
};

// Distributes multi-file read workload across workers

struct Assigner
{
    // `group_by_file` says which pool group serves each file. It is indexed by the ORIGINAL file
    // index:
    //
    //   -1        the synchronous pool
    //   0, 1, 2   an async group - one per MOUNT in this submission
    //
    // An empty list means every file is synchronous. That is what every caller except the streamer
    // wants.
    //
    // There is one group per mount, not one group for all async files. A workload goes to exactly one
    // engine, and each engine serves one mount. If a workload held files from two mounts, the two
    // mounts would no longer be separated.
    //
    // A submission may now be heterogeneous, which it never was before: object storage and filesystem
    // still cannot mix (Streamer::lock_object_plugin rejects that), but the filesystem side can split
    // between the synchronous pool and the async one. Files are therefore GROUPED by that split and
    // each group divided for the pool that will serve it - one workload for the async pool, which has
    // one worker, and `concurrency` for the synchronous one.
    //
    // Dividing a group for the wrong pool is not a correctness bug, only a meaningless one: the async
    // window spans workloads, so 16 slices arriving at one worker still fill it. But the slices do not
    // align to chunk boundaries, so each one leaves a partial chunk at its edges.
    Assigner(const std::vector<FileRanges> & request,
             std::shared_ptr<const Config> config,
             std::vector<int> group_by_file = {});

    // The coalesced transfers, each carrying the worker slices it was divided into. Batches is built
    // per transfer: within a transfer the ranges tile one contiguous span of file and buffer, which is
    // the assumption Batch is built on.
    const std::vector<ContiguousTransfer> & transfers() const;

    // The worker count for the SYNCHRONOUS group (or for object storage): concurrency, or
    // s3_concurrency. NOT a submission-wide figure - the async group always uses one, because its
    // pool has one worker.
    unsigned get_num_workers() const;

    unsigned num_workloads() const;

    // The group this workload belongs to: -1 for the synchronous pool, or the async group id from
    // group_by_file. Workload numbers continue across the groups, so the caller can route on this
    // without adding a field to Workload or Batch.
    //
    // The caller turns the id back into a mount. Assigner never sees an st_dev. It only sees the ids
    // it was given.
    int group_of_workload(unsigned workload_index) const;

    // How many async groups this submission produced. It counts the different group ids among files
    // that produced a transfer. A file with no ranges produces none, so this can be smaller than the
    // number of mounts in group_by_file.
    unsigned num_async_groups() const;

 private:
    // Bytes per workload for one group, and how many workloads it needs. Deliberately pure: writing
    // _num_workloads as a side effect cannot work once there is more than one group.
    size_t bytes_per_worker_for(size_t group_bytes, unsigned max_workers,
                                size_t & remainder_bytesize, unsigned & workloads);
    bool check_object_storage(const std::vector<FileRanges> & request) const;

    // Step 1 - merge each file's ranges that are adjacent in both file and destination. Independent of
    // the worker count, and of how the work is later divided. Returns the total bytes of the request,
    // accumulated during the same pass so the ranges are walked only once.
    size_t coalesce(const std::vector<FileRanges> & request);

    // Step 2 - divide _transfers between the workers by total bytes, splitting a transfer that is
    // larger than a worker's remaining target. Offset and destination advance together across a split,
    // so each slice stays contiguous.
    //
    // Done once per GROUP: the synchronous files and the async ones are divided separately, for the
    // pool that will serve each. Workload indices continue across groups, so they stay unique.
    void assign(size_t total_bytes_to_read);

    // One group's share of step 2. `indices` are positions in _transfers; `workers` is how many
    // workloads this group may use; `first_workload` is where its workload numbering starts; `group`
    // is what group_of_workload() will report for them. Returns how many workloads it used.
    unsigned assign_group(const std::vector<size_t> & indices,
                          unsigned workers,
                          unsigned first_workload,
                          size_t group_bytes,
                          int group);

    // This file's group: -1 when group_by_file is empty, which is every caller but the streamer.
    int group_of_file(unsigned file_index) const;

 private:
    std::shared_ptr<const Config> _config;
    bool _is_object_storage;
    unsigned _num_workers;
    std::vector<ContiguousTransfer> _transfers;
    std::vector<WorkerTasks> _worker_assignments; // ordered by worker index
    unsigned _num_workloads;

    // Each file's group, by original file index. Empty means all synchronous.
    std::vector<int> _group_by_file;

    // Per workload index, its group. Sized _num_workloads.
    std::vector<int> _group_by_workload;

    unsigned _num_async_groups = 0;
};

} // namespace runai::llm::streamer::impl
