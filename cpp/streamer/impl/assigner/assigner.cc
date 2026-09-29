/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/assigner/assigner.h"

#include <map>
#include <string>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/exception/exception.h"
#include "common/storage_uri/storage_uri.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

Assigner::Assigner(const std::vector<FileRanges> & request, std::shared_ptr<const Config> config,
                   std::vector<int> group_by_file) :
_config(config),
_is_object_storage(check_object_storage(request)),
_num_workers(_is_object_storage ? _config->s3_concurrency : _config->concurrency),
_num_workloads(0),
_group_by_file(std::move(group_by_file))
{
    LOG(DEBUG) << "Assigning " << request.size() << " files to " << _num_workers << " workers";

    if (request.empty())
    {
        LOG(WARNING) << "Assigner: No files provided.";
        return; // Nothing to assign
    }

    const size_t total_bytes_to_read = coalesce(request);

    if (_transfers.empty())
    {
        LOG(WARNING) << "Assigner: no ranges to read.";
        return;
    }

    if (total_bytes_to_read == 0)
    {
        LOG(WARNING) << "Total bytes to read is zero.";
    }

    assign(total_bytes_to_read);
}

// Merge each file's ranges into ContiguousTransfers. Single pass over the ranges, which also
// accumulates the byte total and detects unordered input - none of these needs its own scan.
size_t Assigner::coalesce(const std::vector<FileRanges> & request)
{
    size_t total_bytes = 0;

    // The best case is one transfer per file. Growing past that is cheap: ContiguousTransfer's move is
    // noexcept (two vector moves plus PODs), so reallocation moves the range vectors rather than
    // copying them.
    _transfers.reserve(request.size());

    for (unsigned file_index = 0; file_index < request.size(); ++file_index)
    {
        const auto & ranges = request[file_index].ranges;

        // Index of the transfer currently open for extension; _transfers.size() means "none open".
        // Reset per file, which is also what stops a transfer from ever spanning two files.
        size_t open = _transfers.size();
        bool ordered = true;

        for (unsigned i = 0; i < ranges.size(); ++i)
        {
            const auto & range = ranges[i];
            char * const destination = static_cast<char *>(range.dst);

            if (total_bytes > std::numeric_limits<size_t>::max() - range.size)
            {
                LOG(ERROR) << "Total byte size calculation overflow";
                throw common::Exception(common::ResponseCode::InvalidParameterError);
            }
            total_bytes += range.size;

            if (open < _transfers.size())
            {
                auto & current = _transfers[open];

                // Extend only when adjacent in BOTH the file and the destination: a single read fills a
                // single contiguous buffer, so either gap alone breaks the merge.
                if (current.offset + current.size == range.offset &&
                    current.destination + current.size == destination)
                {
                    current.size += range.size;
                    current.range_sizes.push_back(range.size);
                    continue;
                }

                // Noticed here for free rather than in a separate ordering pass
                ordered = ordered && range.offset >= current.offset;
            }

            // NOTE: never hold a reference into _transfers across this push - it may reallocate.
            // The open transfer is tracked by index for exactly that reason.
            open = _transfers.size();
            _transfers.emplace_back();

            auto & opened = _transfers.back();
            opened.file_index = file_index;
            opened.offset = range.offset;
            opened.size = range.size;
            opened.destination = destination;
            opened.first_range_index = i;
            opened.range_sizes.push_back(range.size);
        }

        if (!ordered)
        {
            LOG(DEBUG) << "Ranges of " << request[file_index].path << " are not ordered by ascending"
                       << " offset; coalescing is limited to ranges adjacent in the order given";
        }
    }

    LOG(DEBUG) << "Coalesced the request into " << _transfers.size() << " contiguous transfers";

    return total_bytes;
}

// Divide the transfers between the workers by total bytes. A transfer larger than a worker's remaining
// target is split, with offset and destination advancing together so every slice stays contiguous.
//
// Done per GROUP. A submission's filesystem files can be served by two different pools, and a group
// must be divided for the pool that will serve it: the async pool has one worker, so one workload, while
// the synchronous pool has `concurrency` of them. Dividing the async group 16 ways would not be wrong -
// the window spans workloads - but each slice would leave a partial chunk at its edges.
void Assigner::assign(size_t total_bytes_to_read)
{
    // Split the transfers by the group that will serve them, preserving order within each group so the
    // division below still walks a file's transfers in offset order.
    //
    // Group -1 is the synchronous pool; 0..N-1 are the async groups, one per distinct mount.
    std::vector<size_t> sync_indices;
    size_t sync_bytes = 0;

    std::map<int, std::vector<size_t>> async_indices;
    std::map<int, size_t> async_bytes;

    for (size_t i = 0; i < _transfers.size(); ++i)
    {
        const int group = group_of_file(_transfers[i].file_index);

        if (group < 0)
        {
            sync_indices.push_back(i);
            sync_bytes += _transfers[i].size;
        }
        else
        {
            async_indices[group].push_back(i);
            async_bytes[group] += _transfers[i].size;
        }
    }

    _num_async_groups = static_cast<unsigned>(async_indices.size());

    // Sized for the ceiling, then shrunk to what was used: the synchronous group can take up to
    // _num_workers workloads and each async group exactly one, so this is the most any submission can
    // need. Workload indices index into these.
    const unsigned ceiling = _num_workers + _num_async_groups + 1;
    _worker_assignments.clear();
    _worker_assignments.resize(ceiling);
    _group_by_workload.assign(ceiling, -1);

    unsigned used = 0;

    // The synchronous group first, so that with no async files the workload numbering is identical to
    // what it was before grouping existed - which is what keeps every existing test meaningful.
    used += assign_group(sync_indices, _num_workers, used, sync_bytes, -1);

    // One worker per mount, so one workload each, however many bytes they hold. Walked in group order
    // (std::map) so numbering is deterministic.
    for (const auto & [group, indices] : async_indices)
    {
        used += assign_group(indices, 1, used, async_bytes[group], group);
    }

    _num_workloads = std::max(used, 1u);
    _worker_assignments.resize(_num_workloads);
    _group_by_workload.resize(_num_workloads);

    // Verification - every byte assigned exactly once, and every transfer fully covered by its tasks
    size_t assigned_total = 0;
    for (const auto & assignment : _worker_assignments)
    {
        assigned_total += assignment.total_bytes;
    }

    ASSERT(assigned_total == total_bytes_to_read) << "Verification failed: Total bytes assigned (" << assigned_total
        << ") does not match total bytes requested (" << total_bytes_to_read << ")";

    for (size_t i = 0; i < _transfers.size(); ++i)
    {
        size_t transfer_total = 0;
        for (const auto & task : _transfers[i].tasks)
        {
            transfer_total += task.size;
        }
        ASSERT(transfer_total == _transfers[i].size) << "Transfer " << i << " of file " << _transfers[i].file_index
            << " assigned " << transfer_total << " bytes but covers " << _transfers[i].size;
    }

    LOG(DEBUG) << "Workload assignment verification successful. Total bytes assigned: " << assigned_total;
}

unsigned Assigner::assign_group(const std::vector<size_t> & indices,
                                unsigned workers,
                                unsigned first_workload,
                                size_t group_bytes,
                                int group)
{
    if (indices.empty())
    {
        return 0;
    }

    size_t base_bytes_remainder = 0;
    unsigned workloads = 0;
    const size_t base_bytes_per_worker = bytes_per_worker_for(group_bytes, workers, base_bytes_remainder, workloads);

    size_t position = 0;   // which entry of `indices` is being assigned
    size_t consumed = 0;   // how many of its bytes are already assigned
    unsigned used = 0;

    for (unsigned slot = 0; slot < workloads && position < indices.size(); ++slot)
    {
        const unsigned workload_idx = first_workload + slot;
        const size_t target = (slot == 0 ? base_bytes_per_worker + base_bytes_remainder : base_bytes_per_worker);
        size_t assigned = 0;

        LOG(DEBUG) << "Assigning work to worker " << workload_idx << ", target bytes: " << target
                   << (group < 0 ? " (synchronous)" : " (async group " + std::to_string(group) + ")");

        while (position < indices.size())
        {
            auto & transfer = _transfers[indices[position]];

            // A zero sized transfer is still given a task: its ranges are zero sized, and every range
            // owes exactly one response whatever its size.
            if (transfer.size > 0 && assigned >= target)
            {
                break;
            }

            const size_t remaining = transfer.size - consumed;
            const size_t to_assign = std::min(remaining, target - assigned);

            transfer.tasks.emplace_back(workload_idx,
                                        transfer.file_index,
                                        transfer.offset + consumed,
                                        to_assign,
                                        transfer.destination + consumed);

            LOG(SPAM) << "  Worker " << workload_idx << ": file " << transfer.file_index
                      << " offset " << transfer.offset + consumed << " size " << to_assign;

            assigned += to_assign;
            consumed += to_assign;
            _worker_assignments[workload_idx].total_bytes += to_assign;

            if (consumed == transfer.size)
            {
                ++position;
                consumed = 0;
            }
        }

        _group_by_workload[workload_idx] = group;
        used = slot + 1;

        LOG(DEBUG) << "Finished assignment for worker " << workload_idx << ", total bytes assigned: " << assigned;
    }

    return used;
}

int Assigner::group_of_file(unsigned file_index) const
{
    return file_index < _group_by_file.size() ? _group_by_file[file_index] : -1;
}

int Assigner::group_of_workload(unsigned workload_index) const
{
    return workload_index < _group_by_workload.size() ? _group_by_workload[workload_index] : -1;
}

unsigned Assigner::num_async_groups() const
{
    return _num_async_groups;
}

// The submission's backend kind. Reading it from the first file alone is sound because a submission is
// homogeneous IN THIS RESPECT: Streamer::lock_object_plugin rejects one that mixes filesystem with object
// storage (or two object-storage plugins) with UnsupportedBackendMix, before an Assigner is ever
// constructed. That is what lets one block size cover the whole submission, and what guarantees every
// workload is homogeneous for BackendPools::push to route between filesystem and object storage.
//
// It is NOT true that one worker count covers the whole submission: the filesystem side splits between
// the synchronous pool and the async one, and each group is divided for the pool that will serve it - see
// assign(). Only the object-storage-versus-filesystem axis is submission-wide.
bool Assigner::check_object_storage(const std::vector<FileRanges> & request) const
{
    // The first file WITH RANGES decides. A file with no ranges yields no transfer and reaches no storage,
    // so it must not select the backend - otherwise an empty "s3://..." entry alongside real filesystem
    // files would pick the object-storage worker count and block size for a submission that only ever
    // touches the filesystem. lock_object_plugin skips those entries for the same reason.
    for (const auto & file : request)
    {
        if (file.ranges.empty())
        {
            continue;
        }

        try
        {
            common::s3::StorageUri uri(file.path);
            return true;
        }
        catch(const std::exception& e)
        {
        }

        return false;
    }

    return false;
}

const std::vector<ContiguousTransfer> & Assigner::transfers() const
{
    return _transfers;
}

unsigned Assigner::get_num_workers() const
{
    return _num_workers;
}

// How many bytes each workload of a group should aim for, and how many workloads the group needs.
//
// Deliberately pure: writing _num_workloads as a side effect cannot work once there is more than one
// group, so each group computes its own count and the caller adds them up.
size_t Assigner::bytes_per_worker_for(size_t group_bytes, unsigned max_workers,
                                      size_t & remainder_bytesize, unsigned & workloads)
{
    const size_t block_bytesize = _is_object_storage ? _config->s3_block_bytesize : _config->fs_sync_read_block_bytesize;
    const size_t num_blocks = group_bytes / block_bytesize;

    // Zero-sized files are assigned to one worker: they may hold zero-sized tensors, and every range
    // owes a response whatever its size.
    workloads = static_cast<unsigned>(std::max(std::min(num_blocks, static_cast<size_t>(max_workers)), 1UL));

    const size_t base_bytes_per_worker = num_blocks / workloads * block_bytesize;
    remainder_bytesize = group_bytes - workloads * base_bytes_per_worker;

    LOG(DEBUG) << "Group bytes: " << group_bytes
               << ", Block bytesize: " << block_bytesize
               << ", Num blocks: " << num_blocks
               << ", Num workloads: " << workloads << " out of " << max_workers
               << ", Base bytes/worker: " << base_bytes_per_worker
               << ", Remainder bytesize: " << remainder_bytesize;

    return base_bytes_per_worker;
}

unsigned Assigner::num_workloads() const
{
    return _num_workloads;
}

} // namespace runai::llm::streamer::impl
