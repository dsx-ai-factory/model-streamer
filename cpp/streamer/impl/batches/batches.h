/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/responder/responder.h"
#include "common/storage_uri/storage_uri.h"
#include "streamer/impl/batch/batch.h"
#include "streamer/impl/config/config/config.h"
#include "streamer/impl/request/request.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/assigner/file_read_task/file_read_task.h"

namespace runai::llm::streamer::impl
{

// Transforms one ContiguousTransfer into batches, one batch per worker it was divided between.
// Batches is a group of Batch objects which together read the same contiguous span of one file.
//
// Built per transfer rather than per file: within a transfer the ranges tile one contiguous span of
// both file and destination, which is the assumption Batch is built on. A file with non-adjacent
// ranges simply yields several Batches.

struct Batches
{
    // range_sizes are the transfer's ranges in order, and first_range_index is the index of its first
    // range within the file - so the j-th range of this transfer is request index first_range_index + j.
    // That index is what the caller receives back as the response's `index`, so it must be the index in
    // the file's original range list, not the position within this transfer.
    Batches(SubmissionId submission_id,
           unsigned file_index,
           const std::vector<FileReadTask> & file_read_tasks,
           std::shared_ptr<const Config> config,
           std::shared_ptr<common::Responder> responder,
           const std::string & path,
           const common::s3::S3ClientWrapper::Params & params,
           const std::vector<size_t> & range_sizes,
           unsigned first_range_index);

    Batches(Batches &&) = default;
    Batches & operator=(Batches &&) = default;

    unsigned size() const;

    Batch & operator[](unsigned index);

    size_t total() const;

 private:
    struct BatchItr
    {
        BatchItr(const std::vector<FileReadTask> & file_read_tasks);

        unsigned current_index() const;
        unsigned current_workload_index() const;
        size_t consume(size_t bytesize);

        const FileReadTask & current_read_task() const;
        const FileReadTask & read_task(unsigned i) const;

        unsigned workers() const;
        unsigned workload_index(unsigned index) const;

     private:
        const std::vector<FileReadTask> & _file_read_tasks;
        const unsigned _num_batches;
        unsigned _current_task_index;
        unsigned _current_workload_index;
        size_t _current_worker_bytesize;
    };

    // create all the tasks
    void build_tasks(std::shared_ptr<const Config> config, const std::string & path, const common::s3::S3ClientWrapper::Params & params, const std::vector<size_t> & range_sizes);

    // create tasks of a given range; range_index is the index within the FILE, not within this transfer.
    // Tasks are cut at worker boundaries and at multiples of chunk_bytesize, so none straddles a chunk.
    void handle_request(std::vector<Tasks> & v_tasks, unsigned range_index, size_t request_file_offset, size_t request_size, char * destination, size_t chunk_bytesize);

    SubmissionId _submission_id;

    unsigned _file_index;

    // index within the file of this transfer's first range
    unsigned _first_range_index;

    BatchItr _itr;

    std::vector<Batch> _batches;
    std::shared_ptr<common::Responder> _responder;

    size_t _total = 0;
};

} // namespace runai::llm::streamer::impl
