/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/batches/batches.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "utils/logging/logging.h"

#include "common/exception/exception.h"
#include "streamer/impl/file/file.h"
#include "streamer/impl/s3/s3.h"
#include "common/s3_wrapper/s3_wrapper.h"


namespace runai::llm::streamer::impl
{

Batches::BatchItr::BatchItr(const std::vector<FileReadTask> & file_read_tasks) :
     _file_read_tasks(file_read_tasks),
    _num_batches(file_read_tasks.size()),
    _current_task_index(0)
{
    ASSERT(_num_batches) << "Zero file read requests";
    _current_workload_index = file_read_tasks[_current_task_index].workload_index;
    _current_worker_bytesize = file_read_tasks[_current_task_index].size;
}

unsigned Batches::BatchItr::current_index() const
{
    return _current_task_index;
}

unsigned Batches::BatchItr::current_workload_index() const
{
    return _current_workload_index;
}

const FileReadTask & Batches::BatchItr::read_task(unsigned index) const
{
    ASSERT(index < _num_batches) << "Index overflow " << index << " should be less than " << _num_batches;
    return _file_read_tasks[index];
}

unsigned Batches::BatchItr::workers() const
{
    return _num_batches;
}

unsigned Batches::BatchItr::workload_index(unsigned index) const
{
    return read_task(index).workload_index;
}

const FileReadTask & Batches::BatchItr::current_read_task() const
{
    ASSERT(_current_task_index < _num_batches) << "Batches iterator overflow _current_task_index = " << _current_task_index << " num_batches = " << _num_batches;
    return _file_read_tasks[_current_task_index];
}

size_t Batches::BatchItr::consume(size_t bytesize)
{
    if (bytesize == 0)
    {
        LOG(DEBUG) << "consuming zero bytes request";
    }
    if (_current_worker_bytesize == 0 && bytesize)
    {
        // advance to the next worker
        ++_current_task_index;
        _current_workload_index = workload_index(_current_task_index);
        _current_worker_bytesize = read_task(_current_task_index).size;
    }

    auto to_read = std::min<size_t>(_current_worker_bytesize, bytesize);
    _current_worker_bytesize -= to_read;
    return to_read;
}

Batches::Batches(SubmissionId submission_id,
                 unsigned file_index,
                 const std::vector<FileReadTask> & file_read_tasks,
                 std::shared_ptr<const Config> config,
                 std::shared_ptr<common::Responder> responder,
                 const std::string & path,
                 const common::s3::S3ClientWrapper::Params & params,
                 const std::vector<size_t> & range_sizes,
                 unsigned first_range_index) :
    _submission_id(submission_id),
    _file_index(file_index),
    _first_range_index(first_range_index),
    _itr(file_read_tasks),
    _responder(responder)
{
    _batches.reserve(file_read_tasks.size());
    build_tasks(config, path, params, range_sizes);
}

unsigned Batches::size() const
{
    return _batches.size();
}

Batch & Batches::operator[](unsigned index)
{
    ASSERT(index < _batches.size()) << "Batches overflow ( index " << index << " > size " << _batches.size();
    return _batches[index];
}

size_t Batches::total() const
{
    return _total;
}

void Batches::build_tasks(std::shared_ptr<const Config> config, const std::string & path, const common::s3::S3ClientWrapper::Params & params, const std::vector<size_t> & range_sizes)
{
    const auto num_workers = _itr.workers();
    LOG(DEBUG) << "Building tasks for " <<num_workers << " workers";

    // Cut at the size the BACKEND will request, not always the file system one: object storage slices
    // each task by s3_block_bytesize, so cutting tasks at the file system chunk size would make
    // RUNAI_STREAMER_FS_CHUNK_BYTESIZE change object-storage task granularity - a variable affecting a
    // backend it does not name.
    const size_t chunk_bytesize = params.valid() ? config->s3_block_bytesize : config->fs_async_chunk_bytesize;
    std::vector<Tasks> v_tasks(num_workers);

    auto num_sizes = range_sizes.size();

    // Within a transfer the ranges tile one contiguous span of both the file and the destination, so
    // both cursors advance by the range size. That is exactly what coalescing guarantees.
    size_t request_file_offset = _itr.read_task(0).offset_in_file;

    auto destination_start = static_cast<char *>(_itr.read_task(0).destination);

    auto current_request_destination = destination_start;

    // iterate over the workers and the requests to fill each worker share
    for (unsigned i = 0; i < num_sizes; ++i)
    {
        // create tasks for the entire requested range before sending to the threadpool
        const size_t request_size = range_sizes[i];

        // the response carries the index within the FILE, so offset by this transfer's first range
        const unsigned range_index = _first_range_index + i;

        handle_request(v_tasks, range_index, request_file_offset, request_size, current_request_destination, chunk_bytesize);
        LOG(DEBUG) << "created request index " << range_index << " dst " << static_cast<void *>(current_request_destination);

        current_request_destination += request_size;
        request_file_offset += request_size;
    }

    for (unsigned i = 0; i < num_workers; ++i)
    {
        const auto workload_index = _itr.workload_index(i);
        auto & tasks = v_tasks[i];
        auto size = tasks.size();
        if (size == 0)
        {
            LOG(WARNING) << "Zero tasks for worker index " << _itr.workload_index(i);
            continue;
        }

        // The same chunk size the tasks were cut with, so the chunks cover whole tasks.
        _batches.emplace_back(_submission_id, workload_index, _file_index, path, params, std::move(tasks), _responder, config, chunk_bytesize);
    }

    for (auto & batch : _batches)
    {
        _total += batch.total_bytes();
    }
}

void Batches::handle_request(std::vector<Tasks> & v_tasks, unsigned range_index, size_t request_file_offset, size_t request_size, char * destination, size_t chunk_bytesize)
{
    LOG(DEBUG) << "request file offset " << request_file_offset << " size " << request_size;

    // Task infos in file order, each paired with the batch it belongs to.
    //
    // A vector rather than a map keyed by batch index: cutting at chunk boundaries puts SEVERAL tasks
    // in the same batch, which a map keyed that way cannot hold - it kept only the first and lost the
    // rest.
    std::vector<std::pair<unsigned, Task::Info>> infos;

    auto bytes_to_request = request_size;
    size_t task_offset = request_file_offset;
    size_t destination_offset = 0;
    do
    {
        // Cut at the next chunk boundary as well as at the worker boundary consume() applies, so no
        // task ever straddles one. That is what lets a completed chunk account for a whole number of
        // tasks, which is how responses are emitted once completions arrive out of order.
        //
        // Boundaries are ABSOLUTE file offsets, not relative to the range: under O_DIRECT the chunks
        // have to be block aligned, and a range starts wherever the caller put it. So the first task
        // of a range is short and the rest are whole chunks.
        const size_t to_chunk_boundary = chunk_bytesize - (task_offset % chunk_bytesize);
        const size_t want = std::min(bytes_to_request, to_chunk_boundary);

        auto to_read = _itr.consume(want);
        infos.emplace_back(_itr.current_index(), Task::Info(task_offset, to_read, destination_offset));

        task_offset += to_read;
        bytes_to_request -= to_read;
        destination_offset += to_read;
    } while (bytes_to_request > 0);

    auto request_ptr = std::make_shared<Request>(request_file_offset, _file_index, range_index, infos.size(), request_size, destination);

    // create tasks
    for (auto & [batch_id, info] : infos)
    {
        Task task(request_ptr, std::move(info));
        LOG(SPAM) << task;
        ASSERT(batch_id < v_tasks.size()) << batch_id << " v_tasks.size() " << v_tasks.size();
        v_tasks[batch_id].emplace_back(std::move(task));
    }
}

}; // namespace runai::llm::streamer::impl
