/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/batch/batch.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <set>
#include <map>
#include <vector>

#include "utils/logging/logging.h"

#include "common/exception/exception.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/range/range.h"
#include "streamer/impl/reader/reader.h"
#include "streamer/impl/file/file.h"
#include "streamer/impl/s3/s3.h"

namespace runai::llm::streamer::impl
{

Batch::Batch(SubmissionId submission_id, unsigned workload_index, unsigned file_index, const std::string & path, const common::s3::S3ClientWrapper::Params & params, const Tasks && tasks, std::shared_ptr<common::Responder> responder, std::shared_ptr<const Config> config, size_t chunk_bytesize) :
    submission_id(submission_id),
    workload_index(workload_index),
    file_index(file_index),
    path(path),
    object_storage_params(params),
    tasks(tasks),
    chunks(split_into_chunks(this->tasks, chunk_bytesize)),
    range(tasks),
    responder(responder),
    config(config)
{
    LOG(DEBUG) << "Batch " << path << " range " << range << " ; " << this->tasks.size() << " tasks in "
               << chunks.size() << " chunks";
}

size_t Batch::total_bytes() const
{
    return range.size;
}

size_t Batch::end_offset() const
{
    return range.end;
}

void Batch::execute(std::atomic<bool> & stopped)
{
    LOG(DEBUG) << "Start reading from file " << path;

    auto response_code = common::ResponseCode::Success;
    try
    {
        ASSERT(!is_object_storage()) << "Unsupported reader mode for object storage backends";

        _reader = std::make_unique<File>(path, *config);
        read(*config, stopped);
    }
    catch(const common::Exception & e)
    {
        response_code = e.error();
    }
    catch (...)
    {
        response_code = common::ResponseCode::UnknownError;
    }

    // in case of an error all of the batch's unfinished tasks are failed with the same error code
    // in case of success the finished tasks were already notified
    handle_error(response_code);
}

void Batch::handle_error(common::ResponseCode response_code)
{
    // in case of an error all of the batch's unfinished tasks are failed with the same error code
    // in case of success the finished tasks were already notified

    if (response_code != common::ResponseCode::Success)
    {
        if (response_code != common::ResponseCode::FinishedError)
        {
            LOG(ERROR) << "Failed to read from file " << path << " ; error: " << response_code;
        }
        else
        {
            LOG(SPAM) << "Finished reading from file " << path;
        }

        // Note:
        // At this point no more tasks are expected to finish, since synchronous reading has ended and for asyncronous reading the thread stopped waiting for finished tasks
        for (auto & task : tasks)
        {
            if (task.finished_request(response_code))
            {
                common::Response response(submission_id, file_index, task.request->index, task.request->ret());
                responder->push(std::move(response), task.request->bytesize);
            }
        }
    }
}

// read the entire range and send notifications for each sub range
void Batch::read(const Config & config, std::atomic<bool> & stopped)
{
    if (tasks.empty())
    {
        LOG(DEBUG) << "Empty batch";
        return;
    }

    auto file_offset = range.start;
    // A batch covers one ContiguousTransfer, whose ranges are adjacent in both the file and the
    // destination, so the whole batch writes into one contiguous buffer starting at the first task's
    // destination. Both cursors advance in lockstep below.
    char * buffer = tasks[0].destination();

    size_t num_chunks = range.size / config.fs_sync_read_block_bytesize;

    // seek just once because tasks are consecutive within the range
    _reader->seek(file_offset);

    // read task's range in chunks
    size_t i = 0;
    for (; i < num_chunks && !stopped; ++i)
    {
        _reader->read(config.fs_sync_read_block_bytesize, buffer);

        file_offset += config.fs_sync_read_block_bytesize;
        buffer += config.fs_sync_read_block_bytesize;

        finished_until(file_offset, common::ResponseCode::Success);
    }

    if (file_offset < range.end && !stopped)
    {
        num_chunks++;
        i = 1;
        _reader->read(range.end - file_offset, buffer);
        finished_until(range.end, common::ResponseCode::Success);
    }

    // An empty batch range (range.start == range.end) enters neither branch above, so without this its
    // tasks would never be notified and the submission would wait for responses that never come. That
    // is reachable whenever a transfer carries only zero sized ranges - which the streamer accepts and
    // must still answer, one response per range whatever its size.
    // finished_until only ever advances _unfinished, so this is a no-op in every other case.
    if (!stopped)
    {
        finished_until(range.end, common::ResponseCode::Success);
    }

    LOG(DEBUG) << "Finished reading " << i << "/" << num_chunks << " chunks from file " << path << (stopped ? " - terminated" : " successfully");

    if (stopped)
    {
        throw common::Exception(common::ResponseCode::FinishedError);
    }
}

void Batch::handle_response(const common::backend_api::Response & response, const Task * task_ptr)
{
    // Aborting if a single task failed, we should replace this by a retry mechanism
    if (response.ret != common::ResponseCode::Success)
    {
        LOG(ERROR) << "Error " << response.ret << " while waiting for responses";
        throw common::Exception(response.ret);
    }

    ASSERT(task_ptr != nullptr) << "Received response from a null task";

    handle_task_response(response.ret, task_ptr);
}

void Batch::handle_task_response(const common::ResponseCode response_code, const Task * task_ptr)
{
    // Aborting if a single task failed, we should replace this by a retry mechanism

    ASSERT(task_ptr->request->file_index == file_index) << "Received response from a different file " << task_ptr->request->file_index << " expected " << file_index;

    LOG(SPAM) << "Received object storage response: File index " << file_index << " request index " << task_ptr->request->index << " ret " << response_code;
    if (task_ptr->finished_request(response_code))
    {
        common::Response request_response(submission_id, file_index, task_ptr->request->index, task_ptr->request->ret());
        responder->push(std::move(request_response), task_ptr->request->bytesize);
    }
}

// notify unfinished tasks up to but not including offset end
void Batch::finished_until(size_t file_offset, common::ResponseCode ret /*= common::ResponseCode::Success */)
{
    unsigned i = _unfinished;
    for (; i < tasks.size(); ++i)
    {
        if (file_offset < tasks[i].info.end)
        {
            break;
        }
        if (tasks[i].finished_request(ret))
        {
            const auto & r = tasks[i].request;
            common::Response response(submission_id, file_index, r->index, r->ret());
            LOG(SPAM) << "Sending response " << response;
            responder->push(std::move(response), tasks[i].request->bytesize);
        }
    }
    _unfinished = i;
}

unsigned Batch::finished_until() const
{
    return _unfinished;
}

bool Batch::is_object_storage() const
{
    return object_storage_params.valid();
}

std::ostream & operator<<(std::ostream & os, const Batch & r)
{
    return os << r.path << " range " << r.range << " ; " << r.tasks.size() << " tasks";
}

Batch::Range::Range(size_t start_offset, size_t end_offset) :
    common::Range(start_offset, end_offset - start_offset),
    end(end_offset)
{
    if (end < start)
    {
        LOG(ERROR) << "Invalid range " << start << " - " << end;
        throw common::Exception(common::ResponseCode::InvalidParameterError);
    }
}

Batch::Range::Range(const Tasks & tasks) :
    Range(calculate_start(tasks), calculate_end(tasks))
{}

size_t Batch::Range::calculate_start(const Tasks & tasks)
{
    if (tasks.empty())
    {
        return 0;
    }
    return tasks[0].info.offset;
}

size_t Batch::Range::calculate_end(const Tasks & tasks)
{
    if (tasks.empty())
    {
        return 0;
    }

    return tasks[tasks.size() - 1].info.end;
}

std::ostream & operator<<(std::ostream & os, const Batch::Range & r)
{
    return os << "Range from " << r.start << " to " << r.end;
}


}; // namespace runai::llm::streamer::impl
