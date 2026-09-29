/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "streamer/impl/task/task.h"

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

Task::Info::Info(size_t offset, size_t bytesize, size_t relative_offset) :
    offset(offset),
    bytesize(bytesize),
    end(offset + bytesize),
    relative_offset(relative_offset)
{}

Task::Task(std::shared_ptr<Request> request, size_t offset, size_t bytesize, size_t relative_offset) :
    Task(request, Info(offset, bytesize, relative_offset))
{}

Task::Task(std::shared_ptr<Request> request, Info && info) :
    request(request),
    info(info)
{}

// returns true if the request is completed (i.e. all its tasks were finished)
bool Task::finished_request(common::ResponseCode ret) const
{
    if (_finished)
    {
        // do nothing
        return false;
    }

    _finished = true;
    return request->finished(ret);
}

char * Task::destination() const
{
    ASSERT(request != nullptr) << "Request not initialized";
    return request->dst + info.relative_offset;
}

std::ostream & operator<<(std::ostream & os, const Task & task)
{
    os << "task to read " << task.info.bytesize << " bytes from file offset " << task.info.offset << " to " << task.info.end << " offset (relative to request start) " << task.info.relative_offset;
    return os;
}


}; // namespace runai::llm::streamer::impl
