/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <memory>
#include "common/backend_api/object_storage/object_storage.h"
#include "streamer/impl/request/request.h"

namespace runai::llm::streamer::impl
{

// Task represents a chunk which is read by a single worker thread, and belongs to a single request (i.e. to a single sub range in the file)
// Each request consists of one or several tasks, where each task is read by a different thread

struct Task
{
    // file offsets to read
    struct Info
    {
        Info(size_t offset, size_t bytesize, size_t relative_offset);
        // offset from the beginning of the file
        size_t offset;
        // number of bytes to read
        size_t bytesize;
        // end from the beginning of the file
        size_t end;
        // relative offset from the beginning of the request (e.g. zero for the request's first task)
        size_t relative_offset;
    };

    Task(std::shared_ptr<Request> request, Info && info);
    Task(std::shared_ptr<Request> request, size_t offset, size_t bytesize, size_t relative_offset);

    bool finished_request(common::ResponseCode ret) const;

    char * destination() const;

    std::shared_ptr<Request> request;
    Info info;

 private:
    mutable bool _finished = false;
};

std::ostream & operator<<(std::ostream &, const Task &);

}; // namespace runai::llm::streamer::impl
