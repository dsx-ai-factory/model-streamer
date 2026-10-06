/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <string>
#include <vector>

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::impl
{

// ReadRange - a single range to read, as supplied by the caller: an arbitrary (offset, size) within its
// file, with its own destination. Ranges need not be contiguous in the file, need not be contiguous in
// memory, and need not be ordered. This is the input description; Request below is the streamer's
// internal per-range object built from it.

struct ReadRange
{
    size_t offset;   // source offset within the file
    size_t size;     // bytes to read
    void * dst;      // destination for this range
};

// FileRanges - one file of a submission, with all the ranges requested from it

struct FileRanges
{
    std::string path;
    std::vector<ReadRange> ranges;
};

// Request represents a sub range in a file
// For each request the streamer will issue a correponding response when the sub range is ready
// Reading the sub range can be divided between several workers of the streamer, where each part is represented by a task
// The request is completed and ready to respond when all its tasks are finished
// The request fails if any of the tasks has reported an error

struct Request
{
    Request(size_t file_offset, unsigned file_index, unsigned index, unsigned tasks, size_t bytesize, char * dst);

    // return true if all the request's range had been read
    bool finished(common::ResponseCode result = common::ResponseCode::Success);

    common::ResponseCode ret() const;

    // offset in file
    const size_t offset;

    // file index
    const unsigned file_index;

    // request index in the original list of file's sub ranges
    const unsigned index;


    const size_t bytesize;

    // start offset in destination buffer
    char * dst;

 private:
    // counter of unread chunks (each chunk is handled by a Task object)
    std::atomic<unsigned int> _tasks;

    std::atomic<common::ResponseCode> _ret = common::ResponseCode::Success;
};

}; // namespace runai::llm::streamer::impl
