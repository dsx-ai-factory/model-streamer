/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "streamer/impl/request/request.h"

#include <memory>
#include <utility>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

Request::Request(size_t offset, unsigned file_index, unsigned index, unsigned tasks, size_t bytesize, char * dst) :
    offset(offset),
    file_index(file_index),
    index(index),
    bytesize(bytesize),
    dst(dst),
    _tasks(tasks)
{}

bool Request::finished(common::ResponseCode result /* = common::ResponseCode::Success */)
{
    if (result != common::ResponseCode::Success)
    {
        _ret = result;
    }

    const auto running = _tasks.fetch_sub(1);
    return (running == 1);
}

common::ResponseCode Request::ret() const
{
    return _ret;
}

}; // namespace runai::llm::streamer::impl
