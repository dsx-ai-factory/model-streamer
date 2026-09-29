/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <stddef.h>
#include <memory>
#include <vector>

#include "common/range/range.h"
#include "common/s3_wrapper/s3_wrapper.h"
#include "common/backend_api/response/response.h"

namespace runai::llm::streamer::impl
{

struct Reader
{
    enum class Mode
    {
        Sync  = 0,
        Async = 1,
    };

    Reader(Mode mode) : mode(mode)
    {}

    virtual ~Reader() {}

    virtual void read(size_t bytesize, char * buffer) = 0;

    virtual void seek(size_t offset) = 0;

    // asynchronous
    // request_handle is a handle to the request, it is used to identify the request when the response is received
    virtual void async_read(const common::s3::S3ClientWrapper::Params & params, common::backend_api::ObjectRequestId_t request_handle, const common::Range & range, char * buffer) = 0;
    virtual common::ResponseCode async_response(std::vector<common::backend_api::Response> & responses, unsigned max_responses) = 0;

    // In-flight window (bytes) the backend advertises for submission throttling.
    // Default is unbounded; object-storage readers override it with the plugin value.
    virtual size_t max_inflight_bytes() const { return static_cast<size_t>(-1); }

    const Mode mode;
};

}; // namespace runai::llm::streamer::impl
