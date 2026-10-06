/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <ostream>

#include "common/response_code/response_code.h"
#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::common::backend_api
{

struct Response
{
    Response(ObjectRequestId_t handle, common::ResponseCode ret);
    Response(ObjectRequestId_t handle);
    Response(common::ResponseCode ret);
    Response(const ObjectCompletionEvent_t & event);

    bool operator==(const common::ResponseCode other);
    bool operator!=(const common::ResponseCode other);

    // request id
    ObjectRequestId_t handle;

    // response code
    common::ResponseCode ret;
};

std::ostream & operator<<(std::ostream & os, const Response & response);

}; // namespace runai::llm::streamer::common::backend_api
