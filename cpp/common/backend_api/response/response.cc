/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "common/backend_api/response/response.h"

namespace runai::llm::streamer::common::backend_api
{

Response::Response(const ObjectCompletionEvent_t & event) :
    handle(event.request_id),
    ret(event.response_code)
{}

Response::Response(ObjectRequestId_t handle, common::ResponseCode ret) :
    handle(handle),
    ret(ret)
{}

Response::Response(ObjectRequestId_t handle) :
    Response(handle, common::ResponseCode::Success)
{}

Response::Response(common::ResponseCode ret) :
    Response(ObjectRequestId_t(0), ret)
{}

bool Response::operator==(const common::ResponseCode other)
{
    return ret == other;
}

bool Response::operator!=(const common::ResponseCode other)
{
    return ret != other;
}

std::ostream & operator<<(std::ostream & os, const Response & response)
{
    return os << "Handle: " << response.handle << " Response code: " << response.ret;
}

}; // namespace runai::llm::streamer::common::backend_api
