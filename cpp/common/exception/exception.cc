/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "common/exception/exception.h"

namespace runai::llm::streamer::common
{

Exception::Exception(ResponseCode error) :
    _error(error)
{}

ResponseCode Exception::error() const
{
    return _error;
}

const char * unknown_error = "unknown_error";

const char* Exception::what() const noexcept
{
    try
    {
        return common::description(static_cast<int>(_error));
    }
    catch(const std::exception& e)
    {
    }
    return unknown_error;
}

}; // namespace runai::llm::streamer::common
