/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "common/response_code/response_code.h"

namespace runai::llm::streamer::common
{

struct Exception : std::exception
{
    Exception(ResponseCode error);

    ResponseCode error() const;

    const char* what() const noexcept override;

 private:
    ResponseCode _error;
};

}; // namespace runai::llm::streamer::common
