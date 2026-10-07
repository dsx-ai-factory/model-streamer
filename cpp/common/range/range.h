/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

#include "common/backend_api/object_storage/object_storage.h"

namespace runai::llm::streamer::common
{

struct Range
{
    Range() = default;
    Range(size_t start, size_t size);

    common::backend_api::ObjectRange_t to_backend_api_range() const
    {
        return common::backend_api::ObjectRange_t{start, size};
    }

    size_t start;
    size_t size;
};

std::ostream & operator<<(std::ostream & os, const Range & r);

}; //namespace runai::llm::streamer::common
