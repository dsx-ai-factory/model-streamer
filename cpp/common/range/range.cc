/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/range/range.h"

namespace runai::llm::streamer::common
{

Range::Range(size_t start, size_t size) :
    start(start),
    size(size)
{}

std::ostream & operator<<(std::ostream & os, const Range & r)
{
    return os << "offset : " << r.start << " bytesize: " << r.size;
}

}; //namespace runai::llm::streamer::common
