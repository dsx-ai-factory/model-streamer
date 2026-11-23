/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "posix_io/io_engine/io_engine.h"

#include <unistd.h>

#include <climits>

#include "utils/logging/logging.h"

namespace runai::llm::streamer::posix_io
{

size_t max_read_bytesize(size_t page_size)
{
    // MAX_RW_COUNT = INT_MAX & PAGE_MASK (include/linux/fs.h).
    return static_cast<size_t>(INT_MAX) & ~(page_size - 1);
}

size_t max_read_bytesize()
{
    // No error check, on purpose - do not add one. sysconf(_SC_PAGESIZE) cannot fail: glibc reads it
    // from the auxiliary vector, musl uses a constant, and sysconf returns -1 only for an unknown
    // name. A fallback would be dead code that quietly used a different page size.
    return max_read_bytesize(static_cast<size_t>(::sysconf(_SC_PAGESIZE)));
}

}; // namespace runai::llm::streamer::posix_io
