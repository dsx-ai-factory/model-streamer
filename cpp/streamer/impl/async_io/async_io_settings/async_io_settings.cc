/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streamer/impl/async_io/async_io_settings/async_io_settings.h"

#include <algorithm>

#include "posix_io/io_engine/io_engine.h"
#include "utils/env/env.h"
#include "utils/logging/logging.h"

namespace runai::llm::streamer::impl
{

AsyncIoSettings::AsyncIoSettings(const Config & config, unsigned node_wide_depth, size_t max_read_bytesize) :
    // getenv_positive, not std::max on the raw value: the very next line DIVIDES by this, and a
    // floor applied before the narrowing to `unsigned` does not survive it (env.h).
    _process_group_size(utils::getenv_positive<unsigned>("RUNAI_STREAMER_PROCESS_GROUP_SIZE", 1U)),
    _depth(std::min(std::max(node_wide_depth / _process_group_size, MinDepth), MaxDepth)),
    _chunk_bytesize(std::min(config.fs_async_chunk_bytesize, max_read_bytesize))
{
    const auto divided = node_wide_depth / _process_group_size;

    if (_process_group_size > 1)
    {
        LOG(INFO) << "Queue depth " << node_wide_depth << " over " << _process_group_size
                  << " processes on this node gives " << _depth << " per process";
    }

    // Logged rather than applied silently: a configured number that never materialises is worse than
    // one that is rejected.
    if (divided < MinDepth)
    {
        LOG(WARNING) << "Queue depth resolves to " << divided << " per process ("
                     << node_wide_depth << " over " << _process_group_size
                     << "); raised to " << MinDepth << ", below which reads are effectively serial";
    }
    else if (divided > MaxDepth)
    {
        LOG(WARNING) << "Queue depth " << divided << " per process exceeds the " << MaxDepth
                     << " cap; clamped. Depth beyond this cannot add throughput - bytes in flight is"
                     << " what saturates a device, and this is already far past it";
    }

    if (_chunk_bytesize != config.fs_async_chunk_bytesize)
    {
        LOG(WARNING) << "Chunk size " << config.fs_async_chunk_bytesize << " exceeds the kernel's "
                     << max_read_bytesize << " per read; clamped";
    }
}

AsyncIoSettings::AsyncIoSettings(const Config & config, unsigned node_wide_depth) :
    AsyncIoSettings(config, node_wide_depth, posix_io::max_read_bytesize())
{}

unsigned AsyncIoSettings::depth() const
{
    return _depth;
}

size_t AsyncIoSettings::chunk_bytesize() const
{
    return _chunk_bytesize;
}

unsigned AsyncIoSettings::process_group_size() const
{
    return _process_group_size;
}

std::ostream & operator<<(std::ostream & os, const AsyncIoSettings & settings)
{
    return os << "async io: depth " << settings.depth()
              << " (of " << settings.process_group_size() << " processes on this node)"
              << ", chunk " << settings.chunk_bytesize() << " bytes"
              // The product, not just the factors: bytes in flight is what saturates a device, and it
              // is the only way a small chunk with a large depth reads as what it is.
              << ", up to " << (static_cast<size_t>(settings.depth()) * settings.chunk_bytesize())
              << " bytes in flight";
}

}; // namespace runai::llm::streamer::impl
